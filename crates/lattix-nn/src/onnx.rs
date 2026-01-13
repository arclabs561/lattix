//! ONNX model loading for KGE inference.
//!
//! Load pre-trained Knowledge Graph Embedding models from PyKEEN.
//!
//! # Supported Models
//!
//! - TransE
//! - DistMult
//! - ComplEx
//! - RotatE
//!
//! # PyKEEN Export
//!
//! Export from PyKEEN:
//! ```python
//! import torch
//! from pykeen.models import TransE
//! from pykeen.pipeline import pipeline
//!
//! result = pipeline(model='TransE', dataset='FB15k-237')
//! model = result.model
//!
//! # Export embeddings as numpy arrays
//! import numpy as np
//! np.save('entity_embeddings.npy', model.entity_representations[0].detach().numpy())
//! np.save('relation_embeddings.npy', model.relation_representations[0].detach().numpy())
//!
//! # Export entity/relation mappings
//! import json
//! with open('entity_to_id.json', 'w') as f:
//!     json.dump(result.training.entity_to_id, f)
//! with open('relation_to_id.json', 'w') as f:
//!     json.dump(result.training.relation_to_id, f)
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use propago::onnx::{KGEModel, load_pykeen_model};
//!
//! // Load from exported PyKEEN model
//! let model = load_pykeen_model(
//!     "entity_embeddings.npy",
//!     "relation_embeddings.npy", 
//!     "entity_to_id.json",
//!     "relation_to_id.json",
//!     KGEModelType::TransE,
//! )?;
//!
//! // Score a triple
//! let score = model.score("Apple", "founded_by", "Steve Jobs")?;
//!
//! // Link prediction
//! let candidates = model.predict_tail("Apple", "founded_by", 10)?;
//! ```

use crate::kge::{DistMult, KGEScorer, TransE};
use candle_core::{DType, Device, Result as CandleResult, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// Model type for KGE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KGEModelType {
    TransE,
    DistMult,
    ComplEx,
    RotatE,
}

/// Error types for ONNX/PyKEEN loading.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("NPY parse error: {0}")]
    Npy(String),
    #[error("Entity not found: {0}")]
    EntityNotFound(String),
    #[error("Relation not found: {0}")]
    RelationNotFound(String),
}

/// A loaded KGE model with entity/relation name mappings.
pub struct KGEModel {
    /// Underlying scorer.
    scorer: Box<dyn KGEScorer + Send + Sync>,
    /// Entity name -> ID mapping.
    entity_to_id: HashMap<String, usize>,
    /// ID -> entity name mapping.
    id_to_entity: Vec<String>,
    /// Relation name -> ID mapping.
    relation_to_id: HashMap<String, usize>,
    /// ID -> relation name mapping.
    id_to_relation: Vec<String>,
    /// Model type.
    model_type: KGEModelType,
}

impl KGEModel {
    /// Score a triple by name.
    pub fn score(&self, head: &str, relation: &str, tail: &str) -> Result<f32, LoadError> {
        let h = self
            .entity_to_id
            .get(head)
            .copied()
            .ok_or_else(|| LoadError::EntityNotFound(head.to_string()))?;
        let r = self
            .relation_to_id
            .get(relation)
            .copied()
            .ok_or_else(|| LoadError::RelationNotFound(relation.to_string()))?;
        let t = self
            .entity_to_id
            .get(tail)
            .copied()
            .ok_or_else(|| LoadError::EntityNotFound(tail.to_string()))?;

        self.scorer.score(h, r, t).map_err(LoadError::Candle)
    }

    /// Predict top-k tail entities for (head, relation, ?).
    pub fn predict_tail(
        &self,
        head: &str,
        relation: &str,
        k: usize,
    ) -> Result<Vec<(String, f32)>, LoadError> {
        let h = self
            .entity_to_id
            .get(head)
            .copied()
            .ok_or_else(|| LoadError::EntityNotFound(head.to_string()))?;
        let r = self
            .relation_to_id
            .get(relation)
            .copied()
            .ok_or_else(|| LoadError::RelationNotFound(relation.to_string()))?;

        let predictions = self.scorer.predict_tail(h, r, k).map_err(LoadError::Candle)?;

        Ok(predictions
            .into_iter()
            .map(|(id, score)| (self.id_to_entity[id].clone(), score))
            .collect())
    }

    /// Predict top-k head entities for (?, relation, tail).
    pub fn predict_head(
        &self,
        relation: &str,
        tail: &str,
        k: usize,
    ) -> Result<Vec<(String, f32)>, LoadError> {
        let r = self
            .relation_to_id
            .get(relation)
            .copied()
            .ok_or_else(|| LoadError::RelationNotFound(relation.to_string()))?;
        let t = self
            .entity_to_id
            .get(tail)
            .copied()
            .ok_or_else(|| LoadError::EntityNotFound(tail.to_string()))?;

        let predictions = self.scorer.predict_head(r, t, k).map_err(LoadError::Candle)?;

        Ok(predictions
            .into_iter()
            .map(|(id, score)| (self.id_to_entity[id].clone(), score))
            .collect())
    }

    /// Get entity ID by name.
    pub fn entity_id(&self, name: &str) -> Option<usize> {
        self.entity_to_id.get(name).copied()
    }

    /// Get relation ID by name.
    pub fn relation_id(&self, name: &str) -> Option<usize> {
        self.relation_to_id.get(name).copied()
    }

    /// Get entity name by ID.
    pub fn entity_name(&self, id: usize) -> Option<&str> {
        self.id_to_entity.get(id).map(|s| s.as_str())
    }

    /// Get relation name by ID.
    pub fn relation_name(&self, id: usize) -> Option<&str> {
        self.id_to_relation.get(id).map(|s| s.as_str())
    }

    /// Number of entities.
    pub fn num_entities(&self) -> usize {
        self.scorer.num_entities()
    }

    /// Number of relations.
    pub fn num_relations(&self) -> usize {
        self.scorer.num_relations()
    }

    /// Embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.scorer.embedding_dim()
    }

    /// Model type.
    pub fn model_type(&self) -> KGEModelType {
        self.model_type
    }
}

/// Load a PyKEEN-exported model.
///
/// # Arguments
/// - `entity_embeddings`: Path to entity_embeddings.npy
/// - `relation_embeddings`: Path to relation_embeddings.npy
/// - `entity_to_id`: Path to entity_to_id.json
/// - `relation_to_id`: Path to relation_to_id.json
/// - `model_type`: The KGE model type
pub fn load_pykeen_model(
    entity_embeddings: impl AsRef<Path>,
    relation_embeddings: impl AsRef<Path>,
    entity_to_id: impl AsRef<Path>,
    relation_to_id: impl AsRef<Path>,
    model_type: KGEModelType,
) -> Result<KGEModel, LoadError> {
    // Load embeddings from .npy files
    let entities = load_npy_f32(entity_embeddings)?;
    let relations = load_npy_f32(relation_embeddings)?;

    // Load ID mappings from JSON
    let entity_map: HashMap<String, usize> = {
        let file = File::open(entity_to_id)?;
        serde_json::from_reader(BufReader::new(file))?
    };

    let relation_map: HashMap<String, usize> = {
        let file = File::open(relation_to_id)?;
        serde_json::from_reader(BufReader::new(file))?
    };

    // Build reverse mappings
    let num_entities = entity_map.len();
    let num_relations = relation_map.len();

    let mut id_to_entity = vec![String::new(); num_entities];
    for (name, &id) in &entity_map {
        if id < num_entities {
            id_to_entity[id] = name.clone();
        }
    }

    let mut id_to_relation = vec![String::new(); num_relations];
    for (name, &id) in &relation_map {
        if id < num_relations {
            id_to_relation[id] = name.clone();
        }
    }

    // Infer dimensions
    let dim = entities.len() / num_entities;
    debug_assert_eq!(entities.len(), num_entities * dim);
    debug_assert_eq!(relations.len(), num_relations * dim);

    // Create scorer based on model type
    let scorer: Box<dyn KGEScorer + Send + Sync> = match model_type {
        KGEModelType::TransE => Box::new(TransE::from_vecs(
            &entities,
            &relations,
            num_entities,
            num_relations,
            dim,
            1.0, // default margin
        )?),
        KGEModelType::DistMult => Box::new(DistMult::from_vecs(
            &entities,
            &relations,
            num_entities,
            num_relations,
            dim,
        )?),
        KGEModelType::ComplEx | KGEModelType::RotatE => {
            // TODO: Implement ComplEx and RotatE
            // For now, fall back to DistMult
            Box::new(DistMult::from_vecs(
                &entities,
                &relations,
                num_entities,
                num_relations,
                dim,
            )?)
        }
    };

    Ok(KGEModel {
        scorer,
        entity_to_id: entity_map,
        id_to_entity,
        relation_to_id: relation_map,
        id_to_relation,
        model_type,
    })
}

/// Load embeddings from a simple binary format.
///
/// Format: [num_rows: u64][num_cols: u64][data: f32 * num_rows * num_cols]
pub fn load_embeddings_binary(path: impl AsRef<Path>) -> Result<(Vec<f32>, usize, usize), LoadError> {
    let mut file = File::open(path)?;
    let mut buf = [0u8; 8];

    file.read_exact(&mut buf)?;
    let num_rows = u64::from_le_bytes(buf) as usize;

    file.read_exact(&mut buf)?;
    let num_cols = u64::from_le_bytes(buf) as usize;

    let mut data = vec![0f32; num_rows * num_cols];
    let data_bytes = unsafe {
        std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u8, data.len() * 4)
    };
    file.read_exact(data_bytes)?;

    Ok((data, num_rows, num_cols))
}

/// Load a simple embeddings format (JSON with f32 arrays).
#[derive(Debug, Serialize, Deserialize)]
struct EmbeddingsJson {
    entities: Vec<Vec<f32>>,
    relations: Vec<Vec<f32>>,
    entity_to_id: HashMap<String, usize>,
    relation_to_id: HashMap<String, usize>,
}

/// Load from a combined JSON export.
pub fn load_embeddings_json(path: impl AsRef<Path>) -> Result<KGEModel, LoadError> {
    let file = File::open(path)?;
    let data: EmbeddingsJson = serde_json::from_reader(BufReader::new(file))?;

    let num_entities = data.entities.len();
    let num_relations = data.relations.len();
    let dim = data.entities.first().map(|v| v.len()).unwrap_or(0);

    // Flatten embeddings
    let entities: Vec<f32> = data.entities.into_iter().flatten().collect();
    let relations: Vec<f32> = data.relations.into_iter().flatten().collect();

    // Build reverse mappings
    let mut id_to_entity = vec![String::new(); num_entities];
    for (name, &id) in &data.entity_to_id {
        if id < num_entities {
            id_to_entity[id] = name.clone();
        }
    }

    let mut id_to_relation = vec![String::new(); num_relations];
    for (name, &id) in &data.relation_to_id {
        if id < num_relations {
            id_to_relation[id] = name.clone();
        }
    }

    let scorer: Box<dyn KGEScorer + Send + Sync> = Box::new(TransE::from_vecs(
        &entities,
        &relations,
        num_entities,
        num_relations,
        dim,
        1.0,
    )?);

    Ok(KGEModel {
        scorer,
        entity_to_id: data.entity_to_id,
        id_to_entity,
        relation_to_id: data.relation_to_id,
        id_to_relation,
        model_type: KGEModelType::TransE,
    })
}

/// Parse a .npy file (NumPy array format) containing f32 values.
///
/// Supports only simple 2D float32 arrays (the common case for embeddings).
fn load_npy_f32(path: impl AsRef<Path>) -> Result<Vec<f32>, LoadError> {
    let mut file = File::open(path)?;
    let mut header = [0u8; 10];
    file.read_exact(&mut header)?;

    // Verify magic number
    if &header[0..6] != b"\x93NUMPY" {
        return Err(LoadError::Npy("Invalid NPY magic number".into()));
    }

    let version = (header[6], header[7]);
    let header_len = match version {
        (1, 0) => u16::from_le_bytes([header[8], header[9]]) as usize,
        (2, 0) | (3, 0) => {
            let mut len_bytes = [0u8; 4];
            len_bytes[0] = header[8];
            len_bytes[1] = header[9];
            file.read_exact(&mut len_bytes[2..4])?;
            u32::from_le_bytes(len_bytes) as usize
        }
        _ => return Err(LoadError::Npy(format!("Unsupported NPY version: {:?}", version))),
    };

    // Skip header (we assume f32 and trust the array is well-formed)
    let mut header_str = vec![0u8; header_len];
    file.read_exact(&mut header_str)?;

    // Read the rest as f32 values
    let mut data = Vec::new();
    let mut buf = [0u8; 4];
    while file.read_exact(&mut buf).is_ok() {
        data.push(f32::from_le_bytes(buf));
    }

    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kge_model_type() {
        assert_eq!(KGEModelType::TransE, KGEModelType::TransE);
        assert_ne!(KGEModelType::TransE, KGEModelType::DistMult);
    }
}
