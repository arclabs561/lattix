//! ONNX-based KGE inference implementing KGEModel trait.
//!
//! This module wraps ONNX Runtime models for high-performance inference
//! of pre-trained KGE models exported from PyKEEN, LibKGE, or custom training.
//!
//! # Model Export
//!
//! Export a trained model from PyKEEN:
//!
//! ```python
//! from pykeen.pipeline import pipeline
//! import torch
//!
//! # Train
//! result = pipeline(model='TransE', dataset='FB15k-237')
//! model = result.model
//!
//! # Export to ONNX
//! dummy_h = torch.tensor([[0]], dtype=torch.long)
//! dummy_r = torch.tensor([[0]], dtype=torch.long)
//! dummy_t = torch.tensor([[0]], dtype=torch.long)
//!
//! torch.onnx.export(
//!     model,
//!     (dummy_h, dummy_r, dummy_t),
//!     "transe.onnx",
//!     input_names=["heads", "relations", "tails"],
//!     output_names=["scores"],
//!     dynamic_axes={"heads": {0: "batch"}, ...}
//! )
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use lattix_kge::models::KGEOnnx;
//! use lattix_kge::KGEModel;
//!
//! let model = KGEOnnx::from_file(
//!     "transe.onnx",
//!     "entity_map.json",
//!     "relation_map.json",
//! )?;
//!
//! let score = model.score("Einstein", "won", "NobelPrize")?;
//! ```

#![cfg(feature = "onnx")]

use crate::error::{Error, Result};
use crate::model::{Fact, KGEModel, ProgressCallback};
use crate::training::TrainingConfig;
use ndarray::{Array1, Axis};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Mutex;

/// ONNX-based KGE model for inference.
///
/// Wraps an ONNX model exported from PyKEEN or similar training frameworks.
/// Implements [`KGEModel`] for unified interface.
///
/// # Notes
///
/// - Training is not supported (returns error)
/// - Entity/relation embeddings may not be extractable depending on model export
pub struct KGEOnnx {
    session: Mutex<Session>,
    entity_map: HashMap<String, usize>,
    relation_map: HashMap<String, usize>,
    id_to_entity: Vec<String>,
    embedding_dim: usize,
}

impl std::fmt::Debug for KGEOnnx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KGEOnnx")
            .field("num_entities", &self.entity_map.len())
            .field("num_relations", &self.relation_map.len())
            .field("embedding_dim", &self.embedding_dim)
            .finish()
    }
}

impl KGEOnnx {
    /// Load model from ONNX file and JSON mapping files.
    ///
    /// # Arguments
    /// * `model_path` - Path to ONNX model file
    /// * `entity_map_path` - Path to JSON file mapping entity names to IDs
    /// * `relation_map_path` - Path to JSON file mapping relation names to IDs
    pub fn from_file(
        model_path: impl AsRef<Path>,
        entity_map_path: impl AsRef<Path>,
        relation_map_path: impl AsRef<Path>,
    ) -> Result<Self> {
        // Load mappings
        let entity_map: HashMap<String, usize> = serde_json::from_reader(BufReader::new(
            File::open(entity_map_path).map_err(Error::Io)?,
        ))
        .map_err(Error::Serialization)?;

        let relation_map: HashMap<String, usize> = serde_json::from_reader(BufReader::new(
            File::open(relation_map_path).map_err(Error::Io)?,
        ))
        .map_err(Error::Serialization)?;

        // Reverse entity map for predictions
        let mut id_to_entity = vec![String::new(); entity_map.len()];
        for (k, &v) in &entity_map {
            if v < id_to_entity.len() {
                id_to_entity[v] = k.clone();
            }
        }

        // Load ONNX model
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        Ok(Self {
            session: Mutex::new(session),
            entity_map,
            relation_map,
            id_to_entity,
            embedding_dim: 0, // Unknown without inspecting model
        })
    }

    /// Load with known embedding dimension.
    pub fn from_file_with_dim(
        model_path: impl AsRef<Path>,
        entity_map_path: impl AsRef<Path>,
        relation_map_path: impl AsRef<Path>,
        embedding_dim: usize,
    ) -> Result<Self> {
        let mut model = Self::from_file(model_path, entity_map_path, relation_map_path)?;
        model.embedding_dim = embedding_dim;
        Ok(model)
    }

    fn get_entity_id(&self, entity: &str) -> Result<usize> {
        self.entity_map
            .get(entity)
            .copied()
            .ok_or_else(|| Error::EntityNotFound(entity.to_string()))
    }

    fn get_relation_id(&self, relation: &str) -> Result<usize> {
        self.relation_map
            .get(relation)
            .copied()
            .ok_or_else(|| Error::RelationNotFound(relation.to_string()))
    }
}

impl KGEModel for KGEOnnx {
    fn score(&self, head: &str, relation: &str, tail: &str) -> Result<f32> {
        let h_id = self.get_entity_id(head)?;
        let r_id = self.get_relation_id(relation)?;
        let t_id = self.get_entity_id(tail)?;

        let h_tensor = Array1::from_elem(1, h_id as i64).insert_axis(Axis(1));
        let r_tensor = Array1::from_elem(1, r_id as i64).insert_axis(Axis(1));
        let t_tensor = Array1::from_elem(1, t_id as i64).insert_axis(Axis(1));

        let h_val = Value::from_array(h_tensor)?;
        let r_val = Value::from_array(r_tensor)?;
        let t_val = Value::from_array(t_tensor)?;

        let mut session = self.session.lock().unwrap();
        let outputs = session.run(ort::inputs![
            "heads" => h_val,
            "relations" => r_val,
            "tails" => t_val
        ])?;

        let scores_tuple = outputs["scores"].try_extract_tensor::<f32>()?;
        let scores_slice = scores_tuple.1;
        Ok(scores_slice[0])
    }

    fn train(&mut self, _triples: &[Fact<String>], _config: &TrainingConfig) -> Result<f32> {
        Err(Error::UnsupportedOperation(
            "ONNX models are inference-only. Train in Python with PyKEEN and export.".into(),
        ))
    }

    fn train_with_callback(
        &mut self,
        triples: &[Fact<String>],
        config: &TrainingConfig,
        _callback: ProgressCallback,
    ) -> Result<f32> {
        self.train(triples, config)
    }

    fn entity_embedding(&self, _entity: &str) -> Option<Vec<f32>> {
        // Most ONNX exports don't expose embeddings directly
        None
    }

    fn relation_embedding(&self, _relation: &str) -> Option<Vec<f32>> {
        None
    }

    fn entity_embeddings(&self) -> &HashMap<String, Vec<f32>> {
        static EMPTY: std::sync::OnceLock<HashMap<String, Vec<f32>>> = std::sync::OnceLock::new();
        EMPTY.get_or_init(HashMap::new)
    }

    fn relation_embeddings(&self) -> &HashMap<String, Vec<f32>> {
        static EMPTY: std::sync::OnceLock<HashMap<String, Vec<f32>>> = std::sync::OnceLock::new();
        EMPTY.get_or_init(HashMap::new)
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    fn num_entities(&self) -> usize {
        self.entity_map.len()
    }

    fn num_relations(&self) -> usize {
        self.relation_map.len()
    }

    fn name(&self) -> &'static str {
        "KGE-ONNX"
    }

    fn is_trained(&self) -> bool {
        true // ONNX models are always "trained"
    }

    fn predict_tail(
        &self,
        head: &str,
        relation: &str,
        k: usize,
    ) -> Result<Vec<crate::model::Prediction>> {
        let h_id = self.get_entity_id(head)?;
        let r_id = self.get_relation_id(relation)?;
        let num_entities = self.num_entities();

        let h_tensor = Array1::from_elem(num_entities, h_id as i64).insert_axis(Axis(1));
        let r_tensor = Array1::from_elem(num_entities, r_id as i64).insert_axis(Axis(1));
        let t_tensor =
            Array1::from_iter((0..num_entities).map(|x| x as i64)).insert_axis(Axis(1));

        let h_val = Value::from_array(h_tensor)?;
        let r_val = Value::from_array(r_tensor)?;
        let t_val = Value::from_array(t_tensor)?;

        let mut session = self.session.lock().unwrap();
        let outputs = session.run(ort::inputs![
            "heads" => h_val,
            "relations" => r_val,
            "tails" => t_val
        ])?;

        let scores_tuple = outputs["scores"].try_extract_tensor::<f32>()?;
        let scores_slice = scores_tuple.1;

        let mut results: Vec<(usize, f32)> = scores_slice
            .iter()
            .enumerate()
            .map(|(i, &s)| (i, s))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results
            .into_iter()
            .take(k)
            .map(|(idx, score)| crate::model::Prediction {
                entity: self.id_to_entity[idx].clone(),
                score,
            })
            .collect())
    }

    fn predict_head(
        &self,
        relation: &str,
        tail: &str,
        k: usize,
    ) -> Result<Vec<crate::model::Prediction>> {
        let r_id = self.get_relation_id(relation)?;
        let t_id = self.get_entity_id(tail)?;
        let num_entities = self.num_entities();

        let h_tensor =
            Array1::from_iter((0..num_entities).map(|x| x as i64)).insert_axis(Axis(1));
        let r_tensor = Array1::from_elem(num_entities, r_id as i64).insert_axis(Axis(1));
        let t_tensor = Array1::from_elem(num_entities, t_id as i64).insert_axis(Axis(1));

        let h_val = Value::from_array(h_tensor)?;
        let r_val = Value::from_array(r_tensor)?;
        let t_val = Value::from_array(t_tensor)?;

        let mut session = self.session.lock().unwrap();
        let outputs = session.run(ort::inputs![
            "heads" => h_val,
            "relations" => r_val,
            "tails" => t_val
        ])?;

        let scores_tuple = outputs["scores"].try_extract_tensor::<f32>()?;
        let scores_slice = scores_tuple.1;

        let mut results: Vec<(usize, f32)> = scores_slice
            .iter()
            .enumerate()
            .map(|(i, &s)| (i, s))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results
            .into_iter()
            .take(k)
            .map(|(idx, score)| crate::model::Prediction {
                entity: self.id_to_entity[idx].clone(),
                score,
            })
            .collect())
    }
}
