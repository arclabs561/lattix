use crate::{Error, KGEmbedding, LinkPredictionResult, Result};
use lattix_core::EntityId;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::sync::Mutex;

/// Generic ONNX KGE model wrapper.
pub struct OnnxKGE {
    session: Mutex<Session>,
    entity_map: HashMap<String, usize>,
    relation_map: HashMap<String, usize>,
    id_to_entity: Vec<String>,
    // id_to_relation: Vec<String>,
    embedding_dim: usize,
}

impl OnnxKGE {
    /// Load model and mappings from files.
    pub fn from_file(
        model_path: impl AsRef<Path>,
        entity_map_path: impl AsRef<Path>,
        relation_map_path: impl AsRef<Path>,
    ) -> Result<Self> {
        // Load mappings
        let entity_map: HashMap<String, usize> = serde_json::from_reader(BufReader::new(
            File::open(entity_map_path).map_err(|e| Error::Io(e))?,
        ))
        .map_err(|e| Error::Serialization(e))?;

        let relation_map: HashMap<String, usize> = serde_json::from_reader(BufReader::new(
            File::open(relation_map_path).map_err(|e| Error::Io(e))?,
        ))
        .map_err(|e| Error::Serialization(e))?;

        // Reverse maps
        let mut id_to_entity = vec![String::new(); entity_map.len()];
        for (k, &v) in &entity_map {
            if v < id_to_entity.len() {
                id_to_entity[v] = k.clone();
            }
        }

        let mut id_to_relation = vec![String::new(); relation_map.len()];
        for (k, &v) in &relation_map {
            if v < id_to_relation.len() {
                id_to_relation[v] = k.clone();
            }
        }

        // Load ONNX model
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        // Inspect model to guess dim? Or just assume it works.
        let embedding_dim = 0; // TODO: Infer

        Ok(Self {
            session: Mutex::new(session),
            entity_map,
            relation_map,
            id_to_entity,
            // id_to_relation,
            embedding_dim,
        })
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

impl KGEmbedding for OnnxKGE {
    fn score(&self, head: &str, relation: &str, tail: &str) -> Result<f32> {
        let h_id = self.get_entity_id(head)?;
        let r_id = self.get_relation_id(relation)?;
        let t_id = self.get_entity_id(tail)?;

        // Create tensors using ort 2.0 API: (shape, data)
        let h_val = Tensor::from_array(([1usize, 1], vec![h_id as i64]))?;
        let r_val = Tensor::from_array(([1usize, 1], vec![r_id as i64]))?;
        let t_val = Tensor::from_array(([1usize, 1], vec![t_id as i64]))?;

        let mut session = self.session.lock().unwrap();
        let outputs = session.run(ort::inputs![
            "heads" => h_val,
            "relations" => r_val,
            "tails" => t_val
        ])?;

        let scores = outputs["scores"].try_extract_tensor::<f32>()?;
        let (_shape, data) = scores;
        Ok(data[0])
    }

    fn predict_tail(
        &self,
        head: &str,
        relation: &str,
        k: usize,
    ) -> Result<Vec<LinkPredictionResult>> {
        let h_id = self.get_entity_id(head)?;
        let r_id = self.get_relation_id(relation)?;
        let num_entities = self.num_entities();

        // Create batch tensors using ort 2.0 API
        let h_data: Vec<i64> = vec![h_id as i64; num_entities];
        let r_data: Vec<i64> = vec![r_id as i64; num_entities];
        let t_data: Vec<i64> = (0..num_entities as i64).collect();

        let h_val = Tensor::from_array(([num_entities, 1], h_data))?;
        let r_val = Tensor::from_array(([num_entities, 1], r_data))?;
        let t_val = Tensor::from_array(([num_entities, 1], t_data))?;

        let mut session = self.session.lock().unwrap();
        let outputs = session.run(ort::inputs![
            "heads" => h_val,
            "relations" => r_val,
            "tails" => t_val
        ])?;

        let scores = outputs["scores"].try_extract_tensor::<f32>()?;
        let (_shape, scores_slice) = scores;

        let mut results: Vec<(usize, f32)> = scores_slice
            .iter()
            .enumerate()
            .map(|(i, &s)| (i, s))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k: Vec<LinkPredictionResult> = results
            .into_iter()
            .take(k)
            .enumerate()
            .map(|(rank, (idx, score))| LinkPredictionResult {
                entity: EntityId::new(&self.id_to_entity[idx]),
                score,
                rank: rank + 1,
            })
            .collect();

        Ok(top_k)
    }

    fn predict_head(
        &self,
        relation: &str,
        tail: &str,
        k: usize,
    ) -> Result<Vec<LinkPredictionResult>> {
        let r_id = self.get_relation_id(relation)?;
        let t_id = self.get_entity_id(tail)?;
        let num_entities = self.num_entities();

        // Create batch tensors using ort 2.0 API
        let h_data: Vec<i64> = (0..num_entities as i64).collect();
        let r_data: Vec<i64> = vec![r_id as i64; num_entities];
        let t_data: Vec<i64> = vec![t_id as i64; num_entities];

        let h_val = Tensor::from_array(([num_entities, 1], h_data))?;
        let r_val = Tensor::from_array(([num_entities, 1], r_data))?;
        let t_val = Tensor::from_array(([num_entities, 1], t_data))?;

        let mut session = self.session.lock().unwrap();
        let outputs = session.run(ort::inputs![
            "heads" => h_val,
            "relations" => r_val,
            "tails" => t_val
        ])?;

        let scores = outputs["scores"].try_extract_tensor::<f32>()?;
        let (_shape, scores_slice) = scores;

        let mut results: Vec<(usize, f32)> = scores_slice
            .iter()
            .enumerate()
            .map(|(i, &s)| (i, s))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k: Vec<LinkPredictionResult> = results
            .into_iter()
            .take(k)
            .enumerate()
            .map(|(rank, (idx, score))| LinkPredictionResult {
                entity: EntityId::new(&self.id_to_entity[idx]),
                score,
                rank: rank + 1,
            })
            .collect();

        Ok(top_k)
    }

    fn entity_embedding(&self, _entity: &str) -> Result<Vec<f32>> {
        Err(Error::UnsupportedOperation(
            "Entity embedding extraction not supported by this ONNX model".into(),
        ))
    }

    fn relation_embedding(&self, _relation: &str) -> Result<Vec<f32>> {
        Err(Error::UnsupportedOperation(
            "Relation embedding extraction not supported by this ONNX model".into(),
        ))
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
}

/// TransE ONNX wrapper (alias for OnnxKGE).
pub type TransEOnnx = OnnxKGE;
