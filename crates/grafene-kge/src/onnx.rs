use crate::{Error, KGEmbedding, LinkPredictionResult, Result};
use ndarray::{Array1, Axis};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
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

    fn predict_tail(
        &self,
        head: &str,
        relation: &str,
        k: usize,
    ) -> Result<Vec<LinkPredictionResult>> {
        let h_id = self.get_entity_id(head)?;
        let r_id = self.get_relation_id(relation)?;
        let num_entities = self.num_entities();

        let h_tensor = Array1::from_elem(num_entities, h_id as i64).insert_axis(Axis(1));
        let r_tensor = Array1::from_elem(num_entities, r_id as i64).insert_axis(Axis(1));
        let t_tensor = Array1::from_iter((0..num_entities).map(|x| x as i64)).insert_axis(Axis(1));

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

        let top_k: Vec<LinkPredictionResult> = results
            .into_iter()
            .take(k)
            .enumerate()
            .map(|(rank, (idx, score))| LinkPredictionResult {
                entity: self.id_to_entity[idx].clone(),
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

        let h_tensor = Array1::from_iter((0..num_entities).map(|x| x as i64)).insert_axis(Axis(1));
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

        let top_k: Vec<LinkPredictionResult> = results
            .into_iter()
            .take(k)
            .enumerate()
            .map(|(rank, (idx, score))| LinkPredictionResult {
                entity: self.id_to_entity[idx].clone(),
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

pub type TransEOnnx = OnnxKGE;
