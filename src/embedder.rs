use anyhow::Result;
use ndarray::Array2;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use tokenizers::Tokenizer;

const BATCH_SIZE: usize = 8;
const MIN_SEQ: usize = 32;
const MAX_SEQ: usize = 2048;
const DOC_PREFIX: &str = "search_document: ";
const QUERY_PREFIX: &str = "search_query: ";

pub struct Embedder {
    session: Session,
    tokenizer: Tokenizer,
}

fn padded_len(len: usize) -> usize {
    len.next_power_of_two().clamp(MIN_SEQ, MAX_SEQ)
}

fn l2_normalize(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        vec.iter_mut().for_each(|v| *v /= norm);
    }
}

fn mean_pool(output: &[f32], mask: &[i64], batch: usize, seq: usize, dims: usize) -> Vec<Vec<f32>> {
    (0..batch)
        .map(|i| {
            let mut sum = vec![0.0f32; dims];
            let mut count = 0.0f32;
            for s in (0..seq).filter(|&s| mask[i * seq + s] > 0) {
                let token = &output[(i * seq + s) * dims..][..dims];
                sum.iter_mut().zip(token).for_each(|(acc, x)| *acc += x);
                count += 1.0;
            }
            if count > 0.0 {
                sum.iter_mut().for_each(|v| *v /= count);
            }
            l2_normalize(&mut sum);
            sum
        })
        .collect()
}

impl Embedder {
    pub fn new(model_dir: &Path) -> Result<Self> {
        let session = Session::builder()?
            .commit_from_file(model_dir.join("model_int8.onnx"))
            .map_err(|e| anyhow::anyhow!("failed to load ONNX model: {e}"))?;
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer_nomic.json"))
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;
        Ok(Self { session, tokenizer })
    }

    pub fn embed(&mut self, chunks: &[crate::types::Chunk]) -> Result<Vec<Vec<f32>>> {
        let texts: Vec<String> = chunks.iter().map(|c| format!("{DOC_PREFIX}{}", c.text)).collect();
        let mut order: Vec<usize> = (0..texts.len()).collect();
        order.sort_by_key(|&i| texts[i].len());

        let mut embeddings = vec![Vec::new(); texts.len()];
        for batch in order.chunks(BATCH_SIZE) {
            let batch_texts: Vec<&str> = batch.iter().map(|&i| texts[i].as_str()).collect();
            for (&i, embedding) in batch.iter().zip(self.embed_batch(&batch_texts)?) {
                embeddings[i] = embedding;
            }
        }
        Ok(embeddings)
    }

    pub fn embed_by_text(&mut self, text: &str) -> Result<Vec<f32>> {
        let prefixed = format!("{QUERY_PREFIX}{text}");
        self.embed_batch(&[prefixed.as_str()])?
            .pop()
            .ok_or_else(|| anyhow::anyhow!("model returned no embedding"))
    }

    fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;

        let longest = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);
        let seq = padded_len(longest);
        let batch = texts.len();
        let mut ids = vec![0i64; batch * seq];
        let mut mask = vec![0i64; batch * seq];
        for (i, encoding) in encodings.iter().enumerate() {
            let pairs = encoding.get_ids().iter().zip(encoding.get_attention_mask()).take(seq);
            for (j, (&id, &m)) in pairs.enumerate() {
                ids[i * seq + j] = id as i64;
                mask[i * seq + j] = m as i64;
            }
        }

        let outputs = self.session.run(ort::inputs![
            "input_ids" => Tensor::from_array(Array2::from_shape_vec((batch, seq), ids)?)?,
            "attention_mask" => Tensor::from_array(Array2::from_shape_vec((batch, seq), mask.clone())?)?,
            "token_type_ids" => Tensor::from_array(Array2::<i64>::zeros((batch, seq)))?
        ])?;

        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
        Ok(mean_pool(data, &mask, batch, seq, shape[2] as usize))
    }
}
