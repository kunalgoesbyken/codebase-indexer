use anyhow::Result;
use ndarray::Array2;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use tokenizers::Tokenizer;

const BATCH_SIZE: usize = 8;
const MAX_SEQ: usize = 8192;
const DOC_PREFIX: &str = "search_document: ";
const QUERY_PREFIX: &str = "search_query: ";

pub struct Embedder {
    session: Session,
    tokenizer: Tokenizer,
}

fn next_pow2_len(len: usize) -> usize {
    if len == 0 {
        return 32;
    }
    let mut p = 32usize;
    while p < len {
        p *= 2;
    }
    p.min(MAX_SEQ)
}

fn l2_normalize(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in vec.iter_mut() {
            *v /= norm;
        }
    }
}

fn mean_pool(output: &[f32], mask: &[i64], batch: usize, seq: usize, dims: usize) -> Vec<Vec<f32>> {
    let mut embeddings = Vec::with_capacity(batch);
    for i in 0..batch {
        let mut sum = vec![0.0f32; dims];
        let mut count = 0.0f32;
        for s in 0..seq {
            if mask[i * seq + s] > 0 {
                for d in 0..dims {
                    sum[d] += output[i * seq * dims + s * dims + d];
                }
                count += 1.0;
            }
        }
        if count > 0.0 {
            for d in 0..dims {
                sum[d] /= count;
            }
        }
        l2_normalize(&mut sum);
        embeddings.push(sum);
    }
    embeddings
}

impl Embedder {
    pub fn new(model_dir: &Path) -> Result<Self> {
        let model_path = model_dir.join("model_int8.onnx");
        let tokenizer_path = model_dir.join("tokenizer_nomic.json");

        let session = Session::builder()?
            .commit_from_file(&model_path)
            .map_err(|e| anyhow::anyhow!("failed to load ONNX model: {e}"))?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

        Ok(Self { session, tokenizer })
    }

    pub fn embed(&mut self, chunks: &[crate::types::Chunk]) -> Result<Vec<Vec<f32>>> {
        let texts: Vec<String> = chunks
            .iter()
            .map(|c| format!("{}{}", DOC_PREFIX, c.text))
            .collect();
        let mut embeddings = Vec::with_capacity(chunks.len());
        for batch in texts.chunks(BATCH_SIZE) {
            embeddings.extend(self.embed_batch(batch)?);
        }
        Ok(embeddings)
    }

    pub fn embed_by_text(&mut self, text: &str) -> Result<Vec<f32>> {
        let prefixed = format!("{}{}", QUERY_PREFIX, text);
        self.embed_single(&prefixed)
    }

    fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        if texts.len() == 1 {
            return Ok(vec![self.embed_single(&texts[0])?]);
        }

        let encodings: Vec<_> = texts
            .iter()
            .map(|t| self.tokenizer.encode(t.as_str(), true))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;

        let seq_lens: Vec<usize> = encodings.iter().map(|e| e.get_ids().len()).collect();
        let max_seq = *seq_lens.iter().max().unwrap_or(&1);
        let pad_to = next_pow2_len(max_seq).min(8192);

        let batch_size = texts.len();
        let mut all_ids = vec![0i64; batch_size * pad_to];
        let mut all_masks = vec![0i64; batch_size * pad_to];

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let offset = i * pad_to;
            let len = ids.len().min(pad_to);
            for j in 0..len {
                all_ids[offset + j] = ids[j] as i64;
                all_masks[offset + j] = mask[j] as i64;
            }
        }

        let input_ids = Array2::from_shape_vec((batch_size, pad_to), all_ids)?;
        let attention_mask = Array2::from_shape_vec((batch_size, pad_to), all_masks.clone())?;
        let token_type_ids = Array2::from_shape_vec((batch_size, pad_to), vec![0i64; batch_size * pad_to])?;

        let input_ids_tensor = Tensor::<i64>::from_array(input_ids)?;
        let attention_mask_tensor = Tensor::<i64>::from_array(attention_mask)?;
        let token_type_ids_tensor = Tensor::<i64>::from_array(token_type_ids)?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor
        ])?;

        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;

        let out_batch = shape[0] as usize;
        let out_seq = shape[1] as usize;
        let out_dims = shape[2] as usize;

        Ok(mean_pool(data, &all_masks, out_batch, out_seq.min(pad_to), out_dims))
    }

    fn embed_single(&mut self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();

        let seq_len = ids.len();
        let pad_to = next_pow2_len(seq_len).min(8192);
        let mut ids_padded = ids.clone();
        let mut mask_padded = mask.clone();
        ids_padded.resize(pad_to, 0i64);
        mask_padded.resize(pad_to, 0i64);
        let type_ids = vec![0i64; pad_to];

        let input_ids = Array2::from_shape_vec((1, pad_to), ids_padded)?;
        let attention_mask = Array2::from_shape_vec((1, pad_to), mask_padded.clone())?;
        let token_type_ids = Array2::from_shape_vec((1, pad_to), type_ids)?;

        let input_ids_tensor = Tensor::<i64>::from_array(input_ids)?;
        let attention_mask_tensor = Tensor::<i64>::from_array(attention_mask)?;
        let token_type_ids_tensor = Tensor::<i64>::from_array(token_type_ids)?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor
        ])?;

        let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;

        let out_dims = shape[2] as usize;
        let mut sum = vec![0.0f32; out_dims];
        let mut count = 0.0f32;

        for s in 0..pad_to.min(seq_len) {
            if mask_padded[s] > 0 {
                for d in 0..out_dims {
                    sum[d] += data[s * out_dims + d];
                }
                count += 1.0;
            }
        }

        if count > 0.0 {
            for d in 0..out_dims {
                sum[d] /= count;
            }
        }
        l2_normalize(&mut sum);
        Ok(sum)
    }
}