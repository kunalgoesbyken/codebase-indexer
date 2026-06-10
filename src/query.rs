use crate::types::IndexRecord;
use std::collections::HashMap;

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    dot / (mag_a * mag_b)
}

pub fn rrf_fuse(
    bm25_results: &[(String, usize, usize, f32)],
    vector_results: &[(usize, f32)],
    vector_records: &[IndexRecord],
    k: usize,
    limit: usize,
) -> Vec<(String, usize, String, f32)> {
    let mut scores: HashMap<(String, usize), (f32, String)> = HashMap::new();

    for (rank, (file_path, start_line, _end_line, _score)) in bm25_results.iter().enumerate() {
        let entry = scores.entry((file_path.clone(), *start_line)).or_insert((0.0, String::new()));
        entry.0 += 1.0 / (k as f32 + rank as f32);
    }

    for (rank, (idx, _score)) in vector_results.iter().enumerate() {
        if let Some(record) = vector_records.get(*idx) {
            let key = (record.chunk.file_path.clone(), record.chunk.start_line);
            let entry = scores.entry(key).or_insert((0.0, String::new()));
            entry.0 += 1.0 / (k as f32 + rank as f32);
            entry.1 = record.chunk.text.clone();
        }
    }

    let mut results: Vec<((String, usize), (f32, String))> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap_or(std::cmp::Ordering::Equal));

    results.into_iter()
        .take(limit)
        .map(|((file_path, start_line), (score, text))| (file_path, start_line, text, score))
        .collect()
}
