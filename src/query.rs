use crate::types::Chunk;
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

pub fn rrf_fuse<'a>(rankings: &[Vec<&'a Chunk>], k: f32, limit: usize) -> Vec<(&'a Chunk, f32)> {
    let mut fused: HashMap<(&str, usize), (&'a Chunk, f32)> = HashMap::new();
    for ranking in rankings {
        for (rank, &chunk) in ranking.iter().enumerate() {
            let key = (chunk.file_path.as_str(), chunk.start_line);
            fused.entry(key).or_insert((chunk, 0.0)).1 += 1.0 / (k + rank as f32 + 1.0);
        }
    }
    let mut results: Vec<(&Chunk, f32)> = fused.into_values().collect();
    results.sort_by(|a, b| b.1.total_cmp(&a.1));
    results.truncate(limit);
    results
}
