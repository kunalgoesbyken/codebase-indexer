use std::collections::BinaryHeap;

#[derive(Clone, Debug)]
struct Candidate {
    dist: f32,
    idx: usize,
}

impl Eq for Candidate {}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

fn cosine_dist(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        return 1.0;
    }
    1.0 - (dot / (mag_a * mag_b))
}

fn rand_level(ml: f64) -> usize {
    let r: f64 = rand::random();
    (-r.ln() * ml).floor() as usize
}

#[derive(Clone)]
#[allow(dead_code)]
struct HnswNode {
    vector: Vec<f32>,
    level: usize,
    neighbors: Vec<Vec<usize>>,
}

#[allow(dead_code)]
pub struct HnswIndex {
    nodes: Vec<HnswNode>,
    entry: Option<usize>,
    max_level: usize,
    m: usize,
    m_max: usize,
    m_max0: usize,
    ef_construction: usize,
    ml: f64,
}

impl HnswIndex {
    pub fn new() -> Self {
        let m = 16;
        Self {
            nodes: Vec::new(),
            entry: None,
            max_level: 0,
            m,
            m_max: m,
            m_max0: 2 * m,
            ef_construction: 200,
            ml: 1.0 / (m as f64).ln(),
        }
    }

    pub fn insert(&mut self, vector: Vec<f32>) {
        let level = rand_level(self.ml);
        let idx = self.nodes.len();

        let mut neighbors = Vec::with_capacity(level + 1);
        for _ in 0..=level {
            neighbors.push(Vec::new());
        }
        self.nodes.push(HnswNode {
            vector: vector.clone(),
            level,
            neighbors,
        });

        if let Some(ep) = self.entry {
            let mut curr = ep;
            let max_lvl = self.max_level;

            for lc in (level + 1..=max_lvl).rev() {
                let nearest = self.search_layer(&vector, curr, 1, lc);
                curr = nearest[0];
            }

            for lc in (0..=level.min(max_lvl)).rev() {
                let candidates = self.search_layer(&vector, curr, self.ef_construction, lc);
                let neighbors = self.select_neighbors(&vector, &candidates, if lc == 0 { self.m_max0 } else { self.m_max });

                for &neighbor in &neighbors {
                    self.shrink(neighbor, lc);
                }

                self.nodes[idx].neighbors[lc] = neighbors.clone();

                for &neighbor in &neighbors {
                    while self.nodes[neighbor].neighbors.len() <= lc {
                        self.nodes[neighbor].neighbors.push(Vec::new());
                    }
                    self.nodes[neighbor].neighbors[lc].push(idx);
                }

                if !neighbors.is_empty() {
                    curr = neighbors[0];
                }
            }

            if level > max_lvl {
                self.max_level = level;
                self.entry = Some(idx);
            }
        } else {
            self.entry = Some(idx);
            self.max_level = level;
        }
    }

    fn search_layer(&self, query: &[f32], entry: usize, ef: usize, layer: usize) -> Vec<usize> {
        let mut visited = vec![false; self.nodes.len()];
        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();

        let dist = cosine_dist(query, &self.nodes[entry].vector);
        visited[entry] = true;
        candidates.push(Candidate { dist, idx: entry });
        results.push(Candidate { dist, idx: entry });

        while let Some(c) = candidates.pop() {
            let furthest_dist = results.peek().map(|r| r.dist).unwrap_or(f32::MAX);
            if c.dist > furthest_dist {
                break;
            }

            let neighbors = &self.nodes[c.idx].neighbors;
            if layer < neighbors.len() {
                for &neighbor in &neighbors[layer] {
                    if !visited[neighbor] {
                        visited[neighbor] = true;
                        let d = cosine_dist(query, &self.nodes[neighbor].vector);
                        let furthest = results.peek().map(|r| r.dist).unwrap_or(f32::MAX);
                        if d < furthest || results.len() < ef {
                            candidates.push(Candidate { dist: d, idx: neighbor });
                            results.push(Candidate { dist: d, idx: neighbor });
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        results.into_iter().map(|c| c.idx).collect()
    }

    fn select_neighbors(&self, query: &[f32], candidates: &[usize], m: usize) -> Vec<usize> {
        let mut sorted: Vec<(f32, usize)> = candidates
            .iter()
            .map(|&idx| (cosine_dist(query, &self.nodes[idx].vector), idx))
            .collect();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(m);
        sorted.into_iter().map(|(_, idx)| idx).collect()
    }

    fn shrink(&mut self, idx: usize, layer: usize) {
        if layer >= self.nodes[idx].neighbors.len() {
            return;
        }
        let max_conn = if layer == 0 { self.m_max0 } else { self.m_max };
        if self.nodes[idx].neighbors[layer].len() > max_conn {
            let query = &self.nodes[idx].vector;
            let neighbors = self.nodes[idx].neighbors[layer].clone();
            let selected = self.select_neighbors(query, &neighbors, max_conn);
            self.nodes[idx].neighbors[layer] = selected;
        }
    }

    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<usize> {
        if self.nodes.is_empty() {
            return Vec::new();
        }

        let ep = self.entry.unwrap_or(0);
        let mut curr = ep;

        for lc in (1..=self.max_level).rev() {
            let nearest = self.search_layer(query, curr, 1, lc);
            curr = nearest[0];
        }

        let results = self.search_layer(query, curr, ef.max(k), 0);
        let mut sorted: Vec<(f32, usize)> = results
            .into_iter()
            .map(|idx| (cosine_dist(query, &self.nodes[idx].vector), idx))
            .collect();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(k);
        sorted.into_iter().map(|(_, idx)| idx).collect()
    }

}
