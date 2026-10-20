use serde::{Deserialize, Serialize};
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashSet};

const M: usize = 16;
const M_MAX0: usize = 2 * M;
const EF_CONSTRUCTION: usize = 200;

#[derive(Clone, Copy)]
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
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.total_cmp(&other.dist)
    }
}

fn cosine_dist(a: &[f32], b: &[f32]) -> f32 {
    1.0 - crate::query::cosine_similarity(a, b)
}

fn random_level() -> usize {
    let r: f64 = rand::random();
    (-r.ln() / (M as f64).ln()).floor() as usize
}

fn max_connections(layer: usize) -> usize {
    if layer == 0 { M_MAX0 } else { M }
}

#[derive(Serialize, Deserialize)]
pub struct HnswIndex {
    neighbors: Vec<Vec<Vec<usize>>>,
    entry: Option<usize>,
    max_level: usize,
}

struct Graph<'a> {
    index: &'a mut HnswIndex,
    vectors: &'a [&'a [f32]],
}

impl HnswIndex {
    pub fn build(vectors: &[&[f32]]) -> Self {
        let mut index = Self {
            neighbors: Vec::with_capacity(vectors.len()),
            entry: None,
            max_level: 0,
        };
        let mut graph = Graph { index: &mut index, vectors };
        for idx in 0..vectors.len() {
            graph.insert(idx);
        }
        index
    }

    pub fn is_consistent(&self, node_count: usize) -> bool {
        let has_layer = |node: usize, layer: usize| node < node_count && self.neighbors[node].len() > layer;
        self.neighbors.len() == node_count
            && self.entry.is_none_or(|entry| has_layer(entry, self.max_level))
            && self.neighbors.iter().all(|layers| {
                layers
                    .iter()
                    .enumerate()
                    .all(|(layer, links)| links.iter().all(|&n| has_layer(n, layer)))
            })
    }

    pub fn search(&self, vectors: &[&[f32]], query: &[f32], k: usize, ef: usize) -> Vec<usize> {
        let Some(mut curr) = self.entry else {
            return Vec::new();
        };
        for layer in (1..=self.max_level).rev() {
            curr = self.search_layer(vectors, query, curr, 1, layer)[0].idx;
        }
        self.search_layer(vectors, query, curr, ef.max(k), 0)
            .into_iter()
            .take(k)
            .map(|c| c.idx)
            .collect()
    }

    fn search_layer(
        &self,
        vectors: &[&[f32]],
        query: &[f32],
        entry: usize,
        ef: usize,
        layer: usize,
    ) -> Vec<Candidate> {
        let start = Candidate { dist: cosine_dist(query, vectors[entry]), idx: entry };
        let mut visited = HashSet::from([entry]);
        let mut candidates = BinaryHeap::from([Reverse(start)]);
        let mut results = BinaryHeap::from([start]);

        while let Some(Reverse(nearest)) = candidates.pop() {
            if results.peek().is_some_and(|furthest| nearest.dist > furthest.dist) {
                break;
            }
            for &neighbor in &self.neighbors[nearest.idx][layer] {
                if !visited.insert(neighbor) {
                    continue;
                }
                let candidate = Candidate { dist: cosine_dist(query, vectors[neighbor]), idx: neighbor };
                let is_full = results.len() >= ef;
                if is_full && results.peek().is_some_and(|furthest| candidate.dist >= furthest.dist) {
                    continue;
                }
                candidates.push(Reverse(candidate));
                results.push(candidate);
                if results.len() > ef {
                    results.pop();
                }
            }
        }

        results.into_sorted_vec()
    }
}

impl Graph<'_> {
    fn insert(&mut self, idx: usize) {
        let level = random_level();
        self.index.neighbors.push(vec![Vec::new(); level + 1]);

        let Some(mut curr) = self.index.entry else {
            self.index.entry = Some(idx);
            self.index.max_level = level;
            return;
        };

        let query = self.vectors[idx];
        for layer in (level + 1..=self.index.max_level).rev() {
            curr = self.index.search_layer(self.vectors, query, curr, 1, layer)[0].idx;
        }

        for layer in (0..=level.min(self.index.max_level)).rev() {
            let candidates = self.index.search_layer(self.vectors, query, curr, EF_CONSTRUCTION, layer);
            let neighbors = self.select_diverse(&candidates, max_connections(layer));
            for &neighbor in &neighbors {
                self.connect(neighbor, idx, layer);
            }
            self.index.neighbors[idx][layer] = neighbors;
            curr = candidates[0].idx;
        }

        if level > self.index.max_level {
            self.index.max_level = level;
            self.index.entry = Some(idx);
        }
    }

    fn connect(&mut self, from: usize, to: usize, layer: usize) {
        let links = &mut self.index.neighbors[from][layer];
        links.push(to);
        if links.len() <= max_connections(layer) {
            return;
        }
        let base = self.vectors[from];
        let mut candidates: Vec<Candidate> = links
            .iter()
            .map(|&idx| Candidate { dist: cosine_dist(base, self.vectors[idx]), idx })
            .collect();
        candidates.sort();
        self.index.neighbors[from][layer] = self.select_diverse(&candidates, max_connections(layer));
    }

    fn select_diverse(&self, sorted_candidates: &[Candidate], m: usize) -> Vec<usize> {
        let mut selected: Vec<usize> = Vec::with_capacity(m);
        for candidate in sorted_candidates {
            if selected.len() >= m {
                break;
            }
            let vector = self.vectors[candidate.idx];
            if selected.iter().all(|&s| candidate.dist <= cosine_dist(vector, self.vectors[s])) {
                selected.push(candidate.idx);
            }
        }
        selected
    }
}
