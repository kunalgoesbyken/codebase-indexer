use crate::hnsw::HnswIndex;
use crate::types::IndexRecord;
use anyhow::Result;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

const TOMBSTONE_MARKER: u32 = 0xFFFFFFFF;

fn read_log(bytes: &[u8]) -> Vec<IndexRecord> {
    let mut records = Vec::new();
    let mut pos = 0;
    while pos + 4 <= bytes.len() {
        let len = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        pos += 4;
        if len == TOMBSTONE_MARKER {
            if pos + 4 <= bytes.len() {
                let path_len = u32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
                pos += 4;
                if pos + path_len as usize <= bytes.len() {
                    let tomb_path = std::str::from_utf8(&bytes[pos..pos + path_len as usize]).unwrap_or("");
                    records.retain(|r: &IndexRecord| r.chunk.file_path != tomb_path);
                    pos += path_len as usize;
                }
            }
        } else if len > 0 {
            let end = pos + len as usize;
            if end <= bytes.len() {
                if let Ok(record) = bincode::deserialize::<IndexRecord>(&bytes[pos..end]) {
                    records.push(record);
                }
                pos = end;
            }
        }
    }
    records
}

fn write_record<W: Write>(writer: &mut W, record: &IndexRecord) -> Result<()> {
    let data = bincode::serialize(record)?;
    let len = data.len() as u32;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(&data)?;
    Ok(())
}

fn write_tombstone<W: Write>(writer: &mut W, path: &str) -> Result<()> {
    let path_bytes = path.as_bytes();
    writer.write_all(&TOMBSTONE_MARKER.to_le_bytes())?;
    writer.write_all(&(path_bytes.len() as u32).to_le_bytes())?;
    writer.write_all(path_bytes)?;
    Ok(())
}

fn build_hnsw(records: &[IndexRecord]) -> HnswIndex {
    let mut hnsw = HnswIndex::new();
    for record in records {
        hnsw.insert(record.embedding.clone());
    }
    hnsw
}

pub struct VectorStore {
    records: Vec<IndexRecord>,
    path: Box<Path>,
    hnsw: HnswIndex,
}

impl VectorStore {
    pub fn new(path: &Path) -> Result<Self> {
        let records = if path.exists() {
            let bytes = std::fs::read(path)?;
            read_log(&bytes)
        } else {
            Vec::new()
        };
        let hnsw = build_hnsw(&records);
        Ok(Self {
            records,
            path: path.into(),
            hnsw,
        })
    }

    pub fn tombstone(&mut self, file_path: &str) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        write_tombstone(&mut file, file_path)?;
        file.flush()?;
        self.records.retain(|r| r.chunk.file_path != file_path);
        self.hnsw = build_hnsw(&self.records);
        Ok(())
    }

    pub fn append(&mut self, records: Vec<IndexRecord>) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        for record in &records {
            write_record(&mut file, record)?;
        }
        file.flush()?;
        self.records.extend(records);
        self.hnsw = build_hnsw(&self.records);
        Ok(())
    }

    pub fn records(&self) -> &[IndexRecord] {
        &self.records
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let ef = (k * 2).max(50).min(self.records.len());
        let indices = self.hnsw.search(query, k, ef);
        let mut results: Vec<(usize, f32)> = indices
            .into_iter()
            .map(|idx| {
                let actual_score =
                    crate::query::cosine_similarity(query, &self.records[idx].embedding);
                (idx, actual_score)
            })
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    pub fn search_exact(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = self
            .records
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let score = crate::query::cosine_similarity(query, &r.embedding);
                (i, score)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}
