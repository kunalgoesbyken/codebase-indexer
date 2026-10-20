use crate::hnsw::HnswIndex;
use crate::types::{Chunk, IndexRecord};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const TOMBSTONE_MARKER: u32 = 0xFFFFFFFF;
const ANN_MIN_RECORDS: usize = 1000;

struct LogContents {
    records: Vec<IndexRecord>,
    appended: usize,
    valid_len: usize,
}

#[derive(Serialize, Deserialize)]
struct AnnSnapshot {
    log_len: u64,
    graph: HnswIndex,
}

fn read_u32(bytes: &[u8], pos: usize) -> Option<u32> {
    let word = bytes.get(pos..pos + 4)?;
    Some(u32::from_le_bytes(word.try_into().ok()?))
}

fn read_log(bytes: &[u8]) -> LogContents {
    let mut records: Vec<IndexRecord> = Vec::new();
    let mut appended = 0;
    let mut pos = 0;
    while let Some(len) = read_u32(bytes, pos) {
        let body = pos + 4;
        if len == TOMBSTONE_MARKER {
            let Some(path_len) = read_u32(bytes, body) else { break };
            let path_start = body + 4;
            let path_end = path_start + path_len as usize;
            let Some(path) = bytes.get(path_start..path_end) else { break };
            let path = String::from_utf8_lossy(path);
            records.retain(|r| r.chunk.file_path.as_str() != path.as_ref());
            pos = path_end;
        } else {
            let end = body + len as usize;
            let Some(payload) = bytes.get(body..end) else { break };
            if let Ok(record) = bincode::deserialize(payload) {
                records.push(record);
                appended += 1;
            }
            pos = end;
        }
    }
    LogContents { records, appended, valid_len: pos }
}

fn write_record<W: Write>(writer: &mut W, record: &IndexRecord) -> Result<()> {
    let data = bincode::serialize(record)?;
    writer.write_all(&(data.len() as u32).to_le_bytes())?;
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

pub struct VectorStore {
    records: Vec<IndexRecord>,
    path: PathBuf,
    dead: usize,
}

impl VectorStore {
    pub fn new(path: &Path) -> Result<Self> {
        let bytes = if path.exists() { std::fs::read(path)? } else { Vec::new() };
        let log = read_log(&bytes);
        if log.valid_len < bytes.len() {
            OpenOptions::new().write(true).open(path)?.set_len(log.valid_len as u64)?;
        }
        Ok(Self {
            dead: log.appended - log.records.len(),
            records: log.records,
            path: path.to_path_buf(),
        })
    }

    pub fn records(&self) -> &[IndexRecord] {
        &self.records
    }

    fn open_log(&self) -> Result<BufWriter<File>> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new().create(true).append(true).open(&self.path)?;
        Ok(BufWriter::new(file))
    }

    pub fn tombstone(&mut self, file_path: &str) -> Result<()> {
        let removed = self.records.iter().filter(|r| r.chunk.file_path == file_path).count();
        if removed == 0 {
            return Ok(());
        }
        let mut writer = self.open_log()?;
        write_tombstone(&mut writer, file_path)?;
        writer.flush()?;
        self.records.retain(|r| r.chunk.file_path != file_path);
        self.dead += removed;
        Ok(())
    }

    pub fn append(&mut self, records: Vec<IndexRecord>) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        let mut writer = self.open_log()?;
        for record in &records {
            write_record(&mut writer, record)?;
        }
        writer.flush()?;
        self.records.extend(records);
        Ok(())
    }

    pub fn compact_if_needed(&mut self) -> Result<()> {
        if self.dead <= self.records.len() {
            return Ok(());
        }
        let tmp_path = self.path.with_extension("bin.tmp");
        let mut writer = BufWriter::new(File::create(&tmp_path)?);
        for record in &self.records {
            write_record(&mut writer, record)?;
        }
        writer.into_inner()?.sync_all()?;
        std::fs::rename(&tmp_path, &self.path)?;
        self.dead = 0;
        Ok(())
    }

    fn ann_path(&self) -> PathBuf {
        self.path.with_extension("hnsw")
    }

    fn log_len(&self) -> Result<u64> {
        Ok(std::fs::metadata(&self.path)?.len())
    }

    fn vectors(&self) -> Vec<&[f32]> {
        self.records.iter().map(|r| r.embedding.as_slice()).collect()
    }

    pub fn rebuild_ann_index(&self) -> Result<()> {
        let ann_path = self.ann_path();
        if self.records.len() < ANN_MIN_RECORDS {
            if ann_path.exists() {
                std::fs::remove_file(ann_path)?;
            }
            return Ok(());
        }
        let snapshot = AnnSnapshot {
            log_len: self.log_len()?,
            graph: HnswIndex::build(&self.vectors()),
        };
        std::fs::write(ann_path, bincode::serialize(&snapshot)?)?;
        Ok(())
    }

    fn load_ann_index(&self) -> Option<HnswIndex> {
        let bytes = std::fs::read(self.ann_path()).ok()?;
        let snapshot: AnnSnapshot = bincode::deserialize(&bytes).ok()?;
        let is_current = snapshot.log_len == self.log_len().ok()?;
        let is_usable = is_current && snapshot.graph.is_consistent(self.records.len());
        is_usable.then_some(snapshot.graph)
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<&Chunk> {
        let indices = match self.load_ann_index() {
            Some(graph) => graph.search(&self.vectors(), query, k, (k * 2).max(50)),
            None => self.exact_search(query, k),
        };
        indices.into_iter().map(|idx| &self.records[idx].chunk).collect()
    }

    fn exact_search(&self, query: &[f32], k: usize) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = self
            .records
            .iter()
            .enumerate()
            .map(|(idx, r)| (idx, crate::query::cosine_similarity(query, &r.embedding)))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(k);
        scored.into_iter().map(|(idx, _)| idx).collect()
    }
}
