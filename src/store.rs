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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Chunk;
    use std::path::PathBuf;

    /// Unique scratch dir per test so parallel test threads never collide.
    fn scratch(name: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU32, Ordering};
        static SEQ: AtomicU32 = AtomicU32::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "codebase-indexer-test-{name}-{}-{nanos}-{seq}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch dir");
        dir
    }

    fn record(file_path: &str, start_line: usize, text: &str) -> IndexRecord {
        IndexRecord {
            chunk: Chunk {
                file_path: file_path.to_string(),
                start_line,
                end_line: start_line + 2,
                text: text.to_string(),
            },
            // Fake embedding; no ONNX model involved.
            embedding: vec![0.1, 0.2, 0.3, 0.4],
        }
    }

    fn texts(store: &VectorStore) -> Vec<String> {
        store
            .records()
            .iter()
            .map(|r| r.chunk.text.clone())
            .collect()
    }

    /// Regression test for BUG 1: `cmd_index` used to append a changed file's new
    /// chunks without ever removing its previous chunks, so the append-only log
    /// accumulated stale records forever and searches returned deleted symbols.
    ///
    /// Asserts the tombstone-then-append sequence the fix relies on leaves only
    /// the new records, both in memory and after replaying the log from disk.
    #[test]
    fn tombstone_then_append_keeps_only_new_records() {
        let dir = scratch("tombstone");
        let vectors_path = dir.join("vectors.bin");

        let mut store = VectorStore::new(&vectors_path).expect("new store");
        store
            .append(vec![
                record("a.rs", 1, "alpha_unique_marker_one"),
                record("a.rs", 9, "alpha_second_stale_chunk"),
                record("b.rs", 1, "other_file_untouched"),
            ])
            .expect("initial append");
        assert_eq!(store.records().len(), 3);

        // The fix: drop the file's prior chunks, then append the new ones.
        store.tombstone("a.rs").expect("tombstone");
        store
            .append(vec![record("a.rs", 1, "beta_unique_marker_two")])
            .expect("re-append");

        // In-memory state: only the new a.rs chunk plus untouched b.rs.
        let mut in_memory = texts(&store);
        in_memory.sort();
        assert_eq!(
            in_memory,
            vec!["beta_unique_marker_two", "other_file_untouched"],
            "stale records survived in memory"
        );
        assert!(
            !texts(&store)
                .iter()
                .any(|t| t.contains("alpha_unique_marker_one")),
            "the stale symbol is still searchable in memory"
        );

        // On-disk persistence: re-open and verify read_log replays the tombstone.
        let reopened = VectorStore::new(&vectors_path).expect("reopen store");
        let mut on_disk = texts(&reopened);
        on_disk.sort();
        assert_eq!(
            on_disk,
            vec!["beta_unique_marker_two", "other_file_untouched"],
            "tombstone was not replayed when reading the log from disk"
        );

        // The log itself is append-only, so the stale bytes are still present;
        // what matters is that replay excludes them from the live record set.
        let raw = std::fs::read(&vectors_path).expect("read log");
        assert!(
            raw.windows(23)
                .any(|w| w == b"alpha_unique_marker_one"),
            "sanity check: the log should still physically contain the stale bytes"
        );
        assert_eq!(
            reopened
                .records()
                .iter()
                .filter(|r| r.chunk.file_path == "a.rs")
                .count(),
            1,
            "a.rs should have exactly one live chunk after tombstone + re-append"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A tombstone must be durable on its own: appending nothing afterwards still
    /// leaves the file purged once the log is replayed from disk. This is the case
    /// BUG 2 depends on (a file deleted from disk is tombstoned and never re-added).
    #[test]
    fn tombstone_alone_persists_across_reopen() {
        let dir = scratch("purge");
        let vectors_path = dir.join("vectors.bin");

        let mut store = VectorStore::new(&vectors_path).expect("new store");
        store
            .append(vec![
                record("gone.rs", 1, "gone_unique_marker"),
                record("kept.rs", 1, "kept_unique_marker"),
            ])
            .expect("append");

        store.tombstone("gone.rs").expect("tombstone");
        assert_eq!(texts(&store), vec!["kept_unique_marker"]);

        let reopened = VectorStore::new(&vectors_path).expect("reopen store");
        assert_eq!(
            texts(&reopened),
            vec!["kept_unique_marker"],
            "purged file came back after reopening the log"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Tombstoning a path that was never indexed must be a harmless no-op, since
    /// the index loop tombstones every changed file unconditionally.
    #[test]
    fn tombstone_unknown_path_is_noop() {
        let dir = scratch("noop");
        let vectors_path = dir.join("vectors.bin");

        let mut store = VectorStore::new(&vectors_path).expect("new store");
        store
            .append(vec![record("a.rs", 1, "only_chunk")])
            .expect("append");
        store.tombstone("never_indexed.rs").expect("tombstone");

        assert_eq!(texts(&store), vec!["only_chunk"]);
        let reopened = VectorStore::new(&vectors_path).expect("reopen store");
        assert_eq!(texts(&reopened), vec!["only_chunk"]);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
