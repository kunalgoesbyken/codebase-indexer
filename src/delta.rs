use crate::types::FileState;
use anyhow::Result;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;

pub fn file_hash(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn load_cache(cache_path: &Path) -> Result<HashMap<String, FileState>> {
    if !cache_path.exists() {
        return Ok(HashMap::new());
    }
    let bytes = std::fs::read(cache_path)?;
    let states: Vec<FileState> = bincode::deserialize(&bytes)?;
    Ok(states.into_iter().map(|s| (s.path.clone(), s)).collect())
}

pub fn save_cache(cache_path: &Path, states: &[FileState]) -> Result<()> {
    if let Some(parent) = cache_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let bytes = bincode::serialize(states)?;
    std::fs::write(cache_path, bytes)?;
    Ok(())
}

pub fn detect_changes(
    files: &[std::path::PathBuf],
    cache: &HashMap<String, FileState>,
    root: &Path,
) -> Result<Vec<std::path::PathBuf>> {
    let mut changed = Vec::new();
    for path in files {
        let rel = path
            .strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();
        let hash = file_hash(path)?;
        let modified = std::fs::metadata(path)?.modified()?;
        let mtime = modified
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let needs_update = cache.get(&rel).map_or(true, |cached| {
            cached.hash != hash || cached.last_modified != mtime
        });
        if needs_update {
            changed.push(path.to_path_buf());
        }
    }
    Ok(changed)
}

