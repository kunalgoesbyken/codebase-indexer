use crate::types::FileState;
use anyhow::Result;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

pub fn rel_path(path: &Path, root: &Path) -> String {
    path.strip_prefix(root).unwrap_or(path).to_string_lossy().to_string()
}

fn file_state(path: &Path, rel: String) -> Result<FileState> {
    let hash = format!("{:x}", Sha256::digest(std::fs::read(path)?));
    let last_modified = std::fs::metadata(path)?
        .modified()?
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    Ok(FileState { path: rel, hash, last_modified })
}

pub fn load_cache(cache_path: &Path) -> Result<HashMap<String, FileState>> {
    if !cache_path.exists() {
        return Ok(HashMap::new());
    }
    let states: Vec<FileState> = bincode::deserialize(&std::fs::read(cache_path)?)?;
    Ok(states.into_iter().map(|s| (s.path.clone(), s)).collect())
}

pub fn save_cache(cache_path: &Path, cache: &HashMap<String, FileState>) -> Result<()> {
    if let Some(parent) = cache_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let states: Vec<&FileState> = cache.values().collect();
    std::fs::write(cache_path, bincode::serialize(&states)?)?;
    Ok(())
}

pub fn detect_changes(
    files: &[PathBuf],
    cache: &HashMap<String, FileState>,
    root: &Path,
) -> Result<Vec<(PathBuf, FileState)>> {
    let mut changed = Vec::new();
    for path in files {
        let state = file_state(path, rel_path(path, root))?;
        let is_unchanged = cache
            .get(&state.path)
            .is_some_and(|cached| cached.hash == state.hash && cached.last_modified == state.last_modified);
        if !is_unchanged {
            changed.push((path.clone(), state));
        }
    }
    Ok(changed)
}
