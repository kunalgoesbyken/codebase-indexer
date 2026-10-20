use anyhow::Result;
use ignore::types::TypesBuilder;
use ignore::{DirEntry, WalkBuilder};
use std::path::{Path, PathBuf};

const ALWAYS_PRUNED_DIRS: &[&str] = &[".indexer"];
const ROOT_ONLY_PRUNED_DIRS: &[&str] = &["models"];

fn is_lockfile(path: &Path) -> bool {
    let is_lock_ext = path.extension().is_some_and(|ext| ext == "lock");
    let is_lock_stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .is_some_and(|stem| stem.ends_with("-lock"));
    is_lock_ext || is_lock_stem
}

fn is_pruned_dir(entry: &DirEntry, root: &Path) -> bool {
    if entry.depth() == 0 || !entry.file_type().is_some_and(|t| t.is_dir()) {
        return false;
    }
    let Some(name) = entry.file_name().to_str() else {
        return false;
    };
    let is_root_child = entry.path().parent() == Some(root);
    ALWAYS_PRUNED_DIRS.contains(&name) || (is_root_child && ROOT_ONLY_PRUNED_DIRS.contains(&name))
}

pub fn walk_source_files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut types_builder = TypesBuilder::new();
    types_builder.add_defaults();
    types_builder.add("ts", "*.ts")?;
    types_builder.add("tsx", "*.tsx")?;
    types_builder.add("markdown", "*.md")?;
    types_builder.add("yaml", "*.yaml,*.yml")?;
    types_builder.add("toml", "*.toml")?;
    types_builder.add("json", "*.json")?;
    let file_types = types_builder
        .select("rust")
        .select("go")
        .select("python")
        .select("java")
        .select("cpp")
        .select("ts")
        .select("tsx")
        .select("markdown")
        .select("yaml")
        .select("toml")
        .select("json")
        .build()
        .map_err(|e| anyhow::anyhow!("failed to build file types: {e}"))?;

    let root_owned = root.to_path_buf();
    let mut walker = WalkBuilder::new(root);
    walker.types(file_types);
    walker.filter_entry(move |entry| !is_pruned_dir(entry, &root_owned));

    let mut files = Vec::new();
    for result in walker.build() {
        let entry = result?;
        let is_file = entry.file_type().is_some_and(|t| t.is_file());
        if is_file && !is_lockfile(entry.path()) {
            files.push(entry.into_path());
        }
    }
    Ok(files)
}
