use anyhow::Result;
use ignore::types::TypesBuilder;
use ignore::WalkBuilder;
use std::path::Path;

pub fn walk_source_files(root: &Path) -> Result<Vec<std::path::PathBuf>> {
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

    let mut files = Vec::new();
    for result in WalkBuilder::new(root).types(file_types).build() {
        let entry = result?;
        if !entry.file_type().map_or(false, |t| t.is_file()) {
            continue;
        }
        let path = entry.path();
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.ends_with(".lock") || name == "Cargo.lock" {
                continue;
            }
        }
        files.push(entry.path().to_path_buf());
    }
    Ok(files)
}