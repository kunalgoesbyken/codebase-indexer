use anyhow::Result;
use ignore::types::TypesBuilder;
use ignore::WalkBuilder;
use std::path::Path;

/// Directory names that are always this tool's own artifacts, no matter where
/// they appear in the tree. `.indexer/` is only ever created by this indexer
/// (cache.bin, vectors.bin, bm25/), so pruning it at any depth is safe and also
/// covers the nested-workspace case (a monorepo member that was indexed on its
/// own). Indexing our own index is pure self-pollution.
const ALWAYS_PRUNED_DIRS: &[&str] = &[".indexer"];

/// Directory names that are pruned ONLY when they sit directly at the indexed
/// root. These names are ambiguous: `models/` at *our* repo root holds vendored
/// ML assets (model_int8.onnx 137MB, tokenizer_nomic.json 711KB,
/// special_tokens_map.json), but `models/` nested inside a user's project is
/// almost always genuine source code -- a Django `app/models/*.py` or a Rails
/// `app/models/*.rb` package is one of the most common layouts in existence.
/// Pruning that name everywhere would silently drop the most important files in
/// those codebases, so we anchor it to the root, which is exactly where
/// `main.rs` looks for the model assets (`root.join("models")`).
const ROOT_ONLY_PRUNED_DIRS: &[&str] = &["models"];

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

    // Compare against the walk root as the `ignore` crate yields it, so
    // root-anchored matching works for `.`, relative and absolute roots alike.
    let root_owned = root.to_path_buf();

    let mut walker = WalkBuilder::new(root);
    walker.types(file_types);
    walker.filter_entry(move |entry| {
        // depth() == 0 is the walk root itself. Never prune it: pruning the
        // root (e.g. when indexing a directory that happens to be named
        // `models`, or running `index` from inside `.indexer`) would make the
        // walk return zero files.
        if entry.depth() == 0 {
            return true;
        }
        if !entry.file_type().map_or(false, |t| t.is_dir()) {
            return true;
        }
        let Some(name) = entry.file_name().to_str() else {
            return true;
        };
        if ALWAYS_PRUNED_DIRS.contains(&name) {
            return false;
        }
        // Root-anchored: prune only a direct child of the indexed root.
        if ROOT_ONLY_PRUNED_DIRS.contains(&name)
            && entry.path().parent().map_or(false, |p| p == root_owned.as_path())
        {
            return false;
        }
        true
    });

    let mut files = Vec::new();
    for result in walker.build() {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    struct TempTree(PathBuf);

    impl TempTree {
        fn new(tag: &str) -> Self {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let dir = std::env::temp_dir().join(format!("walker-test-{tag}-{nanos}"));
            fs::create_dir_all(&dir).unwrap();
            TempTree(dir)
        }

        fn path(&self) -> &Path {
            &self.0
        }

        fn write(&self, rel: &str, contents: &str) {
            let full = self.0.join(rel);
            fs::create_dir_all(full.parent().unwrap()).unwrap();
            fs::write(full, contents).unwrap();
        }
    }

    impl Drop for TempTree {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    /// Relative, slash-normalized paths of the walk result, for stable asserts.
    fn rel_names(root: &Path, files: &[PathBuf]) -> Vec<String> {
        files
            .iter()
            .map(|p| {
                p.strip_prefix(root)
                    .unwrap_or(p)
                    .to_string_lossy()
                    .replace('\\', "/")
            })
            .collect()
    }

    #[test]
    fn excludes_indexer_and_root_models_dirs() {
        let tree = TempTree::new("exclude");
        tree.write("demo.rs", "fn main() {}\n");
        tree.write(".indexer/bm25/.managed.json", "{}\n");
        tree.write(".indexer/meta.json", "{}\n");
        tree.write("models/tokenizer_nomic.json", "{}\n");
        tree.write("models/special_tokens_map.json", "{}\n");
        tree.write("Cargo.lock", "# lock\n");

        let files = walk_source_files(tree.path()).unwrap();
        let names = rel_names(tree.path(), &files);

        assert!(names.contains(&"demo.rs".to_string()), "got {names:?}");
        assert!(
            !names.iter().any(|n| n.starts_with(".indexer/")),
            ".indexer/ leaked into results: {names:?}"
        );
        assert!(
            !names.iter().any(|n| n.starts_with("models/")),
            "root models/ leaked into results: {names:?}"
        );
        assert!(
            !names.iter().any(|n| n.ends_with(".lock")),
            "lock file leaked into results: {names:?}"
        );
        assert_eq!(names.len(), 1, "expected only demo.rs, got {names:?}");
    }

    /// Guards the Django/Rails case: a NESTED `models/` dir holds real source
    /// and must still be indexed. Only a root-level `models/` is ours.
    #[test]
    fn keeps_nested_models_dir_with_real_source() {
        let tree = TempTree::new("nested");
        tree.write("manage.py", "# django entrypoint\n");
        tree.write("app/models/user.py", "class User: pass\n");
        tree.write("app/models/order.py", "class Order: pass\n");
        tree.write("app/models/schema.json", "{}\n");
        tree.write("src/app/models/deep.py", "X = 1\n");
        // ...but a nested .indexer/ is still ours and still pruned.
        tree.write("app/.indexer/cache.json", "{}\n");

        let files = walk_source_files(tree.path()).unwrap();
        let names = rel_names(tree.path(), &files);

        for expected in [
            "manage.py",
            "app/models/user.py",
            "app/models/order.py",
            "app/models/schema.json",
            "src/app/models/deep.py",
        ] {
            assert!(
                names.contains(&expected.to_string()),
                "{expected} was wrongly excluded; got {names:?}"
            );
        }
        assert!(
            !names.iter().any(|n| n.contains(".indexer/")),
            "nested .indexer/ leaked into results: {names:?}"
        );
    }

    /// The classic trap: `filter_entry` also sees the root entry. Walking a
    /// directory named `models`, or `.indexer`, must still return its files.
    #[test]
    fn root_entry_is_never_pruned() {
        let tree = TempTree::new("rootentry");
        tree.write("models/main.rs", "fn main() {}\n");
        tree.write(".indexer/main.rs", "fn main() {}\n");

        let models_root = tree.path().join("models");
        let files = walk_source_files(&models_root).unwrap();
        assert_eq!(
            rel_names(&models_root, &files),
            vec!["main.rs".to_string()],
            "walking a dir literally named `models` returned nothing"
        );

        let indexer_root = tree.path().join(".indexer");
        let files = walk_source_files(&indexer_root).unwrap();
        assert_eq!(
            rel_names(&indexer_root, &files),
            vec!["main.rs".to_string()],
            "walking a dir literally named `.indexer` returned nothing"
        );
    }

    /// Root-anchoring must work when the root is the relative path `.`, which
    /// is what `main.rs` defaults to.
    #[test]
    fn relative_dot_root_prunes_and_returns_files() {
        let tree = TempTree::new("dotroot");
        tree.write("demo.rs", "fn main() {}\n");
        tree.write("models/tokenizer_nomic.json", "{}\n");
        tree.write(".indexer/bm25/.managed.json", "{}\n");

        // set_current_dir is process-global, so serialize against any other
        // cwd-sensitive test and always restore, even on panic.
        static CWD_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = CWD_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let original = std::env::current_dir().unwrap();
        std::env::set_current_dir(tree.path()).unwrap();
        let result = walk_source_files(Path::new("."));
        std::env::set_current_dir(original).unwrap();

        let files = result.unwrap();
        let names = rel_names(Path::new("."), &files);
        assert!(!files.is_empty(), "walking `.` returned zero files");
        assert!(
            names.iter().any(|n| n.ends_with("demo.rs")),
            "demo.rs missing from {names:?}"
        );
        assert!(
            !names.iter().any(|n| n.contains("models/")),
            "root models/ leaked when root was `.`: {names:?}"
        );
        assert!(
            !names.iter().any(|n| n.contains(".indexer/")),
            ".indexer/ leaked when root was `.`: {names:?}"
        );
    }

    /// The repo's own source files must still be found when indexing this
    /// project, while its `models/` and `.indexer/` artifacts are skipped.
    #[test]
    fn indexes_this_repo_src_but_not_its_artifacts() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let files = walk_source_files(repo_root).unwrap();
        let names = rel_names(repo_root, &files);

        assert!(
            names.contains(&"src/walker.rs".to_string()),
            "src/walker.rs missing from {names:?}"
        );
        assert!(
            names.iter().filter(|n| n.starts_with("src/")).count() >= 5,
            "expected several src/*.rs files, got {names:?}"
        );
        assert!(
            !names.iter().any(|n| n.starts_with("models/")),
            "vendored models/ leaked: {names:?}"
        );
        assert!(
            !names.iter().any(|n| n.starts_with(".indexer/")),
            ".indexer/ leaked: {names:?}"
        );
        assert!(
            !names.iter().any(|n| n == "Cargo.lock"),
            "Cargo.lock leaked: {names:?}"
        );
    }
}
