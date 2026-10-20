mod bm25;
mod chunker;
mod delta;
mod embedder;
mod hnsw;
mod query;
mod store;
mod types;
mod walker;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

const DEFAULT_MODEL_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/models");
const CANDIDATES_PER_RANKER: usize = 20;
const RRF_K: f32 = 60.0;
const TOP_K: usize = 5;

#[derive(Parser)]
#[command(name = "codebase-indexer", about = "Semantic codebase indexer")]
struct Cli {
    #[arg(long, global = true, env = "CODEBASE_INDEXER_MODELS", default_value = DEFAULT_MODEL_DIR)]
    models: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Index {
        #[arg(default_value = ".")]
        path: PathBuf,
    },
    Query {
        query: String,
        #[arg(default_value = ".")]
        path: PathBuf,
    },
}

struct IndexPaths {
    cache: PathBuf,
    vectors: PathBuf,
    bm25: PathBuf,
}

impl IndexPaths {
    fn new(root: &Path) -> Result<Self> {
        let dir = root.join(".indexer");
        reject_symlinks(&dir)?;
        Ok(Self {
            cache: dir.join("cache.bin"),
            vectors: dir.join("vectors.bin"),
            bm25: dir.join("bm25"),
        })
    }
}

fn reject_symlinks(dir: &Path) -> Result<()> {
    let mut pending = vec![dir.to_path_buf()];
    while let Some(path) = pending.pop() {
        let Ok(meta) = std::fs::symlink_metadata(&path) else {
            continue;
        };
        if meta.file_type().is_symlink() {
            anyhow::bail!("refusing to use index containing a symlink: {}", path.display());
        }
        if meta.is_dir() {
            for entry in std::fs::read_dir(&path)? {
                pending.push(entry?.path());
            }
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Index { path } => cmd_index(&path, &cli.models),
        Command::Query { query, path } => cmd_query(&query, &path, &cli.models),
    }
}

fn index_file(
    path: &Path,
    rel: &str,
    embedder: &mut embedder::Embedder,
    store: &mut store::VectorStore,
    bm25: &mut bm25::Bm25Index,
) -> Result<usize> {
    let source = std::fs::read_to_string(path)?;
    let chunks = chunker::chunk_file_with_rel(path, &source, rel)?;
    let embeddings = embedder.embed(&chunks)?;

    store.tombstone(rel)?;
    bm25.delete_file(rel)?;
    bm25.add_chunks(&chunks)?;
    let records: Vec<types::IndexRecord> = chunks
        .into_iter()
        .zip(embeddings)
        .map(|(chunk, embedding)| types::IndexRecord { chunk, embedding })
        .collect();
    let count = records.len();
    store.append(records)?;
    Ok(count)
}

fn cmd_index(root: &Path, model_dir: &Path) -> Result<()> {
    let paths = IndexPaths::new(root)?;

    println!("Scanning {}...", root.display());
    let files = walker::walk_source_files(root)?;
    println!("Found {} source files", files.len());

    let mut cache = delta::load_cache(&paths.cache)?;
    let walked: HashSet<String> = files.iter().map(|p| delta::rel_path(p, root)).collect();
    let mut deleted: Vec<String> = cache.keys().filter(|rel| !walked.contains(*rel)).cloned().collect();
    deleted.sort();

    let changed = delta::detect_changes(&files, &cache, root)?;
    println!("{} files modified/new, {} deleted", changed.len(), deleted.len());
    if changed.is_empty() && deleted.is_empty() {
        println!("Nothing to index.");
        return Ok(());
    }

    let mut store = store::VectorStore::new(&paths.vectors)?;
    let mut bm25 = bm25::Bm25Index::open_or_create(&paths.bm25)?;

    for rel in &deleted {
        println!("  Purging {rel}...");
        store.tombstone(rel)?;
        bm25.delete_file(rel)?;
        cache.remove(rel);
    }

    let mut total_chunks = 0;
    if !changed.is_empty() {
        let mut embedder = embedder::Embedder::new(model_dir)?;
        for (path, state) in changed {
            println!("  Indexing {}...", state.path);
            match index_file(&path, &state.path, &mut embedder, &mut store, &mut bm25) {
                Ok(count) => {
                    total_chunks += count;
                    cache.insert(state.path.clone(), state);
                }
                Err(e) => {
                    eprintln!("  Failed {}: {e}", state.path);
                    cache.remove(&state.path);
                }
            }
        }
    }

    bm25.commit()?;
    store.compact_if_needed()?;
    store.rebuild_ann_index()?;
    delta::save_cache(&paths.cache, &cache)?;
    println!("Indexed {total_chunks} chunks.");
    Ok(())
}

fn search_bm25(bm25_dir: &Path, query: &str) -> Vec<types::Chunk> {
    if !bm25_dir.exists() {
        return Vec::new();
    }
    bm25::Bm25Index::open_or_create(bm25_dir)
        .and_then(|bm25| bm25.search(query, CANDIDATES_PER_RANKER))
        .unwrap_or_else(|e| {
            eprintln!("BM25 search error: {e}");
            Vec::new()
        })
}

fn cmd_query(query: &str, root: &Path, model_dir: &Path) -> Result<()> {
    let paths = IndexPaths::new(root)?;
    let store = store::VectorStore::new(&paths.vectors)?;
    if store.records().is_empty() {
        println!("No index found. Run `index` first.");
        return Ok(());
    }
    println!("Loaded {} indexed records", store.records().len());
    println!("Searching for: {query}");

    let query_embedding = embedder::Embedder::new(model_dir)?.embed_by_text(query)?;
    let vector_hits = store.search(&query_embedding, CANDIDATES_PER_RANKER);
    let bm25_hits = search_bm25(&paths.bm25, query);
    let fused = query::rrf_fuse(&[bm25_hits.iter().collect(), vector_hits], RRF_K, TOP_K);

    if fused.is_empty() {
        println!("No matching results found.");
        return Ok(());
    }
    for (i, (chunk, score)) in fused.iter().enumerate() {
        println!("\n--- Result {} (score: {score:.4}) ---", i + 1);
        println!("{}:{}-{}", chunk.file_path, chunk.start_line, chunk.end_line);
        println!("{}", chunk.text);
    }
    Ok(())
}
