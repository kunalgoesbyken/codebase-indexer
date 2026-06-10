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
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "codebase-indexer", about = "Semantic codebase indexer")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Index {
        path: Option<PathBuf>,
    },
    Query {
        query: String,
        path: Option<PathBuf>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Index { path } => cmd_index(path.unwrap_or_else(|| PathBuf::from("."))),
        Command::Query { query, path } => {
            cmd_query(&query, path.unwrap_or_else(|| PathBuf::from(".")))
        }
    }
}

fn cmd_index(root: PathBuf) -> Result<()> {
    let index_dir = root.join(".indexer");
    let cache_path = index_dir.join("cache.bin");
    let vectors_path = index_dir.join("vectors.bin");
    let bm25_dir = index_dir.join("bm25");
    let model_dir = root.join("models");

    println!("Scanning {}...", root.display());
    let files = walker::walk_source_files(&root)?;
    if files.is_empty() {
        println!("No source files found.");
        return Ok(());
    }
    println!("Found {} source files", files.len());

    let cache = delta::load_cache(&cache_path)?;
    let changed = delta::detect_changes(&files, &cache, &root)?;
    println!("{} files modified/new", changed.len());

    if changed.is_empty() {
        println!("Nothing to index.");
        return Ok(());
    }

    let mut embedder = embedder::Embedder::new(&model_dir)?;
    let mut store = store::VectorStore::new(&vectors_path)?;

    let bm25_exists = bm25_dir.exists();
    let mut bm25 = if bm25_exists {
        bm25::Bm25Index::open(&bm25_dir)?
    } else {
        bm25::Bm25Index::new(&bm25_dir)?
    };

    let mut total_chunks = 0;

    for path in &changed {
        println!("  Processing {}...", path.display());
        let rel = path
            .strip_prefix(&root)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();
        let source = std::fs::read_to_string(path)?;
        let chunks = chunker::chunk_file_with_rel(path, &source, &rel)?;
        println!("    -> {} chunks", chunks.len());

        if chunks.is_empty() {
            continue;
        }

        match embedder.embed(&chunks) {
            Ok(embeddings) => {
                if let Err(e) = bm25.delete_file(&rel) {
                    eprintln!("  BM25 delete failed for {}: {e}", rel);
                }
                if let Err(e) = bm25.add_chunks(&chunks) {
                    eprintln!("  BM25 index failed for {}: {e}", rel);
                }
                let records: Vec<types::IndexRecord> = chunks
                    .into_iter()
                    .zip(embeddings)
                    .map(|(chunk, embedding)| types::IndexRecord { chunk, embedding })
                    .collect();
                let count = records.len();
                store.append(records)?;
                total_chunks += count;
                println!("    -> embedded {count} chunks");
            }
            Err(e) => {
                eprintln!("  Embedding failed for {}: {e}", path.display());
            }
        }
    }

    bm25.commit()?;

    println!("Wrote {total_chunks} chunks to vectors.bin (append-only log)");

    let new_states: Vec<_> = files
        .iter()
        .map(|path| {
            let rel = path
                .strip_prefix(&root)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string();
            let hash = delta::file_hash(path).unwrap_or_default();
            let modified = std::fs::metadata(path)
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            types::FileState {
                path: rel,
                hash,
                last_modified: modified,
            }
        })
        .collect();
    delta::save_cache(&cache_path, &new_states)?;
    println!("Cache updated.");

    Ok(())
}

fn cmd_query(query: &str, root: PathBuf) -> Result<()> {
    let index_dir = root.join(".indexer");
    let vectors_path = index_dir.join("vectors.bin");
    let bm25_dir = index_dir.join("bm25");
    let model_dir = root.join("models");

    if !vectors_path.exists() {
        println!("No index found. Run `index` first.");
        return Ok(());
    }

    let store = store::VectorStore::new(&vectors_path)?;
    if store.records().is_empty() {
        println!("Index is empty. Run `index` first.");
        return Ok(());
    }
    println!("Loaded {} indexed records", store.records().len());

    let mut embedder = embedder::Embedder::new(&model_dir)?;
    let query_embedding = embedder.embed_by_text(query)?;

    println!("Searching for: {query}");

    let threshold = 1000;
    let vector_results = if store.records().len() < threshold {
        store.search_exact(&query_embedding, 20)
    } else {
        store.search(&query_embedding, 20)
    };

    let bm25_results = if bm25_dir.exists() {
        match bm25::Bm25Index::open(&bm25_dir) {
            Ok(bm25) => match bm25.search(query, 20) {
                Ok(results) => results,
                Err(e) => {
                    eprintln!("BM25 search error: {e}");
                    Vec::new()
                }
            },
            Err(e) => {
                eprintln!("BM25 open error: {e}");
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };

    let fused = query::rrf_fuse(&bm25_results, &vector_results, store.records(), 60, 5);

    if fused.is_empty() {
        println!("No matching results found.");
        return Ok(());
    }

    for (i, (file_path, start_line, text, score)) in fused.iter().enumerate() {
        println!("\n--- Result {} (score: {:.4}) ---", i + 1, score);
        println!("{}:{}", file_path, start_line);
        if !text.is_empty() {
            println!("{}", text);
        }
    }

    Ok(())
}