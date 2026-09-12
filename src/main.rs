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

fn rel_path(path: &std::path::Path, root: &std::path::Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string()
}

fn cmd_index(root: PathBuf) -> Result<()> {
    let index_dir = root.join(".indexer");
    let cache_path = index_dir.join("cache.bin");
    let vectors_path = index_dir.join("vectors.bin");
    let bm25_dir = index_dir.join("bm25");
    let model_dir = root.join("models");

    println!("Scanning {}...", root.display());
    let files = walker::walk_source_files(&root)?;
    println!("Found {} source files", files.len());

    let cache = delta::load_cache(&cache_path)?;

    // Relative paths of everything currently on disk, in the exact same form
    // used as cache keys (strip_prefix + to_string_lossy).
    let walked_rels: std::collections::HashSet<String> =
        files.iter().map(|p| rel_path(p, &root)).collect();

    // BUG 2: files present in the cache but no longer on disk must be purged
    // from BOTH the vector store and the BM25 index.
    let mut deleted: Vec<String> = cache
        .keys()
        .filter(|rel| !walked_rels.contains(*rel))
        .cloned()
        .collect();
    deleted.sort();

    let changed = delta::detect_changes(&files, &cache, &root)?;
    println!("{} files modified/new", changed.len());
    if !deleted.is_empty() {
        println!("{} files deleted from disk", deleted.len());
    }

    if changed.is_empty() && deleted.is_empty() {
        println!("Nothing to index.");
        return Ok(());
    }

    // Store and BM25 are opened before the embedder so that a purge-only run
    // (deletions but no new/modified files) does not require the ONNX model.
    let mut store = store::VectorStore::new(&vectors_path)?;

    let bm25_exists = bm25_dir.exists();
    let mut bm25 = if bm25_exists {
        bm25::Bm25Index::open(&bm25_dir)?
    } else {
        bm25::Bm25Index::new(&bm25_dir)?
    };

    for rel in &deleted {
        println!("  Purging deleted {}...", rel);
        if let Err(e) = store.tombstone(rel) {
            eprintln!("  Vector tombstone failed for {}: {e}", rel);
        }
        if let Err(e) = bm25.delete_file(rel) {
            eprintln!("  BM25 delete failed for {}: {e}", rel);
        }
    }

    // Relative paths that are safe to record in the cache: either indexed
    // successfully this run, or parsed successfully with genuinely no chunks.
    let mut indexed_ok: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut total_chunks = 0;

    if !changed.is_empty() {
        let mut embedder = embedder::Embedder::new(&model_dir)?;

        for path in &changed {
            println!("  Processing {}...", path.display());
            let rel = rel_path(path, &root);
            let source = match std::fs::read_to_string(path) {
                Ok(source) => source,
                Err(e) => {
                    // Not cached: the next run must retry this file.
                    eprintln!("  Read failed for {}: {e}", path.display());
                    continue;
                }
            };
            let chunks = match chunker::chunk_file_with_rel(path, &source, &rel) {
                Ok(chunks) => chunks,
                Err(e) => {
                    eprintln!("  Chunking failed for {}: {e}", path.display());
                    continue;
                }
            };
            println!("    -> {} chunks", chunks.len());

            if chunks.is_empty() {
                // Parsed fine, genuinely nothing to index. Still drop any prior
                // chunks for it, then cache it so we don't re-parse every run.
                if let Err(e) = store.tombstone(&rel) {
                    eprintln!("  Vector tombstone failed for {}: {e}", rel);
                }
                if let Err(e) = bm25.delete_file(&rel) {
                    eprintln!("  BM25 delete failed for {}: {e}", rel);
                }
                indexed_ok.insert(rel);
                continue;
            }

            match embedder.embed(&chunks) {
                Ok(embeddings) => {
                    // BUG 1: drop the file's prior chunks from the vector store,
                    // symmetric with the BM25 delete, before appending new ones.
                    if let Err(e) = store.tombstone(&rel) {
                        eprintln!("  Vector tombstone failed for {}: {e}", rel);
                    }
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
                    indexed_ok.insert(rel);
                    println!("    -> embedded {count} chunks");
                }
                Err(e) => {
                    // BUG 3: leave `rel` out of `indexed_ok` so it is absent from
                    // the new cache and gets retried on the next run.
                    eprintln!("  Embedding failed for {}: {e}", path.display());
                }
            }
        }
    }

    bm25.commit()?;

    println!("Wrote {total_chunks} chunks to vectors.bin (append-only log)");

    // BUG 3: the new cache holds only (a) files indexed successfully this run and
    // (b) files that were already cached and are still unchanged on disk.
    // Files that errored are omitted so the next run retries them, and deleted
    // files drop out because they are not in `walked_rels`.
    let changed_rels: std::collections::HashSet<String> =
        changed.iter().map(|p| rel_path(p, &root)).collect();

    let mut new_states: Vec<types::FileState> = Vec::new();
    for path in &files {
        let rel = rel_path(path, &root);
        if !indexed_ok.contains(&rel) {
            // Unchanged file: carry its existing cache entry forward untouched.
            if !changed_rels.contains(&rel) {
                if let Some(existing) = cache.get(&rel) {
                    new_states.push(existing.clone());
                }
            }
            // Otherwise it was attempted and failed -> omit it entirely.
            continue;
        }
        let hash = match delta::file_hash(path) {
            Ok(hash) => hash,
            Err(e) => {
                eprintln!("  Hashing failed for {}: {e}", rel);
                continue;
            }
        };
        let modified = std::fs::metadata(path)
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        new_states.push(types::FileState {
            path: rel,
            hash,
            last_modified: modified,
        });
    }
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