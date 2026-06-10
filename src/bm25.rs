use crate::types::Chunk;
use anyhow::Result;
use std::path::Path;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Schema, STRING, STORED, TEXT, Value};
use tantivy::{doc, Index, ReloadPolicy, TantivyDocument, Term};

pub struct Bm25Index {
    index: Index,
    file_path_field: Field,
    start_line_field: Field,
    end_line_field: Field,
    text_field: Field,
}

fn build_schema() -> (Schema, Field, Field, Field, Field) {
    let mut builder = Schema::builder();
    let file_path = builder.add_text_field("file_path", STRING | STORED);
    let start_line = builder.add_i64_field("start_line", STORED);
    let end_line = builder.add_i64_field("end_line", STORED);
    let text = builder.add_text_field("text", TEXT | STORED);
    (builder.build(), file_path, start_line, end_line, text)
}

impl Bm25Index {
    pub fn new(index_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(index_dir)?;
        let (schema, file_path_field, start_line_field, end_line_field, text_field) = build_schema();
        let index = Index::create_in_dir(index_dir, schema)?;
        Ok(Self {
            index,
            file_path_field,
            start_line_field,
            end_line_field,
            text_field,
        })
    }

    pub fn open(index_dir: &Path) -> Result<Self> {
        let index = Index::open_in_dir(index_dir)?;
        let schema = index.schema();
        let file_path_field = schema.get_field("file_path").map_err(|e| anyhow::anyhow!("{e}"))?;
        let start_line_field = schema.get_field("start_line").map_err(|e| anyhow::anyhow!("{e}"))?;
        let end_line_field = schema.get_field("end_line").map_err(|e| anyhow::anyhow!("{e}"))?;
        let text_field = schema.get_field("text").map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(Self {
            index,
            file_path_field,
            start_line_field,
            end_line_field,
            text_field,
        })
    }

    pub fn add_chunks(&mut self, chunks: &[Chunk]) -> Result<()> {
        let mut writer = self.index.writer::<TantivyDocument>(50_000_000)?;
        for chunk in chunks {
            let _ = writer.add_document(doc!(
                self.file_path_field => chunk.file_path.as_str(),
                self.start_line_field => chunk.start_line as i64,
                self.end_line_field => chunk.end_line as i64,
                self.text_field => chunk.text.as_str(),
            ));
        }
        writer.commit()?;
        Ok(())
    }

    pub fn commit(&mut self) -> Result<()> {
        let mut writer = self.index.writer::<TantivyDocument>(50_000_000)?;
        writer.commit()?;
        Ok(())
    }

    pub fn delete_file(&mut self, file_path: &str) -> Result<()> {
        let mut writer = self.index.writer::<TantivyDocument>(50_000_000)?;
        let term = Term::from_field_text(self.file_path_field, file_path);
        writer.delete_term(term);
        writer.commit()?;
        Ok(())
    }

    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<(String, usize, usize, f32)>> {
        let reader = self.index.reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;
        let searcher = reader.searcher();
        let query_parser = QueryParser::for_index(&self.index, vec![self.text_field]);
        let parsed = query_parser.parse_query(query)?;
        let top_docs = searcher.search(&parsed, &TopDocs::with_limit(limit))?;
        let mut results = Vec::new();
        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address)?;
            let file_path = doc.get_first(self.file_path_field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let start_line = doc.get_first(self.start_line_field)
                .and_then(|v| v.as_i64())
                .unwrap_or(0) as usize;
            let end_line = doc.get_first(self.end_line_field)
                .and_then(|v| v.as_i64())
                .unwrap_or(0) as usize;
            results.push((file_path, start_line, end_line, score as f32));
        }
        Ok(results)
    }
}