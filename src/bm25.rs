use crate::types::Chunk;
use anyhow::Result;
use std::path::Path;
use tantivy::collector::TopDocs;
use tantivy::directory::MmapDirectory;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Schema, STORED, STRING, TEXT, Value};
use tantivy::{doc, Index, IndexWriter, ReloadPolicy, TantivyDocument, Term};

const WRITER_HEAP_BYTES: usize = 50_000_000;

pub struct Bm25Index {
    index: Index,
    writer: Option<IndexWriter>,
    file_path_field: Field,
    start_line_field: Field,
    end_line_field: Field,
    text_field: Field,
}

fn build_schema() -> Schema {
    let mut builder = Schema::builder();
    builder.add_text_field("file_path", STRING | STORED);
    builder.add_i64_field("start_line", STORED);
    builder.add_i64_field("end_line", STORED);
    builder.add_text_field("text", TEXT | STORED);
    builder.build()
}

fn field(schema: &Schema, name: &str) -> Result<Field> {
    schema.get_field(name).map_err(|e| anyhow::anyhow!("{e}"))
}

impl Bm25Index {
    pub fn open_or_create(index_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(index_dir)?;
        let index = Index::open_or_create(MmapDirectory::open(index_dir)?, build_schema())?;
        let schema = index.schema();
        Ok(Self {
            file_path_field: field(&schema, "file_path")?,
            start_line_field: field(&schema, "start_line")?,
            end_line_field: field(&schema, "end_line")?,
            text_field: field(&schema, "text")?,
            writer: None,
            index,
        })
    }

    fn writer(&mut self) -> Result<&mut IndexWriter> {
        if self.writer.is_none() {
            self.writer = Some(self.index.writer(WRITER_HEAP_BYTES)?);
        }
        Ok(self.writer.as_mut().expect("writer initialized above"))
    }

    pub fn add_chunks(&mut self, chunks: &[Chunk]) -> Result<()> {
        let (file_path, start_line, end_line, text) =
            (self.file_path_field, self.start_line_field, self.end_line_field, self.text_field);
        let writer = self.writer()?;
        for chunk in chunks {
            writer.add_document(doc!(
                file_path => chunk.file_path.as_str(),
                start_line => chunk.start_line as i64,
                end_line => chunk.end_line as i64,
                text => chunk.text.as_str(),
            ))?;
        }
        Ok(())
    }

    pub fn delete_file(&mut self, file_path: &str) -> Result<()> {
        let term = Term::from_field_text(self.file_path_field, file_path);
        self.writer()?.delete_term(term);
        Ok(())
    }

    pub fn commit(&mut self) -> Result<()> {
        if let Some(writer) = self.writer.as_mut() {
            writer.commit()?;
        }
        Ok(())
    }

    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<Chunk>> {
        let reader = self.index.reader_builder().reload_policy(ReloadPolicy::Manual).try_into()?;
        let searcher = reader.searcher();
        let (parsed, _) = QueryParser::for_index(&self.index, vec![self.text_field]).parse_query_lenient(query);
        let top_docs = searcher.search(&parsed, &TopDocs::with_limit(limit))?;
        top_docs
            .into_iter()
            .map(|(_, address)| Ok(self.to_chunk(&searcher.doc(address)?)))
            .collect()
    }

    fn to_chunk(&self, doc: &TantivyDocument) -> Chunk {
        let text = |field| doc.get_first(field).and_then(|v| v.as_str()).unwrap_or_default().to_string();
        let line = |field| doc.get_first(field).and_then(|v| v.as_i64()).unwrap_or_default() as usize;
        Chunk {
            file_path: text(self.file_path_field),
            start_line: line(self.start_line_field),
            end_line: line(self.end_line_field),
            text: text(self.text_field),
        }
    }
}
