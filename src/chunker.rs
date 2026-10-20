use crate::types::Chunk;
use anyhow::Result;
use std::path::Path;
use tree_sitter::{Language, Node, Parser, Range};

const MAX_CHUNK_CHARS: usize = 3000;
const MAX_SPLIT_DEPTH: usize = 32;

enum Format {
    Code(&'static str),
    Markdown,
    Config,
}

fn format_for_path(path: &Path) -> Option<Format> {
    let format = match path.extension()?.to_str()? {
        "rs" => Format::Code("rust"),
        "go" => Format::Code("go"),
        "py" => Format::Code("python"),
        "ts" => Format::Code("typescript"),
        "tsx" => Format::Code("tsx"),
        "java" => Format::Code("java"),
        "cpp" | "cxx" | "cc" | "hpp" | "h" | "c" => Format::Code("cpp"),
        "md" => Format::Markdown,
        "toml" | "yaml" | "yml" | "json" => Format::Config,
        _ => return None,
    };
    Some(format)
}

fn grammar(lang: &str) -> Option<Language> {
    let language = match lang {
        "rust" => tree_sitter_rust::LANGUAGE,
        "go" => tree_sitter_go::LANGUAGE,
        "python" => tree_sitter_python::LANGUAGE,
        "typescript" => tree_sitter_typescript::LANGUAGE_TYPESCRIPT,
        "tsx" => tree_sitter_typescript::LANGUAGE_TSX,
        "java" => tree_sitter_java::LANGUAGE,
        "cpp" => tree_sitter_cpp::LANGUAGE,
        _ => return None,
    };
    Some(language.into())
}

fn chunk_kinds(lang: &str) -> &'static [&'static str] {
    match lang {
        "rust" => &["function_item", "impl_item", "struct_item", "enum_item", "trait_item"],
        "go" => &["function_declaration", "method_declaration", "type_declaration"],
        "python" => &["function_definition", "class_definition", "decorated_definition"],
        "typescript" | "tsx" => &[
            "function_declaration",
            "class_declaration",
            "interface_declaration",
            "type_alias_declaration",
            "enum_declaration",
            "lexical_declaration",
        ],
        "java" => &["class_declaration", "interface_declaration", "enum_declaration", "record_declaration"],
        "cpp" => &["function_definition", "class_specifier", "struct_specifier", "namespace_definition", "template_declaration"],
        _ => &[],
    }
}

fn chunk_size(text: &str) -> usize {
    text.chars().filter(|c| !c.is_whitespace()).count()
}

fn split_to_budget(node: Node, source: &str, depth: usize) -> Vec<Range> {
    let fits = chunk_size(&source[node.byte_range()]) <= MAX_CHUNK_CHARS;
    if fits || node.child_count() == 0 || depth >= MAX_SPLIT_DEPTH {
        return vec![node.range()];
    }
    let mut cursor = node.walk();
    let pieces = node
        .children(&mut cursor)
        .flat_map(|child| split_to_budget(child, source, depth + 1))
        .collect();
    merge_adjacent(pieces, source)
}

fn merge_adjacent(pieces: Vec<Range>, source: &str) -> Vec<Range> {
    let mut merged: Vec<Range> = Vec::with_capacity(pieces.len());
    for piece in pieces {
        if let Some(last) = merged.last_mut()
            && chunk_size(&source[last.start_byte..piece.end_byte]) <= MAX_CHUNK_CHARS
        {
            last.end_byte = piece.end_byte;
            last.end_point = piece.end_point;
            continue;
        }
        merged.push(piece);
    }
    merged
}

fn collect_ranges(root: Node, source: &str, kinds: &[&str]) -> Vec<Range> {
    let mut ranges = Vec::new();
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if kinds.contains(&node.kind()) {
            ranges.extend(split_to_budget(node, source, 0));
            continue;
        }
        let mut cursor = node.walk();
        let children: Vec<Node> = node.children(&mut cursor).collect();
        stack.extend(children.into_iter().rev());
    }
    ranges
}

fn range_chunk(range: &Range, source: &str, rel_path: &str) -> Chunk {
    Chunk {
        file_path: rel_path.to_string(),
        start_line: range.start_point.row + 1,
        end_line: range.end_point.row + 1,
        text: source[range.start_byte..range.end_byte].to_string(),
    }
}

fn chunk_code(lang: &str, source: &str, rel_path: &str) -> Result<Vec<Chunk>> {
    let language = grammar(lang).ok_or_else(|| anyhow::anyhow!("no grammar for language: {lang}"))?;
    let mut parser = Parser::new();
    parser.set_language(&language)?;
    let tree = parser
        .parse(source, None)
        .ok_or_else(|| anyhow::anyhow!("failed to parse {rel_path}"))?;

    let root = tree.root_node();
    let mut ranges = collect_ranges(root, source, chunk_kinds(lang));
    if ranges.is_empty() && !source.trim().is_empty() {
        ranges = split_to_budget(root, source, 0);
    }
    Ok(ranges.iter().map(|r| range_chunk(r, source, rel_path)).collect())
}

fn is_markdown_heading(line: &str) -> bool {
    let trimmed = line.trim_start();
    let after_hashes = trimmed.trim_start_matches('#');
    trimmed.starts_with('#') && (after_hashes.is_empty() || after_hashes.starts_with(char::is_whitespace))
}

fn is_code_fence(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("```") || trimmed.starts_with("~~~")
}

fn push_lines(chunks: &mut Vec<Chunk>, lines: &[&str], start_line: usize, rel_path: &str) {
    let text = lines.join("\n");
    if text.trim().is_empty() {
        return;
    }
    chunks.push(Chunk {
        file_path: rel_path.to_string(),
        start_line,
        end_line: start_line + lines.len() - 1,
        text,
    });
}

fn chunk_lines(source: &str, rel_path: &str, mut starts_section: impl FnMut(&str) -> bool) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut current: Vec<&str> = Vec::new();
    let mut current_size = 0;
    let mut start_line = 1;

    for (i, line) in source.lines().enumerate() {
        let line_size = chunk_size(line);
        let overflows = current_size + line_size > MAX_CHUNK_CHARS;
        if starts_section(line) || overflows {
            push_lines(&mut chunks, &current, start_line, rel_path);
            current.clear();
            current_size = 0;
            start_line = i + 1;
        }
        current.push(line);
        current_size += line_size;
    }
    push_lines(&mut chunks, &current, start_line, rel_path);
    chunks
}

fn chunk_markdown(source: &str, rel_path: &str) -> Vec<Chunk> {
    let mut in_fence = false;
    chunk_lines(source, rel_path, |line| {
        if is_code_fence(line) {
            in_fence = !in_fence;
        }
        !in_fence && is_markdown_heading(line)
    })
}

pub fn chunk_file_with_rel(path: &Path, source: &str, rel_path: &str) -> Result<Vec<Chunk>> {
    let format = format_for_path(path)
        .ok_or_else(|| anyhow::anyhow!("unsupported language: {}", path.display()))?;
    match format {
        Format::Code(lang) => chunk_code(lang, source, rel_path),
        Format::Markdown => Ok(chunk_markdown(source, rel_path)),
        Format::Config => Ok(chunk_lines(source, rel_path, |_| false)),
    }
}
