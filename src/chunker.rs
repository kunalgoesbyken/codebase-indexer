use crate::types::Chunk;
use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;
use tree_sitter::{Node, Parser};

type NodeKindMap = HashMap<&'static str, Vec<&'static str>>;

const MAX_CHUNK_TOKENS: usize = 1000;

fn language_node_kinds() -> NodeKindMap {
    let mut m = NodeKindMap::new();
    m.insert("rust", vec!["function_item", "impl_item", "struct_item", "trait_item"]);
    m.insert("go", vec!["function_declaration", "method_declaration", "type_declaration", "struct_type"]);
    m.insert("python", vec!["function_definition", "class_definition", "decorated_definition"]);
    m.insert("typescript", vec!["function_declaration", "class_declaration", "method_definition", "interface_declaration", "type_alias_declaration", "enum_declaration", "arrow_function"]);
    m.insert("tsx", vec!["function_declaration", "class_declaration", "method_definition", "interface_declaration", "type_alias_declaration", "enum_declaration", "arrow_function"]);
    m.insert("java", vec!["method_declaration", "class_declaration", "interface_declaration", "enum_declaration", "constructor_declaration"]);
    m.insert("cpp", vec!["function_definition", "class_specifier", "struct_specifier", "namespace_definition", "template_declaration"]);
    m
}

fn language_splittable_kinds() -> NodeKindMap {
    let mut m = NodeKindMap::new();
    m.insert("rust", vec!["function_item"]);
    m.insert("go", vec!["function_declaration", "method_declaration"]);
    m.insert("python", vec!["function_definition", "class_definition"]);
    m.insert("typescript", vec!["function_declaration", "method_definition"]);
    m.insert("tsx", vec!["function_declaration", "method_definition"]);
    m.insert("java", vec!["method_declaration", "constructor_declaration"]);
    m.insert("cpp", vec!["function_definition"]);
    m
}

fn language_inner_kinds() -> NodeKindMap {
    let mut m = NodeKindMap::new();
    m.insert("rust", vec![
        "expression_statement",
        "let_declaration",
        "return_expression",
        "match_expression",
        "for_expression",
        "while_expression",
        "loop_expression",
        "if_expression",
        "unsafe_block",
        "macro_invocation",
    ]);
    m.insert("go", vec![
        "expression_statement",
        "short_var_declaration",
        "if_statement",
        "for_statement",
        "switch_statement",
        "return_statement",
        "assignment_statement",
        "defer_statement",
        "go_statement",
    ]);
    m.insert("python", vec![
        "expression_statement",
        "simple_statement",
        "if_statement",
        "for_statement",
        "while_statement",
        "try_statement",
        "with_statement",
        "return_statement",
        "assignment",
        "decorated_definition",
    ]);
    m.insert("typescript", vec![
        "expression_statement",
        "variable_declaration",
        "if_statement",
        "for_statement",
        "while_statement",
        "try_statement",
        "return_statement",
        "switch_statement",
        "throw_statement",
        "lexical_declaration",
    ]);
    m.insert("tsx", vec![
        "expression_statement",
        "variable_declaration",
        "if_statement",
        "for_statement",
        "while_statement",
        "try_statement",
        "return_statement",
        "switch_statement",
        "throw_statement",
        "lexical_declaration",
    ]);
    m.insert("java", vec![
        "expression_statement",
        "if_statement",
        "for_statement",
        "while_statement",
        "try_statement",
        "return_statement",
        "switch_statement",
        "throw_statement",
        "local_variable_declaration",
        "assert_statement",
    ]);
    m.insert("cpp", vec![
        "expression_statement",
        "declaration_statement",
        "if_statement",
        "for_statement",
        "while_statement",
        "return_statement",
        "switch_statement",
        "try_statement",
        "throw_statement",
        "preproc_ifdef",
    ]);
    m
}

fn language_for_path(path: &Path) -> Option<&'static str> {
    match path.extension()?.to_str()? {
        "rs" => Some("rust"),
        "go" => Some("go"),
        "py" => Some("python"),
        "ts" => Some("typescript"),
        "tsx" => Some("tsx"),
        "java" => Some("java"),
        "cpp" | "cxx" | "cc" | "hpp" | "h" | "c" => Some("cpp"),
        "md" => Some("markdown"),
        "toml" => Some("config"),
        "yaml" | "yml" => Some("config"),
        "json" => Some("config"),
        _ => None,
    }
}

fn estimate_tokens(text: &str) -> usize {
    text.split_whitespace().count()
}

fn make_chunk(node: Node, source: &str, rel_path: &str) -> Chunk {
    Chunk {
        file_path: rel_path.to_string(),
        start_line: node.start_position().row + 1,
        end_line: node.end_position().row + 1,
        text: source[node.start_byte()..node.end_byte()].to_string(),
    }
}

fn split_node(node: Node, source: &str, inner_kinds: &[&str], rel_path: &str, depth: usize) -> Vec<Chunk> {
    if depth > 5 {
        return vec![make_chunk(node, source, rel_path)];
    }
    let text = &source[node.start_byte()..node.end_byte()];
    if estimate_tokens(text) <= MAX_CHUNK_TOKENS || inner_kinds.is_empty() {
        return vec![make_chunk(node, source, rel_path)];
    }
    let mut result = Vec::new();
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if inner_kinds.contains(&child.kind()) {
            result.extend(split_node(child, source, inner_kinds, rel_path, depth + 1));
        }
    }
    if result.is_empty() {
        result.push(make_chunk(node, source, rel_path));
    }
    result
}

fn collect_chunks(
    node: Node,
    source: &str,
    kinds: &[&str],
    splittable_kinds: &[&str],
    inner_kinds: &[&str],
    rel_path: &str,
) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    if kinds.contains(&node.kind()) {
        let text = &source[node.start_byte()..node.end_byte()];
        let is_splittable = splittable_kinds.contains(&node.kind());
        if is_splittable && estimate_tokens(text) > MAX_CHUNK_TOKENS && !inner_kinds.is_empty() {
            chunks.extend(split_node(node, source, inner_kinds, rel_path, 0));
        } else {
            chunks.push(make_chunk(node, source, rel_path));
        }
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        chunks.extend(collect_chunks(child, source, kinds, splittable_kinds, inner_kinds, rel_path));
    }
    chunks
}

fn chunk_markdown(source: &str, rel_path: &str) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut current_lines: Vec<String> = Vec::new();
    let mut current_start = 1;

    for (i, line) in source.lines().enumerate() {
        let line_num = i + 1;
        if line.starts_with('#') && !current_lines.is_empty() {
            let text = current_lines.join("\n");
            if !text.trim().is_empty() {
                chunks.push(Chunk {
                    file_path: rel_path.to_string(),
                    start_line: current_start,
                    end_line: line_num - 1,
                    text,
                });
            }
            current_start = line_num;
            current_lines.clear();
        }
        current_lines.push(line.to_string());
    }

    if !current_lines.is_empty() {
        let text = current_lines.join("\n");
        if !text.trim().is_empty() {
            chunks.push(Chunk {
                file_path: rel_path.to_string(),
                start_line: current_start,
                end_line: source.lines().count(),
                text,
            });
        }
    }

    if chunks.is_empty() && !source.trim().is_empty() {
        chunks.push(Chunk {
            file_path: rel_path.to_string(),
            start_line: 1,
            end_line: source.lines().count(),
            text: source.to_string(),
        });
    }

    chunks
}

fn whole_file_chunk(source: &str, rel_path: &str) -> Vec<Chunk> {
    if source.trim().is_empty() {
        return Vec::new();
    }
    vec![Chunk {
        file_path: rel_path.to_string(),
        start_line: 1,
        end_line: source.lines().count(),
        text: source.to_string(),
    }]
}

pub fn chunk_file_with_rel(path: &Path, source: &str, rel_path: &str) -> Result<Vec<Chunk>> {
    let lang = language_for_path(path)
        .ok_or_else(|| anyhow::anyhow!("unsupported language: {}", path.display()))?;

    if lang == "markdown" {
        return Ok(chunk_markdown(source, rel_path));
    }

    if lang == "config" {
        return Ok(whole_file_chunk(source, rel_path));
    }

    let kinds_map = language_node_kinds();
    let kinds = kinds_map.get(lang).ok_or_else(|| {
        anyhow::anyhow!("no node kinds defined for language: {lang}")
    })?;

    let splittable_map = language_splittable_kinds();
    let splittable_kinds = splittable_map.get(lang).map(|v| &v[..]).unwrap_or(&[]);

    let inner_map = language_inner_kinds();
    let inner_kinds = inner_map.get(lang).map(|v| &v[..]).unwrap_or(&[]);

    let mut parser = Parser::new();

    let tree_lang: tree_sitter::Language = match lang {
        "rust" => tree_sitter_rust::LANGUAGE.into(),
        "go" => tree_sitter_go::LANGUAGE.into(),
        "python" => tree_sitter_python::LANGUAGE.into(),
        "typescript" => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        "tsx" => tree_sitter_typescript::LANGUAGE_TSX.into(),
        "java" => tree_sitter_java::LANGUAGE.into(),
        "cpp" => tree_sitter_cpp::LANGUAGE.into(),
        _ => unreachable!(),
    };
    parser.set_language(&tree_lang)?;

    let tree = parser
        .parse(source, None)
        .ok_or_else(|| anyhow::anyhow!("failed to parse {}", path.display()))?;

    let root = tree.root_node();
    let mut chunks = collect_chunks(root, source, kinds, splittable_kinds, inner_kinds, rel_path);

    for chunk in &mut chunks {
        if chunk.file_path.is_empty() {
            chunk.file_path = rel_path.to_string();
        }
    }

    if chunks.is_empty() {
        chunks.push(Chunk {
            file_path: rel_path.to_string(),
            start_line: 1,
            end_line: source.lines().count(),
            text: source.to_string(),
        });
    }

    Ok(chunks)
}