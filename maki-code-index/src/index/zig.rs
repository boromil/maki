use tree_sitter::Node;

use super::common::{
    ChildKind, LanguageExtractor, Section, SkeletonEntry, compact_ws, extract_fields_truncated,
    find_child, node_text,
};

pub(crate) struct ZigExtractor;

fn is_type_value_kind(kind: &str) -> bool {
    matches!(
        kind,
        "struct_declaration" | "enum_declaration" | "union_declaration" | "opaque_declaration"
    )
}

impl ZigExtractor {
    fn import_path(bf_node: Node, source: &[u8]) -> Vec<String> {
        let text = find_child(bf_node, "arguments")
            .and_then(|args| find_child(args, "string"))
            .map(|s| node_text(s, source))
            .unwrap_or_default();
        let inner = text.trim_matches('"');
        let clean = inner.strip_suffix(".zig").unwrap_or(inner);
        clean.split('/').map(String::from).collect()
    }

    fn extract_var_decl(&self, node: Node, source: &[u8]) -> Option<SkeletonEntry> {
        let mut has_pub = false;
        let mut is_const = false;
        let mut name: Option<&str> = None;
        let mut import_node: Option<Node> = None;
        let mut is_type = false;

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "pub" => has_pub = true,
                "const" => is_const = true,
                "var" | "extern" => {}
                "identifier" if name.is_none() => name = Some(node_text(child, source)),
                "builtin_function" => {
                    if find_child(child, "builtin_identifier")
                        .map(|bi| node_text(bi, source) == "@import")
                        .unwrap_or(false)
                    {
                        import_node = Some(child);
                    }
                }
                k if is_type_value_kind(k) => is_type = true,
                _ => {}
            }
        }

        let name = name?;

        if let Some(bf) = import_node {
            return Some(SkeletonEntry::new_import(
                node,
                vec![Self::import_path(bf, source)],
            ));
        }

        let vis = if has_pub { "pub " } else { "" };
        if is_type {
            let container = Self::find_container(node);
            let type_kw = container
                .as_ref()
                .map(|(_, k)| k.strip_suffix("_declaration").unwrap_or(*k))
                .unwrap_or("const");
            let mut entry =
                SkeletonEntry::new(Section::Type, node, format!("{vis}{type_kw} {name}"));
            if let Some((container_node, container_kind)) = container {
                let is_enum = container_kind == "enum_declaration";
                let fields = extract_fields_truncated(
                    container_node,
                    source,
                    "container_field",
                    |field, src| {
                        let fname = field
                            .child_by_field_name("name")
                            .map(|n| node_text(n, src))
                            .unwrap_or("_");
                        if is_enum {
                            fname.to_string()
                        } else {
                            let ftype = field
                                .child_by_field_name("type")
                                .map(|n| format!(": {}", node_text(n, src)))
                                .unwrap_or_default();
                            format!("{fname}{ftype}")
                        }
                    },
                );
                if is_enum {
                    entry = entry.with_child_kind(ChildKind::Brief);
                }
                entry = entry.with_children(fields);
            }
            return Some(entry);
        }

        let kw = if is_const { "const" } else { "var" };
        let ty = node
            .child_by_field_name("type")
            .map(|n| format!(": {}", node_text(n, source)))
            .unwrap_or_default();
        Some(SkeletonEntry::new(
            Section::Constant,
            node,
            format!("{vis}{kw} {name}{ty}"),
        ))
    }

    fn extract_fn(&self, node: Node, source: &[u8]) -> Option<SkeletonEntry> {
        let name = node
            .child_by_field_name("name")
            .map(|n| node_text(n, source))?;
        let params = find_child(node, "parameters")
            .map(|n| node_text(n, source))
            .unwrap_or("()");
        let ret = node
            .child_by_field_name("type")
            .map(|n| format!(" {}", node_text(n, source)))
            .unwrap_or_default();

        let mut vis_parts: Vec<&str> = Vec::new();
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            match child.kind() {
                "pub" => vis_parts.push("pub"),
                "inline" => vis_parts.push("inline"),
                "noinline" => vis_parts.push("noinline"),
                "extern" | "export" => vis_parts.push(child.kind()),
                _ => {}
            }
        }

        let vis = if vis_parts.is_empty() {
            String::new()
        } else {
            format!("{} ", vis_parts.join(" "))
        };
        Some(SkeletonEntry::new(
            Section::Function,
            node,
            compact_ws(&format!("{vis}fn {name}{params}{ret}")).into_owned(),
        ))
    }

    fn find_container(node: Node) -> Option<(Node, &'static str)> {
        static KINDS: &[&str] = &[
            "struct_declaration",
            "enum_declaration",
            "union_declaration",
            "opaque_declaration",
        ];
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if KINDS.contains(&child.kind()) {
                return Some((child, child.kind()));
            }
        }
        None
    }
}

impl LanguageExtractor for ZigExtractor {
    fn extract_nodes(&self, node: Node, source: &[u8], _attrs: &[Node]) -> Vec<SkeletonEntry> {
        let entry = match node.kind() {
            "function_declaration" => self.extract_fn(node, source),
            "variable_declaration" => self.extract_var_decl(node, source),
            _ => None,
        };
        entry.into_iter().collect()
    }

    fn is_doc_comment(&self, node: Node, source: &[u8]) -> bool {
        if node.kind() != "comment" {
            return false;
        }
        let t = node_text(node, source);
        t.starts_with("///") && !t.starts_with("////")
    }

    fn is_module_doc(&self, node: Node, source: &[u8]) -> bool {
        node.kind() == "comment" && node_text(node, source).starts_with("//!")
    }

    fn is_test_node(&self, node: Node, _source: &[u8], _attrs: &[Node]) -> bool {
        node.kind() == "test_declaration"
    }

    fn import_separator(&self) -> &'static str {
        "/"
    }
}
