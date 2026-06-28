use std::{fmt, io, path::{Path, PathBuf}};

use rnix::ast::{self, HasEntry};
use rowan::ast::AstNode;

// ── Parse diagnostic ─────────────────────────────────────────────────────────
// Carried verbatim to the execution/self-healing loop so the LLM sees exact
// byte offsets and message text.

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ParseDiagnostic {
    pub message: String,
    pub byte_start: u32,
    pub byte_end: u32,
}

// ── Error ─────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum NixError {
    Io { path: PathBuf, source: io::Error },
    /// AST is malformed — self-healing loop should trigger re-generation.
    Parse { path: PathBuf, diagnostics: Vec<ParseDiagnostic> },
    /// The requested dotted path does not exist in the file.
    AttrNotFound { attr_path: String },
    /// The node exists but has the wrong value type for the requested operation.
    TypeError { attr_path: String, expected: &'static str },
    /// Package name not present in the target list.
    PackageNotFound { attr_path: String, package: String },
}

impl fmt::Display for NixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(f, "I/O error on {}: {}", path.display(), source)
            }
            Self::Parse { path, diagnostics } => {
                write!(f, "parse errors in {}:", path.display())?;
                for d in diagnostics {
                    write!(f, "\n  {}..{} — {}", d.byte_start, d.byte_end, d.message)?;
                }
                Ok(())
            }
            Self::AttrNotFound { attr_path } => {
                write!(f, "attribute '{}' not found", attr_path)
            }
            Self::TypeError { attr_path, expected } => {
                write!(f, "'{}' is not a {}", attr_path, expected)
            }
            Self::PackageNotFound { attr_path, package } => {
                write!(f, "'{}' not found in '{}'", package, attr_path)
            }
        }
    }
}

impl std::error::Error for NixError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let Self::Io { source, .. } = self {
            Some(source)
        } else {
            None
        }
    }
}

// ── Value representation ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub enum NixValue {
    Bool(bool),
    Int(i64),
    Str(String),
    /// Items as raw source text (may be `vim`, `pkgs.git`, `(callPackage …)`, …).
    PackageList(Vec<String>),
    /// Value node we can locate but not interpret structurally.
    Opaque(String),
}

// ── NixFile ───────────────────────────────────────────────────────────────────

pub struct NixFile {
    pub path: PathBuf,
    source: String,
    root: rnix::Root,
}

impl NixFile {
    /// Read and parse a `.nix` file from disk.
    /// Any parse error is a hard failure so the execution module can detect
    /// a broken file before attempting mutations.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, NixError> {
        let path = path.as_ref().to_path_buf();
        let source = std::fs::read_to_string(&path).map_err(|e| NixError::Io {
            path: path.clone(),
            source: e,
        })?;
        Self::from_source(path, source)
    }

    /// Parse from an in-memory string. Used in tests and when the caller
    /// already holds the source (e.g. after applying a previous patch).
    pub fn from_source(path: impl Into<PathBuf>, source: String) -> Result<Self, NixError> {
        let path = path.into();
        let parse = rnix::Root::parse(&source);

        // ParseError embeds range info in its Display output; we also extract
        // byte offsets from known variants via the enum's first TextRange field.
        let diagnostics: Vec<ParseDiagnostic> = parse
            .errors()
            .iter()
            .map(|e| {
                use rnix::parser::ParseError;
                let (start, end) = match e {
                    ParseError::Unexpected(r)
                    | ParseError::UnexpectedExtra(r)
                    | ParseError::UnexpectedDoubleBind(r) => {
                        (u32::from(r.start()), u32::from(r.end()))
                    }
                    ParseError::UnexpectedWanted(_, r, _) => {
                        (u32::from(r.start()), u32::from(r.end()))
                    }
                    ParseError::DuplicatedArgs(r, _) => {
                        (u32::from(r.start()), u32::from(r.end()))
                    }
                    // EOF variants have no meaningful byte position.
                    _ => (0, 0),
                };
                ParseDiagnostic {
                    message: e.to_string(),
                    byte_start: start,
                    byte_end: end,
                }
            })
            .collect();

        if !diagnostics.is_empty() {
            return Err(NixError::Parse { path, diagnostics });
        }

        Ok(Self { path, source, root: parse.tree() })
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// Read the value at `attr_path` (e.g. `&["environment", "systemPackages"]`).
    pub fn get_attr(&self, attr_path: &[&str]) -> Result<NixValue, NixError> {
        let node = self.resolve_value_node(attr_path)?;
        Ok(classify_value(&node))
    }

    // ── Mutations ─────────────────────────────────────────────────────────────
    // All mutation methods return a *new source string*. They never modify
    // `self` — the caller decides whether to write to disk and under which
    // git strategy. This preserves NixOS immutability semantics at the tool level.

    /// Return new source with `package` appended to the list at `attr_path`.
    /// Handles both `[ a b ]` and `with pkgs; [ a b ]` forms.
    pub fn append_package(
        &self,
        attr_path: &[&str],
        package: &str,
    ) -> Result<String, NixError> {
        let list_node = self.resolve_list_node(attr_path)?;
        let range = list_node.text_range();
        let abs_start = usize::from(range.start());
        let abs_end = usize::from(range.end());
        let list_src = &self.source[abs_start..abs_end];

        let close = list_src
            .rfind(']')
            .expect("rnix List node always contains ]");

        let is_multiline = list_src[..close].contains('\n');

        let new_list = if is_multiline {
            let indent = detect_list_item_indent(list_src).unwrap_or("  ");
            // Strip trailing whitespace before `]` so we insert cleanly.
            let before_close = list_src[..close].trim_end_matches([' ', '\t']);
            format!(
                "{}\n{}{}\n{}]",
                before_close,
                indent,
                package,
                // Preserve whatever whitespace was after the last item before `]`.
                &list_src[close + 1 - (list_src[..close].len() - before_close.len())..close],
            )
        } else {
            // Single-line: `[ a b ]` → `[ a b c ]`
            let before_close = list_src[..close].trim_end_matches(' ');
            format!("{} {}]", before_close, package)
        };

        Ok(splice(&self.source, abs_start, abs_end, &new_list))
    }

    /// Return new source with `package` removed from the list at `attr_path`.
    pub fn remove_package(
        &self,
        attr_path: &[&str],
        package: &str,
    ) -> Result<String, NixError> {
        let list_node = self.resolve_list_node(attr_path)?;
        let list = ast::List::cast(list_node.clone())
            .expect("resolve_list_node guarantees a List node");

        let item = list
            .items()
            .find(|expr| expr.syntax().text().to_string().trim() == package)
            .ok_or_else(|| NixError::PackageNotFound {
                attr_path: attr_path.join("."),
                package: package.to_owned(),
            })?;

        let item_range = item.syntax().text_range();
        let item_start = usize::from(item_range.start());
        let item_end = usize::from(item_range.end());

        // Absorb the leading whitespace/newline so we don't leave a blank line.
        let list_start = usize::from(list_node.text_range().start());
        let prefix = &self.source[list_start..item_start];
        let leading_ws = prefix
            .chars()
            .rev()
            .take_while(|c| *c == ' ' || *c == '\t')
            .count();
        let has_leading_newline = prefix[..prefix.len() - leading_ws].ends_with('\n');
        let remove_from = item_start - leading_ws - usize::from(has_leading_newline);

        Ok(splice(&self.source, remove_from, item_end, ""))
    }

    /// Return new source with the boolean at `attr_path` set to `value`.
    pub fn set_bool(
        &self,
        attr_path: &[&str],
        value: bool,
    ) -> Result<String, NixError> {
        let node = self.resolve_value_node(attr_path)?;
        if !matches!(classify_value(&node), NixValue::Bool(_)) {
            return Err(NixError::TypeError {
                attr_path: attr_path.join("."),
                expected: "boolean",
            });
        }
        let range = node.text_range();
        Ok(splice(
            &self.source,
            usize::from(range.start()),
            usize::from(range.end()),
            if value { "true" } else { "false" },
        ))
    }

    /// Return new source with the string at `attr_path` set to `value`
    /// (properly escaped and quoted).
    pub fn set_string(
        &self,
        attr_path: &[&str],
        value: &str,
    ) -> Result<String, NixError> {
        let node = self.resolve_value_node(attr_path)?;
        if !matches!(classify_value(&node), NixValue::Str(_)) {
            return Err(NixError::TypeError {
                attr_path: attr_path.join("."),
                expected: "string",
            });
        }
        let range = node.text_range();
        let quoted = nix_escape_string(value);
        Ok(splice(
            &self.source,
            usize::from(range.start()),
            usize::from(range.end()),
            &quoted,
        ))
    }

    // ── Internal resolution ───────────────────────────────────────────────────

    fn resolve_value_node(&self, attr_path: &[&str]) -> Result<rowan::SyntaxNode<rnix::NixLanguage>, NixError> {
        let top = top_level_attrset(&self.root).ok_or_else(|| NixError::AttrNotFound {
            attr_path: attr_path.join("."),
        })?;
        find_value_in_attrset(top.syntax(), attr_path).ok_or_else(|| NixError::AttrNotFound {
            attr_path: attr_path.join("."),
        })
    }

    fn resolve_list_node(&self, attr_path: &[&str]) -> Result<rowan::SyntaxNode<rnix::NixLanguage>, NixError> {
        let value = self.resolve_value_node(attr_path)?;
        unwrap_to_list(&value).ok_or_else(|| NixError::TypeError {
            attr_path: attr_path.join("."),
            expected: "list",
        })
    }
}

// ── AST Navigation ────────────────────────────────────────────────────────────

type SyntaxNode = rowan::SyntaxNode<rnix::NixLanguage>;

/// Peel any number of lambda arguments (`{ config, pkgs, ... }:`) to reach
/// the top-level attrset body of a NixOS configuration.
fn top_level_attrset(root: &rnix::Root) -> Option<ast::AttrSet> {
    let mut expr = root.expr()?;
    loop {
        match expr {
            ast::Expr::Lambda(lambda) => {
                expr = lambda.body()?;
            }
            ast::Expr::AttrSet(attrset) => return Some(attrset),
            ast::Expr::Paren(p) => {
                expr = p.expr()?;
            }
            _ => return None,
        }
    }
}

/// Recursively find the value node for `path` inside `attrset_node`.
///
/// Handles both dotted flat keys (`a.b.c = value;`) and nested attrsets
/// (`a = { b = { c = value; }; };`), including mixed forms.
fn find_value_in_attrset(attrset_node: &SyntaxNode, path: &[&str]) -> Option<SyntaxNode> {
    if path.is_empty() {
        return Some(attrset_node.clone());
    }

    let attrset = ast::AttrSet::cast(attrset_node.clone())?;

    for entry in attrset.entries() {
        let ast::Entry::AttrpathValue(kv) = entry else {
            continue;
        };
        let key_parts: Vec<String> = kv
            .attrpath()?
            .attrs()
            .filter_map(attr_ident_text)
            .collect();

        if key_parts.is_empty() {
            continue;
        }

        // Count how many path components this flat key satisfies.
        let consumed = key_parts
            .iter()
            .zip(path.iter())
            .take_while(|(k, p)| k.as_str() == **p)
            .count();

        if consumed == 0 || consumed != key_parts.len() {
            // Partial prefix or no match — skip.
            continue;
        }

        let remaining = &path[consumed..];
        let value_node = kv.value()?.syntax().clone();

        if remaining.is_empty() {
            return Some(value_node);
        }

        // More path components remain → the value must be a nested attrset.
        if matches!(kv.value()?, ast::Expr::AttrSet(_)) {
            return find_value_in_attrset(&value_node, remaining);
        }
    }

    None
}

/// Extract the string name of an `Attr` node, if it is a static identifier or
/// a plain quoted string key. Dynamic keys (`${expr}`) are not navigable.
fn attr_ident_text(attr: ast::Attr) -> Option<String> {
    match attr {
        ast::Attr::Ident(id) => Some(id.ident_token()?.text().to_owned()),
        ast::Attr::Str(s) => {
            let raw = s.syntax().text().to_string();
            // Strip surrounding double-quotes for plain string keys like `"foo"`.
            Some(raw.trim_matches('"').to_owned())
        }
        ast::Attr::Dynamic(_) => None,
    }
}

/// Classify a value `SyntaxNode` into a `NixValue`. Unrecognised nodes become
/// `NixValue::Opaque` — they are valid Nix, just not structurally interesting
/// for the current operation set.
fn classify_value(node: &SyntaxNode) -> NixValue {
    match ast::Expr::cast(node.clone()) {
        Some(ast::Expr::Ident(id)) => {
            match id.ident_token().as_ref().map(|t| t.text()) {
                Some("true") => NixValue::Bool(true),
                Some("false") => NixValue::Bool(false),
                _ => NixValue::Opaque(node.text().to_string()),
            }
        }
        Some(ast::Expr::Str(s)) => {
            // Collect the inner text of a double-quoted string, stripping quotes.
            let raw = s.syntax().text().to_string();
            let inner = raw
                .strip_prefix('"')
                .and_then(|t| t.strip_suffix('"'))
                .unwrap_or(&raw)
                .to_owned();
            NixValue::Str(inner)
        }
        Some(ast::Expr::Literal(lit)) => {
            if let Some(tok) = lit.syntax().first_token() {
                if let Ok(n) = tok.text().parse::<i64>() {
                    return NixValue::Int(n);
                }
            }
            NixValue::Opaque(node.text().to_string())
        }
        Some(ast::Expr::List(list)) => {
            NixValue::PackageList(list_items_as_strings(&list))
        }
        Some(ast::Expr::With(with)) => {
            if let Some(ast::Expr::List(list)) = with.body() {
                return NixValue::PackageList(list_items_as_strings(&list));
            }
            NixValue::Opaque(node.text().to_string())
        }
        _ => NixValue::Opaque(node.text().to_string()),
    }
}

fn list_items_as_strings(list: &ast::List) -> Vec<String> {
    list.items()
        .map(|item| item.syntax().text().to_string().trim().to_owned())
        .collect()
}

/// Drill through `with pkgs; [...]` or plain `[...]` to the underlying `List`
/// `SyntaxNode`. Returns `None` if the value is neither.
fn unwrap_to_list(node: &SyntaxNode) -> Option<SyntaxNode> {
    match ast::Expr::cast(node.clone())? {
        ast::Expr::List(_) => Some(node.clone()),
        ast::Expr::With(with) => {
            if matches!(with.body()?, ast::Expr::List(_)) {
                Some(with.body()?.syntax().clone())
            } else {
                None
            }
        }
        _ => None,
    }
}

// ── Text utilities ────────────────────────────────────────────────────────────

/// Replace `source[start..end]` with `replacement`.
fn splice(source: &str, start: usize, end: usize, replacement: &str) -> String {
    let mut out = String::with_capacity(source.len() - (end - start) + replacement.len());
    out.push_str(&source[..start]);
    out.push_str(replacement);
    out.push_str(&source[end..]);
    out
}

/// Detect the whitespace indent of the first item in a list source fragment.
/// Input is the raw text of a `List` node, e.g. `"[\n    vim\n    git\n  ]"`.
fn detect_list_item_indent(list_src: &str) -> Option<&str> {
    let after_open = list_src.strip_prefix('[')?;
    let nl_pos = after_open.find('\n')?;
    let rest = &after_open[nl_pos + 1..];
    let indent_len = rest
        .chars()
        .take_while(|c| *c == ' ' || *c == '\t')
        .count();
    Some(&rest[..indent_len])
}

/// Produce a valid Nix double-quoted string literal for `s`.
fn nix_escape_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '$' => out.push_str("\\$"),  // prevent Nix string interpolation
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{ config, pkgs, ... }:
{
  environment.systemPackages = with pkgs; [
    vim
    git
  ];

  services.openssh.enable = true;
  networking.hostName = "myhost";
}
"#;

    fn nf(src: &str) -> NixFile {
        NixFile::from_source("test.nix", src.to_owned()).expect("valid Nix")
    }

    #[test]
    fn rejects_broken_ast() {
        let err = NixFile::from_source("bad.nix", "{ foo = ;".to_owned());
        assert!(matches!(err, Err(NixError::Parse { .. })));
    }

    #[test]
    fn reads_package_list() {
        let f = nf(SAMPLE);
        let val = f.get_attr(&["environment", "systemPackages"]).unwrap();
        assert_eq!(val, NixValue::PackageList(vec!["vim".into(), "git".into()]));
    }

    #[test]
    fn reads_bool() {
        let f = nf(SAMPLE);
        assert_eq!(
            f.get_attr(&["services", "openssh", "enable"]).unwrap(),
            NixValue::Bool(true),
        );
    }

    #[test]
    fn reads_string() {
        let f = nf(SAMPLE);
        assert_eq!(
            f.get_attr(&["networking", "hostName"]).unwrap(),
            NixValue::Str("myhost".into()),
        );
    }

    #[test]
    fn append_package_round_trip() {
        let f = nf(SAMPLE);
        let new_src = f
            .append_package(&["environment", "systemPackages"], "firefox")
            .unwrap();
        let f2 = NixFile::from_source("test.nix", new_src).expect("still valid");
        let pkgs = f2.get_attr(&["environment", "systemPackages"]).unwrap();
        assert_eq!(
            pkgs,
            NixValue::PackageList(vec!["vim".into(), "git".into(), "firefox".into()])
        );
    }

    #[test]
    fn remove_package_round_trip() {
        let f = nf(SAMPLE);
        let new_src = f
            .remove_package(&["environment", "systemPackages"], "git")
            .unwrap();
        let f2 = NixFile::from_source("test.nix", new_src).expect("still valid");
        let pkgs = f2.get_attr(&["environment", "systemPackages"]).unwrap();
        assert_eq!(pkgs, NixValue::PackageList(vec!["vim".into()]));
    }

    #[test]
    fn set_bool_round_trip() {
        let f = nf(SAMPLE);
        let new_src = f
            .set_bool(&["services", "openssh", "enable"], false)
            .unwrap();
        assert!(new_src.contains("enable = false"));
        let f2 = NixFile::from_source("test.nix", new_src).unwrap();
        assert_eq!(
            f2.get_attr(&["services", "openssh", "enable"]).unwrap(),
            NixValue::Bool(false),
        );
    }

    #[test]
    fn set_string_round_trip() {
        let f = nf(SAMPLE);
        let new_src = f
            .set_string(&["networking", "hostName"], r#"need"escape"#)
            .unwrap();
        assert!(new_src.contains(r#""need\"escape""#));
    }

    #[test]
    fn missing_attr_returns_error() {
        let f = nf(SAMPLE);
        let err = f.get_attr(&["does", "not", "exist"]);
        assert!(matches!(err, Err(NixError::AttrNotFound { .. })));
    }

    #[test]
    fn type_error_on_wrong_mutation() {
        let f = nf(SAMPLE);
        // `hostName` is a string, not a bool
        let err = f.set_bool(&["networking", "hostName"], true);
        assert!(matches!(err, Err(NixError::TypeError { .. })));
    }
}
