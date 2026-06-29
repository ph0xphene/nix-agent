//! Discovery, listing, and downloading of local GGUF inference models.
//!
//! Resolution is a cascade, highest priority first:
//!   1. `NIX_AGENT_MODEL` — an explicit path to a `.gguf` file.
//!   2. XDG data dir — `~/.local/share/nix-agent/models/*.gguf`.
//!   3. Ollama — valid GGUF blobs under `~/.ollama/models/blobs/`, surfaced so
//!      weights are reused instead of duplicated on disk.
//!
//! This module is pure file/HTTP plumbing — it never loads a model into the
//! inference engine, so it compiles regardless of the `embedded-llm` feature.

use std::collections::HashSet;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use console::Style;
use dialoguer::theme::ColorfulTheme;
use dialoguer::Select;
use directories::{BaseDirs, ProjectDirs};
use indicatif::{ProgressBar, ProgressStyle};

/// Environment override naming an explicit model file.
pub const ENV_MODEL: &str = "NIX_AGENT_MODEL";

/// First four bytes of every GGUF file.
const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// Default target for `models pull` — the 7B Qwen2.5-Coder, Q4_K_M.
pub const DEFAULT_PULL_ALIAS: &str = "qwen2.5-coder-7b";

/// Where a discovered model came from. Drives the listing label and dedup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSource {
    /// Pointed at directly by `NIX_AGENT_MODEL`.
    Env,
    /// Found in the managed XDG models directory.
    XdgCache,
    /// Reused from an existing Ollama blob store.
    Ollama,
}

impl ModelSource {
    pub fn label(self) -> &'static str {
        match self {
            Self::Env => "env",
            Self::XdgCache => "local",
            Self::Ollama => "ollama",
        }
    }
}

/// A GGUF model available on this machine.
#[derive(Debug, Clone)]
pub struct LocalModel {
    /// Display name (file stem, or Ollama blob short hash).
    pub name: String,
    /// Absolute path to the `.gguf` file (or Ollama blob).
    pub path: PathBuf,
    /// On-disk size in bytes.
    pub size_bytes: u64,
    /// Where it was discovered.
    pub source: ModelSource,
}

impl LocalModel {
    fn from_path(path: &Path, source: ModelSource) -> Result<Self> {
        let meta = fs::metadata(path)
            .with_context(|| format!("could not stat model at {}", path.display()))?;
        let name = match source {
            // Ollama blobs are content-addressed (`sha256-<hex>`); show a short hash.
            ModelSource::Ollama => short_blob_name(path),
            _ => path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.to_string_lossy().into_owned()),
        };
        Ok(Self {
            name,
            path: path.to_path_buf(),
            size_bytes: meta.len(),
            source,
        })
    }

    /// Human-readable size, e.g. `4.4 GiB`.
    pub fn human_size(&self) -> String {
        human_bytes(self.size_bytes)
    }
}

/// The managed models directory, `~/.local/share/nix-agent/models/`.
pub fn models_dir() -> Result<PathBuf> {
    let dirs = ProjectDirs::from("", "", "nix-agent")
        .ok_or_else(|| anyhow!("could not determine an XDG data directory for nix-agent"))?;
    Ok(dirs.data_dir().join("models"))
}

/// `~/.ollama/models/blobs/`, if a home directory is resolvable.
fn ollama_blobs_dir() -> Option<PathBuf> {
    let home = BaseDirs::new().map(|b| b.home_dir().to_path_buf())?;
    Some(home.join(".ollama").join("models").join("blobs"))
}

/// Run the full discovery cascade, de-duplicating by canonical path so an Ollama
/// blob symlinked into the XDG dir is not listed twice.
pub fn discover() -> Result<Vec<LocalModel>> {
    let mut out = Vec::new();
    let mut seen: HashSet<PathBuf> = HashSet::new();

    // 1. Explicit environment override.
    if let Some(raw) = std::env::var_os(ENV_MODEL) {
        let p = PathBuf::from(raw);
        if p.is_file() {
            push_unique(&mut out, &mut seen, &p, ModelSource::Env)?;
        }
    }

    // 2. Managed XDG models directory.
    if let Ok(dir) = models_dir() {
        for path in gguf_files_in(&dir) {
            push_unique(&mut out, &mut seen, &path, ModelSource::XdgCache)?;
        }
    }

    // 3. Ollama blob store — only files that really are GGUF.
    if let Some(dir) = ollama_blobs_dir() {
        for path in gguf_blobs_in(&dir) {
            push_unique(&mut out, &mut seen, &path, ModelSource::Ollama)?;
        }
    }

    Ok(out)
}

fn push_unique(
    out: &mut Vec<LocalModel>,
    seen: &mut HashSet<PathBuf>,
    path: &Path,
    source: ModelSource,
) -> Result<()> {
    let key = fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    if seen.insert(key) {
        out.push(LocalModel::from_path(path, source)?);
    }
    Ok(())
}

/// All `*.gguf` files directly inside `dir` (non-recursive). Missing dir → empty.
fn gguf_files_in(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let Ok(entries) = fs::read_dir(dir) else {
        return found;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() && path.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf")) {
            found.push(path);
        }
    }
    found.sort();
    found
}

/// Ollama blobs are extension-less and content-addressed, so we identify GGUF by
/// the magic header rather than the name.
fn gguf_blobs_in(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let Ok(entries) = fs::read_dir(dir) else {
        return found;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() && has_gguf_magic(&path).unwrap_or(false) {
            found.push(path);
        }
    }
    found.sort();
    found
}

/// Read the first four bytes and compare against the GGUF magic.
pub fn has_gguf_magic(path: &Path) -> Result<bool> {
    use std::io::Read;
    let mut file = File::open(path)
        .with_context(|| format!("could not open {} to verify GGUF magic", path.display()))?;
    let mut magic = [0u8; 4];
    match file.read_exact(&mut magic) {
        Ok(()) => Ok(&magic == GGUF_MAGIC),
        // Too small to be a model — definitively not GGUF.
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(false),
        Err(e) => Err(e).with_context(|| format!("reading magic from {}", path.display())),
    }
}

/// Interactively pick one model from `models` via a `dialoguer` menu. Caller
/// guarantees `models` is non-empty.
pub fn prompt_select(models: &[LocalModel]) -> Result<usize> {
    let items: Vec<String> = models
        .iter()
        .map(|m| format!("{}  [{}]  {}", m.name, m.source.label(), m.human_size()))
        .collect();

    let idx = Select::with_theme(&ColorfulTheme::default())
        .with_prompt("Select a model")
        .items(&items)
        .default(0)
        .interact()
        .context("model selection cancelled")?;
    Ok(idx)
}

// ── Pull ────────────────────────────────────────────────────────────────────

/// Resolve a `models pull` spec — either a direct `http(s)://…` URL or a short
/// alias — to a download URL.
pub fn resolve_spec(spec: &str) -> Result<String> {
    if spec.starts_with("http://") || spec.starts_with("https://") {
        return Ok(spec.to_string());
    }
    let url = match spec.to_ascii_lowercase().as_str() {
        "default" | "7b" | "qwen2.5-coder-7b" => hf_url(
            "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        ),
        "3b" | "qwen2.5-coder-3b" => hf_url(
            "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
            "qwen2.5-coder-3b-instruct-q4_k_m.gguf",
        ),
        "1.5b" | "qwen2.5-coder-1.5b" => hf_url(
            "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
            "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        ),
        other => {
            return Err(anyhow!(
                "unknown model alias '{other}'. Use a direct https URL, or one of: \
                 default, 7b, 3b, 1.5b"
            ))
        }
    };
    Ok(url)
}

fn hf_url(repo: &str, file: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/main/{file}?download=true")
}

/// Print a cargo-style status line: a right-aligned, green, bold verb followed
/// by a message, e.g. `  Downloading https://…`.
fn status(verb: &str, message: &str) {
    let label = format!("{verb:>12}");
    println!("{} {}", Style::new().green().bold().apply_to(label), message);
}

/// Download the GGUF named by `spec` into `dest_dir`, streaming with a progress
/// bar. Returns the final on-disk path. The download lands in a `.part` file and
/// is renamed into place only on success, so an interrupted pull never leaves a
/// truncated model that looks valid.
pub fn pull(spec: &str, dest_dir: &Path) -> Result<PathBuf> {
    let url = resolve_spec(spec)?;
    let file_name = url_file_name(&url)
        .ok_or_else(|| anyhow!("could not derive a file name from URL: {url}"))?;
    let dest = dest_dir.join(&file_name);

    if dest.exists() {
        status("Cached", &dest.display().to_string());
        return Ok(dest);
    }

    fs::create_dir_all(dest_dir)
        .with_context(|| format!("could not create model dir {}", dest_dir.display()))?;

    status("Downloading", &url);

    let agent = ureq::AgentBuilder::new().redirects(10).build();
    let resp = agent
        .get(&url)
        .call()
        .with_context(|| format!("HTTP request failed for {url}"))?;

    let total: Option<u64> = resp
        .header("Content-Length")
        .and_then(|h| h.parse::<u64>().ok());

    let pb = match total {
        Some(len) => {
            let pb = ProgressBar::new(len);
            pb.set_style(
                ProgressStyle::with_template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] \
                     {bytes}/{total_bytes} ({bytes_per_sec}, eta {eta})",
                )
                .expect("valid progress template")
                .progress_chars("█▓░"),
            );
            pb
        }
        None => {
            let pb = ProgressBar::new_spinner();
            pb.set_style(
                ProgressStyle::with_template("{spinner:.green} {bytes} ({bytes_per_sec})")
                    .expect("valid spinner template"),
            );
            pb
        }
    };

    let tmp = dest.with_extension("part");
    let mut writer = File::create(&tmp)
        .with_context(|| format!("could not create {}", tmp.display()))?;
    let mut reader = pb.wrap_read(resp.into_reader());

    io::copy(&mut reader, &mut writer)
        .with_context(|| format!("download stream failed for {url}"))?;
    writer.sync_all().ok();
    drop(writer);
    pb.finish_and_clear();

    // Reject anything that did not arrive as a real GGUF before publishing it.
    if !has_gguf_magic(&tmp).unwrap_or(false) {
        fs::remove_file(&tmp).ok();
        return Err(anyhow!(
            "downloaded file is not a valid GGUF (wrong URL or an HTML error page?)"
        ));
    }

    fs::rename(&tmp, &dest)
        .with_context(|| format!("could not move {} into place", tmp.display()))?;

    let size = fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
    status("Downloaded", &format!("{file_name} ({})", human_bytes(size)));

    Ok(dest)
}

fn url_file_name(url: &str) -> Option<String> {
    url.split('?')
        .next()?
        .rsplit('/')
        .next()
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

// ── Formatting helpers ──────────────────────────────────────────────────────

fn short_blob_name(path: &Path) -> String {
    let raw = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    // `sha256-deadbeef…` → `ollama:deadbeef`.
    let hash = raw.rsplit('-').next().unwrap_or(&raw);
    let short: String = hash.chars().take(12).collect();
    format!("ollama:{short}")
}

/// Format a byte count as a binary-prefixed human string.
pub fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut size = bytes as f64;
    let mut unit = 0;
    while size >= 1024.0 && unit < UNITS.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} {}", UNITS[unit])
    } else {
        format!("{size:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_bytes_scales() {
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(1024), "1.0 KiB");
        assert_eq!(human_bytes(4_700_000_000), "4.4 GiB");
    }

    #[test]
    fn aliases_resolve_to_hf_urls() {
        assert!(resolve_spec("default").unwrap().contains("7B-Instruct-GGUF"));
        assert!(resolve_spec("3b").unwrap().contains("3B-Instruct-GGUF"));
        assert_eq!(
            resolve_spec("https://example.com/x.gguf").unwrap(),
            "https://example.com/x.gguf"
        );
        assert!(resolve_spec("bogus").is_err());
    }

    #[test]
    fn url_file_name_strips_query() {
        assert_eq!(
            url_file_name("https://h.co/a/b/model.gguf?download=true").as_deref(),
            Some("model.gguf")
        );
    }

    #[test]
    fn gguf_magic_detects_header() {
        let dir = std::env::temp_dir().join(format!("nixagent-gguf-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let good = dir.join("a.gguf");
        let bad = dir.join("b.bin");
        fs::write(&good, b"GGUF\x00\x01rest").unwrap();
        fs::write(&bad, b"NOPE....").unwrap();
        assert!(has_gguf_magic(&good).unwrap());
        assert!(!has_gguf_magic(&bad).unwrap());
        fs::remove_dir_all(&dir).ok();
    }
}
