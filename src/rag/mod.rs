//! Local RAG index + prompt assembly for the self-healing loop.
//!
//! Two responsibilities:
//!   1. [`NixOptionIndex`] — an embedded `rusqlite` store of the NixOS option
//!      universe (`environment.systemPackages`, `services.openssh.enable`, …),
//!      ingested from a `nixosOptionsDoc` JSON dump and queried with a hybrid
//!      path/description search. This is the *retrieval* half of RAG.
//!   2. [`LocalLlmHealer`] — implements [`CodeHealer`] from the `execution`
//!      module. On a failed build it pulls the broken symbol out of the
//!      [`HealingContext`], grounds it against the index, and assembles a strict,
//!      hallucination-resistant prompt for a local model (Gemma/Qwen via Ollama).
//!
//! Everything here is offline and synchronous against SQLite; only the final
//! model call is async, behind the [`LlmBackend`] seam so it can be swapped for a
//! real Ollama/llama.cpp socket without touching the retrieval or prompt logic.

use std::path::PathBuf;

use rusqlite::{params, params_from_iter, Connection};
use serde_json::Value;

use crate::config::HardwareTier;
use crate::execution::{CodeHealer, CorrelatedNixError, HealingContext, HealingFailure};

// ── Errors ────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum RagError {
    Db(rusqlite::Error),
    Json(serde_json::Error),
    /// The JSON dump did not match the expected option-dump shape.
    Schema(String),
    /// Model acquisition or in-process inference failure.
    Backend(String),
}

impl std::fmt::Display for RagError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Db(e) => write!(f, "rag database error: {}", e),
            Self::Json(e) => write!(f, "rag json error: {}", e),
            Self::Schema(m) => write!(f, "rag schema error: {}", m),
            Self::Backend(m) => write!(f, "inference backend error: {}", m),
        }
    }
}

impl std::error::Error for RagError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Db(e) => Some(e),
            Self::Json(e) => Some(e),
            Self::Schema(_) | Self::Backend(_) => None,
        }
    }
}

impl From<rusqlite::Error> for RagError {
    fn from(e: rusqlite::Error) -> Self {
        Self::Db(e)
    }
}
impl From<serde_json::Error> for RagError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

// ── Option record ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct NixOption {
    pub attribute_path: String,
    pub type_name: String,
    pub description: String,
    pub default_value: Option<String>,
}

// ── Index ─────────────────────────────────────────────────────────────────────

pub struct NixOptionIndex {
    conn: Connection,
}

impl NixOptionIndex {
    /// Open (or create) an on-disk index.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, RagError> {
        Ok(Self {
            conn: Connection::open(path)?,
        })
    }

    /// Open an ephemeral in-memory index (tests, dry runs).
    pub fn open_in_memory() -> Result<Self, RagError> {
        Ok(Self {
            conn: Connection::open_in_memory()?,
        })
    }

    /// Initialise the schema. Idempotent.
    ///
    /// The `attribute_path` PRIMARY KEY already yields a unique BINARY index for
    /// exact lookups and anchored prefix scans; we additionally create a NOCASE
    /// index so case-insensitive prefix queries (`services.openSSH%`) stay on an
    /// index rather than falling back to a full scan.
    pub fn bootstrap(&self) -> Result<(), RagError> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS nix_options (
                 attribute_path TEXT PRIMARY KEY,
                 type_name      TEXT NOT NULL,
                 description    TEXT NOT NULL,
                 default_value  TEXT
             );
             CREATE INDEX IF NOT EXISTS idx_nix_options_path_nocase
                 ON nix_options (attribute_path COLLATE NOCASE);",
        )?;
        Ok(())
    }

    /// Number of options currently stored.
    pub fn count(&self) -> Result<usize, RagError> {
        let n: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM nix_options", [], |r| r.get(0))?;
        Ok(n as usize)
    }

    /// Ingest a `nixosOptionsDoc` JSON dump in a single transaction.
    ///
    /// Accepts both the bare top-level map (`{ "<path>": { … } }`) and the
    /// wrapped form (`{ "options": { … } }`). Per-entry fields are extracted
    /// defensively: `description`/`type` may be plain strings or `{ _type, text }`
    /// objects, and `default` may be absent, a raw value, or a literal wrapper.
    /// Returns the number of options written.
    pub fn ingest_json_dump(&mut self, json_str: &str) -> Result<usize, RagError> {
        let root: Value = serde_json::from_str(json_str)?;
        let map = match &root {
            Value::Object(m) => match m.get("options") {
                Some(Value::Object(inner)) => inner,
                _ => m,
            },
            _ => return Err(RagError::Schema("expected a top-level JSON object".into())),
        };

        let tx = self.conn.transaction()?;
        let mut count = 0usize;
        {
            let mut stmt = tx.prepare(
                "INSERT OR REPLACE INTO nix_options
                     (attribute_path, type_name, description, default_value)
                 VALUES (?1, ?2, ?3, ?4)",
            )?;

            for (path, entry) in map {
                let Some(obj) = entry.as_object() else {
                    continue;
                };
                let type_name = obj
                    .get("type")
                    .and_then(extract_text)
                    .unwrap_or_else(|| "unknown".to_owned());
                let description = obj
                    .get("description")
                    .and_then(extract_text)
                    .unwrap_or_default();
                let default_value = extract_default(obj.get("default"));

                stmt.execute(params![path, type_name, description, default_value])?;
                count += 1;
            }
        }
        tx.commit()?;
        Ok(count)
    }

    /// Hybrid search over option paths and descriptions.
    ///
    /// SQLite's case-insensitive `LIKE` gathers a candidate set (path substring,
    /// or any keyword in path/description); candidates are then re-ranked in Rust
    /// by a composite score (exact path ≫ leaf match ≫ substring ≫ keyword hits)
    /// and truncated to `limit`.
    pub fn search_options(&self, query: &str, limit: usize) -> Result<Vec<NixOption>, RagError> {
        let query = query.trim();
        if query.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let query_lc = query.to_ascii_lowercase();
        let keywords = tokenize(query);

        let mut clauses: Vec<&str> = vec!["attribute_path LIKE ?"];
        let mut binds: Vec<String> = vec![format!("%{}%", query)];
        for kw in &keywords {
            clauses.push("attribute_path LIKE ?");
            binds.push(format!("%{}%", kw));
            clauses.push("description LIKE ?");
            binds.push(format!("%{}%", kw));
        }
        let sql = format!(
            "SELECT attribute_path, type_name, description, default_value \
             FROM nix_options WHERE {}",
            clauses.join(" OR "),
        );

        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(binds.iter()), |row| {
            Ok(NixOption {
                attribute_path: row.get(0)?,
                type_name: row.get(1)?,
                description: row.get(2)?,
                default_value: row.get(3)?,
            })
        })?;

        let mut scored: Vec<(i64, NixOption)> = Vec::new();
        for row in rows {
            let opt = row?;
            let s = score_option(&opt, &query_lc, &keywords);
            scored.push((s, opt));
        }
        scored.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| a.1.attribute_path.cmp(&b.1.attribute_path))
        });
        Ok(scored.into_iter().take(limit).map(|(_, o)| o).collect())
    }
}

// ── JSON extraction helpers ───────────────────────────────────────────────────

/// A field that is either a plain string or a `{ "_type": …, "text": … }` wrapper.
fn extract_text(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Object(m) => m.get("text").and_then(Value::as_str).map(str::to_owned),
        _ => None,
    }
}

/// The `default` field: absent/null → `None`; literal wrapper → its `text`;
/// otherwise the compact JSON rendering of the raw value.
fn extract_default(v: Option<&Value>) -> Option<String> {
    match v {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Object(m)) => match m.get("text").and_then(Value::as_str) {
            Some(text) => Some(text.to_owned()),
            None => Some(Value::Object(m.clone()).to_string()),
        },
        Some(other) => Some(other.to_string()),
    }
}

// ── Ranking ───────────────────────────────────────────────────────────────────

fn tokenize(s: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for tok in s.split(|c: char| !c.is_alphanumeric()) {
        let t = tok.to_ascii_lowercase();
        if t.len() >= 2 && !out.contains(&t) {
            out.push(t);
        }
    }
    out
}

fn score_option(opt: &NixOption, query_lc: &str, keywords: &[String]) -> i64 {
    let path_lc = opt.attribute_path.to_ascii_lowercase();
    let desc_lc = opt.description.to_ascii_lowercase();
    let mut score = 0i64;

    if path_lc == query_lc {
        score += 1000;
    }
    if path_lc.rsplit('.').next() == Some(query_lc) {
        score += 500;
    }
    if path_lc.contains(query_lc) {
        score += 200;
    }
    if path_lc.starts_with(query_lc) {
        score += 100;
    }
    for kw in keywords {
        if path_lc.contains(kw) {
            score += 50;
        }
        if desc_lc.contains(kw) {
            score += 10;
        }
    }
    // Tie-break toward more specific (shorter) option paths.
    score - (opt.attribute_path.len() as i64) / 20
}

// ── LLM backend seam ──────────────────────────────────────────────────────────

/// The async call to the local model. The production implementation is
/// [`EmbeddedLlamaBackend`], which runs a GGUF model in-process; [`StubBackend`]
/// is a deterministic offline stand-in for tests.
#[allow(async_fn_in_trait)]
pub trait LlmBackend {
    async fn complete(&self, prompt: &str) -> anyhow::Result<String>;
}

/// Deterministic, offline backend used by tests and as the default healer type
/// parameter. Performs no inference — real generation uses [`EmbeddedLlamaBackend`].
#[derive(Debug, Clone, Default)]
pub struct StubBackend;

impl LlmBackend for StubBackend {
    async fn complete(&self, prompt: &str) -> anyhow::Result<String> {
        // Emit a *valid, empty* NixOS module so the plan/apply workflow can be
        // exercised end-to-end offline. Real generation requires `embedded-llm`.
        Ok(format!(
            "{{ config, pkgs, ... }}:\n{{\n  \
             # Generated by nix-agent (offline stub backend — no model inference).\n  \
             # Build with `--features embedded-llm` for real generation.\n  \
             # prompt was {} bytes.\n}}\n",
            prompt.len(),
        ))
    }
}

/// In-process GGUF inference backend.
///
/// Picks the model from the host's [`HardwareTier`], downloads it from Hugging
/// Face on first run, and loads it straight into process memory with full GPU
/// offload (`n_gpu_layers = 99`) so llama.cpp uses native Metal (macOS) or
/// Vulkan/HIP (Linux/NixOS) acceleration.
///
/// The native pieces (`llama-cpp-2`, `hf-hub`) are compiled only under the
/// `embedded-llm` feature; without it, [`Self::load`] returns a clear error.
pub struct EmbeddedLlamaBackend {
    tier: HardwareTier,
    model_path: PathBuf,
    #[cfg(feature = "embedded-llm")]
    engine: embedded::Engine,
}

impl EmbeddedLlamaBackend {
    /// Resolve (downloading if necessary) and load the tier's model into memory.
    /// `progress` receives user-facing, English status lines (e.g. the first-run
    /// download notice).
    pub fn load(
        tier: HardwareTier,
        cache_dir: &std::path::Path,
        #[allow(unused_mut, unused_variables)] mut progress: impl FnMut(&str),
    ) -> Result<Self, RagError> {
        #[cfg(not(feature = "embedded-llm"))]
        {
            let _ = cache_dir;
            Err(RagError::Backend(format!(
                "embedded inference for tier {} is not compiled in; rebuild with \
                 `--features embedded-llm` (or `--features metal`/`--features vulkan`), \
                 which requires a C/C++ toolchain and cmake",
                tier.label(),
            )))
        }
        #[cfg(feature = "embedded-llm")]
        {
            let model_path = embedded::ensure_model(tier, cache_dir, &mut progress)?;
            let engine = embedded::Engine::load(&model_path, FULL_GPU_OFFLOAD)
                .map_err(|e| RagError::Backend(format!("failed to load GGUF model: {e}")))?;
            Ok(Self {
                tier,
                model_path,
                engine,
            })
        }
    }

    /// The detected hardware tier this backend was built for.
    pub fn tier(&self) -> HardwareTier {
        self.tier
    }

    /// Local path of the loaded GGUF model file.
    pub fn model_path(&self) -> &std::path::Path {
        &self.model_path
    }
}

impl LlmBackend for EmbeddedLlamaBackend {
    async fn complete(&self, prompt: &str) -> anyhow::Result<String> {
        #[cfg(not(feature = "embedded-llm"))]
        {
            let _ = prompt;
            anyhow::bail!(
                "embedded inference is not compiled in; rebuild with `--features embedded-llm`"
            )
        }
        #[cfg(feature = "embedded-llm")]
        {
            // Cap generation length; the model returns a full Nix expression.
            const MAX_TOKENS: usize = 2048;
            self.engine
                .generate(prompt, MAX_TOKENS)
                .map_err(|e| anyhow::anyhow!("inference failed: {e}"))
        }
    }
}

/// Offload every transformer layer to the GPU. 99 is the conventional llama.cpp
/// "all layers" sentinel, enabling full Metal/Vulkan/HIP acceleration.
#[cfg(feature = "embedded-llm")]
const FULL_GPU_OFFLOAD: u32 = 99;

/// Native inference + model-download glue. Compiled only with `embedded-llm`.
///
/// Targets the `llama-cpp-2` 0.1.x API generation; if you bump that crate and
/// the sampler/context surface shifts, this is the single module to update.
#[cfg(feature = "embedded-llm")]
mod embedded {
    use std::num::NonZeroU32;
    use std::path::{Path, PathBuf};

    use hf_hub::api::sync::ApiBuilder;
    use hf_hub::{Cache, Repo, RepoType};

    use llama_cpp_2::context::params::LlamaContextParams;
    use llama_cpp_2::llama_backend::LlamaBackend;
    use llama_cpp_2::llama_batch::LlamaBatch;
    use llama_cpp_2::model::params::LlamaModelParams;
    use llama_cpp_2::model::{AddBos, LlamaModel};
    use llama_cpp_2::sampling::LlamaSampler;
    use llama_cpp_2::token::LlamaToken;

    use crate::config::HardwareTier;

    use super::RagError;

    /// Look up the tier's GGUF in the hf-hub cache; download it (announcing the
    /// first-run progress line) only if it is missing. Returns the local path.
    pub fn ensure_model(
        tier: HardwareTier,
        cache_dir: &Path,
        progress: &mut impl FnMut(&str),
    ) -> Result<PathBuf, RagError> {
        let repo = tier.model_repo().to_owned();
        let file = tier.model_file();

        // Offline cache probe first, so we only announce a download when needed.
        let cached = Cache::new(cache_dir.to_path_buf())
            .repo(Repo::new(repo.clone(), RepoType::Model))
            .get(file);

        if cached.is_none() {
            progress(&format!(
                "\u{2b07}\u{fe0f} [First Run] Downloading the optimal AI model for your hardware (Size: {:.1} GB)...",
                tier.approx_download_gb(),
            ));
        }

        let api = ApiBuilder::new()
            .with_cache_dir(cache_dir.to_path_buf())
            .build()
            .map_err(|e| RagError::Backend(format!("hf-hub init failed: {e}")))?;

        api.repo(Repo::new(repo, RepoType::Model))
            .get(file)
            .map_err(|e| RagError::Backend(format!("model download failed: {e}")))
    }

    /// A model loaded into process memory, reusable across generations.
    pub struct Engine {
        backend: LlamaBackend,
        model: LlamaModel,
    }

    impl Engine {
        /// Load `path` with `n_gpu_layers` offloaded to the GPU.
        pub fn load(path: &Path, n_gpu_layers: u32) -> anyhow::Result<Self> {
            let backend = LlamaBackend::init()?;
            let params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
            let model = LlamaModel::load_from_file(&backend, path, &params)?;
            Ok(Self { backend, model })
        }

        /// Greedily generate up to `max_tokens` tokens of continuation.
        pub fn generate(&self, prompt: &str, max_tokens: usize) -> anyhow::Result<String> {
            let n_ctx = NonZeroU32::new(8192).expect("non-zero context");
            let ctx_params = LlamaContextParams::default().with_n_ctx(Some(n_ctx));
            let mut ctx = self.model.new_context(&self.backend, ctx_params)?;

            let tokens = self.model.str_to_token(prompt, AddBos::Always)?;
            let mut batch = LlamaBatch::new(8192, 1);
            let last = tokens.len().saturating_sub(1);
            for (i, tok) in tokens.iter().enumerate() {
                batch.add(*tok, i as i32, &[0], i == last)?;
            }
            ctx.decode(&mut batch)?;

            let mut sampler = LlamaSampler::greedy();
            // A single streaming decoder across the loop so multi-byte UTF-8
            // tokens are reassembled correctly.
            let mut decoder = encoding_rs::UTF_8.new_decoder();
            let mut out = String::new();
            // Absolute KV-cache position of the first generated token.
            let start_pos = batch.n_tokens();

            for i in 0..max_tokens {
                let token: LlamaToken = sampler.sample(&ctx, batch.n_tokens() - 1);
                sampler.accept(token);
                if self.model.is_eog_token(token) {
                    break;
                }
                out.push_str(&self.model.token_to_piece(token, &mut decoder, false, None)?);

                batch.clear();
                batch.add(token, start_pos + i as i32, &[0], true)?;
                ctx.decode(&mut batch)?;
            }
            Ok(out)
        }
    }
}

// ── Healer ────────────────────────────────────────────────────────────────────

const SYSTEM_PREAMBLE: &str = "\
You are nix-agent, a NixOS configuration repair engine. You receive a broken Nix
configuration, the exact build error, and a list of VERIFIED NixOS options pulled
from the local option database.

STRICT RULES:
- Output ONLY a single, complete, valid Nix expression. No prose, no markdown.
- Use ONLY option attribute paths that appear in the VERIFIED NIXOS OPTIONS list.
  Never invent, guess, or hallucinate option names, types, or attributes.
- Preserve all unrelated parts of the configuration exactly as given.
- Make the minimal change required to resolve the reported error.
";

const INSTRUCTION_FOOTER: &str = "\
\n## TASK
Return the corrected configuration in full. Output Nix source only.
";

/// A [`CodeHealer`] that grounds re-generation in the local option index.
pub struct LocalLlmHealer<B: LlmBackend = StubBackend> {
    index: NixOptionIndex,
    backend: B,
    /// How many verified options to inject into the prompt.
    max_options: usize,
}

impl LocalLlmHealer<StubBackend> {
    /// Construct with the offline stub backend (tests / dry runs).
    pub fn new(index: NixOptionIndex) -> Self {
        Self::with_backend(index, StubBackend)
    }
}

impl<B: LlmBackend> LocalLlmHealer<B> {
    pub fn with_backend(index: NixOptionIndex, backend: B) -> Self {
        Self {
            index,
            backend,
            max_options: 8,
        }
    }

    pub fn max_options(mut self, n: usize) -> Self {
        self.max_options = n;
        self
    }

    /// Borrow the underlying index (e.g. to bootstrap/ingest before healing).
    pub fn index(&self) -> &NixOptionIndex {
        &self.index
    }

    /// Assemble the initial generation prompt from a natural-language request,
    /// grounded in options retrieved from the local index. This is the `run`
    /// entry point — there is no failure context yet, just the user's intent.
    pub fn assemble_generation_prompt(&self, user_prompt: &str) -> Result<String, RagError> {
        let options = self.index.search_options(user_prompt, self.max_options)?;
        let mut p = String::with_capacity(2048);
        p.push_str(SYSTEM_PREAMBLE);
        render_verified_options(&mut p, &options);
        p.push_str("\n## USER REQUEST\n");
        p.push_str(user_prompt.trim());
        p.push('\n');
        p.push_str(INSTRUCTION_FOOTER);
        Ok(p)
    }

    /// Produce the first candidate configuration for a user request by grounding
    /// it against the index and invoking the local model.
    pub async fn draft(&self, user_prompt: &str) -> anyhow::Result<String> {
        let prompt = self.assemble_generation_prompt(user_prompt)?;
        let raw = self.backend.complete(&prompt).await?;
        Ok(strip_code_fences(&raw))
    }

    /// Assemble the full RAG + error prompt. Pure and deterministic — this is the
    /// testable heart of the healer; the model call merely consumes its output.
    pub fn assemble_prompt(&self, ctx: &HealingContext) -> Result<String, RagError> {
        let (symbol, options) = self.gather_grounding(ctx)?;

        let mut p = String::with_capacity(2048);
        p.push_str(SYSTEM_PREAMBLE);
        render_verified_options(&mut p, &options);

        p.push_str("\n## BUILD FAILURE\n");
        p.push_str(&describe_failure(ctx));
        if let Some(sym) = &symbol {
            p.push_str(&format!("broken symbol: `{}`\n", sym));
        }

        p.push_str("\n## CURRENT (BROKEN) CONFIGURATION\n```nix\n");
        p.push_str(&ctx.previous_code);
        if !ctx.previous_code.ends_with('\n') {
            p.push('\n');
        }
        p.push_str("```\n");

        p.push_str(INSTRUCTION_FOOTER);
        Ok(p)
    }

    /// Decide the retrieval query and pull grounding options from the index.
    /// Syntax (AST) failures carry no symbol, so no option lookup is performed.
    fn gather_grounding(
        &self,
        ctx: &HealingContext,
    ) -> Result<(Option<String>, Vec<NixOption>), RagError> {
        match &ctx.failure {
            HealingFailure::Ast { .. } => Ok((None, Vec::new())),
            HealingFailure::Build { error, correlated } => {
                let query = pick_query(error.symbol.as_deref(), &error.message, correlated);
                let options = match &query {
                    Some(q) => self.index.search_options(q, self.max_options)?,
                    None => Vec::new(),
                };
                Ok((error.symbol.clone().or(query), options))
            }
        }
    }
}

impl<B: LlmBackend> CodeHealer for LocalLlmHealer<B> {
    async fn generate(&mut self, ctx: &HealingContext) -> anyhow::Result<String> {
        let prompt = self.assemble_prompt(ctx)?;
        let raw = self.backend.complete(&prompt).await?;
        Ok(strip_code_fences(&raw))
    }
}

// ── Prompt helpers ────────────────────────────────────────────────────────────

/// Choose the retrieval query: a correlated missing-attribute path is the
/// strongest signal, then the build error's symbol, then the most salient word
/// from the error message.
fn pick_query(
    symbol: Option<&str>,
    message: &str,
    correlated: &Option<CorrelatedNixError>,
) -> Option<String> {
    if let Some(CorrelatedNixError::AttrNotFound { attr_path }) = correlated {
        return Some(attr_path.clone());
    }
    if let Some(sym) = symbol {
        return Some(sym.to_owned());
    }
    tokenize(message).into_iter().find(|w| w.len() >= 3)
}

/// Render the authoritative, grounded option list shared by the generation and
/// healing prompts. Empty results emit an explicit "do not invent" guard.
fn render_verified_options(p: &mut String, options: &[NixOption]) {
    p.push_str("\n## VERIFIED NIXOS OPTIONS (authoritative — use ONLY these)\n");
    if options.is_empty() {
        p.push_str("(no matching options found in the local index — do NOT invent any)\n");
        return;
    }
    for o in options {
        p.push_str(&format!(
            "- `{}` : {}\n    {}\n",
            o.attribute_path,
            o.type_name,
            truncate(&o.description, 200),
        ));
        if let Some(d) = &o.default_value {
            p.push_str(&format!("    default: {}\n", truncate(d, 80)));
        }
    }
}

fn describe_failure(ctx: &HealingContext) -> String {
    match &ctx.failure {
        HealingFailure::Ast { diagnostics } => {
            let mut s = String::from("type: syntax/parse error (rejected before build)\n");
            for d in diagnostics {
                let (line, col) = byte_to_line_col(&ctx.previous_code, d.byte_start);
                s.push_str(&format!(
                    "- {} at line {}, col {} (bytes {}..{})\n",
                    d.message, line, col, d.byte_start, d.byte_end,
                ));
            }
            s
        }
        HealingFailure::Build { error, correlated } => {
            let mut s = format!("type: {:?}\nmessage: {}\n", error.kind, error.message);
            if let Some(loc) = &error.location {
                s.push_str(&format!(
                    "location: {}:{}:{}\n",
                    loc.file.display(),
                    loc.line,
                    loc.column,
                ));
            }
            if let Some(c) = correlated {
                s.push_str(&format!("ast_correlation: {:?}\n", c));
            }
            s
        }
    }
}

/// Strip a leading ```/```nix fence and trailing ``` a model may wrap output in.
fn strip_code_fences(s: &str) -> String {
    let t = s.trim();
    if let Some(rest) = t.strip_prefix("```") {
        let body = rest.split_once('\n').map(|x| x.1).unwrap_or("");
        let body = body.strip_suffix("```").unwrap_or(body);
        return body.trim_end().to_owned();
    }
    t.to_owned()
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_owned();
    }
    let head: String = s.chars().take(max).collect();
    format!("{}…", head.trim_end())
}

/// 0-based byte offset → 1-based `(line, column)`.
fn byte_to_line_col(src: &str, byte: u32) -> (u32, u32) {
    let byte = byte as usize;
    let mut line = 1u32;
    let mut col = 1u32;
    for (i, ch) in src.char_indices() {
        if i >= byte {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{NixBuildError, NixBuildErrorKind, SourceLocation};
    use std::path::PathBuf;
    use std::sync::Mutex;

    const FAKE_DUMP: &str = r#"
    {
      "services.openssh.enable": {
        "description": "Whether to enable the OpenSSH secure shell daemon.",
        "type": "boolean",
        "default": { "_type": "literalExpression", "text": "false" },
        "loc": ["services", "openssh", "enable"]
      },
      "services.openssh.ports": {
        "description": "Specifies on which ports the SSH daemon listens.",
        "type": "list of signed integer",
        "default": [22]
      },
      "networking.firewall.enable": {
        "description": "Whether to enable the firewall.",
        "type": "boolean",
        "default": true
      },
      "environment.systemPackages": {
        "description": "The set of packages that appear in /run/current-system/sw.",
        "type": "list of package",
        "default": []
      }
    }"#;

    fn seeded_index() -> NixOptionIndex {
        let mut idx = NixOptionIndex::open_in_memory().unwrap();
        idx.bootstrap().unwrap();
        let n = idx.ingest_json_dump(FAKE_DUMP).unwrap();
        assert_eq!(n, 4);
        idx
    }

    fn build_ctx(error: NixBuildError, correlated: Option<CorrelatedNixError>, code: &str) -> HealingContext {
        HealingContext {
            attempt: 1,
            max_attempts: 3,
            previous_code: code.to_owned(),
            failure: HealingFailure::Build { error, correlated },
        }
    }

    // ── index ─────────────────────────────────────────────────────────────────

    #[test]
    fn ingest_extracts_fields_and_defaults() {
        let idx = seeded_index();
        assert_eq!(idx.count().unwrap(), 4);

        // literal-wrapper default → its text
        let ssh = &idx.search_options("services.openssh.enable", 1).unwrap()[0];
        assert_eq!(ssh.type_name, "boolean");
        assert_eq!(ssh.default_value.as_deref(), Some("false"));

        // raw bool default → compact JSON
        let fw = &idx.search_options("networking.firewall.enable", 1).unwrap()[0];
        assert_eq!(fw.default_value.as_deref(), Some("true"));

        // raw list default → compact JSON
        let pkgs = &idx.search_options("environment.systemPackages", 1).unwrap()[0];
        assert_eq!(pkgs.default_value.as_deref(), Some("[]"));
    }

    #[test]
    fn exact_path_ranks_first() {
        let idx = seeded_index();
        let hits = idx.search_options("services.openssh.enable", 5).unwrap();
        assert_eq!(hits[0].attribute_path, "services.openssh.enable");
    }

    #[test]
    fn substring_path_search() {
        let idx = seeded_index();
        let hits = idx.search_options("openssh", 5).unwrap();
        let paths: Vec<_> = hits.iter().map(|o| o.attribute_path.as_str()).collect();
        assert!(paths.contains(&"services.openssh.enable"));
        assert!(paths.contains(&"services.openssh.ports"));
        assert!(!paths.contains(&"networking.firewall.enable"));
    }

    #[test]
    fn description_keyword_search() {
        let idx = seeded_index();
        // "firewall" only appears in the networking option's path+description.
        let hits = idx.search_options("firewall", 5).unwrap();
        assert_eq!(hits[0].attribute_path, "networking.firewall.enable");

        // "packages" appears only inside the camelCase path / description.
        let hits = idx.search_options("packages", 5).unwrap();
        assert_eq!(hits[0].attribute_path, "environment.systemPackages");
    }

    #[test]
    fn empty_query_returns_empty() {
        let idx = seeded_index();
        assert!(idx.search_options("   ", 5).unwrap().is_empty());
        assert!(idx.search_options("openssh", 0).unwrap().is_empty());
    }

    #[test]
    fn ingest_accepts_wrapped_options_form() {
        let mut idx = NixOptionIndex::open_in_memory().unwrap();
        idx.bootstrap().unwrap();
        let wrapped = r#"{ "options": { "networking.hostName": { "type": "string", "description": "The hostname." } } }"#;
        assert_eq!(idx.ingest_json_dump(wrapped).unwrap(), 1);
        assert_eq!(idx.search_options("hostName", 1).unwrap()[0].attribute_path, "networking.hostName");
    }

    // ── prompt assembly ─────────────────────────────────────────────────────

    #[test]
    fn prompt_grounds_on_symbol() {
        let healer = LocalLlmHealer::new(seeded_index());
        let err = NixBuildError {
            kind: NixBuildErrorKind::UndefinedVariable,
            message: "undefined variable 'openssh'".to_owned(),
            location: Some(SourceLocation {
                file: PathBuf::from("/etc/nixos/configuration.nix"),
                line: 10,
                column: 7,
            }),
            symbol: Some("openssh".to_owned()),
            raw: String::new(),
        };
        let code = "{ services.openssh.enabel = true; }\n";
        let prompt = healer.assemble_prompt(&build_ctx(err, None, code)).unwrap();

        // Grounding facts present.
        assert!(prompt.contains("services.openssh.enable"));
        // Error context present.
        assert!(prompt.contains("UndefinedVariable"));
        assert!(prompt.contains("broken symbol: `openssh`"));
        assert!(prompt.contains("configuration.nix:10:7"));
        // Anti-hallucination preamble + original code present.
        assert!(prompt.contains("Never invent"));
        assert!(prompt.contains("services.openssh.enabel")); // the broken original
    }

    #[test]
    fn prompt_uses_correlated_attr_path_as_query() {
        let healer = LocalLlmHealer::new(seeded_index());
        let err = NixBuildError {
            kind: NixBuildErrorKind::MissingAttribute,
            message: "attribute 'enable' missing".to_owned(),
            location: None,
            symbol: Some("enable".to_owned()),
            raw: String::new(),
        };
        let correlated = Some(CorrelatedNixError::AttrNotFound {
            attr_path: "services.openssh.enable".to_owned(),
        });
        let prompt = healer
            .assemble_prompt(&build_ctx(err, correlated, "{ }\n"))
            .unwrap();
        // The exact attr path drove retrieval and is grounded first.
        assert!(prompt.contains("services.openssh.enable"));
        assert!(prompt.contains("ast_correlation"));
    }

    #[test]
    fn prompt_for_syntax_error_has_no_options_but_has_position() {
        let healer = LocalLlmHealer::new(seeded_index());
        let ctx = HealingContext {
            attempt: 2,
            max_attempts: 3,
            previous_code: "{ foo = ;".to_owned(),
            failure: HealingFailure::Ast {
                diagnostics: vec![crate::ast::ParseDiagnostic {
                    message: "unexpected token".to_owned(),
                    byte_start: 8,
                    byte_end: 9,
                }],
            },
        };
        let prompt = healer.assemble_prompt(&ctx).unwrap();
        assert!(prompt.contains("do NOT invent any"));
        assert!(prompt.contains("syntax/parse error"));
        assert!(prompt.contains("line 1, col 9"));
    }

    #[test]
    fn generation_prompt_grounds_on_user_request() {
        let healer = LocalLlmHealer::new(seeded_index());
        let prompt = healer
            .assemble_generation_prompt("enable the openssh daemon")
            .unwrap();
        // RAG retrieved the relevant verified option for the request.
        assert!(prompt.contains("services.openssh.enable"));
        // The user's intent and the strict preamble are both present.
        assert!(prompt.contains("USER REQUEST"));
        assert!(prompt.contains("enable the openssh daemon"));
        assert!(prompt.contains("Never invent"));
    }

    // ── generate end-to-end ──────────────────────────────────────────────────

    struct CapturingBackend {
        last_prompt: Mutex<Option<String>>,
        reply: String,
    }
    impl LlmBackend for CapturingBackend {
        async fn complete(&self, prompt: &str) -> anyhow::Result<String> {
            *self.last_prompt.lock().unwrap() = Some(prompt.to_owned());
            Ok(self.reply.clone())
        }
    }

    #[tokio::test]
    async fn generate_assembles_prompt_and_strips_fences() {
        let backend = CapturingBackend {
            last_prompt: Mutex::new(None),
            reply: "```nix\n{ services.openssh.enable = true; }\n```".to_owned(),
        };
        let mut healer = LocalLlmHealer::with_backend(seeded_index(), backend);

        let err = NixBuildError {
            kind: NixBuildErrorKind::UndefinedVariable,
            message: "undefined variable 'openssh'".to_owned(),
            location: None,
            symbol: Some("openssh".to_owned()),
            raw: String::new(),
        };
        let ctx = build_ctx(err, None, "{ services.openssh.enabel = true; }\n");

        let out = healer.generate(&ctx).await.unwrap();

        // Fences stripped → clean Nix.
        assert_eq!(out, "{ services.openssh.enable = true; }");
        assert!(!out.contains("```"));

        // The backend really received the grounded prompt.
        let seen = healer
            .into_captured_prompt()
            .expect("prompt was captured");
        assert!(seen.contains("VERIFIED NIXOS OPTIONS"));
        assert!(seen.contains("services.openssh.enable"));
    }

    // Small helper to extract the captured prompt out of the test backend.
    impl LocalLlmHealer<CapturingBackend> {
        fn into_captured_prompt(self) -> Option<String> {
            self.backend.last_prompt.into_inner().unwrap()
        }
    }
}
