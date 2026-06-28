//! Two-phase `plan` / `apply` workflow.
//!
//! The agent never edits the live system configuration. Instead:
//!   * **`plan`** generates an isolated Nix module, validates it through the
//!     engine's AST gate, and writes an annotated, ready-to-apply plan file.
//!   * **`apply`** installs that validated module into the sandbox directory
//!     (`<config_dir>/modules/ai-generated/<plan-id>.nix`) and runs the
//!     activation engine (`nixos-rebuild test`), rolling back on failure.
//!
//! The orchestrators here are generic over the [`SystemBuilder`] and
//! [`CodeHealer`] seams so the whole flow is exercised offline with mocks.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::config::AppConfig;
use crate::execution::{
    CodeHealer, EngineError, ExecutionEngine, HealEvent, HealingOutcome, SystemBuilder,
};

// ── Errors ────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum PlanError {
    Io { path: PathBuf, source: std::io::Error },
    Engine(EngineError),
    /// The plan file did not contain the expected header.
    Malformed(String),
    /// No plan could be resolved from the given path or id.
    NotFound(String),
    /// The requested id did not match the id stored in the plan file.
    IdMismatch { requested: String, found: String },
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "I/O error on {}: {}", path.display(), source),
            Self::Engine(e) => write!(f, "{}", e),
            Self::Malformed(m) => write!(f, "malformed plan file: {}", m),
            Self::NotFound(a) => write!(f, "no plan found for '{}'", a),
            Self::IdMismatch { requested, found } => {
                write!(f, "no plan with id '{}' (latest plan is '{}')", requested, found)
            }
        }
    }
}

impl std::error::Error for PlanError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Engine(e) => Some(e),
            _ => None,
        }
    }
}

impl From<EngineError> for PlanError {
    fn from(e: EngineError) -> Self {
        Self::Engine(e)
    }
}

// ── Plan ──────────────────────────────────────────────────────────────────────

/// A generated, validated module plus its provenance metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub id: String,
    pub prompt: String,
    pub module_source: String,
}

const HEADER_MARKER: &str = "# nix-agent plan v1";

impl Plan {
    /// Serialize to the on-disk plan format: a metadata comment header (valid
    /// Nix, since every line is a `#` comment) followed by the module body.
    pub fn render(&self) -> String {
        let mut body = self.module_source.trim_end().to_owned();
        body.push('\n');
        format!(
            "{marker}\n# plan-id: {id}\n# prompt: {prompt}\n\n{body}",
            marker = HEADER_MARKER,
            id = self.id,
            prompt = self.prompt.replace('\n', " "),
            body = body,
        )
    }

    /// Parse a plan file produced by [`Self::render`].
    pub fn parse(content: &str) -> Result<Plan, PlanError> {
        let mut id: Option<String> = None;
        let mut prompt = String::new();
        let mut body_lines: Vec<&str> = Vec::new();
        let mut in_header = true;

        for line in content.lines() {
            if in_header {
                let t = line.trim_start();
                if let Some(rest) = t.strip_prefix("# plan-id:") {
                    id = Some(rest.trim().to_owned());
                    continue;
                }
                if let Some(rest) = t.strip_prefix("# prompt:") {
                    prompt = rest.trim().to_owned();
                    continue;
                }
                if t.starts_with(HEADER_MARKER) {
                    continue;
                }
                if t.is_empty() {
                    // Blank line terminates the header block.
                    in_header = false;
                    continue;
                }
                // First substantive line ends the header; it belongs to the body.
                in_header = false;
                body_lines.push(line);
            } else {
                body_lines.push(line);
            }
        }

        let id = id.ok_or_else(|| PlanError::Malformed("missing `# plan-id:` header".to_owned()))?;
        let mut module_source = body_lines.join("\n");
        if !module_source.ends_with('\n') {
            module_source.push('\n');
        }
        Ok(Plan { id, prompt, module_source })
    }
}

// ── Plan id ───────────────────────────────────────────────────────────────────

/// Current UNIX time in seconds (saturating to 0 before the epoch).
pub fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Build a human-readable plan id like `2026-06-29-tmux-keybindings` from the
/// generation date and a slug derived from the prompt.
pub fn make_plan_id(prompt: &str, unix_secs: i64) -> String {
    let (y, m, d) = ymd_from_unix(unix_secs);
    let slug = derive_slug(prompt);
    if slug.is_empty() {
        format!("{y:04}-{m:02}-{d:02}")
    } else {
        format!("{y:04}-{m:02}-{d:02}-{slug}")
    }
}

/// Convert a UNIX timestamp to a civil `(year, month, day)` (UTC), using
/// Howard Hinnant's `civil_from_days` algorithm — no calendar dependency.
fn ymd_from_unix(secs: i64) -> (i64, u32, u32) {
    let days = secs.div_euclid(86_400);
    let z = days + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = if m <= 2 { y + 1 } else { y };
    (year, m as u32, d as u32)
}

/// Up to two salient lowercase tokens from the prompt, joined with `-`.
fn derive_slug(prompt: &str) -> String {
    const STOP: &[&str] = &[
        "add", "install", "enable", "disable", "remove", "with", "the", "and", "or", "custom",
        "configure", "config", "set", "setup", "for", "please", "make", "use", "using", "that",
    ];
    let mut parts: Vec<String> = Vec::new();
    for tok in prompt.split(|c: char| !c.is_ascii_alphanumeric()) {
        let lower = tok.to_ascii_lowercase();
        if lower.len() < 2 || STOP.contains(&lower.as_str()) {
            continue;
        }
        parts.push(lower);
        if parts.len() == 2 {
            break;
        }
    }
    parts.join("-")
}

// ── Outcomes ──────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum PlanOutcome {
    Validated { plan: Plan, attempts: usize },
    Rejected { reason: String, attempts: usize },
}

#[derive(Debug)]
pub enum ApplyOutcome {
    Activated { module_path: PathBuf },
    Failed { reason: String },
}

// ── Orchestration ─────────────────────────────────────────────────────────────

/// `plan` phase: validate `initial_module` (self-healing on AST failures) at the
/// isolated plan file, and — on success — persist an annotated plan. Never
/// touches the live system configuration or runs an activation.
pub async fn create_plan<B, H>(
    cfg: &AppConfig,
    prompt: &str,
    plan_id: &str,
    initial_module: String,
    healer: &mut H,
    builder: B,
    on_event: impl FnMut(HealEvent),
) -> Result<PlanOutcome, PlanError>
where
    B: SystemBuilder,
    H: CodeHealer,
{
    let engine = ExecutionEngine::with_builder(cfg.plan_file.clone(), builder)
        .max_attempts(cfg.max_attempts)
        .restore_on_failure(true);

    let outcome = engine
        .self_healing_loop_with(initial_module, healer, on_event)
        .await?;

    match outcome {
        HealingOutcome::Healed { code, attempts, .. } => {
            let plan = Plan {
                id: plan_id.to_owned(),
                prompt: prompt.to_owned(),
                module_source: code,
            };
            // Replace the raw staged candidate with the annotated plan.
            write_file(&cfg.plan_file, &plan.render())?;
            Ok(PlanOutcome::Validated { plan, attempts })
        }
        HealingOutcome::Exhausted { attempts, last_failure, .. } => Ok(PlanOutcome::Rejected {
            reason: last_failure.short_summary(),
            attempts,
        }),
    }
}

/// `apply` phase: install the already-validated module into the sandbox and run
/// the activation engine once. Rolls back (removes the module) on failure.
pub async fn apply_plan<B>(
    cfg: &AppConfig,
    plan: &Plan,
    builder: B,
) -> Result<ApplyOutcome, PlanError>
where
    B: SystemBuilder,
{
    let module_path = cfg.module_path(&plan.id);
    let engine = ExecutionEngine::with_builder(module_path.clone(), builder)
        .max_attempts(1)
        .restore_on_failure(true);

    match engine.run_once(plan.render()).await? {
        HealingOutcome::Healed { .. } => Ok(ApplyOutcome::Activated { module_path }),
        HealingOutcome::Exhausted { last_failure, .. } => Ok(ApplyOutcome::Failed {
            reason: last_failure.short_summary(),
        }),
    }
}

/// Resolve a plan from a CLI `--plan` argument that is either a path to a plan
/// file or a plan id (matched against the default plan file).
pub fn load_plan(cfg: &AppConfig, arg: &str) -> Result<Plan, PlanError> {
    let path = Path::new(arg);
    if path.is_file() {
        let content = read_file(path)?;
        return Plan::parse(&content);
    }

    let content = std::fs::read_to_string(&cfg.plan_file)
        .map_err(|_| PlanError::NotFound(arg.to_owned()))?;
    let plan = Plan::parse(&content)?;
    if plan.id != arg {
        return Err(PlanError::IdMismatch {
            requested: arg.to_owned(),
            found: plan.id,
        });
    }
    Ok(plan)
}

fn read_file(path: &Path) -> Result<String, PlanError> {
    std::fs::read_to_string(path).map_err(|source| PlanError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn write_file(path: &Path, content: &str) -> Result<(), PlanError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|source| PlanError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
    }
    std::fs::write(path, content).map_err(|source| PlanError::Io {
        path: path.to_path_buf(),
        source,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{AstOnlyBuilder, BuildOutput, HealingContext};
    use std::sync::Mutex;

    // Unique temp dir per test, cleaned up on drop.
    struct TempDir {
        path: PathBuf,
    }
    impl TempDir {
        fn new(tag: &str) -> Self {
            let mut path = std::env::temp_dir();
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            path.push(format!("nix-agent-plan-{tag}-{nanos}"));
            std::fs::create_dir_all(&path).unwrap();
            TempDir { path }
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.path).ok();
        }
    }

    fn cfg_in(dir: &Path) -> AppConfig {
        AppConfig {
            config_dir: dir.to_path_buf(),
            plan_file: dir.join(".nix-agent-plan.nix"),
            max_attempts: 2,
            ..AppConfig::default()
        }
    }

    const VALID_MODULE: &str = "{ config, pkgs, ... }:\n{\n  programs.tmux.enable = true;\n}\n";

    // A healer that always replays the same candidate.
    struct FixedHealer {
        reply: String,
        calls: Mutex<usize>,
    }
    impl CodeHealer for FixedHealer {
        async fn generate(&mut self, _ctx: &HealingContext) -> anyhow::Result<String> {
            *self.calls.lock().unwrap() += 1;
            Ok(self.reply.clone())
        }
    }

    // A builder with a fixed outcome (for apply).
    struct FixedBuilder {
        output: BuildOutput,
    }
    impl SystemBuilder for FixedBuilder {
        async fn build(&self, _staging_path: &Path) -> Result<BuildOutput, EngineError> {
            Ok(self.output.clone())
        }
    }

    fn ok_build() -> BuildOutput {
        BuildOutput { success: true, exit_code: Some(0), stdout: String::new(), stderr: String::new() }
    }
    fn fail_build() -> BuildOutput {
        BuildOutput {
            success: false,
            exit_code: Some(1),
            stdout: String::new(),
            stderr: "error: undefined variable 'pkgs' at /x.nix:1:1\n".to_owned(),
        }
    }

    // ── id + date ─────────────────────────────────────────────────────────────

    #[test]
    fn civil_date_conversion() {
        assert_eq!(ymd_from_unix(0), (1970, 1, 1));
        assert_eq!(ymd_from_unix(86_400), (1970, 1, 2));
        // Leap day, 2000-02-29.
        assert_eq!(ymd_from_unix(951_782_400), (2000, 2, 29));
    }

    #[test]
    fn plan_id_combines_date_and_slug() {
        let id = make_plan_id("add tmux with custom keybindings", 0);
        assert!(id.starts_with("1970-01-01-tmux"), "got {id}");
        assert!(!id.contains("add"));
        assert!(!id.contains("with"));
    }

    #[test]
    fn slug_skips_stopwords() {
        assert_eq!(derive_slug("add tmux"), "tmux");
        assert_eq!(derive_slug("install firefox and enable bluetooth"), "firefox-bluetooth");
        assert_eq!(derive_slug("enable the"), "");
    }

    // ── render / parse ──────────────────────────────────────────────────────

    #[test]
    fn render_parse_round_trip() {
        let plan = Plan {
            id: "2026-06-29-tmux".to_owned(),
            prompt: "add tmux".to_owned(),
            module_source: VALID_MODULE.to_owned(),
        };
        let rendered = plan.render();
        assert!(rendered.starts_with(HEADER_MARKER));
        assert!(rendered.contains("# plan-id: 2026-06-29-tmux"));
        let parsed = Plan::parse(&rendered).unwrap();
        assert_eq!(parsed, plan);
    }

    #[test]
    fn parse_rejects_missing_id() {
        let err = Plan::parse("{ }\n");
        assert!(matches!(err, Err(PlanError::Malformed(_))));
    }

    // ── load_plan ───────────────────────────────────────────────────────────

    #[test]
    fn load_plan_by_path_and_by_id() {
        let dir = TempDir::new("load");
        let cfg = cfg_in(&dir.path);
        let plan = Plan {
            id: "2026-06-29-tmux".to_owned(),
            prompt: "add tmux".to_owned(),
            module_source: VALID_MODULE.to_owned(),
        };
        write_file(&cfg.plan_file, &plan.render()).unwrap();

        // By path.
        let by_path = load_plan(&cfg, cfg.plan_file.to_str().unwrap()).unwrap();
        assert_eq!(by_path.id, "2026-06-29-tmux");
        // By id (matches the default plan file).
        let by_id = load_plan(&cfg, "2026-06-29-tmux").unwrap();
        assert_eq!(by_id, plan);
        // Mismatched id.
        let err = load_plan(&cfg, "2099-01-01-nope");
        assert!(matches!(err, Err(PlanError::IdMismatch { .. })));
    }

    // ── create_plan ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn create_plan_validates_and_writes_plan_file() {
        let dir = TempDir::new("create-ok");
        let cfg = cfg_in(&dir.path);
        let mut healer = FixedHealer { reply: String::new(), calls: Mutex::new(0) };

        let outcome = create_plan(
            &cfg,
            "add tmux",
            "2026-06-29-tmux",
            VALID_MODULE.to_owned(),
            &mut healer,
            AstOnlyBuilder,
            |_| {},
        )
        .await
        .unwrap();

        match outcome {
            PlanOutcome::Validated { plan, attempts } => {
                assert_eq!(attempts, 1);
                assert_eq!(plan.id, "2026-06-29-tmux");
            }
            other => panic!("expected Validated, got {other:?}"),
        }
        // Healer never invoked (initial module was valid).
        assert_eq!(*healer.calls.lock().unwrap(), 0);
        // Plan file on disk is the annotated, re-parseable plan.
        let written = std::fs::read_to_string(&cfg.plan_file).unwrap();
        assert!(written.starts_with(HEADER_MARKER));
        assert_eq!(Plan::parse(&written).unwrap().id, "2026-06-29-tmux");
    }

    #[tokio::test]
    async fn create_plan_rejects_unrepairable_module() {
        let dir = TempDir::new("create-bad");
        let cfg = cfg_in(&dir.path); // max_attempts = 2
        // Healer keeps returning broken Nix → AST gate never passes.
        let mut healer = FixedHealer { reply: "{ foo = ;".to_owned(), calls: Mutex::new(0) };

        let outcome = create_plan(
            &cfg,
            "do something impossible",
            "2026-06-29-impossible",
            "{ foo = ;".to_owned(),
            &mut healer,
            AstOnlyBuilder,
            |_| {},
        )
        .await
        .unwrap();

        match outcome {
            PlanOutcome::Rejected { attempts, reason } => {
                assert_eq!(attempts, 2);
                assert!(reason.contains("syntax"), "got {reason}");
            }
            other => panic!("expected Rejected, got {other:?}"),
        }
        // No plan file left behind.
        assert!(!cfg.plan_file.exists());
    }

    // ── apply_plan ──────────────────────────────────────────────────────────

    #[tokio::test]
    async fn apply_plan_installs_and_activates() {
        let dir = TempDir::new("apply-ok");
        let cfg = cfg_in(&dir.path);
        let plan = Plan {
            id: "2026-06-29-tmux".to_owned(),
            prompt: "add tmux".to_owned(),
            module_source: VALID_MODULE.to_owned(),
        };

        let outcome = apply_plan(&cfg, &plan, FixedBuilder { output: ok_build() })
            .await
            .unwrap();

        match outcome {
            ApplyOutcome::Activated { module_path } => {
                assert_eq!(module_path, cfg.module_path("2026-06-29-tmux"));
                assert!(module_path.exists());
                // Installed module carries the provenance header.
                let body = std::fs::read_to_string(&module_path).unwrap();
                assert!(body.contains("# plan-id: 2026-06-29-tmux"));
            }
            other => panic!("expected Activated, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn apply_plan_rolls_back_on_activation_failure() {
        let dir = TempDir::new("apply-fail");
        let cfg = cfg_in(&dir.path);
        let plan = Plan {
            id: "2026-06-29-tmux".to_owned(),
            prompt: "add tmux".to_owned(),
            module_source: VALID_MODULE.to_owned(),
        };

        let outcome = apply_plan(&cfg, &plan, FixedBuilder { output: fail_build() })
            .await
            .unwrap();

        assert!(matches!(outcome, ApplyOutcome::Failed { .. }));
        // Rollback removed the installed module.
        assert!(!cfg.module_path("2026-06-29-tmux").exists());
    }
}
