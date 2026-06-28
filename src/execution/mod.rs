//! Execution Layer + Self-Healing debug loop.
//!
//! Responsibilities:
//!   1. Drive `nixos-rebuild` (default: the non-destructive `dry-build`) through
//!      `tokio::process`, capturing `stdout`/`stderr` asynchronously with a timeout.
//!   2. Parse NixOS/`nix` diagnostics out of `stderr` into a structured
//!      [`NixBuildError`] (file, line, column, kind, message).
//!   3. Coordinate the self-healing loop: validate AST → build → on failure,
//!      correlate the error with [`crate::ast::NixError`] and hand a structured
//!      [`HealingContext`] back to a [`CodeHealer`] for re-generation (max 3 tries).
//!
//! The build step ([`SystemBuilder`]) and the LLM re-generation step
//! ([`CodeHealer`]) are traits so the loop is fully testable without a live
//! NixOS host, and so the future `rag`/inference module can plug in cleanly.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::time::timeout;

use crate::ast::{NixError, NixFile, ParseDiagnostic};

/// Default cap on self-healing attempts before giving up.
pub const DEFAULT_MAX_ATTEMPTS: usize = 3;
/// Default wall-clock budget for a single `nixos-rebuild` invocation.
pub const DEFAULT_BUILD_TIMEOUT: Duration = Duration::from_secs(900);

// ── Infrastructure errors ─────────────────────────────────────────────────────
// These are failures of the *tooling itself* (couldn't spawn, timed out, healer
// crashed) — distinct from a build that ran fine but reported Nix errors, which
// is the normal, expected input to the healing loop.

#[derive(Debug)]
pub enum EngineError {
    /// Failed to spawn the rebuild process (binary missing, permissions, …).
    Spawn { program: String, source: std::io::Error },
    /// Could not read/write the staged configuration file.
    Io { path: PathBuf, source: std::io::Error },
    /// The build exceeded its time budget and was killed.
    Timeout { secs: u64 },
    /// The injected code generator returned an error.
    Healer(anyhow::Error),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Spawn { program, source } => {
                write!(f, "failed to spawn '{}': {}", program, source)
            }
            Self::Io { path, source } => {
                write!(f, "I/O error on {}: {}", path.display(), source)
            }
            Self::Timeout { secs } => write!(f, "build timed out after {}s", secs),
            Self::Healer(e) => write!(f, "code generator failed: {}", e),
        }
    }
}

impl std::error::Error for EngineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Spawn { source, .. } | Self::Io { source, .. } => Some(source),
            Self::Healer(e) => Some(e.as_ref()),
            Self::Timeout { .. } => None,
        }
    }
}

// ── Build mode ────────────────────────────────────────────────────────────────

/// Which `nixos-rebuild` subcommand to run. Ordered from safest to most invasive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum BuildMode {
    /// Build derivations only; never activates, never needs root. Default.
    DryBuild,
    /// Build, then print what *would* change on activation. No root by default.
    DryActivate,
    /// Build and activate until next reboot. Requires root.
    Test,
}

impl BuildMode {
    /// The `nixos-rebuild` subcommand string.
    pub fn subcommand(self) -> &'static str {
        match self {
            Self::DryBuild => "dry-build",
            Self::DryActivate => "dry-activate",
            Self::Test => "test",
        }
    }

    /// Whether this mode mutates the running system and therefore needs root.
    pub fn requires_root(self) -> bool {
        matches!(self, Self::Test)
    }
}

// ── Raw build output ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize)]
pub struct BuildOutput {
    pub success: bool,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

// ── Parsed Nix diagnostic ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct SourceLocation {
    pub file: PathBuf,
    pub line: u32,
    pub column: u32,
}

/// Coarse classification of a Nix build failure, used to decide how to correlate
/// with [`NixError`] and how to steer the re-prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum NixBuildErrorKind {
    Syntax,
    UndefinedVariable,
    MissingAttribute,
    TypeError,
    Assertion,
    InfiniteRecursion,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct NixBuildError {
    pub kind: NixBuildErrorKind,
    /// The human-readable message (without the leading `error:`).
    pub message: String,
    /// `file:line:column` if the diagnostic carried one.
    pub location: Option<SourceLocation>,
    /// Identifier the diagnostic referred to (variable/attribute name), if any.
    pub symbol: Option<String>,
    /// The raw stderr slice, preserved verbatim for the session trace.
    pub raw: String,
}

// ── Healing data flow ─────────────────────────────────────────────────────────

/// Why an attempt failed. Either the candidate didn't parse (we own precise byte
/// offsets via `rnix`), or it parsed but `nixos-rebuild` rejected it.
#[derive(Debug, Clone, serde::Serialize)]
pub enum HealingFailure {
    /// AST stage caught it before we ever invoked the builder.
    Ast { diagnostics: Vec<ParseDiagnostic> },
    /// Builder ran and reported a Nix error.
    Build {
        error: NixBuildError,
        /// Bridge back to the `ast` error domain when we can establish one.
        correlated: Option<CorrelatedNixError>,
    },
}

impl HealingFailure {
    /// A compact, single-line technical reason suitable for live UI status —
    /// never a wall of stderr.
    pub fn short_summary(&self) -> String {
        match self {
            Self::Ast { diagnostics } => match diagnostics.first() {
                Some(d) => format!("syntax error: {}", d.message),
                None => "syntax error".to_owned(),
            },
            Self::Build { error, .. } => match &error.symbol {
                Some(sym) => format!("{:?}: {}", error.kind, sym),
                None => format!("{:?}", error.kind),
            },
        }
    }
}

/// Serializable mirror of the [`NixError`] variants relevant to healing. Lets the
/// JSON session trace record the AST-domain interpretation of a build failure
/// without forcing `NixError` itself to be `Serialize`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub enum CorrelatedNixError {
    Parse { diagnostics: Vec<ParseDiagnostic> },
    AttrNotFound { attr_path: String },
    TypeError { attr_path: String, expected: String },
}

/// Everything the next prompt needs: which attempt, the code that failed, and why.
#[derive(Debug, Clone, serde::Serialize)]
pub struct HealingContext {
    pub attempt: usize,
    pub max_attempts: usize,
    pub previous_code: String,
    pub failure: HealingFailure,
}

impl HealingContext {
    /// Render a structured, deterministic re-prompt for the code generator.
    ///
    /// For AST failures we translate each byte offset into a `line:column`
    /// (using `previous_code`) so the model gets both representations. For build
    /// failures we surface the file/line/column reported by Nix.
    pub fn reprompt(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "Attempt {}/{} failed. Fix the Nix configuration below.\n\n",
            self.attempt, self.max_attempts
        ));

        match &self.failure {
            HealingFailure::Ast { diagnostics } => {
                s.push_str("=== AST PARSE ERRORS (file never reached nixos-rebuild) ===\n");
                for d in diagnostics {
                    let (line, col) = byte_to_line_col(&self.previous_code, d.byte_start);
                    s.push_str(&format!(
                        "- {} (bytes {}..{}, line {}, col {})\n",
                        d.message, d.byte_start, d.byte_end, line, col
                    ));
                }
            }
            HealingFailure::Build { error, correlated } => {
                s.push_str("=== NIXOS-REBUILD ERROR ===\n");
                s.push_str(&format!("kind: {:?}\n", error.kind));
                s.push_str(&format!("message: {}\n", error.message));
                if let Some(sym) = &error.symbol {
                    s.push_str(&format!("symbol: {}\n", sym));
                }
                if let Some(loc) = &error.location {
                    s.push_str(&format!(
                        "location: {}:{}:{}\n",
                        loc.file.display(),
                        loc.line,
                        loc.column
                    ));
                }
                if let Some(c) = correlated {
                    s.push_str(&format!("correlated_ast_error: {:?}\n", c));
                }
            }
        }

        s.push_str("\n=== PREVIOUS CODE ===\n");
        s.push_str(&self.previous_code);
        s.push('\n');
        s
    }
}

/// Final result of the self-healing loop.
#[derive(Debug, serde::Serialize)]
pub enum HealingOutcome {
    /// A candidate built successfully.
    Healed {
        code: String,
        attempts: usize,
        build: BuildOutput,
    },
    /// Exhausted all attempts without a green build.
    Exhausted {
        attempts: usize,
        last_failure: HealingFailure,
        last_code: String,
    },
}

/// Live progress signal emitted by [`ExecutionEngine::self_healing_loop_with`].
/// Carries only owned, compact data so a front-end can render status lines
/// without ever touching raw build logs.
#[derive(Debug, Clone)]
pub enum HealEvent {
    /// An attempt's validate→build cycle is starting.
    Attempt { attempt: usize, max_attempts: usize },
    /// The attempt failed; `summary` is a one-line technical reason.
    Failed { attempt: usize, summary: String },
    /// About to ask the healer to regenerate before `next_attempt`.
    Regenerating { next_attempt: usize },
}

// ── Pluggable seams ───────────────────────────────────────────────────────────

/// Performs an actual system build of the staged configuration. The real
/// implementation is [`NixosRebuild`]; tests provide scripted mocks.
#[allow(async_fn_in_trait)]
pub trait SystemBuilder {
    /// Build the configuration currently staged at `staging_path`.
    async fn build(&self, staging_path: &Path) -> Result<BuildOutput, EngineError>;
}

/// Produces a fresh candidate from a structured failure context. The real
/// implementation will wrap the local LLM + RAG; the loop only sees this trait.
#[allow(async_fn_in_trait)]
pub trait CodeHealer {
    async fn generate(&mut self, ctx: &HealingContext) -> anyhow::Result<String>;
}

// ── Real builder: nixos-rebuild over tokio::process ───────────────────────────

pub struct NixosRebuild {
    pub mode: BuildMode,
    /// Wrap the invocation in `sudo`. Auto-enabled for root-requiring modes.
    pub use_sudo: bool,
    /// The rebuild binary; overridable for tests/sandboxes.
    pub bin: String,
    pub timeout: Duration,
    /// When true, build the system's default configuration (`/etc/nixos`)
    /// instead of pointing `-I nixos-config` at the staged file. `apply` uses
    /// this so the freshly-installed sandbox module is picked up via the user's
    /// existing `imports` of `modules/ai-generated`.
    pub use_system_config: bool,
}

impl NixosRebuild {
    pub fn new(mode: BuildMode) -> Self {
        Self {
            mode,
            use_sudo: mode.requires_root(),
            bin: "nixos-rebuild".to_owned(),
            timeout: DEFAULT_BUILD_TIMEOUT,
            use_system_config: false,
        }
    }

    /// Build the `(program, args)` pair without spawning. Pure → unit-testable.
    ///
    /// By default Nix is pointed at *our* staged file via `-I nixos-config=<path>`
    /// so the candidate is evaluated without touching `/etc/nixos`. When
    /// `use_system_config` is set, the flag is omitted and the system default
    /// configuration is built. `--show-trace` yields richer diagnostics.
    pub fn argv(&self, staging_path: &Path) -> (String, Vec<String>) {
        let mut args: Vec<String> = Vec::new();
        let program = if self.use_sudo {
            args.push(self.bin.clone());
            "sudo".to_owned()
        } else {
            self.bin.clone()
        };
        args.push(self.mode.subcommand().to_owned());
        if !self.use_system_config {
            args.push("-I".to_owned());
            args.push(format!("nixos-config={}", staging_path.display()));
        }
        args.push("--show-trace".to_owned());
        (program, args)
    }
}

impl SystemBuilder for NixosRebuild {
    async fn build(&self, staging_path: &Path) -> Result<BuildOutput, EngineError> {
        let (program, args) = self.argv(staging_path);

        let mut child = Command::new(&program)
            .args(&args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|source| EngineError::Spawn {
                program: program.clone(),
                source,
            })?;

        // Take the pipe handles so we can read them concurrently while the child
        // runs, then reap with `wait()`. Reading both prevents pipe-buffer
        // deadlock on large `--show-trace` output.
        let mut stdout = child.stdout.take().expect("stdout piped");
        let mut stderr = child.stderr.take().expect("stderr piped");

        let mut out_buf = Vec::new();
        let mut err_buf = Vec::new();

        let read_both = async {
            tokio::try_join!(
                stdout.read_to_end(&mut out_buf),
                stderr.read_to_end(&mut err_buf),
            )
        };

        match timeout(self.timeout, read_both).await {
            Ok(io_result) => {
                io_result.map_err(|source| EngineError::Io {
                    path: staging_path.to_path_buf(),
                    source,
                })?;
            }
            Err(_elapsed) => {
                let _ = child.start_kill();
                let _ = child.wait().await;
                return Err(EngineError::Timeout {
                    secs: self.timeout.as_secs(),
                });
            }
        }

        let status = child.wait().await.map_err(|source| EngineError::Io {
            path: staging_path.to_path_buf(),
            source,
        })?;

        Ok(BuildOutput {
            success: status.success(),
            exit_code: status.code(),
            stdout: String::from_utf8_lossy(&out_buf).into_owned(),
            stderr: String::from_utf8_lossy(&err_buf).into_owned(),
        })
    }
}

/// A [`SystemBuilder`] that performs no system build. It relies entirely on the
/// engine's AST gate (which runs first) and reports success, so the read-only
/// `plan` phase can structurally validate a generated module — and self-heal
/// malformed output — without invoking `nixos-rebuild` or requiring privileges.
#[derive(Debug, Clone, Default)]
pub struct AstOnlyBuilder;

impl SystemBuilder for AstOnlyBuilder {
    async fn build(&self, _staging_path: &Path) -> Result<BuildOutput, EngineError> {
        Ok(BuildOutput {
            success: true,
            exit_code: Some(0),
            stdout: "AST-gate structural validation only; no system build performed".to_owned(),
            stderr: String::new(),
        })
    }
}

// ── The engine ────────────────────────────────────────────────────────────────

pub struct ExecutionEngine<B: SystemBuilder> {
    /// Isolated path where each candidate is staged for the builder. This is
    /// always a sandbox/temp file — never the live system configuration.
    staging_path: PathBuf,
    builder: B,
    max_attempts: usize,
    /// On total failure, restore the pre-loop contents of `staging_path`.
    restore_on_failure: bool,
}

impl ExecutionEngine<NixosRebuild> {
    /// Convenience constructor using the real `nixos-rebuild` builder.
    pub fn nixos(staging_path: impl Into<PathBuf>, mode: BuildMode) -> Self {
        Self::with_builder(staging_path, NixosRebuild::new(mode))
    }
}

impl<B: SystemBuilder> ExecutionEngine<B> {
    pub fn with_builder(staging_path: impl Into<PathBuf>, builder: B) -> Self {
        Self {
            staging_path: staging_path.into(),
            builder,
            max_attempts: DEFAULT_MAX_ATTEMPTS,
            restore_on_failure: true,
        }
    }

    pub fn max_attempts(mut self, n: usize) -> Self {
        self.max_attempts = n.max(1);
        self
    }

    pub fn restore_on_failure(mut self, yes: bool) -> Self {
        self.restore_on_failure = yes;
        self
    }

    /// Coordinate the bounded self-healing debug loop.
    ///
    /// Per attempt:
    ///   1. Validate the candidate's AST locally (cheap; no subprocess). A parse
    ///      error short-circuits to an [`HealingFailure::Ast`] carrying byte offsets.
    ///   2. Otherwise stage the file and run the builder.
    ///   3. Green build → [`HealingOutcome::Healed`].
    ///   4. Red build → parse stderr, correlate with [`NixError`], and (if attempts
    ///      remain) ask the healer for a new candidate.
    ///
    /// Returns `Err(EngineError)` only for *infrastructure* failures (spawn,
    /// timeout, healer crash). A merely-failing build is an `Ok` outcome.
    pub async fn self_healing_loop<H>(
        &self,
        initial_code: String,
        healer: &mut H,
    ) -> Result<HealingOutcome, EngineError>
    where
        H: CodeHealer,
    {
        self.self_healing_loop_with(initial_code, healer, |_| {})
            .await
    }

    /// Same as [`Self::self_healing_loop`], but emits [`HealEvent`]s to `on_event`
    /// so a UI can show compact, live progress (e.g. `[Attempt 2] …`) instead of
    /// dumping raw `nixos-rebuild` stderr.
    pub async fn self_healing_loop_with<H, F>(
        &self,
        initial_code: String,
        healer: &mut H,
        mut on_event: F,
    ) -> Result<HealingOutcome, EngineError>
    where
        H: CodeHealer,
        F: FnMut(HealEvent),
    {
        let backup = self.read_backup()?;
        let mut candidate = initial_code;
        let mut last_failure: Option<HealingFailure> = None;

        for attempt in 1..=self.max_attempts {
            on_event(HealEvent::Attempt {
                attempt,
                max_attempts: self.max_attempts,
            });

            match self.try_once(&candidate).await? {
                AttemptResult::Success(build) => {
                    return Ok(HealingOutcome::Healed {
                        code: candidate,
                        attempts: attempt,
                        build,
                    });
                }
                AttemptResult::Failed(failure) => {
                    on_event(HealEvent::Failed {
                        attempt,
                        summary: failure.short_summary(),
                    });
                    last_failure = Some(failure.clone());

                    if attempt < self.max_attempts {
                        on_event(HealEvent::Regenerating {
                            next_attempt: attempt + 1,
                        });
                        let ctx = HealingContext {
                            attempt,
                            max_attempts: self.max_attempts,
                            previous_code: candidate.clone(),
                            failure,
                        };
                        candidate = healer
                            .generate(&ctx)
                            .await
                            .map_err(EngineError::Healer)?;
                    }
                }
            }
        }

        if self.restore_on_failure {
            self.restore_backup(backup)?;
        }

        Ok(HealingOutcome::Exhausted {
            attempts: self.max_attempts,
            last_failure: last_failure.expect("loop runs at least once"),
            last_code: candidate,
        })
    }

    /// Run a single validate → stage → build cycle with no self-healing.
    ///
    /// Used by `apply`, where the module was already validated during `plan`:
    /// the candidate is staged at `staging_path`, built once, and — on failure —
    /// rolled back. Returns [`HealingOutcome::Healed`] (attempts = 1) on success
    /// or [`HealingOutcome::Exhausted`] on failure.
    pub async fn run_once(&self, candidate: String) -> Result<HealingOutcome, EngineError> {
        let backup = self.read_backup()?;
        match self.try_once(&candidate).await? {
            AttemptResult::Success(build) => Ok(HealingOutcome::Healed {
                code: candidate,
                attempts: 1,
                build,
            }),
            AttemptResult::Failed(failure) => {
                if self.restore_on_failure {
                    self.restore_backup(backup)?;
                }
                Ok(HealingOutcome::Exhausted {
                    attempts: 1,
                    last_failure: failure,
                    last_code: candidate,
                })
            }
        }
    }

    /// One validate → stage → build cycle.
    async fn try_once(&self, candidate: &str) -> Result<AttemptResult, EngineError> {
        // 1. Local AST gate — never feed unparseable code to the builder.
        match NixFile::from_source(&self.staging_path, candidate.to_owned()) {
            Ok(_) => {}
            Err(NixError::Parse { diagnostics, .. }) => {
                return Ok(AttemptResult::Failed(HealingFailure::Ast { diagnostics }));
            }
            // `from_source` only yields `Parse`; any other variant is defensive.
            Err(_) => {
                return Ok(AttemptResult::Failed(HealingFailure::Ast {
                    diagnostics: Vec::new(),
                }));
            }
        }

        // 2. Stage and build.
        self.stage_code(candidate)?;
        let output = self.builder.build(&self.staging_path).await?;
        if output.success {
            return Ok(AttemptResult::Success(output));
        }

        // 3. Parse the diagnostic and bridge it to the ast error domain.
        let error = parse_build_stderr(&output.stderr).unwrap_or_else(|| NixBuildError {
            kind: NixBuildErrorKind::Other,
            message: "nixos-rebuild failed without a parseable error".to_owned(),
            location: None,
            symbol: None,
            raw: output.stderr.clone(),
        });
        let correlated = correlate_with_ast(&error, candidate);

        Ok(AttemptResult::Failed(HealingFailure::Build { error, correlated }))
    }

    fn stage_code(&self, code: &str) -> Result<(), EngineError> {
        if let Some(parent) = self.staging_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|source| EngineError::Io {
                    path: parent.to_path_buf(),
                    source,
                })?;
            }
        }
        std::fs::write(&self.staging_path, code).map_err(|source| EngineError::Io {
            path: self.staging_path.clone(),
            source,
        })
    }

    /// Snapshot existing contents (if any) so we can roll back on total failure.
    fn read_backup(&self) -> Result<Option<String>, EngineError> {
        match std::fs::read_to_string(&self.staging_path) {
            Ok(s) => Ok(Some(s)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(source) => Err(EngineError::Io {
                path: self.staging_path.clone(),
                source,
            }),
        }
    }

    fn restore_backup(&self, backup: Option<String>) -> Result<(), EngineError> {
        match backup {
            Some(original) => self.stage_code(&original),
            // Nothing existed before us — remove what we staged.
            None => match std::fs::remove_file(&self.staging_path) {
                Ok(()) => Ok(()),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
                Err(source) => Err(EngineError::Io {
                    path: self.staging_path.clone(),
                    source,
                }),
            },
        }
    }
}

enum AttemptResult {
    Success(BuildOutput),
    Failed(HealingFailure),
}

// ── stderr parser ─────────────────────────────────────────────────────────────

/// Parse a `nix`/`nixos-rebuild` stderr blob into a structured [`NixBuildError`].
///
/// Handles both the legacy single-line form
/// `error: <msg> at /path/file.nix:LINE:COL`
/// and the modern multi-line form (Nix ≥ 2.4)
/// ```text
/// error: <msg>
///        at /path/file.nix:LINE:COL:
///             N| ...
/// ```
pub fn parse_build_stderr(stderr: &str) -> Option<NixBuildError> {
    // Locate the first real `error:` line (nixos-rebuild interleaves progress).
    let err_line_idx = stderr
        .lines()
        .position(|l| l.trim_start().starts_with("error:"))?;
    let lines: Vec<&str> = stderr.lines().collect();
    let err_line = lines[err_line_idx].trim_start();

    // Strip the `error:` prefix.
    let after = err_line.trim_start_matches("error:").trim();

    // The message may carry an inline `... at <loc>` (legacy form). Split it off.
    let (message, inline_loc) = match split_inline_location(after) {
        Some((msg, loc)) => (msg.to_owned(), Some(loc)),
        None => (after.to_owned(), None),
    };

    // Otherwise scan following lines for an `at <path>:LINE:COL[:]` marker.
    let location = inline_loc.or_else(|| {
        lines[err_line_idx + 1..]
            .iter()
            .take(8) // location appears within a few lines of the message
            .find_map(|l| {
                let t = l.trim();
                t.strip_prefix("at ").and_then(parse_location_token)
            })
    });

    let kind = classify(&message);
    let symbol = extract_quoted(&message);

    Some(NixBuildError {
        kind,
        message,
        location,
        symbol,
        raw: stderr.to_owned(),
    })
}

/// If `s` ends with `... at <path>:LINE:COL`, return `(message_without_loc, loc)`.
fn split_inline_location(s: &str) -> Option<(&str, SourceLocation)> {
    // Search from the right for " at " so message text containing "at" is safe.
    let mut search = s;
    while let Some(rel) = search.rfind(" at ") {
        let candidate = &search[rel + 4..];
        if let Some(loc) = parse_location_token(candidate.trim()) {
            return Some((s[..rel].trim_end(), loc));
        }
        // Not a location; keep looking further left.
        search = &search[..rel];
    }
    None
}

/// Parse a bare `/path/to/file.nix:LINE:COL` (optional trailing `:`) token.
fn parse_location_token(token: &str) -> Option<SourceLocation> {
    let token = token.trim().trim_end_matches(':');
    // rsplit twice: COL, LINE, then the remainder is the path (may hold ':').
    let mut it = token.rsplitn(3, ':');
    let col = it.next()?.parse::<u32>().ok()?;
    let line = it.next()?.parse::<u32>().ok()?;
    let path = it.next()?;
    if path.is_empty() {
        return None;
    }
    Some(SourceLocation {
        file: PathBuf::from(path),
        line,
        column: col,
    })
}

fn classify(message: &str) -> NixBuildErrorKind {
    let m = message.to_ascii_lowercase();
    if m.contains("syntax error") {
        NixBuildErrorKind::Syntax
    } else if m.contains("undefined variable") {
        NixBuildErrorKind::UndefinedVariable
    } else if m.contains("attribute") && m.contains("missing") {
        NixBuildErrorKind::MissingAttribute
    } else if m.contains("infinite recursion") {
        NixBuildErrorKind::InfiniteRecursion
    } else if m.contains("assertion") && m.contains("failed") {
        NixBuildErrorKind::Assertion
    } else if m.contains("cannot coerce")
        || m.contains("is not a")
        || (m.contains("expected") && m.contains("but found"))
    {
        NixBuildErrorKind::TypeError
    } else {
        NixBuildErrorKind::Other
    }
}

/// Extract the first single-quoted identifier from a message, e.g.
/// `undefined variable 'pkgs'` → `pkgs`.
fn extract_quoted(s: &str) -> Option<String> {
    let start = s.find('\'')?;
    let rest = &s[start + 1..];
    let end = rest.find('\'')?;
    Some(rest[..end].to_owned())
}

/// Bridge a parsed build error back to the `ast` error domain so the trace and
/// the re-prompt can speak in `NixError` terms with byte offsets where possible.
fn correlate_with_ast(error: &NixBuildError, code: &str) -> Option<CorrelatedNixError> {
    match error.kind {
        // Re-run the local parser to recover precise byte offsets `nix` doesn't give.
        NixBuildErrorKind::Syntax => {
            match NixFile::from_source("staged.nix", code.to_owned()) {
                Err(NixError::Parse { diagnostics, .. }) => {
                    Some(CorrelatedNixError::Parse { diagnostics })
                }
                _ => None,
            }
        }
        NixBuildErrorKind::MissingAttribute => error
            .symbol
            .as_ref()
            .map(|attr| CorrelatedNixError::AttrNotFound {
                attr_path: attr.clone(),
            }),
        NixBuildErrorKind::TypeError => Some(CorrelatedNixError::TypeError {
            attr_path: error.symbol.clone().unwrap_or_default(),
            expected: "unknown".to_owned(),
        }),
        _ => None,
    }
}

// ── small text util ───────────────────────────────────────────────────────────

/// Convert a 0-based byte offset into 1-based `(line, column)`.
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
    use std::collections::VecDeque;
    use std::sync::Mutex;

    // A temp config path unique to each test, cleaned up on drop.
    struct TempCfg {
        path: PathBuf,
    }
    impl TempCfg {
        fn new(tag: &str) -> Self {
            let mut path = std::env::temp_dir();
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            path.push(format!("nix-agent-{}-{}.nix", tag, nanos));
            TempCfg { path }
        }
    }
    impl Drop for TempCfg {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    // Builder that replays a scripted queue of outputs.
    struct ScriptedBuilder {
        queue: Mutex<VecDeque<BuildOutput>>,
        calls: Mutex<usize>,
    }
    impl ScriptedBuilder {
        fn new(outputs: Vec<BuildOutput>) -> Self {
            Self {
                queue: Mutex::new(outputs.into()),
                calls: Mutex::new(0),
            }
        }
    }
    impl SystemBuilder for ScriptedBuilder {
        async fn build(&self, _staging_path: &Path) -> Result<BuildOutput, EngineError> {
            *self.calls.lock().unwrap() += 1;
            Ok(self
                .queue
                .lock()
                .unwrap()
                .pop_front()
                .expect("scripted builder ran out of responses"))
        }
    }

    // Healer that replays scripted candidate strings.
    struct ScriptedHealer {
        queue: Mutex<VecDeque<String>>,
        seen: Mutex<Vec<HealingContext>>,
    }
    impl ScriptedHealer {
        fn new(candidates: Vec<&str>) -> Self {
            Self {
                queue: Mutex::new(candidates.into_iter().map(String::from).collect()),
                seen: Mutex::new(Vec::new()),
            }
        }
    }
    impl CodeHealer for ScriptedHealer {
        async fn generate(&mut self, ctx: &HealingContext) -> anyhow::Result<String> {
            self.seen.lock().unwrap().push(ctx.clone());
            Ok(self
                .queue
                .lock()
                .unwrap()
                .pop_front()
                .expect("scripted healer ran out of candidates"))
        }
    }

    fn ok_build() -> BuildOutput {
        BuildOutput {
            success: true,
            exit_code: Some(0),
            stdout: String::new(),
            stderr: String::new(),
        }
    }

    fn fail_build(stderr: &str) -> BuildOutput {
        BuildOutput {
            success: false,
            exit_code: Some(1),
            stdout: String::new(),
            stderr: stderr.to_owned(),
        }
    }

    const VALID: &str = "{ pkgs, ... }: { environment.systemPackages = [ pkgs.vim ]; }\n";

    // ── parser ────────────────────────────────────────────────────────────────

    #[test]
    fn parses_modern_multiline_undefined_variable() {
        let stderr = "\
building Nix...
error: undefined variable 'pkgs'

       at /etc/nixos/configuration.nix:10:7:

            9|   environment.systemPackages = [
           10|     pkgs.firefox
             |     ^
           11|   ];
";
        let e = parse_build_stderr(stderr).expect("parsed");
        assert_eq!(e.kind, NixBuildErrorKind::UndefinedVariable);
        assert_eq!(e.symbol.as_deref(), Some("pkgs"));
        let loc = e.location.expect("location");
        assert_eq!(loc.file, PathBuf::from("/etc/nixos/configuration.nix"));
        assert_eq!((loc.line, loc.column), (10, 7));
    }

    #[test]
    fn parses_legacy_inline_syntax_error() {
        let stderr =
            "error: syntax error, unexpected '}' at /etc/nixos/configuration.nix:34:1\n";
        let e = parse_build_stderr(stderr).expect("parsed");
        assert_eq!(e.kind, NixBuildErrorKind::Syntax);
        let loc = e.location.expect("location");
        assert_eq!((loc.line, loc.column), (34, 1));
        // The inline location must be stripped out of the message.
        assert!(!e.message.contains("/etc/nixos"));
        assert!(e.message.contains("syntax error"));
    }

    #[test]
    fn parses_missing_attribute() {
        let stderr = "\
error: attribute 'systemPackages' missing

       at /etc/nixos/configuration.nix:5:3:
";
        let e = parse_build_stderr(stderr).expect("parsed");
        assert_eq!(e.kind, NixBuildErrorKind::MissingAttribute);
        assert_eq!(e.symbol.as_deref(), Some("systemPackages"));
        assert_eq!(e.location.unwrap().line, 5);
    }

    #[test]
    fn returns_none_without_error_line() {
        assert!(parse_build_stderr("building...\nactivating...\n").is_none());
    }

    #[test]
    fn byte_to_line_col_basic() {
        let src = "abc\ndef\nghi";
        assert_eq!(byte_to_line_col(src, 0), (1, 1));
        assert_eq!(byte_to_line_col(src, 4), (2, 1)); // 'd'
        assert_eq!(byte_to_line_col(src, 6), (2, 3)); // 'f'
    }

    #[test]
    fn argv_dry_build_has_no_sudo_and_points_at_staged_file() {
        let nb = NixosRebuild::new(BuildMode::DryBuild);
        let (prog, args) = nb.argv(Path::new("/tmp/c.nix"));
        assert_eq!(prog, "nixos-rebuild");
        assert!(args.contains(&"dry-build".to_owned()));
        assert!(args.contains(&"nixos-config=/tmp/c.nix".to_owned()));
    }

    #[test]
    fn argv_test_mode_wraps_in_sudo() {
        let nb = NixosRebuild::new(BuildMode::Test);
        let (prog, args) = nb.argv(Path::new("/tmp/c.nix"));
        assert_eq!(prog, "sudo");
        assert_eq!(args[0], "nixos-rebuild");
        assert!(args.contains(&"test".to_owned()));
    }

    #[test]
    fn argv_system_config_omits_nixos_config_flag() {
        // `apply` builds the system default config (which imports the sandbox),
        // so no `-I nixos-config=` override is emitted.
        let mut nb = NixosRebuild::new(BuildMode::Test);
        nb.use_system_config = true;
        let (_prog, args) = nb.argv(Path::new("/tmp/ignored.nix"));
        assert!(args.contains(&"test".to_owned()));
        assert!(!args.iter().any(|a| a.starts_with("nixos-config=")));
        assert!(!args.contains(&"-I".to_owned()));
    }

    // ── plan-phase builders ──────────────────────────────────────────────────

    #[tokio::test]
    async fn ast_only_builder_validates_without_system_build() {
        let cfg = TempCfg::new("ast-only");
        let engine = ExecutionEngine::with_builder(cfg.path.clone(), AstOnlyBuilder);
        // Valid Nix passes the AST gate, the AST-only builder reports success.
        let outcome = engine.run_once(VALID.to_owned()).await.unwrap();
        assert!(matches!(outcome, HealingOutcome::Healed { attempts: 1, .. }));
        // The staged module is on disk at the isolated path.
        assert!(cfg.path.exists());
    }

    #[tokio::test]
    async fn ast_only_builder_rejects_broken_nix() {
        let cfg = TempCfg::new("ast-only-bad");
        let engine = ExecutionEngine::with_builder(cfg.path.clone(), AstOnlyBuilder)
            .restore_on_failure(true);
        let outcome = engine.run_once("{ foo = ;".to_owned()).await.unwrap();
        match outcome {
            HealingOutcome::Exhausted { last_failure, .. } => {
                assert!(matches!(last_failure, HealingFailure::Ast { .. }));
            }
            other => panic!("expected Exhausted, got {:?}", other),
        }
        // Rollback removed the broken staged file.
        assert!(!cfg.path.exists());
    }

    #[tokio::test]
    async fn run_once_rolls_back_on_build_failure() {
        let cfg = TempCfg::new("run-once-fail");
        let builder = ScriptedBuilder::new(vec![fail_build(
            "error: undefined variable 'pkgs' at /x.nix:1:1\n",
        )]);
        let engine = ExecutionEngine::with_builder(cfg.path.clone(), builder);
        let outcome = engine.run_once(VALID.to_owned()).await.unwrap();
        match outcome {
            HealingOutcome::Exhausted { attempts, last_failure, .. } => {
                assert_eq!(attempts, 1);
                assert!(matches!(last_failure, HealingFailure::Build { .. }));
            }
            other => panic!("expected Exhausted, got {:?}", other),
        }
        // The module we staged was rolled back.
        assert!(!cfg.path.exists());
    }

    // ── loop ────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn heals_on_second_attempt() {
        let cfg = TempCfg::new("heal2");
        let builder = ScriptedBuilder::new(vec![
            fail_build("error: undefined variable 'pkgs' at /x.nix:1:1\n"),
            ok_build(),
        ]);
        let engine = ExecutionEngine::with_builder(cfg.path.clone(), builder);
        let mut healer = ScriptedHealer::new(vec![VALID]);

        let outcome = engine
            .self_healing_loop(VALID.to_owned(), &mut healer)
            .await
            .expect("no infra error");

        match outcome {
            HealingOutcome::Healed { attempts, .. } => assert_eq!(attempts, 2),
            other => panic!("expected Healed, got {:?}", other),
        }
        // Healer was consulted exactly once (after the first failure).
        assert_eq!(healer.seen.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn exhausts_after_three_failures() {
        let cfg = TempCfg::new("exhaust");
        let builder = ScriptedBuilder::new(vec![
            fail_build("error: undefined variable 'a' at /x.nix:1:1\n"),
            fail_build("error: undefined variable 'b' at /x.nix:1:1\n"),
            fail_build("error: undefined variable 'c' at /x.nix:1:1\n"),
        ]);
        let engine = ExecutionEngine::with_builder(cfg.path.clone(), builder)
            .restore_on_failure(false);
        // Two regenerations between three attempts.
        let mut healer = ScriptedHealer::new(vec![VALID, VALID]);

        let outcome = engine
            .self_healing_loop(VALID.to_owned(), &mut healer)
            .await
            .expect("no infra error");

        match outcome {
            HealingOutcome::Exhausted { attempts, last_failure, .. } => {
                assert_eq!(attempts, 3);
                assert!(matches!(last_failure, HealingFailure::Build { .. }));
            }
            other => panic!("expected Exhausted, got {:?}", other),
        }
        assert_eq!(healer.seen.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn broken_ast_short_circuits_before_build() {
        let cfg = TempCfg::new("ast");
        // If the builder were ever invoked here, it would panic (empty queue).
        let builder = ScriptedBuilder::new(vec![ok_build()]);
        let engine = ExecutionEngine::with_builder(cfg.path.clone(), builder);
        // Initial code is syntactically broken; healer supplies a valid fix.
        let mut healer = ScriptedHealer::new(vec![VALID]);

        let outcome = engine
            .self_healing_loop("{ foo = ;".to_owned(), &mut healer)
            .await
            .expect("no infra error");

        // First failure must be the AST gate, carrying diagnostics with offsets.
        let first = &healer.seen.lock().unwrap()[0];
        match &first.failure {
            HealingFailure::Ast { diagnostics } => assert!(!diagnostics.is_empty()),
            other => panic!("expected Ast failure first, got {:?}", other),
        }
        // Builder only ran once — for the valid second candidate.
        assert!(matches!(outcome, HealingOutcome::Healed { attempts: 2, .. }));
        assert_eq!(*engine_builder_calls(&engine), 1);
    }

    // Helper to peek at the scripted builder's call count through the engine.
    fn engine_builder_calls(engine: &ExecutionEngine<ScriptedBuilder>) -> std::sync::MutexGuard<'_, usize> {
        engine_builder(engine).calls.lock().unwrap()
    }
    fn engine_builder(engine: &ExecutionEngine<ScriptedBuilder>) -> &ScriptedBuilder {
        &engine.builder
    }

    #[tokio::test]
    async fn reprompt_includes_location_and_offsets() {
        // Build-failure reprompt carries file:line:col.
        let ctx = HealingContext {
            attempt: 1,
            max_attempts: 3,
            previous_code: VALID.to_owned(),
            failure: HealingFailure::Build {
                error: parse_build_stderr(
                    "error: undefined variable 'pkgs' at /etc/nixos/configuration.nix:10:7\n",
                )
                .unwrap(),
                correlated: None,
            },
        };
        let p = ctx.reprompt();
        assert!(p.contains("configuration.nix:10:7"));
        assert!(p.contains("PREVIOUS CODE"));

        // AST-failure reprompt translates byte offsets to line/col.
        let diag = vec![ParseDiagnostic {
            message: "unexpected token".to_owned(),
            byte_start: 8,
            byte_end: 9,
        }];
        let ctx2 = HealingContext {
            attempt: 1,
            max_attempts: 3,
            previous_code: "{ foo = ;".to_owned(),
            failure: HealingFailure::Ast { diagnostics: diag },
        };
        let p2 = ctx2.reprompt();
        assert!(p2.contains("bytes 8..9"));
        assert!(p2.contains("line 1"));
    }
}
