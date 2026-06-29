//! `nix-agent` — a local, air-gapped Verified Patch Assistant for NixOS.
//!
//! The agent never edits the live system configuration. Work is split into two
//! explicit phases:
//!   * `ingest <PATH>` — populate the local RAG index from a NixOS options dump.
//!   * `plan "<PROMPT>"` — read-only: generate an isolated module, validate it
//!     through the AST gate, and write a ready-to-apply plan file. No activation.
//!   * `apply --plan <PATH_OR_ID>` — privileged: install the validated module
//!     into the sandbox and run the activation engine (`nixos-rebuild test`).

use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::{Parser, Subcommand};

use nix_agent::config::{AppConfig, HardwareTier};
use nix_agent::execution::{AstOnlyBuilder, NixosRebuild};
use nix_agent::plan::{self, ApplyOutcome, Plan, PlanOutcome};
use nix_agent::rag::{LocalLlmHealer, NixOptionIndex};

#[derive(Parser)]
#[command(
    name = "nix-agent",
    version,
    about = "Local, air-gapped Verified Patch Assistant for NixOS (AST + RAG + self-healing)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Populate the local RAG index from a nixosOptionsDoc JSON dump.
    Ingest {
        /// Path to the options JSON dump.
        path: PathBuf,
    },
    /// Read-only: generate and validate an isolated module, then write a plan.
    Plan {
        /// What you want the system to do, e.g. "add tmux with custom keybindings".
        prompt: String,
        /// Path to a specific GGUF model. If omitted, the agent auto-discovers
        /// local models (XDG + Ollama) and prompts when several are available.
        #[arg(long)]
        model: Option<PathBuf>,
    },
    /// Privileged: install a validated plan into the sandbox and activate it.
    Apply {
        /// Plan id (e.g. 2026-06-29-tmux) or path to a plan file.
        #[arg(long)]
        plan: String,
    },
    /// Manage local GGUF inference models (discovery + downloads).
    Models {
        #[command(subcommand)]
        action: ModelsCmd,
    },
}

#[derive(Subcommand)]
enum ModelsCmd {
    /// List locally available GGUF models, their size, and full path.
    List,
    /// Download a GGUF model (by alias or URL) into the local model directory.
    Pull {
        /// Alias (`default`, `7b`, `3b`, `1.5b`) or a direct https URL to a .gguf.
        #[arg(default_value = nix_agent::models::DEFAULT_PULL_ALIAS)]
        target: String,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    if let Err(err) = dispatch(cli).await {
        ui::failure(&format!("{err:#}"));
        std::process::exit(1);
    }
}

async fn dispatch(cli: Cli) -> anyhow::Result<()> {
    let cfg = AppConfig::load();
    cfg.ensure_storage().with_context(|| {
        format!(
            "failed to create storage directories under {}",
            cfg.model_cache_dir.display()
        )
    })?;

    match cli.command {
        Command::Ingest { path } => cmd_ingest(&cfg, &path),
        Command::Plan { prompt, model } => cmd_plan(&cfg, &prompt, model.as_deref()).await,
        Command::Apply { plan } => cmd_apply(&cfg, &plan).await,
        Command::Models { action } => match action {
            ModelsCmd::List => cmd_models_list(),
            ModelsCmd::Pull { target } => cmd_models_pull(&target),
        },
    }
}

/// `ingest` — read a JSON options dump and index it locally.
fn cmd_ingest(cfg: &AppConfig, path: &Path) -> anyhow::Result<()> {
    ui::banner();
    ui::step(1, 2, "Reading NixOS options dump...");
    let json = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;

    ui::step(2, 2, "Indexing into the local RAG database...");
    let mut index = NixOptionIndex::open(&cfg.rag_db_path)?;
    index.bootstrap()?;
    let count = index.ingest_json_dump(&json)?;

    ui::success(&format!(
        "Indexed {count} options → {}",
        cfg.rag_db_path.display()
    ));
    Ok(())
}

/// `plan` — generate and structurally validate an isolated module. Read-only:
/// it writes a plan file but never installs anything or runs an activation.
async fn cmd_plan(cfg: &AppConfig, prompt: &str, model: Option<&Path>) -> anyhow::Result<()> {
    ui::banner();

    // ── 1. Context: RAG index + hardware tier + model ──────────────────────
    ui::step(1, 3, "Analyzing system context...");
    let index = NixOptionIndex::open(&cfg.rag_db_path)?;
    index.bootstrap()?;
    let known = index.count()?;
    if known == 0 {
        ui::warn("Local RAG index is empty — run `nix-agent ingest <options.json>` first.");
    } else {
        ui::detail(&format!("RAG index: {known} options"));
    }

    let tier = HardwareTier::detect();
    ui::detail(&format!(
        "Hardware tier: {} → {} model ({})",
        tier.label(),
        tier.param_label(),
        tier.model_file()
    ));

    #[cfg(feature = "embedded-llm")]
    let backend = {
        let b = load_inference_backend(cfg, tier, model)?;
        ui::detail(&format!("Model ready: {}", b.model_path().display()));
        b
    };
    #[cfg(not(feature = "embedded-llm"))]
    let backend = {
        let _ = model; // model selection only matters for the real engine
        ui::detail("Inference: offline stub backend (build with `--features embedded-llm`)");
        nix_agent::rag::StubBackend
    };

    let mut healer = LocalLlmHealer::with_backend(index, backend);

    // ── 2. Generation ──────────────────────────────────────────────────────
    ui::step(2, 3, "Generating Nix module...");
    ui::detail(&format!("request: \"{prompt}\""));
    let initial = healer
        .draft(prompt)
        .await
        .context("model failed to generate a module")?;

    // ── 3. AST-gate validation (no activation, no privileges) ──────────────
    ui::step(3, 3, "Validating module (AST gate)...");
    let plan_id = plan::make_plan_id(prompt, plan::now_unix());
    let outcome = plan::create_plan(
        cfg,
        prompt,
        &plan_id,
        initial,
        &mut healer,
        AstOnlyBuilder,
        ui::heal_event,
    )
    .await
    .context("planning failed")?;

    match outcome {
        PlanOutcome::Validated { plan, attempts } => {
            ui::success(&format!(
                "Plan validated — AST gate passed ({attempts} attempt{}).",
                plural(attempts)
            ));
            print_plan_layout(cfg, &plan, attempts);
        }
        PlanOutcome::Rejected { reason, attempts } => {
            ui::failure(&format!(
                "Could not produce a valid module after {attempts} attempt{}.",
                plural(attempts)
            ));
            ui::detail(&format!("last error: {reason}"));
        }
    }
    Ok(())
}

/// Resolve which GGUF the embedded engine should load, following the model
/// manager's cascade: an explicit `--model` path, else auto-discovered local
/// models (prompting when several exist), else the hardware tier's
/// auto-downloaded default.
#[cfg(feature = "embedded-llm")]
fn load_inference_backend(
    cfg: &AppConfig,
    tier: HardwareTier,
    explicit: Option<&Path>,
) -> anyhow::Result<nix_agent::rag::EmbeddedLlamaBackend> {
    use nix_agent::models;
    use nix_agent::rag::EmbeddedLlamaBackend;

    // 1. Explicit --model path always wins.
    if let Some(path) = explicit {
        if !path.is_file() {
            anyhow::bail!("--model path does not exist: {}", path.display());
        }
        ui::detail(&format!("Using model (--model): {}", path.display()));
        return EmbeddedLlamaBackend::load_from_path(tier, path)
            .context("could not load the specified model");
    }

    // 2. Auto-discover local models (XDG cache + Ollama blobs).
    let found = models::discover().context("model discovery failed")?;
    let chosen = match found.len() {
        0 => None,
        1 => Some(found[0].path.clone()),
        _ => {
            ui::detail(&format!("{} local models found:", found.len()));
            let idx = models::prompt_select(&found)?;
            Some(found[idx].path.clone())
        }
    };

    match chosen {
        Some(path) => {
            ui::detail(&format!("Selected model: {}", path.display()));
            EmbeddedLlamaBackend::load_from_path(tier, &path)
                .context("could not load the selected model")
        }
        // 3. Nothing local — fall back to the tier's first-run auto-download.
        None => {
            ui::detail("No local models found — falling back to the tier model (auto-download).");
            EmbeddedLlamaBackend::load(tier, &cfg.model_cache_dir, ui::progress)
                .context("could not initialize the local inference model")
        }
    }
}

/// `models list` — show every locally discoverable GGUF model.
fn cmd_models_list() -> anyhow::Result<()> {
    use nix_agent::models;

    ui::banner();
    println!();
    println!("{}", nix_agent::ui::frame_header("Local GGUF Models"));

    let found = models::discover().context("model discovery failed")?;
    if found.is_empty() {
        ui::warn("No models found in the XDG cache or Ollama store.");
        ui::hint("Pull the recommended model: nix-agent models pull default");
        return Ok(());
    }

    for m in &found {
        ui::field(
            &format!("{} [{}]", m.name, m.source.label()),
            &m.human_size(),
        );
        ui::detail(&m.path.display().to_string());
    }
    println!("{}", nix_agent::ui::frame_rule());
    ui::detail(&format!("{} model(s) available.", found.len()));
    Ok(())
}

/// `models pull` — download a GGUF model into the managed XDG directory.
fn cmd_models_pull(target: &str) -> anyhow::Result<()> {
    use nix_agent::models;

    ui::banner();
    let dir = models::models_dir().context("could not resolve the local model directory")?;
    let path = models::pull(target, &dir).context("model download failed")?;
    ui::success(&format!("Model ready: {}", path.display()));
    ui::hint("Use it with: nix-agent plan \"...\" --model <path>  (or just `plan` — it is auto-discovered)");
    Ok(())
}

/// Environment overrides worth forwarding across a `sudo` re-exec so the
/// escalated process resolves the same paths the user configured.
#[cfg(unix)]
const FORWARDED_ENV: &[&str] = &[
    "NIX_AGENT_CONFIG",
    "NIX_AGENT_CONFIG_DIR",
    "NIX_AGENT_DB",
    "NIX_AGENT_MODEL_CACHE",
    "NIX_AGENT_MODEL",
    "NIX_AGENT_TIMEOUT",
];

/// Ensure `apply` runs as root. If we already are, return and continue.
/// Otherwise re-exec the current binary under `sudo`, preserving the original
/// arguments and any explicitly-set `NIX_AGENT_*` env vars, then exit with the
/// child's status. `apply` performs no inference, so re-execing the real binary
/// (rather than the Vulkan wrapper) is safe — the GPU path is never used here.
#[cfg(unix)]
fn escalate_if_needed() -> anyhow::Result<()> {
    use std::process::{Command, Stdio};

    if nix::unistd::getuid().is_root() {
        return Ok(());
    }

    ui::warn("Writing to the system config requires root — re-running under sudo...");

    let exe = std::env::current_exe().context("could not resolve the current executable path")?;

    // Original CLI arguments, minus argv[0].
    let forwarded_args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();

    let mut cmd = Command::new("sudo");

    // Tell sudo to keep the env vars the user actually set (env_reset scrubs the
    // rest). They are already present in this process's environment, which the
    // child Command inherits, so `--preserve-env=<list>` is enough.
    let to_preserve: Vec<&str> = FORWARDED_ENV
        .iter()
        .copied()
        .filter(|k| std::env::var_os(k).is_some())
        .collect();
    if !to_preserve.is_empty() {
        cmd.arg(format!("--preserve-env={}", to_preserve.join(",")));
    }

    // `--` stops sudo option parsing before our own binary + args.
    cmd.arg("--").arg(&exe).args(&forwarded_args);
    cmd.stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    let status = cmd
        .status()
        .context("failed to invoke `sudo` (is it installed and on PATH?)")?;

    std::process::exit(status.code().unwrap_or(1));
}

/// Non-unix fallback: no privilege model to escalate against, so proceed and let
/// any real permission error surface from the filesystem.
#[cfg(not(unix))]
fn escalate_if_needed() -> anyhow::Result<()> {
    Ok(())
}

/// `apply` — install a validated plan into the sandbox and activate it.
async fn cmd_apply(cfg: &AppConfig, plan_arg: &str) -> anyhow::Result<()> {
    // Self-escalate before any output so the privileged run owns the workflow.
    escalate_if_needed()?;

    ui::banner();

    ui::step(1, 2, "Loading validated plan...");
    let plan = plan::load_plan(cfg, plan_arg).context("could not load the plan")?;
    let module_path = cfg.module_path(&plan.id);
    ui::detail(&format!("Plan: {}", plan.id));
    ui::detail(&format!("Target module: {}", module_path.display()));

    // Interactive, git-style review of the change before any privileged action.
    // The "old" side is whatever module is already installed (empty on first
    // apply); the "new" side is the validated plan body.
    let current = std::fs::read_to_string(&module_path).unwrap_or_default();
    if !nix_agent::ui::render_and_confirm_plan(&current, &plan.module_source) {
        ui::warn("Apply cancelled. System left unchanged.");
        return Ok(());
    }

    ui::step(
        2,
        2,
        "Installing module and activating (nixos-rebuild test)...",
    );
    let mut rebuild = NixosRebuild::new(cfg.build_mode);
    rebuild.use_system_config = true;
    rebuild.timeout = cfg.build_timeout;

    let outcome = plan::apply_plan(cfg, &plan, rebuild)
        .await
        .context("execution layer failure")?;

    match outcome {
        ApplyOutcome::Activated { module_path } => {
            ui::success(&format!(
                "Module installed and system activated: {}",
                module_path.display()
            ));
            show_diff(&cfg.ai_generated_dir());
        }
        ApplyOutcome::Failed { reason } => {
            ui::failure(&format!("Activation failed: {reason}"));
            ui::detail("module rolled back; system left unchanged.");
        }
    }
    Ok(())
}

fn plural(n: usize) -> &'static str {
    if n == 1 {
        ""
    } else {
        "s"
    }
}

/// Render the clean, markdown-like plan summary printed after a successful plan.
fn print_plan_layout(cfg: &AppConfig, plan: &Plan, attempts: usize) {
    println!();
    ui::rule();
    ui::heading(&format!("  Plan: {}", plan.id));
    ui::rule();
    ui::field("Prompt", &plan.prompt);
    ui::field(
        "Module",
        &format!("{} (created on apply)", cfg.module_path(&plan.id).display()),
    );
    ui::field(
        "AST gate",
        &format!("PASSED ({attempts} attempt{})", plural(attempts)),
    );
    ui::field("Plan file", &cfg.plan_file.display().to_string());
    println!();
    ui::code_block(&plan.module_source);
    println!();
    ui::hint(&format!(
        "This was a dry run — nothing was activated. Apply with:\n  sudo nix-agent apply --plan {}",
        plan.id
    ));
}

/// Show a generated module's change as a `git diff`, falling back gracefully
/// when the target is not under version control.
fn show_diff(target: &Path) {
    let dir = target.parent().unwrap_or_else(|| Path::new("."));
    let result = std::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["--no-pager", "diff", "--"])
        .arg(target)
        .output();

    match result {
        Ok(out) if out.status.success() && !out.stdout.is_empty() => {
            ui::diff_block(&String::from_utf8_lossy(&out.stdout));
        }
        _ => {
            ui::detail(&format!("installed module directory: {}", target.display()));
        }
    }
}

/// Clean, high-level terminal output. Keeps the self-healing internals quiet,
/// surfacing only compact step and attempt status — never raw build logs.
mod ui {
    use crossterm::style::Stylize;
    use nix_agent::execution::HealEvent;

    pub fn banner() {
        println!(
            "{}",
            "nix-agent · local air-gapped Verified Patch Assistant for NixOS"
                .cyan()
                .bold()
        );
    }

    pub fn step(n: u32, total: u32, label: &str) {
        println!("{} {}", format!("[{n}/{total}]").dark_grey(), label.bold());
    }

    pub fn detail(msg: &str) {
        println!("      {}", msg.dark_grey());
    }

    /// Model-download / first-run progress, printed verbatim (already formatted).
    /// Only wired in when the embedded inference backend is compiled.
    #[cfg(feature = "embedded-llm")]
    pub fn progress(msg: &str) {
        println!("{}", msg.bold());
    }

    pub fn success(msg: &str) {
        println!("{} {}", "✓".green().bold(), msg.green());
    }

    pub fn warn(msg: &str) {
        println!("{} {}", "!".yellow().bold(), msg.yellow());
    }

    pub fn failure(msg: &str) {
        println!("{} {}", "✗".red().bold(), msg.red());
    }

    pub fn rule() {
        println!(
            "{}",
            "────────────────────────────────────────────".dark_grey()
        );
    }

    pub fn heading(text: &str) {
        println!("{}", text.bold());
    }

    pub fn field(label: &str, value: &str) {
        println!("  {} {}", format!("{label:<11}").dark_grey(), value);
    }

    pub fn code_block(code: &str) {
        println!("{}", "```nix".dark_grey());
        print!("{code}");
        if !code.ends_with('\n') {
            println!();
        }
        println!("{}", "```".dark_grey());
    }

    pub fn hint(text: &str) {
        println!("{}", text.cyan());
    }

    /// Render a self-healing progress event compactly. The first attempt is
    /// implied by the step header, so only retries are annotated.
    pub fn heal_event(ev: HealEvent) {
        match ev {
            HealEvent::Attempt {
                attempt,
                max_attempts,
            } if attempt > 1 => {
                println!(
                    "      {} {}",
                    format!("[Attempt {attempt}/{max_attempts}]").yellow(),
                    "re-validating module...".dark_grey()
                );
            }
            HealEvent::Attempt { .. } => {}
            HealEvent::Failed { attempt, summary } => {
                println!(
                    "      {} {}",
                    format!("[Attempt {attempt}]").yellow(),
                    format!("error detected: {summary}").dark_grey()
                );
            }
            HealEvent::Regenerating { next_attempt } => {
                println!(
                    "      {} {}",
                    format!("[Attempt {next_attempt}]").yellow(),
                    "Repairing: AI is rewriting the module...".dark_grey()
                );
            }
        }
    }

    /// Pretty-print a unified diff with +/-/hunk coloring.
    pub fn diff_block(diff: &str) {
        println!("\n{}", "── resulting git diff ──".dark_grey());
        for line in diff.lines() {
            let styled = if line.starts_with("+++") || line.starts_with("---") {
                line.bold().to_string()
            } else if line.starts_with('+') {
                line.green().to_string()
            } else if line.starts_with('-') {
                line.red().to_string()
            } else if line.starts_with("@@") {
                line.cyan().to_string()
            } else {
                line.dark_grey().to_string()
            };
            println!("{styled}");
        }
    }
}
