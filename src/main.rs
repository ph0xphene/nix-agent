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
    },
    /// Privileged: install a validated plan into the sandbox and activate it.
    Apply {
        /// Plan id (e.g. 2026-06-29-tmux) or path to a plan file.
        #[arg(long)]
        plan: String,
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
        Command::Plan { prompt } => cmd_plan(&cfg, &prompt).await,
        Command::Apply { plan } => cmd_apply(&cfg, &plan).await,
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
async fn cmd_plan(cfg: &AppConfig, prompt: &str) -> anyhow::Result<()> {
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
        let b = nix_agent::rag::EmbeddedLlamaBackend::load(tier, &cfg.model_cache_dir, ui::progress)
            .context("could not initialize the local inference model")?;
        ui::detail(&format!("Model ready: {}", b.model_path().display()));
        b
    };
    #[cfg(not(feature = "embedded-llm"))]
    let backend = {
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

/// `apply` — install a validated plan into the sandbox and activate it.
async fn cmd_apply(cfg: &AppConfig, plan_arg: &str) -> anyhow::Result<()> {
    ui::banner();

    ui::step(1, 2, "Loading validated plan...");
    let plan = plan::load_plan(cfg, plan_arg).context("could not load the plan")?;
    let module_path = cfg.module_path(&plan.id);
    ui::detail(&format!("Plan: {}", plan.id));
    ui::detail(&format!("Target module: {}", module_path.display()));

    ui::step(2, 2, "Installing module and activating (nixos-rebuild test)...");
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
        println!("{}", "────────────────────────────────────────────".dark_grey());
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
