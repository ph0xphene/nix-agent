//! Runtime configuration and host-hardware detection for the `nix-agent` CLI.
//!
//! Holds the paths and tunables that bind the `ast`, `execution`, and `rag`
//! layers into one tool, plus [`HardwareTier`] — which inspects host RAM to pick
//! the largest local model the machine can comfortably run. Everything has
//! sensible offline defaults; the only network access is the one-time model
//! download performed by the embedded inference backend.

use std::path::PathBuf;
use std::time::Duration;

use sysinfo::System;

use crate::execution::BuildMode;

/// Default editable system configuration.
pub const DEFAULT_CONFIG_PATH: &str = "/etc/nixos/configuration.nix";
/// Default per-build wall-clock budget, in seconds.
pub const DEFAULT_TIMEOUT_SECS: u64 = 60;
/// Default cap on self-healing attempts.
pub const DEFAULT_MAX_ATTEMPTS: usize = 3;
/// Default path where `plan` writes the validated plan file.
pub const DEFAULT_PLAN_FILE: &str = ".nix-agent-plan.nix";

/// RAM threshold (GiB) at or above which the high-end 7B model is selected.
pub const RAM_HIGH_END_GIB: u64 = 16;
/// RAM threshold (GiB) at or above which the medium 3B model is selected.
pub const RAM_MEDIUM_GIB: u64 = 8;

const BYTES_PER_GIB: u64 = 1024 * 1024 * 1024;

/// Hardware capability bucket, derived from host RAM, that maps to a concrete
/// quantized GGUF model. CPU-only / older machines land in [`HardwareTier::Low`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum HardwareTier {
    /// >= 16 GiB RAM — 7B model.
    HighEnd,
    /// >= 8 GiB RAM — 3B model.
    Medium,
    /// Everything else (older machines, CPU-only) — 1.5B model.
    Low,
}

impl HardwareTier {
    /// Inspect the host's total RAM and pick a tier.
    pub fn detect() -> Self {
        let mut sys = System::new();
        sys.refresh_memory();
        Self::from_total_ram_bytes(sys.total_memory())
    }

    /// Pure RAM-to-tier mapping (testable without touching real hardware).
    pub fn from_total_ram_bytes(total_bytes: u64) -> Self {
        if total_bytes >= RAM_HIGH_END_GIB * BYTES_PER_GIB {
            Self::HighEnd
        } else if total_bytes >= RAM_MEDIUM_GIB * BYTES_PER_GIB {
            Self::Medium
        } else {
            Self::Low
        }
    }

    /// Human-readable tier name.
    pub fn label(self) -> &'static str {
        match self {
            Self::HighEnd => "HighEnd",
            Self::Medium => "Medium",
            Self::Low => "Low",
        }
    }

    /// Parameter-count label of the selected model, e.g. `7B`.
    pub fn param_label(self) -> &'static str {
        match self {
            Self::HighEnd => "7B",
            Self::Medium => "3B",
            Self::Low => "1.5B",
        }
    }

    /// Hugging Face repository hosting the GGUF weights for this tier.
    /// Qwen2.5-Coder is used throughout — it is tuned for code generation, which
    /// suits emitting valid Nix expressions.
    pub fn model_repo(self) -> &'static str {
        match self {
            Self::HighEnd => "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            Self::Medium => "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
            Self::Low => "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
        }
    }

    /// GGUF file name within [`Self::model_repo`] (Q4_K_M quantization).
    pub fn model_file(self) -> &'static str {
        match self {
            Self::HighEnd => "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
            Self::Medium => "qwen2.5-coder-3b-instruct-q4_k_m.gguf",
            Self::Low => "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        }
    }

    /// Approximate on-disk download size in GB, for the first-run progress line.
    pub fn approx_download_gb(self) -> f64 {
        match self {
            Self::HighEnd => 4.7,
            Self::Medium => 2.1,
            Self::Low => 1.1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AppConfig {
    /// The main system configuration the agent imports into but NEVER edits.
    pub config_path: PathBuf,
    /// Sandbox root under which generated modules are written, at
    /// `<config_dir>/modules/ai-generated/<plan-id>.nix`. Defaults to the
    /// current directory so the workflow can be tested safely without root.
    pub config_dir: PathBuf,
    /// Where `plan` writes the validated, ready-to-apply plan file.
    pub plan_file: PathBuf,
    /// SQLite file backing the local NixOS-options RAG index.
    pub rag_db_path: PathBuf,
    /// Directory used as the Hugging Face cache for downloaded GGUF weights.
    pub model_cache_dir: PathBuf,
    /// Time budget for a single `nixos-rebuild` invocation.
    pub build_timeout: Duration,
    /// Which `nixos-rebuild` mode `apply` runs. Defaults to `test` (build +
    /// activate until reboot) so a successful apply mutates the live system.
    pub build_mode: BuildMode,
    /// Maximum self-healing attempts before giving up.
    pub max_attempts: usize,
}

impl Default for AppConfig {
    fn default() -> Self {
        let cache = cache_root().join("nix-agent");
        Self {
            config_path: PathBuf::from(DEFAULT_CONFIG_PATH),
            config_dir: PathBuf::from("."),
            plan_file: PathBuf::from(DEFAULT_PLAN_FILE),
            rag_db_path: cache.join("rag.db"),
            model_cache_dir: cache.join("models"),
            build_timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            build_mode: BuildMode::Test,
            max_attempts: DEFAULT_MAX_ATTEMPTS,
        }
    }
}

impl AppConfig {
    /// Build a config from defaults, applying any environment overrides:
    /// `NIX_AGENT_CONFIG`, `NIX_AGENT_CONFIG_DIR`, `NIX_AGENT_DB`,
    /// `NIX_AGENT_MODEL_CACHE`, `NIX_AGENT_TIMEOUT` (seconds).
    pub fn load() -> Self {
        let mut cfg = Self::default();

        if let Some(p) = env_path("NIX_AGENT_CONFIG") {
            cfg.config_path = p;
        }
        if let Some(p) = env_path("NIX_AGENT_CONFIG_DIR") {
            cfg.config_dir = p;
        }
        if let Some(p) = env_path("NIX_AGENT_DB") {
            cfg.rag_db_path = p;
        }
        if let Some(p) = env_path("NIX_AGENT_MODEL_CACHE") {
            cfg.model_cache_dir = p;
        }
        if let Some(secs) = env_nonempty("NIX_AGENT_TIMEOUT").and_then(|s| s.parse::<u64>().ok()) {
            cfg.build_timeout = Duration::from_secs(secs);
        }

        cfg
    }

    /// The sandbox directory holding generated modules:
    /// `<config_dir>/modules/ai-generated`.
    pub fn ai_generated_dir(&self) -> PathBuf {
        self.config_dir.join("modules").join("ai-generated")
    }

    /// Destination path for a generated module with the given plan id.
    pub fn module_path(&self, plan_id: &str) -> PathBuf {
        self.ai_generated_dir().join(format!("{plan_id}.nix"))
    }

    /// Ensure the directories backing the RAG database and model cache exist,
    /// creating them (and any parents) on first run. Called at startup.
    pub fn ensure_storage(&self) -> std::io::Result<()> {
        if let Some(parent) = self.rag_db_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        if !self.model_cache_dir.as_os_str().is_empty() {
            std::fs::create_dir_all(&self.model_cache_dir)?;
        }
        Ok(())
    }
}

/// `$XDG_CACHE_HOME`, else `$HOME/.cache`, else a relative `.cache`.
fn cache_root() -> PathBuf {
    if let Some(x) = std::env::var_os("XDG_CACHE_HOME") {
        if !x.is_empty() {
            return PathBuf::from(x);
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        if !home.is_empty() {
            return PathBuf::from(home).join(".cache");
        }
    }
    PathBuf::from(".cache")
}

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var_os(key)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

fn env_nonempty(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sane() {
        let cfg = AppConfig::default();
        assert_eq!(cfg.config_path, PathBuf::from(DEFAULT_CONFIG_PATH));
        assert_eq!(cfg.config_dir, PathBuf::from("."));
        assert_eq!(cfg.plan_file, PathBuf::from(DEFAULT_PLAN_FILE));
        assert!(cfg.rag_db_path.ends_with("nix-agent/rag.db"));
        assert!(cfg.model_cache_dir.ends_with("nix-agent/models"));
        assert_eq!(cfg.build_timeout, Duration::from_secs(60));
        assert_eq!(cfg.max_attempts, 3);
    }

    #[test]
    fn sandbox_paths_are_scoped_to_config_dir() {
        let cfg = AppConfig {
            config_dir: PathBuf::from("/etc/nixos"),
            ..AppConfig::default()
        };
        assert_eq!(
            cfg.ai_generated_dir(),
            PathBuf::from("/etc/nixos/modules/ai-generated")
        );
        assert_eq!(
            cfg.module_path("2026-06-29-tmux"),
            PathBuf::from("/etc/nixos/modules/ai-generated/2026-06-29-tmux.nix")
        );
    }

    #[test]
    fn ensure_storage_creates_dirs() {
        let mut base = std::env::temp_dir();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        base.push(format!("nix-agent-cfg-{nanos}"));

        let cfg = AppConfig {
            rag_db_path: base.join("db").join("rag.db"),
            model_cache_dir: base.join("models"),
            ..AppConfig::default()
        };
        cfg.ensure_storage().unwrap();
        assert!(cfg.rag_db_path.parent().unwrap().is_dir());
        assert!(cfg.model_cache_dir.is_dir());

        std::fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn tier_mapping_respects_thresholds() {
        let gib = |g: u64| g * BYTES_PER_GIB;
        // High-end: at and above 16 GiB.
        assert_eq!(HardwareTier::from_total_ram_bytes(gib(32)), HardwareTier::HighEnd);
        assert_eq!(HardwareTier::from_total_ram_bytes(gib(16)), HardwareTier::HighEnd);
        // Medium: [8, 16) GiB.
        assert_eq!(HardwareTier::from_total_ram_bytes(gib(16) - 1), HardwareTier::Medium);
        assert_eq!(HardwareTier::from_total_ram_bytes(gib(8)), HardwareTier::Medium);
        // Low: below 8 GiB (older machines, CPU-only).
        assert_eq!(HardwareTier::from_total_ram_bytes(gib(8) - 1), HardwareTier::Low);
        assert_eq!(HardwareTier::from_total_ram_bytes(gib(4)), HardwareTier::Low);
        assert_eq!(HardwareTier::from_total_ram_bytes(0), HardwareTier::Low);
    }

    #[test]
    fn tier_model_descriptors_are_consistent() {
        for (tier, params) in [
            (HardwareTier::HighEnd, "7B"),
            (HardwareTier::Medium, "3B"),
            (HardwareTier::Low, "1.5B"),
        ] {
            assert_eq!(tier.param_label(), params);
            assert!(tier.model_file().ends_with(".gguf"));
            assert!(tier.model_repo().contains("GGUF"));
            assert!(tier.approx_download_gb() > 0.0);
        }
    }

    #[test]
    fn detect_returns_a_tier() {
        // Smoke test: detection runs against the real host without panicking.
        let _ = HardwareTier::detect();
    }
}
