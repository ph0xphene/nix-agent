//! Local GGUF model management: discovery (XDG + Ollama), listing, and pulling.

pub mod manager;

pub use manager::{
    discover, human_bytes, models_dir, prompt_select, pull, LocalModel, ModelSource,
    DEFAULT_PULL_ALIAS, ENV_MODEL,
};
