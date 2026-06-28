# nix-agent

> **Talk to your machine. It rewrites itself.**
> A local, air-gapped AI agent that declaratively mutates and self-heals your NixOS configuration — no cloud, no API keys, no Ollama daemon. Just one binary and your GPU.

[![Built with Rust](https://img.shields.io/badge/built_with-Rust-000000?logo=rust)](https://www.rust-lang.org/)
[![NixOS](https://img.shields.io/badge/NixOS-flake-5277C3?logo=nixos&logoColor=white)](https://nixos.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#license)

---

`nix-agent` turns a sentence into a verified system change. You describe what you
want in plain English; the agent retrieves the relevant NixOS options from a
local index, generates a Nix module with an **in-process LLM**, parses it through
an AST gate, runs `nixos-rebuild`, and — if the build breaks — reads the
compiler's own error and repairs its output. Up to three times. Silently.

It never edits your files imperatively. It generates valid Nix, tests it, and
shows you a `git diff`. If anything fails, your original configuration is
restored untouched.

---

## Key Features

- **🧠 Embedded GGUF inference — no Ollama, no network.** Runs
  [Qwen2.5-Coder](https://huggingface.co/Qwen) directly inside the binary via
  `llama.cpp`, with full GPU offload through **Vulkan** (Linux/NixOS) or
  **Metal** (macOS). After the first-run model download, it is fully air-gapped.
- **⚙️ Automatic hardware-tier profiling.** Inspects host RAM at startup and
  selects the largest model your machine can comfortably run — 7B, 3B, or 1.5B —
  with zero configuration.
- **🛡️ AST-gate isolation.** Every generated module is parsed with `rnix` before
  it is ever handed to `nixos-rebuild`. Malformed Nix is caught locally, with
  byte-precise diagnostics, and never reaches your system.
- **♻️ Self-healing build loop.** On a failed build, the agent parses the Nix
  error (file, line, column, symbol), correlates it back to its own AST, and
  re-prompts the model with grounded context — bounded to three attempts.
- **⏪ Automatic rollback.** If the loop exhausts its attempts, the original
  configuration is restored verbatim. Your system is never left in a broken
  intermediate state.
- **📦 One-command install.** Ships as a Nix flake. The Vulkan loader is wrapped
  into the binary, so the GPU survives even when the agent runs under `sudo`.

---

## How It Works

```
  prompt ──▶ [RAG retrieval] ──▶ [LLM generation] ──▶ [AST gate] ──▶ [nixos-rebuild]
                  │                     ▲                                  │
            local options          re-prompt with                    pass │ fail
            (SQLite index)         grounded error ◀───[parse stderr]◀──────┘
                                        │                                  │
                                   (max 3 tries)                      git diff ✓
```

---

## One-Command Quickstart

Run the agent on the fly — no cloning, no build step. Nix fetches, compiles, and
executes it in a single command:

```bash
sudo nix run github:youruser/nix-agent -- run "add tmux with custom keybindings"
```

> Requires Nix with flakes enabled:
> `nix-command` and `flakes` in your `experimental-features`.
> The first invocation compiles the project and downloads the hardware-matched
> model; subsequent runs are instant and offline.

The agent will print its pipeline as it works:

```
[1/3] Analyzing system context...
      RAG index: 18432 options
      Hardware tier: HighEnd → 7B model (qwen2.5-coder-7b-instruct-q4_k_m.gguf)
      Model ready: /root/.cache/nix-agent/models/...
[2/3] Generating Nix module...
[3/3] Testing and activating configuration...
✓ System rebuilt and activated (attempts: 1).
```

---

## Declarative Installation

For a permanent install, add `nix-agent` as a flake input and drop it into your
`environment.systemPackages`.

```nix
# flake.nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nix-agent.url = "github:youruser/nix-agent";
  };

  outputs = { nixpkgs, nix-agent, ... }: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        ./configuration.nix
        ({ pkgs, ... }: {
          environment.systemPackages = [
            nix-agent.packages.${pkgs.system}.default
          ];
        })
      ];
    };
  };
}
```

Then rebuild once, and `nix-agent` is available system-wide:

```bash
sudo nixos-rebuild switch --flake .#myhost
```

> Replace `youruser` with your GitHub handle (or a local `path:` reference) in
> both the quickstart and the input URL.

---

## Usage

### 1. Index the NixOS option corpus (one-time)

The agent grounds every generation in the real NixOS option set. Build the
options dump and ingest it into the local RAG database:

```bash
# Produce options.json from your current nixpkgs:
nix-build -E '(import <nixpkgs/nixos> { configuration = {}; }).config.system.build.manual.optionsJSON' -o options

nix-agent ingest options/share/doc/nixos/options.json
```

### 2. Mutate the system

```bash
# Generate → test → activate, in one shot:
sudo nix-agent run "enable the openssh daemon and open port 22"

# `heal` is an alias for `run`:
sudo nix-agent heal "install firefox and enable bluetooth"
```

If the model's first attempt fails to build, you'll see compact repair status —
not a wall of `stderr`:

```
      [Attempt 1] error detected: UndefinedVariable: pkgs
      [Attempt 2] Repairing: AI is rewriting the module...
✓ System rebuilt and activated (attempts: 2).
```

---

## Hardware Tiers

At startup the agent reads total system RAM and picks a model automatically.

| Tier        | RAM        | Model                          | Approx. download |
| ----------- | ---------- | ------------------------------ | ---------------- |
| **HighEnd** | ≥ 16 GiB   | Qwen2.5-Coder-7B-Instruct      | ~4.7 GB          |
| **Medium**  | ≥ 8 GiB    | Qwen2.5-Coder-3B-Instruct      | ~2.1 GB          |
| **Low**     | < 8 GiB    | Qwen2.5-Coder-1.5B-Instruct    | ~1.1 GB          |

All models are Q4_K_M quantized GGUF and run with `n_gpu_layers = 99` (full GPU
offload).

---

## Environment Variables

Every path and tunable has a sensible default and can be overridden:

| Variable               | Default                              | Description                                          |
| ---------------------- | ------------------------------------ | ---------------------------------------------------- |
| `NIX_AGENT_CONFIG`     | `/etc/nixos/configuration.nix`       | The `.nix` file the agent rewrites and rebuilds.     |
| `NIX_AGENT_DB`         | `~/.cache/nix-agent/rag.db`          | SQLite file backing the local NixOS-options index.   |
| `NIX_AGENT_MODEL_CACHE`| `~/.cache/nix-agent/models`          | Hugging Face cache for downloaded GGUF weights.      |
| `NIX_AGENT_TIMEOUT`    | `60`                                 | Per-`nixos-rebuild` wall-clock budget, in seconds.   |

```bash
# Example: point the agent at a non-standard config and a project-local index.
NIX_AGENT_CONFIG=/etc/nixos/hosts/laptop.nix \
NIX_AGENT_DB=./nix-agent.db \
  sudo -E nix-agent run "enable docker"
```

---

## Building from Source

The flake handles everything — the C/C++ toolchain, `cmake`, the Vulkan SDK, and
`bindgen` — automatically:

```bash
nix build github:youruser/nix-agent     # → ./result/bin/nix-agent
nix develop                             # dev shell with `cargo build --features vulkan`
```

To build with plain `cargo` outside Nix, you must supply the native toolchain
yourself and select an acceleration backend:

```bash
cargo build --release --features vulkan   # Linux / NixOS
cargo build --release --features metal    # macOS
```

The base crate (no `--features`) builds with **zero native dependencies** — the
inference engine is gated behind the `embedded-llm` feature (implied by `vulkan`
and `metal`), so CI and contributors without a C++ toolchain stay fast and green.

---

## Safety Model

`nix-agent` is built to follow the immutability of NixOS, not fight it:

- **No imperative edits.** All AST mutations produce a *new* source string; the
  agent decides when to stage it, and always under a backup.
- **Default build mode is activation-aware.** `run` uses `nixos-rebuild test`
  (build + activate until reboot), so a successful change is real but never
  silently written to your bootloader.
- **Local cache probe before any download.** The model is fetched only on first
  run, with an explicit progress notice; after that the agent is air-gapped.
- **Bounded autonomy.** The self-healing loop is hard-capped at three attempts
  and rolls back on exhaustion.

---

## License

Released under the [MIT License](LICENSE).
