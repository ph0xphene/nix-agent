{
  description = "nix-agent — a local, air-gapped AI agent that declaratively mutates and self-heals NixOS configurations";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    # NixOS / Linux targets: the bundled inference engine ships with the Vulkan
    # backend, and the runtime wrapper hardcodes the Vulkan loader path.
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" ] (system:
      let
        pkgs = import nixpkgs { inherit system; };

        nix-agent = pkgs.rustPlatform.buildRustPackage {
          pname = "nix-agent";
          version = "0.1.0";

          # Flakes only copy git-tracked files, so target/ and other build
          # artifacts are excluded automatically.
          src = ./.;

          cargoLock = {
            lockFile = ./Cargo.lock;
          };

          nativeBuildInputs = [
            pkgs.cmake          # builds the vendored llama.cpp via the cmake crate
            pkgs.pkg-config     # locates the Vulkan loader / system libraries
            pkgs.makeWrapper    # wraps the final binary in postInstall

            # The embedded inference stack additionally needs these to build:
            pkgs.rustPlatform.bindgenHook  # libclang for llama-cpp-sys bindgen
            pkgs.shaderc                   # glslc, compiles GGML's Vulkan shaders
          ];

          buildInputs = [
            pkgs.vulkan-loader

            pkgs.vulkan-headers  # CMake FindVulkan needs these on the target include path
            pkgs.spirv-headers   # GGML's Vulkan backend find_package(SPIRV-Headers)
            pkgs.spirv-tools     # …and links the SPIR-V optimizer alongside it
            pkgs.openssl         # openssl-sys (hf-hub → reqwest/native-tls) links against it
          ];

          # Force the in-process GGUF inference engine with the Vulkan backend.
          buildFeatures = [ "vulkan" ];

          OPENSSL_DIR = "${pkgs.openssl.dev}";
          OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
          OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";
          # Hardcode the Vulkan loader into the binary so the GPU is never lost,
          # even when the agent is launched via `sudo` (which scrubs the
          # environment). Without this, `nixos-rebuild` activation under sudo
          # would fall back to CPU or fail to find libvulkan.so.
          postInstall = ''
            wrapProgram $out/bin/nix-agent \
              --prefix LD_LIBRARY_PATH : "${pkgs.vulkan-loader}/lib"
          '';

          meta = with pkgs.lib; {
            description = "Local, air-gapped AI agent for declarative NixOS configuration mutation and self-healing";
            homepage = "https://github.com/ph0xphene/nix-agent";
            license = licenses.mit;
            mainProgram = "nix-agent";
            platforms = [ "x86_64-linux" "aarch64-linux" ];
          };
        };
      in
      {
        packages.default = nix-agent;
        packages.nix-agent = nix-agent;

        # Enables `nix run github:ph0xphene/nix-agent -- run "..."`.
        apps.default = flake-utils.lib.mkApp {
          drv = nix-agent;
          name = "nix-agent";
        };

        # `nix develop` — a shell with the full toolchain to `cargo build --features vulkan`.
        devShells.default = pkgs.mkShell {
          inputsFrom = [ nix-agent ];
          packages = [
            pkgs.rustc
            pkgs.cargo
            pkgs.clippy
            pkgs.rust-analyzer
          ];
          # Let `cargo run --features vulkan` find the loader during development.
          LD_LIBRARY_PATH = "${pkgs.vulkan-loader}/lib";
        };
      });
}
