{
  description = "nix-agent — a local, air-gapped AI agent that declaratively mutates and self-heals NixOS configurations";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    let
      supportedSystems = [
        "aarch64-darwin"
        "x86_64-darwin"
        "x86_64-linux"
        "aarch64-linux"
      ];
    in
    flake-utils.lib.eachSystem supportedSystems (system:
      let
        pkgs = import nixpkgs { inherit system; };

        isDarwin = pkgs.stdenv.isDarwin;
        isLinux = pkgs.stdenv.isLinux;

        buildFeatures =
          if isDarwin then [ "metal" ]
          else [ "vulkan" ];

        nativeBuildInputs =
          [
            pkgs.cmake
            pkgs.pkg-config
            pkgs.rustPlatform.bindgenHook
          ]
          ++ pkgs.lib.optionals isLinux [
            pkgs.makeWrapper
            pkgs.shaderc
            pkgs.vulkan-headers
          ];

        buildInputs =
          [
            pkgs.openssl
          ]
          ++ pkgs.lib.optionals isLinux [
            pkgs.vulkan-loader
            pkgs.vulkan-headers
            pkgs.spirv-headers
            pkgs.spirv-tools
          ];

        nix-agent = pkgs.rustPlatform.buildRustPackage {
          pname = "nix-agent";
          version = "0.1.0";

          # Flakes only copy git-tracked files, so target/ and other build
          # artifacts are excluded automatically.
          src = ./.;

          cargoLock = {
            lockFile = ./Cargo.lock;
          };

          inherit nativeBuildInputs buildInputs buildFeatures;

          OPENSSL_DIR = "${pkgs.openssl.dev}";
          OPENSSL_LIB_DIR = "${pkgs.openssl.out}/lib";
          OPENSSL_INCLUDE_DIR = "${pkgs.openssl.dev}/include";

          postInstall = pkgs.lib.optionalString isLinux ''
            wrapProgram $out/bin/nix-agent \
              --prefix LD_LIBRARY_PATH : "${pkgs.vulkan-loader}/lib"
          '';

          meta = with pkgs.lib; {
            description = "Local, air-gapped AI agent for declarative NixOS configuration mutation and self-healing";
            homepage = "https://github.com/ph0xphene/nix-agent";
            license = licenses.mit;
            mainProgram = "nix-agent";
            platforms = supportedSystems;
          };
        };
      in
      {
        packages = {
          default = nix-agent;
          nix-agent = nix-agent;
        };

        # Enables `nix run github:ph0xphene/nix-agent -- run "..."`.
        apps = {
          default = flake-utils.lib.mkApp {
            drv = nix-agent;
            name = "nix-agent";
          };

          nix-agent = flake-utils.lib.mkApp {
            drv = nix-agent;
            name = "nix-agent";
          };
        };

        # `nix develop` — a shell with the full toolchain to build locally.
        devShells.default = pkgs.mkShell {
          inputsFrom = [ nix-agent ];
          packages = [
            pkgs.rustc
            pkgs.cargo
            pkgs.clippy
            pkgs.rust-analyzer
          ];
          LD_LIBRARY_PATH = pkgs.lib.optionalString isLinux "${pkgs.vulkan-loader}/lib";
        };
      });
}
