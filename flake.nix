{
  inputs = {
    # Pinned: needs Cargo 1.85+ (edition2024) and gcc < 15 (musl static libstdc++)
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };
        cudaPkgs = import nixpkgs {
          inherit system overlays;
          config.allowUnfree = true;
        };

        cargoToml = builtins.fromTOML (builtins.readFile ./Cargo.toml);
        pname = cargoToml.package.name;
        version = cargoToml.package.version;

        cargoHash = "sha256-Sf4AVCXPEJR+EktVMiLrYndr2vhuWculRVaYggIsA0w=";

        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          targets = [
            "x86_64-unknown-linux-musl"
            "x86_64-pc-windows-gnu"
          ];
        };

        # Common native build inputs for llama.cpp (cmake) and bindgen (libclang)
        commonNativeBuildInputs = with pkgs; [
          cmake
          pkg-config
          llvmPackages.libclang
        ];

        commonEnv = {
          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
          BINDGEN_EXTRA_CLANG_ARGS = "-I${pkgs.glibc.dev}/include -I${pkgs.llvmPackages.libclang.lib}/lib/clang/${pkgs.llvmPackages.libclang.version}/include";
        };
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            rustToolchain
            pkgs.cmake
            pkgs.pkg-config
            pkgs.libclang
            pkgs.llvmPackages.libclang
          ];

          LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
          BINDGEN_EXTRA_CLANG_ARGS = "-I${pkgs.glibc.dev}/include";

          shellHook = ''
            # Set up git hooks if .git exists
            if [ -d .git ]; then
              mkdir -p .git/hooks
              cat > .git/hooks/pre-commit << 'HOOK'
#!/usr/bin/env bash
set -e
cargo fmt -- --check
cargo clippy -- -D warnings
cargo test
HOOK
              chmod +x .git/hooks/pre-commit
            fi
          '';
        };

        packages = {
          # Native linux build (glibc)
          default = pkgs.rustPlatform.buildRustPackage (commonEnv // {
            inherit pname version;
            src = ./.;
            inherit cargoHash;

            nativeBuildInputs = commonNativeBuildInputs;

            meta = with pkgs.lib; {
              description = "Local LLM inference MCP server powered by llama.cpp";
              license = licenses.mit;
            };
          });

          # Static linux build (musl)
          musl = let
            muslPkgs = pkgs.pkgsCross.musl64;
            gccLib = "${muslPkgs.stdenv.cc.cc}/x86_64-unknown-linux-musl/lib";
          in muslPkgs.rustPlatform.buildRustPackage (commonEnv // {
            pname = "${pname}-musl";
            inherit version;
            src = ./.;
            inherit cargoHash;

            nativeBuildInputs = commonNativeBuildInputs;

            CARGO_BUILD_TARGET = "x86_64-unknown-linux-musl";
            PKG_CONFIG_ALL_STATIC = "1";

            # Force fully static binary: shadow dynamic libstdc++/libgcc_s with
            # linker scripts that redirect to static archives. This avoids the
            # gcc 15 PIE/libstdc++.a incompatibility when using -static.
            # See: https://github.com/NixOS/nixpkgs/issues/425367
            preBuild = ''
              OVERRIDE=$TMPDIR/force-static
              mkdir -p $OVERRIDE
              echo "INPUT(${gccLib}/libstdc++.a)" > $OVERRIDE/libstdc++.so
              echo "INPUT(${gccLib}/libstdc++.a)" > $OVERRIDE/libstdc++.so.6
              echo "INPUT(${gccLib}/libgcc_s.a)"  > $OVERRIDE/libgcc_s.so
              echo "INPUT(${gccLib}/libgcc_s.a)"  > $OVERRIDE/libgcc_s.so.1
              export CARGO_TARGET_X86_64_UNKNOWN_LINUX_MUSL_RUSTFLAGS="-C target-feature=+crt-static -C relocation-model=static -C link-args=-L$OVERRIDE -C link-args=-lc"
            '';

            meta = with pkgs.lib; {
              description = "Local LLM inference MCP server powered by llama.cpp (musl static)";
              license = licenses.mit;
            };
          });

          # CUDA build (requires unfree NVIDIA packages)
          cuda = cudaPkgs.rustPlatform.buildRustPackage (commonEnv // {
            pname = "${pname}-cuda";
            inherit version;
            src = ./.;
            inherit cargoHash;

            buildFeatures = [ "cuda" ];

            nativeBuildInputs = commonNativeBuildInputs ++ [
              cudaPkgs.cudaPackages.cuda_nvcc
            ];
            buildInputs = [
              cudaPkgs.cudaPackages.cuda_cudart
              cudaPkgs.cudaPackages.libcublas
            ];

            meta = with cudaPkgs.lib; {
              description = "Local LLM inference MCP server powered by llama.cpp (CUDA)";
              license = licenses.mit;
            };
          });

          # Note: Windows build uses cargo directly in CI (cross-compilation
          # via nix has unresolved bindgen/cmake compatibility issues with
          # mingw and gcc 15). See .github/workflows/release.yml
        };
      }
    );
}
