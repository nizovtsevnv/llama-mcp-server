# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v0.1.1] - 2026-03-08

### Added
- LLM inference engine powered by llama.cpp (GGUF models)
- Configurable MCP tool name (default: `chat`)
- Flexible input: plain text (`text`) or multi-turn conversation (`messages` array)
- Per-request generation parameters: temperature, max_tokens, top_p, top_k, prompt_suffix
- Chat template formatting from GGUF model metadata
- Automatic `<think>...</think>` block stripping
- Dual transport: stdio (default) and HTTP with Bearer auth + session management
- Optional CUDA GPU acceleration via `--features cuda`
- Nix flake with native, musl-static, and CUDA build targets
- CI/CD: GitHub Actions for checks, builds, and releases (6 platforms)

[v0.1.1]: https://github.com/nizovtsevnv/llama-mcp-server/releases/tag/v0.1.1
