# AGENTS.md — Instructions for AI Assistants

This file provides instructions for AI coding assistants (OpenCode, Claude Code, Cursor, Windsurf, etc.) working with the llama-mcp-server codebase.

## Sources of truth

- **README.md** — goals, architecture, protocol, build instructions. Before implementing any feature or module — consult the README. If a task contradicts the README — ask the user, don't guess.
- **Source code** — current state of implementation. Don't assume what's implemented — read the code. Don't suggest changes to files you haven't read.

## Workflow

1. Read README.md (in full or the relevant section)
2. Read the affected source files
3. Follow rules from the Restrictions section below

## Dev environment

```bash
nix develop          # reproducible environment; automatically sets up git hooks
cargo check          # quick compilation check
cargo test           # tests
cargo clippy         # linter
```

## Restrictions

- Do not commit with `--no-verify`
- Do not use `unwrap()` in production code (tests only)
- Do not add dependencies without necessity
- Do not create abstractions "for the future"
- Code comments, doc comments (`///`), log messages, and commit messages must be in English
- Communicate with the user in their language (match the language of the user's messages)
- One commit = one logical unit with a clear message
- Tests go with the code, not "later"
- Pre-commit hooks are mandatory: `cargo fmt`, `cargo clippy`, `cargo test`
