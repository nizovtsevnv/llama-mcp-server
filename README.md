# llama-mcp-server

[![CI](https://github.com/nizovtsevnv/llama-mcp-server/actions/workflows/ci.yml/badge.svg)](https://github.com/nizovtsevnv/llama-mcp-server/actions/workflows/ci.yml)
[![Release](https://github.com/nizovtsevnv/llama-mcp-server/actions/workflows/release.yml/badge.svg)](https://github.com/nizovtsevnv/llama-mcp-server/actions/workflows/release.yml)
[![Crate](https://img.shields.io/crates/v/llama-mcp-server)](https://crates.io/crates/llama-mcp-server)

Local LLM inference MCP server powered by [llama.cpp](https://github.com/ggerganov/llama.cpp).

Standalone binary that exposes a configurable chat tool over MCP (Model Context Protocol) via stdio or HTTP transport using JSON-RPC 2.0.

## Features

- **Configurable tool name** — default `chat`, customizable via `--tool_name`
- **Flexible input** — accepts plain text (`text`) or multi-turn conversation (`messages` array)
- **Per-request parameters** — temperature, max_tokens, top_p, top_k, prompt_suffix overridable per call
- **Chat template** — uses the model's built-in GGUF chat template for prompt formatting
- **Thinking block stripping** — automatically removes `<think>...</think>` blocks from output
- **Optional CUDA** — GPU acceleration via llama.cpp CUDA backend
- **HTTP transport** — MCP Streamable HTTP with Bearer token authentication and session management
- **Dual transport** — stdio (default) or HTTP mode via `--transport` flag

## Architecture

Single crate, four source modules:

| Module | Responsibility |
|---|---|
| `src/main.rs` | CLI parsing (clap), backend/model init, entry point, transport selection |
| `src/engine.rs` | LLM engine: model loading, context creation, prompt formatting, token generation |
| `src/mcp.rs` | JSON-RPC 2.0 dispatch, message/param parsing, MCP tool result helpers, stdio loop |
| `src/http.rs` | HTTP transport: axum server, Bearer auth, session management, per-request context |

## CLI Arguments

```
llama-mcp-server --model <PATH> [OPTIONS]

Options:
  --model <PATH>            Path to GGUF model file [required]
  --n_gpu_layers <N>        GPU layers to offload (0 = CPU, 99 = all) [default: 99]
  --n_ctx <N>               Context window size in tokens [default: 4096]
  --n_batch <N>             Batch size for prompt processing [default: 2048]
  --n_threads <N>           Number of CPU threads [default: 4]
  --tool_name <NAME>        MCP tool name [default: chat]
  --max_tokens <N>          Default max tokens to generate [default: 2048]
  --temperature <F>         Default temperature (0.0 = greedy) [default: 0.7]
  --top_p <F>               Default top-p (nucleus sampling) [default: 0.9]
  --top_k <N>               Default top-k sampling [default: 40]
  --repeat_penalty <F>      Repeat penalty [default: 1.1]
  --seed <N>                Random seed [default: 42]
  --prompt_suffix <TEXT>    Suffix appended to system prompt (e.g. "/think")
  --transport <MODE>        Transport mode: stdio or http [default: stdio]
  --host <HOST>             Host to bind HTTP server [default: 127.0.0.1]
  --port <PORT>             Port for HTTP server [default: 8080]
  --auth <TOKEN>            Bearer token for HTTP authentication (optional)
  --version                 Print version and exit
```

## Build

### Prerequisites

- Rust toolchain (stable)
- CMake (llama.cpp build dependency)
- libclang (bindgen dependency)

### Native build

```bash
cargo build --release
```

### CUDA build

CUDA requires glibc and NVIDIA CUDA toolkit (nvcc, cudart, cuBLAS):

```bash
cargo build --release --features cuda
```

With Nix (requires unfree packages):

```bash
nix build .#cuda
```

## Runtime Dependencies

- **GGUF model file** — any GGUF-format model compatible with llama.cpp (e.g. from [Hugging Face](https://huggingface.co/models?search=gguf))

## Rust Dependencies

| Crate | Purpose |
|---|---|
| `llama-cpp-2` | Rust bindings for llama.cpp |
| `clap` | CLI argument parsing |
| `serde`, `serde_json` | JSON serialization for MCP protocol |
| `tracing`, `tracing-subscriber` | Structured logging to stderr |
| `axum` | HTTP server framework for MCP HTTP transport |
| `tokio` | Async runtime for HTTP transport |
| `uuid` | Session ID generation (UUID v4) |

## MCP Protocol

The server supports two transport modes:

- **stdio** (default) — communicates over stdin/stdout, one JSON object per line
- **HTTP** — MCP Streamable HTTP on `POST /mcp` and `DELETE /mcp`

### HTTP Transport

Start the server in HTTP mode:

```bash
llama-mcp-server --model model.gguf --transport http --port 8080 --auth secret123
```

**Authentication**: when `--auth` is set, all requests must include `Authorization: Bearer <token>`. Without `--auth`, authentication is disabled.

**Sessions**: the `initialize` request returns an `Mcp-Session-Id` header. All subsequent requests must include this header. Sessions are terminated via `DELETE /mcp`.

Example session:

```bash
# Initialize (get session ID)
curl -s -D- -X POST http://127.0.0.1:8080/mcp \
  -H "Authorization: Bearer secret123" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}'

# Use the Mcp-Session-Id from the response headers for subsequent requests
curl -s -X POST http://127.0.0.1:8080/mcp \
  -H "Authorization: Bearer secret123" \
  -H "Content-Type: application/json" \
  -H "Mcp-Session-Id: <session-id>" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list"}'

# Terminate session
curl -s -X DELETE http://127.0.0.1:8080/mcp \
  -H "Authorization: Bearer secret123" \
  -H "Mcp-Session-Id: <session-id>"
```

### Stdio Transport

The server communicates over stdin/stdout using JSON-RPC 2.0, one JSON object per line.

### Initialize

Request:
```json
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}
```

Response:
```json
{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"llama-mcp-server","version":"<version>"}}}
```

### List tools

Request:
```json
{"jsonrpc":"2.0","id":2,"method":"tools/list"}
```

### Chat (simple text)

Request:
```json
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"chat","arguments":{"text":"Hello, how are you?"}}}
```

### Chat (multi-turn)

Request:
```json
{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"chat","arguments":{"messages":[{"role":"system","content":"You are helpful."},{"role":"user","content":"Hi"}],"temperature":0.5}}}
```

Response:
```json
{"jsonrpc":"2.0","id":4,"result":{"content":[{"type":"text","text":"Hello! How can I help you today?"}]}}
```

## CI/CD

GitHub Actions workflows:

- **CI** (`ci.yml`) — runs `cargo fmt`, `cargo clippy`, `cargo test` on every push/PR to `main`/`develop`
- **Release** (`release.yml`) — builds binaries for 6 targets on tag push (`v*`), uploads as release assets

Release targets:

| Artifact | Build method | Notes |
|---|---|---|
| `linux-x86_64` | nix (default) | glibc, CPU only |
| `linux-x86_64-musl` | nix (musl) | Static binary, CPU only |
| `linux-x86_64-cuda` | cargo + CUDA toolkit | glibc, GPU acceleration |
| `windows-x86_64` | cargo (native) | Windows runner |
| `macos-x86_64` | nix (default) | Intel Mac |
| `macos-arm64` | nix (default) | Apple Silicon |

Release process:
1. Create a git tag: `git tag vX.Y.Z && git push --tags`
2. CI builds binaries for all 6 targets (Linux/macOS via nix, Windows/CUDA via cargo)
3. Create a GitHub release from the tag — CI attaches build artifacts automatically

To update `cargoHash` in `flake.nix` after changing dependencies:
```bash
./scripts/update-cargo-hash.sh
```

## Usage

### Claude Desktop (stdio)

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "llama": {
      "command": "/path/to/llama-mcp-server",
      "args": ["--model", "/path/to/model.gguf"]
    }
  }
}
```

### HTTP mode

```bash
llama-mcp-server --model /path/to/model.gguf --transport http --port 8080 --auth mytoken
```

Connect any HTTP-capable MCP client to `http://127.0.0.1:8080/mcp`. All requests require `Authorization: Bearer mytoken` and `Content-Type: application/json`. See [HTTP Transport](#http-transport) for protocol details.

### Any MCP client (stdio)

The server reads JSON-RPC requests from stdin and writes responses to stdout. Logs go to stderr. Connect any MCP-compatible client using stdio transport.
