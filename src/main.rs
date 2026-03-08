use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use tracing::info;

mod engine;
mod http;
mod mcp;

use engine::{EngineConfig, GenerateParams};

#[derive(Parser)]
#[command(name = "llama-mcp-server", version)]
struct Args {
    /// Path to GGUF model file
    #[arg(long)]
    model: PathBuf,

    /// Number of layers to offload to GPU (0 = CPU only, 99 = all)
    #[arg(long, default_value_t = 99)]
    n_gpu_layers: u32,

    /// Context window size in tokens
    #[arg(long, default_value_t = 4096)]
    n_ctx: u32,

    /// Batch size for prompt processing
    #[arg(long, default_value_t = 2048)]
    n_batch: u32,

    /// Number of CPU threads
    #[arg(long, default_value_t = 4)]
    n_threads: i32,

    /// Tool name exposed via MCP
    #[arg(long, default_value = "chat")]
    tool_name: String,

    /// Default max tokens to generate
    #[arg(long, default_value_t = 2048)]
    max_tokens: usize,

    /// Default temperature (0.0 = greedy)
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,

    /// Default top-p (nucleus sampling)
    #[arg(long, default_value_t = 0.9)]
    top_p: f32,

    /// Default top-k sampling
    #[arg(long, default_value_t = 40)]
    top_k: i32,

    /// Repeat penalty
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u32,

    /// Suffix appended to the system prompt (e.g. "/think" for Qwen3)
    #[arg(long)]
    prompt_suffix: Option<String>,

    /// Transport mode: stdio or http
    #[arg(long, default_value = "stdio")]
    transport: String,

    /// Host to bind HTTP server (http transport only)
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port for HTTP server (http transport only)
    #[arg(long, default_value_t = 8080)]
    port: u16,

    /// Bearer token for HTTP authentication (http transport only)
    #[arg(long)]
    auth: Option<String>,
}

fn main() {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let args = Args::parse();

    info!("Initializing llama.cpp backend");

    let backend = llama_cpp_2::llama_backend::LlamaBackend::init()
        .expect("failed to initialize llama.cpp backend");

    let config = EngineConfig {
        n_gpu_layers: args.n_gpu_layers,
        n_ctx: args.n_ctx,
        n_batch: args.n_batch,
        n_threads: args.n_threads,
    };

    info!("Loading model from {}", args.model.display());

    let model = engine::load_model(&backend, &args.model, &config).expect("failed to load model");

    let default_params = GenerateParams {
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        seed: args.seed,
        repeat_penalty: args.repeat_penalty,
        prompt_suffix: args.prompt_suffix.clone(),
    };

    let model_display = args
        .model
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".into());

    info!(
        model = %model_display,
        tool = %args.tool_name,
        ctx_size = args.n_ctx,
        "llama-mcp-server ready"
    );

    match args.transport.as_str() {
        "stdio" => {
            let ctx = engine::create_context(&model, &backend, &config)
                .expect("failed to create context");
            let ctx = RefCell::new(ctx);
            mcp::run_stdio_loop(
                &model,
                &ctx,
                &default_params,
                &args.tool_name,
                &model_display,
            );
        }
        "http" => {
            let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
            rt.block_on(http::run_http_server(
                Arc::new(backend),
                Arc::new(model),
                config,
                default_params,
                args.tool_name,
                model_display,
                &args.host,
                args.port,
                args.auth,
            ));
        }
        other => {
            eprintln!("Unknown transport: {other}");
            std::process::exit(1);
        }
    }
}
