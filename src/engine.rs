use std::num::NonZeroU32;
use std::path::Path;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;

// ---------------------------------------------------------------------------
// Chat message (MCP-facing)
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

// ---------------------------------------------------------------------------
// Engine configuration
// ---------------------------------------------------------------------------

pub struct EngineConfig {
    pub n_gpu_layers: u32,
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_threads: i32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            n_gpu_layers: 99,
            n_ctx: 4096,
            n_batch: 2048,
            n_threads: 4,
        }
    }
}

// ---------------------------------------------------------------------------
// Generation parameters
// ---------------------------------------------------------------------------

pub struct GenerateParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub seed: u32,
    pub repeat_penalty: f32,
    /// Optional suffix appended to the system prompt (e.g. "/think" for Qwen3).
    pub prompt_suffix: Option<String>,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            max_tokens: 2048,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            seed: 42,
            repeat_penalty: 1.1,
            prompt_suffix: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

/// Load model from a GGUF file.
pub fn load_model(
    backend: &LlamaBackend,
    model_path: &Path,
    config: &EngineConfig,
) -> Result<LlamaModel, String> {
    let model_params = LlamaModelParams::default().with_n_gpu_layers(config.n_gpu_layers);

    let model = LlamaModel::load_from_file(backend, model_path, &model_params)
        .map_err(|e| format!("failed to load model: {e}"))?;

    tracing::info!(
        params = model.n_params(),
        layers = model.n_layer(),
        gpu_layers = config.n_gpu_layers,
        "model loaded"
    );

    Ok(model)
}

/// Create an inference context from a loaded model.
pub fn create_context<'a>(
    model: &'a LlamaModel,
    backend: &LlamaBackend,
    config: &EngineConfig,
) -> Result<LlamaContext<'a>, String> {
    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(config.n_ctx))
        .with_n_batch(config.n_batch)
        .with_n_threads(config.n_threads)
        .with_n_threads_batch(config.n_threads);

    model
        .new_context(backend, ctx_params)
        .map_err(|e| format!("failed to create context: {e}"))
}

// ---------------------------------------------------------------------------
// Prompt formatting
// ---------------------------------------------------------------------------

/// Format messages using the model's built-in chat template (from GGUF metadata).
///
/// When `prompt_suffix` is provided, appends it to the first system message
/// (or creates one). This is the generic mechanism for model-specific
/// directives like Qwen3's `/think` / `/no_think`.
pub fn format_prompt(
    model: &LlamaModel,
    messages: &[ChatMessage],
    prompt_suffix: Option<&str>,
) -> Result<String, String> {
    let effective: Vec<ChatMessage> = if let Some(suffix) = prompt_suffix {
        let mut adjusted = Vec::with_capacity(messages.len() + 1);
        if messages.first().is_some_and(|m| m.role == "system") {
            adjusted.push(ChatMessage {
                role: "system".into(),
                content: format!("{} {suffix}", messages[0].content),
            });
            adjusted.extend(messages[1..].iter().map(|m| ChatMessage {
                role: m.role.clone(),
                content: m.content.clone(),
            }));
        } else {
            adjusted.push(ChatMessage {
                role: "system".into(),
                content: suffix.into(),
            });
            adjusted.extend(messages.iter().map(|m| ChatMessage {
                role: m.role.clone(),
                content: m.content.clone(),
            }));
        }
        adjusted
    } else {
        messages.to_vec()
    };

    let chat_messages: Result<Vec<LlamaChatMessage>, _> = effective
        .iter()
        .map(|m| LlamaChatMessage::new(m.role.clone(), m.content.clone()))
        .collect();
    let chat_messages = chat_messages.map_err(|e| format!("invalid chat message: {e}"))?;

    let template = model
        .chat_template(None)
        .map_err(|e| format!("chat template not found in model: {e}"))?;

    model
        .apply_chat_template(&template, &chat_messages, true)
        .map_err(|e| format!("apply chat template: {e}"))
}

/// Strip `<think>...</think>` block from model output.
///
/// Safety net — some models emit thinking blocks even when not requested.
/// Cheap to run unconditionally.
pub fn strip_think_block(text: &str) -> String {
    let Some(start) = text.find("<think>") else {
        return text.to_string();
    };
    let Some(end) = text.find("</think>") else {
        // Unclosed think block — strip from <think> to end.
        return text[..start].to_string();
    };
    let mut result = String::with_capacity(text.len());
    result.push_str(&text[..start]);
    result.push_str(&text[end + "</think>".len()..]);
    result.trim_start().to_string()
}

// ---------------------------------------------------------------------------
// Token decoding
// ---------------------------------------------------------------------------

/// Decode a sequence of tokens back to text.
fn decode_tokens(model: &LlamaModel, tokens: &[LlamaToken]) -> Result<String, String> {
    let mut output = String::new();
    for &token in tokens {
        let piece = model_token_to_string(model, token)?;
        output.push_str(&piece);
    }
    Ok(output)
}

/// Convert a single token to its string representation.
fn model_token_to_string(model: &LlamaModel, token: LlamaToken) -> Result<String, String> {
    let n_vocab = model.n_vocab();
    if token.0 < 0 || token.0 >= n_vocab {
        return Err(format!("token id {} out of range [0, {n_vocab})", token.0));
    }
    #[allow(deprecated)]
    model
        .token_to_str(token, llama_cpp_2::model::Special::Plaintext)
        .map_err(|e| format!("token_to_str: {e}"))
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

pub fn generate(
    model: &LlamaModel,
    ctx: &mut LlamaContext<'_>,
    messages: &[ChatMessage],
    params: &GenerateParams,
) -> Result<String, String> {
    let prompt = format_prompt(model, messages, params.prompt_suffix.as_deref())?;

    // Tokenize.
    let tokens = model
        .str_to_token(&prompt, AddBos::Always)
        .map_err(|e| format!("tokenize: {e}"))?;

    if tokens.is_empty() {
        return Err("empty prompt after tokenization".into());
    }

    let n_ctx = ctx.n_ctx() as usize;
    if tokens.len() >= n_ctx {
        return Err(format!(
            "prompt ({} tokens) exceeds context size ({n_ctx})",
            tokens.len()
        ));
    }

    tracing::debug!(prompt_tokens = tokens.len(), "starting generation");

    // Clear KV cache from any previous generation.
    ctx.clear_kv_cache();

    // Build sampler chain.
    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::penalties(
            64, // penalty_last_n: look back 64 tokens for repeat penalty
            params.repeat_penalty,
            0.0,
            0.0,
        ),
        LlamaSampler::top_k(params.top_k),
        LlamaSampler::top_p(params.top_p, 1),
        LlamaSampler::temp(params.temperature),
        LlamaSampler::dist(params.seed),
    ]);

    // Feed prompt tokens in chunks of n_batch.
    let n_batch = ctx.n_batch() as usize;
    let mut batch = LlamaBatch::new(n_batch.max(1), 1);

    for chunk_start in (0..tokens.len()).step_by(n_batch) {
        let chunk_end = (chunk_start + n_batch).min(tokens.len());
        batch.clear();
        for i in chunk_start..chunk_end {
            let is_last = i == tokens.len() - 1;
            batch
                .add(tokens[i], i as i32, &[0], is_last)
                .map_err(|e| format!("batch add: {e}"))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| format!("decode prompt: {e}"))?;
    }

    // Autoregressive generation.
    let eos = model.token_eos();
    let mut generated: Vec<LlamaToken> = Vec::new();
    let mut n_cur = tokens.len();

    for _ in 0..params.max_tokens {
        let token = sampler.sample(ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if token == eos || model.is_eog_token(token) {
            break;
        }

        generated.push(token);
        n_cur += 1;

        if n_cur >= n_ctx {
            tracing::warn!("context window full, stopping generation");
            break;
        }

        batch.clear();
        batch
            .add(token, (n_cur - 1) as i32, &[0], true)
            .map_err(|e| format!("batch add: {e}"))?;

        ctx.decode(&mut batch).map_err(|e| format!("decode: {e}"))?;
    }

    tracing::debug!(generated_tokens = generated.len(), "generation complete");

    let text = decode_tokens(model, &generated)?;

    // Strip thinking blocks unconditionally — cheap safety net.
    Ok(strip_think_block(&text))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_params_defaults() {
        let p = GenerateParams::default();
        assert!((p.temperature - 0.7).abs() < f32::EPSILON);
        assert_eq!(p.max_tokens, 2048);
        assert_eq!(p.top_k, 40);
    }

    #[test]
    fn engine_config_defaults() {
        let c = EngineConfig::default();
        assert_eq!(c.n_gpu_layers, 99);
        assert_eq!(c.n_ctx, 4096);
    }

    #[test]
    fn strip_think_block_removes_think() {
        assert_eq!(
            strip_think_block("<think>reasoning</think>answer"),
            "answer"
        );
    }

    #[test]
    fn strip_think_block_no_think() {
        assert_eq!(strip_think_block("just answer"), "just answer");
    }

    #[test]
    fn strip_think_block_empty_think() {
        assert_eq!(strip_think_block("<think></think>answer"), "answer");
    }
}
