use std::cell::RefCell;
use std::io::{self, BufRead, Write};

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::model::LlamaModel;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tracing::{debug, error, warn};

use crate::engine::{self, ChatMessage, GenerateParams};

#[derive(Deserialize)]
pub(crate) struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    pub(crate) id: Option<Value>,
    pub(crate) method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Serialize)]
pub(crate) struct JsonRpcResponse {
    jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<Value>,
}

/// Process a JSON-RPC request and return a response.
/// Returns None for notifications that require no response.
pub fn dispatch_request(
    request_json: &str,
    model: &LlamaModel,
    ctx: &RefCell<LlamaContext<'_>>,
    default_params: &GenerateParams,
    tool_name: &str,
    model_display: &str,
) -> Option<String> {
    let request: JsonRpcRequest = match serde_json::from_str(request_json) {
        Ok(r) => r,
        Err(e) => {
            warn!("Invalid JSON-RPC request: {e}");
            let resp = JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: None,
                result: None,
                error: Some(json!({"code": -32700, "message": "Parse error"})),
            };
            return serde_json::to_string(&resp).ok();
        }
    };

    debug!("Received method: {}", request.method);

    if request.method.starts_with("notifications/") {
        return None;
    }

    let response = match request.method.as_str() {
        "initialize" => handle_initialize(&request),
        "tools/list" => handle_tools_list(&request, tool_name, model_display),
        "tools/call" => handle_tools_call(&request, model, ctx, default_params, tool_name),
        _ => JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id,
            result: None,
            error: Some(json!({"code": -32601, "message": "Method not found"})),
        },
    };

    match serde_json::to_string(&response) {
        Ok(json) => Some(json),
        Err(e) => {
            error!("Failed to serialize response: {e}");
            None
        }
    }
}

pub fn run_stdio_loop(
    model: &LlamaModel,
    ctx: &RefCell<LlamaContext<'_>>,
    default_params: &GenerateParams,
    tool_name: &str,
    model_display: &str,
) {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdout = stdout.lock();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                error!("stdin read error: {e}");
                break;
            }
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if let Some(response) = dispatch_request(
            trimmed,
            model,
            ctx,
            default_params,
            tool_name,
            model_display,
        ) {
            let _ = writeln!(stdout, "{response}");
            let _ = stdout.flush();
        }
    }
}

fn handle_initialize(request: &JsonRpcRequest) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        id: request.id.clone(),
        result: Some(json!({
            "protocolVersion": "2024-11-05",
            "capabilities": { "tools": {} },
            "serverInfo": {
                "name": "llama-mcp-server",
                "version": env!("CARGO_PKG_VERSION")
            }
        })),
        error: None,
    }
}

fn handle_tools_list(
    request: &JsonRpcRequest,
    tool_name: &str,
    model_display: &str,
) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0".to_string(),
        id: request.id.clone(),
        result: Some(json!({
            "tools": [{
                "name": tool_name,
                "description": format!(
                    "Send a message to local LLM (model: {model_display}) and return the response"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text message to send (single user message)"
                        },
                        "messages": {
                            "type": "array",
                            "description": "Array of {role, content} messages for multi-turn conversation",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {"type": "string"},
                                    "content": {"type": "string"}
                                }
                            }
                        },
                        "temperature": {
                            "type": "number",
                            "description": "Sampling temperature (0.0 = greedy)"
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Maximum tokens to generate"
                        },
                        "top_p": {
                            "type": "number",
                            "description": "Top-p (nucleus) sampling threshold"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Top-k sampling (number of candidates)"
                        },
                        "prompt_suffix": {
                            "type": "string",
                            "description": "Suffix appended to the system prompt (e.g. \"/think\" for Qwen3)"
                        }
                    }
                }
            }]
        })),
        error: None,
    }
}

fn handle_tools_call(
    request: &JsonRpcRequest,
    model: &LlamaModel,
    ctx: &RefCell<LlamaContext<'_>>,
    default_params: &GenerateParams,
    expected_tool_name: &str,
) -> JsonRpcResponse {
    let tool_name = request
        .params
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if tool_name != expected_tool_name {
        return JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id.clone(),
            result: Some(mcp_error(&format!("Unknown tool: {tool_name}"))),
            error: None,
        };
    }

    let arguments = match request.params.get("arguments").and_then(|v| v.as_object()) {
        Some(args) => args,
        None => {
            return JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id.clone(),
                result: Some(mcp_error("Missing or invalid arguments")),
                error: None,
            };
        }
    };

    let messages = match parse_messages(arguments) {
        Ok(m) => m,
        Err(e) => {
            return JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id.clone(),
                result: Some(mcp_error(&e)),
                error: None,
            };
        }
    };

    let gen_params = parse_generate_params(arguments, default_params);

    match engine::generate(model, &mut ctx.borrow_mut(), &messages, &gen_params) {
        Ok(text) => JsonRpcResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id.clone(),
            result: Some(json!({
                "content": [{"type": "text", "text": text}]
            })),
            error: None,
        },
        Err(e) => {
            error!("Generation failed: {e}");
            JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id.clone(),
                result: Some(mcp_error(&e)),
                error: None,
            }
        }
    }
}

pub(crate) fn parse_messages(
    arguments: &serde_json::Map<String, Value>,
) -> Result<Vec<ChatMessage>, String> {
    if let Some(msgs_val) = arguments.get("messages") {
        let arr = msgs_val.as_array().ok_or("messages must be an array")?;
        if arr.is_empty() {
            return Err("messages array must not be empty".into());
        }
        arr.iter()
            .map(|m| {
                let role = m
                    .get("role")
                    .and_then(|v| v.as_str())
                    .unwrap_or("user")
                    .to_string();
                let content = m
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                Ok(ChatMessage { role, content })
            })
            .collect()
    } else if let Some(text) = arguments.get("text").and_then(|v| v.as_str()) {
        Ok(vec![ChatMessage {
            role: "user".into(),
            content: text.to_string(),
        }])
    } else {
        Err("missing required argument: text or messages".into())
    }
}

pub(crate) fn parse_generate_params(
    arguments: &serde_json::Map<String, Value>,
    defaults: &GenerateParams,
) -> GenerateParams {
    GenerateParams {
        max_tokens: arguments
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(defaults.max_tokens),
        temperature: arguments
            .get("temperature")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(defaults.temperature),
        top_p: arguments
            .get("top_p")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(defaults.top_p),
        top_k: arguments
            .get("top_k")
            .and_then(|v| v.as_i64())
            .map(|v| v as i32)
            .unwrap_or(defaults.top_k),
        seed: defaults.seed,
        repeat_penalty: defaults.repeat_penalty,
        prompt_suffix: arguments
            .get("prompt_suffix")
            .and_then(|v| v.as_str())
            .map(String::from)
            .or_else(|| defaults.prompt_suffix.clone()),
    }
}

pub(crate) fn mcp_error(message: &str) -> Value {
    json!({
        "content": [{"type": "text", "text": message}],
        "isError": true
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_parse_error() {
        let result = parse_and_dispatch("not valid json {{{");
        assert!(result.is_some());
        let resp: Value = serde_json::from_str(&result.unwrap()).expect("valid json");
        assert_eq!(resp["error"]["code"], -32700);
    }

    #[test]
    fn test_dispatch_method_not_found() {
        let input = r#"{"jsonrpc":"2.0","id":1,"method":"unknown/method"}"#;
        let result = parse_and_dispatch(input);
        assert!(result.is_some());
        let resp: Value = serde_json::from_str(&result.unwrap()).expect("valid json");
        assert_eq!(resp["error"]["code"], -32601);
        assert_eq!(resp["id"], 1);
    }

    #[test]
    fn test_dispatch_notification_returns_none() {
        let input = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
        let result = parse_and_dispatch(input);
        assert!(result.is_none());
    }

    #[test]
    fn test_dispatch_initialize() {
        let input = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
        let result = parse_and_dispatch(input);
        assert!(result.is_some());
        let resp: Value = serde_json::from_str(&result.unwrap()).expect("valid json");
        assert_eq!(resp["id"], 1);
        assert_eq!(resp["result"]["protocolVersion"], "2024-11-05");
        assert_eq!(resp["result"]["serverInfo"]["name"], "llama-mcp-server");
    }

    #[test]
    fn test_dispatch_tools_list() {
        let input = r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#;
        let result = parse_and_dispatch(input);
        assert!(result.is_some());
        let resp: Value = serde_json::from_str(&result.unwrap()).expect("valid json");
        let tools = resp["result"]["tools"].as_array().expect("tools array");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "chat");
    }

    #[test]
    fn parse_messages_from_text() {
        let mut args = serde_json::Map::new();
        args.insert("text".into(), Value::String("hello".into()));
        let msgs = parse_messages(&args).expect("parse ok");
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, "user");
        assert_eq!(msgs[0].content, "hello");
    }

    #[test]
    fn parse_messages_from_array() {
        let mut args = serde_json::Map::new();
        args.insert(
            "messages".into(),
            json!([
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"}
            ]),
        );
        let msgs = parse_messages(&args).expect("parse ok");
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[1].content, "Hi");
    }

    #[test]
    fn parse_messages_empty_array() {
        let mut args = serde_json::Map::new();
        args.insert("messages".into(), json!([]));
        assert!(parse_messages(&args).is_err());
    }

    #[test]
    fn parse_messages_missing() {
        let args = serde_json::Map::new();
        assert!(parse_messages(&args).is_err());
    }

    #[test]
    fn parse_generate_params_defaults() {
        let args = serde_json::Map::new();
        let defaults = GenerateParams::default();
        let p = parse_generate_params(&args, &defaults);
        assert!((p.temperature - defaults.temperature).abs() < f32::EPSILON);
        assert_eq!(p.max_tokens, defaults.max_tokens);
    }

    #[test]
    fn parse_generate_params_overrides() {
        let mut args = serde_json::Map::new();
        args.insert("temperature".into(), json!(0.3));
        args.insert("max_tokens".into(), json!(100));
        args.insert("top_k".into(), json!(20));
        let defaults = GenerateParams::default();
        let p = parse_generate_params(&args, &defaults);
        assert!((p.temperature - 0.3).abs() < f32::EPSILON);
        assert_eq!(p.max_tokens, 100);
        assert_eq!(p.top_k, 20);
        assert!((p.top_p - defaults.top_p).abs() < f32::EPSILON);
    }

    #[test]
    fn parse_generate_params_prompt_suffix_override() {
        let mut args = serde_json::Map::new();
        args.insert("prompt_suffix".into(), json!("/think"));
        let defaults = GenerateParams::default();
        assert!(defaults.prompt_suffix.is_none());
        let p = parse_generate_params(&args, &defaults);
        assert_eq!(p.prompt_suffix.as_deref(), Some("/think"));
    }

    #[test]
    fn parse_generate_params_prompt_suffix_default() {
        let args = serde_json::Map::new();
        let defaults = GenerateParams::default();
        let p = parse_generate_params(&args, &defaults);
        assert_eq!(p.prompt_suffix, defaults.prompt_suffix);
    }

    #[test]
    fn parse_generate_params_prompt_suffix_from_cli_default() {
        let args = serde_json::Map::new();
        let defaults = GenerateParams {
            prompt_suffix: Some("/no_think".into()),
            ..GenerateParams::default()
        };
        let p = parse_generate_params(&args, &defaults);
        assert_eq!(p.prompt_suffix.as_deref(), Some("/no_think"));
    }

    /// Helper that dispatches without needing a LlamaModel/LlamaContext.
    /// Only works for methods that don't use tools/call with generation.
    fn parse_and_dispatch(request_json: &str) -> Option<String> {
        let request: JsonRpcRequest = match serde_json::from_str(request_json) {
            Ok(r) => r,
            Err(e) => {
                warn!("Invalid JSON-RPC request: {e}");
                let resp = JsonRpcResponse {
                    jsonrpc: "2.0".to_string(),
                    id: None,
                    result: None,
                    error: Some(json!({"code": -32700, "message": "Parse error"})),
                };
                return serde_json::to_string(&resp).ok();
            }
        };

        if request.method.starts_with("notifications/") {
            return None;
        }

        let response = match request.method.as_str() {
            "initialize" => handle_initialize(&request),
            "tools/list" => handle_tools_list(&request, "chat", "test-model.gguf"),
            _ => JsonRpcResponse {
                jsonrpc: "2.0".to_string(),
                id: request.id,
                result: None,
                error: Some(json!({"code": -32601, "message": "Method not found"})),
            },
        };

        serde_json::to_string(&response).ok()
    }
}
