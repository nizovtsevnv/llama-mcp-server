use std::cell::RefCell;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Router;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use tracing::info;

use crate::engine::{self, EngineConfig, GenerateParams};

struct AppState {
    backend: Arc<LlamaBackend>,
    model: Arc<LlamaModel>,
    config: EngineConfig,
    default_params: GenerateParams,
    tool_name: String,
    model_display: String,
    auth_token: Option<String>,
    sessions: Mutex<HashSet<String>>,
}

#[allow(clippy::too_many_arguments)]
pub async fn run_http_server(
    backend: Arc<LlamaBackend>,
    model: Arc<LlamaModel>,
    config: EngineConfig,
    default_params: GenerateParams,
    tool_name: String,
    model_display: String,
    host: &str,
    port: u16,
    auth_token: Option<String>,
) {
    let state = Arc::new(AppState {
        backend,
        model,
        config,
        default_params,
        tool_name,
        model_display,
        auth_token,
        sessions: Mutex::new(HashSet::new()),
    });

    let app = Router::new()
        .route("/mcp", post(handle_post).delete(handle_delete))
        .with_state(state);

    let addr = format!("{host}:{port}");
    info!("HTTP server listening on {addr}");

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .expect("failed to bind HTTP listener");

    axum::serve(listener, app).await.expect("HTTP server error");
}

fn check_auth(state: &AppState, headers: &HeaderMap) -> Result<(), StatusCode> {
    let expected = match &state.auth_token {
        Some(t) => t,
        None => return Ok(()),
    };

    let auth = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    if let Some(token) = auth.strip_prefix("Bearer ") {
        if token == expected.as_str() {
            return Ok(());
        }
    }

    Err(StatusCode::UNAUTHORIZED)
}

fn get_session_id(headers: &HeaderMap) -> Option<String> {
    headers
        .get("mcp-session-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

async fn handle_post(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: String,
) -> Response {
    if let Err(status) = check_auth(&state, &headers) {
        return status.into_response();
    }

    // Check Content-Type
    let content_type = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if !content_type.contains("application/json") {
        return (
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "Content-Type must be application/json",
        )
            .into_response();
    }

    // Peek at the method to determine if this is an initialize request
    let is_initialize = serde_json::from_str::<serde_json::Value>(&body)
        .ok()
        .and_then(|v| v.get("method").and_then(|m| m.as_str()).map(String::from))
        .as_deref()
        == Some("initialize");

    if !is_initialize {
        // Require valid session ID for non-initialize requests
        let session_id = match get_session_id(&headers) {
            Some(id) => id,
            None => {
                return (StatusCode::BAD_REQUEST, "Missing Mcp-Session-Id header").into_response()
            }
        };

        let sessions = state.sessions.lock().expect("session lock poisoned");
        if !sessions.contains(&session_id) {
            return (StatusCode::BAD_REQUEST, "Invalid session ID").into_response();
        }
    }

    // Dispatch the request using spawn_blocking for LLM inference
    let backend = Arc::clone(&state.backend);
    let model = Arc::clone(&state.model);
    let config = EngineConfig {
        n_gpu_layers: state.config.n_gpu_layers,
        n_ctx: state.config.n_ctx,
        n_batch: state.config.n_batch,
        n_threads: state.config.n_threads,
    };
    let default_params = GenerateParams {
        max_tokens: state.default_params.max_tokens,
        temperature: state.default_params.temperature,
        top_p: state.default_params.top_p,
        top_k: state.default_params.top_k,
        seed: state.default_params.seed,
        repeat_penalty: state.default_params.repeat_penalty,
        prompt_suffix: state.default_params.prompt_suffix.clone(),
    };
    let tool_name = state.tool_name.clone();
    let model_display = state.model_display.clone();
    let request_body = body;

    let result = tokio::task::spawn_blocking(move || {
        let ctx =
            engine::create_context(&model, &backend, &config).expect("failed to create context");
        let ctx = RefCell::new(ctx);
        crate::mcp::dispatch_request(
            &request_body,
            &model,
            &ctx,
            &default_params,
            &tool_name,
            &model_display,
        )
    })
    .await
    .expect("dispatch task panicked");

    match result {
        Some(response_json) => {
            let mut builder = Response::builder()
                .status(StatusCode::OK)
                .header("content-type", "application/json");

            // If this was an initialize request, create a session
            if is_initialize {
                let session_id = uuid::Uuid::new_v4().to_string();
                state
                    .sessions
                    .lock()
                    .expect("session lock poisoned")
                    .insert(session_id.clone());
                builder = builder.header("mcp-session-id", &session_id);
            }

            builder
                .body(Body::from(response_json))
                .expect("valid response body")
        }
        None => {
            // Notification — no response body needed
            (StatusCode::ACCEPTED, "").into_response()
        }
    }
}

async fn handle_delete(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    if let Err(status) = check_auth(&state, &headers) {
        return status.into_response();
    }

    let session_id = match get_session_id(&headers) {
        Some(id) => id,
        None => return (StatusCode::BAD_REQUEST, "Missing Mcp-Session-Id header").into_response(),
    };

    let mut sessions = state.sessions.lock().expect("session lock poisoned");
    if sessions.remove(&session_id) {
        (StatusCode::OK, "Session terminated").into_response()
    } else {
        (StatusCode::NOT_FOUND, "Session not found").into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_auth_valid() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            "Bearer secret123".parse().expect("valid header"),
        );

        let result = check_auth_standalone(Some("secret123"), &headers);
        assert!(result.is_ok());
    }

    #[test]
    fn test_check_auth_invalid() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "authorization",
            "Bearer wrong".parse().expect("valid header"),
        );

        let result = check_auth_standalone(Some("secret123"), &headers);
        assert!(result.is_err());
    }

    #[test]
    fn test_check_auth_missing() {
        let headers = HeaderMap::new();

        let result = check_auth_standalone(Some("secret123"), &headers);
        assert!(result.is_err());
    }

    #[test]
    fn test_no_token_configured() {
        let headers = HeaderMap::new();

        let result = check_auth_standalone(None, &headers);
        assert!(result.is_ok());
    }

    #[test]
    fn test_session_lifecycle() {
        let sessions: Mutex<HashSet<String>> = Mutex::new(HashSet::new());

        // Add a session
        let session_id = uuid::Uuid::new_v4().to_string();
        sessions.lock().expect("lock").insert(session_id.clone());
        assert!(sessions.lock().expect("lock").contains(&session_id));

        // Remove the session
        assert!(sessions.lock().expect("lock").remove(&session_id));
        assert!(!sessions.lock().expect("lock").contains(&session_id));
    }

    #[test]
    fn test_missing_session_id() {
        let headers = HeaderMap::new();
        assert!(get_session_id(&headers).is_none());
    }

    #[test]
    fn test_get_session_id_present() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "mcp-session-id",
            "test-id-123".parse().expect("valid header"),
        );
        assert_eq!(get_session_id(&headers), Some("test-id-123".to_string()));
    }

    /// Standalone auth check that doesn't need AppState.
    fn check_auth_standalone(token: Option<&str>, headers: &HeaderMap) -> Result<(), StatusCode> {
        let expected = match token {
            Some(t) => t,
            None => return Ok(()),
        };

        let auth = headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");

        if let Some(bearer_token) = auth.strip_prefix("Bearer ") {
            if bearer_token == expected {
                return Ok(());
            }
        }

        Err(StatusCode::UNAUTHORIZED)
    }
}
