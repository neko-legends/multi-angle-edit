//! Multi-Angle Edit — Tauri backend.
//!
//! Re-renders an input image from a new camera angle using Qwen-Image-Edit plus
//! the Multiple-Angles LoRA, driven by an interactive 3D angle picker in the
//! frontend. Runs locally on the user's own GPU.
//!
//! Architecture mirrors the rest of the Neko suite: a Python worker is bundled as
//! a resource, an app-managed virtual environment is created on first use, and a
//! shared NekoLegendsAI store dedups PyTorch wheels + Hugging Face models across
//! every app. See `neko_store.rs`.

use serde::{Deserialize, Serialize};
use std::{
    env,
    fs,
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::{AtomicBool, Ordering},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tauri::{AppHandle, Emitter, Manager, State};

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

mod neko_store;
use neko_store::NekoStore;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

const EMBEDDED_WORKER: &str = include_str!("../../python/multi_angle_worker.py");
const NEKO_APP_SLUG: &str = "multiangleedit";

const DEFAULT_AGENT_API_PORT: u16 = 17339;
const AGENT_APP_ID: &str = "multi-angle-edit";
const AGENT_APP_NAME: &str = "Multi-Angle Edit";
const AGENT_API_BIND_ADDRESS: &str = "127.0.0.1";
const AGENT_API_REGISTRY_FILE: &str = "agent-api-registry.json";

// ---------------------------------------------------------------------------
// Types shared with the frontend
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GpuDevice {
    /// CUDA device index, or -1 for CPU.
    index: i32,
    name: String,
    /// Total VRAM in megabytes, when known.
    vram_mb: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AngleRequest {
    /// Absolute path to the source image on disk.
    image_path: String,
    /// Prompt string produced by the 3D angle picker, e.g.
    /// "<sks> right side view high-angle shot close-up".
    angle_prompt: String,
    /// CUDA device index to run on (-1 = CPU).
    device_index: i32,
    /// Optional extra natural-language guidance appended to the angle prompt.
    extra_prompt: Option<String>,
    /// Optional output directory override. Defaults next to the app.
    output_dir: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WorkerJob {
    image_path: String,
    angle_prompt: String,
    extra_prompt: String,
    device_index: i32,
    env_dir: String,
    models_dir: String,
    output_path: String,
}

// ---------------------------------------------------------------------------
// Agent API types
// ---------------------------------------------------------------------------

struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AgentServerStatus {
    enabled: bool,
    port: u16,
    url: String,
    openapi_url: String,
    busy: bool,
    active_job_id: Option<String>,
    message: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
struct AgentApiRegistryEntry {
    app_id: String,
    app_name: String,
    default_port: u16,
    bind_address: String,
    port: u16,
    enabled: bool,
    url: String,
    openapi_url: String,
    busy: bool,
    active_job_id: Option<String>,
    last_seen: Option<String>,
    note: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct AgentApiRegistry {
    updated_at: String,
    apps: Vec<AgentApiRegistryEntry>,
}

#[derive(Clone, Default)]
struct AgentServerState {
    inner: Arc<Mutex<AgentServerControl>>,
}

#[derive(Default)]
struct AgentServerControl {
    enabled: bool,
    port: u16,
    stop: Option<Arc<AtomicBool>>,
}

impl AgentServerControl {
    fn port(&self) -> u16 {
        if self.port == 0 {
            read_registered_agent_api_port().unwrap_or(DEFAULT_AGENT_API_PORT)
        } else {
            self.port
        }
    }
}

fn app_data_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|error| format!("Unable to resolve app data dir: {error}"))?;
    fs::create_dir_all(&dir).map_err(|error| format!("Unable to create app data dir: {error}"))?;
    Ok(dir)
}

/// Resolve the shared NekoLegendsAI store, defaulting its location to the
/// executable's drive so portable installs on `D:` land the store at
/// `D:\NekoLegendsAI` next to the app.
fn neko_store(app: &AppHandle) -> Result<NekoStore, String> {
    let app_data = app_data_dir(app)?;
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf));
    Ok(NekoStore::resolve_with_volume_hint(
        NEKO_APP_SLUG,
        &app_data,
        exe_dir.as_deref(),
    ))
}

fn default_output_dir(app: &AppHandle) -> Result<PathBuf, String> {
    // Prefer an `output` folder next to the executable; fall back to app data.
    let base = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .map(|p| p.join("output"))
        .unwrap_or(app_data_dir(app)?.join("output"));
    fs::create_dir_all(&base).map_err(|error| format!("Unable to create output dir: {error}"))?;
    Ok(base)
}

fn hide_window(command: &mut Command) {
    #[cfg(target_os = "windows")]
    {
        command.creation_flags(CREATE_NO_WINDOW);
    }
}

/// Locate the bundled Python worker, materializing the embedded copy under app
/// data so it works both in dev and from an installed bundle.
fn worker_path(app: &AppHandle) -> Result<PathBuf, String> {
    // Bundled resource first.
    if let Ok(resource) = app
        .path()
        .resolve("python/multi_angle_worker.py", tauri::path::BaseDirectory::Resource)
    {
        if resource.exists() {
            return Ok(resource);
        }
    }
    // Fallback: write the embedded copy under app data.
    let dir = app_data_dir(app)?.join("worker");
    fs::create_dir_all(&dir).map_err(|error| format!("Unable to create worker dir: {error}"))?;
    let path = dir.join("multi_angle_worker.py");
    fs::write(&path, EMBEDDED_WORKER)
        .map_err(|error| format!("Unable to materialize worker: {error}"))?;
    Ok(path)
}

/// Find a usable system Python (3.10–3.12) to bootstrap the venv with.
fn find_python() -> Option<String> {
    let candidates: Vec<&str> = if cfg!(target_os = "windows") {
        vec!["py -3.11", "py -3.12", "py -3.10", "python", "python3"]
    } else {
        vec!["python3.11", "python3.12", "python3.10", "python3", "python"]
    };
    for candidate in candidates {
        if python_ok(candidate) {
            return Some(candidate.to_string());
        }
    }
    None
}

fn python_ok(spec: &str) -> bool {
    let mut parts = spec.split_whitespace();
    let Some(exe) = parts.next() else {
        return false;
    };
    let mut command = Command::new(exe);
    for arg in parts {
        command.arg(arg);
    }
    command.arg("--version");
    hide_window(&mut command);
    match command.output() {
        Ok(output) => {
            let text = format!(
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            // Accept Python 3.10, 3.11, 3.12.
            text.contains("Python 3.10")
                || text.contains("Python 3.11")
                || text.contains("Python 3.12")
        }
        Err(_) => false,
    }
}

fn command_from_python_spec(spec: &str) -> Command {
    let mut parts = spec.split_whitespace();
    let exe = parts.next().unwrap_or("python");
    let mut command = Command::new(exe);
    for arg in parts {
        command.arg(arg);
    }
    hide_window(&mut command);
    command
}

// ---------------------------------------------------------------------------
// GPU detection
// ---------------------------------------------------------------------------

/// Enumerate CUDA GPUs via `nvidia-smi`. Always includes a CPU fallback entry.
#[tauri::command]
fn list_gpus() -> Vec<GpuDevice> {
    let mut devices = Vec::new();
    let mut command = Command::new("nvidia-smi");
    command.args([
        "--query-gpu=index,name,memory.total",
        "--format=csv,noheader,nounits",
    ]);
    hide_window(&mut command);
    if let Ok(output) = command.output() {
        if output.status.success() {
            let text = String::from_utf8_lossy(&output.stdout);
            for line in text.lines() {
                let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if parts.len() >= 2 {
                    let index = parts[0].parse::<i32>().unwrap_or(0);
                    let name = parts[1].to_string();
                    let vram_mb = parts.get(2).and_then(|v| v.parse::<u64>().ok());
                    devices.push(GpuDevice {
                        index,
                        name,
                        vram_mb,
                    });
                }
            }
        }
    }
    devices.push(GpuDevice {
        index: -1,
        name: "CPU (slow)".to_string(),
        vram_mb: None,
    });
    devices
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

#[tauri::command]
fn generate_angle(app: AppHandle, request: AngleRequest) -> Result<String, String> {
    run_angle_generation(&app, request)
}

fn run_angle_generation(app: &AppHandle, request: AngleRequest) -> Result<String, String> {
    let python = find_python().ok_or_else(|| {
        "Python 3.10, 3.11, or 3.12 was not found on PATH. Install it and try again.".to_string()
    })?;
    let worker = worker_path(app)?;
    let store = neko_store(app)?;

    let output_dir = match &request.output_dir {
        Some(dir) if !dir.trim().is_empty() => {
            let p = PathBuf::from(dir);
            fs::create_dir_all(&p)
                .map_err(|error| format!("Unable to create output dir: {error}"))?;
            p
        }
        _ => default_output_dir(app)?,
    };

    let stem = Path::new(&request.image_path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "image".to_string());
    let job_id = uuid::Uuid::new_v4().to_string();
    let short = &job_id[..8];
    let output_path = output_dir.join(format!("{stem}_angle_{short}.png"));

    let job = WorkerJob {
        image_path: request.image_path.clone(),
        angle_prompt: request.angle_prompt.clone(),
        extra_prompt: request.extra_prompt.clone().unwrap_or_default(),
        device_index: request.device_index,
        env_dir: store.env_dir.display().to_string(),
        models_dir: store.models_dir.display().to_string(),
        output_path: output_path.display().to_string(),
    };

    let jobs_dir = app_data_dir(app)?.join("jobs");
    fs::create_dir_all(&jobs_dir).map_err(|error| format!("Unable to create jobs dir: {error}"))?;
    let job_path = jobs_dir.join(format!("{job_id}.json"));
    fs::write(
        &job_path,
        serde_json::to_string_pretty(&job)
            .map_err(|error| format!("Unable to serialize job: {error}"))?,
    )
    .map_err(|error| format!("Unable to write job: {error}"))?;

    let mut command = command_from_python_spec(&python);
    store.apply_env(&mut command);
    command
        .arg(&worker)
        .arg("--job")
        .arg(&job_path)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command
        .spawn()
        .map_err(|error| format!("Unable to start Python worker: {error}"))?;

    // Stream stdout JSON events to the frontend.
    if let Some(stdout) = child.stdout.take() {
        let app_evt = app.clone();
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines().map_while(Result::ok) {
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) {
                    let _ = app_evt.emit("angle-worker-event", value);
                } else {
                    let _ = app_evt.emit(
                        "angle-worker-event",
                        serde_json::json!({ "type": "setup", "message": line }),
                    );
                }
            }
        });
    }
    if let Some(stderr) = child.stderr.take() {
        let app_evt = app.clone();
        thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines().map_while(Result::ok) {
                let _ = app_evt.emit(
                    "angle-worker-event",
                    serde_json::json!({ "type": "setup", "message": line }),
                );
            }
        });
    }

    let status = child
        .wait()
        .map_err(|error| format!("Worker failed: {error}"))?;
    if !status.success() {
        return Err("Angle generation failed. See the log for details.".to_string());
    }
    if !output_path.exists() {
        return Err("Worker finished but produced no output image.".to_string());
    }
    Ok(output_path.display().to_string())
}

/// Report where the shared store resolved to, for the UI to display.
#[tauri::command]
fn store_info(app: AppHandle) -> Result<serde_json::Value, String> {
    store_info_value(&app)
}

fn store_info_value(app: &AppHandle) -> Result<serde_json::Value, String> {
    let store = neko_store(app)?;
    Ok(serde_json::json!({
        "root": store.root.display().to_string(),
        "shared": store.shared,
        "envDir": store.env_dir.display().to_string(),
        "modelsDir": store.models_dir.display().to_string(),
    }))
}

#[tauri::command]
fn open_path(app: AppHandle, path: String) -> Result<(), String> {
    use tauri_plugin_opener::OpenerExt;
    app.opener()
        .open_path(path, None::<&str>)
        .map_err(|error| format!("Unable to open path: {error}"))
}

// ---------------------------------------------------------------------------
// Agent API
// ---------------------------------------------------------------------------

fn validate_agent_api_port(port: Option<u16>) -> Result<u16, String> {
    let port = port
        .or_else(read_registered_agent_api_port)
        .unwrap_or(DEFAULT_AGENT_API_PORT);
    if port == 0 {
        return Err("Agent API port must be between 1 and 65535.".to_string());
    }
    Ok(port)
}

fn agent_api_url(port: u16) -> String {
    format!("http://{AGENT_API_BIND_ADDRESS}:{port}")
}

fn timestamp_string() -> String {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs().to_string())
        .unwrap_or_else(|_| "0".to_string())
}

fn shared_neko_legends_dir() -> Option<PathBuf> {
    let base = if cfg!(target_os = "windows") {
        env::var_os("APPDATA")
            .map(PathBuf::from)
            .or_else(|| env::var_os("USERPROFILE").map(PathBuf::from))
    } else if cfg!(target_os = "macos") {
        env::var_os("HOME")
            .map(PathBuf::from)
            .map(|home| home.join("Library").join("Application Support"))
    } else {
        env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
    }?;
    Some(base.join("NekoLegends"))
}

fn agent_api_registry_path() -> Option<PathBuf> {
    Some(shared_neko_legends_dir()?.join(AGENT_API_REGISTRY_FILE))
}

fn read_agent_api_registry() -> AgentApiRegistry {
    let updated_at = timestamp_string();
    let Some(path) = agent_api_registry_path() else {
        return AgentApiRegistry {
            updated_at,
            apps: Vec::new(),
        };
    };
    fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or(AgentApiRegistry {
            updated_at,
            apps: Vec::new(),
        })
}

fn read_registered_agent_api_port() -> Option<u16> {
    read_agent_api_registry()
        .apps
        .into_iter()
        .find(|entry| entry.app_id == AGENT_APP_ID)
        .map(|entry| entry.port)
        .filter(|port| *port > 0)
}

fn publish_agent_api_status(status: &AgentServerStatus) {
    let Some(path) = agent_api_registry_path() else {
        return;
    };
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let mut registry = read_agent_api_registry();
    let updated_at = timestamp_string();
    let entry = AgentApiRegistryEntry {
        app_id: AGENT_APP_ID.to_string(),
        app_name: AGENT_APP_NAME.to_string(),
        default_port: DEFAULT_AGENT_API_PORT,
        bind_address: AGENT_API_BIND_ADDRESS.to_string(),
        port: status.port,
        enabled: status.enabled,
        url: status.url.clone(),
        openapi_url: status.openapi_url.clone(),
        busy: status.busy,
        active_job_id: status.active_job_id.clone(),
        last_seen: Some(updated_at.clone()),
        note: Some("Local Agent API.".to_string()),
    };
    if let Some(existing) = registry
        .apps
        .iter_mut()
        .find(|entry| entry.app_id == AGENT_APP_ID)
    {
        *existing = entry;
    } else {
        registry.apps.push(entry);
    }
    registry.updated_at = updated_at;
    if let Ok(raw) = serde_json::to_string_pretty(&registry) {
        let _ = fs::write(path, raw);
    }
}

fn agent_status_from(agent_state: &AgentServerState) -> Result<AgentServerStatus, String> {
    let (enabled, port) = {
        let control = agent_state
            .inner
            .lock()
            .map_err(|_| "Unable to lock agent server state.".to_string())?;
        (control.enabled, control.port())
    };
    let status = AgentServerStatus {
        enabled,
        port,
        url: agent_api_url(port),
        openapi_url: format!("{}/openapi.json", agent_api_url(port)),
        busy: false,
        active_job_id: None,
        message: if enabled {
            "Agent API is enabled.".to_string()
        } else {
            "Agent API is off.".to_string()
        },
    };
    publish_agent_api_status(&status);
    Ok(status)
}

fn read_http_request(stream: &mut TcpStream) -> Result<HttpRequest, String> {
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .map_err(|error| format!("Unable to set read timeout: {error}"))?;
    let mut data = Vec::new();
    let mut buffer = [0_u8; 4096];
    let mut expected_len: Option<usize> = None;

    loop {
        let bytes_read = stream
            .read(&mut buffer)
            .map_err(|error| format!("Unable to read agent request: {error}"))?;
        if bytes_read == 0 {
            break;
        }
        data.extend_from_slice(&buffer[..bytes_read]);
        if let Some(header_end) = find_header_end(&data) {
            if expected_len.is_none() {
                let headers = String::from_utf8_lossy(&data[..header_end]);
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        if name.eq_ignore_ascii_case("content-length") {
                            value.trim().parse::<usize>().ok()
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);
                expected_len = Some(header_end + 4 + content_length);
            }
            if expected_len.is_some_and(|len| data.len() >= len) {
                break;
            }
        }
        if data.len() > 2 * 1024 * 1024 {
            return Err("Agent request is too large.".to_string());
        }
    }

    let header_end = find_header_end(&data).ok_or_else(|| "Invalid HTTP request.".to_string())?;
    let headers = String::from_utf8_lossy(&data[..header_end]);
    let mut lines = headers.lines();
    let request_line = lines
        .next()
        .ok_or_else(|| "Invalid HTTP request line.".to_string())?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let raw_path = parts.next().unwrap_or("/").to_string();
    let path = raw_path.split('?').next().unwrap_or("/").to_string();
    let body_start = header_end + 4;
    let body = if body_start <= data.len() {
        data[body_start..].to_vec()
    } else {
        Vec::new()
    };
    Ok(HttpRequest { method, path, body })
}

fn find_header_end(data: &[u8]) -> Option<usize> {
    data.windows(4).position(|window| window == b"\r\n\r\n")
}

fn write_json_response(
    stream: &mut TcpStream,
    status: &str,
    payload: serde_json::Value,
) -> Result<(), String> {
    let body = serde_json::to_vec(&payload)
        .map_err(|error| format!("Unable to serialize agent response: {error}"))?;
    let headers = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: content-type\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream
        .write_all(headers.as_bytes())
        .and_then(|_| stream.write_all(&body))
        .map_err(|error| format!("Unable to write agent response: {error}"))
}

fn write_empty_response(stream: &mut TcpStream, status: &str) -> Result<(), String> {
    let headers = format!(
        "HTTP/1.1 {status}\r\nContent-Length: 0\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: content-type\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\nConnection: close\r\n\r\n"
    );
    stream
        .write_all(headers.as_bytes())
        .map_err(|error| format!("Unable to write agent response: {error}"))
}

fn agent_openapi(port: u16) -> serde_json::Value {
    serde_json::json!({
        "openapi": "3.1.0",
        "info": {
            "title": "Multi-Angle Edit Agent API",
            "version": env!("CARGO_PKG_VERSION")
        },
        "servers": [{ "url": agent_api_url(port) }],
        "paths": {
            "/health": { "get": { "summary": "Check API status" } },
            "/openapi.json": { "get": { "summary": "Fetch this OpenAPI document" } },
            "/gpus": { "get": { "summary": "List available GPUs" } },
            "/store": { "get": { "summary": "Shared NekoLegends store info" } },
            "/generate": {
                "post": {
                    "summary": "Run a multi-angle edit",
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["imagePath", "anglePrompt", "deviceIndex"],
                                    "properties": {
                                        "imagePath": { "type": "string" },
                                        "anglePrompt": { "type": "string" },
                                        "deviceIndex": { "type": "integer" },
                                        "extraPrompt": { "type": "string" },
                                        "outputDir": { "type": "string" }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    })
}

fn handle_agent_route(
    request: HttpRequest,
    app: &AppHandle,
    agent_state: &AgentServerState,
) -> Result<serde_json::Value, String> {
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/health") => {
            let status = agent_status_from(agent_state)?;
            Ok(serde_json::json!({
                "ok": true,
                "app": "Multi-Angle Edit",
                "enabled": status.enabled,
                "port": status.port
            }))
        }
        ("GET", "/openapi.json") => Ok(agent_openapi(agent_status_from(agent_state)?.port)),
        ("GET", "/gpus") => {
            serde_json::to_value(list_gpus()).map_err(|error| error.to_string())
        }
        ("GET", "/store") => store_info_value(app),
        ("POST", "/generate") => {
            let request: AngleRequest = serde_json::from_slice(&request.body)
                .map_err(|error| format!("Invalid JSON request: {error}"))?;
            let output_path = run_angle_generation(app, request)?;
            Ok(serde_json::json!({ "ok": true, "outputPath": output_path }))
        }
        _ => Err(format!(
            "No agent endpoint for {} {}",
            request.method, request.path
        )),
    }
}

fn handle_agent_stream(mut stream: TcpStream, app: &AppHandle, agent_state: &AgentServerState) {
    let result = read_http_request(&mut stream).and_then(|request| {
        if request.method == "OPTIONS" {
            return write_empty_response(&mut stream, "204 No Content");
        }
        match handle_agent_route(request, app, agent_state) {
            Ok(payload) => write_json_response(&mut stream, "200 OK", payload),
            Err(error) => write_json_response(
                &mut stream,
                "400 Bad Request",
                serde_json::json!({ "ok": false, "error": error }),
            ),
        }
    });
    if let Err(error) = result {
        let _ = write_json_response(
            &mut stream,
            "400 Bad Request",
            serde_json::json!({ "ok": false, "error": error }),
        );
    }
}

fn run_agent_server(
    listener: TcpListener,
    app: AppHandle,
    agent_state: AgentServerState,
    stop: Arc<AtomicBool>,
) {
    let _ = listener.set_nonblocking(true);
    while !stop.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, _)) => handle_agent_stream(stream, &app, &agent_state),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_millis(80));
            }
            Err(_) => {
                thread::sleep(Duration::from_millis(150));
            }
        }
    }
}

#[tauri::command]
fn get_agent_server_status(
    agent_state: State<'_, AgentServerState>,
) -> Result<AgentServerStatus, String> {
    agent_status_from(agent_state.inner())
}

fn set_agent_server_enabled_inner(
    app: AppHandle,
    agent_state: &AgentServerState,
    enabled: bool,
    port: Option<u16>,
) -> Result<AgentServerStatus, String> {
    let port = validate_agent_api_port(port)?;
    {
        let mut control = agent_state
            .inner
            .lock()
            .map_err(|_| "Unable to lock agent server state.".to_string())?;

        if control.enabled && (!enabled || control.port() != port) {
            if let Some(stop) = control.stop.take() {
                stop.store(true, Ordering::SeqCst);
            }
            control.enabled = false;
        }
        control.port = port;

        if enabled && !control.enabled {
            let listener = TcpListener::bind((AGENT_API_BIND_ADDRESS, port))
                .map_err(|error| format!("Unable to start Agent API: {error}"))?;
            let stop = Arc::new(AtomicBool::new(false));
            thread::spawn({
                let app = app.clone();
                let agent_state = agent_state.clone();
                let stop = stop.clone();
                move || run_agent_server(listener, app, agent_state, stop)
            });
            control.enabled = true;
            control.stop = Some(stop);
        }
    }

    agent_status_from(agent_state)
}

#[tauri::command]
fn set_agent_server_enabled(
    app: AppHandle,
    agent_state: State<'_, AgentServerState>,
    enabled: bool,
    port: Option<u16>,
) -> Result<AgentServerStatus, String> {
    set_agent_server_enabled_inner(app, agent_state.inner(), enabled, port)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AgentServerState::default())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            list_gpus,
            generate_angle,
            store_info,
            open_path,
            get_agent_server_status,
            set_agent_server_enabled
        ])
        .run(tauri::generate_context!())
        .expect("error while running Multi-Angle Edit");
}
