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

use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use serde::{Deserialize, Serialize};
use std::{
    env, fs,
    io::{BufRead, BufReader, Read, Write},
    net::{TcpListener, TcpStream, ToSocketAddrs},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::{AtomicBool, Ordering},
    sync::{mpsc, Arc, Mutex},
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
const QWEN_MODEL_CACHE_DIR: &str = "models--Qwen--Qwen-Image-Edit-2511";
const QWEN_GGUF_MODEL_CACHE_DIR: &str = "models--unsloth--Qwen-Image-Edit-2511-GGUF";
const ANGLES_LORA_CACHE_DIR: &str = "models--fal--Qwen-Image-Edit-2511-Multiple-Angles-LoRA";
const VENICE_APP_DATA_DIR: &str = "community.venice.media.local";
const VENICE_DISCOVERY_FILE: &str = "control-api.json";
const VENICE_RECENT_MODELS_STORAGE_KEY: &str = "veniceMediaLocal:recentModels:v1";
const DEFAULT_WINDOW_WIDTH: u32 = 1600;
const DEFAULT_WINDOW_HEIGHT: u32 = 900;
const MIN_WINDOW_WIDTH: u32 = 900;
const MIN_WINDOW_HEIGHT: u32 = 600;
const WINDOW_PREFERENCES_FILE: &str = "window-preferences.json";
const RESIZE_PREFERENCE_SAVE_DELAY: Duration = Duration::from_millis(500);
const SMALL_MONITOR_WIDTH_RATIO: f64 = 0.9;
const SMALL_MONITOR_HEIGHT_RATIO: f64 = 0.86;
const ROUND_WINDOW_SIZE_TO: u32 = 10;

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

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct LocalModelStatus {
    env_ready: bool,
    torch_ready: bool,
    deps_ready: bool,
    model_ready: bool,
    lora_ready: bool,
    store_root: String,
    env_dir: String,
    models_dir: String,
    message: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct VeniceMediaStatus {
    available: bool,
    message: String,
    url: Option<String>,
    model: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct GenerationResult {
    provider: String,
    output_path: Option<String>,
    data_url: Option<String>,
    message: String,
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WindowSizePreference {
    width: u32,
    height: u32,
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

/// The app's Python environment directory.
///
/// A shared store can sit on a drive used by both Windows and Linux (dual boot),
/// so on Linux the shared venv gets its own `<slug>-linux` folder instead of
/// reusing (and clobbering) the Windows venv and its install markers.
fn app_env_dir(store: &NekoStore) -> PathBuf {
    if cfg!(target_os = "windows") || !store.shared {
        return store.env_dir.clone();
    }
    store
        .env_dir
        .with_file_name(format!("{NEKO_APP_SLUG}-{}", env::consts::OS))
}

/// Folder the app's executable lives in, for the "output next to the app"
/// default. On Linux an AppImage runs from a temporary read-only mount, so use
/// the folder holding the `.AppImage` file itself.
fn app_install_dir() -> Option<PathBuf> {
    #[cfg(target_os = "linux")]
    if let Some(appimage) = env::var_os("APPIMAGE").map(PathBuf::from) {
        return appimage.parent().map(Path::to_path_buf);
    }
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
}

fn default_output_dir(app: &AppHandle) -> Result<PathBuf, String> {
    // Prefer an `output` folder next to the executable; fall back to app data.
    let base = app_install_dir()
        .map(|p| p.join("output"))
        .unwrap_or(app_data_dir(app)?.join("output"));
    if let Err(error) = fs::create_dir_all(&base) {
        // Read-only install location (e.g. /usr/bin on Linux): use app data.
        if cfg!(target_os = "windows") {
            return Err(format!("Unable to create output dir: {error}"));
        }
        let fallback = app_data_dir(app)?.join("output");
        fs::create_dir_all(&fallback)
            .map_err(|error| format!("Unable to create output dir: {error}"))?;
        return Ok(fallback);
    }
    Ok(base)
}

fn venv_python_path(env_dir: &Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        env_dir.join("Scripts").join("python.exe")
    } else {
        env_dir.join("bin").join("python")
    }
}

fn hf_repo_cache_ready(hf_home: &Path, repo_cache_dir: &str) -> bool {
    let snapshots = hf_home.join("hub").join(repo_cache_dir).join("snapshots");
    fs::read_dir(snapshots)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .any(|entry| entry.path().is_dir())
}

#[tauri::command]
fn local_model_status(app: AppHandle) -> Result<LocalModelStatus, String> {
    let store = neko_store(&app)?;
    let env_dir = app_env_dir(&store);
    let env_ready = venv_python_path(&env_dir).exists();
    let torch_ready = env_dir.join(".multiangle-torch-ready").exists();
    let deps_ready = env_dir.join(".multiangle-deps-gguf-ready").exists();
    let hf_home = store.hf_home();
    let model_ready = hf_repo_cache_ready(&hf_home, QWEN_MODEL_CACHE_DIR)
        && hf_repo_cache_ready(&hf_home, QWEN_GGUF_MODEL_CACHE_DIR);
    let lora_ready = hf_repo_cache_ready(&hf_home, ANGLES_LORA_CACHE_DIR);

    let message = if !env_ready {
        "First local generation will create a Python environment, install PyTorch/diffusers/GGUF support, then download quantized Qwen model files.".to_string()
    } else if !torch_ready || !deps_ready {
        "Python environment exists; first local generation will finish installing PyTorch/diffusers/GGUF support before model download.".to_string()
    } else if !model_ready || !lora_ready {
        "Local runtime is ready; first local generation will download Qwen-Image-Edit 2511, a VRAM-selected GGUF transformer quant, and the Multiple-Angles LoRA into the shared model cache.".to_string()
    } else {
        "Local runtime and quantized Qwen model cache are ready.".to_string()
    };

    Ok(LocalModelStatus {
        env_ready,
        torch_ready,
        deps_ready,
        model_ready,
        lora_ready,
        store_root: store.root.display().to_string(),
        env_dir: env_dir.display().to_string(),
        models_dir: store.models_dir.display().to_string(),
        message,
    })
}

#[cfg_attr(not(target_os = "windows"), allow(unused_variables))]
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
    if let Ok(resource) = app.path().resolve(
        "python/multi_angle_worker.py",
        tauri::path::BaseDirectory::Resource,
    ) {
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
        vec![
            "python3.11",
            "python3.12",
            "python3.10",
            "python3",
            "python",
        ]
    };
    for candidate in &candidates {
        if python_ok(candidate) {
            return Some(candidate.to_string());
        }
    }
    // Rolling-release distros (Arch/Omarchy) ship a newer Python 3 only. The
    // bootstrap phase of the worker is stdlib-only, and it builds the venv with
    // `uv venv --python 3.11` when uv is available, so a newer system Python is
    // fine for starting it.
    if !cfg!(target_os = "windows") {
        for candidate in ["python3", "python"] {
            if python_minor_version(candidate).is_some_and(|minor| minor >= 10) {
                return Some(candidate.to_string());
            }
        }
    }
    None
}

/// Minor version of a Python 3 interpreter (`3.14.1` -> 14).
fn python_minor_version(spec: &str) -> Option<u32> {
    let output = command_from_python_spec(spec).arg("--version").output().ok()?;
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let version = text.trim().strip_prefix("Python 3.")?;
    version.split('.').next()?.trim().parse().ok()
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
async fn generate_angle(app: AppHandle, request: AngleRequest) -> Result<String, String> {
    tauri::async_runtime::spawn_blocking(move || run_angle_generation(&app, request))
        .await
        .map_err(|error| format!("Generation task failed: {error}"))?
}

fn run_angle_generation(app: &AppHandle, request: AngleRequest) -> Result<String, String> {
    let python = find_python().ok_or_else(|| {
        if cfg!(target_os = "windows") {
            "Python 3.10, 3.11, or 3.12 was not found on PATH. Install it and try again.".to_string()
        } else {
            "Python 3.10 or newer (python3) was not found on PATH. Install it and try again.".to_string()
        }
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
        env_dir: app_env_dir(&store).display().to_string(),
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
    command.env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True");
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
        "envDir": app_env_dir(&store).display().to_string(),
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

fn percent_decode_lossy(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;

    while index < bytes.len() {
        if bytes[index] == b'%' && index + 2 < bytes.len() {
            if let Ok(hex) = u8::from_str_radix(&value[index + 1..index + 3], 16) {
                out.push(hex);
                index += 3;
                continue;
            }
        }
        out.push(bytes[index]);
        index += 1;
    }

    String::from_utf8_lossy(&out).into_owned()
}

fn normalize_preview_path(path: &str) -> PathBuf {
    let trimmed = path.trim();
    let without_scheme = trimmed
        .strip_prefix("file://")
        .unwrap_or(trimmed)
        .trim_start_matches("localhost/");
    #[cfg_attr(not(target_os = "windows"), allow(unused_mut))]
    let mut normalized = percent_decode_lossy(without_scheme);

    #[cfg(target_os = "windows")]
    {
        if normalized.starts_with('/') && normalized.as_bytes().get(2) == Some(&b':') {
            normalized.remove(0);
        }
        if normalized.starts_with(r"\\?\") {
            normalized = normalized[4..].to_string();
        }
        normalized = normalized.replace('/', r"\");
    }

    PathBuf::from(normalized)
}

#[tauri::command]
fn image_preview_data_url(path: String) -> Result<String, String> {
    let path = normalize_preview_path(&path);
    image_path_to_data_url(&path)
}

fn image_path_to_data_url(path: &Path) -> Result<String, String> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let mime = match extension.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        _ => return Err("Preview supports PNG, JPG, JPEG, and WebP images.".to_string()),
    };

    let metadata =
        fs::metadata(&path).map_err(|error| format!("Unable to read image metadata: {error}"))?;
    if metadata.len() > 100 * 1024 * 1024 {
        return Err("Image is too large to preview in the app.".to_string());
    }

    let bytes = fs::read(&path).map_err(|error| format!("Unable to read image: {error}"))?;
    Ok(format!(
        "data:{mime};base64,{}",
        BASE64_STANDARD.encode(bytes)
    ))
}

fn venice_app_data_dir() -> Option<PathBuf> {
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
    Some(base.join(VENICE_APP_DATA_DIR))
}

fn venice_discovery_path() -> Option<PathBuf> {
    Some(venice_app_data_dir()?.join(VENICE_DISCOVERY_FILE))
}

fn read_venice_discovery() -> Result<serde_json::Value, String> {
    let path = venice_discovery_path()
        .ok_or_else(|| "Could not resolve Venice Media Local app data folder.".to_string())?;
    let raw = fs::read_to_string(&path).map_err(|_| {
        format!(
            "Venice Media Local is not linked yet. Open Venice Media Local, enable AI Agent Remote Control, then try again. Expected discovery file: {}",
            path.display()
        )
    })?;
    serde_json::from_str(&raw)
        .map_err(|error| format!("Venice Media Local discovery file is invalid: {error}"))
}

fn venice_control_address(discovery: &serde_json::Value) -> String {
    let bind_address = discovery
        .get("bindAddress")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    if bind_address.starts_with("127.0.0.1:") || bind_address.starts_with("localhost:") {
        return bind_address.to_string();
    }

    discovery
        .get("address")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("127.0.0.1:9876")
        .trim()
        .trim_start_matches("http://")
        .trim_start_matches("https://")
        .to_string()
}

fn venice_control_token(discovery: &serde_json::Value) -> Result<String, String> {
    discovery
        .get("token")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
        .ok_or_else(|| "Venice Media Local discovery file has no API token.".to_string())
}

fn decode_chunked_body(body: &[u8]) -> Result<Vec<u8>, String> {
    let mut decoded = Vec::new();
    let mut index = 0;
    while index < body.len() {
        let Some(line_end) = body[index..]
            .windows(2)
            .position(|window| window == b"\r\n")
            .map(|pos| index + pos)
        else {
            return Err("Invalid chunked Venice response.".to_string());
        };
        let size_line = String::from_utf8_lossy(&body[index..line_end]);
        let size = usize::from_str_radix(size_line.split(';').next().unwrap_or("").trim(), 16)
            .map_err(|error| format!("Invalid chunk size from Venice Media Local: {error}"))?;
        index = line_end + 2;
        if size == 0 {
            break;
        }
        if index + size > body.len() {
            return Err("Incomplete chunked Venice response.".to_string());
        }
        decoded.extend_from_slice(&body[index..index + size]);
        index += size + 2;
    }
    Ok(decoded)
}

fn read_venice_response(stream: &mut TcpStream) -> Result<serde_json::Value, String> {
    stream
        .set_read_timeout(Some(Duration::from_secs(180)))
        .map_err(|error| format!("Unable to set Venice read timeout: {error}"))?;
    let mut data = Vec::new();
    stream
        .read_to_end(&mut data)
        .map_err(|error| format!("Unable to read Venice Media Local response: {error}"))?;

    let header_end = find_header_end(&data)
        .ok_or_else(|| "Invalid HTTP response from Venice Media Local.".to_string())?;
    let headers = String::from_utf8_lossy(&data[..header_end]);
    let status = headers
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(0);
    let lower_headers = headers.to_ascii_lowercase();
    let mut body = data[header_end + 4..].to_vec();
    if lower_headers.contains("transfer-encoding: chunked") {
        body = decode_chunked_body(&body)?;
    }

    if status >= 400 || status == 0 {
        let detail = String::from_utf8_lossy(&body);
        return Err(format!(
            "Venice Media Local returned HTTP {status}: {}",
            detail.trim()
        ));
    }

    serde_json::from_slice(&body)
        .map_err(|error| format!("Venice Media Local returned invalid JSON: {error}"))
}

fn venice_json_request(
    address: &str,
    method: &str,
    path: &str,
    token: &str,
    body: Option<&serde_json::Value>,
) -> Result<serde_json::Value, String> {
    let body_bytes = match body {
        Some(value) => serde_json::to_vec(value)
            .map_err(|error| format!("Unable to serialize Venice request: {error}"))?,
        None => Vec::new(),
    };
    let addr = address
        .to_socket_addrs()
        .map_err(|error| {
            format!("Unable to resolve Venice Media Local address {address}: {error}")
        })?
        .next()
        .ok_or_else(|| format!("No socket address found for Venice Media Local at {address}"))?;
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(3)).map_err(|error| {
        format!("Could not connect to Venice Media Local at {address}. Is it open with AI Agent Remote Control enabled? {error}")
    })?;
    stream
        .set_write_timeout(Some(Duration::from_secs(30)))
        .map_err(|error| format!("Unable to set Venice write timeout: {error}"))?;
    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: {address}\r\nAuthorization: Bearer {token}\r\nContent-Type: application/json\r\nAccept: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body_bytes.len()
    );
    stream
        .write_all(request.as_bytes())
        .and_then(|_| stream.write_all(&body_bytes))
        .map_err(|error| format!("Unable to write Venice Media Local request: {error}"))?;
    read_venice_response(&mut stream)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct VeniceEditModelChoice {
    model: String,
    source: &'static str,
}

fn venice_edit_model_ids(state: &serde_json::Value) -> Vec<String> {
    state
        .get("models")
        .and_then(|models| models.get("editModels"))
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|model| model.get("id").and_then(serde_json::Value::as_str))
        .filter(|id| !id.trim().is_empty())
        .map(str::to_string)
        .collect()
}

fn venice_state_model_candidate<'a>(
    value: &'a serde_json::Value,
    path: &[&str],
) -> Option<&'a str> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    current
        .as_str()
        .or_else(|| current.get("id").and_then(serde_json::Value::as_str))
        .or_else(|| current.get("model").and_then(serde_json::Value::as_str))
        .or_else(|| current.get("modelId").and_then(serde_json::Value::as_str))
        .map(str::trim)
        .filter(|id| !id.is_empty())
}

fn venice_selected_edit_model_from_state(state: &serde_json::Value) -> Option<String> {
    const SELECTED_MODEL_PATHS: &[&[&str]] = &[
        &["selectedModels", "edit"],
        &["selectedModels", "editModel"],
        &["selectedModels", "dit"],
        &["selectedModels", "ditModel"],
        &["selected", "edit"],
        &["selected", "editModel"],
        &["currentModels", "edit"],
        &["currentModels", "editModel"],
        &["activeModels", "edit"],
        &["activeModels", "editModel"],
        &["settings", "selectedModels", "edit"],
        &["settings", "selectedModels", "editModel"],
        &["settings", "selectedEditModel"],
        &["settings", "editModel"],
        &["settings", "selectedDitModel"],
        &["settings", "ditModel"],
    ];

    SELECTED_MODEL_PATHS
        .iter()
        .find_map(|path| venice_state_model_candidate(state, path))
        .map(str::to_string)
}

fn extract_json_object_at(bytes: &[u8], start: usize) -> Option<&[u8]> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (offset, byte) in bytes[start..].iter().copied().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            continue;
        }

        match byte {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(&bytes[start..=start + offset]);
                }
            }
            _ => {}
        }
    }

    None
}

fn recent_edit_models_from_leveldb_bytes(bytes: &[u8]) -> Vec<String> {
    let key = VENICE_RECENT_MODELS_STORAGE_KEY.as_bytes();
    let mut models = Vec::new();
    let mut search_start = 0usize;

    while search_start < bytes.len() {
        let Some(relative_pos) = bytes[search_start..]
            .windows(key.len())
            .position(|window| window == key)
        else {
            break;
        };
        let key_end = search_start + relative_pos + key.len();
        let Some(json_start) = bytes[key_end..]
            .iter()
            .position(|byte| *byte == b'{')
            .map(|offset| key_end + offset)
        else {
            break;
        };

        if let Some(json_bytes) = extract_json_object_at(bytes, json_start) {
            if let Ok(value) = serde_json::from_slice::<serde_json::Value>(json_bytes) {
                if let Some(model) = value
                    .get("edit")
                    .and_then(serde_json::Value::as_array)
                    .and_then(|items| items.first())
                    .and_then(serde_json::Value::as_str)
                    .map(str::trim)
                    .filter(|id| !id.is_empty())
                {
                    models.push(model.to_string());
                }
            }
        }

        search_start = key_end;
    }

    models
}

fn venice_webview_leveldb_dir() -> Option<PathBuf> {
    if cfg!(target_os = "windows") {
        env::var_os("LOCALAPPDATA").map(PathBuf::from).map(|base| {
            base.join(VENICE_APP_DATA_DIR)
                .join("EBWebView")
                .join("Default")
                .join("Local Storage")
                .join("leveldb")
        })
    } else {
        None
    }
}

fn read_venice_recent_edit_model() -> Option<String> {
    let dir = venice_webview_leveldb_dir()?;
    let mut files = fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let path = entry.path();
            let extension = path
                .extension()
                .and_then(|value| value.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            if !matches!(extension.as_str(), "log" | "ldb" | "sst") {
                return None;
            }
            let modified = entry
                .metadata()
                .and_then(|metadata| metadata.modified())
                .unwrap_or(UNIX_EPOCH);
            Some((modified, path))
        })
        .collect::<Vec<_>>();
    files.sort_by(|(left_time, left_path), (right_time, right_path)| {
        left_time
            .cmp(right_time)
            .then_with(|| left_path.cmp(right_path))
    });

    let mut selected = None;
    for (_, path) in files {
        let Ok(bytes) = fs::read(path) else {
            continue;
        };
        for model in recent_edit_models_from_leveldb_bytes(&bytes) {
            selected = Some(model);
        }
    }
    selected
}

fn choose_venice_edit_model_with_recent(
    state: &serde_json::Value,
    recent_model: Option<String>,
) -> Option<VeniceEditModelChoice> {
    let edit_model_ids = venice_edit_model_ids(state);
    let accepts_model = |model: &str| {
        edit_model_ids.is_empty() || edit_model_ids.iter().any(|known| known == model)
    };

    if let Some(model) =
        venice_selected_edit_model_from_state(state).filter(|model| accepts_model(model))
    {
        return Some(VeniceEditModelChoice {
            model,
            source: "selected in Venice Media Local",
        });
    }

    if let Some(model) = recent_model.filter(|model| accepts_model(model)) {
        return Some(VeniceEditModelChoice {
            model,
            source: "most recent Venice Media Local edit model",
        });
    }

    edit_model_ids
        .into_iter()
        .next()
        .map(|model| VeniceEditModelChoice {
            model,
            source: "first Venice Media Local edit model",
        })
}

fn choose_venice_edit_model(state: &serde_json::Value) -> Option<VeniceEditModelChoice> {
    choose_venice_edit_model_with_recent(state, read_venice_recent_edit_model())
}

fn venice_state() -> Result<(String, String, serde_json::Value), String> {
    let discovery = read_venice_discovery()?;
    let address = venice_control_address(&discovery);
    let token = venice_control_token(&discovery)?;
    let state = venice_json_request(&address, "GET", "/api/v1/state", &token, None)?;
    Ok((address, token, state))
}

#[tauri::command]
fn venice_media_status() -> VeniceMediaStatus {
    match venice_state() {
        Ok((address, _, state)) => {
            if let Some(choice) = choose_venice_edit_model(&state) {
                VeniceMediaStatus {
                    available: true,
                    message: format!(
                        "Venice Media Local is linked at {address}; using {source}: {model}.",
                        source = choice.source,
                        model = choice.model,
                    ),
                    url: Some(format!("http://{address}")),
                    model: Some(choice.model),
                }
            } else {
                VeniceMediaStatus {
                    available: false,
                    message: format!(
                        "Venice Media Local is linked at {address}, but no edit/DiT model is available."
                    ),
                    url: Some(format!("http://{address}")),
                    model: None,
                }
            }
        }
        Err(error) => VeniceMediaStatus {
            available: false,
            message: error,
            url: None,
            model: None,
        },
    }
}

fn venice_angle_prompt(angle_prompt: &str, extra_prompt: Option<&str>) -> String {
    let cleaned_angle = angle_prompt
        .replace("<sks>", "")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let mut prompt = format!(
        "Re-render this source image from a new camera angle: {cleaned_angle}. Preserve the source image's subject identity, outfit, materials, background, lighting direction, color palette, and overall scene style unless the prompt additions explicitly change them."
    );
    if let Some(extra) = extra_prompt
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        prompt.push_str(" Prompt additions: ");
        prompt.push_str(extra);
    }
    prompt
}

#[tauri::command]
async fn generate_angle_venice(request: AngleRequest) -> Result<GenerationResult, String> {
    tauri::async_runtime::spawn_blocking(move || generate_angle_venice_blocking(request))
        .await
        .map_err(|error| format!("Venice generation task failed: {error}"))?
}

fn generate_angle_venice_blocking(request: AngleRequest) -> Result<GenerationResult, String> {
    let (address, token, state) = venice_state()?;
    let model_choice = choose_venice_edit_model(&state).ok_or_else(|| {
        "Venice Media Local did not report an available edit/DiT model.".to_string()
    })?;
    let data_url = image_path_to_data_url(&normalize_preview_path(&request.image_path))?;
    let prompt = venice_angle_prompt(&request.angle_prompt, request.extra_prompt.as_deref());
    let body = serde_json::json!({
        "model": model_choice.model.clone(),
        "prompt": prompt,
        "images": [data_url],
        "navigate": true,
    });
    let result = venice_json_request(&address, "POST", "/api/v1/edit-image", &token, Some(&body))?;
    let output_path = result
        .get("filePath")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    let data_url = result
        .get("dataUrl")
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_string);

    if output_path.is_none() && data_url.is_none() {
        return Err(
            "Venice Media Local finished but did not return an image path or preview data."
                .to_string(),
        );
    }

    Ok(GenerationResult {
        provider: "venice".to_string(),
        output_path,
        data_url,
        message: format!(
            "Online edit finished in Venice Media Local using {source}: {model}.",
            source = model_choice.source,
            model = model_choice.model,
        ),
    })
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
        ("GET", "/gpus") => serde_json::to_value(list_gpus()).map_err(|error| error.to_string()),
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

fn env_truthy(name: &str) -> bool {
    env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn agent_api_headless_requested() -> bool {
    env::args().any(|arg| arg == "--headless" || arg == "--serve-agent-api")
        || env_truthy("NEKO_AGENT_HEADLESS")
        || env_truthy("MULTI_ANGLE_EDIT_HEADLESS")
}

fn agent_api_cli_port() -> Option<u16> {
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--agent-api-port" || arg == "--agent-port" {
            return args.next().and_then(|value| value.parse::<u16>().ok());
        }
        if let Some(value) = arg
            .strip_prefix("--agent-api-port=")
            .or_else(|| arg.strip_prefix("--agent-port="))
        {
            return value.parse::<u16>().ok();
        }
    }
    env::var("MULTI_ANGLE_EDIT_AGENT_API_PORT")
        .or_else(|_| env::var("NEKO_AGENT_API_PORT"))
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
}

fn hide_main_window(app: &tauri::App) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.hide();
    }
}

fn clamp_window_dimension(value: u32, min: u32, monitor_max: u32) -> u32 {
    let max = monitor_max.max(1);
    value.max(min.min(max)).min(max)
}

fn clamp_window_size(
    size: tauri::PhysicalSize<u32>,
    monitor_width: u32,
    monitor_height: u32,
) -> tauri::PhysicalSize<u32> {
    tauri::PhysicalSize::new(
        clamp_window_dimension(size.width, MIN_WINDOW_WIDTH, monitor_width),
        clamp_window_dimension(size.height, MIN_WINDOW_HEIGHT, monitor_height),
    )
}

fn round_window_dimension(value: f64) -> u32 {
    let multiple = ROUND_WINDOW_SIZE_TO.max(1) as f64;
    (value / multiple).round().max(1.0) as u32 * ROUND_WINDOW_SIZE_TO.max(1)
}

fn initial_window_size(monitor_width: u32, monitor_height: u32) -> tauri::PhysicalSize<u32> {
    let landscape = monitor_width >= monitor_height;
    let size = if landscape && monitor_width >= 3000 && monitor_height >= 1800 {
        // 4K class: launch one resolution tier down, e.g. 3840x2160 -> 1920x1080.
        tauri::PhysicalSize::new(monitor_width / 2, monitor_height / 2)
    } else if landscape && monitor_width >= 2300 && monitor_height >= 1300 {
        // 2K / 1440p class: launch one tier down, e.g. 2560x1440 -> 1280x720.
        tauri::PhysicalSize::new(monitor_width / 2, monitor_height / 2)
    } else if landscape && monitor_width >= 1800 && monitor_height >= 1000 {
        // Literal half-size is too cramped on 1080p for the three-column tool.
        tauri::PhysicalSize::new(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
    } else {
        tauri::PhysicalSize::new(
            round_window_dimension(monitor_width as f64 * SMALL_MONITOR_WIDTH_RATIO),
            round_window_dimension(monitor_height as f64 * SMALL_MONITOR_HEIGHT_RATIO),
        )
    };

    clamp_window_size(size, monitor_width, monitor_height)
}

fn main_monitor_size(window: &tauri::WebviewWindow) -> tauri::PhysicalSize<u32> {
    let monitor_size = window
        .current_monitor()
        .ok()
        .flatten()
        .or_else(|| window.primary_monitor().ok().flatten())
        .map(|monitor| *monitor.size());
    let monitor_width = monitor_size
        .as_ref()
        .map(|size| size.width)
        .unwrap_or(DEFAULT_WINDOW_WIDTH);
    let monitor_height = monitor_size
        .as_ref()
        .map(|size| size.height)
        .unwrap_or(DEFAULT_WINDOW_HEIGHT);
    tauri::PhysicalSize::new(monitor_width, monitor_height)
}

fn window_preferences_path(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(app_data_dir(app)?.join(WINDOW_PREFERENCES_FILE))
}

fn read_window_size_preference(app: &AppHandle) -> Option<tauri::PhysicalSize<u32>> {
    let path = window_preferences_path(app).ok()?;
    let text = fs::read_to_string(path).ok()?;
    let pref: WindowSizePreference = serde_json::from_str(&text).ok()?;
    if pref.width < 100 || pref.height < 100 {
        return None;
    }
    Some(tauri::PhysicalSize::new(pref.width, pref.height))
}

fn write_window_size_preference(path: &Path, size: tauri::PhysicalSize<u32>) -> Result<(), String> {
    if size.width < 100 || size.height < 100 {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("Unable to create window preference dir: {error}"))?;
    }
    let pref = WindowSizePreference {
        width: size.width,
        height: size.height,
    };
    let text = serde_json::to_string_pretty(&pref)
        .map_err(|error| format!("Unable to serialize window preference: {error}"))?;
    fs::write(path, text).map_err(|error| format!("Unable to save window preference: {error}"))
}

fn apply_initial_window_size(app: &tauri::App) {
    let Some(window) = app.get_webview_window("main") else {
        return;
    };
    let monitor = main_monitor_size(&window);
    let app_handle = app.handle().clone();
    let size = read_window_size_preference(&app_handle)
        .unwrap_or_else(|| initial_window_size(monitor.width, monitor.height));
    let size = clamp_window_size(size, monitor.width, monitor.height);

    if let Err(error) = window.set_size(tauri::Size::Physical(size)) {
        eprintln!("Failed to initialize window size: {error}");
    }
    let _ = window.center();
}

fn install_window_resize_persistence(app: &tauri::App) {
    let Some(window) = app.get_webview_window("main") else {
        return;
    };
    let Ok(path) = window_preferences_path(&app.handle().clone()) else {
        return;
    };
    let launch_size = window.inner_size().ok();
    let (tx, rx) = mpsc::channel::<tauri::PhysicalSize<u32>>();

    thread::spawn(move || {
        while let Ok(mut latest) = rx.recv() {
            while let Ok(next) = rx.recv_timeout(RESIZE_PREFERENCE_SAVE_DELAY) {
                latest = next;
            }
            if let Err(error) = write_window_size_preference(&path, latest) {
                eprintln!("Failed to save window preference: {error}");
            }
        }
    });

    window.on_window_event(move |event| {
        let size = match event {
            tauri::WindowEvent::Resized(size) => *size,
            tauri::WindowEvent::ScaleFactorChanged { new_inner_size, .. } => *new_inner_size,
            _ => return,
        };
        if Some(size) == launch_size {
            return;
        }
        let _ = tx.send(size);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn model_state() -> serde_json::Value {
        json!({
            "models": {
                "editModels": [
                    { "id": "firered-image-edit" },
                    { "id": "gpt-image-2-edit" },
                    { "id": "nano-banana-2-edit" }
                ]
            }
        })
    }

    #[test]
    fn extracts_latest_recent_edit_model_from_leveldb_bytes() {
        let bytes = br#"noise veniceMediaLocal:recentModels:v1 x {"edit":["gpt-image-2-edit"]} more veniceMediaLocal:recentModels:v1 y {"image":["flux"],"edit":["nano-banana-2-edit","gpt-image-2-edit"]} end"#;

        let models = recent_edit_models_from_leveldb_bytes(bytes);

        assert_eq!(
            models,
            vec![
                "gpt-image-2-edit".to_string(),
                "nano-banana-2-edit".to_string()
            ]
        );
    }

    #[test]
    fn chooses_selected_model_from_state_before_recent_model() {
        let mut state = model_state();
        state["selectedModels"] = json!({ "edit": "gpt-image-2-edit" });

        let choice =
            choose_venice_edit_model_with_recent(&state, Some("nano-banana-2-edit".to_string()))
                .expect("model choice");

        assert_eq!(choice.model, "gpt-image-2-edit");
        assert_eq!(choice.source, "selected in Venice Media Local");
    }

    #[test]
    fn chooses_recent_edit_model_before_first_available_model() {
        let state = model_state();

        let choice =
            choose_venice_edit_model_with_recent(&state, Some("nano-banana-2-edit".to_string()))
                .expect("model choice");

        assert_eq!(choice.model, "nano-banana-2-edit");
        assert_eq!(choice.source, "most recent Venice Media Local edit model");
    }

    #[test]
    fn falls_back_to_first_advertised_edit_model_without_hardcoded_preference() {
        let state = model_state();

        let choice = choose_venice_edit_model_with_recent(&state, None).expect("model choice");

        assert_eq!(choice.model, "firered-image-edit");
        assert_eq!(choice.source, "first Venice Media Local edit model");
    }

    #[test]
    fn smart_window_size_halves_4k_displays() {
        let size = initial_window_size(3840, 2160);

        assert_eq!((size.width, size.height), (1920, 1080));
    }

    #[test]
    fn smart_window_size_halves_1440p_displays() {
        let size = initial_window_size(2560, 1440);

        assert_eq!((size.width, size.height), (1280, 720));
    }

    #[test]
    fn smart_window_size_keeps_1080p_usable() {
        let size = initial_window_size(1920, 1080);

        assert_eq!((size.width, size.height), (1600, 900));
    }

    #[test]
    fn smart_window_size_uses_most_of_small_displays() {
        let size = initial_window_size(1366, 768);

        assert_eq!((size.width, size.height), (1230, 660));
    }

    #[test]
    fn venice_prompt_preserves_source_and_appends_user_text() {
        let prompt = venice_angle_prompt(
            "<sks> right side view high-angle shot close-up",
            Some("add a soft rim light"),
        );

        assert!(prompt.contains("right side view high-angle shot close-up"));
        assert!(prompt.contains("Preserve the source image's subject identity"));
        assert!(prompt.contains("Prompt additions: add a soft rim light"));
        assert!(!prompt.contains("<sks>"));
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(AgentServerState::default())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let headless = agent_api_headless_requested();
            if !headless {
                apply_initial_window_size(app);
                install_window_resize_persistence(app);
            }
            if headless {
                hide_main_window(app);
                let handle = app.handle().clone();
                let agent_state = app.state::<AgentServerState>().inner().clone();
                if let Err(error) =
                    set_agent_server_enabled_inner(handle, &agent_state, true, agent_api_cli_port())
                {
                    eprintln!("Unable to start Agent API: {error}");
                }
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            list_gpus,
            generate_angle,
            generate_angle_venice,
            image_preview_data_url,
            local_model_status,
            venice_media_status,
            store_info,
            open_path,
            get_agent_server_status,
            set_agent_server_enabled
        ])
        .run(tauri::generate_context!())
        .expect("error while running Multi-Angle Edit");
}
