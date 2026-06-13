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
    fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
};
use tauri::{AppHandle, Emitter, Manager};

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

mod neko_store;
use neko_store::NekoStore;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

const EMBEDDED_WORKER: &str = include_str!("../../python/multi_angle_worker.py");
const NEKO_APP_SLUG: &str = "multiangleedit";

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
// Paths & store
// ---------------------------------------------------------------------------

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
    let python = find_python().ok_or_else(|| {
        "Python 3.10, 3.11, or 3.12 was not found on PATH. Install it and try again.".to_string()
    })?;
    let worker = worker_path(&app)?;
    let store = neko_store(&app)?;

    let output_dir = match &request.output_dir {
        Some(dir) if !dir.trim().is_empty() => {
            let p = PathBuf::from(dir);
            fs::create_dir_all(&p)
                .map_err(|error| format!("Unable to create output dir: {error}"))?;
            p
        }
        _ => default_output_dir(&app)?,
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

    let jobs_dir = app_data_dir(&app)?.join("jobs");
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
    let store = neko_store(&app)?;
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            list_gpus,
            generate_angle,
            store_info,
            open_path
        ])
        .run(tauri::generate_context!())
        .expect("error while running Multi-Angle Edit");
}
