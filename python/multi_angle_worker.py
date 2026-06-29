"""Multi-Angle Edit — Qwen-Image-Edit camera-angle re-render worker.

This worker re-renders an input image from a new camera angle using
Qwen-Image-Edit (2509/2511) together with the fal "Multiple-Angles" LoRA,
driven through the Hugging Face ``diffusers`` library.

It follows the proven two-phase bootstrap/relaunch pattern used by the sibling
DepthMapAIGenerator worker:

  * Phase 1 (system Python): create/validate an app-managed virtual environment,
    install GPU (or CPU) PyTorch and the ML runtime packages into it, then
    re-exec *this same script* with the venv's Python and MULTIANGLE_IN_VENV=1.
  * Phase 2 (venv Python): skip all bootstrapping, import torch/diffusers, load
    the pipeline + LoRA, run inference, write the PNG, and emit ``done``.

Input contract (launched by the Rust backend)::

    python qwen_angle_worker.py --job <path-to-job.json>

The job JSON uses camelCase keys: imagePath, anglePrompt, extraPrompt,
deviceIndex, envDir, modelsDir, outputPath.

Output contract: NDJSON events on stdout, one JSON object per line, each flushed
immediately. Event shapes:

    {"type":"setup","message":"..."}
    {"type":"progress","message":"...","percent":42}
    {"type":"done","outputPath":"..."}
    {"type":"error","message":"..."}

A nonzero exit OR a missing output file is treated as failure by the parent, so
the PNG is written to exactly job['outputPath'] before ``done`` is emitted, and
the process exits nonzero after any ``error`` event.

The parent process already exports the shared NekoLegendsAI cache environment
variables (HF_HOME, HF_HUB_CACHE, HUGGINGFACE_HUB_CACHE, PIP_CACHE_DIR,
UV_CACHE_DIR, TORCH_HOME). This worker respects them and never overrides them.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Env var marker that tells the re-exec'd process it is already inside the venv.
IN_VENV_ENV = "MULTIANGLE_IN_VENV"

# The suite standardizes on cu128: it is the only PyTorch build that covers both
# the RTX 5090 (Blackwell, sm_120) and the RTX 3090 (Ampere, sm_86).
PYTORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu128"

# Bootstrap packaging tools (only used by the stdlib-venv install path).
BOOTSTRAP_PACKAGES = ["pip", "setuptools", "wheel"]

# CUDA / CPU PyTorch. We deliberately do not hard-pin torch so that whatever the
# cu128 index currently ships (torch 2.7.x or newer) is installed cleanly.
PYTORCH_PACKAGES = ["torch", "torchvision"]

# Diffusers / Transformers ML runtime. peft is required to load LoRA weights.
# gguf is required for the quantized Qwen transformer path.
ML_PACKAGES = [
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "pillow",
    "huggingface_hub",
    "peft",
    "gguf",
]

# Marker files inside envDir so re-runs skip already-completed install steps.
TORCH_MARKER = ".multiangle-torch-ready"
DEPS_MARKER = ".multiangle-deps-gguf-ready"
LEGACY_DEPS_MARKER = ".multiangle-deps-ready"

# Hugging Face model + LoRA identifiers.
QWEN_MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
QWEN_GGUF_MODEL_ID = "unsloth/Qwen-Image-Edit-2511-GGUF"
ANGLES_LORA_ID = "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA"

QWEN_GGUF_FILES = {
    "Q2_K": "qwen-image-edit-2511-Q2_K.gguf",
    "Q3_K_M": "qwen-image-edit-2511-Q3_K_M.gguf",
    "Q4_K_M": "qwen-image-edit-2511-Q4_K_M.gguf",
    "Q5_K_M": "qwen-image-edit-2511-Q5_K_M.gguf",
    "Q6_K": "qwen-image-edit-2511-Q6_K.gguf",
    "Q8_0": "qwen-image-edit-2511-Q8_0.gguf",
}
DEFAULT_QWEN_GGUF_QUANT = "Q8_0"
QWEN_TRANSFORMER_IGNORE_PATTERNS = [
    "transformer/*.safetensors",
    "transformer/*.bin",
    "transformer/*.pt",
    "transformer/*.gguf",
    "transformer/*.index.json",
]

# Inference defaults.
DEFAULT_STEPS = 30
DEFAULT_TRUE_CFG_SCALE = 4.0
DEFAULT_SEED = 42
DEFAULT_PRESERVATION_GUIDANCE = (
    "preserve the source image's subject identity, outfit, materials, "
    "background, lighting direction, color palette, and overall scene style "
    "unless the prompt additions explicitly change them"
)
LOW_VRAM_OFFLOAD_THRESHOLD_GB = 28.0
MAX_SOURCE_SIDE = 1024
MAX_SOURCE_PIXELS = 1024 * 1024
SOURCE_SIZE_MULTIPLE = 16


# --------------------------------------------------------------------------- #
# Event emitter (identical idiom to the sibling worker)
# --------------------------------------------------------------------------- #

def emit(event_type: str, message: str | None = None, **payload: Any) -> None:
    """Print a single NDJSON event to stdout and flush immediately."""
    event: dict[str, Any] = {"type": event_type}
    if message:
        event["message"] = message
    event.update(payload)
    print(json.dumps(event, ensure_ascii=False), flush=True)


# --------------------------------------------------------------------------- #
# Venv / bootstrap helpers (stdlib only — no third-party imports here)
# --------------------------------------------------------------------------- #

def venv_python(env_dir: Path) -> Path:
    """Return the path to the venv's Python interpreter for this platform."""
    if platform.system() == "Windows":
        return env_dir / "Scripts" / "python.exe"
    return env_dir / "bin" / "python"


def sanitize_installer_line(line: str) -> str | None:
    """Collapse noisy pip/uv output into short, user-friendly status messages.

    Returns ``None`` for lines that should be suppressed.
    """
    if line.startswith("Collecting "):
        package = line.removeprefix("Collecting ").split()[0]
        return f"Resolving dependency: {package}"
    if line.startswith("Downloading ") or line.startswith("Downloaded "):
        return "Downloading dependency packages"
    if line.startswith("Installing collected packages"):
        return "Installing dependency packages"
    if line.startswith("Successfully installed"):
        return "Dependency install step finished"
    # uv-style progress lines.
    if line.startswith("Resolved ") or line.startswith("Prepared ") or line.startswith("Installed "):
        return line
    if line.startswith("ERROR:") or " error:" in line.lower() or "failed" in line.lower():
        return line
    return None


def run_checked(command: list[str], message: str) -> None:
    """Run a subprocess, streaming sanitized status through ``setup`` events.

    Raises ``RuntimeError`` (with the captured tail of output) on nonzero exit.
    """
    emit("setup", f"{message}. This can take several minutes.")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    tail: list[str] = []
    last_status: str | None = None
    for line in process.stdout:
        line = line.strip()
        if line:
            tail.append(line)
            tail = tail[-18:]
            status = sanitize_installer_line(line)
            if status and status != last_status:
                emit("setup", status)
                last_status = status
    code = process.wait()
    if code != 0:
        details = "\n".join(tail) if tail else "No installer output was captured."
        raise RuntimeError(
            f"{message} failed. Multi-Angle Edit uses its own virtual environment, "
            f"but it needs a compatible system Python (3.10, 3.11, or 3.12) on PATH "
            f"to create it.\n\nLast installer output:\n{details}"
        )


def nvidia_smi_available() -> bool:
    """Return True if an NVIDIA GPU is present (via ``nvidia-smi``)."""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except Exception:
        return False
    return True


def ensure_env(env_dir: Path, device_index: int) -> Path:
    """Create/validate the venv and install all dependencies into it.

    Uses ``uv`` when available (preferred), otherwise the stdlib ``venv`` module.
    Marker files inside ``env_dir`` make completed install steps idempotent.

    Returns the path to the venv's Python interpreter.
    """
    env_dir.mkdir(parents=True, exist_ok=True)
    python_path = venv_python(env_dir)
    use_uv = shutil.which("uv") is not None

    # --- Create the virtual environment ----------------------------------- #
    if not python_path.exists():
        if use_uv:
            emit("setup", f"Creating Python environment with uv: {env_dir}")
            run_checked(
                ["uv", "venv", "--python", "3.11", str(env_dir)],
                "Creating virtual environment",
            )
        else:
            emit("setup", f"Creating Python environment: {env_dir}")
            venv.EnvBuilder(with_pip=True).create(env_dir)

    if not python_path.exists():
        raise RuntimeError(
            f"Failed to create the virtual environment at {env_dir}. "
            f"Ensure a compatible system Python (3.10-3.12) is installed and on PATH."
        )

    torch_marker = env_dir / TORCH_MARKER
    deps_marker = env_dir / DEPS_MARKER
    legacy_deps_marker = env_dir / LEGACY_DEPS_MARKER

    # If everything is already installed, short-circuit.
    if torch_marker.exists() and deps_marker.exists():
        return python_path

    # Build the install-command prefix for the chosen installer.
    def install_cmd(extra: list[str]) -> list[str]:
        if use_uv:
            return ["uv", "pip", "install", "--python", str(python_path), *extra]
        return [str(python_path), "-m", "pip", "install", *extra]

    # --- Bootstrap packaging tools (stdlib-venv path only) ---------------- #
    if not use_uv and not torch_marker.exists():
        run_checked(
            install_cmd(["--upgrade", *BOOTSTRAP_PACKAGES]),
            "Updating pip",
        )

    # --- Install PyTorch (CUDA cu128 or CPU) ------------------------------ #
    if not torch_marker.exists():
        want_cuda = device_index >= 0 and nvidia_smi_available()
        if want_cuda:
            run_checked(
                install_cmd(
                    ["--index-url", PYTORCH_CUDA_INDEX_URL, *PYTORCH_PACKAGES]
                ),
                "Installing CUDA PyTorch packages (cu128)",
            )
        else:
            run_checked(
                install_cmd(PYTORCH_PACKAGES),
                "Installing CPU PyTorch packages",
            )
        torch_marker.write_text("ok\n", encoding="utf-8")

    # --- Install diffusers + ML runtime ----------------------------------- #
    if not deps_marker.exists():
        run_checked(
            install_cmd([*ML_PACKAGES]),
            "Installing diffusers ML runtime packages",
        )
        deps_marker.write_text("ok\n", encoding="utf-8")
        legacy_deps_marker.write_text("ok\n", encoding="utf-8")

    return python_path


def relaunch_in_env(job_path: Path, env_dir: Path, device_index: int) -> int:
    """Phase 1: ensure the venv, then re-exec this script with the venv Python.

    Streams the child's stdout straight through to our own stdout so the parent
    Rust process sees a single continuous NDJSON event stream.
    """
    python_path = ensure_env(env_dir, device_index)
    env = os.environ.copy()
    env[IN_VENV_ENV] = "1"
    args = [
        str(python_path),
        str(Path(__file__).resolve()),
        "--job",
        str(job_path),
    ]
    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return process.wait()


# --------------------------------------------------------------------------- #
# Inference (phase 2 — only runs inside the venv; third-party imports allowed)
# --------------------------------------------------------------------------- #

def ensure_model_assets(gguf_filename: str) -> Path:
    """Make model downloads an explicit visible setup phase."""
    from huggingface_hub import hf_hub_download, snapshot_download

    emit(
        "progress",
        "Downloading or verifying Qwen-Image-Edit 2511 pipeline files without 16-bit transformer shards.",
        percent=5,
    )
    snapshot_download(QWEN_MODEL_ID, ignore_patterns=QWEN_TRANSFORMER_IGNORE_PATTERNS)
    emit(
        "progress",
        f"Downloading or verifying Qwen GGUF transformer quant: {gguf_filename}.",
        percent=10,
    )
    gguf_path = hf_hub_download(QWEN_GGUF_MODEL_ID, filename=gguf_filename)
    emit(
        "progress",
        "Downloading or verifying Multiple-Angles LoRA files.",
        percent=15,
    )
    snapshot_download(ANGLES_LORA_ID)
    return Path(gguf_path)


def cuda_memory_gb() -> tuple[float | None, float | None]:
    """Return the current CUDA device's total/free VRAM in GiB, when available."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None, None
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        total_gb = props.total_memory / (1024**3)
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(torch.cuda.current_device())
            total_gb = total_bytes / (1024**3)
            free_gb = free_bytes / (1024**3)
        except Exception:
            free_gb = None
        return total_gb, free_gb
    except Exception:
        return None, None


def cuda_total_vram_gb() -> float | None:
    """Return the current CUDA device's total VRAM in GiB, when available."""
    total_gb, _ = cuda_memory_gb()
    return total_gb


def choose_qwen_gguf_quant(device: str) -> tuple[str, str]:
    """Choose a Qwen GGUF quant from current GPU capacity, with env override."""
    override = os.environ.get("MULTIANGLE_QWEN_GGUF_QUANT", "").strip().upper().replace("-", "_")
    if override:
        if override not in QWEN_GGUF_FILES:
            available = ", ".join(QWEN_GGUF_FILES)
            raise RuntimeError(
                f"Unsupported MULTIANGLE_QWEN_GGUF_QUANT='{override}'. "
                f"Choose one of: {available}."
            )
        emit("setup", f"Using requested Qwen GGUF quant override: {override}.")
        return override, QWEN_GGUF_FILES[override]

    if not device.startswith("cuda"):
        quant = "Q4_K_M"
        emit("setup", f"Selected Qwen GGUF quant for CPU fallback: {quant}.")
        return quant, QWEN_GGUF_FILES[quant]

    total_gb, free_gb = cuda_memory_gb()
    total = total_gb or 0.0
    free = free_gb if free_gb is not None else total

    if total >= 30.0 and free >= 22.0:
        quant = "Q8_0"
    elif total >= 24.0 and free >= 18.0:
        quant = "Q6_K"
    elif total >= 18.0 and free >= 14.0:
        quant = "Q5_K_M"
    elif total >= 14.0 and free >= 10.0:
        quant = "Q4_K_M"
    elif total >= 10.0 and free >= 7.0:
        quant = "Q3_K_M"
    else:
        quant = "Q2_K"

    total_label = f"{total_gb:.1f} GiB" if total_gb is not None else "unknown"
    free_label = f"{free_gb:.1f} GiB" if free_gb is not None else "unknown"
    emit(
        "setup",
        f"Selected Qwen GGUF quant: {quant} for {device} "
        f"(total VRAM {total_label}, free {free_label}).",
    )
    return quant, QWEN_GGUF_FILES.get(quant, QWEN_GGUF_FILES[DEFAULT_QWEN_GGUF_QUANT])


def enable_pipe_offload(pipe: Any, method_name: str, device: str) -> None:
    """Call a diffusers offload method across version-specific signatures."""
    method = getattr(pipe, method_name)
    try:
        method(device=device)
        return
    except TypeError:
        pass

    if device.startswith("cuda:"):
        try:
            method(gpu_id=int(device.split(":", 1)[1]))
            return
        except TypeError:
            pass
    method()


def configure_pipeline_memory(pipe: Any, device: str) -> None:
    """Enable conservative memory settings before inference."""
    if not device.startswith("cuda"):
        pipe.to(device)
        return

    for method_name, label in [
        ("enable_attention_slicing", "Enabled attention slicing."),
        ("enable_vae_slicing", "Enabled VAE slicing."),
        ("enable_vae_tiling", "Enabled VAE tiling."),
    ]:
        method = getattr(pipe, method_name, None)
        if method is None:
            continue
        try:
            method()
            emit("setup", label)
        except Exception:
            pass

    vram_gb = cuda_total_vram_gb()
    prefer_sequential = vram_gb is not None and vram_gb <= LOW_VRAM_OFFLOAD_THRESHOLD_GB

    if prefer_sequential and hasattr(pipe, "enable_sequential_cpu_offload"):
        try:
            enable_pipe_offload(pipe, "enable_sequential_cpu_offload", device)
            emit(
                "setup",
                f"Enabled sequential CPU offload for {vram_gb:.1f} GB GPU memory savings.",
            )
            return
        except Exception as error:  # noqa: BLE001 - fall through to model offload.
            emit("setup", f"Sequential CPU offload unavailable ({error}); trying model offload.")

    if hasattr(pipe, "enable_model_cpu_offload"):
        try:
            enable_pipe_offload(pipe, "enable_model_cpu_offload", device)
            emit("setup", "Enabled model CPU offload for GPU memory savings.")
            return
        except Exception as error:  # noqa: BLE001 - fall through to direct placement.
            emit(
                "setup",
                f"Model CPU offload unavailable ({error}); moving pipeline directly to {device}.",
            )

    pipe.to(device)


def resize_source_for_vram(source: Any) -> Any:
    """Shrink oversized inputs to keep Qwen inference inside 24GB-class VRAM."""
    width, height = source.size
    if width <= 0 or height <= 0:
        return source

    side_scale = min(1.0, MAX_SOURCE_SIDE / max(width, height))
    pixel_scale = min(1.0, (MAX_SOURCE_PIXELS / float(width * height)) ** 0.5)
    scale = min(side_scale, pixel_scale)
    if scale >= 0.999:
        return source

    new_width = max(SOURCE_SIZE_MULTIPLE, int(width * scale))
    new_height = max(SOURCE_SIZE_MULTIPLE, int(height * scale))
    new_width = max(SOURCE_SIZE_MULTIPLE, (new_width // SOURCE_SIZE_MULTIPLE) * SOURCE_SIZE_MULTIPLE)
    new_height = max(SOURCE_SIZE_MULTIPLE, (new_height // SOURCE_SIZE_MULTIPLE) * SOURCE_SIZE_MULTIPLE)

    from PIL import Image

    resampling = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    emit(
        "setup",
        f"Resized source for local VRAM safety: {width}x{height} -> {new_width}x{new_height}.",
    )
    return source.resize((new_width, new_height), resampling)


def load_pipeline(device: str, dtype: Any, gguf_path: Path, quant_name: str) -> Any:
    """Load the Qwen-Image-Edit pipeline and attach the Multiple-Angles LoRA."""
    try:
        from diffusers import (
            GGUFQuantizationConfig,
            QwenImageEditPipeline,
            QwenImageTransformer2DModel,
        )

        emit("setup", f"Loading Qwen GGUF transformer ({quant_name}): {gguf_path.name}")
        transformer = QwenImageTransformer2DModel.from_single_file(
            str(gguf_path),
            config=QWEN_MODEL_ID,
            subfolder="transformer",
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            torch_dtype=dtype,
        )

        emit("setup", f"Loading Qwen Image Edit pipeline: {QWEN_MODEL_ID}")
        pipe = QwenImageEditPipeline.from_pretrained(
            QWEN_MODEL_ID,
            transformer=transformer,
            torch_dtype=dtype,
        )
    except Exception as error:  # noqa: BLE001 - surface a clear model-fetch error.
        raise RuntimeError(
            f"Failed to load the quantized Qwen-Image-Edit model. "
            f"Base='{QWEN_MODEL_ID}', GGUF='{QWEN_GGUF_MODEL_ID}/{gguf_path.name}'.\n{error}"
        ) from error

    # --- Attach the fal Multiple-Angles LoRA ------------------------------ #
    emit("setup", f"Loading Multiple-Angles LoRA: {ANGLES_LORA_ID}")
    try:
        pipe.load_lora_weights(ANGLES_LORA_ID)
    except Exception as error:  # noqa: BLE001 - LoRA load failure is fatal.
        raise RuntimeError(
            f"Failed to load the Multiple-Angles LoRA '{ANGLES_LORA_ID}'. "
            f"Ensure 'peft' is installed and the LoRA repo is reachable.\n{error}"
        ) from error

    # --- Device placement / memory savings -------------------------------- #
    configure_pipeline_memory(pipe, device)

    return pipe


def run_inference(job: dict[str, Any]) -> None:
    """Phase 2 core: load the model, run the edit, and write the PNG."""
    import torch
    from PIL import Image

    image_path = Path(job["imagePath"])
    angle_prompt = str(job.get("anglePrompt", "")).strip()
    extra_prompt = str(job.get("extraPrompt", "") or "").strip()
    device_index = int(job.get("deviceIndex", -1))
    output_path = Path(job["outputPath"])

    if not image_path.exists():
        raise RuntimeError(f"Source image not found: {image_path}")

    # --- Resolve device + dtype ------------------------------------------- #
    if device_index >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "A CUDA device was requested, but torch.cuda is not available."
            )
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {device_index} requested, but only "
                f"{torch.cuda.device_count()} device(s) are available."
            )
        torch.cuda.set_device(device_index)
        device = f"cuda:{device_index}"
        dtype = torch.bfloat16
    else:
        device = "cpu"
        dtype = torch.float32
    emit("setup", f"Using device: {device} (dtype={dtype})")

    quant_name, gguf_filename = choose_qwen_gguf_quant(device)

    # --- Download / verify model cache before pipeline construction -------- #
    gguf_path = ensure_model_assets(gguf_filename)

    # --- Load pipeline ----------------------------------------------------- #
    pipe = load_pipeline(device, dtype, gguf_path, quant_name)

    # --- Build the final prompt ------------------------------------------- #
    # anglePrompt carries the required <sks> trigger token + angle vocabulary.
    # Keep the preservation clause short so it does not drown out the LoRA token.
    prompt_parts = [angle_prompt, DEFAULT_PRESERVATION_GUIDANCE]
    if extra_prompt:
        prompt_parts.append(extra_prompt)
    final_prompt = ", ".join(part for part in prompt_parts if part)
    emit("setup", f"Prompt: {final_prompt}")

    # --- Load the source image -------------------------------------------- #
    emit("progress", "Loading source image", percent=10)
    with Image.open(image_path) as raw:
        source = raw.convert("RGB")
    source = resize_source_for_vram(source)

    # --- Reproducible generator ------------------------------------------- #
    gen_device = device if device.startswith("cuda") else "cpu"
    generator = torch.Generator(device=gen_device).manual_seed(DEFAULT_SEED)

    # --- Run inference ----------------------------------------------------- #
    emit("progress", "Running Qwen-Image-Edit inference", percent=30)

    call_kwargs: dict[str, Any] = {
        "image": source,
        "prompt": final_prompt,
        "num_inference_steps": DEFAULT_STEPS,
        "generator": generator,
    }
    # true_cfg_scale is the Qwen-Image-Edit guidance knob; gracefully degrade to
    # guidance_scale on pipelines (e.g. the AutoPipeline fallback) that lack it.
    def run_pipeline_safely(**kwargs: Any) -> Any:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            return pipe(**kwargs)
        except RuntimeError as error:
            if "out of memory" in str(error).lower():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                raise RuntimeError(
                    "CUDA ran out of VRAM during local generation. Multi-Angle Edit "
                    "used low-memory offload and resized large inputs, but this image "
                    "still needs more free GPU memory. Close other GPU apps, choose a "
                    "GPU with more free VRAM, or switch the provider to Venice Media Local."
                ) from error
            raise

    try:
        result = run_pipeline_safely(
            true_cfg_scale=DEFAULT_TRUE_CFG_SCALE,
            **call_kwargs,
        )
    except TypeError:
        emit("setup", "Pipeline does not accept true_cfg_scale; using guidance_scale.")
        result = run_pipeline_safely(
            guidance_scale=DEFAULT_TRUE_CFG_SCALE,
            **call_kwargs,
        )

    emit("progress", "Saving output image", percent=90)

    # --- Save the result --------------------------------------------------- #
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(output_path, "PNG")

    if not output_path.exists():
        raise RuntimeError(f"Inference completed but no file was written to {output_path}")

    emit("done", outputPath=str(output_path))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-Angle Edit Qwen angle worker")
    parser.add_argument("--job", required=True, help="Path to the job JSON file")
    args = parser.parse_args()

    job_path = Path(args.job)
    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001 - cannot proceed without the job.
        emit("error", f"Failed to read job file '{job_path}': {error}")
        return 1

    # Phase 1: when launched under the system Python, bootstrap + relaunch.
    if os.environ.get(IN_VENV_ENV) != "1":
        try:
            env_dir = Path(job["envDir"])
            device_index = int(job.get("deviceIndex", -1))
            return relaunch_in_env(job_path, env_dir, device_index)
        except Exception as error:  # noqa: BLE001 - bootstrap failure.
            emit("error", str(error))
            return 1

    # Phase 2: inside the venv — go straight to inference.
    try:
        run_inference(job)
        return 0
    except Exception as error:  # noqa: BLE001 - top-level worker error.
        emit("error", str(error))
        return 1


if __name__ == "__main__":
    sys.exit(main())
