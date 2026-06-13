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
ML_PACKAGES = [
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "pillow",
    "huggingface_hub",
    "peft",
]

# Marker files inside envDir so re-runs skip already-completed install steps.
TORCH_MARKER = ".multiangle-torch-ready"
DEPS_MARKER = ".multiangle-deps-ready"

# Hugging Face model + LoRA identifiers.
QWEN_MODEL_ID = "Qwen/Qwen-Image-Edit"
ANGLES_LORA_ID = "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA"

# Inference defaults.
DEFAULT_STEPS = 30
DEFAULT_TRUE_CFG_SCALE = 4.0
DEFAULT_SEED = 42


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

def load_pipeline(device: str, dtype: Any) -> Any:
    """Load the Qwen-Image-Edit pipeline and attach the Multiple-Angles LoRA."""
    import torch  # noqa: F401  (ensures torch is importable before model load)
    from diffusers import AutoPipelineForImage2Image

    emit("setup", f"Loading Qwen Image Edit pipeline: {QWEN_MODEL_ID}")

    pipe = None
    try:
        # Preferred path: the dedicated Qwen-Image-Edit pipeline class.
        from diffusers import QwenImageEditPipeline

        pipe = QwenImageEditPipeline.from_pretrained(
            QWEN_MODEL_ID,
            torch_dtype=dtype,
        )
    except ImportError:
        emit(
            "setup",
            "QwenImageEditPipeline not available on this diffusers version; "
            "falling back to AutoPipelineForImage2Image.",
        )
        pipe = AutoPipelineForImage2Image.from_pretrained(
            QWEN_MODEL_ID,
            torch_dtype=dtype,
        )
    except Exception as error:  # noqa: BLE001 - surface a clear model-fetch error.
        raise RuntimeError(
            f"Failed to load the Qwen-Image-Edit model '{QWEN_MODEL_ID}'. "
            f"The model may be unavailable or the download was interrupted.\n{error}"
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
    if device.startswith("cuda"):
        # CPU offload keeps the model footprint small enough for a 24GB 3090.
        # It requires accelerate, so guard it and fall back to a direct .to().
        try:
            pipe.enable_model_cpu_offload(device=device)
            emit("setup", "Enabled model CPU offload for GPU memory savings.")
        except Exception as error:  # noqa: BLE001 - offload is best-effort.
            emit(
                "setup",
                f"Model CPU offload unavailable ({error}); moving pipeline "
                f"directly to {device}.",
            )
            pipe.to(device)
    else:
        pipe.to(device)

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

    # --- Load pipeline ----------------------------------------------------- #
    pipe = load_pipeline(device, dtype)

    # --- Build the final prompt ------------------------------------------- #
    # anglePrompt already carries the required <sks> trigger token + angle
    # vocabulary from the 3D picker; we only append optional extra guidance.
    final_prompt = angle_prompt + (f", {extra_prompt}" if extra_prompt else "")
    emit("setup", f"Prompt: {final_prompt}")

    # --- Load the source image -------------------------------------------- #
    emit("progress", "Loading source image", percent=10)
    with Image.open(image_path) as raw:
        source = raw.convert("RGB")

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
    try:
        result = pipe(true_cfg_scale=DEFAULT_TRUE_CFG_SCALE, **call_kwargs)
    except TypeError:
        emit("setup", "Pipeline does not accept true_cfg_scale; using guidance_scale.")
        result = pipe(guidance_scale=DEFAULT_TRUE_CFG_SCALE, **call_kwargs)

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
