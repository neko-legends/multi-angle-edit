# Multi-Angle Edit

Local, GPU-powered **multi-angle image editor**. Load an image, dial in a new
camera angle with an interactive 3D picker, and re-render the shot from that
angle using **Qwen-Image-Edit** plus the **Multiple-Angles LoRA** — all on your
own machine.

Part of the Neko Legends / ForPublic suite. It shares the same Tauri + React +
Python-worker pattern as DepthMap AI Generator and ImageToHunyuan3D, and uses the
shared **NekoLegendsAI** store so PyTorch wheels and Hugging Face models are
downloaded once and reused across every app in the suite.

## What it does

1. **Pick an image.** PNG / JPG / WebP.
2. **Aim the camera.** An interactive Three.js viewport lets you orbit the
   subject: drag to set azimuth (0–360°) and elevation (−30°…60°), scroll to set
   distance (close-up / medium / wide). Sliders and quick-select presets stay in
   sync with the 3D handles.
3. **Choose a GPU.** Detected via `nvidia-smi`; CPU fallback is available but
   slow. Built and tested for an RTX 5090 + RTX 3090 workstation.
4. **Generate.** The angle is formatted into the LoRA's prompt vocabulary
   (`<sks> right side view high-angle shot close-up`), and the image is
   re-rendered from the new viewpoint. Progress streams live; the result lands in
   the output folder next to the app.

## Why standalone (and not just a ComfyUI node)

The angle-picker idea is inspired by the excellent
[ComfyUI-qwenmultiangle](https://github.com/jtydhr88/ComfyUI-qwenmultiangle)
node, but this is an **original standalone app** with no ComfyUI dependency. The
real multi-angle re-render genuinely needs the Qwen-Image-Edit model plus the
`fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA` weights, which is why this app
exists for people with their own GPUs. (A no-GPU "approximate angle" path that
routes the same picker through a hosted general image editor lives in the
companion Venice Media Local app.)

## Requirements

- Windows (other platforms can build from source via Tauri).
- An NVIDIA GPU for real-speed generation. The suite standardizes on **CUDA
  cu128**, the only build that covers both Blackwell (RTX 5090, sm_120) and
  Ampere (RTX 3090, sm_86).
- Python 3.10, 3.11, or 3.12 on PATH (used once to bootstrap the app-managed
  virtual environment). `uv` is used automatically when present for fast,
  hardlinked installs.
- Node.js 20+ and the Rust toolchain to build from source.

## First run

On the first generation the app creates its own virtual environment and installs
PyTorch (cu128) + diffusers, then downloads the Qwen-Image-Edit model and the
Multiple-Angles LoRA into the shared store. This is a large one-time download.
Subsequent runs — and other suite apps that need the same wheels — reuse the
shared cache.

## The shared NekoLegendsAI store

Models and caches live in a single visible folder (default `D:\NekoLegendsAI`)
shared across the suite:

```
NekoLegendsAI/
  models/   HF_HOME — models downloaded once
  cache/    pip / uv / torch caches
  envs/
    multiangleedit/   this app's virtual environment
```

The app discovers the store automatically and falls back to an app-local store
if none is available, so it always works standalone. See
[NekoLegendsAI-Shared](../NekoLegendsAI-Shared) for the resolver details. The
resolved store path is shown in the app footer.

## Development

```bash
npm install
npm run tauri:dev      # run the app in dev mode
npm run tauri:build    # produce an installer under src-tauri/target/release/bundle/nsis
```

The Python worker lives at `python/qwen_angle_worker.py` and is bundled as a
Tauri resource. It is a two-phase script: launched under the system Python it
bootstraps the venv, then re-execs itself inside that venv to run inference.

## License

MIT.
