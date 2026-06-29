// App.tsx — Multi-Angle Edit main UI.
//
// Three-column creative tool:
//   left   = source image select + before/after thumbnails
//   center = the interactive 3D AnglePicker + live formatted prompt
//   right  = GPU select, angle sliders + preset dropdowns, extra guidance,
//            Generate button, progress bar, and a scrolling event log.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import {
  Image as ImageIcon,
  Cpu,
  FolderOpen,
  Sparkles,
  Wand2,
  Check,
  Copy,
} from "lucide-react";
import packageJson from "../package.json";
import AnglePicker from "./components/AnglePicker";
import {
  formatAnglePrompt,
  AZIMUTH_PRESETS,
  ELEVATION_PRESETS,
  DISTANCE_PRESETS,
  wrapAzimuth,
  clamp,
  type AngleValue,
} from "./lib/anglePrompt";

// ---- types mirrored from the Rust backend --------------------------------
interface GpuDevice {
  index: number;
  name: string;
  vramMb: number | null;
}

interface StoreInfo {
  root: string;
  shared: boolean;
  envDir: string;
  modelsDir: string;
}

interface AgentServerStatus {
  enabled: boolean;
  port: number;
  url: string;
  openapiUrl: string;
  busy: boolean;
  activeJobId: string | null;
  message: string;
}

interface LocalModelStatus {
  envReady: boolean;
  torchReady: boolean;
  depsReady: boolean;
  modelReady: boolean;
  loraReady: boolean;
  storeRoot: string;
  envDir: string;
  modelsDir: string;
  message: string;
}

interface VeniceMediaStatus {
  available: boolean;
  message: string;
  url: string | null;
  model: string | null;
}

interface GenerationResult {
  provider: string;
  outputPath?: string | null;
  dataUrl?: string | null;
  message: string;
}

type WorkerEvent =
  | { type: "setup"; message: string }
  | { type: "progress"; message: string; percent?: number }
  | { type: "done"; outputPath: string }
  | { type: "error"; message: string };

interface LogLine {
  id: number;
  kind: "setup" | "progress" | "done" | "error" | "info";
  text: string;
}

type GenerationProvider = "local" | "venice";

const DEFAULT_ANGLE: AngleValue = { azimuth: 0, elevation: 0, distance: 5 };
const APP_VERSION = packageJson.version;
const IMAGE_EXTENSIONS = new Set(["png", "jpg", "jpeg", "webp"]);
const PROMPT_ADDITION_PRESETS = [
  {
    label: "Preserve source",
    text: "preserve the same subject identity, outfit, materials, background, lighting, and color palette",
  },
  {
    label: "Keep background",
    text: "keep the original background and scene layout",
  },
  {
    label: "Match lighting",
    text: "match the original lighting direction and color mood",
  },
];

function normalizeLocalImagePath(path: string) {
  const trimmed = path.trim();
  if (!trimmed) return trimmed;

  let normalized = trimmed;
  if (trimmed.startsWith("file://")) {
    try {
      const url = new URL(trimmed);
      const pathName = decodeURIComponent(url.pathname);
      normalized =
        url.hostname && url.hostname !== "localhost"
          ? `//${url.hostname}${pathName}`
          : pathName;
    } catch {
      normalized = trimmed.replace(/^file:\/+/, "");
    }
  }

  if (/^\/[a-zA-Z]:\//.test(normalized)) {
    normalized = normalized.slice(1);
  }

  return normalized.startsWith("\\\\?\\") ? normalized.slice(4) : normalized;
}

function isSupportedImagePath(path: string) {
  const normalized = normalizeLocalImagePath(path);
  const extension = normalized.split(".").pop()?.toLowerCase();
  return extension ? IMAGE_EXTENSIONS.has(extension) : false;
}

function hasTauriRuntime() {
  return "__TAURI_INTERNALS__" in window;
}

function parseTqdmPercent(message: string) {
  const match = message.match(/(\d{1,3})%\|/);
  if (!match) return null;
  return clamp(Number(match[1]), 0, 100);
}

function inferWorkerProgress(message: string, previous: number | null) {
  const current = previous ?? 0;
  const text = message.trim();
  const tqdmPercent = parseTqdmPercent(text);

  let inferred: number | null = null;
  if (text.includes("Creating Python environment")) inferred = 3;
  else if (text.includes("Installing CUDA PyTorch")) inferred = 8;
  else if (text.includes("Installing CPU PyTorch")) inferred = 8;
  else if (text.includes("Installing diffusers")) inferred = 16;
  else if (text.includes("Downloading dependency packages")) inferred = 12;
  else if (text.includes("Dependency install step finished")) inferred = 20;
  else if (text.includes("Downloading or verifying Qwen")) inferred = 22;
  else if (text.startsWith("Fetching ") && tqdmPercent != null) {
    inferred = 22 + tqdmPercent * 0.18;
  } else if (text.includes("Downloading or verifying Multiple-Angles")) {
    inferred = 40;
  } else if (text.includes("Loading Qwen Image Edit pipeline")) inferred = 42;
  else if (text.startsWith("Loading pipeline components") && tqdmPercent != null) {
    inferred = 42 + tqdmPercent * 0.16;
  } else if (
    (text.startsWith("Loading weights") ||
      text.startsWith("Loading checkpoint shards")) &&
    tqdmPercent != null
  ) {
    inferred = 48 + tqdmPercent * 0.1;
  } else if (text.includes("Loading Multiple-Angles LoRA")) inferred = 60;
  else if (text.includes("Enabled") && text.includes("offload")) inferred = 64;
  else if (text.includes("Loading source image")) inferred = 66;
  else if (text.includes("Running Qwen-Image-Edit inference")) inferred = 70;
  else if (text.includes("/30") && tqdmPercent != null) {
    inferred = 70 + tqdmPercent * 0.2;
  } else if (text.includes("Saving output image")) inferred = 92;

  return inferred == null ? current : Math.max(current, Math.round(inferred));
}

export default function App() {
  const [imagePath, setImagePath] = useState<string | undefined>(undefined);
  const [angle, setAngle] = useState<AngleValue>(DEFAULT_ANGLE);
  const [gpus, setGpus] = useState<GpuDevice[]>([]);
  const [deviceIndex, setDeviceIndex] = useState<number>(0);
  const [generationProvider, setGenerationProvider] =
    useState<GenerationProvider>("local");
  const [extraPrompt, setExtraPrompt] = useState<string>("");
  const [store, setStore] = useState<StoreInfo | null>(null);
  const [localStatus, setLocalStatus] = useState<LocalModelStatus | null>(null);
  const [veniceStatus, setVeniceStatus] = useState<VeniceMediaStatus | null>(
    null,
  );

  const [agent, setAgent] = useState<AgentServerStatus | null>(null);
  const [agentPort, setAgentPort] = useState<number>(17339);
  const [agentBusy, setAgentBusy] = useState(false);

  const [running, setRunning] = useState(false);
  const [percent, setPercent] = useState<number | null>(null);
  const [log, setLog] = useState<LogLine[]>([]);
  const [logCopied, setLogCopied] = useState(false);
  const [outputPath, setOutputPath] = useState<string | undefined>(undefined);
  const [outputDir, setOutputDir] = useState<string | undefined>(undefined);
  const [outputPreviewSrc, setOutputPreviewSrc] = useState<string | undefined>(
    undefined,
  );
  const [imagePreviewSrc, setImagePreviewSrc] = useState<string | undefined>(
    undefined,
  );
  const [runStatus, setRunStatus] = useState<string>("");
  const [beforeDropActive, setBeforeDropActive] = useState(false);

  const logIdRef = useRef(0);
  const logEndRef = useRef<HTMLDivElement | null>(null);
  const logCopyResetRef = useRef<number | null>(null);
  const beforeFrameRef = useRef<HTMLDivElement | null>(null);
  const dragHasSupportedImageRef = useRef(false);
  const imagePreviewRequestIdRef = useRef(0);
  const outputPreviewRequestIdRef = useRef(0);

  const anglePrompt = useMemo(() => formatAnglePrompt(angle), [angle]);
  const trimmedExtraPrompt = extraPrompt.trim();
  const busyLog = useMemo(() => log.slice(-6), [log]);
  const logText = useMemo(() => log.map((line) => line.text).join("\n"), [log]);

  const pushLog = useCallback((kind: LogLine["kind"], text: string) => {
    setLog((prev) => {
      const next = [...prev, { id: logIdRef.current++, kind, text }];
      // Cap the log to keep the DOM light.
      return next.length > 400 ? next.slice(next.length - 400) : next;
    });
  }, []);

  const copyLog = useCallback(async () => {
    if (!logText) return;

    try {
      const copyWithTextarea = () => {
        const textarea = document.createElement("textarea");
        textarea.value = logText;
        textarea.setAttribute("readonly", "");
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        textarea.style.pointerEvents = "none";
        document.body.appendChild(textarea);
        textarea.select();
        const copied = document.execCommand("copy");
        textarea.remove();
        if (!copied) throw new Error("clipboard copy was not available");
      };

      if (navigator.clipboard?.writeText) {
        try {
          await navigator.clipboard.writeText(logText);
        } catch {
          copyWithTextarea();
        }
      } else {
        copyWithTextarea();
      }

      setLogCopied(true);
      if (logCopyResetRef.current != null) {
        window.clearTimeout(logCopyResetRef.current);
      }
      logCopyResetRef.current = window.setTimeout(() => {
        setLogCopied(false);
        logCopyResetRef.current = null;
      }, 1400);
    } catch (e) {
      pushLog("error", `Log copy failed: ${String(e)}`);
    }
  }, [logText, pushLog]);

  const appendPromptAddition = useCallback((addition: string) => {
    setExtraPrompt((current) => {
      const trimmedCurrent = current.trim();
      if (!trimmedCurrent) return addition;
      if (trimmedCurrent.toLowerCase().includes(addition.toLowerCase())) {
        return current;
      }
      return `${trimmedCurrent}; ${addition}`;
    });
  }, []);

  const loadSourceImage = useCallback(
    (path: string, action = "Loaded") => {
      const normalizedPath = normalizeLocalImagePath(path);
      const requestId = ++imagePreviewRequestIdRef.current;
      setImagePath(normalizedPath);
      setImagePreviewSrc(undefined);
      setOutputPath(undefined);
      setOutputDir(undefined);
      setOutputPreviewSrc(undefined);
      setPercent(null);
      setRunStatus("");
      pushLog("info", `${action} image: ${normalizedPath}`);

      invoke<string>("image_preview_data_url", { path: normalizedPath })
        .then((previewSrc) => {
          if (requestId === imagePreviewRequestIdRef.current) {
            setImagePreviewSrc(previewSrc);
          }
        })
        .catch((e) => {
          if (requestId !== imagePreviewRequestIdRef.current) return;
          setImagePath(undefined);
          pushLog("error", `Image preview failed: ${String(e)}`);
        });
    },
    [pushLog],
  );

  const refreshLocalStatus = useCallback(() => {
    if (!hasTauriRuntime()) return;

    invoke<LocalModelStatus>("local_model_status")
      .then(setLocalStatus)
      .catch((e) => pushLog("error", `Local model status failed: ${String(e)}`));
  }, [pushLog]);

  const refreshVeniceStatus = useCallback(() => {
    if (!hasTauriRuntime()) return;

    invoke<VeniceMediaStatus>("venice_media_status")
      .then(setVeniceStatus)
      .catch((e) =>
        setVeniceStatus({
          available: false,
          message: `Venice Media status failed: ${String(e)}`,
          url: null,
          model: null,
        }),
      );
  }, []);

  const loadOutputPreview = useCallback(
    (path: string) => {
      const requestId = ++outputPreviewRequestIdRef.current;
      setOutputPreviewSrc(undefined);
      invoke<string>("image_preview_data_url", { path })
        .then((previewSrc) => {
          if (requestId === outputPreviewRequestIdRef.current) {
            setOutputPreviewSrc(previewSrc);
          }
        })
        .catch((e) => pushLog("error", `Output preview failed: ${String(e)}`));
    },
    [pushLog],
  );

  // ---- initial backend queries -------------------------------------------
  useEffect(() => {
    if (!hasTauriRuntime()) return;

    invoke<GpuDevice[]>("list_gpus")
      .then((devices) => {
        setGpus(devices);
        // Prefer the first real GPU; fall back to whatever exists.
        const firstGpu = devices.find((d) => d.index >= 0);
        setDeviceIndex(firstGpu ? firstGpu.index : devices[0]?.index ?? -1);
      })
      .catch((e) => pushLog("error", `Failed to list GPUs: ${String(e)}`));

    invoke<StoreInfo>("store_info")
      .then(setStore)
      .catch((e) => pushLog("error", `Failed to read store info: ${String(e)}`));

    invoke<AgentServerStatus>("get_agent_server_status")
      .then((status) => {
        setAgent(status);
        if (status.port > 0) setAgentPort(status.port);
      })
      .catch((e) =>
        pushLog("error", `Failed to read agent status: ${String(e)}`),
      );
  }, [pushLog]);

  useEffect(() => {
    if (!hasTauriRuntime()) return;

    refreshLocalStatus();
    refreshVeniceStatus();
    const interval = window.setInterval(refreshVeniceStatus, 8000);
    return () => window.clearInterval(interval);
  }, [refreshLocalStatus, refreshVeniceStatus]);

  // ---- worker event stream ------------------------------------------------
  useEffect(() => {
    if (!hasTauriRuntime()) return;

    let unlisten: UnlistenFn | undefined;
    listen<WorkerEvent>("angle-worker-event", (event) => {
      const payload = event.payload;
      switch (payload.type) {
        case "setup":
          pushLog("setup", payload.message);
          setRunStatus(payload.message);
          setPercent((prev) => inferWorkerProgress(payload.message, prev));
          break;
        case "progress":
          pushLog("progress", payload.message);
          setRunStatus(payload.message);
          setPercent((prev) => {
            const explicit =
              typeof payload.percent === "number"
                ? clamp(payload.percent, 0, 100)
                : null;
            const inferred = inferWorkerProgress(payload.message, prev);
            return Math.max(prev ?? 0, explicit ?? 0, inferred);
          });
          break;
        case "done":
          pushLog("done", `Done: ${payload.outputPath}`);
          setRunStatus("Generation finished.");
          setPercent(100);
          break;
        case "error":
          pushLog("error", payload.message);
          setRunStatus(payload.message);
          break;
      }
    })
      .then((fn) => {
        unlisten = fn;
      })
      .catch((e) => pushLog("error", `Event stream failed: ${String(e)}`));
    return () => {
      if (unlisten) unlisten();
    };
  }, [pushLog]);

  // Auto-scroll the log.
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [log]);

  useEffect(() => {
    return () => {
      if (logCopyResetRef.current != null) {
        window.clearTimeout(logCopyResetRef.current);
      }
    };
  }, []);

  // Native Tauri drag/drop gives us real file paths; accept drops only over
  // the Before frame so the rest of the app keeps its normal drag behavior.
  useEffect(() => {
    if (!hasTauriRuntime()) return;

    const isOverBeforeFrame = (position: { x: number; y: number }) => {
      const rect = beforeFrameRef.current?.getBoundingClientRect();
      if (!rect) return false;
      const scale = window.devicePixelRatio || 1;
      const x = position.x / scale;
      const y = position.y / scale;
      return x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom;
    };

    let unlisten: UnlistenFn | undefined;
    getCurrentWebview()
      .onDragDropEvent((event) => {
        const payload = event.payload;

        if (payload.type === "leave") {
          dragHasSupportedImageRef.current = false;
          setBeforeDropActive(false);
          return;
        }

        if (payload.type === "enter") {
          dragHasSupportedImageRef.current = payload.paths.some(isSupportedImagePath);
          setBeforeDropActive(
            !running &&
              dragHasSupportedImageRef.current &&
              isOverBeforeFrame(payload.position),
          );
          return;
        }

        if (payload.type === "over") {
          setBeforeDropActive(
            !running &&
              dragHasSupportedImageRef.current &&
              isOverBeforeFrame(payload.position),
          );
          return;
        }

        const insideBeforeFrame = isOverBeforeFrame(payload.position);
        dragHasSupportedImageRef.current = false;
        setBeforeDropActive(false);
        if (!insideBeforeFrame || running) return;

        const droppedImage = payload.paths.find(isSupportedImagePath);
        if (droppedImage) {
          loadSourceImage(droppedImage, "Dropped");
        } else {
          pushLog("error", "Drop a PNG, JPG, JPEG, or WebP image.");
        }
      })
      .then((fn) => {
        unlisten = fn;
      })
      .catch((e) => pushLog("error", `Image drag/drop failed: ${String(e)}`));

    return () => {
      if (unlisten) unlisten();
    };
  }, [loadSourceImage, pushLog, running]);

  // ---- actions ------------------------------------------------------------
  const pickImage = useCallback(async () => {
    try {
      const selected = await openDialog({
        filters: [
          { name: "Images", extensions: ["png", "jpg", "jpeg", "webp"] },
        ],
        multiple: false,
      });
      if (typeof selected === "string") {
        loadSourceImage(selected);
      }
    } catch (e) {
      pushLog("error", `Image selection failed: ${String(e)}`);
    }
  }, [loadSourceImage, pushLog]);

  const generate = useCallback(async () => {
    if (!imagePath || running) return;
    setRunning(true);
    setPercent(0);
    setOutputPath(undefined);
    setOutputDir(undefined);
    setOutputPreviewSrc(undefined);
    try {
      const request = {
        imagePath,
        anglePrompt,
        deviceIndex,
        extraPrompt: trimmedExtraPrompt || undefined,
      };

      if (generationProvider === "venice") {
        setRunStatus("Sending image edit to Venice Media Local...");
        setPercent(10);
        pushLog("info", `Generating online with Venice Media: ${anglePrompt}`);
        const result = await invoke<GenerationResult>("generate_angle_venice", {
          request,
        });
        if (result.dataUrl) {
          setOutputPreviewSrc(result.dataUrl);
        }
        if (result.outputPath) {
          setOutputPath(result.outputPath);
          const idx = Math.max(
            result.outputPath.lastIndexOf("\\"),
            result.outputPath.lastIndexOf("/"),
          );
          setOutputDir(idx > 0 ? result.outputPath.slice(0, idx) : undefined);
          if (!result.dataUrl) loadOutputPreview(result.outputPath);
          pushLog("done", `Venice output saved: ${result.outputPath}`);
        } else {
          pushLog("done", result.message);
        }
        setRunStatus(result.message);
        setPercent(100);
        refreshVeniceStatus();
        return;
      }

      if (localStatus) {
        pushLog("info", localStatus.message);
        setRunStatus(localStatus.message);
      }
      pushLog("info", `Generating locally: ${anglePrompt}`);
      const result = await invoke<string>("generate_angle", { request });
      setOutputPath(result);
      const idx = Math.max(result.lastIndexOf("\\"), result.lastIndexOf("/"));
      setOutputDir(idx > 0 ? result.slice(0, idx) : undefined);
      loadOutputPreview(result);
      setPercent(100);
      setRunStatus("Local generation finished.");
      pushLog("done", `Output saved: ${result}`);
      refreshLocalStatus();
    } catch (e) {
      setRunStatus(`Generation failed: ${String(e)}`);
      pushLog("error", `Generation failed: ${String(e)}`);
    } finally {
      setRunning(false);
    }
  }, [
    imagePath,
    running,
    anglePrompt,
    deviceIndex,
    trimmedExtraPrompt,
    generationProvider,
    localStatus,
    loadOutputPreview,
    pushLog,
    refreshLocalStatus,
    refreshVeniceStatus,
  ]);

  const openOutputFolder = useCallback(() => {
    if (!outputDir) return;
    invoke("open_path", { path: outputDir }).catch((e) =>
      pushLog("error", `Could not open folder: ${String(e)}`),
    );
  }, [outputDir, pushLog]);

  const toggleAgent = useCallback(
    async (enabled: boolean) => {
      setAgentBusy(true);
      try {
        const status = await invoke<AgentServerStatus>(
          "set_agent_server_enabled",
          { enabled, port: agentPort },
        );
        setAgent(status);
        if (status.port > 0) setAgentPort(status.port);
        pushLog(
          "info",
          enabled
            ? `Agent API enabled at ${status.url}`
            : "Agent API disabled.",
        );
      } catch (e) {
        pushLog("error", `Agent API change failed: ${String(e)}`);
      } finally {
        setAgentBusy(false);
      }
    },
    [agentPort, pushLog],
  );

  const providerStatus =
    generationProvider === "local"
      ? localStatus?.message ??
        "Local Qwen GGUF runtime status will appear here after startup."
      : veniceStatus?.message ??
        "Checking Venice Media Local link...";
  const providerReady =
    generationProvider === "local" || (veniceStatus?.available ?? false);
  const busyTitle =
    generationProvider === "venice"
      ? "Sending to Venice Media Local"
      : localStatus?.modelReady && localStatus.loraReady
        ? "Generating locally"
        : "Preparing local Qwen GGUF model";
  const busyNote =
    generationProvider === "venice"
      ? "The online edit is running through Venice Media Local."
      : localStatus?.modelReady && localStatus.loraReady
        ? "Local inference is running. The app will unlock when the image is ready."
        : "First local run can install packages and download the selected Qwen GGUF quant.";
  const busyPercent = percent == null ? 0 : clamp(percent, 0, 100);

  // ---- render -------------------------------------------------------------
  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <span className="brand-name">Multi-Angle Edit</span>
          <span className="brand-version">v{APP_VERSION}</span>
        </div>
        <span className="brand-tagline">
          Re-light any subject from a brand-new camera angle
        </span>
      </header>

      <main className="layout">
        {/* LEFT: image + before/after */}
        <section className="col col-left">
          <div className="card">
            <div className="card-title">
              <ImageIcon size={16} /> Source image
            </div>
            <button className="picker-btn" onClick={pickImage} disabled={running}>
              <FolderOpen size={16} /> Choose image…
            </button>
            <div className="thumb-stack">
              <figure className="thumb">
                <figcaption>Before</figcaption>
                <div
                  ref={beforeFrameRef}
                  className={`thumb-frame thumb-drop-target${
                    beforeDropActive ? " thumb-drop-active" : ""
                  }`}
                  aria-label="Drop source image here"
                >
                  {imagePreviewSrc ? (
                    <img src={imagePreviewSrc} alt="source" />
                  ) : imagePath ? (
                    <div className="thumb-empty">Loading image…</div>
                  ) : (
                    <div className="thumb-empty">Drop image here or choose one</div>
                  )}
                  {beforeDropActive && (
                    <div className="drop-overlay">Drop image</div>
                  )}
                </div>
              </figure>
              <figure className="thumb">
                <figcaption>After</figcaption>
                <div className="thumb-frame">
                  {outputPreviewSrc ? (
                    <img src={outputPreviewSrc} alt="result" />
                  ) : outputPath ? (
                    <div className="thumb-empty">Loading result…</div>
                  ) : (
                    <div className="thumb-empty">
                      {running ? "Rendering…" : "Result appears here"}
                    </div>
                  )}
                </div>
              </figure>
            </div>
            {outputPath && (
              <button className="ghost-btn" onClick={openOutputFolder}>
                <FolderOpen size={15} /> Open output folder
              </button>
            )}
          </div>
        </section>

        {/* CENTER: 3D picker + prompt readout */}
        <section className="col col-center">
          <div className="card viewport-card">
            <div className="card-title">
              <Wand2 size={16} /> Angle viewport
              <span className="hint">
                left-drag to orbit · middle-drag/scroll to zoom
              </span>
            </div>
            <div className="viewport-wrap">
              <AnglePicker
                imagePreviewSrc={imagePreviewSrc}
                value={angle}
                onChange={setAngle}
              />
            </div>
            <div className="prompt-readout">
              <span className="prompt-label">angle prompt</span>
              <code>{anglePrompt}</code>
            </div>
          </div>
        </section>

        {/* RIGHT: controls */}
        <section className="col col-right">
          <div className="card provider-panel">
            <div className="card-title">
              <Sparkles size={16} /> Generation provider
            </div>
            <select
              className="select"
              value={generationProvider}
              onChange={(e) =>
                setGenerationProvider(e.target.value as GenerationProvider)
              }
              disabled={running}
            >
              <option value="local">Local GPU - Qwen GGUF Edit</option>
              <option value="venice">Online - Venice Media Local</option>
            </select>
            <div
              className={`provider-status ${
                providerReady ? "provider-ok" : "provider-warn"
              }`}
            >
              {providerStatus}
            </div>
          </div>

          <div className="card">
            <div className="card-title">
              <Cpu size={16} /> Compute device
            </div>
            <select
              className="select"
              value={deviceIndex}
              onChange={(e) => setDeviceIndex(Number(e.target.value))}
              disabled={running || generationProvider !== "local"}
            >
              {gpus.map((g) => (
                <option key={g.index} value={g.index}>
                  {g.name}
                  {g.vramMb != null
                    ? ` — ${(g.vramMb / 1024).toFixed(1)} GB`
                    : ""}
                </option>
              ))}
            </select>
          </div>

          <div className="card">
            <div className="card-title">Camera angle</div>

            <AngleControl
              label="Azimuth"
              unit="°"
              min={0}
              max={360}
              step={1}
              value={angle.azimuth}
              onChange={(v) =>
                setAngle((a) => ({ ...a, azimuth: wrapAzimuth(v) }))
              }
              presets={AZIMUTH_PRESETS}
              disabled={running}
            />
            <AngleControl
              label="Elevation"
              unit="°"
              min={-30}
              max={60}
              step={1}
              value={angle.elevation}
              onChange={(v) =>
                setAngle((a) => ({ ...a, elevation: clamp(v, -30, 60) }))
              }
              presets={ELEVATION_PRESETS}
              disabled={running}
            />
            <AngleControl
              label="Distance"
              unit=""
              min={0}
              max={10}
              step={0.1}
              value={angle.distance}
              onChange={(v) =>
                setAngle((a) => ({ ...a, distance: clamp(v, 0, 10) }))
              }
              presets={DISTANCE_PRESETS}
              disabled={running}
            />
          </div>

          <div className="card">
            <div className="card-title">Prompt additions (optional)</div>
            <div className="prompt-preset-row">
              {PROMPT_ADDITION_PRESETS.map((preset) => (
                <button
                  key={preset.label}
                  className="prompt-chip"
                  type="button"
                  onClick={() => appendPromptAddition(preset.text)}
                  disabled={running}
                >
                  {preset.label}
                </button>
              ))}
            </div>
            <textarea
              className="textarea"
              placeholder="e.g. add soft rim light, keep the same dress, make the room moodier"
              value={extraPrompt}
              onChange={(e) => setExtraPrompt(e.target.value)}
              disabled={running}
              rows={2}
            />
          </div>

          <button
            className="generate-btn"
            onClick={generate}
            disabled={running || !imagePath || !providerReady}
          >
            <Sparkles size={18} />
            {running
              ? generationProvider === "venice"
                ? "Sending…"
                : "Forging…"
              : "Generate"}
          </button>

          {(running || runStatus || percent != null) && (
            <div className="run-panel">
              {runStatus && <div className="run-status">{runStatus}</div>}
              {(running || percent != null) && (
                <div className="progress">
                  <div
                    className="progress-bar"
                    style={{ width: `${percent ?? 0}%` }}
                  />
                </div>
              )}
            </div>
          )}

          <div className="card log-card">
            <div className="card-title log-title">
              <span>Log</span>
              <button
                className="icon-btn log-copy-btn"
                type="button"
                onClick={copyLog}
                disabled={log.length === 0}
                title={logCopied ? "Copied log" : "Copy log"}
                aria-label={logCopied ? "Copied log" : "Copy log"}
              >
                {logCopied ? <Check size={14} /> : <Copy size={14} />}
              </button>
            </div>
            <div className="log">
              {log.length === 0 ? (
                <div className="log-empty">Activity will appear here.</div>
              ) : (
                log.map((l) => (
                  <div key={l.id} className={`log-line log-${l.kind}`}>
                    {l.text}
                  </div>
                ))
              )}
              <div ref={logEndRef} />
            </div>
          </div>

          <div className="card agent-panel">
            <div className="card-title">
              <Cpu size={16} /> AI Agent Control
            </div>
            <p className="agent-desc">
              Local HTTP API for AI agents. Off by default, binds to
              127.0.0.1.
            </p>
            <div className="agent-row">
              <label className="agent-toggle">
                <input
                  type="checkbox"
                  checked={agent?.enabled ?? false}
                  disabled={agentBusy}
                  onChange={(e) => toggleAgent(e.target.checked)}
                />
                <span>{agent?.enabled ? "Enabled" : "Disabled"}</span>
              </label>
              <div className="agent-port">
                <label>Port</label>
                <input
                  className="agent-port-input"
                  type="number"
                  min={1}
                  max={65535}
                  value={agentPort}
                  disabled={agentBusy || (agent?.enabled ?? false)}
                  onChange={(e) =>
                    setAgentPort(Number(e.target.value) || 17339)
                  }
                />
              </div>
            </div>
            {agent?.enabled && (
              <div className="agent-urls">
                <div>
                  <span className="agent-url-label">url</span>
                  <code>{agent.url}</code>
                </div>
                <div>
                  <span className="agent-url-label">openapi</span>
                  <code>{agent.openapiUrl}</code>
                </div>
              </div>
            )}
          </div>
        </section>
      </main>

      <footer className="app-footer">
        {store ? (
          <span>
            Models: <code>{store.root}</code> (
            {store.shared ? "shared" : "local"})
          </span>
        ) : (
          <span>Resolving store…</span>
        )}
        {localStatus && (
          <span className="footer-status">
            {localStatus.modelReady && localStatus.loraReady
              ? "Qwen GGUF cache ready"
              : "Qwen GGUF download needed on first local run"}
          </span>
        )}
      </footer>

      {running && (
        <div className="busy-overlay" role="alertdialog" aria-modal="true">
          <div className="busy-modal">
            <div className="busy-kicker">
              {generationProvider === "venice" ? "Online provider" : "Local provider"}
            </div>
            <div className="busy-heading-row">
              <h2>{busyTitle}</h2>
              <span>{Math.round(busyPercent)}%</span>
            </div>
            <div className="progress progress-large">
              <div
                className="progress-bar"
                style={{ width: `${busyPercent}%` }}
              />
            </div>
            <p className="busy-status">
              {runStatus || "Starting generation..."}
            </p>
            <p className="busy-note">{busyNote}</p>
            {busyLog.length > 0 && (
              <div className="busy-log">
                {busyLog.map((line) => (
                  <div key={line.id} className={`log-line log-${line.kind}`}>
                    {line.text}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ---- a labelled slider + numeric field + preset dropdown ------------------
interface AngleControlProps {
  label: string;
  unit: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (value: number) => void;
  presets: { label: string; value: number }[];
  disabled?: boolean;
}

function AngleControl({
  label,
  unit,
  min,
  max,
  step,
  value,
  onChange,
  presets,
  disabled,
}: AngleControlProps) {
  // Selecting a preset sets the value; show "Custom" when off a preset.
  const matched = presets.find((p) => Math.abs(p.value - value) < 0.001);
  return (
    <div className="angle-control">
      <div className="angle-control-head">
        <label>{label}</label>
        <span className="angle-value">
          {step < 1 ? value.toFixed(1) : Math.round(value)}
          {unit}
        </span>
      </div>
      <input
        className="slider"
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        disabled={disabled}
      />
      <select
        className="select select-sm"
        value={matched ? String(matched.value) : "custom"}
        onChange={(e) => {
          if (e.target.value !== "custom") onChange(Number(e.target.value));
        }}
        disabled={disabled}
      >
        {!matched && <option value="custom">Custom</option>}
        {presets.map((p) => (
          <option key={p.label} value={String(p.value)}>
            {p.label}
          </option>
        ))}
      </select>
    </div>
  );
}
