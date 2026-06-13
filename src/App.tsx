// App.tsx — Multi-Angle Edit main UI.
//
// Three-column creative tool:
//   left   = source image select + before/after thumbnails
//   center = the interactive 3D AnglePicker + live formatted prompt
//   right  = GPU select, angle sliders + preset dropdowns, extra guidance,
//            Generate button, progress bar, and a scrolling event log.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { invoke, convertFileSrc } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import {
  Image as ImageIcon,
  Cpu,
  FolderOpen,
  Sparkles,
  Wand2,
  Camera,
} from "lucide-react";
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

const DEFAULT_ANGLE: AngleValue = { azimuth: 0, elevation: 0, distance: 5 };

export default function App() {
  const [imagePath, setImagePath] = useState<string | undefined>(undefined);
  const [angle, setAngle] = useState<AngleValue>(DEFAULT_ANGLE);
  const [gpus, setGpus] = useState<GpuDevice[]>([]);
  const [deviceIndex, setDeviceIndex] = useState<number>(0);
  const [extraPrompt, setExtraPrompt] = useState<string>("");
  const [store, setStore] = useState<StoreInfo | null>(null);

  const [running, setRunning] = useState(false);
  const [percent, setPercent] = useState<number | null>(null);
  const [log, setLog] = useState<LogLine[]>([]);
  const [outputPath, setOutputPath] = useState<string | undefined>(undefined);
  const [outputDir, setOutputDir] = useState<string | undefined>(undefined);

  const logIdRef = useRef(0);
  const logEndRef = useRef<HTMLDivElement | null>(null);

  const anglePrompt = useMemo(() => formatAnglePrompt(angle), [angle]);

  const pushLog = useCallback((kind: LogLine["kind"], text: string) => {
    setLog((prev) => {
      const next = [...prev, { id: logIdRef.current++, kind, text }];
      // Cap the log to keep the DOM light.
      return next.length > 400 ? next.slice(next.length - 400) : next;
    });
  }, []);

  // ---- initial backend queries -------------------------------------------
  useEffect(() => {
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
  }, [pushLog]);

  // ---- worker event stream ------------------------------------------------
  useEffect(() => {
    let unlisten: UnlistenFn | undefined;
    listen<WorkerEvent>("angle-worker-event", (event) => {
      const payload = event.payload;
      switch (payload.type) {
        case "setup":
          pushLog("setup", payload.message);
          break;
        case "progress":
          pushLog("progress", payload.message);
          if (typeof payload.percent === "number") {
            setPercent(clamp(payload.percent, 0, 100));
          }
          break;
        case "done":
          pushLog("done", `Done: ${payload.outputPath}`);
          setPercent(100);
          break;
        case "error":
          pushLog("error", payload.message);
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
        setImagePath(selected);
        setOutputPath(undefined);
        pushLog("info", `Loaded image: ${selected}`);
      }
    } catch (e) {
      pushLog("error", `Image selection failed: ${String(e)}`);
    }
  }, [pushLog]);

  const generate = useCallback(async () => {
    if (!imagePath || running) return;
    setRunning(true);
    setPercent(0);
    setOutputPath(undefined);
    pushLog("info", `Generating: ${anglePrompt}`);
    try {
      const result = await invoke<string>("generate_angle", {
        request: {
          imagePath,
          anglePrompt,
          deviceIndex,
          extraPrompt: extraPrompt.trim() || undefined,
        },
      });
      setOutputPath(result);
      // Derive the output directory for the "open folder" button.
      const idx = Math.max(result.lastIndexOf("\\"), result.lastIndexOf("/"));
      setOutputDir(idx > 0 ? result.slice(0, idx) : undefined);
      setPercent(100);
      pushLog("done", `Output saved: ${result}`);
    } catch (e) {
      pushLog("error", `Generation failed: ${String(e)}`);
    } finally {
      setRunning(false);
    }
  }, [imagePath, running, anglePrompt, deviceIndex, extraPrompt, pushLog]);

  const openOutputFolder = useCallback(() => {
    if (!outputDir) return;
    invoke("open_path", { path: outputDir }).catch((e) =>
      pushLog("error", `Could not open folder: ${String(e)}`),
    );
  }, [outputDir, pushLog]);

  // ---- render -------------------------------------------------------------
  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <Camera size={22} className="brand-icon" />
          <span className="brand-name">Multi-Angle Edit</span>
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
                <div className="thumb-frame">
                  {imagePath ? (
                    <img src={convertFileSrc(imagePath)} alt="source" />
                  ) : (
                    <div className="thumb-empty">No image loaded</div>
                  )}
                </div>
              </figure>
              <figure className="thumb">
                <figcaption>After</figcaption>
                <div className="thumb-frame">
                  {outputPath ? (
                    <img src={convertFileSrc(outputPath)} alt="result" />
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
              <span className="hint">drag to orbit · scroll to zoom</span>
            </div>
            <div className="viewport-wrap">
              <AnglePicker
                imagePath={imagePath}
                value={angle}
                onChange={setAngle}
              />
            </div>
            <div className="prompt-readout">
              <span className="prompt-label">prompt</span>
              <code>{anglePrompt}</code>
            </div>
          </div>
        </section>

        {/* RIGHT: controls */}
        <section className="col col-right">
          <div className="card">
            <div className="card-title">
              <Cpu size={16} /> Compute device
            </div>
            <select
              className="select"
              value={deviceIndex}
              onChange={(e) => setDeviceIndex(Number(e.target.value))}
              disabled={running}
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
            <div className="card-title">Extra guidance (optional)</div>
            <textarea
              className="textarea"
              placeholder="e.g. keep the lighting warm, preserve background"
              value={extraPrompt}
              onChange={(e) => setExtraPrompt(e.target.value)}
              disabled={running}
              rows={2}
            />
          </div>

          <button
            className="generate-btn"
            onClick={generate}
            disabled={running || !imagePath}
          >
            <Sparkles size={18} />
            {running ? "Forging…" : "Generate"}
          </button>

          {(running || percent != null) && (
            <div className="progress">
              <div
                className="progress-bar"
                style={{ width: `${percent ?? 0}%` }}
              />
            </div>
          )}

          <div className="card log-card">
            <div className="card-title">Log</div>
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
      </footer>
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
