// anglePrompt.ts
//
// Pure helpers that translate the interactive 3D angle picker's numeric state
// (azimuth / elevation / distance) into the exact Qwen-Image-Edit "Multiple
// Angles" LoRA vocabulary, plus the preset tables that drive App.tsx's
// quick-select dropdowns.
//
// The trigger token "<sks>" is what the LoRA was trained against; the three
// descriptive phrases follow it in a fixed order: azimuth, elevation, distance.
//
// --- self-checks (informal unit tests, verified by inspection) ------------
//   formatAnglePrompt({ azimuth: 0,   elevation: 0,  distance: 5 })
//     === '<sks> front view eye-level shot medium shot'
//   formatAnglePrompt({ azimuth: 90,  elevation: 60, distance: 1 })
//     === '<sks> right side view high-angle shot close-up'
//   formatAnglePrompt({ azimuth: 180, elevation: -30, distance: 9 })
//     === '<sks> back view low-angle shot wide shot'
//   formatAnglePrompt({ azimuth: 359, elevation: 0,  distance: 0 })
//     === '<sks> front view eye-level shot close-up'   (359 snaps to 0/front)
//   formatAnglePrompt({ azimuth: 315, elevation: 30, distance: 4 })
//     === '<sks> front-left quarter view elevated shot medium shot'
// --------------------------------------------------------------------------

export interface AngleValue {
  /** Horizontal orbit angle in degrees, 0-360 (wraps). 0 = facing camera. */
  azimuth: number;
  /** Vertical orbit angle in degrees, clamped to [-30, 60]. */
  elevation: number;
  /** Orbit distance bucketed into close-up / medium / wide. Range [0, 10]. */
  distance: number;
}

export interface Preset {
  label: string;
  value: number;
}

/** The eight azimuth buckets the LoRA understands, around the full circle. */
export const AZIMUTH_PRESETS: Preset[] = [
  { label: "Front view", value: 0 },
  { label: "Front-right quarter view", value: 45 },
  { label: "Right side view", value: 90 },
  { label: "Back-right quarter view", value: 135 },
  { label: "Back view", value: 180 },
  { label: "Back-left quarter view", value: 225 },
  { label: "Left side view", value: 270 },
  { label: "Front-left quarter view", value: 315 },
];

/** Representative elevation values for each elevation phrase. */
export const ELEVATION_PRESETS: Preset[] = [
  { label: "Low-angle shot", value: -30 },
  { label: "Eye-level shot", value: 0 },
  { label: "Elevated shot", value: 30 },
  { label: "High-angle shot", value: 60 },
];

/** Representative distance values for each distance phrase. */
export const DISTANCE_PRESETS: Preset[] = [
  { label: "Close-up", value: 1 },
  { label: "Medium shot", value: 5 },
  { label: "Wide shot", value: 9 },
];

/** Phrase lookup keyed by the canonical azimuth bucket value. */
const AZIMUTH_PHRASES: Record<number, string> = {
  0: "front view",
  45: "front-right quarter view",
  90: "right side view",
  135: "back-right quarter view",
  180: "back view",
  225: "back-left quarter view",
  270: "left side view",
  315: "front-left quarter view",
};

/** Wrap any angle into the [0, 360) range. */
export function wrapAzimuth(azimuth: number): number {
  const wrapped = azimuth % 360;
  return wrapped < 0 ? wrapped + 360 : wrapped;
}

/** Clamp a number into an inclusive range. */
export function clamp(value: number, min: number, max: number): number {
  if (Number.isNaN(value)) return min;
  return Math.min(max, Math.max(min, value));
}

/** Snap a continuous azimuth to the nearest of the eight LoRA buckets. */
export function snapAzimuth(azimuth: number): number {
  const wrapped = wrapAzimuth(azimuth);
  // Each bucket spans 45deg; nearest-bucket = round to multiple of 45.
  const snapped = (Math.round(wrapped / 45) * 45) % 360;
  return snapped;
}

/** Map a continuous azimuth to its LoRA phrase. */
export function azimuthPhrase(azimuth: number): string {
  return AZIMUTH_PHRASES[snapAzimuth(azimuth)] ?? "front view";
}

/** Map a continuous elevation to its LoRA phrase. */
export function elevationPhrase(elevation: number): string {
  const e = clamp(elevation, -30, 60);
  if (e <= -15) return "low-angle shot";
  if (e <= 15) return "eye-level shot";
  if (e <= 45) return "elevated shot";
  return "high-angle shot";
}

/** Map a continuous orbit distance to its LoRA phrase. */
export function distancePhrase(distance: number): string {
  const d = clamp(distance, 0, 10);
  if (d <= 3.33) return "close-up";
  if (d <= 6.66) return "medium shot";
  return "wide shot";
}

/**
 * Compose the full prompt string fed to the backend, e.g.
 * "<sks> right side view high-angle shot close-up".
 */
export function formatAnglePrompt(value: AngleValue): string {
  return [
    "<sks>",
    azimuthPhrase(value.azimuth),
    elevationPhrase(value.elevation),
    distancePhrase(value.distance),
  ].join(" ");
}
