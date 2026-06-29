// AnglePicker.tsx
//
// Interactive 3D camera-angle picker built directly on Three.js (no
// react-three-fiber, no OrbitControls dependency — a small hand-rolled orbit
// implementation keeps the contract tight and the bundle lean).
//
// Controlled component: the PARENT owns { azimuth, elevation, distance } and
// passes it in via `value`; the picker reflects that state into the orbiting
// camera and reports user drags / wheel scrolls back through `onChange`.
//
// Conventions:
//   azimuth   0-360 deg, wraps. 0 = camera in front of the subject (+Z).
//   elevation -30..60 deg, clamped. positive = looking down from above.
//   distance  0..10, mapped to a real camera radius; bucketed into
//             close-up / medium / wide by anglePrompt.ts.

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { clamp, wrapAzimuth, type AngleValue } from "../lib/anglePrompt";

export interface AnglePickerProps {
  imagePreviewSrc?: string;
  value: AngleValue;
  onChange: (value: AngleValue) => void;
}

// Map the abstract distance scale [0,10] to a physical orbit radius. Smaller
// number = closer to the subject. Tuned so close-up / medium / wide read well.
function distanceToRadius(distance: number): number {
  const d = clamp(distance, 0, 10);
  // 0 -> 2.2 (tight), 10 -> 9 (far). Linear is fine for a viewport preview.
  return 2.2 + (d / 10) * 6.8;
}

export default function AnglePicker({
  imagePreviewSrc,
  value,
  onChange,
}: AnglePickerProps) {
  const mountRef = useRef<HTMLDivElement | null>(null);

  // Latest props mirrored into refs so the imperative loop / handlers always
  // read fresh values without re-running the heavy setup effect.
  const valueRef = useRef<AngleValue>(value);
  const onChangeRef = useRef(onChange);
  valueRef.current = value;
  onChangeRef.current = onChange;

  // Three.js objects that the texture-loading effect also needs to touch.
  const subjectRef = useRef<THREE.Mesh | null>(null);
  const placeholderRef = useRef<THREE.Group | null>(null);

  // ---- one-time scene setup ------------------------------------------------
  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101018);

    const camera = new THREE.PerspectiveCamera(
      45,
      mount.clientWidth / Math.max(1, mount.clientHeight),
      0.1,
      1000,
    );

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, mount.clientHeight);
    mount.appendChild(renderer.domElement);
    renderer.domElement.style.display = "block";
    renderer.domElement.style.cursor = "grab";
    renderer.domElement.style.touchAction = "none";

    // Lighting.
    const ambient = new THREE.AmbientLight(0xffffff, 0.85);
    scene.add(ambient);
    const key = new THREE.DirectionalLight(0xffffff, 1.0);
    key.position.set(3, 5, 4);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xb14bff, 0.35);
    fill.position.set(-4, 2, -3);
    scene.add(fill);

    // Ground grid for orientation.
    const grid = new THREE.GridHelper(20, 20, 0xb14bff, 0x2a2a3a);
    (grid.material as THREE.Material).opacity = 0.5;
    (grid.material as THREE.Material).transparent = true;
    grid.position.y = -1.6;
    scene.add(grid);

    // Axis indicator at the origin.
    const axes = new THREE.AxesHelper(1.4);
    scene.add(axes);

    // --- subject -----------------------------------------------------------
    // A flat image card (front-facing toward +Z, i.e. azimuth 0). Starts with a
    // neutral material; the texture effect swaps the front face's map in.
    const cardGeo = new THREE.PlaneGeometry(2.2, 3.0);
    const cardMat = new THREE.MeshStandardMaterial({
      color: 0x3a3a4a,
      roughness: 0.85,
      metalness: 0.0,
      side: THREE.DoubleSide,
    });
    const card = new THREE.Mesh(cardGeo, cardMat);
    card.visible = false; // shown only once a texture loads
    scene.add(card);
    subjectRef.current = card;

    // Neutral placeholder figure (a simple "bust": box body + sphere head)
    // shown when no image is loaded so there is always something to orbit.
    const placeholder = new THREE.Group();
    const bodyMat = new THREE.MeshStandardMaterial({
      color: 0x4a4a5e,
      roughness: 0.6,
      metalness: 0.1,
    });
    const body = new THREE.Mesh(new THREE.BoxGeometry(1.6, 2.0, 1.0), bodyMat);
    body.position.y = -0.1;
    placeholder.add(body);
    const head = new THREE.Mesh(
      new THREE.SphereGeometry(0.55, 32, 24),
      new THREE.MeshStandardMaterial({
        color: 0xb14bff,
        roughness: 0.4,
        metalness: 0.2,
      }),
    );
    head.position.y = 1.25;
    placeholder.add(head);
    // A small "nose" cone marking the front (+Z) so orientation is obvious.
    const nose = new THREE.Mesh(
      new THREE.ConeGeometry(0.14, 0.4, 16),
      new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.5 }),
    );
    nose.rotation.x = Math.PI / 2;
    nose.position.set(0, 1.25, 0.6);
    placeholder.add(nose);
    scene.add(placeholder);
    placeholderRef.current = placeholder;

    // --- camera positioning from the controlled value --------------------
    const target = new THREE.Vector3(0, 0, 0);
    function applyCamera() {
      const v = valueRef.current;
      const az = THREE.MathUtils.degToRad(wrapAzimuth(v.azimuth));
      const el = THREE.MathUtils.degToRad(clamp(v.elevation, -30, 60));
      const radius = distanceToRadius(v.distance);
      // azimuth 0 -> +Z (in front). Spherical to cartesian:
      const cosEl = Math.cos(el);
      camera.position.set(
        radius * cosEl * Math.sin(az),
        radius * Math.sin(el),
        radius * cosEl * Math.cos(az),
      );
      camera.lookAt(target);
    }
    applyCamera();

    // --- render loop -------------------------------------------------------
    let frameId = 0;
    const render = () => {
      applyCamera();
      renderer.render(scene, camera);
      frameId = requestAnimationFrame(render);
    };
    frameId = requestAnimationFrame(render);

    // --- resize handling ---------------------------------------------------
    const handleResize = () => {
      if (!mount) return;
      const w = mount.clientWidth;
      const h = Math.max(1, mount.clientHeight);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    const resizeObserver = new ResizeObserver(handleResize);
    resizeObserver.observe(mount);
    window.addEventListener("resize", handleResize);

    // --- pointer drag -> orbit / zoom --------------------------------------
    type DragMode = "orbit" | "zoom";
    let dragMode: DragMode | null = null;
    let lastX = 0;
    let lastY = 0;

    const onPointerDown = (e: PointerEvent) => {
      if (e.button !== 0 && e.button !== 1) return;
      e.preventDefault();
      dragMode = e.button === 1 ? "zoom" : "orbit";
      lastX = e.clientX;
      lastY = e.clientY;
      renderer.domElement.style.cursor =
        dragMode === "zoom" ? "ns-resize" : "grabbing";
      renderer.domElement.setPointerCapture(e.pointerId);
    };
    const onPointerMove = (e: PointerEvent) => {
      if (!dragMode) return;
      e.preventDefault();
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      const v = valueRef.current;

      if (dragMode === "zoom") {
        // Drag down = zoom out, drag up = zoom in. Mirrors wheel direction.
        const distance = clamp(v.distance + dy * 0.035, 0, 10);
        onChangeRef.current({ ...v, distance });
        return;
      }

      // 0.5 deg per pixel feels natural for both axes.
      const azimuth = wrapAzimuth(v.azimuth + dx * 0.5);
      const elevation = clamp(v.elevation + dy * 0.5, -30, 60);
      onChangeRef.current({ ...v, azimuth, elevation });
    };
    const endDrag = (e: PointerEvent) => {
      dragMode = null;
      renderer.domElement.style.cursor = "grab";
      try {
        renderer.domElement.releasePointerCapture(e.pointerId);
      } catch {
        /* capture may already be gone */
      }
    };
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const v = valueRef.current;
      // Scroll down (positive deltaY) = zoom out = larger distance.
      const distance = clamp(v.distance + (e.deltaY > 0 ? 0.6 : -0.6), 0, 10);
      onChangeRef.current({ ...v, distance });
    };
    const preventMiddleClick = (e: MouseEvent) => {
      if (e.button === 1) e.preventDefault();
    };

    const el = renderer.domElement;
    el.addEventListener("pointerdown", onPointerDown);
    el.addEventListener("pointermove", onPointerMove);
    el.addEventListener("pointerup", endDrag);
    el.addEventListener("pointercancel", endDrag);
    el.addEventListener("pointerleave", endDrag);
    el.addEventListener("wheel", onWheel, { passive: false });
    el.addEventListener("auxclick", preventMiddleClick);

    // --- teardown ----------------------------------------------------------
    return () => {
      cancelAnimationFrame(frameId);
      resizeObserver.disconnect();
      window.removeEventListener("resize", handleResize);
      el.removeEventListener("pointerdown", onPointerDown);
      el.removeEventListener("pointermove", onPointerMove);
      el.removeEventListener("pointerup", endDrag);
      el.removeEventListener("pointercancel", endDrag);
      el.removeEventListener("pointerleave", endDrag);
      el.removeEventListener("wheel", onWheel);
      el.removeEventListener("auxclick", preventMiddleClick);

      // Dispose geometry / materials / textures.
      scene.traverse((obj) => {
        const mesh = obj as THREE.Mesh;
        if (mesh.geometry) mesh.geometry.dispose();
        const mat = mesh.material as
          | THREE.Material
          | THREE.Material[]
          | undefined;
        if (Array.isArray(mat)) {
          mat.forEach((m) => disposeMaterial(m));
        } else if (mat) {
          disposeMaterial(mat);
        }
      });
      grid.dispose();
      axes.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === mount) {
        mount.removeChild(renderer.domElement);
      }
      subjectRef.current = null;
      placeholderRef.current = null;
    };
  }, []);

  // ---- load / swap the image texture when the source preview changes ------
  useEffect(() => {
    const card = subjectRef.current;
    const placeholder = placeholderRef.current;
    if (!card) return;

    if (!imagePreviewSrc) {
      card.visible = false;
      if (placeholder) placeholder.visible = true;
      return;
    }

    let cancelled = false;
    const loader = new THREE.TextureLoader();
    loader.load(
      imagePreviewSrc,
      (texture) => {
        if (cancelled) {
          texture.dispose();
          return;
        }
        texture.colorSpace = THREE.SRGBColorSpace;
        const mat = card.material as THREE.MeshStandardMaterial;
        // Dispose any previous texture before swapping.
        if (mat.map) mat.map.dispose();
        mat.map = texture;
        mat.color.set(0xffffff);
        mat.needsUpdate = true;

        // Fit the card to the image aspect ratio (max 3 units tall/wide).
        const img = texture.image as { width: number; height: number };
        if (img && img.width && img.height) {
          const aspect = img.width / img.height;
          const maxDim = 3.0;
          let w = maxDim;
          let h = maxDim;
          if (aspect >= 1) h = maxDim / aspect;
          else w = maxDim * aspect;
          card.geometry.dispose();
          card.geometry = new THREE.PlaneGeometry(w, h);
        }
        card.visible = true;
        if (placeholder) placeholder.visible = false;
      },
      undefined,
      () => {
        // On load failure fall back to the placeholder figure.
        if (cancelled) return;
        card.visible = false;
        if (placeholder) placeholder.visible = true;
      },
    );

    return () => {
      cancelled = true;
    };
  }, [imagePreviewSrc]);

  return <div className="angle-canvas" ref={mountRef} />;
}

function disposeMaterial(mat: THREE.Material) {
  const m = mat as THREE.MeshStandardMaterial;
  if (m.map) m.map.dispose();
  mat.dispose();
}
