/**
 * main.js — Three.js 3D Ising Model Visualiser
 *
 * Renders the spin lattice as instanced spheres:
 *   spin +1 → blue
 *   spin -1 → red
 *
 * Physics runs in a Web Worker (worker.js) via WASM (ising.wasm).
 * The worker sends spin arrays each frame; Three.js updates instance matrices.
 *
 * innerHTML is used only with SVG strings built entirely from numeric data —
 * no user input is ever interpolated into the SVG markup.
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// ─── State ────────────────────────────────────────────────────────────────────

const state = {
  n: 15,
  geometry: 2,
  j: 1.0,
  h: 0.0,
  temperature: 2.0,
  running: false,
  step: 0,
  magnetisation: 0,
};

// ─── Worker ───────────────────────────────────────────────────────────────────

const worker = new Worker("./worker.js", { type: "module" });

worker.onmessage = (e) => {
  const msg = e.data;
  switch (msg.type) {
    case "ready":
      setStatus("Ready — click ▶ Run");
      break;
    case "frame":
      updateMeshes(msg.spins);
      state.step = msg.step;
      state.magnetisation = msg.magnetisation;
      updateHUD();
      break;
    case "sweep_done":
      parseSweepCSV(msg.csv);
      setStatus("Sweep complete ✓");
      document.getElementById("btn-sweep").disabled = false;
      document.getElementById("btn-run").disabled = false;
      break;
  }
};

function workerSend(msg) { worker.postMessage(msg); }

// ─── Three.js Setup ───────────────────────────────────────────────────────────

const canvas = document.getElementById("canvas");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x070710);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

scene.add(new THREE.AmbientLight(0xffffff, 0.5));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
dirLight.position.set(1, 2, 3);
scene.add(dirLight);
const backLight = new THREE.DirectionalLight(0x4466ff, 0.4);
backLight.position.set(-2, -1, -2);
scene.add(backLight);

// ─── Instanced Meshes ─────────────────────────────────────────────────────────

const GEO      = new THREE.SphereGeometry(0.42, 10, 8);
const MAT_UP   = new THREE.MeshPhongMaterial({ color: 0x3b82f6, shininess: 80 });
const MAT_DOWN = new THREE.MeshPhongMaterial({ color: 0xef4444, shininess: 80 });

let meshUp = null;
let meshDown = null;
let spinPositions = [];

function buildPositions(n) {
  spinPositions = [];
  const offset = -(n - 1) / 2;
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      for (let k = 0; k < n; k++)
        spinPositions.push(new THREE.Vector3(i + offset, j + offset, k + offset));
}

function initMeshes(n) {
  if (meshUp)   scene.remove(meshUp);
  if (meshDown) scene.remove(meshDown);

  const size = n * n * n;
  meshUp   = new THREE.InstancedMesh(GEO, MAT_UP,   size);
  meshDown = new THREE.InstancedMesh(GEO, MAT_DOWN, size);
  meshUp.count = 0;
  meshDown.count = 0;
  scene.add(meshUp);
  scene.add(meshDown);

  const dist = n * 1.5;
  camera.position.set(dist, dist * 0.7, dist);
  controls.target.set(0, 0, 0);
  controls.update();
}

const _dummy = new THREE.Object3D();

function updateMeshes(spins) {
  if (!meshUp || !meshDown) return;
  let upIdx = 0, downIdx = 0;
  for (let i = 0; i < spins.length; i++) {
    _dummy.position.copy(spinPositions[i]);
    _dummy.updateMatrix();
    if (spins[i] > 0) meshUp.setMatrixAt(upIdx++, _dummy.matrix);
    else               meshDown.setMatrixAt(downIdx++, _dummy.matrix);
  }
  meshUp.count = upIdx;
  meshDown.count = downIdx;
  meshUp.instanceMatrix.needsUpdate = true;
  meshDown.instanceMatrix.needsUpdate = true;
}

// ─── Charts ───────────────────────────────────────────────────────────────────

const CHARTS = [
  { id: "chart-e",   key: "E",   label: "⟨E⟩ / spin",      color: "#60a5fa" },
  { id: "chart-m",   key: "M",   label: "|⟨M⟩| / spin",    color: "#34d399" },
  { id: "chart-cv",  key: "Cv",  label: "Heat Capacity Cv", color: "#fb923c" },
  { id: "chart-chi", key: "chi", label: "Susceptibility χ", color: "#e879f9" },
];

let sweepRows = [];

function parseSweepCSV(csv) {
  sweepRows = [];
  const lines = csv.trim().split("\n").slice(1);
  for (const line of lines) {
    const [T, E, M, Cv, chi] = line.split(",").map(Number);
    sweepRows.push({ T, E, M, Cv, chi });
  }
  renderCharts();
  announceTc();
}

function announceTc() {
  if (sweepRows.length < 3) return;
  let maxDm = 0, tc = sweepRows[0].T;
  for (let i = 1; i < sweepRows.length - 1; i++) {
    const dm = Math.abs(sweepRows[i - 1].M - sweepRows[i + 1].M);
    if (dm > maxDm) { maxDm = dm; tc = sweepRows[i].T; }
  }
  document.getElementById("hud-tc").textContent = tc.toFixed(2);
}

// Build SVG purely from numbers — no user content is interpolated
function buildSVG(rows, key, label, color) {
  const W = 280, H = 145;
  const P = { t: 14, r: 8, b: 26, l: 42 };
  const pw = W - P.l - P.r, ph = H - P.t - P.b;

  const xs = rows.map(r => r.T);
  const ys = rows.map(r => r[key]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const y0 = Math.min(...ys), y1 = Math.max(...ys);
  const ry = y1 - y0 || 1;

  const sx = x => P.l + (x - x0) / (x1 - x0) * pw;
  const sy = y => P.t + ph - (y - y0) / ry * ph;
  const path = rows.map((r, i) =>
    `${i ? "L" : "M"}${sx(r.T).toFixed(1)},${sy(r[key]).toFixed(1)}`).join(" ");

  // Build SVG DOM safely using createElementNS
  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("width", "100%");

  const ax1 = document.createElementNS(NS, "line");
  ax1.setAttribute("x1", P.l); ax1.setAttribute("y1", P.t);
  ax1.setAttribute("x2", P.l); ax1.setAttribute("y2", P.t + ph);
  ax1.setAttribute("stroke", "#374151"); ax1.setAttribute("stroke-width", "1");
  svg.appendChild(ax1);

  const ax2 = document.createElementNS(NS, "line");
  ax2.setAttribute("x1", P.l); ax2.setAttribute("y1", P.t + ph);
  ax2.setAttribute("x2", P.l + pw); ax2.setAttribute("y2", P.t + ph);
  ax2.setAttribute("stroke", "#374151"); ax2.setAttribute("stroke-width", "1");
  svg.appendChild(ax2);

  for (const v of [y0, y0 + ry / 2, y1]) {
    const tick = document.createElementNS(NS, "line");
    tick.setAttribute("x1", P.l - 3); tick.setAttribute("y1", sy(v).toFixed(1));
    tick.setAttribute("x2", P.l);     tick.setAttribute("y2", sy(v).toFixed(1));
    tick.setAttribute("stroke", "#374151"); tick.setAttribute("stroke-width", "1");
    svg.appendChild(tick);

    const lbl = document.createElementNS(NS, "text");
    lbl.setAttribute("x", P.l - 5);
    lbl.setAttribute("y", (sy(v) + 3).toFixed(1));
    lbl.setAttribute("text-anchor", "end");
    lbl.setAttribute("font-size", "8");
    lbl.setAttribute("fill", "#6b7280");
    lbl.textContent = v.toFixed(2);
    svg.appendChild(lbl);
  }

  for (const v of [x0, (x0 + x1) / 2, x1]) {
    const tick = document.createElementNS(NS, "line");
    tick.setAttribute("x1", sx(v).toFixed(1)); tick.setAttribute("y1", P.t + ph);
    tick.setAttribute("x2", sx(v).toFixed(1)); tick.setAttribute("y2", P.t + ph + 3);
    tick.setAttribute("stroke", "#374151"); tick.setAttribute("stroke-width", "1");
    svg.appendChild(tick);

    const lbl = document.createElementNS(NS, "text");
    lbl.setAttribute("x", sx(v).toFixed(1));
    lbl.setAttribute("y", P.t + ph + 11);
    lbl.setAttribute("text-anchor", "middle");
    lbl.setAttribute("font-size", "8");
    lbl.setAttribute("fill", "#6b7280");
    lbl.textContent = v.toFixed(1);
    svg.appendChild(lbl);
  }

  const pathEl = document.createElementNS(NS, "path");
  pathEl.setAttribute("d", path);
  pathEl.setAttribute("fill", "none");
  pathEl.setAttribute("stroke", color);
  pathEl.setAttribute("stroke-width", "1.5");
  pathEl.setAttribute("stroke-linejoin", "round");
  svg.appendChild(pathEl);

  const titleEl = document.createElementNS(NS, "text");
  titleEl.setAttribute("x", P.l + 4);
  titleEl.setAttribute("y", P.t + 10);
  titleEl.setAttribute("font-size", "9");
  titleEl.setAttribute("fill", color);
  titleEl.setAttribute("font-weight", "600");
  titleEl.textContent = label;
  svg.appendChild(titleEl);

  return svg;
}

function renderCharts() {
  for (const { id, key, label, color } of CHARTS) {
    const el = document.getElementById(id);
    if (!el) continue;
    el.replaceChildren();
    if (sweepRows.length < 2) {
      const p = document.createElement("p");
      p.className = "placeholder";
      p.textContent = `Run sweep to see ${label}`;
      el.appendChild(p);
    } else {
      el.appendChild(buildSVG(sweepRows, key, label, color));
    }
  }
}

// ─── HUD ─────────────────────────────────────────────────────────────────────

function updateHUD() {
  document.getElementById("hud-step").textContent = state.step.toLocaleString();
  document.getElementById("hud-mag").textContent  = state.magnetisation.toFixed(3);
  document.getElementById("hud-t").textContent    = state.temperature.toFixed(2);
}

function setStatus(msg) {
  document.getElementById("status").textContent = msg;
}

// ─── Render Loop ──────────────────────────────────────────────────────────────

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

function onResize() {
  const el = document.getElementById("viewport");
  const w = el.clientWidth || window.innerWidth;
  const h = el.clientHeight || window.innerHeight;
  if (w === 0 || h === 0) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

window.addEventListener("resize", onResize);
// Also observe the viewport element directly for size changes
new ResizeObserver(onResize).observe(document.getElementById("viewport"));

// ─── Controls ────────────────────────────────────────────────────────────────

function initSim() {
  state.n           = parseInt(document.getElementById("ctrl-n").value);
  state.j           = parseFloat(document.getElementById("ctrl-j").value);
  state.h           = parseFloat(document.getElementById("ctrl-h").value);
  state.temperature = parseFloat(document.getElementById("ctrl-t").value);

  buildPositions(state.n);
  initMeshes(state.n);
  sweepRows = [];
  renderCharts();

  workerSend({
    type: "init",
    n: state.n,
    geometry: 2,
    j: state.j,
    h: state.h,
    temperature: state.temperature,
    seed: Date.now(),
  });
}

function bindControls() {
  const sliders = [
    { id: "ctrl-t", label: "val-t", key: "temperature", dec: 2 },
    { id: "ctrl-j", label: "val-j", key: "j",           dec: 1 },
    { id: "ctrl-h", label: "val-h", key: "h",           dec: 1 },
  ];

  for (const { id, label, key, dec } of sliders) {
    const el  = document.getElementById(id);
    const lbl = document.getElementById(label);
    el.addEventListener("input", () => {
      const v = parseFloat(el.value);
      state[key] = v;
      lbl.textContent = v.toFixed(dec);
      workerSend({ type: "set", temperature: state.temperature, j: state.j, h: state.h });
    });
  }

  const nEl  = document.getElementById("ctrl-n");
  const nLbl = document.getElementById("val-n");
  nLbl.textContent = nEl.value;
  nEl.addEventListener("change", () => {
    nLbl.textContent = nEl.value;
    if (state.running) {
      workerSend({ type: "pause" });
      state.running = false;
      document.getElementById("btn-run").textContent = "▶ Run";
    }
    initSim();
  });

  document.getElementById("btn-run").addEventListener("click", () => {
    state.running = !state.running;
    if (state.running) {
      workerSend({ type: "start" });
      document.getElementById("btn-run").textContent = "⏸ Pause";
      setStatus("Simulating…");
    } else {
      workerSend({ type: "pause" });
      document.getElementById("btn-run").textContent = "▶ Run";
      setStatus("Paused");
    }
  });

  document.getElementById("btn-reset").addEventListener("click", () => {
    workerSend({ type: "reset" });
    state.step = 0;
    updateHUD();
  });

  document.getElementById("btn-sweep").addEventListener("click", () => {
    if (state.running) {
      workerSend({ type: "pause" });
      state.running = false;
      document.getElementById("btn-run").textContent = "▶ Run";
    }
    document.getElementById("btn-sweep").disabled = true;
    document.getElementById("btn-run").disabled = true;
    setStatus("Running temperature sweep… (~20s)");
    workerSend({ type: "sweep", tMin: 0.5, tMax: 5.0, steps: 46, warmup: 300, samples: 150 });
  });
}

// ─── Boot ─────────────────────────────────────────────────────────────────────

animate();
bindControls();

// Two rAF frames ensures the CSS grid has fully laid out before we measure
requestAnimationFrame(() => requestAnimationFrame(() => {
  onResize();
  initSim();
}));
