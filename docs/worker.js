/**
 * Web Worker: runs the WASM Ising physics engine off the main thread.
 *
 * Messages IN  (from main thread):
 *   { type: "init",   n, geometry, j, h, seed, temperature }
 *   { type: "set",    temperature?, j?, h? }
 *   { type: "reset" }
 *   { type: "start" }
 *   { type: "pause" }
 *   { type: "sweep",  tMin, tMax, steps, warmup, samples }
 *
 * Messages OUT (to main thread):
 *   { type: "ready" }
 *   { type: "frame",  spins: Int8Array (transferred), magnetisation, step }
 *   { type: "sweep_done", csv }
 */

import init, { IsingWasm } from "./pkg/ising.js";

let wasmReady = false;
let sim = null;
let running = false;
let stepCount = 0;
let temperature = 2.0;
let frameTimer = null;

async function initWasm() {
  await init();
  wasmReady = true;
}

function tick() {
  if (!sim || !running) return;
  sim.step(temperature);
  stepCount++;

  // get_spins_copy() returns a JS Array from Rust Vec<i8>
  // Convert to Int8Array and transfer (zero extra copy on transfer)
  const raw = sim.get_spins_copy();
  const spins = new Int8Array(raw);
  const mag = sim.magnetisation();

  self.postMessage(
    { type: "frame", spins, magnetisation: mag, step: stepCount },
    [spins.buffer]
  );
}

self.onmessage = async (e) => {
  const msg = e.data;

  // Ensure WASM is ready before any sim operation
  if (!wasmReady) await initWasm();

  switch (msg.type) {
    case "init": {
      sim = new IsingWasm(
        msg.n,
        msg.geometry ?? 2,       // default: Cubic3D
        msg.j ?? 1.0,
        msg.h ?? 0.0,
        BigInt(msg.seed ?? 42)
      );
      temperature = msg.temperature ?? 2.0;
      stepCount = 0;
      running = false;
      self.postMessage({ type: "ready" });
      break;
    }

    case "set": {
      if (msg.temperature !== undefined) temperature = msg.temperature;
      if (sim && (msg.j !== undefined || msg.h !== undefined)) {
        sim.set_params(msg.j ?? 1.0, msg.h ?? 0.0);
      }
      break;
    }

    case "reset": {
      if (sim) { sim.randomise(); stepCount = 0; }
      break;
    }

    case "start": {
      running = true;
      if (!frameTimer) frameTimer = setInterval(tick, 32); // ~30fps
      break;
    }

    case "pause": {
      running = false;
      clearInterval(frameTimer);
      frameTimer = null;
      break;
    }

    case "sweep": {
      // Stop live sim during sweep
      running = false;
      clearInterval(frameTimer);
      frameTimer = null;
      if (!sim) break;

      sim.randomise();
      const csv = sim.temperature_sweep(
        msg.tMin ?? 0.5,
        msg.tMax ?? 5.0,
        msg.steps ?? 46,
        msg.warmup ?? 500,
        msg.samples ?? 200,
      );
      self.postMessage({ type: "sweep_done", csv });
      break;
    }
  }
};

// Pre-load WASM in the background so it's ready when "init" arrives
initWasm();
