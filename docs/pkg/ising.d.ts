/* tslint:disable */
/* eslint-disable */

/**
 * The live simulation state exposed to JavaScript.
 *
 * JS creates one instance, then calls step() in a loop from a Web Worker.
 * The spin array is read directly from WASM memory via a typed array view.
 */
export class IsingWasm {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Parse sweep CSV and fit critical exponents. Returns JSON string:
     * { tc, beta, alpha, gamma, beta_err, alpha_err, gamma_err,
     *   theory_beta, theory_alpha, theory_gamma }
     * Returns empty string if fitting fails (too few points, etc).
     */
    fit_exponents(csv: string, window: number): string;
    /**
     * Copy of spin array as Int8Array — safe to transfer to main thread.
     */
    get_spins_copy(): Int8Array;
    /**
     * Average magnetisation |<M>| of the current configuration.
     */
    magnetisation(): number;
    /**
     * Lattice dimension N.
     */
    n(): number;
    /**
     * Create a new lattice.
     * geometry: 0 = Square2D, 1 = Triangular2D, 2 = Cubic3D
     */
    constructor(n: number, geometry: number, j: number, h: number, seed: bigint);
    /**
     * Randomise the lattice (reset).
     */
    randomise(): void;
    /**
     * Update J and h without recreating the lattice.
     */
    set_params(j: number, h: number): void;
    /**
     * Number of spins (N² or N³).
     */
    size(): number;
    /**
     * Pointer into WASM linear memory for the spin array.
     * JS reads this as Int8Array for zero-copy access.
     */
    spins_ptr(): number;
    /**
     * Run one full Metropolis sweep (N² spin-flip attempts).
     */
    step(temperature: number): void;
    /**
     * Run a full temperature sweep and return CSV bytes.
     * t_min, t_max, steps, warmup, samples — same as CLI.
     */
    temperature_sweep(t_min: number, t_max: number, steps: number, warmup: number, samples: number): string;
    /**
     * Run `n` sweeps — useful for warm-up from JS without per-frame overhead.
     */
    warm_up(temperature: number, n: number): void;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_isingwasm_free: (a: number, b: number) => void;
    readonly isingwasm_fit_exponents: (a: number, b: number, c: number, d: number) => [number, number];
    readonly isingwasm_get_spins_copy: (a: number) => [number, number];
    readonly isingwasm_magnetisation: (a: number) => number;
    readonly isingwasm_n: (a: number) => number;
    readonly isingwasm_new: (a: number, b: number, c: number, d: number, e: bigint) => number;
    readonly isingwasm_randomise: (a: number) => void;
    readonly isingwasm_set_params: (a: number, b: number, c: number) => void;
    readonly isingwasm_size: (a: number) => number;
    readonly isingwasm_spins_ptr: (a: number) => number;
    readonly isingwasm_step: (a: number, b: number) => void;
    readonly isingwasm_temperature_sweep: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
    readonly isingwasm_warm_up: (a: number, b: number, c: number) => void;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
