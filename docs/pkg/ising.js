/* @ts-self-types="./ising.d.ts" */

/**
 * The live simulation state exposed to JavaScript.
 *
 * JS creates one instance, then calls step() in a loop from a Web Worker.
 * The spin array is read directly from WASM memory via a typed array view.
 */
export class IsingWasm {
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        IsingWasmFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_isingwasm_free(ptr, 0);
    }
    /**
     * Parse sweep CSV and fit critical exponents. Returns JSON string:
     * { tc, beta, alpha, gamma, beta_err, alpha_err, gamma_err,
     *   theory_beta, theory_alpha, theory_gamma }
     * Returns empty string if fitting fails (too few points, etc).
     * @param {string} csv
     * @param {number} window
     * @returns {string}
     */
    fit_exponents(csv, window) {
        let deferred2_0;
        let deferred2_1;
        try {
            const ptr0 = passStringToWasm0(csv, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len0 = WASM_VECTOR_LEN;
            const ret = wasm.isingwasm_fit_exponents(this.__wbg_ptr, ptr0, len0, window);
            deferred2_0 = ret[0];
            deferred2_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred2_0, deferred2_1, 1);
        }
    }
    /**
     * Copy of spin array as Int8Array — safe to transfer to main thread.
     * @returns {Int8Array}
     */
    get_spins_copy() {
        const ret = wasm.isingwasm_get_spins_copy(this.__wbg_ptr);
        var v1 = getArrayI8FromWasm0(ret[0], ret[1]).slice();
        wasm.__wbindgen_free(ret[0], ret[1] * 1, 1);
        return v1;
    }
    /**
     * Average magnetisation |<M>| of the current configuration.
     * @returns {number}
     */
    magnetisation() {
        const ret = wasm.isingwasm_magnetisation(this.__wbg_ptr);
        return ret;
    }
    /**
     * Lattice dimension N.
     * @returns {number}
     */
    n() {
        const ret = wasm.isingwasm_n(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Create a new lattice.
     * geometry: 0 = Square2D, 1 = Triangular2D, 2 = Cubic3D
     * @param {number} n
     * @param {number} geometry
     * @param {number} j
     * @param {number} h
     * @param {bigint} seed
     */
    constructor(n, geometry, j, h, seed) {
        const ret = wasm.isingwasm_new(n, geometry, j, h, seed);
        this.__wbg_ptr = ret >>> 0;
        IsingWasmFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Randomise the lattice (reset).
     */
    randomise() {
        wasm.isingwasm_randomise(this.__wbg_ptr);
    }
    /**
     * Update J and h without recreating the lattice.
     * @param {number} j
     * @param {number} h
     */
    set_params(j, h) {
        wasm.isingwasm_set_params(this.__wbg_ptr, j, h);
    }
    /**
     * Number of spins (N² or N³).
     * @returns {number}
     */
    size() {
        const ret = wasm.isingwasm_size(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Pointer into WASM linear memory for the spin array.
     * JS reads this as Int8Array for zero-copy access.
     * @returns {number}
     */
    spins_ptr() {
        const ret = wasm.isingwasm_spins_ptr(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * Run one full Metropolis sweep (N² spin-flip attempts).
     * @param {number} temperature
     */
    step(temperature) {
        wasm.isingwasm_step(this.__wbg_ptr, temperature);
    }
    /**
     * Run one Wolff cluster flip. Returns cluster size.
     * Falls back to Metropolis for J ≤ 0 or h ≠ 0.
     * @param {number} temperature
     * @returns {number}
     */
    step_wolff(temperature) {
        const ret = wasm.isingwasm_step_wolff(this.__wbg_ptr, temperature);
        return ret >>> 0;
    }
    /**
     * Run a full temperature sweep and return CSV.
     * use_wolff: true = Wolff cluster algorithm (faster near Tc),
     *            false = Metropolis (default, works for any J/h)
     * @param {number} t_min
     * @param {number} t_max
     * @param {number} steps
     * @param {number} warmup
     * @param {number} samples
     * @param {boolean} use_wolff
     * @returns {string}
     */
    temperature_sweep(t_min, t_max, steps, warmup, samples, use_wolff) {
        let deferred1_0;
        let deferred1_1;
        try {
            const ret = wasm.isingwasm_temperature_sweep(this.__wbg_ptr, t_min, t_max, steps, warmup, samples, use_wolff);
            deferred1_0 = ret[0];
            deferred1_1 = ret[1];
            return getStringFromWasm0(ret[0], ret[1]);
        } finally {
            wasm.__wbindgen_free(deferred1_0, deferred1_1, 1);
        }
    }
    /**
     * Run `n` sweeps — useful for warm-up from JS without per-frame overhead.
     * @param {number} temperature
     * @param {number} n
     */
    warm_up(temperature, n) {
        wasm.isingwasm_warm_up(this.__wbg_ptr, temperature, n);
    }
    /**
     * Run `n` Wolff cluster flips for warm-up.
     * @param {number} temperature
     * @param {number} n
     */
    warm_up_wolff(temperature, n) {
        wasm.isingwasm_warm_up_wolff(this.__wbg_ptr, temperature, n);
    }
}
if (Symbol.dispose) IsingWasm.prototype[Symbol.dispose] = IsingWasm.prototype.free;

function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_throw_6ddd609b62940d55: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./ising_bg.js": import0,
    };
}

const IsingWasmFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_isingwasm_free(ptr >>> 0, 1));

function getArrayI8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getInt8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedInt8ArrayMemory0 = null;
function getInt8ArrayMemory0() {
    if (cachedInt8ArrayMemory0 === null || cachedInt8ArrayMemory0.byteLength === 0) {
        cachedInt8ArrayMemory0 = new Int8Array(wasm.memory.buffer);
    }
    return cachedInt8ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return decodeText(ptr, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

const cachedTextEncoder = new TextEncoder();

if (!('encodeInto' in cachedTextEncoder)) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasm;
function __wbg_finalize_init(instance, module) {
    wasm = instance.exports;
    wasmModule = module;
    cachedInt8ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('ising_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
