// Pure-integer 4 KiB geometric vRAM page store wrapper over core.wasm binary ABI.
// Adapted from afflom/vRAM (MIT OR Apache-2.0): supports sparse multi-leaf addressing,
// 7 lossless geometric 4 KiB representations, chunked >16 MiB transfers, and a page-aligned blob allocator.

export const VRAM_PAGE_BYTES = 4096;
export const VRAM_MAX_TRANSFER_BYTES = 16 * 1024 * 1024;
export const VRAM_DEFAULT_MAX_PAGES = 65536;
export const VRAM_DEFAULT_MAX_BACKING_BYTES = 64 * 1024 * 1024;
export const VRAM_TAG_NAMES = Object.freeze([
  'fill',
  'repeated',
  'sparse',
  'selector',
  'corrected',
  'packed',
  'shared',
  'residual',
]);

const ERROR_MESSAGES = Object.freeze({
  [-1]: 'InvalidVramHandle',
  [-2]: 'VramOutOfRange',
  [-3]: 'VramInvalidArgument',
  [-4]: 'VramCapacityExceeded',
  [-5]: 'VramInternalError',
});

const encoder = new TextEncoder();
const decoder = new TextDecoder();

function vramError(code, op) {
  const label = ERROR_MESSAGES[code] || 'VramError';
  const err = new Error(`${label} (${code}) in ${op}`);
  err.code = label;
  err.errno = code;
  return err;
}

function requireSafeNonNegInt(value, name) {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0) {
    throw vramError(-3, name);
  }
  return value;
}

function requireI32NonNeg(value, name) {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 0 || value > 0x7fffffff) {
    throw vramError(-3, name);
  }
  return value;
}

function asUint8(input) {
  if (input instanceof Uint8Array) return input;
  if (input instanceof ArrayBuffer) return new Uint8Array(input);
  if (ArrayBuffer.isView(input)) {
    return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  }
  if (typeof input === 'string') return encoder.encode(input);
  throw new TypeError('vRAM payload must be Uint8Array, ArrayBuffer, typed view, or string');
}

export async function loadVramExports({ exports = null, wasmUrl = null, wasmBytes = null } = {}) {
  if (exports && typeof exports.vram_create === 'function') {
    return exports;
  }
  if (wasmBytes) {
    const { instance } = await WebAssembly.instantiate(wasmBytes, {});
    return instance.exports;
  }
  if (!wasmUrl) {
    throw new Error('loadVramExports requires preloaded exports, wasmBytes, or wasmUrl');
  }
  if (typeof WebAssembly.instantiateStreaming === 'function') {
    try {
      const { instance } = await WebAssembly.instantiateStreaming(fetch(wasmUrl), {});
      return instance.exports;
    } catch {
      // Fallback when server lacks application/wasm MIME or under file/mock fetch.
    }
  }
  const resp = await fetch(wasmUrl);
  if (!resp.ok) {
    throw new Error(`Failed to fetch core.wasm for vRAM (${resp.status})`);
  }
  const buf = await resp.arrayBuffer();
  const { instance } = await WebAssembly.instantiate(buf, {});
  return instance.exports;
}

export function createVramStore(
  exports,
  { maxPages = VRAM_DEFAULT_MAX_PAGES, maxBackingBytes = VRAM_DEFAULT_MAX_BACKING_BYTES } = {},
) {
  if (!exports || typeof exports.vram_create !== 'function') {
    throw new Error('core.wasm exports missing vram_create');
  }
  const reqMaxPages = requireI32NonNeg(maxPages, 'maxPages');
  const reqMaxBacking = requireI32NonNeg(maxBackingBytes, 'maxBackingBytes');
  const resolvedMaxPages = reqMaxPages === 0 ? VRAM_DEFAULT_MAX_PAGES : reqMaxPages;

  const {
    memory,
    holo_alloc,
    holo_free,
    vram_create,
    vram_destroy,
    vram_write_page,
    vram_read_page,
    vram_write_range,
    vram_read_range,
    vram_discard_pages,
    vram_stats_json,
  } = exports;

  const handle = vram_create(reqMaxPages, reqMaxBacking);
  if (handle <= 0) {
    throw vramError(handle, 'vram_create');
  }

  let destroyed = false;
  let nextFreePage = 0;
  const freeIntervals = []; // sorted [{ start, count }]
  const blobs = new Map(); // key -> { startPage, pageCount, byteLength }

  const ensureAlive = () => {
    if (destroyed) throw new Error('VramStore has been destroyed');
  };

  const withScratch = (size, fn) => {
    if (size === 0) return fn(0, 0);
    const ptr = holo_alloc(size);
    if (!ptr) throw new Error(`holo_alloc(${size}) returned null`);
    try {
      return fn(ptr, size);
    } finally {
      holo_free(ptr, size);
    }
  };

  const freeInterval = (start, count) => {
    if (count <= 0) return;
    freeIntervals.push({ start, count });
    freeIntervals.sort((a, b) => a.start - b.start);
    const merged = [];
    for (const seg of freeIntervals) {
      const last = merged[merged.length - 1];
      if (last && last.start + last.count === seg.start) {
        last.count += seg.count;
      } else {
        merged.push({ ...seg });
      }
    }
    // Trim trailing free tail back to nextFreePage.
    while (merged.length > 0 && merged[merged.length - 1].start + merged[merged.length - 1].count === nextFreePage) {
      nextFreePage = merged.pop().start;
    }
    freeIntervals.length = 0;
    freeIntervals.push(...merged);
  };

  const allocInterval = (pageCount) => {
    for (let i = 0; i < freeIntervals.length; i += 1) {
      const seg = freeIntervals[i];
      if (seg.count >= pageCount) {
        const start = seg.start;
        if (seg.count === pageCount) {
          freeIntervals.splice(i, 1);
        } else {
          seg.start += pageCount;
          seg.count -= pageCount;
        }
        return start;
      }
    }
    if (nextFreePage + pageCount > resolvedMaxPages) {
      throw vramError(-2, 'allocInterval');
    }
    const start = nextFreePage;
    nextFreePage += pageCount;
    return start;
  };

  const writePage = (pageIdx, data) => {
    ensureAlive();
    const p = requireI32NonNeg(pageIdx, 'pageIdx');
    const bytes = asUint8(data);
    if (bytes.byteLength !== VRAM_PAGE_BYTES) {
      throw new RangeError(`writePage requires exactly ${VRAM_PAGE_BYTES} bytes, got ${bytes.byteLength}`);
    }
    const rc = withScratch(VRAM_PAGE_BYTES, (ptr) => {
      new Uint8Array(memory.buffer, ptr, VRAM_PAGE_BYTES).set(bytes);
      return vram_write_page(handle, p, ptr, VRAM_PAGE_BYTES);
    });
    if (rc < 0) throw vramError(rc, 'vram_write_page');
    return rc;
  };

  const readPage = (pageIdx) => {
    ensureAlive();
    const p = requireI32NonNeg(pageIdx, 'pageIdx');
    return withScratch(VRAM_PAGE_BYTES, (ptr) => {
      const rc = vram_read_page(handle, p, ptr, VRAM_PAGE_BYTES);
      if (rc < 0) throw vramError(rc, 'vram_read_page');
      return new Uint8Array(memory.buffer, ptr, VRAM_PAGE_BYTES).slice();
    });
  };

  // Note: transfers > 16 MiB loop over sequential chunks; callers needing cross-chunk atomicity
  // (such as putBlob) allocate disjoint pages and discard the allocated span on error.
  const writeBytes = (offset, data) => {
    ensureAlive();
    const baseOffset = requireSafeNonNegInt(offset, 'offset');
    const bytes = asUint8(data);
    const maxBytes = resolvedMaxPages * VRAM_PAGE_BYTES;
    if (baseOffset + bytes.byteLength > maxBytes) {
      throw vramError(-2, 'writeBytes');
    }
    if (bytes.byteLength === 0) return;
    const scratchLen = Math.min(bytes.byteLength, VRAM_MAX_TRANSFER_BYTES);
    withScratch(scratchLen, (ptr) => {
      for (let pos = 0; pos < bytes.byteLength; pos += VRAM_MAX_TRANSFER_BYTES) {
        const chunkLen = Math.min(VRAM_MAX_TRANSFER_BYTES, bytes.byteLength - pos);
        const curOff = baseOffset + pos;
        const lo = curOff >>> 0;
        const hi = Math.floor(curOff / 0x100000000) >>> 0;
        new Uint8Array(memory.buffer, ptr, chunkLen).set(bytes.subarray(pos, pos + chunkLen));
        const rc = vram_write_range(handle, lo | 0, hi | 0, ptr, chunkLen);
        if (rc < 0) throw vramError(rc, 'vram_write_range');
      }
    });
  };

  const readBytes = (offset, length) => {
    ensureAlive();
    const baseOffset = requireSafeNonNegInt(offset, 'offset');
    const totalLen = requireSafeNonNegInt(length, 'length');
    const maxBytes = resolvedMaxPages * VRAM_PAGE_BYTES;
    if (baseOffset + totalLen > maxBytes) {
      throw vramError(-2, 'readBytes');
    }
    if (totalLen === 0) return new Uint8Array(0);
    const out = new Uint8Array(totalLen);
    const scratchLen = Math.min(totalLen, VRAM_MAX_TRANSFER_BYTES);
    withScratch(scratchLen, (ptr) => {
      for (let pos = 0; pos < totalLen; pos += VRAM_MAX_TRANSFER_BYTES) {
        const chunkLen = Math.min(VRAM_MAX_TRANSFER_BYTES, totalLen - pos);
        const curOff = baseOffset + pos;
        const lo = curOff >>> 0;
        const hi = Math.floor(curOff / 0x100000000) >>> 0;
        const rc = vram_read_range(handle, lo | 0, hi | 0, ptr, chunkLen);
        if (rc < 0) throw vramError(rc, 'vram_read_range');
        out.set(new Uint8Array(memory.buffer, ptr, chunkLen), pos);
      }
    });
    return out;
  };

  const discardPages = (startPage, pageCount) => {
    ensureAlive();
    const start = requireI32NonNeg(startPage, 'startPage');
    const count = requireI32NonNeg(pageCount, 'pageCount');
    const rc = vram_discard_pages(handle, start, count);
    if (rc < 0) throw vramError(rc, 'vram_discard_pages');
    return rc;
  };

  const putBlob = (key, data) => {
    ensureAlive();
    const k = String(key || '');
    if (!k) throw new Error('putBlob requires non-empty key');
    const bytes = asUint8(data);
    const pageCount = Math.max(1, Math.ceil(bytes.byteLength / VRAM_PAGE_BYTES));
    const startPage = allocInterval(pageCount);
    try {
      writeBytes(startPage * VRAM_PAGE_BYTES, bytes);
    } catch (err) {
      try {
        vram_discard_pages(handle, startPage, pageCount);
      } catch {}
      freeInterval(startPage, pageCount);
      throw err;
    }
    const prev = blobs.get(k);
    blobs.set(k, { startPage, pageCount, byteLength: bytes.byteLength });
    if (prev) {
      vram_discard_pages(handle, prev.startPage, prev.pageCount);
      freeInterval(prev.startPage, prev.pageCount);
    }
    return { key: k, startPage, pageCount, byteLength: bytes.byteLength };
  };

  const getBlob = (key) => {
    ensureAlive();
    const entry = blobs.get(String(key || ''));
    if (!entry) return null;
    return readBytes(entry.startPage * VRAM_PAGE_BYTES, entry.byteLength);
  };

  const hasBlob = (key) => !destroyed && blobs.has(String(key || ''));

  const deleteBlob = (key) => {
    ensureAlive();
    const k = String(key || '');
    const entry = blobs.get(k);
    if (!entry) return false;
    blobs.delete(k);
    discardPages(entry.startPage, entry.pageCount);
    freeInterval(entry.startPage, entry.pageCount);
    return true;
  };

  const listBlobs = () =>
    Array.from(blobs.entries()).map(([key, info]) => ({ key, ...info }));

  const stats = () => {
    ensureAlive();
    const packed = vram_stats_json(handle);
    if (packed === -1n) throw vramError(-1, 'vram_stats_json');
    const outPtr = Number((packed >> 32n) & 0xffffffffn);
    const outLen = Number(packed & 0xffffffffn);
    try {
      const jsonText = decoder.decode(new Uint8Array(memory.buffer, outPtr, outLen));
      return { ...JSON.parse(jsonText), handle, blobs: blobs.size };
    } finally {
      holo_free(outPtr, outLen);
    }
  };

  const destroy = () => {
    if (destroyed) return;
    destroyed = true;
    blobs.clear();
    freeIntervals.length = 0;
    vram_destroy(handle);
  };

  return Object.freeze({
    handle,
    pageBytes: VRAM_PAGE_BYTES,
    maxPages: resolvedMaxPages,
    writePage,
    readPage,
    writeBytes,
    readBytes,
    discardPages,
    putBlob,
    getBlob,
    hasBlob,
    deleteBlob,
    listBlobs,
    stats,
    destroy,
  });
}
