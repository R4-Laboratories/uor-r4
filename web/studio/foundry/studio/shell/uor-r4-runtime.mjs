// Real in-browser UOR-R4 runtime bridge:
// - Built-in models: `uor-r4:qwen3-0.6b` (default) & `uor-r4:qwen3-1.7b` (`UORR4V13` pure-integer via `./uor-v13-worker.mjs`)
// - Zero M1 dummy/seed models and zero weight-file upload controls.

const OPFS_V13_DIR = 'uor-r4-v13-packs';
const IDB_STORE = 'blobs';
const IDB_V13_NAME = 'uor-r4-v13-cache';
const DEFAULT_MODEL_SHORT = 'qwen3-0.6b';

const BASE = new URL('./', import.meta.url);

const FALLBACK_V13_MODEL = Object.freeze({
  id: 'qwen3-0.6b',
  kind: 'stitch-packs',
  primary: true,
  label: 'UOR-R4 Qwen3-0.6B',
  format: 'UORR4V13',
  manifest_url:
    'https://raw.githubusercontent.com/Casey-allard/lexlean-r4/081cb4f9cdf0e4e9dd1a160fa014615cb81db10e/manifest.json',
  stitch_manifest_url:
    'https://raw.githubusercontent.com/Casey-allard/lexlean-r4/081cb4f9cdf0e4e9dd1a160fa014615cb81db10e/manifest.stitch.json',
  manifest_sha256: '6bc863c2eb5a5a1e79f280eedcb3a90caa806c5fba0c708861f176b18b4a2d8e',
  stitch_manifest_sha256: 'fd29db5b1fbc3aa13c8a7803599ec2f143218a09140d4e771086d03bd608bc4d',
  container_sha256: '617ea77fd9c3d15285e351bd4f094cc0845cf5bc038e042a3fb5aadf0e406bda',
  size_bytes: 429484190,
  pack_count: 9,
  piece_count: 410,
  last_pack: 'pack_08.pack',
  context_window: 2048,
  note: 'Pure-integer UORR4V13 Qwen3-0.6B (410 pieces, 9 Stitch 2-of-3 parity packs). Cached locally in OPFS/IndexedDB.',
});

const FALLBACK_V13_17B_MODEL = Object.freeze({
  id: 'qwen3-1.7b',
  kind: 'stitch-packs',
  primary: false,
  label: 'UOR-R4 Qwen3-1.7B',
  format: 'UORR4V13',
  manifest_url:
    'https://raw.githubusercontent.com/Casey-allard/lexlean-r4/b35a6dcce1c527d91402a07dfa9ebd0928ee28f7/manifest.json',
  stitch_manifest_url:
    'https://raw.githubusercontent.com/Casey-allard/lexlean-r4/b35a6dcce1c527d91402a07dfa9ebd0928ee28f7/manifest.stitch.json',
  manifest_sha256: 'ac10aca3d9259786d483028d73acbfda5363ecadb45471a7e4604bb62970b195',
  stitch_manifest_sha256: 'dca8eacb0d8115e83405a70e2e5361d63ec37c3e200a4c761835ee89fc5b2d3b',
  container_sha256: '2c362d73046851ff69dfa7e1de197d4f0fee14031e9e53e29bf278c41b6bc8bf',
  size_bytes: 1140250782,
  pack_count: 23,
  piece_count: 1088,
  last_pack: 'pack_22.pack',
  context_window: 2048,
  note: 'Pure-integer UORR4V13 Qwen3-1.7B (1,088 pieces, 23 Stitch 2-of-3 parity packs). Streamed from OPFS/IndexedDB.',
});

const BUNDLED_FALLBACK_CATALOG = Object.freeze({
  schema: 'uor-r4.studio-models/1',
  models: [FALLBACK_V13_MODEL, FALLBACK_V13_17B_MODEL],
});

let v13Worker = null;
let catalogCache = null;
let loadedModelId = null;
let loadedCapabilities = null;
let defaultSpeculative = true;
let opfsStatus = 'untested';
let pendingOp = Promise.resolve();
let nextMsgId = 1;
const pendingV13Replies = new Map();
const progressListeners = new Set();

let hydrationState = {
  modelId: `uor-r4:${DEFAULT_MODEL_SHORT}`,
  phase: 'idle',
  percent: 0,
  packsDone: 0,
  packsTotal: 9,
  piecesDone: 0,
  piecesTotal: 410,
  bytesLoaded: 0,
  bytesTotal: 429484190,
  cached: false,
  cacheBackend: null,
  recoveredPieces: [],
  error: null,
};

function notifyProgress(patch) {
  hydrationState = { ...hydrationState, ...patch };
  for (const fn of progressListeners) {
    try {
      fn(hydrationState);
    } catch {}
  }
}

export function onProgress(fn) {
  progressListeners.add(fn);
  try {
    fn(hydrationState);
  } catch {}
  return () => progressListeners.delete(fn);
}

export function getHydrationState() {
  return { ...hydrationState };
}

function sanitizeMirrorBase(raw) {
  const s = String(raw || '').trim();
  if (!s) return '';
  try {
    const parsed = new URL(s, globalThis.location?.href || 'http://127.0.0.1/');
    if (!['http:', 'https:'].includes(parsed.protocol)) return '';
    const sameOrigin = Boolean(globalThis.location?.origin && parsed.origin === globalThis.location.origin);
    const isLoopback =
      parsed.hostname === '127.0.0.1' ||
      parsed.hostname === 'localhost' ||
      parsed.hostname === '[::1]';
    if (!sameOrigin && !isLoopback) return '';
    return parsed.href;
  } catch {
    return '';
  }
}

function getV13MirrorBase(shortId = DEFAULT_MODEL_SHORT) {
  try {
    const map = globalThis.__UOR_V13_MIRRORS__;
    if (map && typeof map === 'object' && typeof map[shortId] === 'string' && map[shortId].trim()) {
      return sanitizeMirrorBase(map[shortId]);
    }
    const storedMapRaw = globalThis.localStorage?.getItem('uor.v13.mirrors');
    if (storedMapRaw) {
      try {
        const storedMap = JSON.parse(storedMapRaw);
        if (storedMap && typeof storedMap[shortId] === 'string' && storedMap[shortId].trim()) {
          return sanitizeMirrorBase(storedMap[shortId]);
        }
      } catch {}
    }
    if (shortId === DEFAULT_MODEL_SHORT) {
      if (typeof globalThis.__UOR_V13_MIRROR__ === 'string' && globalThis.__UOR_V13_MIRROR__.trim()) {
        return sanitizeMirrorBase(globalThis.__UOR_V13_MIRROR__);
      }
      const q = new URLSearchParams(globalThis.location?.search || '').get('v13Mirror');
      if (q && q.trim()) return sanitizeMirrorBase(q);
      const s = globalThis.localStorage?.getItem('uor.v13.mirror');
      if (s && s.trim()) return sanitizeMirrorBase(s);
    }
  } catch {}
  return '';
}

function normalizeModelId(modelId) {
  const raw = String(modelId || `uor-r4:${DEFAULT_MODEL_SHORT}`).trim();
  const short = raw.startsWith('uor-r4:') ? raw.slice('uor-r4:'.length) : raw;
  if (short === 'qwen3-0.6b-foundry') return 'qwen3-0.6b';
  return short;
}

function lastPackForEntry(entry) {
  if (entry?.last_pack) return entry.last_pack;
  return entry?.id === 'qwen3-1.7b' ? 'pack_22.pack' : 'pack_08.pack';
}

function toFullModelId(shortId) {
  return shortId.startsWith('uor-r4:') ? shortId : `uor-r4:${shortId}`;
}

async function openFallbackIdb(name = IDB_V13_NAME) {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(name, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(IDB_STORE)) {
        db.createObjectStore(IDB_STORE);
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function hasCachedFiles(containerSha, files) {
  if (!containerSha || !Array.isArray(files) || !files.length) return false;
  try {
    if (navigator.storage?.getDirectory) {
      const root = await navigator.storage.getDirectory();
      const base = await root.getDirectoryHandle(OPFS_V13_DIR, { create: false });
      const scoped = await base.getDirectoryHandle(containerSha, { create: false });
      for (const file of files) {
        await scoped.getFileHandle(file, { create: false });
      }
      return true;
    }
  } catch {}
  try {
    const db = await openFallbackIdb(IDB_V13_NAME);
    const existing = await new Promise((resolve) => {
      const tx = db.transaction(IDB_STORE, 'readonly');
      const store = tx.objectStore(IDB_STORE);
      let idx = 0;
      const checkNext = () => {
        if (idx >= files.length) {
          resolve(true);
          return;
        }
        const req = store.getKey(`${containerSha}/${files[idx]}`);
        idx += 1;
        req.onsuccess = () => {
          if (req.result === undefined) {
            resolve(false);
          } else {
            checkNext();
          }
        };
        req.onerror = () => resolve(false);
      };
      checkNext();
    });
    return existing;
  } catch {
    return false;
  }
}

async function isV13Cached(containerSha, lastPack = 'pack_08.pack') {
  return hasCachedFiles(containerSha, [lastPack]);
}

async function hasOfflineV13Cache(containerSha, lastPack = 'pack_08.pack') {
  return hasCachedFiles(containerSha, ['manifest.json', 'manifest.stitch.json', lastPack]);
}

function ensureV13Worker() {
  if (v13Worker) return v13Worker;
  v13Worker = new Worker(new URL('./uor-v13-worker.mjs', BASE), { type: 'module' });
  v13Worker.addEventListener('message', ({ data }) => {
    if (!data || typeof data !== 'object') return;
    if (data.action === 'v13_progress') {
      if (data.cacheBackend && data.cacheBackend !== 'network') {
        opfsStatus = data.cacheBackend;
      }
      notifyProgress({
        phase: data.phase,
        percent: data.percent,
        packsDone: data.packsDone,
        packsTotal: data.packsTotal,
        piecesDone: data.piecesDone,
        piecesTotal: data.piecesTotal,
        bytesLoaded: data.bytesLoaded,
        bytesTotal: data.bytesTotal,
        cached: Boolean(data.cached),
        cacheBackend: data.cacheBackend || null,
        recoveredPieces: Array.isArray(data.recoveredPieces) ? data.recoveredPieces : [],
        error: null,
      });
      return;
    }
    const entry = pendingV13Replies.get(data.id);
    if (!entry) return;
    if (data.action === 'v13_token') {
      entry.onToken?.(data);
      return;
    }
    if (data.action === 'v13_error') {
      pendingV13Replies.delete(data.id);
      const err = new Error(data.error || 'UOR-R4 V13 operation failed');
      if (data.cancelled) err.name = 'AbortError';
      entry.reject(err);
      return;
    }
    if (entry.expectAction && data.action !== entry.expectAction) {
      return;
    }
    pendingV13Replies.delete(data.id);
    entry.resolve(data);
  });
  v13Worker.addEventListener('error', (err) => {
    for (const [id, entry] of pendingV13Replies.entries()) {
      pendingV13Replies.delete(id);
      entry.reject(new Error(err?.message || 'UOR-R4 V13 worker error'));
    }
  });
  return v13Worker;
}

function callV13Worker(action, payload, expectAction, { onToken, signal } = {}) {
  const w = ensureV13Worker();
  const id = `v13-${nextMsgId++}`;
  return new Promise((resolve, reject) => {
    let settled = false;
    let cancelTimer = null;
    const onAbort = () => {
      try {
        w.postMessage({ action: 'cancel_v13', id });
      } catch {}
      cancelTimer = setTimeout(() => {
        if (!settled && pendingV13Replies.has(id)) {
          pendingV13Replies.delete(id);
          settled = true;
          reject(Object.assign(new Error('V13 operation cancelled'), { name: 'AbortError' }));
        }
      }, 15000);
    };
    if (signal) {
      if (signal.aborted) {
        reject(Object.assign(new Error('V13 operation cancelled'), { name: 'AbortError' }));
        return;
      }
      signal.addEventListener('abort', onAbort, { once: true });
    }
    pendingV13Replies.set(id, {
      expectAction,
      onToken,
      resolve: (value) => {
        if (cancelTimer) clearTimeout(cancelTimer);
        if (settled) return;
        settled = true;
        signal?.removeEventListener('abort', onAbort);
        resolve(value);
      },
      reject: (err) => {
        if (cancelTimer) clearTimeout(cancelTimer);
        if (settled) return;
        settled = true;
        signal?.removeEventListener('abort', onAbort);
        reject(err);
      },
    });
    w.postMessage({ action, id, payload });
  });
}

function serializeOp(fn) {
  const next = pendingOp.then(fn, fn);
  pendingOp = next.catch(() => {});
  return next;
}

export async function getCatalog({ refresh = false } = {}) {
  if (!catalogCache || refresh) {
    try {
      const resp = await fetch(new URL('./models.json', BASE), { cache: 'no-store' });
      if (resp.ok) {
        catalogCache = await resp.json();
      } else {
        catalogCache = BUNDLED_FALLBACK_CATALOG;
      }
    } catch {
      catalogCache = BUNDLED_FALLBACK_CATALOG;
    }
  }
  const models = [];
  for (const entry of catalogCache.models || []) {
    if (entry.id === 'synthetic-seed-5' || entry.id === 'm1-discrete-trunk-seed-1') continue;
    if (entry.format !== 'UORR4V13' && entry.kind !== 'stitch-packs') continue;
    const cached = await isV13Cached(entry.container_sha256, lastPackForEntry(entry));
    models.push({
      ...entry,
      apiName: toFullModelId(entry.id),
      cached,
      loaded: loadedModelId === entry.id,
    });
  }
  return {
    schema: catalogCache.schema || 'uor-r4.studio-models/1',
    status: getStatus(),
    hydration: getHydrationState(),
    models,
  };
}

export function getStatus() {
  return {
    projectStatus: 'Pre-alpha · M1 complete (1 of 8 milestones) · D90 Studio-only V13 Foundry',
    engine: loadedCapabilities?.engine || 'uor-foundry-v13/wasm',
    loadedModelId: loadedModelId ? toFullModelId(loadedModelId) : null,
    capabilities: loadedCapabilities,
    opfsStatus,
    hydration: getHydrationState(),
    weightsMatched: Boolean(
      loadedCapabilities &&
        (loadedCapabilities.expected_sha256 == null || loadedCapabilities.sha256_verified === true),
    ),
  };
}

async function loadV13Entry(entry, { signal } = {}) {
  const mirrorBaseUrl = getV13MirrorBase(entry.id);
  const fullModelId = toFullModelId(entry.id);
  loadedModelId = null;
  loadedCapabilities = null;
  notifyProgress({
    modelId: fullModelId,
    phase: 'hydrating',
    percent: 0,
    packsDone: 0,
    packsTotal: entry.pack_count || (entry.id === 'qwen3-1.7b' ? 23 : 9),
    piecesDone: 0,
    piecesTotal: entry.piece_count || (entry.id === 'qwen3-1.7b' ? 1088 : 410),
    bytesLoaded: 0,
    bytesTotal: entry.size_bytes || 0,
    cached: false,
    cacheBackend: null,
    recoveredPieces: [],
    error: null,
  });
  try {
    const resp = await callV13Worker(
      'load_v13',
      {
        modelId: fullModelId,
        manifestUrl: entry.manifest_url,
        stitchManifestUrl: entry.stitch_manifest_url,
        manifestSha256: entry.manifest_sha256,
        stitchManifestSha256: entry.stitch_manifest_sha256,
        containerSha256: entry.container_sha256,
        sizeBytes: entry.size_bytes,
        mirrorBaseUrl,
        lastBackend: opfsStatus !== 'untested' ? opfsStatus : undefined,
      },
      'v13_loaded',
      { signal },
    );
    loadedModelId = entry.id;
    if (resp.cacheBackend && resp.cacheBackend !== 'network') {
      opfsStatus = resp.cacheBackend;
    }
    loadedCapabilities = {
      engine: 'uor-foundry-v13/wasm',
      format: 'UORR4V13',
      can_predict: false,
      can_generate: true,
      artifact_sha256: resp.status.container_sha256,
      expected_sha256: entry.container_sha256 || resp.status.container_sha256,
      sha256_verified: resp.status.container_sha256 === (entry.container_sha256 || resp.status.container_sha256),
      float_ops: 0,
      dense_matmul_ops: 0,
      n_layers: resp.status.n_layers,
      d_model: resp.status.d_model,
      vocab: resp.status.vocab,
      max_seq: resp.status.max_seq,
      cacheBackend: resp.cacheBackend,
      recoveredPieces: resp.recoveredPieces || [],
    };
    const isCached = Boolean(resp.reusedMount || resp.cacheBackend === 'opfs' || resp.cacheBackend === 'indexeddb');
    notifyProgress({
      modelId: fullModelId,
      phase: 'ready',
      percent: 100,
      cached: isCached,
      cacheBackend: resp.cacheBackend,
      recoveredPieces: resp.recoveredPieces || [],
      error: null,
    });
    return {
      modelId: fullModelId,
      capabilities: loadedCapabilities,
      sha256: loadedCapabilities.artifact_sha256,
      bytes: entry.size_bytes,
      cacheBackend: resp.cacheBackend,
      recoveredPieces: resp.recoveredPieces || [],
      cached: isCached,
    };
  } catch (err) {
    loadedModelId = null;
    loadedCapabilities = null;
    notifyProgress({
      modelId: fullModelId,
      phase: 'error',
      error: String(err?.message || err),
    });
    throw err;
  }
}

async function ensureModelLoadedQueued(shortId, { signal } = {}) {
  if (loadedModelId === shortId && loadedCapabilities) {
    return {
      modelId: toFullModelId(shortId),
      capabilities: loadedCapabilities,
      sha256: loadedCapabilities.artifact_sha256,
      cached: true,
    };
  }
  const cat = await getCatalog();
  const entry = cat.models.find((m) => m.id === shortId);
  if (!entry) {
    throw new Error(`Unknown UOR-R4 model id: ${shortId}`);
  }
  return loadV13Entry(entry, { signal });
}

export async function ensureModelLoaded(modelId, { signal } = {}) {
  const shortId = normalizeModelId(modelId);
  return serializeOp(() => ensureModelLoadedQueued(shortId, { signal }));
}

export async function generate({
  modelId = `uor-r4:${DEFAULT_MODEL_SHORT}`,
  prompt,
  systemPrompt = '',
  messages,
  maxNewTokens = 768,
  stopSequences = ['</tool_call>'],
  sliceTokens = 4,
  prefillSliceTokens = 32,
  speculative,
  signal,
  onToken,
} = {}) {
  const shortId = normalizeModelId(modelId);
  return serializeOp(async () => {
    await ensureModelLoadedQueued(shortId, { signal });
    const effectiveSpec = typeof speculative === 'boolean' ? speculative : defaultSpeculative;
    const result = await callV13Worker(
      'generate_v13',
      {
        systemPrompt,
        messages,
        prompt,
        maxNewTokens,
        stopSequences,
        sliceTokens,
        prefillSliceTokens,
        speculative: effectiveSpec,
      },
      'v13_complete',
      { onToken, signal },
    );
    return {
      fullText: result.fullText,
      tokenCount: result.tokensEmitted,
      stopReason: result.stopReason,
      float_ops: result.float_ops ?? 0,
      dense_matmul_ops: result.dense_matmul_ops ?? 0,
      prefixReusedTokens: result.prefixReusedTokens ?? 0,
      draftProposed: result.draftProposed ?? 0,
      draftAccepted: result.draftAccepted ?? 0,
      mlpMode: result.mlpMode || 'DenseExact',
      capabilities: loadedCapabilities,
      weightsMatched:
        loadedCapabilities?.expected_sha256 == null || loadedCapabilities?.sha256_verified === true,
    };
  });
}

export async function prewarmIfCached({
  modelId = `uor-r4:${DEFAULT_MODEL_SHORT}`,
  systemPrompt = '',
  signal,
} = {}) {
  const shortId = normalizeModelId(modelId);
  if (loadedModelId !== shortId) {
    const cat = await getCatalog();
    const entry = cat.models.find((m) => m.id === shortId);
    if (!entry) {
      return { ok: false, skipped: 'unknown_model', cached: false };
    }
    const cached = await hasOfflineV13Cache(entry.container_sha256, lastPackForEntry(entry));
    if (!cached) {
      return { ok: false, skipped: 'not_cached', cached: false };
    }
  }
  return serializeOp(async () => {
    if (loadedModelId !== shortId) {
      const cat = await getCatalog();
      const entry = cat.models.find((m) => m.id === shortId);
      if (!entry || !(await hasOfflineV13Cache(entry.container_sha256, lastPackForEntry(entry)))) {
        return { ok: false, skipped: 'not_cached', cached: false };
      }
      await loadV13Entry(entry, { signal });
    }
    return callV13Worker(
      'prewarm_v13',
      { systemPrompt: String(systemPrompt || '') },
      'v13_prewarmed',
      { signal },
    );
  });
}

export async function configure({
  speculative,
  shortlistK,
  mlpShortlistK,
  mlpMode,
  flockWindow,
  signal,
} = {}) {
  const cfg = {};
  if (typeof speculative === 'boolean') cfg.speculative = speculative;
  if (Number.isInteger(shortlistK)) cfg.shortlist_k = shortlistK;
  if (Number.isInteger(mlpShortlistK)) cfg.mlp_shortlist_k = mlpShortlistK;
  if (typeof mlpMode === 'string' && mlpMode) cfg.mlp_policy = mlpMode;
  if (Number.isInteger(flockWindow)) cfg.flock_window = flockWindow;
  const res = await serializeOp(() => callV13Worker('configure_v13', cfg, 'v13_configured', { signal }));
  if (typeof speculative === 'boolean') {
    defaultSpeculative = speculative;
  }
  return res;
}

export async function testStitchRecover({ mirrorBaseUrl = '', groupIndex = 0, signal } = {}) {
  return callV13Worker(
    'test_stitch_recover',
    {
      manifestUrl: FALLBACK_V13_MODEL.manifest_url,
      stitchManifestUrl: FALLBACK_V13_MODEL.stitch_manifest_url,
      manifestSha256: FALLBACK_V13_MODEL.manifest_sha256,
      stitchManifestSha256: FALLBACK_V13_MODEL.stitch_manifest_sha256,
      mirrorBaseUrl: mirrorBaseUrl ? sanitizeMirrorBase(mirrorBaseUrl) : getV13MirrorBase(),
      groupIndex,
    },
    'stitch_recover_result',
    { signal },
  );
}
