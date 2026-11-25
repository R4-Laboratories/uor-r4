// Opt-in browser client for the SpiralCore Stateless Edge Relay (R4-39, Casey-allard/lexlean-r4).
// Makes zero network requests on boot; only executes requests when a user or tool explicitly invokes it
// and a relay endpoint has been configured via localStorage or setRelayUrl().

const RELAY_STORAGE_KEY = 'uor.spiralcore.relayUrl';

function isLoopbackHost(hostname) {
  const host = String(hostname || '').toLowerCase();
  return host === '127.0.0.1' || host === 'localhost' || host === '[::1]' || host === '::1';
}

export function normalizeRelayEndpoint(raw) {
  const value = String(raw || '').trim();
  if (!value) return null;
  let parsed;
  try {
    parsed = new URL(value);
  } catch {
    throw Object.assign(new Error(`Invalid SpiralCore relay URL: ${value}`), { code: 'InvalidRelayUrl' });
  }
  const allowedProtocols = new Set(['https:', 'http:', 'wss:', 'ws:']);
  if (!allowedProtocols.has(parsed.protocol)) {
    throw Object.assign(new Error(`Unsupported SpiralCore relay protocol: ${parsed.protocol}`), {
      code: 'InvalidRelayUrl',
    });
  }
  if ((parsed.protocol === 'http:' || parsed.protocol === 'ws:') && !isLoopbackHost(parsed.hostname)) {
    throw Object.assign(
      new Error('Insecure http:// or ws:// SpiralCore relay is permitted on loopback hosts only'),
      { code: 'InsecureRelayUrl' },
    );
  }
  if (parsed.username || parsed.password) {
    throw Object.assign(new Error('Embedded credentials are not allowed in SpiralCore relay URLs'), {
      code: 'InvalidRelayUrl',
    });
  }
  if (!parsed.pathname || parsed.pathname === '/') {
    parsed.pathname = '/spvs';
  }
  return parsed.href;
}

function toHttpEndpoint(normalizedUrl) {
  const parsed = new URL(normalizedUrl);
  if (parsed.protocol === 'ws:') parsed.protocol = 'http:';
  else if (parsed.protocol === 'wss:') parsed.protocol = 'https:';
  return parsed.href;
}

export function createSpiralCoreRelayClient({
  getStorage = () => globalThis.localStorage,
  fetchImpl = (...args) => globalThis.fetch(...args),
  srcCid = 3,
  dstCid = 2,
} = {}) {
  let overrideUrl = null;
  let reqCounter = 0;

  const readConfiguredUrl = () => {
    if (overrideUrl !== null) return overrideUrl || null;
    try {
      const raw = getStorage()?.getItem?.(RELAY_STORAGE_KEY) || '';
      return raw ? normalizeRelayEndpoint(raw) : null;
    } catch {
      return null;
    }
  };

  const setRelayUrl = (raw) => {
    const trimmed = String(raw || '').trim();
    const normalized = trimmed ? normalizeRelayEndpoint(trimmed) : null;
    overrideUrl = normalized || '';
    try {
      const storage = getStorage();
      if (storage) {
        if (normalized) storage.setItem?.(RELAY_STORAGE_KEY, normalized);
        else storage.removeItem?.(RELAY_STORAGE_KEY);
      }
    } catch {}
    return normalized;
  };

  const getStatus = () => {
    const endpoint = readConfiguredUrl();
    return {
      configured: Boolean(endpoint),
      endpoint,
      protocol: endpoint ? new URL(endpoint).protocol.replace(/:$/, '') : null,
      srcCid,
      dstCid,
    };
  };

  const send = async (op, params = {}, { signal } = {}) => {
    const endpoint = readConfiguredUrl();
    if (!endpoint) {
      throw Object.assign(
        new Error('SpiralCore relay is not configured. Call spiralCoreRelay.setRelayUrl(...) with an https:// or wss:// relay URL.'),
        { code: 'RelayNotConfigured' },
      );
    }
    const reqId = `spvs-${Date.now()}-${++reqCounter}`;
    const httpUrl = toHttpEndpoint(endpoint);
    const resp = await fetchImpl(httpUrl, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        req_id: reqId,
        spvs_magic: 'SPVS',
        src_cid: srcCid,
        dst_cid: dstCid,
        op,
        params,
      }),
      credentials: 'omit',
      signal,
    });
    const payload = await resp.json().catch(() => null);
    if (!resp.ok || !payload?.ok) {
      const errCode = payload?.error_code || `HTTP_${resp.status}`;
      const errMsg = payload?.error_message || `SpiralCore relay ${op} failed (${errCode})`;
      throw Object.assign(new Error(errMsg), { code: errCode, status: resp.status, payload });
    }
    return payload.data;
  };

  return Object.freeze({
    setRelayUrl,
    getStatus,
    send,
    dnsQuery: ({ name, type = 'A', signal } = {}) => send('dns_query', { name, type }, { signal }),
    httpRequest: ({ method = 'GET', url, headers = {}, bodyB64 = null, maxBytes, signal } = {}) =>
      send('http_request', { method, url, headers, body_b64: bodyB64, max_bytes: maxBytes }, { signal }),
    webSearch: ({ query, limit = 5, signal } = {}) => send('web_search', { query, limit }, { signal }),
    gitSmartHttp: ({ repoUrl, pathAndQuery, method = 'GET', contentType = '', authBearer = null, bodyB64 = null, signal } = {}) =>
      send(
        'git_smart_http',
        {
          repo_url: repoUrl,
          path_and_query: pathAndQuery,
          method,
          content_type: contentType,
          auth_bearer: authBearer,
          body_b64: bodyB64,
        },
        { signal },
      ),
  });
}
