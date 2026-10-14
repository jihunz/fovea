export class ApiError extends Error {
  constructor(status, detail, body) { super(detail || `HTTP ${status}`); this.status = status; this.detail = detail; this.body = body; this.network = status === 0; }
}

function qs(query) {
  if (!query) return '';
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(query)) {
    if (v == null || v === '' || v === false) continue;
    if (Array.isArray(v)) { if (v.length) p.set(k, v.join(',')); }
    else p.set(k, String(v));
  }
  const s = p.toString();
  return s ? `?${s}` : '';
}

// ------------------------------------------------------------------ connection state
// One place knows whether the server answers; the shell shows a banner while it does not.
const connSubs = new Set();
let online = true;
function setOnline(v) { if (v === online) return; online = v; connSubs.forEach(fn => { try { fn(v); } catch (e) { console.error(e); } }); }
export function onConnection(fn) { connSubs.add(fn); return () => connSubs.delete(fn); }
export const isOnline = () => online;
// Every endpoint that starts background work answers {job}: the shell hears about it at once (Activity, banners)
// instead of on its next poll.
const jobSubs = new Set();
export function onJobStarted(fn) { jobSubs.add(fn); return () => jobSubs.delete(fn); }
export const NETWORK_MESSAGE = 'Fovea server not reachable — check that it is still running';

/** A readable message from any error body: FastAPI 422 lists become "field: problem", HTML error pages from a
    proxy become a short text, never a wall of markup. */
export function detailOf(data, status, statusText) {
  if (data && typeof data === 'object') {
    const d = data.detail;
    if (typeof d === 'string') return d;
    if (Array.isArray(d)) {
      return d.slice(0, 3).map(x => {
        if (!x || typeof x !== 'object') return String(x);
        const loc = Array.isArray(x.loc) ? x.loc.filter(p => p !== 'body' && p !== 'query').join('.') : '';
        return `${loc ? loc + ': ' : ''}${x.msg || 'invalid'}`;
      }).join('; ') + (d.length > 3 ? ` (+${d.length - 3} more)` : '');
    }
    if (d && typeof d === 'object') return d.message || JSON.stringify(d).slice(0, 200);
    return statusText || `HTTP ${status}`;
  }
  if (typeof data === 'string' && data.trim()) {
    const text = data.replace(/<style[\s\S]*?<\/style>|<script[\s\S]*?<\/script>/gi, ' ').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
    return (text.length > 200 ? text.slice(0, 200) + '…' : text) || statusText || `HTTP ${status}`;
  }
  return statusText || `HTTP ${status}`;
}

export async function api(path, { method = 'GET', body, query, raw = false, signal } = {}) {
  const url = path + qs(query);
  const init = { method, headers: {}, signal };
  if (body !== undefined) { init.headers['Content-Type'] = 'application/json'; init.body = JSON.stringify(body); }
  let res;
  try { res = await fetch(url, init); }
  catch (e) {
    if (e && e.name === 'AbortError') throw e;
    setOnline(false);
    throw new ApiError(0, NETWORK_MESSAGE, null);
  }
  setOnline(true);
  if (raw) return res;
  const ct = res.headers.get('content-type') || '';
  const data = ct.includes('application/json') ? await res.json().catch(() => null) : await res.text().catch(() => '');
  if (!res.ok) throw new ApiError(res.status, detailOf(data, res.status, res.statusText), data);
  if (method !== 'GET' && data && typeof data === 'object' && data.job && data.job.id && ['queued', 'running'].includes(data.job.status)) {
    jobSubs.forEach(fn => { try { fn(data.job); } catch (e) { console.error(e); } });
  }
  return data;
}
export const get = (path, query, opts = {}) => api(path, { ...opts, query });
export const post = (path, body, opts = {}) => api(path, { ...opts, method: 'POST', body });
export const put = (path, body, opts = {}) => api(path, { ...opts, method: 'PUT', body });
export const patch = (path, body, opts = {}) => api(path, { ...opts, method: 'PATCH', body });
export const del = (path, opts = {}) => api(path, { ...opts, method: 'DELETE' });

/** Full image. Pass `v` (the image's mtime) to get an immutable, version-keyed URL; without it the browser
    revalidates (ETag) on every view, so an edited file or a reused id never shows a stale picture. */
export const imgUrl = (id, v) => `/api/img/${id}${v != null ? `?v=${Math.round(v)}` : ''}`;
// Bump THUMB_RENDER whenever the server changes how thumbnails are drawn: versioned URLs are cached for good.
export const THUMB_RENDER = 3;   // 3: 16-bit, ICC and large non-JPEG thumbnails render differently (backend-io)
/** `v` = the image's mtime. With it the browser may cache the thumbnail indefinitely (a changed file gets a
    new URL); without it every view revalidates with a cheap 304. */
export const thumbUrl = (id, s = 256, v) => `/api/thumb/${id}?s=${s}${v != null ? `&v=${Math.round(v)}.${THUMB_RENDER}` : ''}`;
export { qs };

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

/**
 * Follow a background job via SSE (polling fallback).
 * Resolves with the final snapshot for 'done' AND 'cancelled' — check snap.status: a cancelled job may
 * already have written part of its work, and snap.result is null. Rejects when the job errors or no longer
 * exists. A dropped connection is NOT a failure: the stream reconnects, then falls back to polling with
 * backoff, and only a 404 or an 'error' status ends the watch (opts.onConnection(false|true) reports gaps).
 * The promise carries stop(): close the stream without settling, for views that unmount mid-job.
 */
export function watchJob(jobId, onUpdate, opts = {}) {
  let es = null, done = false, sseErrors = 0;
  const TERMINAL = ['done', 'error', 'cancelled'];
  const conn = (v) => { try { opts.onConnection && opts.onConnection(v); } catch (e) { /* view gone */ } };
  const p = new Promise((resolve, reject) => {
    const finish = (snap) => {
      if (done) return;
      done = true; if (es) es.close();
      if (snap.status === 'error') reject(new Error(snap.error || 'Job failed')); else resolve(snap);
    };
    const fail = (e) => { if (!done) { done = true; if (es) es.close(); reject(e); } };
    const poll = async () => {
      let wait = 700, misses = 0;
      while (!done) {
        try {
          const { job } = await get(`/api/jobs/${jobId}`);
          if (done) return;
          if (misses) conn(true);
          misses = 0; wait = 700;
          onUpdate && onUpdate(job);
          if (TERMINAL.includes(job.status)) { finish(job); return; }
        } catch (e) {
          if (done) return;
          if (e instanceof ApiError && e.status === 404) { fail(new Error('The job is no longer known to the server (it may have restarted)')); return; }
          misses++; conn(false);
          wait = Math.min(10000, misses === 1 ? 1000 : wait * 2);
        }
        await sleep(wait);
      }
    };
    try {
      es = new EventSource(`/api/jobs/${jobId}/events`);
      es.onopen = () => { if (sseErrors) conn(true); sseErrors = 0; };
      es.onmessage = (ev) => {
        if (done) return;
        let snap;
        try { snap = JSON.parse(ev.data); } catch (e) { return; }
        if (snap.status === 'missing') { finish({ status: 'error', error: 'Job not found' }); return; }
        onUpdate && onUpdate(snap);
        if (TERMINAL.includes(snap.status)) finish(snap);
      };
      es.onerror = () => {
        if (done) return;
        sseErrors++;
        // The browser retries a dropped stream by itself; after a few failures (or a hard close) poll instead.
        if (es.readyState === EventSource.CLOSED || sseErrors >= 3) { es.close(); conn(false); poll(); }
      };
    } catch (e) { poll(); }
  });
  p.stop = () => { done = true; if (es) es.close(); };
  return p;
}
