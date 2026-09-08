export class ApiError extends Error {
  constructor(status, detail, body) { super(detail || `HTTP ${status}`); this.status = status; this.detail = detail; this.body = body; }
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

export async function api(path, { method = 'GET', body, query, raw = false, signal } = {}) {
  const url = path + qs(query);
  const init = { method, headers: {}, signal };
  if (body !== undefined) { init.headers['Content-Type'] = 'application/json'; init.body = JSON.stringify(body); }
  const res = await fetch(url, init);
  if (raw) return res;
  const ct = res.headers.get('content-type') || '';
  const data = ct.includes('application/json') ? await res.json().catch(() => null) : await res.text();
  if (!res.ok) {
    const detail = data && typeof data === 'object' ? (data.detail || JSON.stringify(data)) : (data || res.statusText);
    throw new ApiError(res.status, typeof detail === 'string' ? detail : JSON.stringify(detail), data);
  }
  return data;
}
export const get = (path, query, opts = {}) => api(path, { ...opts, query });
export const post = (path, body, opts = {}) => api(path, { ...opts, method: 'POST', body });
export const put = (path, body, opts = {}) => api(path, { ...opts, method: 'PUT', body });
export const patch = (path, body, opts = {}) => api(path, { ...opts, method: 'PATCH', body });
export const del = (path, opts = {}) => api(path, { ...opts, method: 'DELETE' });

export const imgUrl = (id) => `/api/img/${id}`;
export const thumbUrl = (id, s = 256) => `/api/thumb/${id}?s=${s}`;
export { qs };

/** Follow a background job via SSE (polling fallback). Resolves with the final snapshot. */
export function watchJob(jobId, onUpdate) {
  return new Promise((resolve, reject) => {
    let done = false;
    const finish = (snap) => { if (done) return; done = true; if (snap.status === 'error') reject(new Error(snap.error || 'Job failed')); else resolve(snap); };
    let es;
    const poll = async () => {
      try {
        while (!done) {
          const { job } = await get(`/api/jobs/${jobId}`);
          onUpdate && onUpdate(job);
          if (['done', 'error', 'cancelled'].includes(job.status)) { finish(job); return; }
          await new Promise(r => setTimeout(r, 700));
        }
      } catch (e) { if (!done) { done = true; reject(e); } }
    };
    try {
      es = new EventSource(`/api/jobs/${jobId}/events`);
      es.onmessage = (ev) => {
        const snap = JSON.parse(ev.data);
        if (snap.status === 'missing') { es.close(); finish({ status: 'error', error: 'Job not found' }); return; }
        onUpdate && onUpdate(snap);
        if (['done', 'error', 'cancelled'].includes(snap.status)) { es.close(); finish(snap); }
      };
      es.onerror = () => { es.close(); if (!done) poll(); };
    } catch (e) { poll(); }
  });
}
