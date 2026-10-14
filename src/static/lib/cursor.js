// Paged, filtered image list shared by Explore, Inspect and Annotate — a FROZEN working set.
//
// The list a person works through must not shift under them. Reviewing "Unreviewed" or labeling "No boxes"
// removes each finished image from the live query, so plain OFFSET paging skipped one image per edit (after 200
// labeled images the next page started 200 images too far). Here, index i always names the same image once it is
// loaded, and pages beyond what is loaded are anchored on the last loaded image instead of on a raw offset:
//   · fast path — one request: the page is fetched one row early and is accepted only when that row is the anchor;
//   · exact path — the first image after the anchor in the live order (neighbor) gives the offset to read from.
// Images that stop matching stay in the working set (a reviewed image is still image #57); images that start
// matching appear only after refresh(). `total` is the loaded prefix plus what the live list still holds after it.
//
// OFFSET is never trusted blindly: SQLite 3.51.0 (the build bundled with current Python) mis-applies OFFSET when a
// filter is an EXISTS subquery (cls=, issue=<flag>, ai=) — it skips one row per matching box, not per image. Every
// page read at an offset is therefore checked (the anchor row, or the first row's position), and a page that does
// not line up is read by key instead (after=<id>, or the ordered id list), which LIMIT alone answers correctly.
import { get, post } from './api.js';

export class ImageCursor {
  constructor({ dsId, filters = {}, sort = 'id', order = 'asc', pageSize = 120, boxes = false }) {
    this.dsId = dsId; this.filters = { ...filters }; this.sort = sort; this.order = order; this.pageSize = pageSize; this.boxes = boxes;
    this.items = []; this.total = null; this.loading = new Map(); this.version = 0;
    this.byId = new Map();   // image id -> index in the working set
    this.shift = 0;          // live offset minus working-set index at the frontier, learned from the last alignment
    this.hint = null;        // positionOf's answer for "the first live image after anchor X": its page fetch reuses it
  }
  get params() { return { ...this.filters, sort: this.sort, order: this.order, boxes: this.boxes ? 1 : 0 }; }
  get queryParams() { return { ...this.filters, sort: this.sort, order: this.order }; }
  reset() { this.items = []; this.total = null; this.loading.clear(); this.byId.clear(); this.shift = 0; this.hint = null; this.version++; }
  /** Re-query from scratch: images that started matching the filters since the list opened are included. */
  refresh() { this.reset(); return this.loadPage(0); }
  get loadedCount() { return this.items.filter(Boolean).length; }

  loadPage(page) {
    const key = page;
    if (this.loading.has(key)) return this.loading.get(key);
    const v = this.version;
    const p = this._fetchPage(page, v).then((res) => {
      if (this.loading.get(key) === p) this.loading.delete(key);
      return res;
    }, (e) => {
      // Only our own entry: after reset() a newer request for the same page may own this key.
      if (this.loading.get(key) === p) this.loading.delete(key);
      throw e;
    });
    this.loading.set(key, p);
    return p;
  }

  _list(offset, limit, extra = null) { return get(`/api/datasets/${this.dsId}/images`, { ...this.params, offset, limit, ...(extra || {}) }); }
  _position(id) { return get(`/api/datasets/${this.dsId}/images/position`, { ...this.queryParams, id }); }

  /** `limit` live rows from live offset `off`, checked: when the first row is not at `off` the server's OFFSET is
      unreliable, and the rows are taken from the ordered id list (LIMIT only) instead. `known` (id -> live
      position, just asked) spares the check when the first row is one of those images. */
  async _readAt(off, limit, known = null) {
    const res = await this._list(off, limit);
    if (off === 0) return res;
    if (res.items.length) {
      if (known && known.get(res.items[0].id) === off) return res;
      const p = await this._position(res.items[0].id); if (p.position === off) return res;
    }
    else if (res.total == null || off >= res.total) return res;        // really past the end (an overshoot is empty too)
    const r = await get(`/api/datasets/${this.dsId}/images/ids`, { ...this.queryParams, limit: off + limit });
    const ids = (r.ids || []).slice(off, off + limit);
    if (!ids.length) return { ...res, items: [] };
    const rows = await post(`/api/datasets/${this.dsId}/images/by-ids`, { ids, boxes: this.boxes ? 1 : 0 });
    return { ...res, total: r.total != null && r.total < off + limit ? r.total : res.total, items: rows.items || [] };
  }

  async _fetchPage(page, v) {
    const ps = this.pageSize, start = page * ps;
    // Sequential loads align exactly: wait for the page before this one instead of guessing past it.
    if (page > 0 && this.loading.has(page - 1)) { try { await this.loading.get(page - 1); } catch (e) { /* align below */ } }
    if (v !== this.version) return { items: [], total: this.total };
    let j = start - 1;
    while (j >= 0 && !this.items[j]) j--;
    if (j < 0) {                                  // nothing before this page: the live order is the working set
      const res = await this._readAt(start, ps);
      this._store(start, res.items, start, res.total, v);
      return res;
    }
    const anchor = this.items[j], gap = start - 1 - j;
    if (gap === 0) {
      if (start + this.shift - 1 >= 0) {          // fast path: one request, accepted only when it starts at the anchor
        const guess = start + this.shift - 1;
        const res = await this._list(guess, ps + 1);
        if (v !== this.version) return res;
        if (res.items.length && res.items[0].id === anchor.id) {
          this._store(start, res.items.slice(1), guess + 1, res.total, v);
          return { ...res, items: res.items.slice(1) };
        }
      }
      // By key: the rows right after the anchor in the live order — exact whatever moved before it, and no OFFSET.
      const [res, ap] = await Promise.all([this._list(0, ps, { after: anchor.id }), this._position(anchor.id).catch(() => ({ position: null }))]);
      if (v !== this.version) return res;
      if (!res.items.length) { this._endAt(j + 1); return { ...res, items: [] }; }
      let off = ap.position != null ? ap.position + 1 : null;
      if (off == null) { const fp = await this._position(res.items[0].id); if (v !== this.version) return res; off = fp.position != null ? fp.position : start + this.shift; }
      this.shift = off - start;
      this._store(start, res.items, off, res.total, v);
      return res;
    }
    // The first live image after the anchor and its live position. positionOf has usually just asked exactly that
    // (a deep link opens far down the list): two serial round trips saved — ~130 ms on a 47k audit-sorted list.
    const hint = this._takeHint(anchor.id, v);
    const nb = hint ? { item: hint.item } : await get(`/api/datasets/${this.dsId}/images/neighbor`, { ...this.queryParams, id: anchor.id, dir: 'next' });
    if (v !== this.version) return { items: [], total: this.total };
    if (!nb.item) { this._endAt(j + 1); return { items: [], total: this.total }; }
    const pos = hint ? { position: hint.position } : await this._position(nb.item.id);
    if (v !== this.version) return { items: [], total: this.total };
    if (pos.position == null) { this._endAt(j + 1); return { items: [], total: this.total }; }
    const off = pos.position + gap;
    const res = await this._readAt(off, ps, hint && hint.known);
    if (v !== this.version) return res;
    this.shift = off - start;
    this._store(start, res.items, off, res.total, v);
    return res;
  }

  /** positionOf's neighbor + position for this anchor, if it is fresh (same list version, under 2 s old). */
  _takeHint(anchorId, v) {
    const h = this.hint;
    if (!h || h.anchor !== anchorId || h.v !== v || Date.now() - h.at > 2000) return null;
    this.hint = null;
    return h;
  }

  _lastLoaded() { for (let i = this.items.length - 1; i >= 0; i--) if (this.items[i]) return i; return -1; }

  _store(start, items, liveOffset, liveTotal, v) {
    if (v !== this.version) return;
    items.forEach((it, k) => {
      const i = start + k;
      if (this.items[i]) return;                  // frozen: a loaded index never changes its image
      this.items[i] = it;
      if (!this.byId.has(it.id)) this.byId.set(it.id, i);
    });
    const end = start + items.length;             // first index after this page
    if (this._lastLoaded() < end) {               // the frontier: re-derive how much list remains after it
      const rest = Math.max(0, (liveTotal ?? 0) - liveOffset);
      this.total = start + Math.max(items.length, rest);
    } else if (this.total == null || this.total < end) this.total = end;
  }

  _endAt(n) { if (this._lastLoaded() < n) this.total = n; }

  async ensure(index) {
    if (index < 0) return null;
    if (this.total != null && index >= this.total) return null;
    if (!this.items[index]) await this.loadPage(Math.floor(index / this.pageSize));
    return this.items[index] || null;
  }
  async at(index) { return this.ensure(index); }
  indexOfId(id) {
    const i = this.byId.get(id);
    if (i != null && this.items[i] && this.items[i].id === id) return i;
    return this.items.findIndex(it => it && it.id === id);
  }
  /** Working-set index of an image id, loading its page if needed; -1 when the image is not in this list
      (it never matched, or it started matching only after the list was opened — see refresh()). */
  async positionOf(id) {
    const i = this.indexOfId(id);
    if (i >= 0) return i;
    const v = this.version;
    try {
      const res = await get(`/api/datasets/${this.dsId}/images/position`, { ...this.queryParams, id });
      if (res.position == null || res.position < 0) return -1;
      const T = res.position;
      let f = -1;                                  // end of the contiguous loaded prefix
      while (this.items[f + 1]) f++;
      if (f < 0) { await this.ensure(T); return this.indexOfId(id); }
      if (this.total != null && f >= this.total - 1) return -1;
      const nb = await get(`/api/datasets/${this.dsId}/images/neighbor`, { ...this.queryParams, id: this.items[f].id, dir: 'next' });
      if (!nb.item) return -1;
      const P = nb.item.id === id ? T : (await get(`/api/datasets/${this.dsId}/images/position`, { ...this.queryParams, id: nb.item.id })).position;
      if (P == null || T < P) return -1;
      if (v === this.version) this.hint = { anchor: this.items[f].id, item: nb.item, position: P, v, at: Date.now(), known: new Map([[nb.item.id, P], [id, T]]) };
      const idx = f + 1 + (T - P);
      if (this.total != null && idx >= this.total) this.total = idx + 1;
      await this.ensure(idx);
      return this.indexOfId(id);
    } catch (e) { return -1; }
  }
  update(item) { const i = this.indexOfId(item.id); if (i >= 0) this.items[i] = { ...this.items[i], ...item }; }
  prefetch(index, n = 1) { for (let k = 1; k <= n; k++) { const j = index + k; if (this.total == null || j < this.total) { if (!this.items[j]) this.loadPage(Math.floor(j / this.pageSize)).catch(() => {}); } } }
}
