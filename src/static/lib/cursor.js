// Paged, filtered image list shared by Explore, Inspect and Annotate.
import { get } from './api.js';

export class ImageCursor {
  constructor({ dsId, filters = {}, sort = 'id', order = 'asc', pageSize = 120, boxes = false }) {
    this.dsId = dsId; this.filters = { ...filters }; this.sort = sort; this.order = order; this.pageSize = pageSize; this.boxes = boxes;
    this.items = []; this.total = null; this.loading = new Map(); this.version = 0;
  }
  get params() { return { ...this.filters, sort: this.sort, order: this.order, boxes: this.boxes ? 1 : 0 }; }
  reset() { this.items = []; this.total = null; this.loading.clear(); this.version++; }
  get loadedCount() { return this.items.filter(Boolean).length; }
  async loadPage(page) {
    const key = page; const v = this.version;
    if (this.loading.has(key)) return this.loading.get(key);
    const p = get(`/api/datasets/${this.dsId}/images`, { ...this.params, offset: page * this.pageSize, limit: this.pageSize }).then(res => {
      if (v !== this.version) return res;
      this.total = res.total;
      res.items.forEach((it, i) => { this.items[page * this.pageSize + i] = it; });
      this.loading.delete(key);
      return res;
    }).catch(e => { this.loading.delete(key); throw e; });
    this.loading.set(key, p);
    return p;
  }
  async ensure(index) {
    if (index < 0) return null;
    if (this.total != null && index >= this.total) return null;
    if (!this.items[index]) await this.loadPage(Math.floor(index / this.pageSize));
    return this.items[index] || null;
  }
  async at(index) { return this.ensure(index); }
  indexOfId(id) { return this.items.findIndex(it => it && it.id === id); }
  async positionOf(id) {
    const i = this.indexOfId(id);
    if (i >= 0) return i;
    try {
      const res = await get(`/api/datasets/${this.dsId}/images/position`, { ...this.params, id });
      if (res.position != null && res.position >= 0) { await this.ensure(res.position); const j = this.indexOfId(id); if (j >= 0) return j; return res.position; }
    } catch (e) { /* fallthrough */ }
    return -1;
  }
  update(item) { const i = this.indexOfId(item.id); if (i >= 0) this.items[i] = { ...this.items[i], ...item }; }
  prefetch(index, n = 1) { for (let k = 1; k <= n; k++) { const j = index + k; if (this.total == null || j < this.total) { if (!this.items[j]) this.loadPage(Math.floor(j / this.pageSize)).catch(() => {}); } } }
}
