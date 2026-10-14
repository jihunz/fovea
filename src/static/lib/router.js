import { h } from './dom.js';

function compile(pattern) {
  const keys = [];
  const re = new RegExp('^' + pattern.replace(/\/:([a-zA-Z0-9_]+)(\*)?/g, (_, k, star) => { keys.push(k); return star ? '/(.+)' : '/([^/]+)'; }) + '/?$');
  return { re, keys };
}

/**
 * History router. Every dispatch gets a sequence number: an async route handler checks router.isCurrent(ctx)
 * after each await and stops if the user has navigated on meanwhile, so a slow fetch never renders the wrong
 * page over the current URL.
 *
 * ctx = { path, query: URLSearchParams, params, route, seq, prevPath, prevSearch }. prevSearch is the query
 * the page last showed (including changes a view made itself with replaceQuery / pushQuery), so a view can
 * tell a same-page query change (Back, a link with other filters) from its own URL sync.
 */
class Router {
  constructor() { this.routes = []; this.current = null; this.before = []; this.after = []; this._started = false; this.seq = 0; this.syncedPath = null; this.syncedSearch = ''; }
  add(pattern, handler) { const { re, keys } = compile(pattern); this.routes.push({ pattern, re, keys, handler }); return this; }
  match(path) {
    for (const r of this.routes) {
      const m = r.re.exec(path);
      if (m) { const params = {}; r.keys.forEach((k, i) => params[k] = decodeURIComponent(m[i + 1])); return { route: r, params }; }
    }
    return null;
  }
  navigate(to, { replace = false, state = null } = {}) {
    if (to === location.pathname + location.search && !replace) return this.dispatch();
    history[replace ? 'replaceState' : 'pushState'](state, '', to);
    return this.dispatch();
  }
  _queryUrl(query) {
    const url = new URL(location.href);
    url.search = '';
    for (const [k, v] of Object.entries(query)) if (v != null && v !== '' && v !== false) url.searchParams.set(k, v);
    return url;
  }
  /** Rewrite the query of the current entry (a view syncing its state into the URL); no dispatch. */
  replaceQuery(query) {
    const url = this._queryUrl(query);
    history.replaceState(history.state, '', url.pathname + url.search);
    this.syncedSearch = url.search;
    this.current = { ...this.current, query: url.searchParams };
  }
  /** Like replaceQuery, but as a new history entry, so Back returns to the previous state (e.g. a folder). */
  pushQuery(query) {
    const url = this._queryUrl(query);
    if (url.pathname + url.search === location.pathname + location.search) return;
    history.pushState(null, '', url.pathname + url.search);
    this.syncedSearch = url.search;
    this.current = { ...this.current, query: url.searchParams };
  }
  /** True while ctx is still the latest navigation. */
  isCurrent(ctx) { return !!ctx && ctx.seq === this.seq; }
  async dispatch() {
    const path = location.pathname;
    const query = new URLSearchParams(location.search);
    const m = this.match(path);
    const ctx = { path, query, params: m ? m.params : {}, route: m ? m.route : null, seq: ++this.seq,
                  prevPath: this.syncedPath, prevSearch: this.syncedSearch, search: location.search };
    for (const fn of this.before) {
      let r;
      try { r = await fn(ctx); } catch (e) { console.error(e); }
      if (r === false) return;
      if (ctx.seq !== this.seq) return;
    }
    this.current = ctx;
    this.syncedPath = path; this.syncedSearch = location.search;
    try {
      if (m) await m.route.handler(ctx);
      else if (this.notFound) this.notFound(ctx);
    } finally {
      if (ctx.seq === this.seq) for (const fn of this.after) { try { fn(ctx); } catch (e) { console.error(e); } }
    }
  }
  start() {
    if (this._started) return; this._started = true;
    window.addEventListener('popstate', () => this.dispatch());
    document.addEventListener('click', (e) => {
      if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      const a = e.target.closest('a[href]');
      if (!a || a.target === '_blank' || a.hasAttribute('download') || a.dataset.external != null) return;
      const href = a.getAttribute('href');
      if (!href || !href.startsWith('/') || href.startsWith('//') || href.startsWith('/api/')) return;
      e.preventDefault();
      this.navigate(href);
    });
    return this.dispatch();
  }
}

export const router = new Router();
export function link(href, props = {}, ...children) { return h('a', { href, ...props }, ...children); }
