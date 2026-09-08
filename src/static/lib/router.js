import { h } from './dom.js';

function compile(pattern) {
  const keys = [];
  const re = new RegExp('^' + pattern.replace(/\/:([a-zA-Z0-9_]+)(\*)?/g, (_, k, star) => { keys.push(k); return star ? '/(.+)' : '/([^/]+)'; }) + '/?$');
  return { re, keys };
}

class Router {
  constructor() { this.routes = []; this.current = null; this.before = []; this._started = false; }
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
  replaceQuery(query) {
    const url = new URL(location.href);
    url.search = '';
    for (const [k, v] of Object.entries(query)) if (v != null && v !== '' && v !== false) url.searchParams.set(k, v);
    history.replaceState(null, '', url.pathname + url.search);
    this.current = { ...this.current, query: url.searchParams };
  }
  async dispatch() {
    const path = location.pathname;
    const query = new URLSearchParams(location.search);
    const m = this.match(path);
    const ctx = { path, query, params: m ? m.params : {}, route: m ? m.route : null };
    for (const fn of this.before) if (await fn(ctx) === false) return;
    this.current = ctx;
    if (m) await m.route.handler(ctx);
    else if (this.notFound) this.notFound(ctx);
  }
  start() {
    if (this._started) return; this._started = true;
    window.addEventListener('popstate', () => this.dispatch());
    document.addEventListener('click', (e) => {
      if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      const a = e.target.closest('a[href]');
      if (!a || a.target === '_blank' || a.hasAttribute('download') || a.dataset.external != null) return;
      const href = a.getAttribute('href');
      if (!href || !href.startsWith('/') || href.startsWith('//')) return;
      e.preventDefault();
      this.navigate(href);
    });
    return this.dispatch();
  }
}

export const router = new Router();
export function link(href, props = {}, ...children) { return h('a', { href, ...props }, ...children); }
