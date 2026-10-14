export function createStore(initial = {}) {
  let state = { ...initial };
  const subs = new Set();
  return {
    get: () => state,
    /** Shallow-merge a patch. Subscribers run only when some value actually changed (by identity), so a poll that
        re-sets an unchanged list does not rebuild the sidebar and drop keyboard focus. */
    set(patch) {
      const next = typeof patch === 'function' ? patch(state) : patch;
      if (!next) return;
      let changed = false;
      for (const k of Object.keys(next)) if (state[k] !== next[k]) { changed = true; break; }
      if (!changed) return;
      const prev = state;
      state = { ...state, ...next };
      subs.forEach(fn => { try { fn(state, prev); } catch (e) { console.error(e); } });
    },
    /** subscribe(fn) runs on every change; subscribe(fn, select) only when select(state) changes. */
    subscribe(fn, select) {
      if (!select) { subs.add(fn); return () => subs.delete(fn); }
      let last = select(state);
      const w = (s) => { const v = select(s); if (v !== last) { last = v; fn(s); } };
      subs.add(w); return () => subs.delete(w);
    },
  };
}

// Simple event bus
export function createBus() {
  const map = new Map();
  return {
    on(evt, fn) { if (!map.has(evt)) map.set(evt, new Set()); map.get(evt).add(fn); return () => map.get(evt)?.delete(fn); },
    emit(evt, payload) { map.get(evt)?.forEach(fn => { try { fn(payload); } catch (e) { console.error(e); } }); },
  };
}
