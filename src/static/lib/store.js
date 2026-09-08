export function createStore(initial = {}) {
  let state = { ...initial };
  const subs = new Set();
  return {
    get: () => state,
    set(patch) {
      const next = typeof patch === 'function' ? patch(state) : patch;
      state = { ...state, ...next };
      subs.forEach(fn => { try { fn(state); } catch (e) { console.error(e); } });
    },
    subscribe(fn) { subs.add(fn); return () => subs.delete(fn); },
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
