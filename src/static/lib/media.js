// Image loading helpers shared by every surface that shows dataset pixels.
//
// Problems this solves:
//  1. A tile whose image has not painted yet is just a dark rectangle — and because the box overlay
//     is drawn immediately, it reads as "boxes floating on nothing", i.e. broken. The overlay stays
//     hidden until there is an image under it.
//  2. A thumbnail that genuinely fails (file deleted, drive unmounted, index built elsewhere) used to
//     stay silently black forever. Now it says so.
//  3. Visual comfort (docs/visual-comfort.md). People step through thousands of images, often
//     consecutive video frames. Every image change must be ONE luminance step: the current picture stays
//     until the next is decoded, then both swap in the same frame. No fade, no blank stage, no animated
//     placeholder — for near-identical frames an instant cut changes almost nothing on screen, while a
//     fade or a blank forces a full dark→bright cycle (and a pupil response) on every keypress.

const SIZES = [128, 256, 512, 1024]; // must match THUMB_SIZES in fovea/config.py

/** Pick the thumbnail size to request for a box `cssPx` wide, accounting for the display's pixel ratio. */
export function thumbPx(cssPx) {
  const want = Math.round(cssPx * (window.devicePixelRatio || 1));
  return SIZES.find((s) => want <= s) || SIZES[SIZES.length - 1];
}

/**
 * Track `img`'s load state on `holder` as is-loading → is-ready | is-error.
 * Safe for cached images (which never fire load) and for repeated calls.
 */
export function trackImage(img, holder) {
  let settled = false;
  // A holder can be re-tracked (new src, retry, a recycled tile): only the latest call may settle it, or a
  // stale load/error listener from the previous src would mark the new picture ready or broken.
  const gen = holder._trackGen = (holder._trackGen || 0) + 1;
  const settle = (ok) => {
    if (settled || holder._trackGen !== gen) return;
    settled = true;
    holder.classList.remove('is-loading');
    holder.classList.add(ok ? 'is-ready' : 'is-error');
  };
  holder.classList.remove('is-ready', 'is-error', 'is-idle');
  holder.classList.add('is-loading');
  const hasSrc = !!img.currentSrc || !!img.getAttribute('src');
  if (hasSrc && img.complete) {
    // Already decoded (memory/disk cache) — settle synchronously so there is no placeholder flash.
    settle(img.naturalWidth > 0);
  } else {
    // Reveal only once decoded. "load" can fire before the pixels are ready, and revealing then paints
    // an empty frame first — a second luminance step instead of one.
    img.addEventListener('load', () => {
      const done = () => settle(img.naturalWidth > 0);
      if (typeof img.decode === 'function') img.decode().then(done, done); else done();
    }, { once: true });
    img.addEventListener('error', () => settle(false), { once: true });
  }
  return img;
}

/** Load and decode `url`. Resolves with the <img> once its pixels are ready to paint in one frame.
    Browsers may postpone decode() while the tab is hidden; after 800 ms the loaded image is used anyway
    (it then decodes at first paint), so a step can never hang on it. */
export function decodedImage(url, { priority } = {}) {
  let im = null, cancelled = false, rejectFn = null;
  const p = new Promise((resolve, reject) => {
    rejectFn = reject;
    im = new Image();
    im.decoding = 'async';
    // The frame the user is waiting for goes ahead of prefetches and gallery thumbnails on the six HTTP/1.1
    // connections a browser opens per host.
    if (priority) { try { im.fetchPriority = priority; } catch (e) { /* older browsers */ } im.setAttribute('fetchpriority', priority); }
    im.onload = () => {
      let settled = false;
      const done = () => { if (!settled && !cancelled) { settled = true; resolve(im); } };
      if (typeof im.decode === 'function') { im.decode().then(done, done); setTimeout(done, 800); } else done();
    };
    im.onerror = () => { const e = new Error('Image could not be loaded'); e.code = 'load'; reject(e); };
    im.src = url;
  });
  /** Abort the download (dropping the src cancels the request) and reject with code 'cancelled'. */
  p.cancel = () => {
    if (cancelled || !im || im.complete) return;
    cancelled = true;
    im.onload = im.onerror = null;
    im.removeAttribute('src');
    const e = new Error('cancelled'); e.code = 'cancelled'; rejectFn(e);
  };
  p.image = () => im;
  return p;
}

/**
 * Decoded frames for sequential viewers (Inspect, Annotate, Compare).
 *
 * get(url) resolves with an already-decoded <img>, so the viewer keeps the current picture on screen and
 * replaces it in a single frame. prefetch() warms the neighbours so stepping is instant and never shows a
 * wait at all. Bounded LRU: one decoded 4K frame is ~33 MB.
 *  · An entry pushed out of the LRU before it finished downloading is cancelled, so a held arrow key over large
 *    frames cannot pile up full-resolution downloads nobody will look at.
 *  · A URL that failed (404: file moved or deleted) is remembered for `failTtl` ms: stepping back and forth
 *    across it neither re-requests it nor prefetches it again.
 */
export function createFrameCache({ max = 5, failTtl = 30000 } = {}) {
  const frames = new Map();   // url -> { p: Promise<HTMLImageElement>, done: bool }
  const failed = new Map();   // url -> time of the failure
  const recentlyFailed = (url) => { const t = failed.get(url); if (t == null) return false; if (Date.now() - t < failTtl) return true; failed.delete(url); return false; };
  function get(url, { priority } = {}) {
    const hit = frames.get(url);
    if (hit) {
      frames.delete(url); frames.set(url, hit);
      if (priority === 'high' && !hit.done && hit.p.image) { const im = hit.p.image(); try { im.fetchPriority = 'high'; } catch (e) { /* ignore */ } }
      return hit.p;
    }
    if (recentlyFailed(url)) { const e = new Error('Image could not be loaded'); e.code = 'load'; return Promise.reject(e); }
    const p = decodedImage(url, { priority });
    const entry = { p, done: false };
    frames.set(url, entry);
    p.then(() => { entry.done = true; }, (e) => {
      if (frames.get(url) === entry) frames.delete(url);
      if (e && e.code === 'load') failed.set(url, Date.now());          // cancelled loads are not failures
    });
    while (frames.size > max) {
      const [oldUrl, old] = frames.entries().next().value;
      frames.delete(oldUrl);
      if (!old.done && old.p.cancel) old.p.cancel();
    }
    return p;
  }
  return {
    get,
    prefetch(urls) { for (const u of urls) if (u && !recentlyFailed(u)) get(u, { priority: 'low' }).catch(() => {}); },
    /** True when `url` is decoded and can be painted in this frame. */
    ready: (url) => { const e = frames.get(url); return !!(e && e.done); },
    /** Stop downloading `url` if it is still in flight (a navigation that was superseded). */
    cancel(url) { const e = frames.get(url); if (e && !e.done) { frames.delete(url); if (e.p.cancel) e.p.cancel(); } },
    failedRecently: recentlyFailed,
    clear() { for (const e of frames.values()) if (!e.done && e.p.cancel) e.p.cancel(); frames.clear(); },
  };
}


/**
 * Viewport-driven thumbnail loading for galleries.
 *
 *  · Deterministic. An image is requested when its tile comes within `margin` of the scroller
 *    (IntersectionObserver, plus a scroll check as a safety net) — never through `loading="lazy"`, a browser
 *    heuristic that can defer in-viewport tiles indefinitely.
 *  · Bounded. At most `maxInflight` requests at once, nearest to the viewport first. Browsers open only six
 *    HTTP/1.1 connections per host: a fast scroll used to queue 300–1,000 thumbnails, and every page fetch
 *    and the Inspect/Annotate image then waited seconds behind them.
 *  · Cancellable. A tile that leaves the margin (or is unobserved) before its image arrives gives the slot
 *    back — dropping the src aborts the request — and is queued again if it comes back.
 *  · Honest about failures. One quiet retry, then the cause is classified on `pic.dataset.err`
 *    ('missing' 404 · 'unsupported' 415 · 'decode' · 'network') instead of calling every hiccup "file missing".
 *
 * observe(img, pic, watch): `img.dataset.src` holds the URL; `pic` goes is-idle → is-loading → is-ready |
 * is-error and reveals the picture only once decoded (docs/visual-comfort.md). retarget() swaps the URL of a
 * tile that is already showing a picture without blanking it (the browser keeps the old pixels until the new
 * ones arrive). pause()/resume() free every connection for a full-size viewer opened over the gallery.
 */
export function createImageLoader({ root = null, margin = '800px', maxInflight = 6 } = {}) {
  const entries = new Map();   // watch target -> entry
  const loaded = new Set();    // URLs that loaded fine this session: re-shown at once, never queued
  let inflight = 0, paused = false, disposed = false, pumpQueued = false, order = 0;
  const slack = parseFloat(String(margin)) || 0;

  const io = new IntersectionObserver((list) => {
    for (const e of list) {
      const en = entries.get(e.target);
      if (!en) continue;
      en.near = e.isIntersecting;
      if (!en.near && en.state === 'loading' && !en.forced) abort(en);   // scrolled past before it arrived
    }
    schedulePump();
  }, { root, rootMargin: margin, threshold: 0 });

  function schedulePump() { if (!pumpQueued && !disposed) { pumpQueued = true; queueMicrotask(pump); } }
  function setState(pic, cls) { pic.classList.remove('is-idle', 'is-loading', 'is-ready', 'is-error'); pic.classList.add(cls); }
  function release(en) { if (en.counted) { en.counted = false; inflight--; } }

  function abort(en) {
    if (en.state !== 'loading') return;
    en.gen++; release(en); en.state = 'idle';
    if (en.shown) { if (en.img.getAttribute('src') !== en.shownUrl) en.img.src = en.shownUrl; }   // keep the old pixels
    else { en.img.removeAttribute('src'); setState(en.pic, 'is-idle'); }
  }

  function reveal(en, gen, url) {
    if (en.gen !== gen) return;
    if (!(en.img.naturalWidth > 0)) { fail(en, gen, url); return; }
    en.shown = true; en.shownUrl = url;
    delete en.pic.dataset.err;
    setState(en.pic, 'is-ready');
  }

  function fail(en, gen, url) {
    if (en.gen !== gen) return;
    en.state = 'failed';
    setState(en.pic, 'is-error');
    en.pic.dataset.err = 'checking';
    // Ask the server what went wrong; an <img> error event does not say.
    fetch(url, { cache: 'no-store' }).then((r) => r.status, () => 0).then((status) => {
      if (en.gen !== gen) return;
      en.pic.dataset.err = status === 404 || status === 410 ? 'missing' : status === 415 ? 'unsupported' : status >= 200 && status < 300 ? 'decode' : 'network';
    });
  }

  function start(en, forced = false) {
    const { img, pic } = en;
    const url = en.url, gen = ++en.gen;
    en.state = 'loading'; en.forced = forced;
    if (!en.counted) { en.counted = true; inflight++; }
    if (!en.shown) setState(pic, 'is-loading');
    img.setAttribute('fetchpriority', 'low');
    const onLoad = () => {
      if (en.gen !== gen) return;
      release(en); en.state = 'done'; loaded.add(url);
      // Reveal only once decoded: revealing at "load" can paint an empty frame first (two luminance steps).
      if (typeof img.decode === 'function') img.decode().then(() => reveal(en, gen, url), () => reveal(en, gen, url));
      else reveal(en, gen, url);
      schedulePump();
    };
    const onError = () => {
      if (en.gen !== gen) return;
      release(en);
      if (en.retries < 1) {                     // one quiet retry: most failures are transient
        en.retries++; en.state = 'idle'; en.retryAt = performance.now() + 700;
        setTimeout(() => { if (entries.get(en.target) === en) schedulePump(); }, 720);
      } else fail(en, gen, url);
      schedulePump();
    };
    img.addEventListener('load', onLoad, { once: true });
    img.addEventListener('error', onError, { once: true });
    img.removeAttribute('data-src');
    img.src = url;
    // Already decoded in memory (a tile scrolled back into view): show it in this frame, no placeholder flash.
    if (img.complete && img.naturalWidth > 0) { en.gen++; release(en); en.state = 'done'; loaded.add(url); reveal(en, en.gen, url); }
  }

  function viewport() {
    if (root) { const r = root.getBoundingClientRect(); return { top: r.top, bottom: r.bottom }; }
    return { top: 0, bottom: window.innerHeight || 0 };
  }

  function pump() {
    pumpQueued = false;
    if (paused || disposed || inflight >= maxInflight) return;
    const now = performance.now();
    const cands = [];
    for (const en of entries.values()) if (en.state === 'idle' && en.near && !(en.retryAt > now)) cands.push(en);
    if (!cands.length) return;
    if (cands.length > maxInflight - inflight) {
      // Nearest to the visible area first, so what the user is looking at arrives before the margin does.
      const vp = viewport();
      for (const en of cands) {
        const r = en.target.getBoundingClientRect();
        en.dist = r.bottom < vp.top ? vp.top - r.bottom : r.top > vp.bottom ? r.top - vp.bottom : 0;
      }
      cands.sort((a, b) => a.dist - b.dist || a.order - b.order);
    }
    for (const en of cands) { if (inflight >= maxInflight) break; start(en); }
  }

  // Safety net. A hidden/occluded tab runs no IntersectionObserver callbacks, and a quirk could leave a tile
  // permanently blank — the exact failure this module exists to prevent. A throttled geometric check costs
  // nothing and makes "no image ever loads" impossible.
  const flushVisible = (extra = slack) => {
    if (document.hidden || disposed) return;
    const vp = viewport();
    for (const [target, en] of [...entries]) {
      if (!target.isConnected) { entries.delete(target); io.unobserve(target); abort(en); continue; }
      if (en.state !== 'idle') continue;
      const r = target.getBoundingClientRect();
      en.near = !(r.bottom < vp.top - extra || r.top > vp.bottom + extra);
    }
    schedulePump();
  };
  const onVisible = () => flushVisible();
  document.addEventListener('visibilitychange', onVisible);
  let ticking = false;
  const onScroll = () => {
    if (ticking || !entries.size) return;
    ticking = true;
    requestAnimationFrame(() => { ticking = false; flushVisible(); });
  };
  const scroller = root || window;
  scroller.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll, { passive: true });

  function entryFor(img, pic, target) {
    let en = entries.get(target);
    if (!en) {
      en = { target, img, pic, url: '', state: 'idle', near: false, gen: 0, retries: 0, counted: false, shown: false, shownUrl: '', order: order++ };
      entries.set(target, en);
      io.observe(target);
    }
    en.img = img; en.pic = pic;
    return en;
  }

  return {
    /** Load `img` (URL in `img.dataset.src`) when `watch` (default `pic`) nears the viewport. */
    observe(img, pic, watch) {
      const target = watch || pic;
      const en = entryFor(img, pic, target);
      en.url = img.dataset.src || img.getAttribute('src') || en.url;
      en.retries = 0;
      if (en.state === 'loading') abort(en);
      en.state = 'idle';
      if (!en.shown) setState(pic, 'is-idle');   // neutral, and no boxes drawn on top of nothing
      if (loaded.has(en.url)) start(en, true);    // cached: show now — waiting a frame for the observer would blink
      else schedulePump();
    },
    /** Point an observed tile at another URL (e.g. a different thumbnail size) without blanking it. */
    retarget(target, url) {
      const en = entries.get(target);
      if (!en || !url || en.url === url) return;
      if (en.state === 'loading') abort(en);
      en.url = url; en.retries = 0; en.state = 'idle';
      if (loaded.has(url)) start(en, true); else schedulePump();
    },
    /** Load right now regardless of position and of the in-flight cap (the image the user jumped to). */
    loadNow(img, pic) {
      let en = [...entries.values()].find(e => e.img === img);
      if (!en) en = entryFor(img, pic, pic);
      if (img.dataset.src) en.url = img.dataset.src;
      if (en.state !== 'loading' && en.state !== 'done') start(en, true);
    },
    /** Forget one tile (unmounted or replaced). An in-flight request is cancelled. */
    unobserve(target) {
      const en = entries.get(target);
      if (!en) return;
      abort(en); entries.delete(target); io.unobserve(target);
      schedulePump();
    },
    /** Forget every tile — the grid was rebuilt; none of the pending pictures will be shown. */
    reset() {
      for (const [target, en] of entries) { abort(en); io.unobserve(target); }
      entries.clear();
    },
    /** Stop loading (cancel what is in flight) while a full-size viewer needs the connections. */
    pause() { paused = true; for (const en of entries.values()) if (!en.forced) abort(en); },
    resume() { if (!paused) return; paused = false; flushVisible(); },
    isLoaded: (url) => loaded.has(url),
    /** Record a URL the caller has already loaded and decoded (e.g. a grid preloading its first screen). */
    prime: (url) => { loaded.add(url); },
    stats: () => ({ inflight, entries: entries.size, queued: [...entries.values()].filter(e => e.state === 'idle' && e.near).length }),
    flushVisible,
    dispose() {
      disposed = true;
      for (const en of entries.values()) abort(en);
      io.disconnect(); entries.clear();
      scroller.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onScroll);
      document.removeEventListener('visibilitychange', onVisible);
    },
  };
}
