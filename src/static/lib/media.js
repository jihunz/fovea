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
  const settle = (ok) => {
    if (settled) return;
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
export function decodedImage(url) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.decoding = 'async';
    im.onload = () => {
      let settled = false;
      const done = () => { if (!settled) { settled = true; resolve(im); } };
      if (typeof im.decode === 'function') { im.decode().then(done, done); setTimeout(done, 800); } else done();
    };
    im.onerror = () => reject(new Error('Image could not be loaded'));
    im.src = url;
  });
}

/**
 * Decoded frames for sequential viewers (Inspect, Annotate, Compare).
 *
 * get(url) resolves with an already-decoded <img>, so the viewer keeps the current picture on screen and
 * replaces it in a single frame. prefetch() warms the neighbours so stepping is instant and never shows a
 * wait at all. Bounded LRU: one decoded 4K frame is ~33 MB.
 */
export function createFrameCache({ max = 5 } = {}) {
  const frames = new Map();   // url -> Promise<HTMLImageElement>
  function get(url) {
    const hit = frames.get(url);
    if (hit) { frames.delete(url); frames.set(url, hit); return hit; }
    const p = decodedImage(url);
    frames.set(url, p);
    p.catch(() => { if (frames.get(url) === p) frames.delete(url); });   // failures are retried next time
    while (frames.size > max) frames.delete(frames.keys().next().value);
    return p;
  }
  return {
    get,
    prefetch(urls) { for (const u of urls) if (u) get(u).catch(() => {}); },
    clear() { frames.clear(); },
  };
}


/**
 * Explicit viewport-driven image loading.
 *
 * `loading="lazy"` is a browser *heuristic*: it can defer indefinitely — a backgrounded or occluded
 * tab, or a grid whose layout it has not settled on — leaving in-viewport tiles permanently blank
 * with no request ever sent. A gallery that already virtualises its own list should not delegate
 * that decision. This observer requests each image when it comes within `margin` of the scroller,
 * which is deterministic and lets images arrive slightly *before* they are scrolled into view.
 *
 * Usage: create one per scroll container, call observe(img, holder) per tile, dispose() on teardown.
 */
export function createImageLoader({ root = null, margin = '800px' } = {}) {
  const pending = new Map(); // img -> holder
  const start = (img, holder) => {
    const src = img.dataset.src;
    if (!src || img.getAttribute('src') === src) return;
    img.removeAttribute('data-src');
    img.src = src;
    trackImage(img, holder);
  };
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      const holder = pending.get(e.target);
      if (holder) { start(holder.img, holder.pic); pending.delete(e.target); }
      io.unobserve(e.target);
    }
  }, { root, rootMargin: margin, threshold: 0 });

  // A hidden/occluded tab does not run IntersectionObserver at all, so nothing would be requested
  // while the user is away and the gallery would still be blank the moment they come back. Flush
  // whatever is on screen as soon as the document becomes visible again.
  const flushVisible = (slackPx = 0) => {
    if (document.hidden) return;
    const h = window.innerHeight || 0;
    for (const [target, holder] of [...pending]) {
      if (!target.isConnected) { pending.delete(target); io.unobserve(target); continue; }   // tile was discarded
      const r = target.getBoundingClientRect();
      if (r.bottom < -slackPx || r.top > h + slackPx) continue;
      start(holder.img, holder.pic);
      pending.delete(target);
      io.unobserve(target);
    }
  };
  const onVisible = () => flushVisible();
  document.addEventListener('visibilitychange', onVisible);

  // Safety net. IntersectionObserver is the fast path, but if it ever fails to deliver — a tab that
  // was occluded while the grid was built, a browser quirk — the gallery would sit permanently blank,
  // which is the exact failure this module exists to prevent. A throttled scroll check costs nothing
  // and makes "no image ever loads" impossible.
  let ticking = false;
  const onScroll = () => {
    if (ticking || !pending.size) return;
    ticking = true;
    requestAnimationFrame(() => { ticking = false; flushVisible(200); });
  };
  const scroller = root || window;
  scroller.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll, { passive: true });

  return {
    /** Defer `img` until `watch` nears the viewport. `img.dataset.src` holds the real URL. */
    observe(img, pic, watch) {
      pic.classList.add('is-idle');   // deferred: neutral, and nothing drawn on top of nothing
      const target = watch || pic;
      pending.set(target, { img, pic });
      io.observe(target);
    },
    /** Load right now regardless of position (e.g. the image the user just jumped to). */
    loadNow(img, pic) { start(img, pic); },
    /** Forget one tile (it is being replaced). */
    unobserve(target) { pending.delete(target); io.unobserve(target); },
    /** Forget every pending tile — the grid was rebuilt, none of them will ever be shown. */
    reset() { for (const target of pending.keys()) io.unobserve(target); pending.clear(); },
    flushVisible,
    dispose() {
      io.disconnect(); pending.clear();
      scroller.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onScroll);
      document.removeEventListener('visibilitychange', onVisible);
    },
  };
}
