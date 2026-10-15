// Settings: appearance and visual comfort (this device, shown at once), AI providers and keys, model weights,
// storage, server, plugins (each loads on its own, with its own error).
/** replaceChildren that skips null / false and flattens arrays (DOM replaceChildren would print "null"). */
const put = (el, ...c) => { el.replaceChildren(...c.flat(Infinity).filter(x => x != null && x !== false && x !== true)); return el; };
export function install(fovea) {}

const PRIVACY = {
  anthropic: 'Each request sends the image (resized) and your class names and card text to Anthropic. Review your organization’s data-retention settings in the Anthropic console.',
  gemini: 'Each request sends the image and your class names and card text to Google. Data sent with a free-tier key may be used to improve Google’s products — use a paid key for private data.',
  openai: 'Each request sends the image and your class names and card text to OpenAI (with store: false, so OpenAI does not keep the response for later retrieval).',
  mock: 'Test provider: nothing is sent anywhere.',
};
const KEY_PREFIX = { anthropic: ['sk-ant-', 'Anthropic keys start with sk-ant-'], openai: ['sk-', 'OpenAI keys start with sk-'], gemini: ['AIza', 'Gemini API keys start with AIza'] };
const COORD_HELP = {
  auto: 'Auto: pick from the model name (Qwen 2.5 → pixels, Qwen 3 → 0–1000, otherwise 0–1000)',
  qwen3: 'Qwen 3 / 3.x: boxes as x1,y1,x2,y2 on a 0–1000 grid',
  qwen25: 'Qwen 2.5: boxes in pixels of the image sent (Fovea resizes to multiples of 28)',
  gemini: 'Gemini style: [y1,x1,y2,x2] on a 0–1000 grid',
  norm: 'Normalised: x1,y1,x2,y2 between 0 and 1',
};
/** The provider setting `batch_mode` only matters where a provider reads it: OpenAI then sends its synchronous
    requests on the Flex tier (batch prices, slower answers). Provider batch APIs (about half price, results within
    24 h) are chosen per run instead — Economy in the batch dialog. */
export const FLEX_PROVIDERS = new Set(['openai']);
export const hasFlexTier = (p) => !!p && (FLEX_PROVIDERS.has(p.id) || !!(p.capabilities && p.capabilities.flex));
/** What a browser on another machine may not change (server.request_is_local false, FOVEA_ALLOW_REMOTE_AI unset),
    mirroring api/ai.py: keys (any provider), the settings of cloud and key-bearing providers (the OpenAI-compatible
    server too: its URL decides where images go), and calls to cloud models (Test, Refresh list). Those controls are
    drawn disabled instead of failing with a 403 on click. Before /api/meta answers nothing is locked. */
export function remoteLocks(p, server, remoteAllowed) {
  const remote = !!(server && server.request_is_local === false && !remoteAllowed);
  const cloud = !!p && p.kind === 'cloud';
  return { remote, key: remote && !!(p && p.key), settings: remote && !!p && (cloud || !!p.needs_key || !!p.key || p.id === 'openai_compat'), call: remote && cloud };
}
/** Settings → Usage. Runs of a cloud model with no price on file are booked at $0 (GET /api/ai/usage lists them under
    `unpriced`): they read "cost unknown" and stay out of the dollar figures, never "$0". Runs are counted once each
    (by_model), not once per provider a two-stage run used. */
export function usageSummary(usage, fmt) {
  const u = usage || {}, un = u.unpriced || {}, unBy = un.by_provider || {};
  const sum = (o) => Object.values(o || {}).reduce((n, v) => n + (Number(v.runs) || 0), 0);
  const runs = sum(u.by_model) || sum(u.by_provider);
  const rows = Object.entries(u.by_provider || {}).map(([id, v]) => {
    const x = unBy[id] || {}, xr = Number(x.runs) || 0, xi = Number(x.images) || 0, imgs = Number(v.images) || 0;
    const text = !xr ? `${fmt.usd(v.usd || 0)} · ${fmt.plural(imgs, 'image')}`
      : xr >= (Number(v.runs) || 0) && !(v.usd > 0) ? `cost unknown · ${fmt.plural(imgs, 'image')}`
      : xi ? `${fmt.usd(v.usd || 0)} · ${fmt.plural(Math.max(0, imgs - xi), 'image')} + ${fmt.plural(xi, 'image')} cost unknown`
      : `${fmt.usd(v.usd || 0)} · ${fmt.plural(imgs, 'image')} + ${fmt.plural(xr, 'run')} cost unknown`;
    return { id, usd: Number(v.usd) || 0, text, unpriced: xr };
  }).sort((a, b) => b.usd - a.usd || b.unpriced - a.unpriced);
  const xr = Number(un.runs) || 0, all = xr && xr >= runs && !(u.total_usd > 0);
  const note = !xr ? '' : `${all ? (xr === 1 ? 'This run' : `All ${fmt.num(xr)} runs`) : fmt.plural(xr, 'run')}${un.images ? ` (${fmt.plural(un.images, 'image')})` : ''} used a model with no price on file, so ${xr === 1 ? 'its' : 'their'} cost is unknown${all ? '' : ' and not in this total'}.`;
  return { total: all ? 'Cost unknown' : fmt.usd(u.total_usd || 0), runs: `across ${fmt.plural(runs, 'run')}`, note, rows };
}
/** The Spend cap row's usage line. Besides what was billed in the last 30 days it says what runs still going have
    committed but not billed yet (GET /api/ai/usage caps[id].reserved_usd: Economy runs waiting at the provider, the
    remaining images of Now runs) — new runs only get the cap minus both, so without it a refusal looks wrong. */
export function capUsageText(capU, spentFallback, unpricedRuns, fmt) {
  const spent = capU && capU.spent_30d != null ? Number(capU.spent_30d) || 0 : Number(spentFallback) || 0;
  const held = Number(capU && capU.reserved_usd) || 0;
  return `spent ${fmt.usd(spent)} in the last 30 days${held > 0 ? ` · reserved by runs still going ${fmt.usd(held)}` : ''}`
    + (unpricedRuns ? ` · ${fmt.plural(unpricedRuns, 'run')} with no price on file not counted` : '');
}
/** A provider head's warning when its 30-day cap leaves no room for a new run (the server refuses it then,
    runner.cap_remaining): reached by what was billed, or taken by what runs still going hold. null = room left. */
export function capStatus(capU, fmt) {
  if (!capU || capU.cap_usd == null) return null;
  const cap = Number(capU.cap_usd), spent = Number(capU.spent_30d) || 0, held = Number(capU.reserved_usd) || 0;
  if (cap - spent <= 0) return `Spend cap reached (${fmt.usd(spent)} of ${fmt.usd(cap)})`;
  if (held > 0 && cap - spent - held <= 0) return `Spend cap held by runs still going (${fmt.usd(held)} of ${fmt.usd(cap)})`;
  return null;
}
const GROUPS = [['local', 'On this machine'], ['cloud', 'Cloud'], ['test', 'Testing']];
const groupOf = (p) => p.id === 'mock' ? 'test' : p.id === 'openai_compat' ? 'local' : p.kind === 'cloud' ? 'cloud' : 'local';

export async function render(ctx, fovea) {
  const { h, icon, ui, fmt, cls } = fovea;
  const api = fovea.api;
  const page = h('div', { class: 'page' }); const inner = h('div', { class: 'page-inner narrow' }); page.appendChild(inner);
  document.title = 'Settings · Fovea';
  let alive = true;
  const offs = [];
  fovea.setView('settings', page, () => { alive = false; offs.forEach(f => f()); });
  const row = (label, ...value) => h('div', { class: 'set-row' }, h('span', { class: 'lbl' }, label), h('div', { class: 'val' }, ...value));
  const cardEl = (title, sub, body, id) => h('section', { class: 'card mb-16', id: id || null, 'aria-label': title }, h('div', { class: 'card-header' }, h('h3', title), sub ? h('span', { class: 'sub' }, sub) : null), h('div', { class: 'card-body settings-list' }, body));
  const waiting = () => h('div', { class: 'wait-row' }, ui.spinner(), h('span', 'Loading…'));
  const failed = (e, retry) => h('div', { class: 'callout danger small' }, icon('alertCircle', 14), h('div', { class: 'grow' }, fovea.plainError(e)), h('button', { class: 'btn btn-sm', onClick: retry }, 'Try again'));

  inner.appendChild(h('div', { class: 'page-head' }, h('div', { class: 'grow' }, h('h1', { class: 'page-title' }, 'Settings'), h('div', { class: 'page-sub' }, 'Appearance and comfort for this device; AI providers, models and storage for this server.'))));

  // ---- appearance (local, immediate)
  const themeSeg = ui.seg([{ value: 'system', label: 'System', icon: 'monitor' }, { value: 'light', label: 'Light', icon: 'sun' }, { value: 'dark', label: 'Dark', icon: 'moon' }], fovea.theme.get(), v => fovea.theme.set(v), { label: 'Theme' });
  offs.push(fovea.state.subscribe((s) => themeSeg.setValue(s.theme), (s) => s.theme));   // the sidebar / palette can change it too
  inner.appendChild(cardEl('Appearance', 'this device', [
    row('Theme', themeSeg, h('span', { class: 'hint' }, 'Neither theme is less tiring for everyone — people differ, and often not in the way they expect. Try both for a session.')),
  ]));

  // ---- visual comfort (per device: depends on the room and the display)
  const c = fovea.comfort.get();
  const dimLabel = h('output', { class: 'mono small', style: 'width:40px;text-align:right' }, `${Math.round(c.dim * 100)}%`);
  const dim = h('input', { type: 'range', class: 'slider', min: 0, max: 0.6, step: 0.05, value: c.dim, style: 'width:200px', 'aria-label': 'Screen dimming', 'aria-valuetext': `${Math.round(c.dim * 100)} percent`,
    onInput: (e) => { const v = Number(e.target.value); dimLabel.textContent = `${Math.round(v * 100)}%`; e.target.setAttribute('aria-valuetext', `${Math.round(v * 100)} percent`); fovea.comfort.set({ dim: v }); } });
  const palPreview = h('div', { class: 'row gap-4', 'aria-hidden': 'true' });
  const drawPal = () => put(palPreview, ...Array.from({ length: 10 }, (_, i) => h('span', { class: 'swatch', style: `width:16px;height:16px;background:${fovea.colors.classColor(i)}`, title: `class ${i}` })));
  drawPal();
  inner.appendChild(cardEl('Visual comfort', 'this device', [
    row('Screen dimming', h('div', { class: 'row' }, dim, dimLabel),
      h('span', { class: 'hint' }, 'Darkens everything Fovea shows. Match the screen to the room — a bright screen in a dim room is the most tiring combination. Lowering brightness helps; colour-temperature (“blue light”) filters were not shown to.')),
    row('Image surround', ui.seg([{ value: 'black', label: 'Black' }, { value: 'dark', label: 'Dark gray' }, { value: 'gray', label: 'Gray' }], c.stage, v => fovea.comfort.set({ stage: v }), { label: 'Image surround' }),
      h('span', { class: 'hint' }, 'Background around images in Inspect, Annotate and Compare. A lighter surround shrinks the brightness jump between an image and its edges.')),
    row('Class colours', h('div', { class: 'row wrap' }, ui.seg([{ value: 'default', label: 'Standard' }, { value: 'cvd', label: 'Colour-blind safe' }], c.palette || 'default', v => { fovea.comfort.set({ palette: v }); drawPal(); fovea.bus.emit('colors:changed', { palette: v }); }, { label: 'Class colours' }), palPreview),
      h('span', { class: 'hint' }, 'Colour-blind safe uses the Okabe–Ito order, which stays distinct with red–green and blue–yellow colour-vision deficiency. Open pages redraw with the new colours when they next draw boxes.')),
    row('Rest reminder', ui.seg([{ value: 0, label: 'Off' }, { value: 30, label: '30 min' }, { value: 45, label: '45 min' }, { value: 60, label: '60 min' }], c.rest, v => fovea.comfort.set({ rest: v }), { label: 'Rest reminder' }),
      h('span', { class: 'hint' }, 'A quiet note after continuous work; five idle minutes count as a break. Suggests a real break — a 20-second glance was not shown to help.')),
    row('Image changes', h('span', { class: 'small muted' }, 'Always instant: the current image stays until the next one is ready, with no fade, blank frame or animated placeholder.')),
  ]));

  // ---- AI providers
  const aiBody = h('div', null, waiting());
  inner.appendChild(cardEl('AI providers', 'this server', aiBody, 'ai-providers'));
  const weightsBody = h('div', null, waiting());
  inner.appendChild(cardEl('Model weights', 'local detectors and segmenters', weightsBody, 'weights'));
  const storageBody = h('div', null, waiting());
  inner.appendChild(cardEl('Storage & server', null, storageBody));
  const pluginsBody = h('div', null, waiting());
  inner.appendChild(cardEl('Plugins', null, pluginsBody));

  const open = new Set(JSON.parse(sessionStorage.getItem('fovea.settings.open') || '[]'));
  const remember = () => { try { sessionStorage.setItem('fovea.settings.open', JSON.stringify([...open])); } catch (e) { /* private mode */ } };
  let P = null, usage = null, meta = null;
  const provNodes = new Map();

  async function loadProviders() {
    try { [P, usage] = await Promise.all([api.get('/api/ai/providers'), api.get('/api/ai/usage', { days: 30 }).catch(() => null)]); }
    catch (e) { if (alive) put(aiBody, failed(e, () => { put(aiBody, waiting()); loadProviders(); })); return; }
    if (!alive) return;
    drawProviders();
  }
  const remoteSlot = h('div');
  /** Whether the provider blocks on screen were drawn locked (remoteLocks) — null before the first draw. */
  let drawnLocked = null;
  const lockOf = (p) => remoteLocks(p, meta && meta.server, P && P.remote_ai_allowed);
  const remoteNow = () => lockOf(null).remote;
  /** Remote-access note: depends on /api/meta, which may arrive after the provider list — updated on its own so
      a late answer never rebuilds (and resets) the provider blocks someone is typing in. Only when that answer
      says this browser is on another machine are the blocks drawn again, locked (nothing typed there could be saved). */
  function drawRemote() {
    const server = meta && meta.server;
    if (!P) return;
    const remoteNote = server && !server.request_is_local && !P.remote_ai_allowed
      ? h('div', { class: 'callout warn small mb-12' }, icon('shield', 14), h('span', 'You are using Fovea from another machine. API keys, cloud providers and their settings can only be changed or used from the computer Fovea runs on, so they are read-only here (start Fovea with FOVEA_ALLOW_REMOTE_AI=1 to allow other machines).'))
      : server && !server.loopback_only ? h('div', { class: 'callout small mb-12' }, icon('shield', 14), h('span', `This server listens on ${server.bind_host}. ${P.remote_ai_allowed ? 'Other machines may use the cloud providers and the saved keys (FOVEA_ALLOW_REMOTE_AI is set).' : 'Cloud providers and keys are refused for requests from other machines (set FOVEA_ALLOW_REMOTE_AI=1 to allow).'}`)) : null;
    put(remoteSlot, remoteNote);
    if (drawnLocked !== null && remoteNow() !== drawnLocked) drawProviders();
  }
  function drawProviders() {
    drawnLocked = remoteNow();
    const dev = P.device || {};
    const s = P.settings || {};
    const devSel = h('select', { class: 'select auto', 'aria-label': 'Device for local models', onChange: (e) => saveSettings({ device: e.target.value }, devState) },
      (dev.options || ['auto', 'cpu', 'mps', 'cuda']).map(o => h('option', { value: o, selected: o === (s.device || 'auto') }, { auto: 'Automatic (CUDA › Apple GPU › CPU)', cpu: 'CPU', mps: 'Apple GPU (MPS)', cuda: 'NVIDIA GPU (CUDA)' }[o] || o)));
    const keepSel = h('select', { class: 'select auto', 'aria-label': 'Models kept in memory', onChange: (e) => saveSettings({ keep_models: Number(e.target.value) }, devState) },
      [1, 2, 3, 4].map(n => h('option', { value: n, selected: n === (s.keep_models || 2) }, `${n} model${n > 1 ? 's' : ''} in memory`)));
    const devState = h('span', { class: 'small muted' }, dev.resolved ? `Using ${dev.resolved}` : dev.reason || '');
    const rolesState = h('span', { class: 'small muted' });
    const rolesSel = h('select', { class: 'select auto', 'aria-label': 'How dataset folders are found', onChange: (e) => saveSettings({ folder_roles: e.target.value }, rolesState) },
      [['ai', 'AI first, then folder names'], ['names', 'Folder names only']].map(([v, t]) => h('option', { value: v, selected: v === (s.folder_roles || 'ai') }, t)));
    const groups = GROUPS.map(([g, label]) => {
      const list = P.providers.filter(p => groupOf(p) === g);
      if (!list.length) return null;
      return h('div', { class: 'mb-12' }, h('div', { class: 'issue-group' }, label), list.map(p => providerBlock(p)));
    });
    const totals = usage ? usageBlock() : null;
    drawRemote();
    put(aiBody,
      remoteSlot,
      row('Local inference', h('div', { class: 'row wrap' }, devSel, keepSel, devState), h('span', { class: 'hint' }, 'Where YOLO, YOLOE and SAM run. Automatic picks the fastest device available; more models in memory switch faster but use more RAM.')),
      row('Find dataset folders', h('div', { class: 'row wrap' }, rolesSel, rolesState), h('span', { class: 'hint' }, 'When a dataset’s folders are not called images/ and labels/ (imgs/ + annotations/, train_images/ …), the first AI provider that can write text reads the folder names and file counts — never images or labels — and says which hold the images and labels. Without one, or with “Folder names only”, common names are matched.')),
      h('div', { class: 'mt-12' }, ...groups),
      totals,
      h('div', { class: 'xs faint mt-8' }, `Prices as of ${P.prices_as_of || 'unknown'} · batch, audit and benchmark runs show an estimate first; a reached spend cap blocks every cloud call · keys are stored on this server only and are never sent back to the browser.`));
  }
  function usageBlock() {
    const sm = usageSummary(usage, fmt);
    const max = Math.max(0.000001, ...sm.rows.map(r => r.usd));
    const name = (id) => (P.providers.find(p => p.id === id) || {}).name || id;
    return row('Usage, last 30 days', h('div', { class: 'row' }, h('span', { class: 'strong' }, sm.total), h('span', { class: 'small muted' }, sm.runs)),
      sm.rows.length ? h('div', { class: 'hbars usage-bars mt-4' }, sm.rows.map(r => h('div', { class: 'hbar' }, h('div', { class: 'name' }, h('span', { class: 'nm' }, name(r.id))),
        h('div', { class: 'track' }, h('i', { style: `width:${100 * r.usd / max}%;background:var(--accent)` })), h('div', { class: 'val' }, r.text)))) : h('span', { class: 'hint' }, 'No AI runs yet.'),
      sm.note ? h('span', { class: 'hint warn-text' }, sm.note) : null,
      h('span', { class: 'hint' }, 'Per run and per dataset in each dataset’s AI tab → Runs & cost.'));
  }
  async function saveSettings(patch, stateEl) {
    try { const r = await api.put('/api/ai/settings', patch); P.settings = r.settings; P.device = r.device; stateEl.textContent = r.device && r.device.resolved ? `Using ${r.device.resolved} · saved` : 'Saved'; stateEl.className = 'small ok-text'; }
    catch (e) { stateEl.textContent = fovea.plainError(e); stateEl.className = 'small danger-text'; }
  }

  function statusOf(p) {
    if (p.config && p.config.enabled === false) return ['off', 'Turned off'];
    if (p.status && !p.status.available) return [p.key && !p.key.present && p.needs_key ? 'warn' : 'bad', p.key && !p.key.present && p.needs_key ? 'Needs a key' : (p.status.reason || 'Unavailable')];
    // --- ai-ui hook (r2): a reached 30-day spend cap refuses every call to this provider; say so in the head
    const capFull = p.kind === 'cloud' && usage && usage.caps ? capStatus(usage.caps[p.id], fmt) : null;
    if (capFull) return ['warn', capFull];
    // --- end ai-ui hook
    if (p.key && p.key.present) return ['on', p.key.source === 'env' ? 'Key from environment' : 'Key saved'];
    // Reachable at FOVEA_OPENAI_COMPAT_BASE_URL or the default Ollama address: ready without a saved URL.
    if (p.id === 'openai_compat' && !(p.config && p.config.base_url) && !(p.status && p.status.available)) return ['warn', 'Set a server URL'];
    return ['on', 'Ready'];
  }
  function providerBlock(p) {
    const [st, stText] = statusOf(p);
    const isOpen = open.has(p.id);
    const body = h('div', { class: 'prov-body', hidden: !isOpen });
    const head = h('button', { class: 'prov-head', type: 'button', 'aria-expanded': String(isOpen), onClick: () => {
      const now = body.hidden; body.hidden = !now; head.setAttribute('aria-expanded', String(now)); wrap.classList.toggle('open', now);
      if (now) { open.add(p.id); if (!body.firstChild) fillBody(p, body); } else open.delete(p.id);
      remember();
    } },
      h('span', { class: `dot ${st}` }),
      h('div', { style: 'min-width:0' }, h('div', { class: 'nm' }, p.name), h('div', { class: 'loc' }, p.id === 'openai_compat' && p.config.base_url ? `Images go to ${p.config.base_url}` : p.location)),
      h('span', { class: cls('st', st === 'on' ? 'ok-text' : st === 'bad' ? 'danger-text' : st === 'warn' ? 'warn-text' : 'faint') }, stText),
      h('span', { class: 'chev' }, icon('chevronRight', 14)));
    const wrap = h('div', { class: cls('prov', isOpen && 'open'), 'data-provider': p.id }, head, body);
    if (isOpen) fillBody(p, body);
    provNodes.set(p.id, wrap);
    return wrap;
  }
  function replaceProvider(entry) {
    const i = P.providers.findIndex(x => x.id === entry.id);
    if (i >= 0) P.providers[i] = entry;
    const old = provNodes.get(entry.id);
    if (old && old.isConnected) old.replaceWith(providerBlock(entry));
  }
  function fillBody(p, body) {
    const cfg = p.config || {};
    const status = h('span', { class: 'small', 'aria-live': 'polite' });
    const say = (text, kind = 'ok') => { status.textContent = text; status.className = `small ${kind === 'ok' ? 'ok-text' : kind === 'bad' ? 'danger-text' : 'muted'}`; };
    const saveCfg = async (patch, okText = 'Saved') => {
      try { const entry = await api.put(`/api/ai/providers/${p.id}/config`, patch); Object.assign(p, entry); say(okText); refreshHead(entry); fovea.bus.emit('ai:providers'); return true; }
      catch (e) { say(fovea.plainError(e), 'bad'); return false; }
    };
    const refreshHead = (entry) => {
      const [st, stText] = statusOf(entry);
      const wrap = provNodes.get(p.id); if (!wrap) return;
      const dot = wrap.querySelector('.prov-head .dot'), stEl = wrap.querySelector('.prov-head .st');
      if (dot) dot.className = `dot ${st}`;
      if (stEl) { stEl.textContent = stText; stEl.className = cls('st', st === 'on' ? 'ok-text' : st === 'bad' ? 'danger-text' : st === 'warn' ? 'warn-text' : 'faint'); }
    };
    // A browser on another machine sees what it may not change, disabled (the server would refuse it with a 403).
    const lock = lockOf(p);
    const locked = (el, on) => { if (on) el.querySelectorAll('button, input, select, textarea').forEach(x => { x.disabled = true; }); return el; };
    const parts = [];
    if (lock.key || lock.settings) parts.push(h('div', { class: 'callout small mb-8 prov-lock' }, icon('shield', 14),
      h('span', `Read-only from another machine: the ${[lock.key && 'key', lock.settings && 'settings', lock.call && 'Test'].filter(Boolean).join(', ').replace(/, ([^,]*)$/, ' and $1')} of ${p.name} only work on the computer Fovea runs on.`)));
    // -- key (write-only)
    if (p.key) {
      const k = p.key;
      const keyBox = h('div', { class: 'col gap-6' });
      const form = () => {
        const inp = h('input', { class: 'input mono', type: 'password', autocomplete: 'off', spellcheck: 'false', placeholder: `Paste the ${p.name} API key`, 'aria-label': `${p.name} API key`, name: `fovea-${p.id}-key` });
        const note = h('span', { class: 'hint' });
        const pre = KEY_PREFIX[p.id];
        inp.addEventListener('input', () => { note.textContent = pre && inp.value && !inp.value.trim().startsWith(pre[0]) ? `This does not look like a key for ${p.name} (${pre[1]}). You can still save it.` : ''; note.className = note.textContent ? 'hint warn-text' : 'hint'; });
        const save = async (btn) => {
          const v = inp.value.trim();
          if (!v) { inp.focus(); return; }
          btn.disabled = true;
          try {
            const r = await api.put(`/api/ai/providers/${p.id}/key`, { key: v }); fovea.bus.emit('ai:providers');
            inp.value = '';                          // never kept in the page
            replaceProvider(r.provider);
            const again = provNodes.get(p.id); const st = again && again.querySelector('.key-status'); if (st) { st.textContent = 'Saved — use Test to check it'; st.className = 'small ok-text key-status'; }
          } catch (e) { btn.disabled = false; note.textContent = fovea.plainError(e); note.className = 'hint danger-text'; }
        };
        inp.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.isComposing) { e.preventDefault(); save(saveBtn); } });
        const saveBtn = h('button', { class: 'btn btn-primary btn-sm', onClick: (e) => save(e.currentTarget) }, 'Save key');
        return h('div', { class: 'col gap-4' }, h('div', { class: 'key-form' }, inp, saveBtn, k.present ? h('button', { class: 'btn btn-sm btn-ghost', onClick: () => drawKey() }, 'Cancel') : null), note,
          h('span', { class: 'hint' }, 'Sent once to this server and stored in its data directory (owner-only file); it is never shown again — only the last characters.'));
      };
      const drawKey = (editing = false) => {
        if (k.present && k.source === 'env') {
          put(keyBox, h('div', { class: 'key-line' }, icon('key', 13), h('span', `From the environment variable ${k.env || ''}`), h('span', { class: 'hint-mono' }, k.hint || 'saved'), ui.chip('read-only', { cls: 'outline' })),
            h('span', { class: 'hint' }, 'Remove the variable and restart Fovea to manage this key here.'));
        } else if (k.present && !editing) {
          put(keyBox, h('div', { class: 'key-line' }, icon('key', 13), h('span', 'Key saved'), h('span', { class: 'hint-mono', title: 'last characters only' }, k.hint || '••••'),
            h('span', { class: 'small muted' }, 'stored in this server’s data directory'), h('span', { class: 'small key-status' }),
            h('span', { class: 'spacer' }),
            h('button', { class: 'btn btn-sm', onClick: () => drawKey(true) }, 'Replace'),
            h('button', { class: 'btn btn-sm btn-danger', onClick: async (e) => {
              if (!await ui.confirm({ title: `Remove the ${p.name} key?`, message: `${p.name} can no longer be used until a key is saved again.`, okLabel: 'Remove', danger: true })) return;
              try { const r = await api.del(`/api/ai/providers/${p.id}/key`); replaceProvider(r.provider); fovea.bus.emit('ai:providers'); } catch (err) { ui.toast(fovea.plainError(err), { type: 'error' }); }
            } }, 'Remove')));
        } else put(keyBox, form());
        locked(keyBox, lock.key);
        if (editing) { const i = keyBox.querySelector('input'); i && i.focus(); }
      };
      drawKey();
      parts.push(row(p.needs_key ? 'API key' : 'API key (optional)', keyBox));
    }
    // -- server URL + coordinates (OpenAI-compatible)
    if (p.id === 'openai_compat') {
      // An empty field still has an address: FOVEA_OPENAI_COMPAT_BASE_URL or the default Ollama one (p.endpoint).
      const ep = p.endpoint && p.endpoint.source !== 'settings' ? p.endpoint : null;
      const url = h('input', { class: 'input mono', value: cfg.base_url || '', placeholder: ep ? ep.url : 'http://localhost:11434/v1', 'aria-label': 'Server URL', spellcheck: 'false' });
      const urlBtn = h('button', { class: 'btn btn-sm', onClick: () => saveCfg({ base_url: url.value.trim() }, 'Server URL saved') }, 'Save');
      url.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.isComposing) { e.preventDefault(); urlBtn.click(); } });
      const now = !ep ? '' : ep.source === 'env' ? ` Now using ${ep.url} from ${ep.env || 'the environment'}; a URL saved here takes its place.` : ` Now using the default Ollama address, ${ep.url}.`;
      parts.push(locked(row('Server URL', h('div', { class: 'key-form' }, url, urlBtn), h('span', { class: 'hint' }, `Any server with the OpenAI chat-completions API: Ollama, vLLM, LM Studio, llama.cpp. Images go to this address.${now}`)), lock.settings));
      const coordHint = h('span', { class: 'hint' }, COORD_HELP[cfg.coord || 'auto']);
      parts.push(locked(row('Box coordinates', h('select', { class: 'select auto', 'aria-label': 'Box coordinate format', onChange: (e) => { coordHint.textContent = COORD_HELP[e.target.value]; saveCfg({ coord: e.target.value }); } },
        Object.keys(COORD_HELP).map(o => h('option', { value: o, selected: o === (cfg.coord || 'auto') }, { auto: 'Automatic', qwen3: 'Qwen 3 (0–1000)', qwen25: 'Qwen 2.5 (pixels)', gemini: 'Gemini style (y first)', norm: 'Normalised 0–1' }[o]))), coordHint), lock.settings));
    }
    // -- default model
    if (p.models && p.models.length) {
      const sel = h('select', { class: 'select', 'aria-label': 'Default model', style: 'max-width:520px', onChange: (e) => saveCfg({ default_model: e.target.value }, 'Default model saved') });
      const fill = (models) => put(sel, ...models.map(m => h('option', { value: m.id, selected: m.id === (cfg.default_model || p.default_model) },
        `${m.name || m.id}${m.recommended ? ' (recommended)' : ''}${m.price ? ` — $${m.price.in} / $${m.price.out} per 1M tokens` : m.size_mb ? ` — ${m.size_mb} MB` : ''}`)));
      fill(p.models);
      const note = h('span', { class: 'hint' });
      const showNote = () => { const m = (p.models || []).find(x => x.id === sel.value); note.textContent = m && m.notes ? m.notes : ''; };
      sel.addEventListener('change', showNote); showNote();
      sel.disabled = lock.settings;
      const canRefresh = p.kind === 'cloud' ? !!(p.key && p.key.present) : p.id === 'openai_compat';
      parts.push(row('Default model', h('div', { class: 'row wrap' }, sel, canRefresh ? h('button', { class: 'btn btn-sm', disabled: lock.call, 'data-tip': 'Ask the provider which models this key can use', onClick: async (e) => {
        const b = e.currentTarget; b.disabled = true; say('Asking the provider…', 'muted');
        try { const r = await api.get(`/api/ai/providers/${p.id}/models`, { refresh: 1 }); p.models = r.models || p.models; fill(p.models); showNote(); say(r.error ? `Using the built-in list (${r.error})` : `${fmt.plural((r.models || []).length, 'model')} available`, r.error ? 'bad' : 'ok'); }
        catch (err) { say(fovea.plainError(err), 'bad'); }
        b.disabled = false;
      } }, icon('refresh', 12), 'Refresh list') : null), note));
    }
    // -- effort, cost controls (cloud)
    if (p.kind === 'cloud' && p.id !== 'mock' || p.id === 'openai_compat') {
      parts.push(locked(row('Effort', ui.seg([{ value: 'low', label: 'Fast' }, { value: 'medium', label: 'Balanced' }, { value: 'high', label: 'Accurate' }], cfg.effort || 'low', (v) => saveCfg({ effort: v }, 'Effort saved'), { label: 'Effort' }),
        h('span', { class: 'hint' }, 'More effort lets the model think longer: usually tighter boxes, slower and costlier (thinking tokens are billed as output). The AI panel and forms use this for every model of this provider unless you pick another effort there.')), lock.settings));   // ai-ui hook (r2): Balanced reaches Annotate and the AI tab
    }
    if (p.kind === 'cloud') {
      const capU = usage && usage.caps ? usage.caps[p.id] : null;
      const spent = usage && usage.by_provider && usage.by_provider[p.id] ? usage.by_provider[p.id].usd : 0;   // when no cap is set
      const unpricedRuns = Number((((usage && usage.unpriced) || {}).by_provider || {})[p.id]?.runs) || 0;   // booked at $0: the cap cannot see them
      const cap = h('input', { class: 'input', type: 'number', min: 0, step: '0.5', value: cfg.spend_cap_usd ?? '', placeholder: 'no cap', 'aria-label': 'Spend cap in US dollars', style: 'width:120px' });
      const capBtn = h('button', { class: 'btn btn-sm', onClick: () => saveCfg({ spend_cap_usd: cap.value === '' ? null : Number(cap.value) }, cap.value === '' ? 'Cap removed' : 'Cap saved') }, 'Save');
      cap.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); capBtn.click(); } });
      parts.push(locked(row('Spend cap', h('div', { class: 'row wrap' }, h('div', { class: 'input-affix' }, h('span', { class: 'affix' }, '$'), cap), capBtn, h('span', { class: 'small muted' }, capUsageText(capU, spent, unpricedRuns, fmt))),
        h('span', { class: 'hint' }, 'Rolling 30 days. What runs still going have reserved counts against it too; new runs are refused once it is reached, and a running batch stops cleanly at its own budget.')), lock.settings));
      if (hasFlexTier(p)) {
        parts.push(locked(row('Flex tier', h('div', { class: 'row' }, ui.switchBtn(!!cfg.batch_mode, (v) => saveCfg({ batch_mode: v }, v ? 'Flex tier on' : 'Flex tier off'), { label: 'Flex tier' }), h('span', { class: 'small' }, 'Cheaper, slower')),
          h('span', { class: 'hint' }, `Every ${p.name} request — single-image proposals too — goes to the Flex tier: batch prices, but answers can take minutes. Models without a Flex tier run at the standard price.`)), lock.settings));
      }
      if (p.capabilities && p.capabilities.batch_api) {
        parts.push(row('Batch API', h('span', { class: 'small' }, 'Choose ', h('b', 'Economy'), ' in the batch dialog (Annotate → Batch…, or AI → Label images): about half price, results within 24 h.')));
      }
    }
    if (p.kind === 'cloud' || p.id === 'openai_compat') {
      const conc = h('input', { class: 'input', type: 'number', min: 1, max: 16, step: 1, value: cfg.concurrency || 4, 'aria-label': 'Parallel requests', style: 'width:80px', onChange: (e) => saveCfg({ concurrency: Number(e.target.value) }, 'Saved') });
      parts.push(locked(row('Parallel requests', conc, h('span', { class: 'hint' }, 'Requests in flight during a batch. Lower it if the provider rate-limits you (Fovea waits and retries on 429 either way).')), lock.settings));
    }
    parts.push(locked(row('Enabled', h('div', { class: 'row' }, ui.switchBtn(cfg.enabled !== false, (v) => saveCfg({ enabled: v }, v ? 'Enabled' : 'Turned off — hidden from AI pickers'), { label: 'Enabled' }), h('span', { class: 'small muted' }, 'Show this provider in the AI pickers'))), lock.settings));
    // -- privacy + test
    const priv = p.id === 'openai_compat' ? `Images go to ${cfg.base_url || (p.endpoint && p.endpoint.url) || 'the server URL above'}. Nothing leaves this machine when that server runs here.` : PRIVACY[p.id] || (p.kind === 'local' ? 'Runs on this machine — images never leave it.' : p.location);
    parts.push(h('div', { class: 'privacy mt-8' }, icon(p.kind === 'cloud' && p.id !== 'mock' ? 'cloud' : 'shield', 13), h('span', priv)));
    const testBtn = h('button', { class: 'btn btn-sm', onClick: async () => {
      testBtn.disabled = true; say('Testing…', 'muted');
      try { const r = await api.post(`/api/ai/providers/${p.id}/test`); say(`${r.ok ? '✓' : '✕'} ${r.message || (r.ok ? 'Works' : 'Failed')}${r.latency_ms != null ? ` · ${fmt.num(r.latency_ms)} ms` : ''}`, r.ok ? 'ok' : 'bad'); }
      catch (e) { say(fovea.plainError(e), 'bad'); }
      testBtn.disabled = false;
    } }, icon('zap', 12), 'Test');
    testBtn.disabled = lock.call;
    parts.push(h('div', { class: 'row mt-8' }, testBtn, status));
    put(body, ...parts);
  }

  // ---- weights
  let W = null;
  const jobsWatch = new Map();
  async function loadWeights() {
    try { W = await api.get('/api/ai/weights'); } catch (e) { if (alive) put(weightsBody, failed(e, () => { put(weightsBody, waiting()); loadWeights(); })); return; }
    if (!alive) return;
    drawWeights();
    // Re-attach to downloads that were started earlier (another tab, before a reload).
    try { const { jobs } = await api.get('/api/jobs', { kind: 'ai_weights', active: 1 }); jobs.forEach(j => j.extra && j.extra.name && follow(j.extra.name, j)); } catch (e) { /* none */ }
  }
  const FAMILY = { yolo: 'Detector (COCO)', yoloe: 'Open vocabulary', sam: 'Segment Anything', text: 'Text encoder' };
  function drawWeights() {
    const rows = (W.weights || []).map(w => {
      const action = h('td', { class: 'right' });
      const tr = h('tr', { 'data-weight': w.name },
        h('td', h('div', { class: 'strong mono small' }, w.name), h('div', { class: 'xs faint' }, w.note || '')),
        h('td', { class: 'small' }, FAMILY[w.family] || w.family),
        h('td', { class: 'num' }, w.size_mb != null ? `${w.size_mb} MB` : '–'),
        h('td', { class: 'small muted' }, w.license || ''),
        action);
      drawAction(w, action);
      return tr;
    });
    put(weightsBody, 
      h('div', { class: 'small muted mb-8' }, 'Downloaded only when you click Download, from the Ultralytics releases on GitHub, into ', h('span', { class: 'mono' }, W.model_dir || 'the model folder'), '. You can also drop .pt files there yourself.'),
      h('div', { class: 'table-scroll' }, h('table', { class: 'table weights-table' }, h('thead', h('tr', h('th', 'Weights'), h('th', 'Kind'), h('th', { class: 'num' }, 'Size'), h('th', 'Licence'), h('th', { class: 'right' }, ''))), h('tbody', rows))));
  }
  function drawAction(w, cell) {
    const j = jobsWatch.get(w.name);
    if (w.downloaded) { put(cell, h('span', { class: 'small ok-text nowrap' }, icon('check', 12), ' Downloaded')); return; }
    if (j) {
      const bar = ui.progress(j.snap.progress || 0, { indeterminate: !j.snap.total });
      put(cell, h('div', { class: 'row', style: 'min-width:220px' }, bar, ui.elapsedClock((j.snap.started_at || j.snap.created_at || Date.now() / 1000) * 1000),
        h('button', { class: 'btn btn-sm btn-ghost', onClick: () => api.post(`/api/jobs/${j.snap.id}/cancel`).catch((e) => ui.toast(fovea.plainError(e), { type: 'error' })) }, 'Cancel')));
      j.bar = bar;
      return;
    }
    put(cell, h('button', { class: 'btn btn-sm', onClick: async (e) => {
      const btn = e.currentTarget;              // currentTarget is null once the await below returns
      btn.disabled = true;
      try { const r = await api.post('/api/ai/weights/download', { name: w.name }); follow(w.name, r.job); fovea.jobs.refresh(); }
      catch (err) { btn.disabled = false; ui.toast(fovea.plainError(err), { type: 'error' }); }
    } }, icon('download', 12), `Download ${w.size_mb ? `${w.size_mb} MB` : ''}`));
  }
  function follow(name, job) {
    if (jobsWatch.has(name)) return;
    const entry = { snap: job, bar: null };
    jobsWatch.set(name, entry);
    const redraw = () => { const tr = weightsBody.querySelector(`tr[data-weight="${CSS.escape(name)}"]`); const w = W && W.weights.find(x => x.name === name); if (tr && w) drawAction(w, tr.lastChild); };
    redraw();
    const watch = api.watchJob(job.id, (snap) => { entry.snap = snap; if (entry.bar) { entry.bar.className = `progress ${snap.total ? '' : 'indeterminate'}`; entry.bar.firstChild.style.width = `${Math.round((snap.progress || 0) * 100)}%`; } });
    offs.push(() => watch.stop());
    watch.then((snap) => {
      jobsWatch.delete(name);
      if (!alive) return;
      if (snap.status === 'done') { ui.toast(`${name} downloaded`, { type: 'ok' }); loadWeights(); loadProviders(); }
      else { ui.toast(`Download of ${name} cancelled`); redraw(); }
    }).catch((e) => { jobsWatch.delete(name); if (alive) { ui.toast(`Download of ${name} failed: ${fovea.plainError(e)}`, { type: 'error' }); redraw(); } });
  }

  // ---- storage, server, plugins (/api/meta)
  async function loadMeta() {
    try { meta = await api.get('/api/meta'); }
    catch (e) { if (alive) { put(storageBody, failed(e, () => { put(storageBody, waiting()); loadMeta(); })); put(pluginsBody); } return; }
    if (!alive) return;
    const tc = meta.thumb_cache || {};
    const tcText = (c) => c.files == null ? 'counting…' : `${fmt.num(c.files)} files · ${fmt.bytes(c.bytes)}${c.max_bytes ? ` of ${fmt.bytes(c.max_bytes)} max` : ''}`;
    const cacheEl = h('span', tcText(tc));
    // The first count after a server start runs in the background: ask again until it has a number.
    if (tc.files == null) (async () => { for (let i = 0; i < 20 && alive && cacheEl.isConnected !== false; i++) { await new Promise(r => setTimeout(r, 1500)); try { const c = (await api.get('/api/cache/stats')).thumb_cache || {}; if (c.files != null) { cacheEl.textContent = tcText(c); return; } } catch (e) { return; } } })();
    const cropEl = h('span', 'counting…');
    const cropText = (r) => `${fmt.num(r.files)} files · ${fmt.bytes(r.bytes)}${r.limit_bytes ? ` of ${fmt.bytes(r.limit_bytes)} max` : ''}`;
    const loadCrops = () => api.get('/api/crops/cache').then(r => { if (alive) cropEl.textContent = cropText(r); }).catch(() => { cropEl.textContent = '–'; });
    loadCrops();
    const recount = async () => {
      try { const c = (await api.get('/api/cache/stats')).thumb_cache || {}; if (alive && c.files != null) cacheEl.textContent = tcText(c); } catch (e) { /* the row keeps its text */ }
      loadCrops();
    };
    const clear = (url, el, what) => async (e) => {
      const btn = e.currentTarget;
      if (!await ui.confirm({ title: `Clear the ${what}?`, message: `Cached ${what === 'thumbnail cache' ? 'thumbnails' : 'object crops'} are deleted and rebuilt when needed. Your images and labels are not touched.`, okLabel: 'Clear' })) return;
      btn.disabled = true;
      try { const r = await api.post(url); ui.toast(`Removed ${fmt.plural(r.removed ?? r.files ?? 0, 'cached file')}`); await recount(); } catch (err) { ui.toast(fovea.plainError(err), { type: 'error' }); }
      btn.disabled = false;
    };
    /** One button for both caches: thumbnails and object crops are drawn again when they are next needed. */
    const clearAll = async (e) => {
      const btn = e.currentTarget;
      if (!await ui.confirm({ title: 'Clear all caches?', message: 'Thumbnails and object crops are deleted and drawn again when they are next shown, so the first pages load a little slower. Images, labels, review marks and the index are not touched.', okLabel: 'Clear caches' })) return;
      btn.disabled = true;
      const res = await Promise.allSettled([api.post('/api/cache/clear'), api.post('/api/crops/cache/clear')]);
      const removed = res.reduce((n, r) => n + (r.status === 'fulfilled' ? Number(r.value.removed) || 0 : 0), 0);
      const failed = res.filter(r => r.status === 'rejected');
      if (failed.length) ui.toast(`Not every cache was cleared: ${fovea.plainError(failed[0].reason)}`, { type: 'error' });
      else ui.toast(`Caches cleared · ${fmt.plural(removed, 'file')} removed`, { type: 'ok' });
      await recount();
      btn.disabled = false;
    };
    const srv = meta.server || {};
    put(storageBody, 
      row('Version', h('span', { class: 'mono' }, meta.version)),
      row('Data directory', h('span', { class: 'mono small break' }, meta.data_dir), h('span', { class: 'hint' }, 'Index database, thumbnails, crops, label versions and saved keys.')),
      row('Model directory', h('span', { class: 'mono small break' }, meta.model_dir)),
      row('Thumbnail cache', h('div', { class: 'row wrap' }, cacheEl, h('button', { class: 'btn btn-sm', onClick: clear('/api/cache/clear', cacheEl, 'thumbnail cache') }, 'Clear'))),
      row('Object crop cache', h('div', { class: 'row wrap' }, cropEl, h('button', { class: 'btn btn-sm', onClick: clear('/api/crops/cache/clear', cropEl, 'crop cache') }, 'Clear')),
        h('span', { class: 'hint' }, 'Box crops shown by Objects and the class cards.')),
      row('All caches', h('div', { class: 'row wrap' }, h('button', { class: 'btn btn-sm', onClick: clearAll }, icon('trash', 12), 'Clear caches')),
        h('span', { class: 'hint' }, 'Thumbnails and object crops together — for example after many images were edited outside Fovea, or to free disk space.')),
      row('Host path mapping', meta.host_path ? h('span', { class: 'mono small break' }, `${meta.host_path} → ${meta.container_mount}`) : h('span', { class: 'faint' }, 'none (running directly on the host)')),
      srv.bind_host ? row('Server', h('span', { class: 'small' }, `Listening on ${srv.bind_host}${srv.loopback_only ? ' — this computer only' : ' — reachable from other machines'}${srv.token_required ? ' · access token required' : ''}`),
        srv.allowed_hosts && srv.allowed_hosts.length ? h('span', { class: 'hint' }, `Allowed host names: ${srv.allowed_hosts.join(', ')}`) : null) : null);
    put(pluginsBody, ...(meta.plugins.length ? meta.plugins.map(p => row(p.name, h('span', { class: 'small' }, p.description || ''), p.error ? ui.chip('error: ' + p.error, { type: 'danger' }) : null))
      : [h('div', { class: 'small faint' }, 'Add a package under fovea/plugins/ exposing PLUGIN to extend Fovea.')]));
    drawRemote();
  }

  loadMeta(); loadProviders(); loadWeights();
  if (ctx.query.get('s') === 'ai' || location.hash === '#ai-providers') setTimeout(() => { const t = document.getElementById('ai-providers'); t && t.scrollIntoView({ block: 'start' }); }, 50);
}
