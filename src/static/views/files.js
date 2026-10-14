// Files tab: browse the dataset root with previews (images, text, YOLO label files with class names and boxes).
import { decodedImage } from '../lib/media.js';
/** replaceChildren that skips null / false and flattens arrays (DOM replaceChildren would print "null"). */
const put = (el, ...c) => { el.replaceChildren(...c.flat(Infinity).filter(x => x != null && x !== false && x !== true)); return el; };

export function install(fovea) {
  fovea.registerTab({ id: 'files', label: 'Files', icon: 'folder', order: 40, render });
  if (fovea.keys && fovea.keys.registerKeys) fovea.keys.registerKeys('files', 'Files', [
    ['↑ ↓', 'Move in the folder list'], ['PageUp PageDown', 'Move 15 rows'], ['Home End', 'First / last entry'],
    ['Enter', 'Open the folder or preview the file'], ['Backspace', 'Up one folder'],
  ], 40);
}

const PAGE = 500;

async function render(el, { ctx, dataset, fovea }) {
  const { h, icon, ui, fmt, cls } = fovea;
  const ds = dataset;
  const enc = encodeURIComponent(ds.id);
  const names = ds.classes || [];
  const root = h('div', { class: 'files' });
  const top = h('div', { class: 'files-top' });
  const list = h('div', { class: 'files-list', role: 'listbox', 'aria-label': 'Folder contents', tabindex: '-1' });
  const preview = h('div', { class: 'files-preview', 'aria-live': 'polite' });
  root.appendChild(top); root.appendChild(h('div', { class: 'files-body' }, list, preview)); el.appendChild(root);
  let path = ctx.query.get('path') || '';
  let loadSeq = 0, openSeq = 0, ctrl = null, alive = true;
  const fileUrl = (rel, extra = '') => `/api/datasets/${enc}/file?path=${encodeURIComponent(rel)}${extra}`;
  const emptyPreview = () => put(preview, h('div', { class: 'empty' }, icon('fileSearch', 30), h('p', 'Select a file to preview it')));
  emptyPreview();

  /** Open a folder. A folder change is a history entry (Back goes up again); only the newest request renders. */
  async function load(p, { push = true, offset = 0 } = {}) {
    const my = ++loadSeq;
    // docs hook: a folder opened from the list (Enter, Backspace, click) keeps the focus in the list, on the folder we
    // came from when going up — the rows are replaced, which would otherwise drop the focus to <body> and end ↑ ↓ Enter.
    const keepFocus = offset === 0 && list.contains(document.activeElement), from = path;
    if (ctrl) ctrl.abort();
    ctrl = new AbortController();
    if (offset === 0) {
      path = p;
      if (push) fovea.router.pushQuery({ path: p || null }); else fovea.router.replaceQuery({ path: p || null });
      renderCrumbs(p, null);
      put(list, h('div', { class: 'wait-row' }, ui.spinner()));
      emptyPreview();
    }
    let data;
    try { data = await fovea.api.get(`/api/datasets/${enc}/files`, { path: p, offset, limit: PAGE }, { signal: ctrl.signal }); }
    catch (e) {
      if (my !== loadSeq || !alive || (e && e.name === 'AbortError')) return;
      put(list, h('div', { class: 'empty compact' }, h('p', fovea.plainError(e)), p ? h('button', { class: 'btn btn-sm', onClick: () => load(parentOf(p)) }, icon('cornerUpLeft', 12), 'Up one folder') : null));
      return;
    }
    if (my !== loadSeq || !alive) return;
    renderCrumbs(p, data);
    if (offset === 0) put(list);
    else { const m = list.querySelector('.load-more'); if (m) m.remove(); }
    if (offset === 0 && p) list.appendChild(row({ name: '..', is_dir: true, up: true }, () => load(parentOf(p))));
    if (offset === 0 && !data.entries.length) list.appendChild(h('div', { class: 'empty compact' }, h('p', 'Empty folder')));
    data.entries.forEach(e => list.appendChild(row(e, () => e.is_dir ? load(e.rel) : open(e))));
    if (data.truncated) {
      const next = offset + data.entries.length;
      list.appendChild(h('div', { class: 'load-more' }, h('button', { class: 'btn btn-sm', onClick: (ev) => { ev.currentTarget.disabled = true; load(p, { offset: next }); } }, `Show more (${fmt.num(data.total - next)} left)`)));
    }
    if (offset === 0) {
      const rows = [...list.querySelectorAll('.file-row')], t = (keepFocus && rows.find(r => r._entry && r._entry.rel === from)) || rows[0];
      if (t && document.activeElement === document.body) { t.tabIndex = 0; if (keepFocus) { t.focus(); t.scrollIntoView({ block: 'nearest' }); } }
    }
  }
  const parentOf = (p) => p.split('/').slice(0, -1).join('/');

  function renderCrumbs(p, data) {
    const parts = p ? p.split('/') : [];
    const crumbs = [h('a', { href: '#', 'aria-label': 'Dataset root', title: 'Dataset root', onClick: (e) => { e.preventDefault(); load(''); } }, icon('database', 14))];
    parts.forEach((name, i) => { const rel = parts.slice(0, i + 1).join('/'); crumbs.push(h('span', { class: 'sep' }, '/'), i === parts.length - 1 ? h('span', { title: name, class: 'strong' }, name) : h('a', { href: '#', title: name, onClick: (e) => { e.preventDefault(); load(rel); } }, name)); });
    const full = data ? data.root_host + (p ? '/' + p : '') : (ds.root_host + (p ? '/' + p : ''));
    put(top, h('nav', { class: 'crumbs', 'aria-label': 'Folder' }, ...crumbs),
      h('span', { class: 'root-path mono small' }, ui.pathText(full, { tail: 40 })),
      data && data.total != null ? h('span', { class: 'xs faint nowrap' }, fmt.plural(data.total, 'item')) : null,
      h('button', { class: 'btn btn-sm', 'aria-label': 'Copy folder path', onClick: () => ui.copyText(full) }, icon('copy', 12), 'Copy path'));
  }

  function row(e, act) {
    const ic = e.up ? 'cornerUpLeft' : e.is_dir ? 'folder' : e.kind === 'image' ? 'image' : e.kind === 'text' ? 'fileText' : 'file';
    const r = h('div', { class: cls('file-row', e.is_dir && 'dir'), role: 'option', tabindex: '-1', 'aria-selected': 'false', 'aria-label': e.up ? 'Parent folder' : e.name, title: e.up ? 'Up one folder' : e.name, onClick: () => { select(r); act(); } },
      icon(ic), h('span', { class: 'fname' }, e.up ? '..' : ui.midText(e.name, { tail: 12, title: e.name })), e.is_dir || e.size == null ? null : h('span', { class: 'size' }, fmt.bytes(e.size)));
    r._entry = e;
    return r;
  }
  function select(r) {
    list.querySelectorAll('.file-row.active').forEach(x => { x.classList.remove('active'); x.setAttribute('aria-selected', 'false'); });
    r.classList.add('active'); r.setAttribute('aria-selected', 'true');
    list.querySelectorAll('.file-row[tabindex="0"]').forEach(x => { x.tabIndex = -1; });
    r.tabIndex = 0;
  }
  // Keyboard: arrows move, Enter opens, Backspace goes up a folder (roving tab stop: one Tab reaches the list).
  list.addEventListener('keydown', (e) => {
    if (e.isComposing || e.metaKey || e.ctrlKey || e.altKey) return;
    const rows = [...list.querySelectorAll('.file-row')];
    if (!rows.length) return;
    let i = rows.indexOf(document.activeElement);
    const move = (n) => { e.preventDefault(); const t = rows[Math.max(0, Math.min(rows.length - 1, n))]; rows.forEach(x => { x.tabIndex = -1; }); t.tabIndex = 0; t.focus(); t.scrollIntoView({ block: 'nearest' }); };
    if (e.key === 'ArrowDown') move(i + 1);
    else if (e.key === 'ArrowUp') move(i < 0 ? 0 : i - 1);
    else if (e.key === 'Home') move(0);
    else if (e.key === 'End') move(rows.length - 1);
    else if (e.key === 'PageDown') move(i + 15);
    else if (e.key === 'PageUp') move(i - 15);
    else if (e.key === 'Enter' && i >= 0) { e.preventDefault(); rows[i].click(); }
    else if (e.key === 'Backspace' && path) { e.preventDefault(); load(parentOf(path)); }
  });
  list.addEventListener('focusin', (e) => { if (e.target === list) { const r = list.querySelector('.file-row.active') || list.querySelector('.file-row'); if (r) { r.tabIndex = 0; r.focus(); } } });

  const downloadBtn = (e) => h('a', { class: 'btn btn-sm', href: fileUrl(e.rel, '&raw=1'), download: e.name, 'data-external': '' }, icon('download', 13), 'Download');
  const head = (e, ...extra) => h('div', { class: 'pv-head' }, ui.midText(e.rel, { tail: 24, cls: 'mono small' }), h('span', { class: 'xs faint nowrap' }, fmt.bytes(e.size)), ...extra, downloadBtn(e));

  async function open(e) {
    const my = ++openSeq;
    if (e.kind === 'image') {
      // Decode first, then swap in one step: the previous preview stays until the next image is ready.
      let img;
      const wait = setTimeout(() => { if (my === openSeq && !preview.querySelector('.pv-img')) put(preview, h('div', { class: 'wait-row' }, ui.spinner())); }, 0);
      try { img = await decodedImage(fileUrl(e.rel), { priority: 'high' }); img.alt = e.name; }
      catch (err) { clearTimeout(wait); if (my === openSeq) put(preview, head(e), h('div', { class: 'callout warn small' }, icon('alert', 14), h('span', 'This image cannot be shown in the browser. Download it to view it.'))); return; }
      clearTimeout(wait);
      if (my !== openSeq || !alive) return;
      put(preview, head(e, e.image_id ? h('a', { class: 'btn btn-sm', href: `/d/${enc}/annotate?img=${e.image_id}` }, icon('pen', 12), 'Annotate') : ui.chip('not indexed', { type: 'warn', title: 'This file is not part of the index (outside the image folders, or added after the last scan)' })),
        h('div', { class: 'pv-img' }, img));
      return;
    }
    if (e.kind === 'text') {
      const pre = h('pre');
      const info = h('div', { class: 'small faint mt-8' });
      const more = h('div', { class: 'mt-8' });
      put(preview, head(e), h('div', { class: 'wait-row' }, ui.spinner()));
      let offset = 0;
      const chunk = async () => {
        let d;
        try { d = await fovea.api.get(`/api/datasets/${enc}/file`, { path: e.rel, offset }); }
        catch (err) { if (my === openSeq) put(preview, head(e), h('div', { class: 'callout danger small' }, icon('alertCircle', 14), h('span', fovea.plainError(err)))); return; }
        if (my !== openSeq || !alive) return;
        if (d.empty) { put(preview, head(e), h('div', { class: 'empty compact' }, icon('file', 26), h('p', 'Empty file (0 bytes).'))); return; }
        if (d.binary) { put(preview, head(e), h('div', { class: 'empty compact' }, icon('file', 26), h('p', 'This looks like a binary file, so it is not shown as text.'), downloadBtn(e))); return; }
        if (offset === 0) {
          const parts = [head(e)];
          const label = labelPreview(e, d.text);
          if (label) parts.push(label);
          parts.push(pre, info, more);
          put(preview, ...parts);
        }
        pre.textContent += d.text;
        offset = d.next_offset ?? (offset + (d.shown || 0));
        const shown = d.next_offset != null ? d.next_offset : d.size;
        info.textContent = d.truncated ? `Showing ${fmt.bytes(shown)} of ${fmt.bytes(d.size)}${d.encoding && d.encoding !== 'utf-8' ? ` · decoded as ${d.encoding}` : ''}` : (d.encoding && d.encoding !== 'utf-8' ? `Decoded as ${d.encoding}` : '');
        put(more, d.truncated && d.next_offset != null ? h('button', { class: 'btn btn-sm', onClick: (ev) => { ev.currentTarget.disabled = true; chunk(); } }, 'Load more') : null);
      };
      chunk();
      return;
    }
    put(preview, h('div', { class: 'empty' }, icon('file', 30), h('p', `${e.name} · ${fmt.bytes(e.size)}`), h('p', { class: 'small faint' }, 'No preview for this file type.'), downloadBtn(e)));
  }

  /** A YOLO label file: its rows with class names, and its boxes drawn on the image it belongs to. */
  function labelPreview(e, text) {
    if (!/\.txt$/i.test(e.name)) return null;
    const lines = text.split(/\r?\n/).filter(l => l.trim());
    if (!lines.length || lines.length > 5000) return null;
    const rows = lines.map(l => l.trim().split(/\s+/).map(Number));
    if (!rows.every(r => r.length >= 5 && r.every(Number.isFinite) && Number.isInteger(r[0]) && r[0] >= 0)) return null;
    const boxes = rows.map(r => [r[0], r[1], r[2], r[3], r[4]]);
    const counts = new Map(); boxes.forEach(b => counts.set(b[0], (counts.get(b[0]) || 0) + 1));
    const box = h('div', { class: 'col gap-8 mb-12' },
      h('div', { class: 'row wrap gap-6' }, h('span', { class: 'small strong' }, `YOLO labels · ${fmt.plural(boxes.length, 'box', 'boxes')}`),
        [...counts].sort((a, b) => a[0] - b[0]).map(([c, n]) => h('span', { class: 'chip outline' }, h('span', { class: 'swatch', style: `background:${fovea.colors.classColor(c)}` }), `${c} ${fovea.colors.className(names, c)} × ${n}`))));
    const imgSlot = h('div');
    box.appendChild(imgSlot);
    // Find the image this label file belongs to (same stem, indexed) and draw the boxes on it.
    const stem = e.name.replace(/\.txt$/i, '');
    fovea.api.get(`/api/datasets/${enc}/images`, { q: stem, limit: 8 }).then((r) => {
      const hit = (r.items || []).find(it => it.rel_path.split('/').pop().replace(/\.[^.]+$/, '') === stem);
      if (!hit || !alive) return;
      decodedImage(fovea.api.thumbUrl(hit.id, 640, hit.mtime)).then((img) => {
        if (!alive) return;
        img.alt = hit.rel_path;
        put(imgSlot, h('div', { class: 'xs faint mb-4' }, 'On ', h('a', { href: `/d/${enc}/annotate?img=${hit.id}`, class: 'mono' }, hit.rel_path)),
          h('div', { class: 'pv-img' }, img, fovea.boxLayer(boxes, { names, labels: true, stroke: 1.5 })));
      }).catch(() => {});
    }).catch(() => {});
    return box;
  }

  await load(path, { push: false });
  const destroy = () => { alive = false; if (ctrl) ctrl.abort(); };
  return destroy;
}
