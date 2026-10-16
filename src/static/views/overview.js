// Overview tab: dataset health at a glance — counts, class balance, label issues, data quality (duplicates,
// sequences, image quality), AI label audit, label history (snapshots), the dataset's folders and its classes
// (names and AI class cards).
import { sortSplits } from './home.js';
import { KIND_COUNT } from './ai.js';
import { parseClassNames, duplicateNames } from '../lib/classnames.js';
/** replaceChildren that skips null / false and flattens arrays (DOM replaceChildren would print "null"). */
const put = (el, ...c) => { el.replaceChildren(...c.flat(Infinity).filter(x => x != null && x !== false && x !== true)); return el; };

export function install(fovea) { fovea.registerTab({ id: 'overview', label: 'Overview', icon: 'layout', order: 10, render }); }

const ISSUE_ICON = { missing_label: 'fileSearch', empty_label: 'file', bad_format: 'alert', bad_class: 'tag', out_of_range: 'maximize', zero_area: 'minus', tiny_box: 'zoomIn', duplicate_box: 'copy', image_unreadable: 'image' };
const QA_ICON = { near_dup: 'copy', leak_split: 'copyX', dup_same_role: 'copy', dup_label_mismatch: 'gitCompare', seq_gap: 'film', seq_flicker: 'film', seq_jump: 'move', seq_count_spike: 'barChart', shape_outlier: 'square', blurry: 'eyeOff', dark: 'moon', overexposed: 'sun' };
const QA_GROUP_LABEL = { duplicates: 'Duplicates', sequence: 'Sequences', shape: 'Box shapes', quality: 'Image quality' };
const AUDIT_KINDS = ['missing', 'loose', 'swap', 'spurious'];
const AUDIT_FALLBACK = { missing: 'Likely missed objects', loose: 'Loose boxes', swap: 'Possibly wrong class', spurious: 'Boxes without support' };
const DRAFT_KEY = (id) => `fovea.classDraft.${id}`;
const ACTIVE = ['queued', 'running'];

/** The QA job the Data quality card follows: any active qa job of this dataset — from /qa/summary, the shell's
    job poll or dataset.job — the running one first, then the oldest queued. */
export function qaJobFor(summary, jobs, dsId, dsJob = null) {
  const list = [];
  for (const j of [...(jobs || []), summary && summary.job, dsJob]) {
    if (!j || j.kind !== 'qa' || !ACTIVE.includes(j.status) || (j.dataset_id && j.dataset_id !== dsId)) continue;
    const i = list.findIndex(x => x.id === j.id);
    if (i < 0) list.push(j);
    else if (j.status === 'running' && list[i].status !== 'running') list[i] = j;   // the fresher of two snapshots
  }
  list.sort((a, b) => (a.status === 'running' ? 0 : 1) - (b.status === 'running' ? 0 : 1) || (a.created_at || 0) - (b.created_at || 0));
  return list[0] || null;
}
/** What the Data quality card shows: 'running' | 'scanning' | 'never' | 'stale' | 'ok'. A qa job anywhere wins
    over the summary's own status (which can lag a job that was queued a moment ago, N2). */
export function qaMode(summary, job) {
  const s = summary || {};
  if (job || s.status === 'running') return 'running';
  if (s.scanning) return 'scanning';
  if (!s.computed_at) return 'never';
  return s.status === 'stale' ? 'stale' : 'ok';
}
/** The median of the non-zero per-class box counts — the base of stats' under-represented rule (< 10% of it). */
export function medianBoxes(classes) {
  const n = (classes || []).map(c => c.boxes || 0).filter(v => v > 0).sort((a, b) => a - b);
  return n.length ? n[Math.floor(n.length / 2)] : 0;
}
/** Overview links into Explore: an empty query would make Explore restore the last session's filters, so "all
    images" says so explicitly (sort=id is Explore's default order). */
export function exploreQuery(q) {
  const p = new URLSearchParams(q || {});
  if (![...p.keys()].length) p.set('sort', 'id');
  return p.toString();
}
/** Annotate over the audit suspects, worst first (→ goes to the next suspect, not the next image of the dataset). */
export const suspectLink = (enc, imageId) => `/d/${enc}/annotate?img=${imageId}&ai=audit&sort=audit&order=desc`;
/** Review links open the most suspicious images first: a QA flag by its own score (sort=flag ranks by the flags the
    issue filter names), audit findings by the audit score. The order is spelled out — Explore and Annotate start
    every sort but 'audit' ascending, and both carry an explicit order on to the next view. */
export const flagReview = (code) => ({ issue: code, sort: 'flag', order: 'desc' });
export const auditReview = (q) => ({ ...q, sort: 'audit', order: 'desc' });
/** Scan problems the layout notes above already state (e.g. a labels-only folder) are not repeated in the report. */
const SAME_AS_LAYOUT_NOTE = { labels_without_images: /no matching images folder/i };
export function withoutLayoutDupes(rep, layoutNotes) {
  if (!rep) return rep;
  const notes = (layoutNotes || []).map(String);
  const problems = rep.problems.filter(p => !(SAME_AS_LAYOUT_NOTE[p.code] && notes.some(n => SAME_AS_LAYOUT_NOTE[p.code].test(n))));
  if (problems.length === rep.problems.length) return rep;
  return { ...rep, problems, shownAbove: rep.problems.length - problems.length,
    warn: problems.filter(p => p.severity === 'warn').length + rep.notes.length, info: problems.filter(p => p.severity === 'info').length };
}
/** An audit tile: findings of one kind (boxes, or predicted objects for "missing") and the images the link opens. */
export function auditTileText(kind, findings, images) {
  const unit = kind === 'missing' ? ['object', 'objects'] : ['box', 'boxes'];
  const n = Number(findings) || 0;
  return { unit: n === 1 ? unit[0] : unit[1], where: images == null ? '' : `in ${Number(images).toLocaleString('en-US')} image${images === 1 ? '' : 's'}` };
}
/** A suspect row's findings in the AI tab's words, in the tiles' order: "1 missing box · 2 loose boxes" (kinds with
    none are left out). */
export const auditCountsText = (counts) => {
  const c = counts || {}, order = [...AUDIT_KINDS, ...Object.keys(c).filter(k => !AUDIT_KINDS.includes(k))];
  return order.filter(k => Number(c[k]) > 0).map(k => { const n = Number(c[k]); return `${n.toLocaleString('en-US')} ${(KIND_COUNT[k] || [k, k])[n === 1 ? 0 : 1]}`; }).join(' · ');
};
/** How many automatic label versions the server keeps (GET /snapshots: keep_auto versions of more than small_files
    files and, counted apart, keep_small smaller ones; versions saved by hand are never dropped). */
export function retentionText(r) {
  const big = Number(r && r.keep_auto) || 20, small = Number(r && r.keep_small) || 0, files = Number(r && r.small_files) || 20;
  return small
    ? `The newest ${big} automatic versions of more than ${files} files are kept, plus the newest ${small} smaller ones; versions you save stay until you delete them.`
    : `The newest ${big} automatic versions are kept; versions you save stay until you delete them.`;
}
/** The restore confirmation's lead. A full restore rewrites only the files that differ from the version (the diff's
    images_changed); with no diff at hand it says that in words rather than counting every file of the version. */
export function restoreText(s, imageIds, summary, fmt) {
  const when = `${fmt.ago(s.created_at)} (${fmt.date(s.created_at)})`;
  if (imageIds) return imageIds.length === 1 ? `This image’s label file will be put back as it was ${when}.` : `${fmt.plural(imageIds.length, 'label file')} will be put back as they were ${when}.`;
  if (summary && summary.images_changed != null) {
    const n = Number(summary.images_changed) || 0, same = Math.max(0, (Number(s.n_files) || 0) - n);
    return `${fmt.plural(n, 'changed label file')} will be put back as ${n === 1 ? 'it was' : 'they were'} ${when}.${same ? ` The other ${fmt.num(same)} already match it and are not touched.` : ''}`;
  }
  return `Every label file that differs from this version will be put back as it was ${when}. Files that already match (the version holds ${fmt.plural(s.n_files, 'file')}) are not touched.`;
}
/** "+3 boxes" / "−1 box" for a diff summary; null for a side with nothing to report. */
export function diffBoxText(sm, fmt) {
  const t = (sign, n) => (n ? `${sign}${fmt.plural(Number(n), 'box', 'boxes')}` : null);
  return { added: t('+', sm && sm.boxes_added), removed: t('−', sm && sm.boxes_removed) };
}
/** The Dataset card's split column: as wide as its longest name (wide characters count double, plus the cell's
    padding), at least 110 px and at most 32 characters — a name is shortened only when it is really long. A table
    column ignores a width that mixes in a percentage (it falls back to equal shares), so the cap is in ch. */
export function splitColWidth(splits) {
  const w = Math.max(0, ...(splits || []).filter(Boolean).map(s => [...String(s)].reduce((n, c) => n + (c.codePointAt(0) >= 0x1100 ? 2 : 1), 0)));
  return `max(110px, calc(${Math.min(w, 32)}ch + 28px))`;
}
/** Placeholder class names ("class_3", "3", "") that say nothing about the class — the cue to suggest names. */
export const isPlaceholderName = (n) => n == null || /^\s*$/.test(String(n)) || /^(class[_ -]?)?\d+$/i.test(String(n).trim());
/** The latest index's report (dataset.last_scan) as a compact summary: problems with counts, warnings first. */
export function scanReport(last) {
  if (!last || typeof last !== 'object') return null;
  const problems = (Array.isArray(last.problems) ? last.problems : []).filter(p => p && typeof p === 'object' && (p.count || 0) > 0)
    .map(p => ({ code: p.code || 'other', label: p.label || String(p.code || 'Problem').replace(/_/g, ' '), severity: p.severity === 'info' ? 'info' : 'warn',
      count: Number(p.count) || 0, note: p.note || '', examples: (Array.isArray(p.examples) ? p.examples : []).slice(0, 5).map(String) }));
  // Reports written before problems were structured only carry plain notes.
  const loose = problems.length ? [] : (Array.isArray(last.notes) ? last.notes : []).filter(Boolean).map(String);
  problems.sort((a, b) => (a.severity === 'warn' ? 0 : 1) - (b.severity === 'warn' ? 0 : 1));
  return { at: last.at || null, images: last.images ?? null, elapsed: last.elapsed ?? null, indexed: last.indexed ?? null, removed: last.removed ?? null,
    problems, notes: loose, warn: problems.filter(p => p.severity === 'warn').length + loose.length, info: problems.filter(p => p.severity === 'info').length };
}

async function render(el, { dataset, fovea, refresh }) {
  const { h, icon, ui, fmt, link, cls } = fovea;
  const { classColor, className, splitAlpha, withAlpha } = fovea.colors;
  const api = fovea.api;
  const page = h('div', { class: 'page' });
  const inner = h('div', { class: 'page-inner' });
  page.appendChild(inner); el.appendChild(page);
  let ds = dataset, alive = true, drawSeq = 0, first = true, stats = null, issueLabels = {};
  const enc = encodeURIComponent(ds.id);
  const explore = (q) => `/d/${enc}/explore?${exploreQuery(q)}`;
  const go = (url) => fovea.router.navigate(url);
  const card = (title, sub, body, span, actions) => h('section', { class: `card ${span || ''}`, 'aria-label': title },
    h('div', { class: 'card-header' }, h('h3', title), sub ? h('span', { class: 'sub' }, sub) : null, actions ? h('div', { class: 'actions' }, actions) : null),
    h('div', { class: 'card-body' }, body));
  const clickRow = (props) => ({ ...props, role: 'link', tabindex: '0' });

  // Cards that load on their own keep their nodes across redraws (no flash, no lost scroll position).
  const qaBody = h('div'), qaActions = h('div', { class: 'row gap-4' });
  const auditBody = h('div');
  const histBody = h('div'), histActions = h('div', { class: 'row gap-4' });
  const classesBox = h('div'), classesActions = h('div', { class: 'row gap-4' });
  const datasetBox = h('div'), datasetActions = h('div', { class: 'row gap-4' });

  // ------------------------------------------------------------------ main draw (serialised)
  async function draw() {
    const my = ++drawSeq;
    if (ds.status !== 'ready' && !ds.image_count) {
      put(inner, ui.emptyState({ icon: 'refresh', title: ds.status === 'scanning' ? 'Indexing the dataset…' : 'Not indexed yet', message: 'Statistics appear as soon as the index is built.' }));
      return;
    }
    if (first) put(inner, h('div', { class: 'wait-row' }, ui.spinner(), h('span', 'Loading statistics…')));
    let r;
    try { r = await api.get(`/api/datasets/${enc}/stats`); }
    catch (e) {
      if (my !== drawSeq || !alive) return;
      if (first) put(inner, ui.emptyState({ icon: 'alert', title: 'Could not load statistics', message: fovea.plainError(e), action: h('button', { class: 'btn', onClick: () => draw() }, icon('refresh', 13), 'Try again') }));
      else ui.toast(`Statistics not refreshed: ${fovea.plainError(e)}`, { type: 'error' });
      return;
    }
    if (my !== drawSeq || !alive) return;
    stats = r.stats;
    issueLabels = { ...(stats.issue_labels || r.issue_labels || {}) };
    const back = classFocus();      // build() moves the class editor into the new grid: keep a field being typed in
    const content = build(stats);
    const top = page.scrollTop;
    put(inner, content);            // one swap: never a blank page between two states
    page.scrollTop = top;
    back();
    first = false;
  }

  // The tile counts label-file problems only; issue=any would also match every data-quality flag (near-duplicates…).
  const labelIssueCodes = (st) => (st.issues || []).filter(i => i.severity !== 'info' && i.count).map(i => i.code).join(',') || 'any';
  function build(stats) {
    const t = stats.totals; const rv = stats.reviews; const reviewed = rv.approved + rv.flagged + rv.excluded;
    const out = document.createDocumentFragment();
    const empties = stats.classes.filter(c => c.empty);
    const unknown = stats.classes.filter(c => c.unknown);
    out.appendChild(h('div', { class: 'stat-grid kpi mb-16' },
      ui.statTile({ label: 'Images', value: fmt.num(t.images), sub: `${fmt.plural(t.sequences, 'sequence')}`, icon: 'images', onClick: () => go(explore({})) }),
      ui.statTile({ label: 'Labeled', value: fmt.pct(t.labeled, t.images), sub: `${fmt.num(t.labeled)} with a label file`, icon: 'tag', onClick: () => go(explore({ labeled: 'file' })), tip: 'Images that have a label file (including empty ones)' }),
      ui.statTile({ label: 'Unlabeled', value: fmt.num(t.unlabeled), sub: t.empty ? `+ ${fmt.num(t.empty)} empty label files` : 'no label file', icon: 'alertCircle', tone: t.unlabeled ? 'warn' : null, onClick: () => go(explore({ labeled: '0' })) }),
      ui.statTile({ label: 'Boxes', value: fmt.num(t.boxes), sub: `${t.avg_boxes} per labeled image`, icon: 'square', onClick: () => go(`/d/${enc}/objects`) }),
      ui.statTile({ label: 'Classes', value: fmt.num(stats.classes.length), sub: unknown.length ? `${unknown.length} unknown ids` : empties.length ? `${empties.length} without boxes` : (ds.classes_source === 'inferred' ? 'inferred from labels' : `from ${ds.classes_source}`), icon: 'tags', tone: unknown.length || empties.length ? 'warn' : null }),
      ui.statTile({ label: 'Label issues', value: fmt.num(t.with_issues), sub: 'images with problems', icon: 'alert', tone: t.with_issues ? 'warn' : 'ok', onClick: () => go(explore({ issue: labelIssueCodes(stats) })) }),
      ui.statTile({ label: 'Reviewed', value: fmt.pct(reviewed, t.images), sub: `${fmt.num(rv.approved)} ✓ · ${fmt.num(rv.flagged)} ⚑ · ${fmt.num(rv.excluded)} ✕`, icon: 'checkCircle', onClick: () => go(explore({ review: 'any' })) }),
    ));
    const grid = h('div', { class: 'ov-grid' });
    out.appendChild(grid);
    grid.appendChild(classCard(stats));
    grid.appendChild(healthCard(stats));
    grid.appendChild(card('Data quality', null, qaBody, 'span-6', qaActions));
    grid.appendChild(card('AI label audit', null, auditBody, 'span-6', h('a', { class: 'btn btn-sm', href: `/d/${enc}/ai?s=audit` }, icon('sparkles', 12), 'Audit setup')));
    grid.appendChild(splitsCard(stats));
    grid.appendChild(bpiCard(stats));
    grid.appendChild(sizeCard(stats));
    grid.appendChild(heatCard(stats));
    grid.appendChild(dimsCard(stats));
    grid.appendChild(card('History', 'label versions', histBody, 'span-4', histActions));
    grid.appendChild(card('Dataset', ds.layout && ds.layout.kind, datasetBox, 'span-7', datasetActions));
    grid.appendChild(card('Classes', `${fmt.num((ds.classes || []).length)} · ${ds.classes_source || ''}`, classesBox, 'span-5', classesActions));
    renderDataset();
    renderClasses();
    return out;
  }

  // ------------------------------------------------------------------ class distribution
  // A split's legend swatch: the same lightness step as its bar segments, with a full-strength edge.
  const swatchFill = (a) => `background:color-mix(in srgb, var(--text-2) ${Math.round(a * 100)}%, transparent)`;
  function classCard(stats) {
    const t = stats.totals;
    const maxBoxes = Math.max(1, ...stats.classes.map(c => c.boxes));
    const splits = sortSplits(stats.splits.map(s => s.split));
    const alphaOf = (s) => splitAlpha(splits.indexOf(s));
    const label = (s) => s || '(no split)';
    const rows = stats.classes.map(c => {
      const color = classColor(c.cls);
      const tip = splits.filter(s => c.by_split[s]).map(s => `${label(s)}: ${fmt.num(c.by_split[s])}`).join(' · ');
      return h('div', clickRow({ class: cls('hbar clickable', c.empty && 'empty-cls'), onClick: () => go(explore({ cls: c.cls })), 'aria-label': `${c.cls} ${c.name}: ${fmt.num(c.boxes)} boxes in ${fmt.num(c.images)} images`, title: tip || null }),
        h('div', { class: 'name' }, h('span', { class: 'swatch', style: `background:${color}` }), h('span', { class: 'nm', title: c.name }, `${c.cls} · ${c.name}`),
          c.unknown ? ui.chip('not in class list', { type: 'warn' }) : c.empty ? ui.chip('empty', { type: 'warn' }) : null),
        // One outlined bar per class (its length on the shared scale), split into segments by lightness steps.
        h('div', { class: 'track' }, c.boxes ? h('span', { class: 'bar', style: `width:${100 * c.boxes / maxBoxes}%;--c:${color}` },
          splits.map(s => (c.by_split[s] || 0) ? h('i', { style: `flex:${c.by_split[s]} 1 0;background:${withAlpha(color, alphaOf(s))}` }) : null)) : null),
        h('div', { class: 'val' }, h('b', fmt.num(c.boxes)), ` · ${fmt.num(c.images)} img`));
    });
    const legend = splits.length > 1 ? h('div', { class: 'legend mt-12', 'aria-label': 'Split shading' }, h('span', { class: 'faint' }, 'Bar shading by split:'),
      splits.map(s => h('span', h('span', { class: 'legend-swatch', style: swatchFill(alphaOf(s)) }), label(s)))) : null;
    return card('Class distribution', `${fmt.num(t.boxes)} boxes`, h('div', null,
      rows.length ? h('div', { class: 'hbars', style: rows.length > 24 ? 'max-height:520px;overflow-y:auto;padding-right:4px' : '' }, rows) : h('div', { class: 'faint' }, 'No boxes yet'), legend), 'span-8');
  }

  // ------------------------------------------------------------------ health
  function healthCard(stats) {
    const t = stats.totals, rv = stats.reviews, reviewed = rv.approved + rv.flagged + rv.excluded;
    const issueRows = stats.issues.map(i => h('div', { class: 'issue-row' }, icon(ISSUE_ICON[i.code] || 'alert', 14), h('span', { class: 'lbl', title: i.label }, i.label), h('span', { class: 'n' }, fmt.num(i.count)),
      link(explore({ issue: i.code }), { class: 'btn btn-sm', 'aria-label': `Review ${i.label}` }, 'Review')));
    const empties = stats.classes.filter(c => c.empty);
    const under = stats.classes.filter(c => c.under_represented && !c.empty && !c.unknown);
    const absent = stats.classes.filter(c => !c.empty && (c.missing_in || []).length);
    const names = (list) => list.slice(0, 6).map(c => c.name).join(', ') + (list.length > 6 ? ` and ${list.length - 6} more` : '');
    return card('Health check', t.with_issues ? `${fmt.plural(t.with_issues, 'image')} need attention` : 'No label issues', h('div', null,
      issueRows.length ? issueRows : h('div', { class: 'callout ok small' }, icon('checkCircle', 14), h('span', 'All label files parsed cleanly.')),
      empties.length ? h('div', { class: 'callout warn small mt-12' }, icon('tags', 14), h('span', `${empties.length === 1 ? 'Class' : 'Classes'} without any box: ${names(empties)}. Label some, or remove trailing unused classes in the class list below.`)) : null,
      under.length ? h('div', { class: 'callout warn small mt-8' }, icon('barChart', 14), h('span', `Under-represented (fewer than 10% of the median class’s ${fmt.plural(medianBoxes(stats.classes), 'box', 'boxes')}): ${names(under)}.`)) : null,
      absent.length ? h('div', { class: 'callout small mt-8' }, icon('info', 14), h('span', absent.slice(0, 4).map(c => `${c.name} has no boxes in ${c.missing_in.join(', ')}`).join('; ') + (absent.length > 4 ? '; …' : '') + '.')) : null,
      h('div', { class: 'divider' }),
      h('div', { class: 'small strong muted mb-8' }, 'Review progress'),
      h('div', { class: 'review-bar', style: 'height:8px', role: 'img', 'aria-label': `${fmt.pct(reviewed, t.images)} reviewed` }, h('i', { class: 'a', style: `width:${100 * rv.approved / Math.max(1, t.images)}%` }), h('i', { class: 'f', style: `width:${100 * rv.flagged / Math.max(1, t.images)}%` }), h('i', { class: 'x', style: `width:${100 * rv.excluded / Math.max(1, t.images)}%` })),
      h('div', { class: 'legend mt-8' }, h('span', h('span', { class: 'swatch', style: 'background:var(--approve)' }), `Approved ${fmt.num(rv.approved)}`), h('span', h('span', { class: 'swatch', style: 'background:var(--flag)' }), `Flagged ${fmt.num(rv.flagged)}`), h('span', h('span', { class: 'swatch', style: 'background:var(--exclude)' }), `Excluded ${fmt.num(rv.excluded)}`), h('span', { class: 'faint' }, `${fmt.num(t.images - reviewed)} unreviewed`)),
      h('div', { class: 'row wrap mt-12' }, link(explore({ review: 'none' }), { class: 'btn btn-sm' }, icon('eye', 13), 'Review unreviewed'), link(explore({ review: 'flagged' }), { class: 'btn btn-sm' }, icon('flag', 13), 'Flagged')),
    ), 'span-4');
  }

  // ------------------------------------------------------------------ splits & charts
  // span-5 with snug number columns: a split name such as "test-1854-1860-포함" stays whole from ~1100 px up, and
  // a longer one keeps both ends (middle ellipsis, full name in the tooltip).
  function splitsCard(stats) {
    const order = sortSplits(stats.splits.map(s => s.split));
    const rows = order.map(name => stats.splits.find(s => s.split === name));
    return card('Splits', fmt.plural(stats.splits.length, 'split'), h('div', { class: 'table-scroll' }, h('table', { class: 'table snug' },
      h('thead', h('tr', h('th', 'Split'), h('th', { class: 'num' }, 'Images'), h('th', { class: 'num' }, 'Labeled'), h('th', { class: 'num' }, 'Boxes'), h('th', { class: 'num' }, 'Per img'))),
      h('tbody', rows.map((s, i) => h('tr', clickRow({ class: 'clickable', onClick: () => go(explore(s.split ? { split: s.split } : {})), 'aria-label': `Explore split ${s.split || 'no split'}` }),
        h('td', { class: 'name-cell' }, h('div', { class: 'row gap-6', style: 'min-width:0' }, h('span', { class: 'legend-swatch', style: `${swatchFill(splitAlpha(i))};width:10px` }), s.split ? ui.midText(s.split, { tail: 8 }) : h('span', { class: 'faint' }, '(no split)'))),
        h('td', { class: 'num' }, fmt.num(s.images)), h('td', { class: 'num' }, fmt.pct(s.labeled, s.images)), h('td', { class: 'num' }, fmt.num(s.boxes)), h('td', { class: 'num' }, s.labeled ? (s.boxes / s.labeled).toFixed(2) : '–')))))), 'span-5');
  }
  function vbars(items, { height = 120, color, labels = true, tip }) {
    const max = Math.max(1, ...items.map(b => b.count));
    return h('div', { class: 'vbars', style: `height:${height}px`, role: 'img', 'aria-label': items.map(b => `${b.label}: ${fmt.num(b.count)}`).join(', ') },
      items.map(b => h('div', { class: 'vb', 'data-tip': tip ? tip(b) : `${b.label}: ${fmt.num(b.count)}` },
        labels ? h('span', { class: 'n' }, b.count ? fmt.compact(b.count) : '0') : null,
        h('i', { class: b.count ? '' : 'zero', style: `height:${b.count ? Math.max(2, 100 * b.count / max) : 2}%${color ? `;background:${color}` : ''}` }), h('span', { class: 'l' }, b.label))));
  }
  // Boxes per image with the classes the boxes are: each bar (the images holding k boxes) is split into the classes
  // of those boxes, the most common classes named in the legend. Picking a class in the legend shows that class's
  // own count per image instead (how many images hold 1, 2 … 10+ of it). The pick survives redraws.
  const BPI_TOP = 10;
  let bpiFocus = null;
  function bpiCard(stats) {
    const by = new Map(stats.boxes_per_image.map(b => [b.k, b.count]));
    // Every bucket 0…10+ is drawn, empty ones too: a missing bar is not the same as a short one.
    const items = Array.from({ length: 11 }, (_, k) => ({ k, label: k === 10 ? '10+' : String(k), count: k === 0 ? (by.get(0) ?? stats.totals.empty ?? 0) : (by.get(k) || 0) }));
    const any = items.some(b => b.count);
    if (!any) return card('Boxes per image', 'labeled images', ui.emptyState({ icon: 'square', title: 'No labeled images', message: 'Counts appear once images have label files.' }), 'span-4');
    if (!stats.boxes_per_image_classes) return card('Boxes per image', 'labeled images', vbars(items, { tip: (b) => `${b.label} ${b.k === 1 ? 'box' : 'boxes'}: ${fmt.num(b.count)} images` }), 'span-4');
    const ranked = stats.classes.filter(c => c.boxes > 0).sort((a, b2) => b2.boxes - a.boxes || a.cls - b2.cls);
    const top = ranked.slice(0, BPI_TOP), rest = ranked.slice(BPI_TOP), topIds = new Set(top.map(c => c.cls));
    const nameOf = (c) => className(ds.classes, c);
    const comp = new Map();                 // k -> Map(cls -> boxes)
    for (const r of stats.boxes_per_image_classes) { if (!comp.has(r.k)) comp.set(r.k, new Map()); comp.get(r.k).set(r.cls, r.boxes); }
    if (bpiFocus != null && !ranked.some(c => c.cls === bpiFocus)) bpiFocus = null;
    const body = h('div');
    const section = card('Boxes per image', '', body, 'span-4');
    const sub = h('span', { class: 'sub' });
    section.querySelector('.card-header h3').after(sub);
    paint();
    return section;

    function paint() {
      if (bpiFocus == null) {
        sub.textContent = 'labeled images · split by class';
        const bars = items.map(b => {
          const m = comp.get(b.k) || new Map();
          const total = [...m.values()].reduce((n, v) => n + v, 0);
          const segs = top.map(c => ({ value: m.get(c.cls) || 0, color: classColor(c.cls) }));
          const other = [...m].filter(([c]) => !topIds.has(c)).reduce((n, [, v]) => n + v, 0);
          if (other) segs.push({ value: other, color: 'var(--text-3)' });
          const named = [...m].sort((x, y) => y[1] - x[1]).slice(0, 4).map(([c, v]) => `${nameOf(c)} ${fmt.num(v)} (${Math.round(100 * v / total)}%)`);
          return { ...b, segs, tip: `${b.label} ${b.k === 1 ? 'box' : 'boxes'}: ${fmt.num(b.count)} images${named.length ? ' · ' + named.join(' · ') + (m.size > 4 ? ' …' : '') : ''}` };
        });
        put(body, stackBars(bars), legend());
      } else {
        const c = bpiFocus, nm = nameOf(c), color = classColor(c);
        sub.textContent = `labeled images · ${nm} per image`;
        const per = new Map(stats.class_per_image.filter(r => r.cls === c).map(r => [r.k, r.images]));
        const holding = [...per.values()].reduce((n, v) => n + v, 0);
        const own = items.map(b => ({ k: b.k, label: b.label, count: b.k === 0 ? Math.max(0, (stats.totals.labeled || 0) - holding) : (per.get(b.k) || 0) }));
        put(body, vbars(own, { color, tip: (b) => `${b.label} × ${nm}: ${fmt.num(b.count)} images` }), legend());
      }
    }
    function legend() {
      const chip = (c) => h('button', { class: cls('chip outline clickable bpi-cls', bpiFocus === c.cls && 'active'), type: 'button', 'aria-pressed': String(bpiFocus === c.cls),
        'data-tip': bpiFocus === c.cls ? 'Back to all classes' : `How many ${nameOf(c.cls)} boxes each image holds`,
        onClick: () => { bpiFocus = bpiFocus === c.cls ? null : c.cls; paint(); } },
        h('span', { class: 'swatch', style: `background:${classColor(c.cls)}` }), h('span', { class: 'truncate' }, nameOf(c.cls)));
      return h('div', { class: 'legend bpi-legend mt-8', 'aria-label': 'Classes' }, top.map(chip),
        rest.length ? h('span', { class: 'chip outline', title: rest.map(c => nameOf(c.cls)).join(', ') }, h('span', { class: 'swatch', style: 'background:var(--text-3)' }), `${rest.length} other`) : null,
        bpiFocus != null ? h('button', { class: 'btn btn-ghost btn-sm', type: 'button', onClick: () => { bpiFocus = null; paint(); } }, 'All classes') : null);
    }
  }
  /** vbars whose bars are split into coloured segments (`segs`: {value, color}, bottom first). */
  function stackBars(items, { height = 120 } = {}) {
    const max = Math.max(1, ...items.map(b => b.count));
    return h('div', { class: 'vbars', style: `height:${height}px`, role: 'img', 'aria-label': items.map(b => b.tip).join('; ') },
      items.map(b => {
        const total = b.segs.reduce((n, sg) => n + sg.value, 0);
        const bar = h('i', { class: cls(b.count && total ? 'stack' : b.count ? 'nobox' : 'zero'), style: `height:${b.count ? Math.max(2, 100 * b.count / max) : 2}%` },
          b.count && total ? b.segs.filter(sg => sg.value).map(sg => h('b', { style: `flex:${sg.value} 1 0;background:${sg.color}` })) : null);
        return h('div', { class: 'vb', 'data-tip': b.tip }, h('span', { class: 'n' }, b.count ? fmt.compact(b.count) : '0'), bar, h('span', { class: 'l' }, b.label));
      }));
  }
  function sizeCard(stats) {
    const any = stats.box_area.some(b => b.count);
    return card('Box size & shape', 'area relative to the image', any ? h('div', null,
      vbars(stats.box_area, { height: 100 }),
      h('div', { class: 'small faint mt-12 mb-4' }, 'Aspect ratio (width / height, in pixels)'),
      vbars(stats.box_aspect, { height: 90, color: 'var(--info)' }),
    ) : ui.emptyState({ icon: 'square', title: 'No boxes yet' }), 'span-4');
  }
  function heatCard(stats) {
    const hm = stats.heatmap;
    // span-3 beside the five-column Splits card (span-5): the map is at most 240 px wide either way.
    if (!hm || !hm.max) return card('Box centres', 'where objects appear', ui.emptyState({ icon: 'target', title: 'No boxes yet' }), 'span-3');
    // Drawn in the shape of the most common frame, so "top left" on the map is top left in the images.
    const d0 = stats.dimensions && stats.dimensions[0];
    const aspect = d0 && d0.width && d0.height ? Math.max(0.5, Math.min(2.4, d0.width / d0.height)) : 16 / 9;
    return card('Box centres', 'where objects appear', h('div', null,
      h('div', { class: 'heat', style: `grid-template-columns:repeat(${hm.n},1fr);grid-template-rows:repeat(${hm.n},1fr);aspect-ratio:${aspect};max-width:${Math.round(240 * aspect)}px`, role: 'img', 'aria-label': `Heatmap of box centres, busiest cell ${fmt.num(hm.max)} boxes` },
        hm.cells.flatMap((row, y) => row.map((v, x) => h('i', { style: `opacity:${0.06 + 0.94 * v / hm.max}`, 'data-tip': `${fmt.num(v)} boxes (column ${x + 1}, row ${y + 1})` })))),
      h('div', { class: 'heat-scale' }, h('span', '0'), h('span', { class: 'ramp' }), h('span', fmt.num(hm.max)), h('span', { class: 'spacer' }), h('span', `${hm.n}×${hm.n} grid`))), 'span-3');
  }
  function dimsCard(stats) {
    const t = stats.totals;
    const top = stats.dimensions[0];
    return card('Image dimensions', `${fmt.plural(stats.dimensions_distinct, 'size')}${top ? ` · mostly ${top.width}×${top.height}` : ''}`, stats.dimensions.length ? h('div', { class: 'table-scroll' }, h('table', { class: 'table' },
      h('thead', h('tr', h('th', 'Width × height'), h('th', { class: 'num' }, 'Images'), h('th', 'Share'))),
      h('tbody', stats.dimensions.map(d => h('tr', h('td', { class: 'mono' }, `${d.width} × ${d.height}`), h('td', { class: 'num' }, fmt.num(d.count)), h('td', { style: 'width:40%' }, h('div', { class: 'bar', title: fmt.pct(d.count, t.images) }, h('i', { style: `width:${100 * d.count / Math.max(1, t.images)}%` })))))))) :
      ui.emptyState({ icon: 'image', title: 'No sizes yet' }), 'span-4');
  }

  // ------------------------------------------------------------------ data quality (qa engine)
  // The card follows the QA job itself — the one /qa/summary reports, or any qa job of this dataset in the shell's
  // job poll (checks queued by a rescan, started in another tab). Its progress block is kept across summary
  // reloads, so the live numbers never freeze on a detached copy while the banner moves on.
  let qaSum = null, qaLoading = false, qaAgain = false, qaRetry = null, seqAll = false;
  const qaLive = { id: null, watch: null, el: null };
  const stopQaWatch = () => { if (qaLive.watch) qaLive.watch.stop(); qaLive.id = null; qaLive.watch = null; qaLive.el = null; };
  const qaJobText = (j) => {
    const t = fovea.jobs.text(j), l = fovea.jobs.label('qa');
    const r = t.startsWith(l) ? t.slice(l.length).replace(/^\s*·\s*/, '') : t;
    return r ? r[0].toUpperCase() + r.slice(1) : 'Starting…';
  };
  function qaProgress(job) {
    if (qaLive.id === job.id && qaLive.el) return qaLive.el;
    stopQaWatch();
    const txt = h('span', { class: 'small grow truncate' }, qaJobText(job));
    const bar = ui.progress(job.progress || 0, { indeterminate: !job.total });
    const el = h('div', { class: 'col gap-4 mb-12 qa-live' }, h('div', { class: 'row' }, icon('scan', 13), txt, ui.elapsedClock(Date.now() - (job.elapsed || 0) * 1000)), bar);
    qaLive.id = job.id; qaLive.el = el;
    const w = qaLive.watch = api.watchJob(job.id, (snap) => {
      if (qaLive.id !== job.id) return;
      txt.textContent = qaJobText(snap);
      bar.className = `progress ${snap.total ? '' : 'indeterminate'}`;
      bar.firstChild.style.width = `${snap.total ? Math.round((snap.progress || 0) * 100) : 40}%`;
    });
    const end = (ok) => { if (qaLive.id !== job.id) return; stopQaWatch(); if (alive) { loadQA(); if (ok) draw(); } };
    w.then(() => end(true)).catch(() => end(false));
    return el;
  }
  function qaJobNow() {
    const cur = fovea.state.get().currentDataset;
    return qaJobFor(qaSum, fovea.state.get().jobs, ds.id, cur && cur.id === ds.id ? cur.job : null);
  }
  async function loadQA() {
    if (qaLoading) { qaAgain = true; return; }
    qaLoading = true;
    let s;
    try { s = await api.get(`/api/datasets/${enc}/qa/summary`); }
    catch (e) { qaLoading = false; if (alive && !qaSum) put(qaBody, h('div', { class: 'small danger-text' }, `Could not load data-quality results: ${fovea.plainError(e)}`)); return; }
    qaLoading = false;
    if (!alive) return;
    qaSum = s;
    renderQA();
    if (qaAgain) { qaAgain = false; loadQA(); }
  }
  function renderQA() {
    const s = qaSum; if (!s || !alive) return;
    const job = qaJobNow();
    const mode = qaMode(s, job);
    const running = mode === 'running';
    clearTimeout(qaRetry);
    // "running" with no job to follow: it finished between two looks — ask again shortly.
    if (running && !job) qaRetry = setTimeout(() => { if (alive) loadQA(); }, 2500);
    if (!running && qaLive.id) stopQaWatch();
    put(qaActions,
      s.refreshing ? h('span', { class: 'xs faint', title: 'Flags of the images you edited are being re-checked' }, 'updating after your edits…')
        : s.computed_at ? h('span', { class: 'xs faint', title: fmt.date(s.computed_at) }, `checked ${fmt.ago(s.computed_at)}`) : null,
      h('button', { class: 'btn btn-sm', disabled: running || mode === 'scanning', onClick: runQA }, icon('refresh', 12), s.computed_at ? 'Check again' : 'Run checks'));
    const labels = { ...(s.labels || {}), ...pick(issueLabels, Object.keys(s.counts || {})) };
    const parts = [];
    if (running) parts.push(job ? qaProgress(job) : h('div', { class: 'col gap-4 mb-12 qa-live' }, h('div', { class: 'row' }, icon('scan', 13), h('span', { class: 'small grow' }, 'Checking…')), ui.progress(0, { indeterminate: true })));
    else if (mode === 'scanning') parts.push(h('div', { class: 'row small muted mb-12 qa-live' }, icon('refresh', 13), h('span', 'Indexing is running — the checks start by themselves when it finishes.')));
    if (!s.computed_at && !running) {
      if (mode !== 'scanning') parts.push(h('div', { class: 'empty compact' }, h('p', 'Duplicate images, frames that leak across splits, sequence gaps, unusual box shapes and blurry or dark images have not been checked yet.'),
        h('button', { class: 'btn btn-sm btn-primary', onClick: runQA }, icon('scan', 13), 'Run checks')));
      put(qaBody, ...parts); return;
    }
    if (mode === 'stale') {
      const what = [s.stale_images ? fmt.plural(s.stale_images, 'image') + ' changed' : '', s.stale_labels ? fmt.plural(s.stale_labels, 'label file') + ' edited' : ''].filter(Boolean).join(' · ');
      if (what) parts.push(h('div', { class: 'callout small mb-8' }, icon('info', 14), h('span', `${what} since the last check — results below may be out of date.`)));
    }
    const groups = s.checks || { all: Object.keys(s.counts || {}) };
    const total = Object.values(s.counts || {}).reduce((a, b) => a + b, 0);
    if (s.computed_at) {
      parts.push(h('div', { class: 'small muted mb-8' }, `${fmt.num(s.unique_images ?? s.images)} distinct images of ${fmt.num(s.images)}`,
        s.dup_groups ? ` · ${fmt.plural(s.dup_groups, 'group')} of near-duplicates` : '', s.exact_dups ? ` · ${fmt.num(s.exact_dups)} exact copies` : ''));
      if (!total) parts.push(h('div', { class: 'callout ok small' }, icon('checkCircle', 14), h('span', 'No duplicates, sequence problems, odd shapes or poor-quality images found.')));
      for (const [g, codes] of Object.entries(groups)) {
        const rows = codes.filter(c => (s.counts || {})[c]).map(c => h('div', { class: 'issue-row' }, icon(QA_ICON[c] || 'alert', 14),
          h('span', { class: 'lbl', title: labels[c] || c }, labels[c] || c), h('span', { class: 'n' }, fmt.num(s.counts[c])),
          link(explore(flagReview(c)), { class: 'btn btn-sm', 'aria-label': `Review ${labels[c] || c}`, 'data-tip': 'Opens Explore, most suspicious first' }, 'Review')));
        if (rows.length) parts.push(h('div', { class: 'issue-group' }, QA_GROUP_LABEL[g] || g), ...rows);
      }
      const span = s.split_spanning_sequences || [];
      if (span.length) {
        parts.push(h('div', { class: 'issue-group' }, `Sequences in more than one split (${fmt.num(span.length)})`),
          h('div', { class: 'small faint mb-4' }, 'Frames of one video in both training and evaluation splits inflate validation scores. Export → split builder keeps each sequence in one split.'),
          h('div', { class: cls('seq-list', seqAll && span.length > 5 && 'open') }, (seqAll ? span : span.slice(0, 5)).map(q => h('div', { class: 'issue-row' }, icon('film', 14), h('span', { class: 'lbl' }, ui.midText(q.seq, { tail: 10 })),
            h('span', { class: 'n', title: Object.entries(q.by_split || {}).map(([k, v]) => `${k}: ${v}`).join(' · ') }, (q.splits || []).join(' + ')),
            link(explore({ seq: q.seq }), { class: 'btn btn-sm', 'aria-label': `Show sequence ${q.seq}` }, 'Show')))),
          span.length > 5 ? h('button', { class: 'btn btn-sm btn-ghost mt-4', 'aria-expanded': String(seqAll), onClick: () => { seqAll = !seqAll; renderQA(); } },
            seqAll ? 'Show fewer' : `Show ${fmt.num(span.length - 5)} more${span.length >= 200 ? ' (the first 200 are listed)' : ''}`) : null);
      }
    }
    put(qaBody, ...parts);
  }
  async function runQA() {
    try { await api.post(`/api/datasets/${enc}/qa`, {}); await fovea.jobs.refresh(); loadQA(); }
    catch (e) { ui.toast(fovea.plainError(e), { type: 'error' }); }
  }

  // ------------------------------------------------------------------ AI audit
  async function loadAudit() {
    let a, facets = null;
    try {
      [a, facets] = await Promise.all([api.get(`/api/datasets/${enc}/audit/summary`, { top: 6 }),
        // images per finding kind (what each tile's link opens): issue facet counts for the whole dataset
        api.get(`/api/datasets/${enc}/images/facets`).then(r => r && r.facets, () => null)]);
    }
    catch (e) { if (alive) put(auditBody, h('div', { class: 'small danger-text' }, `Could not load the audit: ${fovea.plainError(e)}`)); return; }
    if (!alive) return;
    if (!a.run) {
      put(auditBody, h('div', { class: 'empty compact' }, h('p', 'A detector (local or cloud) compares its own detections with your labels and ranks images by how likely their labels are wrong: missed objects, loose boxes, swapped classes, boxes on background.'),
        link(`/d/${enc}/ai?s=audit`, { class: 'btn btn-sm btn-primary' }, icon('fileSearch', 13), 'Set up a label audit')));
      return;
    }
    const run = a.run;
    const running = run.status === 'running' || (run.job && ['queued', 'running'].includes(run.job.status));
    const lab = (k) => issueLabels[`ai_${k}`] || AUDIT_FALLBACK[k];
    put(auditBody, 
      h('div', { class: 'small muted mb-8' }, `${run.provider}${run.model ? ` · ${run.model}` : ''} · ${fmt.num(a.images_audited)} images audited · ${fmt.ago(run.finished_at || run.created_at)}`,
        run.cost_usd ? ` · ${fmt.usd(run.cost_usd)}` : '', running ? h('span', { class: 'warn-text' }, ' · running') : null),
      h('div', { class: 'audit-kinds' }, AUDIT_KINDS.map(k => {
        const n = (a.counts || {})[k] || 0;
        const imgs = facets && facets.issue ? (facets.issue[`ai_${k}`] || 0) : null;
        const t = auditTileText(k, n, n ? imgs : null);
        const inner = [h('span', { class: 'v' }, fmt.num(n), h('span', { class: 'u' }, ` ${t.unit}`)), h('span', { class: 'k' }, lab(k)), t.where ? h('span', { class: 'w' }, t.where) : null];
        return n ? link(explore(auditReview({ issue: `ai_${k}` })), { class: 'audit-kind', 'aria-label': `Review ${lab(k)}` }, ...inner) : h('div', { class: 'audit-kind faint' }, ...inner);
      })),
      a.images_flagged ? h('div', { class: 'row wrap mt-12' }, link(explore(auditReview({ ai: 'audit' })), { class: 'btn btn-sm btn-primary' }, icon('eye', 13), `Review ${fmt.plural(a.images_flagged, 'suspect image')}`),
        h('span', { class: 'xs faint' }, 'most likely wrong first')) : h('div', { class: 'callout ok small mt-12' }, icon('checkCircle', 14), h('span', 'No open suspects.')),
      a.top && a.top.length ? h('div', { class: 'top-list mt-12' }, h('div', { class: 'issue-group' }, 'Most suspicious'),
        // one row per image: the server's summary already drops repeats. Counts that do not fit beside the whole
        // path wrap under it (fovea.css .top-list), so the file name's frame number is never cut.
        a.top.map(it => { const counts = auditCountsText(it.counts);
          return link(suspectLink(enc, it.image_id), { title: it.rel_path }, h('span', { class: 'pth' }, h('span', { class: 'sc' }, it.score.toFixed(2)), ui.midText(it.rel_path, { tail: 16 })),
            counts ? h('span', { class: 'cnt xs faint', title: counts }, counts) : null); })) : null);
  }

  // ------------------------------------------------------------------ history (snapshots)
  async function loadHistory() {
    put(histActions, h('button', { class: 'btn btn-sm', onClick: saveVersion }, icon('save', 12), 'Save version'));
    let r;
    try { r = await api.get(`/api/datasets/${enc}/snapshots`, { limit: 200 }); }
    catch (e) { if (alive) put(histBody, h('div', { class: 'small danger-text' }, `Could not load label versions: ${fovea.plainError(e)}`)); return; }
    if (!alive) return;
    const snaps = r.snapshots || [];
    if (!snaps.length) {
      put(histBody, h('div', { class: 'small muted' }, `Fovea saves the affected label files automatically before every bulk edit, AI write or restore. ${retentionText(r)} Save a version yourself before a big change.`));
      return;
    }
    const shown = snaps.slice(0, 6);
    put(histBody, ...shown.map(snapRow), snaps.length > shown.length ? h('button', { class: 'btn btn-sm btn-ghost mt-8', onClick: () => allVersions(snaps, r) }, `All ${fmt.num(snaps.length)} versions…`) : null);
  }
  /** Why an automatic version exists, from the operation that triggered it. */
  function opText(s) {
    const op = s.op || '';
    if (s.kind === 'manual' || op === 'manual') return 'saved by you';
    if (op.startsWith('restore')) return `before restoring “${op.slice(8).trim() || 'a version'}”`;
    if (op.startsWith('bulk')) return { 'bulk set_class': 'before a bulk class change', 'bulk delete_class': 'before deleting a class’s boxes', 'bulk remap': 'before a class remap', 'bulk clear': 'before clearing labels' }[op] || 'before a bulk label edit';
    if (op === 'ai_accept') return 'before accepting AI proposals';
    if (op.startsWith('ai_')) return `before an AI ${op.slice(3)} write`;
    if (op === 'objects') return 'before an Objects edit';
    if (op.startsWith('range')) return 'before a range edit';
    return op ? `before ${op.replace(/_/g, ' ')}` : 'automatic';
  }
  function snapRow(s) {
    return h('div', { class: 'snap-row' },
      h('span', { class: 'nm', title: s.name || '' }, s.name || (s.kind === 'manual' ? `Version ${s.id}` : `Automatic · ${opText(s)}`)),
      h('div', { class: 'acts' },
        h('button', { class: 'btn btn-sm', onClick: () => openDiff(s) }, 'Changes'),
        h('button', { class: 'btn btn-sm btn-icon', 'aria-label': 'More', 'data-tip': 'Restore or delete', onClick: (e) => ui.menu(e.currentTarget, [
          { label: 'Restore all files…', icon: 'history', onClick: () => restore(s) },
          { label: 'Delete this version…', icon: 'trash', danger: true, onClick: () => deleteSnap(s) },
        ]) }, icon('moreH', 13))),
      h('span', { class: 'meta', title: fmt.date(s.created_at) }, `${s.name ? `${opText(s)} · ` : ''}${fmt.plural(s.n_files, 'file')} · ${fmt.ago(s.created_at)}`));
  }
  function allVersions(snaps, r) {
    ui.drawer({ title: 'Label versions', width: 520, body: h('div', null, h('div', { class: 'small muted mb-8' }, retentionText(r)), snaps.map(snapRow)) });
  }
  async function saveVersion() {
    const name = await ui.prompt({ title: 'Save a version of the labels', label: 'Name', placeholder: 'e.g. Before relabeling class 3', okLabel: 'Save' });
    if (name == null) return;
    try {
      const r = await api.post(`/api/datasets/${enc}/snapshots`, { name: name.trim() });
      ui.toast(r.snapshot_id ? 'Label version saved' : 'Saving the label version…', { type: 'ok' });
      fovea.jobs.refresh();
      loadHistory();
    } catch (e) { ui.toast(fovea.plainError(e), { type: 'error' }); }
  }
  // A restore can finish in a few ms, before the shell's job poll ever sees it — then no job:finished arrives.
  // So the page follows its own restore job and refreshes what the restore changed.
  const restoreWatches = new Set();
  function followRestore(job, onDone) {
    if (!job || !job.id) return;
    const w = api.watchJob(job.id);
    restoreWatches.add(w);
    w.then(async (snap) => {
      restoreWatches.delete(w);
      if (!alive) return;
      if (snap && snap.status === 'cancelled') ui.toast('Restore cancelled');
      try { ds = await refresh(); } catch (e) { /* the header keeps its last state */ }
      if (!alive) return;
      draw(); loadHistory(); loadQA();
      if (onDone) onDone(snap);
    }).catch((e) => { restoreWatches.delete(w); if (alive) { ui.toast(`Restore failed: ${fovea.plainError(e)}`, { type: 'error' }); loadHistory(); } });
  }
  /** A full restore rewrites only the files that differ from the version, so it says how many when the diff is
      known: the Changes drawer passes its summary; from the ⋯ menu it is asked for first. */
  async function restore(s, imageIds, onDone, summary) {
    if (!imageIds && !summary) {
      try { summary = (await api.get(`/api/datasets/${enc}/snapshots/${s.id}/diff`, { limit: 1 })).summary || null; }
      catch (e) { summary = null; }             // the confirmation falls back to wording that needs no count
      if (!alive) return false;
      if (summary && !summary.images_changed) { ui.toast('Nothing to restore — the label files already match this version.', { type: 'ok' }); return false; }
    }
    const ok = await ui.confirm({ title: imageIds ? 'Restore this image’s labels?' : `Restore “${s.name || 'this version'}”?`, okLabel: 'Restore', danger: true,
      message: h('div', { class: 'col gap-8' }, h('p', { style: 'margin:0' }, restoreText(s, imageIds, summary, fmt)),
        h('p', { class: 'small muted', style: 'margin:0' }, 'The current files are saved as a new version first, so this can be undone. A label file that did not exist in that version is removed.')) });
    if (!ok) return false;
    try {
      const r = await api.post(`/api/datasets/${enc}/snapshots/${s.id}/restore`, imageIds ? { image_ids: imageIds } : {});
      ui.toast('Restoring labels…');
      fovea.jobs.refresh();
      followRestore(r && r.job, onDone);
      return true;
    } catch (e) { ui.toast(fovea.plainError(e), { type: 'error' }); return false; }
  }
  async function deleteSnap(s) {
    if (!await ui.confirm({ title: 'Delete this label version?', message: `“${s.name || 'Automatic'}” (${fmt.date(s.created_at)}) can no longer be restored afterwards. Label files are not touched.`, okLabel: 'Delete', danger: true })) return;
    try { await api.del(`/api/datasets/${enc}/snapshots/${s.id}`); loadHistory(); }
    catch (e) { ui.toast(fovea.plainError(e), { type: 'error' }); }
  }
  function openDiff(s) {
    const head = h('div', { class: 'col gap-8' }, h('div', { class: 'wait-row' }, ui.spinner(), h('span', 'Comparing with the files on disk…')));
    const list = h('div');
    const more = h('div', { class: 'row mt-8' });
    let offset = 0, total = 0, seq = 0, summary = null;
    // After a restore from this drawer, compare again (one swap when the answer arrives): restored images drop out.
    const requery = () => { if (d.backdrop.isConnected) page(true); };
    // Enabled once the comparison says something differs: restoring a version that matches would only add a version.
    const restoreAll = h('button', { class: 'btn btn-danger', disabled: true, 'data-tip': 'Comparing with the files on disk…', onClick: async () => { if (await restore(s, null, null, summary)) d.close(); } }, icon('history', 13), 'Restore all…');
    const setRestoreAll = (on, tip) => { restoreAll.disabled = !on; restoreAll.setAttribute('data-tip', tip); };
    const d = ui.drawer({ title: `Changes since “${s.name || 'Automatic'}”`, width: 560,
      body: h('div', { class: 'col gap-12' }, h('div', { class: 'small muted' }, `Saved ${fmt.date(s.created_at)} · ${fmt.plural(s.n_files, 'file')}. Lists what is different in the label files now.`), head, list, more),
      footer: (m) => [h('button', { class: 'btn', onClick: () => m.close() }, 'Close'), restoreAll] });
    const page = async (reset = false) => {
      const my = ++seq;
      if (reset) offset = 0;
      let r;
      try { r = await api.get(`/api/datasets/${enc}/snapshots/${s.id}/diff`, { offset, limit: 50 }); }
      catch (e) {
        if (my !== seq) return;
        put(head, h('div', { class: 'callout danger small' }, icon('alertCircle', 14), h('span', fovea.plainError(e))));
        if (!summary) setRestoreAll(true, 'Put back every label file that differs from this version');
        return;
      }
      if (!d.backdrop.isConnected || my !== seq) return;
      total = r.total || 0;
      if (offset === 0) {
        const sm = r.summary || {};
        summary = { ...sm, images_changed: sm.images_changed ?? total };
        setRestoreAll(summary.images_changed > 0, summary.images_changed > 0 ? `Put back the ${fmt.plural(summary.images_changed, 'changed label file')}` : 'Nothing to restore — the label files match this version');
        const per = Object.entries(sm.per_class || {}).sort((a, b) => Number(a[0]) - Number(b[0]));
        const bx = diffBoxText(sm, fmt);
        put(head, 
          total ? h('div', { class: 'row wrap gap-12' }, h('span', { class: 'strong' }, fmt.plural(summary.images_changed, 'image') + ' changed'),
            bx.added ? h('span', { class: 'ok-text mono' }, bx.added) : null, bx.removed ? h('span', { class: 'danger-text mono' }, bx.removed) : null)
            : h('div', { class: 'callout ok small' }, icon('checkCircle', 14), h('span', 'No differences — the label files match this version.')),
          per.length ? h('div', { class: 'diff-cls' }, per.map(([c, v]) => h('span', { class: 'chip outline' }, h('span', { class: 'swatch', style: `background:${classColor(Number(c))}` }), `${className(ds.classes, Number(c))} `,
            v.added ? h('span', { class: 'ok-text' }, `+${v.added}`) : null, v.removed ? h('span', { class: 'danger-text' }, ` −${v.removed}`) : null))) : null);
      }
      const rows = (r.items || []).map(it => {
        const before = (it.before || []).length, after = (it.after || []).length;
        return h('div', { class: 'diff-item' },
          h('a', { href: `/d/${enc}/annotate?img=${it.image_id}`, style: 'min-width:0;display:flex', title: it.rel_path, onClick: () => d.close() }, ui.midText(it.rel_path, { tail: 18 })),
          h('span', { class: 'mono small', title: `${fmt.plural(before, 'box', 'boxes')} then → ${after} now` }, `${before} → ${after}`),
          h('button', { class: 'btn btn-sm', onClick: (e) => { const b = e.currentTarget; restore(s, [it.image_id], requery).then(ok => { if (ok) { b.disabled = true; b.textContent = 'Restoring…'; } }); } }, 'Restore'));
      });
      if (reset) put(list, rows); else rows.forEach(n => list.appendChild(n));
      offset += (r.items || []).length;
      put(more, offset < total ? h('button', { class: 'btn btn-sm', onClick: (e) => { e.currentTarget.disabled = true; page(); } }, `Show more (${fmt.num(total - offset)} left)`) : null);
    };
    page();
  }

  // ------------------------------------------------------------------ dataset folders
  function renderDataset() {
    const lay = ds.layout || {};
    const root = (ds.root_host || '').replace(/\/+$/, '');
    const rel = (p) => (p && root && p.startsWith(root + '/')) ? p.slice(root.length + 1) : p;
    const cellPath = (p, extra) => h('td', { class: 'name-cell mono small', title: p || '' }, h('div', { class: 'row gap-4', style: 'min-width:0' }, ui.midText(rel(p) || '—', { tail: 22, title: p }), extra || null));
    put(datasetBox, 
      h('div', { class: 'kv mb-12' }, h('b', 'Root'), h('span', { style: 'min-width:0;display:flex' }, ui.pathText(ds.root_host, { tail: 40 }))),
      h('div', { class: 'table-scroll' }, h('table', { class: 'table fixed' }, h('colgroup', h('col', { style: `width:${splitColWidth((lay.sources || []).map(s => s.split))}` }), h('col'), h('col')),
        h('thead', h('tr', h('th', 'Split'), h('th', 'Images'), h('th', 'Labels'))),
        h('tbody', (lay.sources || []).map(s => h('tr', h('td', { class: 'name-cell' }, s.split ? ui.midText(s.split, { tail: 8 }) : '—'),
          cellPath(s.list_file_host || s.img_dir_host),
          s.list_file ? h('td', { class: 'small muted' }, 'next to each image (images → labels)') : cellPath(s.label_dir_host, s.label_dir_exists ? null : ui.chip('will be created', { cls: 'outline' }))))))),
      (lay.notes || []).length ? h('div', { class: 'col gap-4 mt-12' }, lay.notes.map(n => h('div', { class: 'callout small' }, icon('info', 14), h('span', n)))) : null,
      formatSection(lay),
      h('div', { class: 'row wrap mt-12 small muted' }, icon('fileText', 14), lay.data_yaml_host ? h('span', { class: 'grow', style: 'min-width:0;display:flex' }, ui.pathText(lay.data_yaml_host, { tail: 30 })) : h('span', { class: 'grow' }, 'No data.yaml'),
        h('button', { class: 'btn btn-sm', onClick: writeYaml }, icon('save', 13), lay.data_yaml ? 'Rewrite data.yaml…' : 'Write data.yaml…')),
      scanSection(withoutLayoutDupes(scanReport(ds.last_scan), lay.notes), rel));
    put(datasetActions, h('a', { class: 'btn btn-sm', href: `/api/datasets/${enc}/card.md`, download: '', 'data-external': '', 'data-tip': 'DATASET_CARD.md — splits, classes with their cards, data quality, review progress and label issues' }, icon('fileText', 12), 'Dataset card'));
  }
  /** Annotations converted from another format: the format card (where its reading rule came from, the official
      links, licence and citation), the counts, and "Convert again" (replaces edits; a label version is saved first). */
  function formatSection(lay) {
    const fmts = Object.entries(lay.formats || {}).filter(([, f]) => f.state === 'converted' || f.state === 'pending');
    if (!fmts.length) return null;
    const SRC = { catalog: 'built-in format', ai: 'reading rule written by AI from sample lines', web: 'reading rule from the official documentation' };
    return h('div', { class: 'col gap-8 mt-12' }, fmts.map(([fid, f]) => {
      const c = f.card || {}, n = f.counts || {};
      return h('div', { class: 'fmt-box col gap-4' },
        h('div', { class: 'row wrap gap-6' }, ui.chip(c.name || fid, { icon: 'fileText', type: 'accent' }), h('span', { class: 'small muted grow' }, `Annotations converted to YOLO · ${SRC[c.source] || c.source || ''}`),
          f.state === 'converted' ? h('button', { class: 'btn btn-sm', 'data-tip': 'Write the YOLO labels again from the original annotation files. Edits made here are replaced; a label version is saved first.', onClick: () => reconvert(c.name || fid) }, icon('refresh', 12), 'Convert again') : ui.chip('converted on the next index', { cls: 'outline' })),
        n.labels != null ? h('div', { class: 'small muted' }, `${fmt.num(n.labels)} label files · ${fmt.num(n.boxes || 0)} boxes${n.regions ? ` · ${fmt.num(n.regions)} regions` : ''}${n.missing ? ` · ${fmt.num(n.missing)} images without an annotation` : ''}${n.unknown ? ` · ${fmt.num(n.unknown)} entries of unknown classes skipped` : ''}`) : null,
        (c.urls || []).length ? h('div', { class: 'row wrap gap-8 small' }, icon('external', 12), c.urls.slice(0, 3).map(u => h('a', { href: u, target: '_blank', rel: 'noopener noreferrer', 'data-external': '' }, u.replace(/^https?:\/\//, '').replace(/\/$/, '')))) : null,
        c.model ? h('div', { class: 'xs faint' }, `Reading rule written by ${c.model}${c.usd != null && !c.cached ? ` ($${Number(c.usd).toFixed(c.usd < 0.1 ? 3 : 2)})` : ''}`) : null,
        c.citation ? h('div', { class: 'xs faint' }, `Cite: ${c.citation}`) : null,
        c.license ? h('div', { class: 'xs faint' }, `Licence: ${c.license}`) : null,
        c.notes ? h('div', { class: 'xs faint' }, c.notes) : null);
    }));
  }
  async function reconvert(name) {
    const ok = await ui.confirm({ title: `Convert the ${name} annotations again?`, okLabel: 'Convert again', danger: true,
      message: 'The YOLO labels are written again from the original annotation files, then the dataset is re-indexed. Edits made in Fovea to these labels are replaced — a label version is saved first, so History → Restore brings them back.' });
    if (!ok) return;
    try { await api.post(`/api/datasets/${enc}/annotations/reconvert`); ui.toast('Converting the original annotations again…', { type: 'ok' }); fovea.jobs && fovea.jobs.refresh && fovea.jobs.refresh(); }
    catch (e) { ui.toast(fovea.plainError(e), { type: 'error' }); }
  }
  // The latest index's report: one line, collapsed by default; problems with counts and up to five examples each.
  let scanOpen = false;
  function scanSection(rep, rel) {
    const when = `Indexed ${fmt.date(ds.scanned_at)} · registered ${fmt.date(ds.created_at)}`;
    const n = rep ? rep.problems.length + rep.notes.length : 0;
    if (!rep || !n) return h('div', { class: 'row mt-8 small muted' }, icon('clock', 14), h('span', { class: 'grow' }, when, rep && !rep.shownAbove ? ' · no problems in the last index' : ''));
    const secs = (v) => v == null ? '' : v < 60 ? ` in ${Number(v).toFixed(1)} s` : ` in ${fmt.clock(v)}`;
    const d = h('details', { class: 'scan-report mt-8', open: scanOpen || null, onToggle: (e) => { scanOpen = e.currentTarget.open; } },
      h('summary', { class: 'row small muted' }, icon('clock', 14),
        h('span', { class: 'grow', style: 'min-width:0' }, when, rep.images != null ? ` · ${fmt.plural(rep.images, 'image')}${secs(rep.elapsed)}` : ''),
        rep.warn ? ui.chip(fmt.plural(rep.warn, 'problem'), { type: 'warn' }) : null,
        rep.info ? ui.chip(fmt.plural(rep.info, 'note'), { cls: 'outline' }) : null,
        h('span', { class: 'chev' }, icon('chevronRight', 13))),
      h('div', { class: 'scan-body' },
        rep.problems.map(p => h('div', { class: 'scan-prob' },
          h('div', { class: 'issue-row' }, icon(p.severity === 'warn' ? 'alert' : 'info', 14), h('span', { class: 'lbl', title: p.note || p.label }, p.label), h('span', { class: 'n' }, fmt.num(p.count))),
          p.examples.length ? h('ul', { class: 'scan-ex', 'aria-label': `Examples: ${p.label}` }, p.examples.map(x => h('li', { class: 'mono xs' }, ui.midText(rel(x) || x, { tail: 26, title: x })))) : null)),
        rep.notes.map(t => h('div', { class: 'issue-row' }, icon('alert', 14), h('span', { class: 'lbl', title: t }, t)))));
    return d;
  }
  async function writeYaml() {
    let plan;
    try { plan = await api.get(`/api/datasets/${enc}/export/yaml`); } catch (e) { ui.toast(fovea.plainError(e), { type: 'error' }); return; }
    const ok = await ui.confirm({ title: plan.exists ? 'Update data.yaml?' : 'Write data.yaml?', okLabel: plan.exists ? 'Update' : 'Write',
      message: h('div', { class: 'col gap-8' }, h('div', { class: 'small muted' }, plan.exists ? `Updates ${plan.path_host}. Comments and other keys are kept; the previous file is saved as data.yaml.bak.` : `Creates ${plan.path_host}.`),
        h('pre', { class: 'mono small yaml-pre' }, plan.proposed || '')) });
    if (!ok) return;
    try { const r = await api.post(`/api/datasets/${enc}/export/yaml`); ui.toast(`Wrote ${r.path}`, { type: 'ok' }); ds = await refresh(); renderDataset(); }
    catch (e) { ui.toast(fovea.plainError(e), { type: 'error' }); }
  }

  // ------------------------------------------------------------------ classes editor + class cards
  // A working copy: unsaved names never leak into other views, survive redraws, and are kept (per tab session)
  // if the person leaves the tab before saving.
  let names = [...(ds.classes || [])], dirty = false, saving = false, cards = {};
  try { const draft = JSON.parse(sessionStorage.getItem(DRAFT_KEY(ds.id)) || 'null'); if (draft && Array.isArray(draft.names) && JSON.stringify(draft.base) === JSON.stringify(ds.classes || [])) { names = draft.names; dirty = true; } } catch (e) { /* no draft */ }
  const persistDraft = () => { try { if (dirty) sessionStorage.setItem(DRAFT_KEY(ds.id), JSON.stringify({ base: ds.classes || [], names })); else sessionStorage.removeItem(DRAFT_KEY(ds.id)); } catch (e) { /* private mode */ } };
  const onUnload = (e) => { if (dirty) { e.preventDefault(); e.returnValue = ''; } };
  window.addEventListener('beforeunload', onUnload);
  const boxCount = (i) => { const c = stats && stats.classes.find(x => x.cls === i); return c ? c.boxes : 0; };
  // Rows are rebuilt only when the list itself changes (load, add, remove, save, discard, cards). A keystroke updates
  // the row's mark, the footer and "Suggest names" in place: rebuilding the inputs under the caret moved focus to
  // the page after the first letter, and the rest of the name ran global shortcuts (g e opened Explore).
  let footDyn = null, dupesEl = null, saveBtn = null, footKey = '';
  /** Which class-name field has focus (and where its caret is), so a rebuild puts the person back where they were. */
  function classFocus() {
    const sc = classesBox.querySelector('.class-list-scroll');
    const top = sc ? sc.scrollTop : 0;
    const a = document.activeElement;
    const at = a && classesBox.contains(a) ? [...classesBox.querySelectorAll('.class-edit input')].indexOf(a) : -1;
    const sel = at >= 0 ? [a.selectionStart, a.selectionEnd, a.selectionDirection] : null;
    return () => {
      const sc2 = classesBox.querySelector('.class-list-scroll');
      if (sc2 && top) sc2.scrollTop = top;
      if (at < 0 || !classesBox.isConnected) return;
      const inp = classesBox.querySelectorAll('.class-edit input')[at];
      if (!inp || document.activeElement === inp) return;
      inp.focus({ preventScroll: true });
      try { inp.setSelectionRange(...sel); } catch (e) { /* not a text field any more */ }
    };
  }
  function renderClasses() {
    const back = classFocus();
    const rows = names.map((n, i) => {
      // title: the whole name on hover where a narrow card clips it
      const inp = h('input', { class: 'input input-sm', value: n, title: n, 'aria-label': `Name of class ${i}`, spellcheck: 'false',
        onInput: (e) => { names[i] = e.target.value; e.target.title = e.target.value; const row = e.target.closest('.class-edit'); if (row) row.classList.toggle('dirty', (ds.classes || [])[i] !== names[i]); setDirty(true); },
        onKeydown: (e) => { if (e.key === 'Enter' && !e.isComposing) { e.preventDefault(); saveClasses(); } },
        // Several names pasted into one field (a JSON list, a names: block, lines): they fill this row and the ones after.
        onPaste: (e) => {
          const text = (e.clipboardData || window.clipboardData).getData('text');
          const r = parseClassNames(text);
          if (r.error || r.names.length < 2) return;
          e.preventDefault();
          applyPastedNames(r.names, i);
        } });
      const cc = cards[String(i)];
      const hasCard = cc && (cc.description || cc.prompt || cc.include || cc.exclude || (cc.examples || []).length);
      const removable = i === names.length - 1 && boxCount(i) === 0 && i >= 0;
      return h('div', { class: cls('class-edit', (ds.classes || [])[i] !== n && 'dirty') },
        h('span', { class: 'idx' }, i), h('span', { class: 'swatch', style: `background:${classColor(i)}` }), inp,
        h('span', { class: 'n', title: 'boxes' }, fmt.num(boxCount(i))),
        h('div', { class: 'row gap-4' },
          boxCount(i) ? link(`/d/${enc}/objects?cls=${i}`, { class: 'btn btn-sm btn-ghost', 'aria-label': `Examples of class ${i}`, 'data-tip': 'See every box of this class in Objects' }, icon('images', 12), h('span', { class: 'lbl-wide' }, 'Examples'))
            : h('span', { class: 'btn btn-sm btn-ghost', 'aria-hidden': 'true', style: 'visibility:hidden' }, icon('images', 12), h('span', { class: 'lbl-wide' }, 'Examples')),
          h('button', { class: 'btn btn-sm', 'data-tip': hasCard ? 'Edit the class card (AI prompt, definition, examples)' : 'Describe this class for AI providers', onClick: () => editCard(i) }, 'Card', hasCard ? h('span', { class: 'card-dot', 'aria-label': 'has a card' }) : null),
          removable ? h('button', { class: 'btn btn-sm btn-ghost btn-icon', 'aria-label': `Remove class ${i}`, 'data-tip': 'Remove this unused last class', onClick: () => { names.pop(); setDirty(true); renderClasses(); } }, icon('trash', 12)) : null));
    });
    const unnamed = (ds.classes || []).filter(isPlaceholderName).length;
    const many = unnamed > 0 && unnamed * 2 >= (ds.classes || []).length;
    dupesEl = h('div'); footDyn = h('span', { style: 'display:contents' }); footKey = '';
    saveBtn = h('button', { class: 'btn btn-sm btn-primary', onClick: saveClasses }, 'Save names');
    put(classesBox,
      many ? h('div', { class: 'callout small mb-12' }, icon('info', 14), h('span', `${unnamed === (ds.classes || []).length ? 'These classes have' : `${unnamed} of ${(ds.classes || []).length} classes have`} no real names yet. `,
        'Use ', h('span', { class: 'inline-ico', 'aria-label': 'Examples' }, icon('images', 12)), ' Examples next to a class to see its boxes, or let a vision model suggest names — you review every suggestion before it is applied.')) : null,
      h('div', { class: 'class-list-scroll' }, rows.length ? rows : h('div', { class: 'small faint' }, 'No classes yet — add one, or they are inferred from the labels.')),
      dupesEl,
      h('div', { class: 'row wrap mt-12' },
        h('button', { class: 'btn btn-sm', onClick: () => { names.push(`class_${names.length}`); setDirty(true); renderClasses(); const ins = classesBox.querySelectorAll('.class-edit input'); const last = ins[ins.length - 1]; if (last) { last.focus(); last.select(); } } }, icon('plus', 12), 'Add class'),
        h('button', { class: 'btn btn-sm', 'data-tip': 'Paste every name at once: JSON (["person", "car"], {"0": "person"}, COCO categories), a data.yaml names: block, or one per line', onClick: openPasteNames }, icon('fileText', 12), 'Paste names'),
        h('span', { class: 'spacer' }), footDyn, saveBtn),
      h('div', { class: 'xs faint mt-8' }, 'Names are saved to the index and used by data.yaml exports. Class cards describe a class for AI providers (prompt, definition, include / exclude, example boxes).'));
    renderClassFoot();
    back();
  }
  /** What follows each keystroke: unsaved state, duplicates, the Save / Discard / Suggest buttons. */
  function renderClassFoot() {
    if (!footDyn) return;
    const dupes = [...new Set(names.map(n => n.trim()).filter((n, i, a) => n && a.indexOf(n) !== i))];
    const unnamed = (ds.classes || []).filter(isPlaceholderName).length;
    const many = unnamed > 0 && unnamed * 2 >= (ds.classes || []).length;
    saveBtn.disabled = !dirty || saving || dupes.length > 0;
    saveBtn.textContent = saving ? 'Saving…' : 'Save names';
    const key = JSON.stringify([dirty, saving, dupes, many, (ds.classes || []).length]);
    if (key === footKey) return;
    footKey = key;
    put(classesActions, (ds.classes || []).length ? h('button', { class: cls('btn btn-sm', many && 'btn-primary'), disabled: dirty, 'data-tip': dirty ? 'Save or discard your edits first' : 'A vision model looks at a few boxes of each class and proposes a name, a prompt and a definition',
      onClick: suggestNames }, icon('sparkles', 12), 'Suggest names with AI') : null);
    put(dupesEl, dupes.length ? h('div', { class: 'small warn-text mt-8' }, `Duplicate names: ${dupes.join(', ')}`) : null);
    put(footDyn,
      dirty ? h('span', { class: 'xs warn-text' }, 'Unsaved changes') : null,
      dirty ? h('button', { class: 'btn btn-sm btn-ghost', onClick: () => { names = [...(ds.classes || [])]; setDirty(false); renderClasses(); } }, 'Discard') : null);
  }
  function setDirty(v) {
    dirty = v; persistDraft();
    renderClassFoot();
  }
  /** Pasted names replace the list from row `at` on. Classes after them stay while they still have boxes (dropping
      one would turn its boxes into "Invalid class id"); unused ones at the end go. Nothing is saved until Save. */
  function applyPastedNames(list, at = 0) {
    const next = names.slice(0, at).concat(list);
    for (let i = next.length; i < names.length; i++) next.push(names[i]);
    while (next.length > at + list.length && boxCount(next.length - 1) === 0) next.pop();
    const kept = next.length - at - list.length;
    names = next; setDirty(true); renderClasses();
    ui.toast(`Pasted ${fmt.plural(list.length, 'class name')}${kept ? ` · ${fmt.plural(kept, 'later class', 'later classes')} kept (still ${kept === 1 ? 'has' : 'have'} boxes)` : ''} — Save names to keep them`);
  }
  function openPasteNames() {
    const area = h('textarea', { class: 'input mono', rows: 10, spellcheck: 'false', style: 'width:100%', 'aria-label': 'Class names',
      placeholder: '["person", "car", "dog"]\n{"0": "person", "1": "car"}\nnames:\n  0: person\nor one name per line' });
    const preview = h('div', { class: 'col gap-6', 'aria-live': 'polite' });
    let parsed = { names: [] }, okBtn;
    const update = () => {
      parsed = parseClassNames(area.value);
      const dupes = duplicateNames(parsed.names);
      okBtn.disabled = !parsed.names.length || !!parsed.error || dupes.length > 0;
      if (!area.value.trim()) { put(preview, h('div', { class: 'small faint' }, 'Accepts a JSON list or {id: name} map, {"names": …}, COCO "categories", a data.yaml names: block, a Python list, or names one per line / comma-separated. Class ids follow the order (or the ids given).')); return; }
      if (parsed.error) { put(preview, h('div', { class: 'small warn-text' }, parsed.error)); return; }
      const fmtName = { json: 'JSON', yaml: 'YAML', lines: 'one per line', commas: 'comma-separated' }[parsed.format] || '';
      put(preview,
        h('div', { class: 'small muted' }, `${fmt.plural(parsed.names.length, 'class', 'classes')}${fmtName ? ` · read as ${fmtName}` : ''}`),
        h('div', { class: 'row wrap gap-4' }, parsed.names.slice(0, 60).map((n, i) => ui.chip(`${i} ${n}`, { cls: 'outline' })), parsed.names.length > 60 ? h('span', { class: 'small faint' }, `+${parsed.names.length - 60} more`) : null),
        dupes.length ? h('div', { class: 'small warn-text' }, `Duplicate names: ${dupes.join(', ')}`) : null);
    };
    area.addEventListener('input', update);
    ui.modal({ title: 'Paste class names', size: 'lg', initialFocus: area,
      body: h('div', { class: 'col gap-12' }, area, preview,
        h('div', { class: 'xs faint' }, 'Replaces the names from class 0 on. Classes after the pasted ones keep their names while they still have boxes. Review, then Save names.')),
      footer: (a) => [h('button', { class: 'btn', onClick: () => a.close() }, 'Cancel'),
        okBtn = h('button', { class: 'btn btn-primary', disabled: true, onClick: () => { a.close(); applyPastedNames(parsed.names, 0); } }, 'Use these names')] });
    update();
  }
  async function saveClasses() {
    if (!dirty || saving) return;
    const vals = names.map((n, i) => n.trim() || `class_${i}`);
    if (new Set(vals).size !== vals.length) { ui.toast('Two classes have the same name', { type: 'error' }); return; }
    saving = true; renderClassFoot();
    try {
      await api.patch(`/api/datasets/${enc}`, { classes: vals });
      ds = await refresh();
      names = [...(ds.classes || vals)]; saving = false; setDirty(false);
      renderClasses();                          // trimmed / filled-in names, rows no longer marked
      ui.toast('Class names saved', { type: 'ok' });
      draw();
    } catch (e) { saving = false; renderClassFoot(); ui.toast(`Class names not saved: ${fovea.plainError(e)}`, { type: 'error' }); }
  }
  /** "Suggest names with AI" — the dialog belongs to the AI tab (ai.js); without it, the AI tab opens it. */
  async function suggestNames() {
    if (dirty) { ui.toast('Save or discard your name edits first'); return; }
    let m = null;
    try { m = await import('./ai.js'); } catch (e) { console.error(e); }
    if (!alive) return;
    if (!m || typeof m.openSuggestNames !== 'function') { go(`/d/${enc}/ai?tab=cards&suggest=1`); return; }
    try {
      await m.openSuggestNames(fovea, ds, { onApplied: () => { if (!alive) return; loadCards(); redrawSoon(); } });
    } catch (e) { ui.toast(fovea.plainError(e), { type: 'error' }); }
  }
  async function loadCards() {
    try { const r = await api.get(`/api/datasets/${enc}/classes/cards`); if (alive) { cards = r.cards || {}; renderClasses(); } } catch (e) { /* the editor works without cards */ }
  }
  function editCard(i) {
    const cur = { description: '', include: '', exclude: '', prompt: '', examples: [], ...(cards[String(i)] || {}) };
    const nm = className(names, i);
    const ta = (k, label, ph, rows = 3) => { const t = h('textarea', { class: 'input', rows, placeholder: ph, 'aria-label': label }); t.value = cur[k] || ''; return [h('label', { class: 'label' }, label), t]; };
    // Field names and hints match AI → Class cards (ai.js), which edits the same card.
    const [lp, prompt] = ta('prompt', 'What should a model look for?', `A short noun phrase, e.g. “person lying on the ground” (default: “${nm}”)`, 2);
    const [ld, desc] = ta('description', 'Definition', 'One or two sentences that tell people and models what counts as this class.');
    const [li, inc] = ta('include', 'Box it when…', 'e.g. partly hidden people; people mid-fall', 2);
    const [le, exc] = ta('exclude', 'Don’t box…', 'e.g. mannequins, posters, reflections', 2);
    let examples = (cur.examples || []).slice();
    const exGrid = h('div', { class: 'ex-grid' });
    const pool = h('div', { class: 'ex-grid' });
    const exInfo = h('div', { class: 'xs faint' });
    const cropUrl = (e) => e.box ? `/api/crop/${e.image_id}?box=${e.box.map(v => (+v).toFixed(5)).join(',')}&pad=0.2&s=160` : null;
    const keyOf = (e) => `${e.image_id}:${e.row}`;
    const renderEx = () => {
      exInfo.textContent = `${examples.length} of 24 · click to remove · good examples (+) and look-alikes that are not this class (−)`;
      put(exGrid, ...(examples.length ? examples.map(e => h('button', { class: cls('ex', e.polarity < 0 ? 'neg' : 'on'), type: 'button', title: `${e.polarity < 0 ? 'Negative' : 'Positive'} example — click to remove`, onClick: () => { examples = examples.filter(x => keyOf(x) !== keyOf(e)); renderEx(); renderPool(); } },
        cropUrl(e) ? h('img', { src: cropUrl(e), alt: '', loading: 'lazy' }) : h('span', { class: 'xs' }, `#${e.image_id}`), h('span', { class: 'mark' }, e.polarity < 0 ? '−' : '+'))) : [h('div', { class: 'small faint' }, 'No examples yet.')]));
    };
    let poolItems = [];
    const renderPool = () => put(pool, ...poolItems.map(b => {
      const e = { image_id: b.image_id, row: b.row, box: [b.xc, b.yc, b.w, b.h] };
      const at = examples.find(x => keyOf(x) === keyOf(e));
      return h('button', { class: cls('ex', at && (at.polarity < 0 ? 'neg' : 'on')), type: 'button', title: `${b.rel_path} — click: positive example, ⇧click: negative`, onClick: (ev) => {
        if (at) examples = examples.filter(x => keyOf(x) !== keyOf(e));
        else if (examples.length < 24) examples.push({ ...e, polarity: ev.shiftKey ? -1 : 1 });
        else { ui.toast('A card holds at most 24 examples'); return; }
        renderEx(); renderPool();
      } }, h('img', { src: cropUrl(e), alt: '', loading: 'lazy' }), at ? h('span', { class: 'mark' }, at.polarity < 0 ? '−' : '+') : null);
    }));
    const loadPool = async (btn) => {
      btn.disabled = true;
      put(pool, h('div', { class: 'wait-row' }, ui.spinner()));
      try {
        const r = await api.get(`/api/datasets/${enc}/boxes`, { cls: i, sort: 'random', limit: 24, seed: Math.floor(Math.random() * 1e6) });
        poolItems = r.items || [];
        if (!poolItems.length) put(pool, h('div', { class: 'small faint' }, 'This class has no boxes to pick from.')); else renderPool();
      } catch (e) { put(pool, h('div', { class: 'small danger-text' }, fovea.plainError(e))); }
      btn.disabled = false; btn.lastChild.textContent = 'Show other boxes';
    };
    renderEx();
    const d = ui.drawer({ title: `Class card · ${i} ${nm}`, width: 520,
      body: h('div', { class: 'col gap-12' },
        h('div', { class: 'small muted' }, 'Models use the first field (or the class name) to find this class, and the definition with “Box it when…” / “Don’t box…” to decide borderline cases. The same card as AI → Class cards; saved with this dataset.'),
        h('div', { class: 'field' }, lp, prompt), h('div', { class: 'field' }, ld, desc), h('div', { class: 'fields cols-2' }, h('div', { class: 'field' }, li, inc), h('div', { class: 'field' }, le, exc)),
        h('div', { class: 'field' }, h('span', { class: 'label' }, 'Example boxes'), exInfo, exGrid),
        h('div', { class: 'field' }, h('div', { class: 'row' }, h('span', { class: 'label grow' }, `Boxes of “${nm}” in this dataset`),
          boxCount(i) ? link(`/d/${enc}/objects?cls=${i}`, { class: 'btn btn-sm btn-ghost', 'data-tip': 'Every box of this class, as crops (Objects)', onClick: (e) => {
            const edited = [[prompt, 'prompt'], [desc, 'description'], [inc, 'include'], [exc, 'exclude']].some(([t, k]) => t.value !== (cur[k] || '')) || JSON.stringify(examples) !== JSON.stringify(cur.examples || []);
            if (!edited || e.metaKey || e.ctrlKey) return;
            e.preventDefault();
            ui.confirm({ title: 'Leave without saving the card?', message: 'Your changes to this class card are not saved yet.', okLabel: 'Leave', danger: true }).then(ok => { if (ok) go(`/d/${enc}/objects?cls=${i}`); });
          } }, icon('external', 12), 'View all') : null,
          h('button', { class: 'btn btn-sm', onClick: (e) => loadPool(e.currentTarget) }, icon('images', 12), h('span', 'Pick examples'))), pool),
        cur.updated_at ? h('div', { class: 'xs faint' }, `Last saved ${fmt.date(cur.updated_at)}`) : null),
      footer: (m) => [h('button', { class: 'btn', onClick: () => m.close() }, 'Cancel'), h('button', { class: 'btn btn-primary', onClick: async (ev) => {
        const btn = ev.currentTarget;            // currentTarget is null once the await below returns
        btn.disabled = true;
        try {
          const r = await api.put(`/api/datasets/${enc}/classes/${i}/card`, { prompt: prompt.value, description: desc.value, include: inc.value, exclude: exc.value, examples: examples.map(e => ({ image_id: e.image_id, row: e.row, polarity: e.polarity, box: e.box })) });
          cards[String(i)] = r.card || r;
          fovea.bus.emit('ai:cards', { dataset_id: ds.id, cls: i });
          ui.toast('Class card saved', { type: 'ok' });
          m.close(); renderClasses();
        } catch (e) { btn.disabled = false; ui.toast(fovea.plainError(e), { type: 'error' }); }
      } }, 'Save card')],
      initialFocus: prompt });
    return d;
  }

  function pick(o, keys) { const r = {}; for (const k of keys) if (o[k]) r[k] = o[k]; return r; }

  // ------------------------------------------------------------------ lifecycle
  await draw();
  loadQA(); loadAudit(); loadHistory(); loadCards();
  const redrawSoon = fovea.debounce(() => { if (alive) draw(); }, 400);
  const offs = [
    fovea.bus.on('dataset:scanned', (id) => { if (id === ds.id) { ds = fovea.state.get().currentDataset || ds; if (!dirty) names = [...(ds.classes || [])]; draw(); loadQA(); loadHistory(); } }),
    fovea.bus.on('dataset:updated', (d) => { if (d && d.id === ds.id) { const changed = JSON.stringify(d.classes) !== JSON.stringify(ds.classes); ds = { ...ds, ...d }; if (changed && !dirty) { names = [...(ds.classes || [])]; } renderDataset(); if (changed) renderClasses(); } }),
    fovea.bus.on('review:changed', redrawSoon),
    fovea.bus.on('labels:changed', redrawSoon),
    fovea.bus.on('snapshots:changed', (e) => { if (!e || e.dataset === ds.id) loadHistory(); }),
    fovea.bus.on('ai:cards', (e) => { if (e && e.dataset_id === ds.id) loadCards(); }),
    fovea.bus.on('job:finished', (e) => {
      if (!e || e.dataset !== ds.id) return;
      if (e.kind === 'qa') loadQA();
      if (e.kind === 'snapshot' || e.kind === 'restore' || e.kind.startsWith('ai_')) loadHistory();
      if (e.kind === 'ai_audit') loadAudit();
      if (e.kind === 'restore' || e.kind === 'ai_accept' || e.kind === 'ai_batch') redrawSoon();
    }),
  ];
  // Jobs this page did not start (an audit launched from the AI tab, QA after a rescan) end without a banner
  // watcher here: notice them through the shell's job poll.
  let lastKinds = new Set(fovea.state.get().jobs.filter(j => j.dataset_id === ds.id).map(j => j.kind));
  offs.push(fovea.state.subscribe((s) => {
    const now = new Set(s.jobs.filter(j => j.dataset_id === ds.id).map(j => j.kind));
    for (const k of lastKinds) if (!now.has(k)) {
      if (k === 'qa') loadQA();
      if (k === 'ai_audit') loadAudit();
      if (k === 'snapshot' || k === 'restore' || k.startsWith('ai_')) loadHistory();
    }
    if (now.has('qa') && !lastKinds.has('qa')) loadQA();
    // Another qa job took over (the queued one started, a second run was queued): follow it at once.
    else if (qaSum) { const qj = qaJobNow(); if (qj && qj.id !== qaLive.id) renderQA(); }
    lastKinds = now;
  }, (s) => s.jobs));
  return () => {
    alive = false; offs.forEach(f => f()); window.removeEventListener('beforeunload', onUnload);
    stopQaWatch(); clearTimeout(qaRetry); restoreWatches.forEach(w => w.stop && w.stop());
    if (dirty) ui.toast('Unsaved class names are kept — they come back when you return to Overview.');
  };
}
