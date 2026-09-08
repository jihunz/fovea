// Export tab: subsets (zip / copy), review lists, data.yaml.
export function install(fovea) { fovea.registerTab({ id: 'export', label: 'Export', icon: 'download', order: 50, render }); }

async function render(el, { dataset, fovea }) {
  const { h, icon, ui, fmt } = fovea;
  const { classColor } = fovea.colors;
  const ds = dataset; const names = ds.classes || [];
  const page = h('div', { class: 'page' }); const inner = h('div', { class: 'page-inner' }); page.appendChild(inner); el.appendChild(page);
  inner.appendChild(h('div', { class: 'mb-16' }, h('h1', { class: 'page-title' }, 'Export'), h('div', { class: 'page-sub' }, 'Build subsets, review lists and Ultralytics data.yaml from the index. Files on disk are never modified unless you write into the dataset root explicitly.')));
  const grid = h('div', { class: 'export-grid' }); inner.appendChild(grid);

  // ------------------------------------------------------------ subset builder
  const f = { split: new Set((ds.splits || []).filter(Boolean)), cls: new Set(), only: false, labeled: '', excludeReview: new Set(['excluded']), includeUnlabeled: true, resize: 'none', max: 1280, w: 640, hh: 640, yaml: true };
  const countEl = h('div', { class: 'preview-count' }, '…');
  const filtersOf = () => {
    const o = {};
    if (f.split.size && f.split.size !== (ds.splits || []).filter(Boolean).length) o.split = [...f.split].join(',');
    if (f.cls.size) { if (f.only) o.only_cls = [...f.cls].join(','); else o.cls = [...f.cls].join(','); }
    if (f.labeled) o.labeled = f.labeled;
    if (f.excludeReview.size) o.exclude_review = [...f.excludeReview].join(',');
    return o;
  };
  const refreshCount = fovea.debounce(async () => { try { const r = await fovea.api.get(`/api/datasets/${ds.id}/images`, { ...filtersOf(), limit: 1 }); countEl.textContent = `${fmt.num(r.total)} images`; } catch (e) { countEl.textContent = '?'; } }, 150);
  const chk = (label, checked, on) => h('label', { class: 'check' }, h('input', { type: 'checkbox', checked, onChange: (e) => { on(e.target.checked); refreshCount(); } }), label);
  const resizeOpts = h('div', { class: 'row wrap' });
  const renderResize = () => {
    resizeOpts.innerHTML = '';
    resizeOpts.appendChild(ui.seg([{ value: 'none', label: 'Original' }, { value: 'max', label: 'Max side' }, { value: 'exact', label: 'Exact' }], f.resize, v => { f.resize = v; renderResize(); }));
    if (f.resize === 'max') resizeOpts.appendChild(h('input', { class: 'input input-sm', type: 'number', style: 'width:90px', value: f.max, onChange: e => f.max = Number(e.target.value) }));
    if (f.resize === 'exact') { resizeOpts.appendChild(h('input', { class: 'input input-sm', type: 'number', style: 'width:80px', value: f.w, onChange: e => f.w = Number(e.target.value) })); resizeOpts.appendChild(h('span', '×')); resizeOpts.appendChild(h('input', { class: 'input input-sm', type: 'number', style: 'width:80px', value: f.hh, onChange: e => f.hh = Number(e.target.value) })); }
  };
  renderResize();
  const resizeSpec = () => f.resize === 'none' ? null : f.resize === 'max' ? { mode: 'max', size: f.max } : { mode: 'exact', width: f.w, height: f.hh };
  const jobBox = h('div');
  const subsetCard = card('Subset', 'images + labels in Ultralytics layout', h('div', { class: 'col gap-12' },
    (ds.splits || []).filter(Boolean).length ? h('div', { class: 'field' }, h('span', { class: 'label' }, 'Splits'), h('div', { class: 'row wrap' }, (ds.splits || []).filter(Boolean).map(s => chk(s, true, v => v ? f.split.add(s) : f.split.delete(s))))) : null,
    h('div', { class: 'field' }, h('span', { class: 'label' }, 'Classes', h('span', { class: 'hint' }, '— images containing any of')), h('div', { class: 'row wrap' }, names.length ? names.map((n, i) => chk(h('span', { class: 'row gap-4' }, h('span', { class: 'swatch', style: `background:${classColor(i)}` }), `${i} ${n}`), false, v => v ? f.cls.add(i) : f.cls.delete(i))) : h('span', { class: 'faint small' }, 'no classes')),
      h('div', { class: 'row' }, chk('Only these classes (no other class present)', false, v => f.only = v))),
    h('div', { class: 'field' }, h('span', { class: 'label' }, 'Label state'), h('select', { class: 'select', style: 'width:auto', onChange: e => { f.labeled = e.target.value; refreshCount(); } }, [['', 'Any'], ['1', 'Labeled with boxes'], ['nobox', 'No boxes'], ['0', 'No label file']].map(([v, l]) => h('option', { value: v }, l)))),
    h('div', { class: 'field' }, h('span', { class: 'label' }, 'Review'), h('div', { class: 'row wrap' }, chk('Skip excluded', true, v => v ? f.excludeReview.add('excluded') : f.excludeReview.delete('excluded')), chk('Skip flagged', false, v => v ? f.excludeReview.add('flagged') : f.excludeReview.delete('flagged')), chk('Include images without labels', true, v => f.includeUnlabeled = v))),
    h('div', { class: 'field' }, h('span', { class: 'label' }, 'Resize'), resizeOpts, h('span', { class: 'hint' }, 'Normalized YOLO labels stay valid after resizing. Resized images are re-encoded as JPEG.')),
    h('div', { class: 'field' }, h('div', { class: 'row' }, chk('Write data.yaml', true, v => f.yaml = v))),
    h('div', { class: 'divider' }),
    h('div', { class: 'row' }, countEl, h('span', { class: 'spacer' }),
      h('button', { class: 'btn', onClick: () => downloadZip() }, icon('download', 14), 'Download ZIP'),
      h('button', { class: 'btn btn-primary', onClick: () => copyTo() }, icon('folderOpen', 14), 'Copy to folder…')),
    jobBox,
  ));
  grid.appendChild(subsetCard);
  refreshCount();

  async function downloadZip() {
    const res = await fetch(`/api/datasets/${ds.id}/export/zip`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ filters: filtersOf(), resize: resizeSpec(), include_unlabeled: f.includeUnlabeled, include_yaml: f.yaml }) });
    if (!res.ok) { ui.toast('Export failed: ' + (await res.text()), { type: 'error' }); return; }
    const blob = await res.blob(); const a = h('a', { href: URL.createObjectURL(blob), download: `${ds.id}-subset.zip` }); document.body.appendChild(a); a.click(); a.remove();
    ui.toast(`ZIP ready (${fmt.bytes(blob.size)})`, { type: 'ok' });
  }
  async function copyTo() {
    const target = await fovea.pickPath({ title: 'Choose target folder (a new folder will be filled)', files: false, selectFiles: false });
    if (!target) return;
    try {
      const { job } = await fovea.api.post(`/api/datasets/${ds.id}/export/copy`, { filters: filtersOf(), target_dir: target, resize: resizeSpec(), include_unlabeled: f.includeUnlabeled, include_yaml: f.yaml });
      const bar = ui.progress(0); const txt = h('span', { class: 'small muted' }, 'Copying…');
      jobBox.innerHTML = ''; jobBox.appendChild(h('div', { class: 'col gap-4' }, txt, bar));
      const snap = await fovea.api.watchJob(job.id, (s) => { txt.textContent = s.message; bar.firstChild.style.width = `${Math.round(s.progress * 100)}%`; });
      txt.textContent = `Done: ${fmt.num(snap.result.images)} images, ${fmt.num(snap.result.labels)} labels → ${snap.result.target}`;
      ui.toast('Subset copied', { type: 'ok' });
    } catch (e) { ui.toast(e.message, { type: 'error' }); }
  }

  // ------------------------------------------------------------ lists
  const lf = { status: 'excluded', style: 'host', split: '' };
  const listCard = card('Review lists', 'plain-text path lists', h('div', { class: 'col gap-12' },
    h('div', { class: 'field' }, h('span', { class: 'label' }, 'Which images'), h('select', { class: 'select', onChange: e => lf.status = e.target.value }, [['excluded', 'Excluded'], ['flagged', 'Flagged'], ['approved', 'Approved'], ['none', 'Unreviewed'], ['any', 'All reviewed'], ['all', 'Every image (training list)'], ['keep', 'Every image except excluded (training list)']].map(([v, l]) => h('option', { value: v }, l)))),
    (ds.splits || []).filter(Boolean).length ? h('div', { class: 'field' }, h('span', { class: 'label' }, 'Split'), h('select', { class: 'select', onChange: e => lf.split = e.target.value }, [['', 'All splits'], ...(ds.splits || []).filter(Boolean).map(s => [s, s])].map(([v, l]) => h('option', { value: v }, l)))) : null,
    h('div', { class: 'field' }, h('span', { class: 'label' }, 'Path style'), ui.seg([{ value: 'host', label: 'Absolute (host)' }, { value: 'rel', label: 'Relative' }], lf.style, v => lf.style = v)),
    h('div', { class: 'row' }, h('span', { class: 'spacer' }),
      h('button', { class: 'btn', onClick: () => listAction(false) }, icon('download', 14), 'Download .txt'),
      h('button', { class: 'btn', onClick: () => listAction(true) }, icon('save', 14), 'Write into dataset root')),
  ));
  grid.appendChild(listCard);
  async function listAction(write) {
    const filters = {}; if (lf.split) filters.split = lf.split;
    if (lf.status === 'all') {} else if (lf.status === 'keep') filters.exclude_review = 'excluded'; else filters.review = lf.status;
    const filename = `${ds.id}-${lf.status}${lf.split ? '-' + lf.split : ''}`;
    try {
      if (write) { const r = await fovea.api.post(`/api/datasets/${ds.id}/export/list`, { filters, style: lf.style, write: true, filename }); ui.toast(`Wrote ${r.lines} lines → ${r.path}`, { type: 'ok' }); return; }
      const res = await fetch(`/api/datasets/${ds.id}/export/list`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ filters, style: lf.style, filename }) });
      const blob = await res.blob(); const a = h('a', { href: URL.createObjectURL(blob), download: filename + '.txt' }); document.body.appendChild(a); a.click(); a.remove();
    } catch (e) { ui.toast(e.message, { type: 'error' }); }
  }

  // ------------------------------------------------------------ yaml
  grid.appendChild(card('data.yaml', 'Ultralytics dataset config', h('div', { class: 'col gap-12' },
    h('pre', { class: 'mono small', style: 'margin:0;padding:10px;background:var(--surface-2);border-radius:6px;white-space:pre-wrap' }, previewYaml()),
    h('div', { class: 'row' }, h('span', { class: 'small faint grow' }, ds.layout && ds.layout.data_yaml_host ? `Current: ${ds.layout.data_yaml_host}` : 'No data.yaml in dataset root yet'),
      h('button', { class: 'btn', onClick: async () => { try { const r = await fovea.api.post(`/api/datasets/${ds.id}/export/yaml`); ui.toast(`Wrote ${r.path}`, { type: 'ok' }); } catch (e) { ui.toast(e.message, { type: 'error' }); } } }, icon('save', 14), 'Write data.yaml')))));

  function previewYaml() {
    const lines = [`path: ${ds.root_host}`];
    for (const s of (ds.layout && ds.layout.sources) || []) lines.push(`${s.split || 'train'}: ${s.list_file_host || s.img_dir_host}`);
    lines.push(`nc: ${names.length}`, 'names:'); names.forEach((n, i) => lines.push(`  ${i}: ${n}`));
    return lines.join('\n');
  }
  function card(title, sub, body) { return h('div', { class: 'card' }, h('div', { class: 'card-header' }, h('h3', title), h('span', { class: 'sub' }, sub)), h('div', { class: 'card-body' }, body)); }
}
