// Home: dataset registry cards + Add dataset wizard + file browser picker.
let F;
export function install(fovea) { F = fovea; F.openAddDataset = openAddDataset; F.pickPath = pickPath; }

export async function render(ctx, fovea) {
  const { h, icon, logo, fmt, ui, link } = fovea;
  const page = h('div', { class: 'page' });
  const inner = h('div', { class: 'page-inner' });
  page.appendChild(inner);
  fovea.setView('home', page);
  let datasets = fovea.state.get().datasets;
  const search = h('input', { class: 'input', placeholder: 'Filter datasets…' });
  const grid = h('div', { class: 'ds-cards' });

  const renderGrid = () => {
    const q = search.value.trim().toLowerCase();
    const list = datasets.filter(d => !q || (d.name + ' ' + d.id + ' ' + d.root_host).toLowerCase().includes(q));
    fovea.mount ? null : null;
    grid.innerHTML = '';
    grid.appendChild(h('div', { class: 'ds-card add-card', onClick: openAddDataset }, icon('plus', 28), h('div', { class: 'strong' }, 'Add dataset'), h('div', { class: 'small' }, 'Folder, data.yaml or train.txt')));
    list.forEach(d => grid.appendChild(card(d)));
  };

  function card(d) {
    const rv = d.review || {}; const total = d.image_count || 0;
    const pct = (n) => total ? (100 * n / total) + '%' : '0%';
    const reviewed = (rv.approved || 0) + (rv.flagged || 0) + (rv.excluded || 0);
    const kind = (d.layout && d.layout.kind) || '';
    const cover = d.cover_image_id ? h('img', { src: fovea.api.thumbUrl(d.cover_image_id, 512), loading: 'lazy', alt: '' }) : h('div', { class: 'row', style: 'height:100%;justify-content:center;color:#666' }, icon('images', 28));
    const el = h('div', { class: 'ds-card', onClick: () => fovea.router.navigate(`/d/${d.id}`) },
      h('div', { class: 'cover' }, cover, ui.chip(kind, { cls: 'kind', type: 'accent' }),
        h('button', { class: 'btn btn-sm btn-icon menu-btn', onClick: (e) => { e.stopPropagation(); cardMenu(e.currentTarget, d); } }, icon('moreH', 14))),
      h('div', { class: 'body' },
        h('div', { class: 'name' }, d.name, h('span', { class: `status-pill` }, h('span', { class: `dot ${d.status}` }))),
        h('div', { class: 'path truncate', title: d.root_host }, d.root_host),
        d.status === 'scanning' || (d.job && d.job.status === 'running') ? ui.progress(d.job ? d.job.progress : 0, { indeterminate: !d.job || !d.job.total }) : null,
        h('div', { class: 'meta' }, h('span', h('b', fmt.num(total)), ' images'), h('span', h('b', fmt.num(d.label_count)), ' labeled'), h('span', h('b', fmt.num(d.box_count)), ' boxes'), h('span', h('b', (d.classes || []).length), ' classes')),
        h('div', { class: 'review-bar', 'data-tip': `Approved ${rv.approved || 0} · Flagged ${rv.flagged || 0} · Excluded ${rv.excluded || 0}` }, h('i', { class: 'a', style: `width:${pct(rv.approved || 0)}` }), h('i', { class: 'f', style: `width:${pct(rv.flagged || 0)}` }), h('i', { class: 'x', style: `width:${pct(rv.excluded || 0)}` })),
        h('div', { class: 'foot' }, h('span', `${fmt.pct(reviewed, total)} reviewed`), h('span', '·'), h('span', (d.splits || []).filter(Boolean).join(' / ') || 'no splits'), h('span', { class: 'spacer' }), h('span', fmt.ago(d.opened_at || d.created_at))),
      ));
    return el;
  }

  function cardMenu(anchor, d) {
    ui.menu(anchor, [
      { label: 'Open', icon: 'external', onClick: () => fovea.router.navigate(`/d/${d.id}`) },
      { label: 'Explore', icon: 'grid', onClick: () => fovea.router.navigate(`/d/${d.id}/explore`) },
      { label: 'Annotate', icon: 'pen', onClick: () => fovea.router.navigate(`/d/${d.id}/annotate`) },
      { sep: true },
      { label: 'Rescan', icon: 'refresh', onClick: () => rescan(d.id, false) },
      { label: 'Re-detect layout & rescan', icon: 'scan', onClick: () => rescan(d.id, true) },
      { label: 'Copy path', icon: 'copy', onClick: () => ui.copyText(d.root_host) },
      { sep: true },
      { label: 'Remove from Fovea', icon: 'trash', danger: true, onClick: async () => {
        if (await ui.confirm({ title: `Remove “${d.name}”?`, message: 'Only the Fovea index and review marks are removed. Files on disk are untouched.', okLabel: 'Remove', danger: true })) {
          await fovea.api.del(`/api/datasets/${d.id}`); ui.toast('Dataset removed'); datasets = await fovea.loadDatasets(); renderGrid();
        } } },
    ]);
  }

  async function rescan(id, redetect) {
    try { await fovea.api.post(`/api/datasets/${id}/scan`, { redetect }); ui.toast('Scan started'); datasets = await fovea.loadDatasets(); renderGrid(); }
    catch (e) { ui.toast(e.message, { type: 'error' }); }
  }

  const unsub = fovea.state.subscribe(s => { datasets = s.datasets; renderGrid(); });
  const cleanup = () => unsub();
  fovea.setView('home', page, cleanup);

  inner.appendChild(datasets.length ? h('div') : h('div', { class: 'hero' }, logo(56), h('div', null, h('h2', 'Welcome to Fovea'), h('p', 'A local-first workspace for reviewing and labeling computer-vision datasets. Register a YOLO-style folder to index images, labels and boxes, then explore, review and annotate — no upload, no accounts.'))));
  inner.appendChild(h('div', { class: 'row mb-16' }, h('div', null, h('h1', { class: 'page-title' }, 'Datasets'), h('div', { class: 'page-sub' }, `${datasets.length} registered · index stored locally`)), h('span', { class: 'spacer' }),
    h('div', { class: 'input-wrap', style: 'width:240px' }, icon('search'), search), h('button', { class: 'btn btn-primary', onClick: openAddDataset }, icon('plus', 14), 'Add dataset')));
  inner.appendChild(grid);
  search.addEventListener('input', renderGrid);
  renderGrid();
}

// ------------------------------------------------------------------ file browser picker
export function pickPath({ title = 'Choose a folder', files = true, start = '', accept = null, selectFiles = true } = {}) {
  const { h, icon, ui, fmt } = F;
  return new Promise((resolve) => {
    let current = start || '';
    let selected = null;
    const pathInput = h('input', { class: 'input mono', placeholder: '/path/to/folder', value: current });
    const list = h('div', { class: 'fb-list' });
    const status = h('div', { class: 'small faint' });
    const load = async (p) => {
      list.innerHTML = ''; list.appendChild(h('div', { class: 'row', style: 'padding:14px;justify-content:center' }, ui.spinner()));
      try {
        const data = await F.api.get('/api/fs/browse', { path: p, files: files ? 1 : 0 });
        current = data.current; pathInput.value = current; selected = null;
        list.innerHTML = '';
        if (data.parent) list.appendChild(h('div', { class: 'fb-item dir', onClick: () => load(data.parent) }, icon('cornerUpLeft'), h('span', '..')));
        data.dirs.forEach(d => list.appendChild(h('div', { class: 'fb-item dir', onClick: () => load(d.path) }, icon('folder'), h('span', d.name))));
        if (files) data.files.forEach(f => {
          const ok = !accept || accept.some(ext => f.name.toLowerCase().endsWith(ext));
          const row = h('div', { class: `fb-item ${ok ? '' : 'faint'}`, onClick: () => { if (selectFiles && ok) { done(f.path); } }, onDblclick: () => { if (selectFiles && ok) done(f.path); } },
            icon(f.is_image ? 'image' : 'fileText'), h('span', f.name), h('span', { class: 'size' }, fmt.bytes(f.size)));
          list.appendChild(row);
        });
        status.textContent = `${data.dirs.length} folders · ${data.files.length} files`;
      } catch (e) { list.innerHTML = ''; list.appendChild(h('div', { class: 'empty' }, h('p', e.message))); }
    };
    let api;
    const done = (p) => { api.close(); resolve(p); };
    api = ui.modal({
      title, size: 'lg', onClose: () => resolve(null),
      body: h('div', { class: 'fb' },
        h('div', { class: 'fb-path' }, icon('folderOpen', 16), pathInput, h('button', { class: 'btn', onClick: () => load(pathInput.value.trim()) }, 'Go')),
        list, status),
      footer: (a) => [h('span', { class: 'small faint grow' }, files && selectFiles ? 'Click a file to select it, or use the current folder.' : ''), h('button', { class: 'btn', onClick: () => a.close() }, 'Cancel'), h('button', { class: 'btn btn-primary', onClick: () => done(current) }, 'Use this folder')],
    });
    pathInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); load(pathInput.value.trim()); } });
    load(current);
  });
}

// ------------------------------------------------------------------ add dataset wizard
export function openAddDataset(prefillPath = '') {
  const { h, icon, ui, fmt } = F;
  let detected = null;
  const pathInput = h('input', { class: 'input mono', placeholder: '/data/my-dataset  ·  data.yaml  ·  train.txt', value: prefillPath });
  const nameInput = h('input', { class: 'input', placeholder: 'Dataset name' });
  const idInput = h('input', { class: 'input mono', placeholder: 'dataset-id' });
  const result = h('div');
  const detectBtn = h('button', { class: 'btn', onClick: () => detect() }, icon('scan', 14), 'Detect');
  let registerBtn;

  async function detect() {
    const p = pathInput.value.trim(); if (!p) return;
    detectBtn.disabled = true; result.innerHTML = ''; result.appendChild(h('div', { class: 'row small faint' }, ui.spinner(), 'Detecting layout…'));
    try {
      const res = await F.api.get('/api/fs/detect', { path: p });
      detected = res;
      const lay = res.layout;
      if (!nameInput.value) nameInput.value = res.suggested_name;
      if (!idInput.value) idInput.value = res.suggested_id;
      const bad = lay.kind === 'unknown' || !lay.sources.length;
      result.innerHTML = '';
      result.appendChild(h('div', { class: 'detect-box' },
        h('div', { class: 'row' }, ui.chip(lay.kind, { type: bad ? 'danger' : 'accent' }), h('span', { class: 'small muted' }, LAYOUT_DESC[lay.kind] || ''), h('span', { class: 'spacer' }),
          lay.data_yaml ? ui.chip('data.yaml', { icon: 'fileText' }) : null),
        lay.sources.length ? h('table', { class: 'table' }, h('thead', h('tr', h('th', 'Split'), h('th', 'Images'), h('th', 'Labels'), h('th', 'Source'))),
          h('tbody', lay.sources.map(s => h('tr', h('td', s.split || h('span', { class: 'faint' }, '—')), h('td', { class: 'num' }, fmt.num(s.image_count)),
            h('td', { class: 'num' }, s.list_file ? 'auto' : (s.label_dir_exists ? fmt.num(s.label_count) : h('span', { class: 'faint' }, 'none'))),
            h('td', { class: 'mono truncate', style: 'max-width:260px', title: s.list_file_host || s.img_dir_host }, (s.list_file_host || s.img_dir_host).split('/').slice(-3).join('/')))))) : null,
        lay.classes.length ? h('div', { class: 'row wrap gap-4' }, h('span', { class: 'small muted' }, 'Classes:'), lay.classes.map((c, i) => ui.chip(`${i} ${c}`, { cls: 'outline' }))) : h('div', { class: 'small faint' }, 'No class names found — they will be inferred from labels (you can rename them later).'),
        lay.notes.map(n => h('div', { class: 'callout warn small' }, icon('info', 14), n)),
      ));
      registerBtn.disabled = bad;
    } catch (e) { result.innerHTML = ''; result.appendChild(h('div', { class: 'callout danger' }, icon('alertCircle'), e.message)); registerBtn.disabled = true; }
    finally { detectBtn.disabled = false; }
  }

  const m = ui.modal({
    title: 'Add dataset', size: 'lg',
    body: h('div', { class: 'col gap-16' },
      h('div', { class: 'field' }, h('label', { class: 'label' }, 'Path', h('span', { class: 'hint' }, '— folder, data.yaml or train.txt (host paths are mapped automatically)')),
        h('div', { class: 'row' }, pathInput, h('button', { class: 'btn', onClick: async () => { const p = await pickPath({ title: 'Choose dataset folder or file', files: true, start: pathInput.value.trim(), accept: ['.yaml', '.yml', '.txt'] }); if (p) { pathInput.value = p; detect(); } } }, icon('folderOpen', 14), 'Browse'), detectBtn)),
      result,
      h('div', { class: 'fields cols-2' }, h('div', { class: 'field' }, h('label', { class: 'label' }, 'Name'), nameInput), h('div', { class: 'field' }, h('label', { class: 'label' }, 'ID', h('span', { class: 'hint' }, 'used in URLs')), idInput)),
      h('div', { class: 'small faint' }, 'Registering builds a local index (images, labels, boxes, sizes). Nothing is uploaded or modified.'),
    ),
    footer: (api) => [h('button', { class: 'btn', onClick: () => api.close() }, 'Cancel'),
      registerBtn = h('button', { class: 'btn btn-primary', disabled: true, onClick: async () => {
        registerBtn.disabled = true;
        try {
          const res = await F.api.post('/api/datasets', { path: pathInput.value.trim(), name: nameInput.value.trim() || undefined, id: idInput.value.trim() || undefined });
          api.close(); ui.toast(`Registered “${res.dataset.name}” — indexing…`, { type: 'ok' });
          await F.loadDatasets(); F.router.navigate(`/d/${res.dataset.id}`);
        } catch (e) { ui.toast(e.message, { type: 'error' }); registerBtn.disabled = false; }
      } }, icon('database', 14), 'Register & index')],
  });
  pathInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); detect(); } });
  pathInput.addEventListener('change', () => detect());
  if (prefillPath) detect();
}

const LAYOUT_DESC = {
  yaml: 'Ultralytics data.yaml', ultralytics: 'images/<split> + labels/<split>', 'split-first': '<split>/images + <split>/labels',
  flat: 'images/ + labels/', mixed: 'images and labels in one folder', bare: 'images only (labels/ will be created)', list: 'train.txt image list', unknown: 'not recognised',
};
