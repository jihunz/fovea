// Overview tab: dataset card + health check.
export function install(fovea) { fovea.registerTab({ id: 'overview', label: 'Overview', icon: 'layout', order: 10, render }); }

async function render(el, { dataset, fovea, refresh }) {
  const { h, icon, ui, fmt, link } = fovea;
  const { classColor, className } = fovea.colors;
  const page = h('div', { class: 'page' });
  const inner = h('div', { class: 'page-inner' });
  page.appendChild(inner); el.appendChild(page);
  let ds = dataset;
  const explore = (q) => `/d/${ds.id}/explore?${new URLSearchParams(q).toString()}`;

  async function draw() {
    inner.innerHTML = '';
    if (ds.status !== 'ready' && ds.image_count === 0) {
      inner.appendChild(ui.emptyState({ icon: 'refresh', title: ds.status === 'scanning' ? 'Indexing dataset…' : 'Not indexed yet', message: 'Statistics appear as soon as the index is built.' }));
      return;
    }
    let stats;
    try { ({ stats } = await fovea.api.get(`/api/datasets/${ds.id}/stats`)); }
    catch (e) { inner.appendChild(ui.emptyState({ icon: 'alert', title: 'Could not load stats', message: e.message })); return; }
    const t = stats.totals; const rv = stats.reviews; const reviewed = rv.approved + rv.flagged + rv.excluded;
    const names = ds.classes || [];

    // ---- stat tiles
    inner.appendChild(h('div', { class: 'stat-grid mb-16' },
      ui.statTile({ label: 'Images', value: fmt.num(t.images), sub: `${t.sequences} sequences`, icon: 'images', onClick: () => fovea.router.navigate(explore({})) }),
      ui.statTile({ label: 'Labeled', value: fmt.pct(t.labeled, t.images), sub: `${fmt.num(t.labeled)} with label file`, icon: 'tag', onClick: () => fovea.router.navigate(explore({ labeled: '1' })) }),
      ui.statTile({ label: 'Unlabeled', value: fmt.num(t.unlabeled), sub: `${fmt.num(t.empty)} empty label files`, icon: 'alertCircle', tone: t.unlabeled ? 'warn' : null, onClick: () => fovea.router.navigate(explore({ labeled: '0' })) }),
      ui.statTile({ label: 'Boxes', value: fmt.num(t.boxes), sub: `${t.avg_boxes} per labeled image`, icon: 'square' }),
      ui.statTile({ label: 'Classes', value: String(names.length), sub: stats.classes.some(c => c.unknown) ? 'unknown ids present' : (ds.classes_source === 'inferred' ? 'inferred from labels' : `from ${ds.classes_source}`), icon: 'tags', tone: stats.classes.some(c => c.unknown) ? 'warn' : null }),
      ui.statTile({ label: 'Issues', value: fmt.num(t.with_issues), sub: 'images with problems', icon: 'alert', tone: t.with_issues ? 'danger' : 'ok', onClick: () => fovea.router.navigate(explore({ issue: 'any' })) }),
      ui.statTile({ label: 'Reviewed', value: fmt.pct(reviewed, t.images), sub: `${fmt.num(rv.approved)} ✓ · ${fmt.num(rv.flagged)} ⚑ · ${fmt.num(rv.excluded)} ✕`, icon: 'checkCircle', onClick: () => fovea.router.navigate(explore({ review: 'any' })) }),
    ));

    const grid = h('div', { class: 'ov-grid' });
    inner.appendChild(grid);

    // ---- class distribution
    const maxBoxes = Math.max(1, ...stats.classes.map(c => c.boxes));
    const splitsPresent = stats.splits.map(s => s.split);
    const splitColor = (s) => ({ train: 'var(--accent)', val: 'var(--info)', test: 'var(--warn)', '': 'var(--accent)' })[s] || 'var(--text-3)';
    grid.appendChild(card('Class distribution', `${fmt.num(t.boxes)} boxes`, h('div', null,
      stats.classes.length ? h('div', { class: 'hbars' }, stats.classes.map(c => h('div', { class: 'hbar clickable', onClick: () => fovea.router.navigate(explore({ cls: c.cls })) },
        h('div', { class: 'name' }, h('span', { class: 'swatch', style: `background:${classColor(c.cls)}` }), h('span', { title: c.name }, `${c.cls} · ${c.name}`), c.unknown ? ui.chip('unknown', { type: 'warn' }) : null),
        h('div', { class: 'track' }, splitsPresent.map(s => h('i', { style: `width:${100 * (c.by_split[s] || 0) / maxBoxes}%;background:${splitColor(s)};opacity:${s === 'train' || s === '' ? 1 : .75}`, 'data-tip': `${s || 'all'}: ${fmt.num(c.by_split[s] || 0)}` }))),
        h('div', { class: 'val' }, `${fmt.num(c.boxes)} · ${fmt.num(c.images)} img`)))) : h('div', { class: 'faint' }, 'No boxes yet'),
      splitsPresent.length > 1 ? h('div', { class: 'legend mt-12' }, splitsPresent.map(s => h('span', h('span', { class: 'swatch', style: `background:${splitColor(s)}` }), s || 'all'))) : null,
    ), 'span-8'));

    // ---- health
    const issueRows = stats.issues.map(i => h('div', { class: 'issue-row' }, icon(ISSUE_ICON[i.code] || 'alert', 14), h('span', i.label), h('span', { class: 'n' }, fmt.num(i.count)),
      link(explore({ issue: i.code }), { class: 'btn btn-sm' }, 'Review')));
    const imbalance = stats.classes.filter(c => c.boxes > 0 && c.boxes < maxBoxes * 0.05 && !c.unknown);
    grid.appendChild(card('Health check', t.with_issues ? `${fmt.num(t.with_issues)} images need attention` : 'No issues found', h('div', null,
      issueRows.length ? issueRows : h('div', { class: 'callout ok small' }, icon('checkCircle', 14), 'All label files parsed cleanly.'),
      imbalance.length ? h('div', { class: 'callout warn small mt-12' }, icon('barChart', 14), h('span', `Under-represented: ${imbalance.map(c => c.name).join(', ')} (< 5% of the largest class)`)) : null,
      h('div', { class: 'divider' }),
      h('div', { class: 'small strong muted mb-8' }, 'Review progress'),
      h('div', { class: 'review-bar', style: 'height:8px' }, h('i', { class: 'a', style: `width:${100 * rv.approved / Math.max(1, t.images)}%` }), h('i', { class: 'f', style: `width:${100 * rv.flagged / Math.max(1, t.images)}%` }), h('i', { class: 'x', style: `width:${100 * rv.excluded / Math.max(1, t.images)}%` })),
      h('div', { class: 'legend mt-8' }, h('span', h('span', { class: 'swatch', style: 'background:var(--approve)' }), `Approved ${fmt.num(rv.approved)}`), h('span', h('span', { class: 'swatch', style: 'background:var(--flag)' }), `Flagged ${fmt.num(rv.flagged)}`), h('span', h('span', { class: 'swatch', style: 'background:var(--exclude)' }), `Excluded ${fmt.num(rv.excluded)}`), h('span', { class: 'faint' }, `${fmt.num(t.images - reviewed)} unreviewed`)),
      h('div', { class: 'row mt-12' }, link(explore({ review: 'none' }), { class: 'btn btn-sm' }, icon('eye', 13), 'Review unreviewed'), link(explore({ review: 'flagged' }), { class: 'btn btn-sm' }, icon('flag', 13), 'Flagged')),
    ), 'span-4'));

    // ---- splits
    grid.appendChild(card('Splits', `${stats.splits.length} split${stats.splits.length === 1 ? '' : 's'}`, h('table', { class: 'table' },
      h('thead', h('tr', h('th', 'Split'), h('th', { class: 'right' }, 'Images'), h('th', { class: 'right' }, 'Labeled'), h('th', { class: 'right' }, 'Boxes'), h('th', { class: 'right' }, 'Boxes/img'))),
      h('tbody', stats.splits.map(s => h('tr', { class: 'clickable', onClick: () => fovea.router.navigate(explore({ split: s.split })) }, h('td', h('span', { class: 'swatch', style: `background:${splitColor(s.split)};margin-right:6px` }), s.split || h('span', { class: 'faint' }, '(no split)')),
        h('td', { class: 'num' }, fmt.num(s.images)), h('td', { class: 'num' }, fmt.pct(s.labeled, s.images)), h('td', { class: 'num' }, fmt.num(s.boxes)), h('td', { class: 'num' }, s.labeled ? (s.boxes / s.labeled).toFixed(2) : '–')))),
    ), 'span-4'));

    // ---- boxes per image
    const bpiMax = Math.max(1, ...stats.boxes_per_image.map(b => b.count));
    grid.appendChild(card('Boxes per image', 'labeled images', h('div', { class: 'vbars' }, stats.boxes_per_image.map(b => h('div', { class: 'vb', 'data-tip': `${b.k}${b.k === 10 ? '+' : ''} boxes: ${fmt.num(b.count)} images` }, h('span', { class: 'n' }, fmt.compact(b.count)), h('i', { style: `height:${Math.max(2, 100 * b.count / bpiMax)}%` }), h('span', { class: 'l' }, b.k === 10 ? '10+' : String(b.k))))), 'span-4'));

    // ---- box size + aspect
    const areaMax = Math.max(1, ...stats.box_area.map(b => b.count));
    const aspMax = Math.max(1, ...stats.box_aspect.map(b => b.count));
    grid.appendChild(card('Box size & shape', 'relative to image area', h('div', null,
      h('div', { class: 'vbars', style: 'height:100px' }, stats.box_area.map(b => h('div', { class: 'vb', 'data-tip': `${b.label}: ${fmt.num(b.count)}` }, h('span', { class: 'n' }, fmt.compact(b.count)), h('i', { style: `height:${Math.max(2, 100 * b.count / areaMax)}%` }), h('span', { class: 'l' }, b.label)))),
      h('div', { class: 'small faint mt-12 mb-8' }, 'Aspect ratio (w/h)'),
      h('div', { class: 'vbars', style: 'height:80px' }, stats.box_aspect.map(b => h('div', { class: 'vb', 'data-tip': `${b.label}: ${fmt.num(b.count)}` }, h('i', { style: `height:${Math.max(2, 100 * b.count / aspMax)}%;background:var(--info)` }), h('span', { class: 'l' }, b.label)))),
    ), 'span-4'));

    // ---- heatmap
    const hm = stats.heatmap;
    grid.appendChild(card('Box center heatmap', 'where objects appear', h('div', { class: 'heat', style: `grid-template-columns:repeat(${hm.n},1fr)` },
      hm.cells.flatMap((row, y) => row.map((v, x) => h('i', { style: `opacity:${hm.max ? (0.06 + 0.94 * v / hm.max) : 0.06}`, 'data-tip': `${fmt.num(v)} boxes` })))), 'span-4'));

    // ---- dimensions
    grid.appendChild(card('Image dimensions', `${stats.dimensions_distinct} distinct`, h('table', { class: 'table' }, h('tbody', stats.dimensions.map(d => h('tr', h('td', { class: 'mono' }, `${d.width} × ${d.height}`), h('td', { class: 'num' }, fmt.num(d.count)), h('td', { style: 'width:40%' }, h('div', { class: 'bar' }, h('i', { style: `width:${100 * d.count / Math.max(1, t.images)}%` }))))))), 'span-4'));

    // ---- dataset info & classes editor
    const lay = ds.layout || {};
    const classInputs = [];
    const classEditor = h('div', { class: 'col gap-4' });
    const renderClasses = () => {
      classEditor.innerHTML = '';
      names.forEach((n, i) => { const inp = h('input', { class: 'input input-sm', value: n }); classInputs[i] = inp; classEditor.appendChild(h('div', { class: 'class-edit' }, h('span', { class: 'idx' }, i), h('span', { class: 'swatch', style: `background:${classColor(i)}` }), inp)); });
    };
    renderClasses();
    const saveClasses = async () => {
      const vals = classInputs.slice(0, names.length).map((inp, i) => inp.value.trim() || `class_${i}`);
      await fovea.api.patch(`/api/datasets/${ds.id}/`.replace(/\/$/, ''), { classes: vals });
      ds = await refresh(); ui.toast('Class names saved', { type: 'ok' }); draw();
    };
    grid.appendChild(card('Dataset', lay.kind, h('div', { class: 'ov-grid' },
      h('div', { class: 'span-6' },
        h('table', { class: 'table fixed' }, h('colgroup', h('col', { style: 'width:70px' }), h('col'), h('col')), h('thead', h('tr', h('th', 'Split'), h('th', 'Images'), h('th', 'Labels'))), h('tbody', (lay.sources || []).map(s => h('tr', h('td', s.split || '—'), h('td', { class: 'mono small', title: s.list_file_host || s.img_dir_host }, h('div', { class: 'truncate' }, s.list_file_host || s.img_dir_host)), h('td', { class: 'mono small', title: s.label_dir_host }, h('div', { class: 'truncate' }, s.list_file ? 'images→labels rule' : (s.label_dir_host + (s.label_dir_exists ? '' : ' (will be created)')))))))),
        h('div', { class: 'row mt-12 small muted' }, icon('fileText', 14), lay.data_yaml_host ? h('span', { class: 'mono truncate grow', title: lay.data_yaml_host }, lay.data_yaml_host) : h('span', { class: 'grow' }, 'No data.yaml'),
          h('button', { class: 'btn btn-sm', onClick: async () => { try { const r = await fovea.api.post(`/api/datasets/${ds.id}/export/yaml`); ui.toast(`Wrote ${r.path}`, { type: 'ok' }); ds = await refresh(); draw(); } catch (e) { ui.toast(e.message, { type: 'error' }); } } }, icon('save', 13), lay.data_yaml ? 'Rewrite data.yaml' : 'Write data.yaml')),
        h('div', { class: 'row mt-8 small muted' }, icon('clock', 14), `Indexed ${fmt.date(ds.scanned_at)} · registered ${fmt.date(ds.created_at)}`)),
      h('div', { class: 'span-6' },
        h('div', { class: 'row mb-8' }, h('span', { class: 'small strong muted' }, `Classes (${names.length})`), ui.chip(ds.classes_source, { cls: 'outline' }), h('span', { class: 'spacer' }),
          h('button', { class: 'btn btn-sm', onClick: () => { names.push(`class_${names.length}`); renderClasses(); } }, icon('plus', 12), 'Add'),
          h('button', { class: 'btn btn-sm btn-primary', onClick: saveClasses }, 'Save names')),
        classEditor),
    ), 'span-12'));
  }

  function card(title, sub, body, span) { return h('div', { class: `card ${span}` }, h('div', { class: 'card-header' }, h('h3', title), sub ? h('span', { class: 'sub' }, sub) : null), h('div', { class: 'card-body' }, body)); }

  await draw();
  const off = fovea.bus.on('dataset:scanned', async (id) => { if (id === ds.id) { ds = fovea.state.get().currentDataset || ds; draw(); } });
  const off2 = fovea.bus.on('review:changed', () => draw());
  return () => { off(); off2(); };
}

const ISSUE_ICON = { missing_label: 'fileSearch', empty_label: 'file', bad_format: 'alert', bad_class: 'tag', out_of_range: 'maximize', zero_area: 'minus', tiny_box: 'zoomIn', duplicate_box: 'copy', image_unreadable: 'image' };
