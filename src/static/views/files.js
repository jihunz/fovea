// Files tab: browse the dataset root with previews.
export function install(fovea) { fovea.registerTab({ id: 'files', label: 'Files', icon: 'folder', order: 40, render }); }

async function render(el, { ctx, dataset, fovea }) {
  const { h, icon, ui, fmt, cls } = fovea;
  const ds = dataset;
  const root = h('div', { class: 'files' });
  const top = h('div', { class: 'files-top' });
  const list = h('div', { class: 'files-list' });
  const preview = h('div', { class: 'files-preview' }, h('div', { class: 'empty' }, icon('fileSearch', 30), h('p', 'Select a file to preview it')));
  root.appendChild(top); root.appendChild(h('div', { class: 'files-body' }, list, preview)); el.appendChild(root);
  let path = ctx.query.get('path') || '';
  let active = null;

  async function load(p) {
    path = p; active = null;
    fovea.router.replaceQuery({ path: p || null });
    list.innerHTML = ''; list.appendChild(h('div', { class: 'row', style: 'padding:14px;justify-content:center' }, ui.spinner()));
    try {
      const data = await fovea.api.get(`/api/datasets/${ds.id}/files`, { path: p });
      top.innerHTML = '';
      top.appendChild(h('div', { class: 'crumbs' }, h('a', { href: '#', onClick: (e) => { e.preventDefault(); load(''); } }, icon('database', 14)), h('span', { class: 'sep' }, '/'),
        data.crumbs.flatMap((c, i) => [h('a', { href: '#', onClick: (e) => { e.preventDefault(); load(c.rel); } }, c.name), i < data.crumbs.length - 1 ? h('span', { class: 'sep' }, '/') : null])));
      top.appendChild(h('span', { class: 'spacer' }));
      top.appendChild(h('span', { class: 'small faint mono truncate', style: 'max-width:40vw' }, data.root_host + (p ? '/' + p : '')));
      top.appendChild(h('button', { class: 'btn btn-sm', onClick: () => ui.copyText(data.root_host + (p ? '/' + p : '')) }, icon('copy', 12), 'Copy'));
      list.innerHTML = '';
      if (p) list.appendChild(h('div', { class: 'file-row dir', onClick: () => load(p.split('/').slice(0, -1).join('/')) }, icon('cornerUpLeft'), h('span', '..')));
      if (!data.entries.length) list.appendChild(h('div', { class: 'empty' }, h('p', 'Empty folder')));
      data.entries.forEach(e => {
        const row = h('div', { class: cls('file-row', e.is_dir && 'dir'), onClick: () => e.is_dir ? load(e.rel) : open(e, row) },
          icon(e.is_dir ? 'folder' : e.kind === 'image' ? 'image' : e.kind === 'text' ? 'fileText' : 'file'), h('span', { class: 'truncate' }, e.name), e.is_dir ? null : h('span', { class: 'size' }, fmt.bytes(e.size)));
        list.appendChild(row);
      });
      if (data.truncated) list.appendChild(h('div', { class: 'small faint', style: 'padding:10px 16px' }, 'List truncated at 5,000 entries'));
    } catch (e) { list.innerHTML = ''; list.appendChild(h('div', { class: 'empty' }, h('p', e.message))); }
  }
  async function open(e, row) {
    list.querySelectorAll('.file-row').forEach(r => r.classList.remove('active')); row.classList.add('active');
    preview.innerHTML = '';
    if (e.kind === 'image') {
      preview.appendChild(h('div', { class: 'row mb-8' }, h('span', { class: 'mono small grow truncate' }, e.rel), e.image_id ? h('button', { class: 'btn btn-sm', onClick: () => fovea.router.navigate(`/d/${ds.id}/annotate?img=${e.image_id}`) }, icon('pen', 12), 'Annotate') : ui.chip('not indexed', { type: 'warn' })));
      preview.appendChild(h('img', { src: `/api/datasets/${ds.id}/file?path=${encodeURIComponent(e.rel)}`, alt: e.name }));
      return;
    }
    if (e.kind === 'text') {
      preview.appendChild(h('div', { class: 'row mb-8' }, h('span', { class: 'mono small grow truncate' }, e.rel), h('span', { class: 'xs faint' }, fmt.bytes(e.size))));
      const pre = h('pre', 'Loading…'); preview.appendChild(pre);
      try { const d = await fovea.api.get(`/api/datasets/${ds.id}/file`, { path: e.rel }); pre.textContent = d.text + (d.truncated ? '\n… (truncated)' : ''); }
      catch (err) { pre.textContent = err.message; }
      return;
    }
    preview.appendChild(h('div', { class: 'empty' }, icon('file', 30), h('p', `${e.name} · ${fmt.bytes(e.size)}`), h('a', { class: 'btn btn-sm', href: `/api/datasets/${ds.id}/file?path=${encodeURIComponent(e.rel)}&raw=1`, target: '_blank' }, 'Open raw')));
  }
  await load(path);
}
