// HTML box overlay positioned in % over an image container that matches the image aspect ratio.
import { h } from './dom.js';
import { classColor, className } from './colors.js';

export function boxLayer(boxes, { names = [], filter = null, labels = false, active = -1, onClick, stroke = 2, dim = false, styleFor } = {}) {
  const layer = h('div', { class: 'ov', style: 'position:absolute;inset:0;pointer-events:none;overflow:hidden;' });
  boxes.forEach((b, i) => {
    const cls = b[0];
    if (filter && !filter.has(cls)) return;
    const color = classColor(cls);
    const left = (b[1] - b[3] / 2) * 100, top = (b[2] - b[4] / 2) * 100, w = b[3] * 100, hgt = b[4] * 100;
    const custom = styleFor ? styleFor(b, i) : null;
    const el = h('div', {
      class: `bx ${i === active ? 'active' : ''}`,
      style: `position:absolute;left:${left}%;top:${top}%;width:${w}%;height:${hgt}%;border:${stroke}px ${custom?.dashed ? 'dashed' : 'solid'} ${custom?.color || color};box-sizing:border-box;` +
        `background:${custom?.fill || (dim ? 'transparent' : color + '22')};${i === active ? 'box-shadow:0 0 0 1px #fff, 0 0 0 3px ' + color + '99;' : ''}${onClick ? 'pointer-events:auto;cursor:pointer;' : ''}`,
      onClick: onClick ? (e) => { e.stopPropagation(); onClick(i, b); } : null,
    });
    if (labels) {
      const text = custom?.label != null ? custom.label : className(names, cls);
      el.appendChild(h('span', { style: `position:absolute;left:-${stroke}px;top:-${stroke}px;transform:translateY(-100%);background:${custom?.color || color};color:#fff;font:600 10px/1.4 var(--font);padding:1px 5px;border-radius:3px 3px 0 0;white-space:nowrap;max-width:200px;overflow:hidden;text-overflow:ellipsis;` }, text));
    }
    layer.appendChild(el);
  });
  return layer;
}
