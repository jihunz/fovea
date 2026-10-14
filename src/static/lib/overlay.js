// HTML box overlay positioned in % over an image container that matches the image aspect ratio.
// For interactive views (Inspect): one element per box. Grids draw one SVG per tile instead (explore.js).
import { h } from './dom.js';
import { classColor, className, textOn } from './colors.js';

const hex2 = (a) => Math.round(Math.max(0, Math.min(1, a)) * 255).toString(16).padStart(2, '0');

/**
 * boxes: [[cls, xc, yc, w, h], ...] normalised. Options:
 *   names, filter (Set of classes to show), labels (bool), active (index), onClick(i, b), stroke (px),
 *   dim (no fill), fill (true = 13% tint, false = outline only, or an alpha 0–1), styleFor(b, i) → {color, dashed,
 *   fill, label} per box.
 * Labels never vanish at the image edges: a box touching the top gets its label inside the box, one near the
 * right edge is labelled from its right side, and label text is black or white — whichever is readable on
 * the class colour (≥ 4.5:1).
 */
export function boxLayer(boxes, { names = [], filter = null, labels = false, active = -1, onClick, stroke = 2, dim = false, fill = true, styleFor } = {}) {
  const layer = h('div', { class: 'ov', style: 'position:absolute;inset:0;pointer-events:none;overflow:visible;' });
  const tint = dim || fill === false ? 0 : (typeof fill === 'number' ? fill : 0.13);
  boxes.forEach((b, i) => {
    const cls = b[0];
    if (filter && !filter.has(cls)) return;
    const color = classColor(cls);
    const left = (b[1] - b[3] / 2) * 100, top = (b[2] - b[4] / 2) * 100, w = b[3] * 100, hgt = b[4] * 100;
    const custom = styleFor ? styleFor(b, i) : null;
    const col = (custom && custom.color) || color;
    const bg = custom && custom.fill != null ? custom.fill : (tint && /^#[0-9a-f]{6}$/i.test(col) ? col + hex2(tint) : 'transparent');
    const isActive = i === active;
    const el = h('div', {
      class: `bx ${isActive ? 'active' : ''}`,
      style: `position:absolute;left:${left}%;top:${top}%;width:${w}%;height:${hgt}%;border:${stroke}px ${custom && custom.dashed ? 'dashed' : 'solid'} ${col};box-sizing:border-box;` +
        `background:${bg};${isActive ? `box-shadow:0 0 0 1px #fff, 0 0 0 3px ${col}99;z-index:2;` : ''}${onClick ? 'pointer-events:auto;cursor:pointer;' : ''}`,
      onClick: onClick ? (e) => { e.stopPropagation(); onClick(i, b); } : null,
    });
    if (labels) {
      const text = custom && custom.label != null ? custom.label : className(names, cls);
      const inside = top < 7;                       // no room above: label inside the box's top edge
      const fromRight = left + w > 70 && left > 45;  // near the right edge: grow leftwards from the right side
      const pos = `${inside ? `top:0;` : `bottom:100%;margin-bottom:${stroke > 1 ? -1 : 0}px;`}${fromRight ? `right:-${stroke}px;` : `left:-${stroke}px;`}`;
      const radius = inside ? (fromRight ? '0 0 0 3px' : '0 0 3px 0') : '3px 3px 0 0';
      el.appendChild(h('span', { class: 'bx-label', style: `position:absolute;${pos}background:${col};color:${textOn(col)};font:600 10px/1.4 var(--font);padding:1px 5px;border-radius:${radius};white-space:nowrap;max-width:200px;overflow:hidden;text-overflow:ellipsis;` }, text));
    }
    layer.appendChild(el);
  });
  return layer;
}
