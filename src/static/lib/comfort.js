// Visual-comfort preferences (see docs/visual-comfort.md).
//
// Stored per device, not per account: the right values depend on the room and the display in front of
// the person, and the same person may sit at a bright office monitor one day and a laptop at night.
//
//   dim    0 – 0.6   darkens everything Fovea draws. Lowering screen brightness eased eye fatigue in a
//                    randomised trial where colour-temperature ("blue light") software did not; comfort
//                    also peaks at moderate screen luminance, and a bright screen in a dim room is the
//                    most tiring combination.
//   stage  black | dark | gray   the surround of images in Inspect, Annotate and Compare. A lighter
//                    surround shrinks the luminance jump between the image and its edges.
//   rest   0 | minutes   opt-in reminder after that much continuous activity (evidence is mixed, so off
//                    by default; see rest reminder below).
import { h } from './dom.js';
import { icon } from './icons.js';

const KEY = 'fovea.comfort';
const DEFAULTS = { dim: 0, stage: 'dark', rest: 0 };
let prefs = load();

function load() {
  try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(KEY) || '{}') }; } catch (e) { return { ...DEFAULTS }; }
}
function save() { try { localStorage.setItem(KEY, JSON.stringify(prefs)); } catch (e) { /* private mode: session only */ } }

function apply() {
  const root = document.documentElement;
  const dim = Math.max(0, Math.min(0.6, Number(prefs.dim) || 0));
  if (dim > 0) { root.setAttribute('data-dim', ''); root.style.setProperty('--dim', String(dim)); }
  else { root.removeAttribute('data-dim'); root.style.removeProperty('--dim'); }
  if (prefs.stage === 'black' || prefs.stage === 'gray') root.setAttribute('data-stage', prefs.stage);
  else root.removeAttribute('data-stage');
}

// ------------------------------------------------------------------ rest reminder
// Software reminders to take breaks changed behaviour and eased symptoms for as long as they ran; 20-second
// breaks inside a demanding 40-minute task did not help; accuracy in high-rate visual inspection fell with
// time on task. So: off unless asked for, active time only (five idle minutes count as a break taken),
// suggests a real break rather than a glance, and is a static note — no dialog, sound or motion.
const IDLE_BREAK_MS = 5 * 60 * 1000;
let activeSince = 0, lastInput = 0, banner = null;

function onInput() {
  const now = Date.now();
  if (!activeSince || now - lastInput > IDLE_BREAK_MS) activeSince = now;
  lastInput = now;
  maybeRemind(now);
}
function maybeRemind(now = Date.now()) {
  if (!prefs.rest || banner || !activeSince) return;
  if (now - activeSince >= prefs.rest * 60 * 1000) showReminder(Math.round((now - activeSince) / 60000));
}
function hideReminder() { if (banner) { banner.remove(); banner = null; } }
function showReminder(minutes) {
  const done = () => { hideReminder(); activeSince = Date.now(); };
  banner = h('div', { class: 'rest-banner', role: 'status' },
    icon('eye', 15),
    h('div', { class: 'grow' },
      h('div', { class: 'strong' }, `${minutes} minutes of continuous work`),
      h('div', { class: 'small' }, 'Look at something far away — 6 m or more — for a minute or two. A few seconds is not enough.')),
    h('button', { class: 'btn btn-sm', onClick: done }, 'OK'));
  document.body.appendChild(banner);
}

let started = false;
function start() {
  if (started) return;
  started = true;
  for (const t of ['keydown', 'pointerdown', 'wheel']) window.addEventListener(t, onInput, { capture: true, passive: true });
  setInterval(() => { if (Date.now() - lastInput > IDLE_BREAK_MS) activeSince = 0; }, 30 * 1000);
}

export const comfort = {
  get: () => ({ ...prefs }),
  set(patch) {
    prefs = { ...prefs, ...patch };
    save(); apply();
    if (!prefs.rest) hideReminder();
    maybeRemind();
  },
  init() { apply(); start(); },
};
