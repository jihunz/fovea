"""AI UI helpers (static/views/ai-panel.js, static/views/ai.js), exercised in Node.

What these pin down:
* An OpenAI-compatible server that does not answer (or whose configured models are not pulled) is never offered as
  ready: the picker dims it with the reason, and the default two-stage classifier skips it (N4).
* "Economy" (a provider batch API at about half price) is offered only where the provider has a batch API, the request
  carries the flag, and the estimate leads with the batch price.
* Class-name suggestions: the answer is normalised, names are checked for empties and duplicates against the whole
  class list (as Overview's editor requires), and the class list written back keeps every other class.
* "Fill empty" says exactly what the runner does: only images WITHOUT a label file are written; an empty label file is
  a deliberate background image and is left alone (tip, hint, estimate note and confirmation).
* An Economy run resumed after a stop mid-submit reports the images that were never sent (they are deliberately not
  resubmitted) in the job line, the job summary and the run's provider-batch summary.
* The batch form, driven through a small fake DOM: a preview that is still running (or already shown) when the scope,
  model or classes change never lands under the new settings; a class mapping made in the form's own "classes → yours"
  row is used at once (no stale copy of the profile); typing ids into "These image ids" selects that scope, so the
  estimate and Start use those ids, not the previously checked radio.
* Review round 2: Settings' Fast / Balanced / Accurate reaches every request (a picked model stores no effort, so the
  provider default applies); a reached 30-day spend cap blocks the provider in the picker, the panel and the form, and
  the run budget is pre-filled with what is left of the cap; a cloud model with no price reads "cost unknown" (never
  "free" or "$0"), sends no budget and is refused up front under a cap; a double click / Enter twice on Start submits
  one job; job boxes show a provider wait once; Propose names its provider's rate-limit wait (GET /api/ai/waits);
  "Random sample N" draws a new seeded sample per run; Stop in "Suggest names" skips the classes not sent yet
  (POST /api/ai/name-classes/cancel) and says what was billed.
Skipped when Node is not installed.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "static"
NODE = shutil.which("node")

SCENARIOS = r"""
globalThis.location = { hostname: 'localhost' };
globalThis.navigator = globalThis.navigator || { platform: 'MacIntel' };
const m = await import('./views/ai-panel.js');
const t = await import('./views/ai.js');
const out = {};

// ---- providers: unreachable / checking / not pulled (N4)
const compatDown = { id: 'openai_compat', name: 'OpenAI-compatible', kind: 'local', capabilities: { detect: true, classify: true, generate: true },
  status: { available: false, reason: 'Not reachable at 127.0.0.1:11434' }, models: [{ id: 'qwen3-vl:8b' }] };
const compatChecking = { ...compatDown, status: { available: null, checking: true, reason: null } };
const compatChecking2 = { ...compatDown, status: { available: false, reason: 'Checking 127.0.0.1:11434…' } };   // the server's own shape
const compatUnpulled = { ...compatDown, status: { available: true }, models: [{ id: 'qwen3-vl:8b', pulled: false }, { id: 'llava:7b', pulled: true, name: 'LLaVA 7B' }] };
const compatNonePulled = { ...compatDown, status: { available: true }, models: [{ id: 'a', pulled: false }, { id: 'b', pulled: false }] };
const yolo = { id: 'yolo', name: 'YOLO', kind: 'local', capabilities: { detect: true, open_vocab: false, confidence: true }, models: [{ id: 'yolo12m', default: true }] };
const mock = { id: 'mock', name: 'Mock', kind: 'cloud', capabilities: { detect: true, classify: true, generate: true, batch_api: true, open_vocab: true },
  models: [{ id: 'mock-fast', default: true, recommended: true, price: { in: 0.5, out: 2 } }] };
const claude = { id: 'anthropic', name: 'Anthropic Claude', kind: 'cloud', needs_key: true, key: { present: true, hint: '…0001' },
  capabilities: { detect: true, classify: true, generate: true, batch_api: true, open_vocab: true },
  models: [{ id: 'claude-haiku-5-5', name: 'Claude Haiku 5.5', default: true, price: { in: 1, out: 5 } }] };
const gem = { ...claude, id: 'gemini', name: 'Gemini', capabilities: { detect: true, classify: true, generate: true, batch_api: false, open_vocab: true },
  models: [{ id: 'gemini-x', default: true, price: { in: 0.3, out: 2.5 } }] };
const P = { providers: [yolo, compatDown, mock, claude, gem], remote_ai_allowed: true };
const st = (p) => m.providerState(P, p);
out.down = { ...st(compatDown), tone: m.providerTone(P, compatDown) };
out.checking = { ...st(compatChecking), tone: m.providerTone(P, compatChecking), isChecking: m.isChecking(compatChecking) };
out.checking2 = { ...st(compatChecking2), tone: m.providerTone(P, compatChecking2) };
out.downModel = m.modelProblem({ id: 'q', available: false, reason: 'Not pulled on 127.0.0.1:11434 — run: ollama pull q' });
out.unpulled = { state: st(compatUnpulled), def: (m.defaultModel(compatUnpulled) || {}).id, problem: m.modelProblem(compatUnpulled.models[0]) };
out.nonePulled = st(compatNonePulled);
out.classifierDefault = null;
const desc = m.describeSource({ providers: [compatChecking] }, {}, { provider: 'openai_compat', model: 'qwen3-vl:8b' });
out.describeChecking = { ok: desc.ok, checking: !!desc.checking };

// The picker: a fake fovea whose h() builds plain objects and whose menu() returns its items.
const flat = (n) => (n == null ? '' : Array.isArray(n) ? n.map(flat).join(' ') : typeof n === 'object' ? flat(n.kids) : String(n));
const fakeFovea = { h: (tag, props, ...kids) => ({ tag, props, kids }), ui: { menu: (a, items) => ({ items, classList: { add() {} } }) }, router: { navigate() {} } };
const menu = m.pickSource(null, { fovea: fakeFovea, P: { providers: [yolo, compatDown, compatUnpulled] }, prof: {}, dsId: 'x', current: null, onPick() {}, purpose: 'detect', allowTwoStage: false });
out.menu = menu.items.filter((i) => !i.header && !i.sep).map((i) => ({ text: flat(i.label), disabled: !!i.disabled, settings: !!i.onClick && !!i.disabled === false && /Settings/.test(flat(i.label)) }));
const genMenu = m.pickSource(null, { fovea: fakeFovea, P, prof: {}, dsId: 'x', current: null, onPick() {}, purpose: 'generate', allowTwoStage: false });
out.genMenu = genMenu.items.filter((i) => !i.header && !i.sep && !i.disabled && i.onClick).map((i) => flat(i.label));

// ---- economy (provider batch API)
out.eco = {
  cloud: m.economyFor(P, {}, { provider: 'anthropic', model: 'claude-haiku-5-5' }).ok,
  noBatch: m.economyFor(P, {}, { provider: 'gemini', model: 'gemini-x' }).ok,
  local: m.economyFor(P, {}, { provider: 'yolo', model: 'yolo12m' }).ok,
  twoStage: m.economyFor(P, { twoStage: { detector: { provider: 'yolo', model: 'yolo12m', source_classes: ['person'] }, classifier: { provider: 'anthropic', model: 'claude-haiku-5-5' } } }, { pipeline: 'two_stage' }).ok,
};
out.withSpeed = m.withSpeed({ provider: 'anthropic', options: { effort: 'low' } }, 'economy');
out.withSpeedNow = m.withSpeed({ provider: 'anthropic', options: { effort: 'low' } }, 'now');
out.priceEco = m.priceLine({ usd: [1, 2], usd_batch: [0.5, 1] }, 'economy');
out.priceNow = m.priceLine({ usd: [1, 2], usd_batch: [0.5, 1] }, 'now');
// the server's shape: params.provider_batches + result.provider (last poll); status 'waiting' while at the provider
const ecoRun = { id: 3, status: 'waiting', images: 40, created_at: 100, params: { speed: 'economy', batch_id: 'msgbatch_01', target: 'proposals',
  provider_batches: [{ id: 'msgbatch_01', n: 25, state: 'running', submitted_at: 120 }, { id: 'msgbatch_02', n: 15, state: 'running', submitted_at: 130 }] },
  result: { images_done: 0, provider: { state: 'running', counts: { processing: 30, succeeded: 9, errored: 1 }, next_check_at: 999 } } };
out.pb = m.providerBatchOf(ecoRun);
out.pbOpen = [m.batchOpen(ecoRun), m.batchOpen({ ...ecoRun, status: 'done' })];
// the summary as result.batch / job.extra.batch (the job's copy is fresher)
out.pbJob = m.providerBatchOf({ ...ecoRun, result: { batch: { state: 'running', counts: { processing: 40 } } },
  job: { status: 'running', extra: { speed: 'economy', images_done: 7, batch: { state: 'canceling', counts: { processing: 20, succeeded: 20 }, next_check_at: 5, filed: 7 } } } });
out.pbFallback = m.providerBatchOf({ id: 5, status: 'waiting', params: { speed: 'economy', provider_batches: [{ id: 'b1', n: 4, state: 'ended', ingested: true, counts: { succeeded: 4 } }, { id: 'b2', n: 6, state: 'canceling', counts: { processing: 6 } }] }, result: {} });
out.pbNone = m.providerBatchOf({ id: 4, status: 'done', params: { target: 'proposals' } });
out.batchState = [m.batchStateText('running'), m.batchStateText('canceling'), m.batchStateText('ended'), m.batchStateText('weird_state')];

// ---- class-name suggestions
out.norm = t.normSuggestions({ items: [{ cls: 1, name: ' fallen_person ', prompt: 'person lying on the ground', description: 'd', confidence: 0.9, examples: [{ image_id: 5, box: [0.5, 0.5, 0.1, 0.2] }] }, { cls: 'x' }],
  usage: { cost_usd: 0.01 }, warnings: ['sk-ant-api03-SECRET123456 leaked'] });
out.normList = t.normSuggestions([{ cls: 0, name: 'a' }]).items.length;
// the server's own item: numeric confidence plus its label, per-class error and warnings
out.normServer = t.normSuggestions({ items: [{ cls: 10, current_name: 'class_10', boxes: 0, examples: [], error: 'No labeled boxes of this class to look at' },
  { cls: 2, name: 'sitting_person', prompt: 'person sitting', description: 'x', confidence: 0.55, confidence_label: 'medium', warnings: ['a.jpg: bad'] }] }).items;
out.conf = [t.confWord(0.9), t.confWord(0.5), t.confWord(0.1), t.confWord('High'), t.confWord(null)];
const names = ['class_0', 'class_1', 'car', 'class_3'];
out.problems = t.nameProblems(names, { 0: 'person', 1: 'Person', 3: 'car' });
out.problemsOk = t.nameProblems(names, { 0: 'standing_person', 1: 'fallen_person' });
out.problemsEmpty = t.nameProblems(names, { 0: '  ' });
out.renamed = t.renamed(['a', 'b'], { 3: 'd', 0: ' A ' });
out.est = t.namingEstimate({ price: { in: 1, out: 5 } }, 12, 6);
out.estFree = t.namingEstimate({}, 12, 6);

// ---- Fill empty: exact wording (the runner skips every image that has a label file, empty ones included)
out.fillTip = m.TARGETS.find((x) => x.value === 'fill').tip;
out.fillHint = m.TARGET_HINT.fill;
out.fillEffect = [m.fillEffect({ images: 2, no_label_file: 1, empty_label_file: 1, labeled: 0, boxes: 0 }),
  m.fillEffect({ images: 10, no_label_file: 3, empty_label_file: 2, labeled: 5, boxes: 9 }),
  m.fillEffect({ images: 4, no_label_file: 4, empty_label_file: 0 }),
  m.fillEffect({ images: 3, no_label_file: 1, empty_label_file: 0, labeled: 2 })];

// ---- Economy resumed after a stop mid-submit: images never sent are counted, not resubmitted
const { fmt } = await import('./lib/dom.js');
const fv = { fmt };
out.sumNotSent = m.jobSummary(fv, { status: 'done', total: 1000, done: 1000, extra: { speed: 'economy' }, result: { images_done: 1000, boxes: 12, not_submitted: 6808, speed: 'economy' } });
out.sumPlain = m.jobSummary(fv, { status: 'done', total: 1, done: 1, result: { images_done: 1, boxes: 3, written: 1 } });
out.notSentText = [m.notSentText(1), m.notSentText(6808), m.notSentText(0)];
out.pbNotSent = m.providerBatchOf({ id: 9, status: 'waiting', params: { speed: 'economy', provider_batches: [{ id: 'b1', n: 1000, state: 'running' }] },
  result: { not_submitted: 6808 } }).not_submitted;

// ---- latestOnly: only the newest answer may land
const lo = m.latestOnly();
const a1 = lo.begin();
const a2 = lo.begin();                          // a second begin abandons the first
out.latest = { firstStale: !a1.current(), firstAborted: a1.signal.aborted, second: a2.current(), running: lo.running };
const dropped = lo.drop();
out.latest.afterDrop = { dropped, second: a2.current(), aborted: a2.signal.aborted, running: lo.running, again: lo.drop() };
// ---- effort: Settings' three levels reach every request; a picked model follows the provider's default
const claudeMed = { ...claude, config: { effort: 'medium' } };
const PE = { providers: [yolo, claudeMed], remote_ai_allowed: true };
const profE = m.profileOf('effort-ds');
out.effort = {
  values: ['fast', 'balanced', 'accurate', undefined].map(m.effortValue),
  defaults: [m.defaultEffort(claudeMed), m.defaultEffort({ config: { effort: 'high' } }), m.defaultEffort({ config: {} }), m.defaultEffort(null)],
  of: [m.effortOf(PE, { provider: 'anthropic', model: 'claude-haiku-5-5' }), m.effortOf(PE, { provider: 'anthropic', model: 'x', effort: 'accurate' })],
  sent: (await m.requestFor(fakeFovea, PE, profE, { provider: 'anthropic', model: 'claude-haiku-5-5' }, ['a', 'b'], { dsId: 'effort-ds' })).body.options.effort,
  sentChosen: (await m.requestFor(fakeFovea, PE, profE, { provider: 'anthropic', model: 'claude-haiku-5-5', effort: 'fast' }, ['a', 'b'], { dsId: 'effort-ds' })).body.options.effort,
  labels: m.EFFORTS.map((e) => e.label),
};
const picked = [];
const pm = m.pickSource(null, { fovea: fakeFovea, P: PE, prof: {}, dsId: 'x', current: { provider: 'yolo', model: 'yolo12m' }, onPick: (s2) => picked.push(s2), allowTwoStage: false });
pm.items.filter((i) => i.onClick && /Claude Haiku/.test(flat(i.label))).forEach((i) => i.onClick());
const pm2 = m.pickSource(null, { fovea: fakeFovea, P: PE, prof: {}, dsId: 'x', current: { provider: 'anthropic', model: 'claude-x', effort: 'accurate' }, onPick: (s2) => picked.push(s2), allowTwoStage: false });
pm2.items.filter((i) => i.onClick && /Claude Haiku/.test(flat(i.label))).forEach((i) => i.onClick());
out.effort.picked = picked;
out.effort.defaultSource = m.defaultSource({ providers: [claudeMed] }, {});

// ---- 30-day spend caps: a reached cap blocks the provider everywhere, with the numbers
const PC = { providers: [yolo, claude], remote_ai_allowed: true, caps: { anthropic: { cap_usd: 1, spent_30d: 1.95 } } };
out.capReached = m.providerState(PC, claude);
out.capTone = m.providerTone(PC, claude);
const capMenu = m.pickSource(null, { fovea: fakeFovea, P: PC, prof: {}, dsId: 'x', current: null, onPick() {}, allowTwoStage: false });
out.capMenu = capMenu.items.filter((i) => !i.header && !i.sep).map((i) => flat(i.label)).filter((t) => /spend cap/.test(t));
const PL = { ...PC, caps: { anthropic: { cap_usd: 2.15, spent_30d: 1.98 } } };
out.capLeft = { state: m.providerState(PL, claude).ok, cap: m.sourceCap(PL, {}, { provider: 'anthropic', model: 'claude-haiku-5-5' }) };
out.capLeft.cap = { left: Math.round(out.capLeft.cap.left * 100) / 100, cap: out.capLeft.cap.cap, name: out.capLeft.cap.p.name };
out.capLocal = m.capOf({ caps: { yolo: { cap_usd: 0, spent_30d: 0 } } }, yolo);
out.hasCap = [m.hasCap(PL, claude), m.hasCap({ providers: [] }, { ...claude, config: { spend_cap_usd: 3 } }), m.hasCap({}, claude)];

// ---- cost kinds: a cloud model with no price on file is "cost unknown", never "free" or "$0"
out.costKind = [m.costKind({ usd: [0, 0] }, false), m.costKind({ usd: [0, 0], priced: false }, true), m.costKind({ usd: [0, 0] }, true),
  m.costKind({ usd: [0.1, 0.2] }, true), m.costKind({ usd: [0.1, 0.2], priced: false }, true)];
out.unpricedText = [m.unpricedText({ unpriced: ['claude-sonnet-9-9'] }, 'x'), m.unpricedText({}, 'Claude Sonnet 9 9')];

// ---- job box: a provider wait shows once, without the counts the server appends to it
out.jobNote = [
  m.jobNote({ message: 'Anthropic Claude rate limited — retrying in 10 s · 2/12 · 4 boxes · $0.0115' }),
  m.jobNote({ message: 'x', extra: { wait: 'Gemini provider busy — retrying in 3 s' } }),
  m.jobNote({ message: '2/12 · 4 boxes · $0.0115' }),
  m.jobNote({ message: 'Loading models…' }),
  m.jobNote({ message: 'Starting…' }),
  m.jobNote({ message: 'Anthropic Claude rate limited — retrying in 7 s · Auditing 3 of 40' }),
];

// ---- the preview line names a price only where one applies
out.previewHint = [m.previewHint(m.describeSource(P, {}, { provider: 'yolo', model: 'yolo12m' })),
  m.previewHint(m.describeSource(P, {}, { provider: 'anthropic', model: 'claude-haiku-5-5' })),
  m.previewHint(m.describeSource(P, {}, { provider: 'anthropic', model: 'claude-haiku-5-5' }), true)];

// ---- provider waits during a synchronous Propose
out.waitNames = [m.waitNames(P, { provider: 'anthropic' }), m.waitNames(P, { provider: 'yolo' }),
  m.waitNames(P, { provider: 'anthropic', options: { pipeline: 'two_stage', detector: { provider: 'yolo' }, classifier: { provider: 'anthropic' } } })];
const waits = { waits: [{ reason: 'Gemini rate limited', seconds_left: 50 }, { reason: 'Anthropic Claude rate limited', seconds_left: 32 }, { reason: 'Anthropic Claude provider busy', seconds_left: 4 }] };
out.pickWait = m.pickWait(waits, ['Anthropic Claude']);
out.waitSentence = m.waitSentence(out.pickWait);
out.pickWaitNone = m.pickWait(waits, ['OpenAI']);

// ---- AI tab helpers
out.ph = t.placeholderClasses(['class_0', 'person', 'class_2', 'class_3'], {}, { 0: 12, 1: 4 });
out.phNoCounts = t.placeholderClasses(['class_0', 'person'], {}, null);
out.phCard = t.placeholderClasses(['class_0', 'class_1'], { 1: { prompt: 'fallen person' } }, { 0: 1, 1: 1 }).all;
out.classList = [t.classList(['a', 'b', 'c', 'd', 'e'], [0]), t.classList(['a', 'b', 'c', 'd', 'e'], [0, 1]), t.classList(['a', 'b', 'c', 'd', 'e'], [0, 1, 2, 3, 4])];
out.stopped = [t.stoppedText([{ error: 'Stopped before this class was sent' }, { error: 'Stopped before this class was sent' }, { name: 'x' }]), t.stoppedText([{ name: 'x' }])];
const seeds = new Set(Array.from({ length: 20 }, () => t.newSeed()));
out.seeds = { distinct: seeds.size, ok: [...seeds].every((v) => Number.isInteger(v) && v > 0 && v < 2 ** 31) };
const live = { ...claude, models: [...claude.models, { id: 'claude-sonnet-9-9', name: 'Claude Sonnet 9 9', notes: 'Not in Fovea\'s price table' }] };
const CR = { P: { providers: [yolo, live] } };
out.untracked = [
  t.costUntracked(CR, { provider: 'anthropic', model: 'claude-sonnet-9-9', cost_usd: 0, input_tokens: 11000, output_tokens: 888 }),
  t.costUntracked(CR, { provider: 'anthropic', model: 'claude-haiku-5-5', cost_usd: 0, input_tokens: 0, output_tokens: 0 }),
  t.costUntracked(CR, { provider: 'anthropic', model: 'claude-haiku-5-5', cost_usd: 0.01, input_tokens: 9 }),
  t.costUntracked(CR, { provider: 'mock+anthropic', model: 'a+b', cost_usd: 0, result: { warnings: ['no price on file for b; cost not tracked (3×)'] } }),
  t.costUntracked(CR, { provider: 'yolo', model: 'yolo12m', cost_usd: 0, input_tokens: 0 }),
];
// ---- a stock COCO model none of whose classes maps onto the dataset's is not the default when another local model
// can work, the picker says it needs a mapping, and a request for it is refused with fix 'map'
const mapCalls = [];
const fvMap = { ...fakeFovea, api: { get: async () => ({}), post: async (path, body) => {
  mapCalls.push(body);
  const auto = { dsauto: 0 }[body.dataset_id], cm = body.class_map || {};
  const cls = cm['0'] != null ? Number(cm['0']) : auto != null ? auto : null;
  return { items: [{ model_id: 0, model_name: 'person', cls, how: cls == null ? null : cm['0'] != null ? 'explicit' : 'synonym' }], mapped: cls == null ? 0 : 1 };
} } };
const yoloe = { id: 'yoloe', name: 'YOLOE', kind: 'local', capabilities: { detect: true, open_vocab: true, confidence: true }, models: [{ id: 'yoloe-26s-seg', default: true }] };
const PD = { providers: [yolo, yoloe, claude], remote_ai_allowed: true };
const pick = (s2) => s2 && `${s2.provider}:${s2.model}`;
out.defaults = {
  placeholders: pick(await m.usefulDefault(fvMap, PD, m.profileOf('dsph'), 'dsph', ['class_0', 'class_1'], {})),
  named: pick(await m.usefulDefault(fvMap, PD, m.profileOf('dsnamed'), 'dsnamed', ['fallen', 'standing'], {})),
  described: pick(await m.usefulDefault(fvMap, PD, m.profileOf('dscard'), 'dscard', ['class_0', 'class_1'], { 1: { prompt: 'person lying on the floor' } })),
  synonym: pick(await m.usefulDefault(fvMap, PD, m.profileOf('dsauto'), 'dsauto', ['pedestrian', 'fallen'], {})),
  noOther: pick(await m.usefulDefault(fvMap, { providers: [yolo, claude], remote_ai_allowed: true }, m.profileOf('dsnamed'), 'dsnamed', ['fallen', 'standing'], {})),
};
out.known = [m.knownMapped('dsph', m.profileOf('dsph'), yolo, 'yolo12m'), m.knownMapped('dsauto', {}, yolo, 'yolo12m'), m.knownMapped('dsnever', {}, yolo, 'yolo12m')];
const hintMenu = (dsId) => m.pickSource(null, { fovea: fakeFovea, P: PD, prof: {}, dsId, current: null, onPick() {}, allowTwoStage: false })
  .items.filter((i) => i.onClick && !i.disabled).map((i) => flat(i.label)).filter((x) => /yolo/i.test(x));
out.mapHint = { unmapped: hintMenu('dsph'), mapped: hintMenu('dsauto'), unknown: hintMenu('dsnever') };
const reqUnmapped = await m.requestFor(fvMap, PD, m.profileOf('dsph'), { provider: 'yolo', model: 'yolo12m' }, ['class_0', 'class_1'], { dsId: 'dsph' });
const reqMapped = await m.requestFor(fvMap, PD, { ...m.profileOf('dsx'), maps: { 'yolo:yolo12m': { 0: 1 } } }, { provider: 'yolo', model: 'yolo12m' }, ['class_0', 'class_1'], { dsId: 'dsx' });
const fv404 = { ...fakeFovea, api: { get: async () => ({}), post: async () => { const e = new Error('Not Found'); e.status = 404; throw e; } } };
const reqOld = await m.requestFor(fv404, PD, m.profileOf('dsold'), { provider: 'yolo', model: 'yolo12m' }, ['person', 'class_1'], { dsId: 'dsold' });
out.req = { unmapped: { ok: reqUnmapped.ok, fix: reqUnmapped.fix, reason: reqUnmapped.reason }, mapped: reqMapped.body, old: reqOld.body };

// ---- a path in a narrow caption: split chip, the shared head that gives way, the end that always stays
const pl = (it) => { const n = m.pathLine(fakeFovea, it, { cls: 'xs faint' }); return { cls: n.props.class, title: n.props.title, kids: n.kids.filter(Boolean).map((k) => [k.props.class, flat(k)]) }; };
out.pathLine = {
  korean: pl({ rel_path: 'images/test-1854-1860-포함/images_GOPR0812_01016.jpg', split: 'test-1854-1860-포함' }),
  noSplit: pl({ rel_path: 'frames/cam2/frame_0042.jpg', split: '' }),
  root: pl({ rel_path: 'x.jpg' }),
};
console.log(JSON.stringify(out));
"""

# The batch form (ai-panel.js batchForm) needs a DOM: a minimal one, enough for dom.js and the form's own queries.
FORM_PRELUDE = r"""
globalThis.location = { hostname: 'localhost' };
globalThis.navigator = globalThis.navigator || { platform: 'MacIntel' };
class FNode {
  constructor() { this.childNodes = []; this.parentNode = null; }
  get firstChild() { return this.childNodes[0] || null; }
  appendChild(c) {
    if (c instanceof FFrag) { [...c.childNodes].forEach((k) => this.appendChild(k)); return c; }
    if (c.parentNode) c.parentNode.removeChild(c);
    c.parentNode = this; this.childNodes.push(c); return c;
  }
  removeChild(c) { const i = this.childNodes.indexOf(c); if (i >= 0) this.childNodes.splice(i, 1); c.parentNode = null; return c; }
  remove() { if (this.parentNode) this.parentNode.removeChild(this); }
  get textContent() { return this.childNodes.map((c) => c.textContent).join(''); }
  set textContent(v) { this.childNodes.forEach((c) => { c.parentNode = null; }); this.childNodes = []; if (v !== '' && v != null) this.appendChild(new FText(String(v))); }
}
class FText extends FNode { constructor(t) { super(); this.data = t; } get textContent() { return this.data; } }
class FFrag extends FNode {}
class FEl extends FNode {
  constructor(tag) {
    super(); this.tagName = tag.toUpperCase(); this.attrs = {}; this.listeners = {}; this.className = ''; this.style = {}; this.dataset = {};
    this.disabled = false; this.checked = false; this.value = ''; const el = this;
    this.classList = {
      get s() { return new Set(el.className.split(/\s+/).filter(Boolean)); },
      add(...c) { const s = this.s; c.forEach((x) => s.add(x)); el.className = [...s].join(' '); },
      remove(...c) { const s = this.s; c.forEach((x) => s.delete(x)); el.className = [...s].join(' '); },
      toggle(c, on) { const s = this.s; const v = on === undefined ? !s.has(c) : !!on; if (v) s.add(c); else s.delete(c); el.className = [...s].join(' '); return v; },
      contains(c) { return this.s.has(c); },
    };
  }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  hasAttribute(k) { return k in this.attrs; }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  removeEventListener() {}
  fire(t) {
    const e = { type: t, target: this, currentTarget: this, stopPropagation() {}, preventDefault() {} };
    (this.listeners[t] || []).forEach((f) => f(e));
    if (typeof this['on' + t] === 'function') this['on' + t](e);
  }
  click() { if (!this.disabled) this.fire('click'); }
  focus() { globalThis.FOCUSED = this; }
  matches(sel) {
    const m = /^([a-z0-9]*)((?:\.[\w-]+)*)((?:\[[^\]]+\])*)$/i.exec(sel.trim());
    if (!m) return false;
    if (m[1] && this.tagName !== m[1].toUpperCase()) return false;
    for (const c of (m[2].match(/\.[\w-]+/g) || [])) if (!this.classList.contains(c.slice(1))) return false;
    for (const a of (m[3].match(/\[[^\]]+\]/g) || [])) {
      const [k, v] = a.slice(1, -1).split('=');
      const val = k === 'value' ? this.value : this.getAttribute(k);
      if (v === undefined ? val == null : String(val) !== v.replace(/"/g, '')) return false;
    }
    return true;
  }
  querySelectorAll(sel) {
    const out = [];
    const walk = (n) => n.childNodes.forEach((c) => { if (c instanceof FEl) { if (sel.split(',').some((x) => c.matches(x))) out.push(c); walk(c); } });
    walk(this);
    return out;
  }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
  closest() { return null; }
  getBoundingClientRect() { return { top: 0, bottom: 0, left: 0, right: 0, width: 0, height: 0 }; }
}
globalThis.Node = FNode;
globalThis.document = { createElement: (t) => new FEl(t), createTextNode: (t) => new FText(t), createDocumentFragment: () => new FFrag(),
  addEventListener() {}, removeEventListener() {} };

const dom = await import('./lib/dom.js');
const m = await import('./views/ai-panel.js');
const { h, fmt } = dom;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const text = (el) => (el ? el.textContent : '');
const calls = [];
const previews = [];                 // pending POST /api/ai/preview calls, answered by the scenario
const handlers = {};
const bus = { on(n, f) { (handlers[n] = handlers[n] || []).push(f); return () => {}; }, emit(n, e) { (handlers[n] || []).forEach((f) => f(e)); } };
const api = {
  async get(path, q) {
    calls.push(['GET', path, q]);
    if (/\/images$/.test(path)) return { items: String(q.ids).split(',').map(Number).map((id) => ({ id, rel_path: `img${id}.jpg`, width: 640, height: 640, boxes: [] })) };
    if (path === '/api/ai/usage') return { caps: globalThis.CAPS || {} };
    if (path === '/api/ai/providers') return globalThis.PROVIDERS || { providers: [] };
    return {};
  },
  async put() { return {}; },
  post(path, body, opts = {}) {
    calls.push(['POST', path, body]);
    if (path === '/api/ai/mapping') {
      const cm = body.class_map || {};
      return Promise.resolve({ items: [{ model_id: 0, model_name: 'person', cls: cm['0'] != null ? cm['0'] : null, how: cm['0'] != null ? 'explicit' : null }], mapped: cm['0'] != null ? 1 : 0 });
    }
    if (path === '/api/ai/estimate') {
      const base = { images: body.count ?? (body.image_ids || []).length, usd: [0, 0], seconds_per_image: 0.1 };
      return Promise.resolve(globalThis.EST_HOOK ? globalThis.EST_HOOK(body, base) : base);
    }
    if (path === '/api/ai/preview') {
      // A server that answers even after the browser gave up: only the form's own guard can drop the late reply.
      return new Promise((resolve) => previews.push({ body, signal: opts.signal, answer: () => resolve({ items: body.image_ids.map((id) => ({ image_id: id, preds: [{ cls: 0, xc: 0.5, yc: 0.5, w: 0.1, h: 0.1, conf: 0.9 }] })) }) }));
    }
    if (path === '/api/ai/batch') return new Promise((r) => setTimeout(() => r({ job: { id: 'j1', status: 'queued' }, run_id: 1 }), globalThis.BATCH_DELAY || 0));
    if (path === '/api/ai/propose' && globalThis.PROPOSE_HOOK) return globalThis.PROPOSE_HOOK(body, opts);
    return Promise.resolve({});
  },
  thumbUrl: (id) => `/t/${id}`,
};
const seg = (items, value, onChange) => {
  const el = h('div', { class: 'seg' }, items.map((it) => h('button', { type: 'button', 'data-value': String(it.value), onClick: () => { el.setValue(it.value); onChange(it.value); } }, it.label)));
  el.setValue = (v) => { el.value = v; };
  el.value = value;
  return el;
};
const fovea = {
  h, fmt, bus, api,
  keys: { is: () => false },
  ui: { seg, kbd: (k) => h('kbd', k), spinner: () => h('span', { class: 'spinner' }), elapsedClock: () => h('span', { class: 'elapsed' }), progress: () => h('div', { class: 'progress' }, h('i')),
    confirm: async (o) => { (globalThis.CONFIRMS = globalThis.CONFIRMS || []).push(o); return globalThis.CONFIRM_ANSWER ?? true; },
    prompt: async () => null, toast() {}, menu: () => h('div'), modal: () => ({}),
    midText: (t, o) => h('span', { class: 'mid', title: t }, t) },
  icon: (n) => h('i', { class: `ico ico-${n}` }),
  link: (href, props, ...kids) => h('a', { ...props, href }, ...kids),
  colors: { classColor: () => '#888888', className: (names, c) => names[c] || `class_${c}` },
  media: { trackImage() {} },
};
const yolo = { id: 'yolo', name: 'YOLO', kind: 'local', capabilities: { detect: true, open_vocab: false, confidence: true }, models: [{ id: 'yolo12m', default: true }] };
const mock = { id: 'mock', name: 'Mock', kind: 'cloud', capabilities: { detect: true, open_vocab: true, confidence: false }, models: [{ id: 'mock-fast', name: 'Mock fast', default: true }] };
const P = { providers: [yolo, mock], remote_ai_allowed: true };
const scope = (id, ids) => ({ id, label: id.toUpperCase(), count: async () => ids.length, resolve: async () => ({ image_ids: ids }), sample: async (k) => ids.slice(0, k) });
const radios = (f) => f.el.querySelectorAll('input[type=radio]').map((r) => r.checked);
const button = (f, word) => [...f.el.querySelectorAll('button'), ...f.footer()].find((b) => text(b).includes(word));
const out = {};
"""

FORM_SCENARIOS = FORM_PRELUDE + r"""

// ---- 1. a class mapping made in the form's own "classes → yours" row is used at once
{
  const ds = { id: 'dsmap', classes: ['class_0', 'class_1'] };
  const f = m.batchForm({ fovea, ds, P, scopes: [scope('a', [1, 2, 3, 4, 5])], src: { provider: 'yolo', model: 'yolo12m', effort: 'fast' } });
  await sleep(320);
  const est = f.el.querySelector('.aib-est'), start = f.footer()[1];
  out.mapBefore = { est: text(est), startDisabled: start.disabled };
  const sel = f.el.querySelectorAll('select').find((x) => x.getAttribute('aria-label') === 'Map another model class');
  out.mapSelect = !!sel;
  sel.value = '0'; sel.fire('change');
  await sleep(320);
  out.mapAfter = { est: text(est), startDisabled: start.disabled, profile: m.profileOf('dsmap').maps };
  f.destroy();
}

// ---- 2. a preview still running when the settings change never lands under them
{
  const ds = { id: 'dsprev', classes: ['person'] };
  const f = m.batchForm({ fovea, ds, P, scopes: [scope('a', [1, 2, 3]), scope('b', [4, 5, 6])], src: { provider: 'mock', model: 'mock-fast', effort: 'fast' } });
  await sleep(320);
  const prevEl = f.el.querySelector('.aib-prev');
  const prevBtn = button(f, 'Preview on 3 images');
  prevBtn.click(); await sleep(20);
  out.p1 = { sent: previews.length, ids: previews[0] && previews[0].body.image_ids, btnDisabled: prevBtn.disabled };
  const rb = f.el.querySelectorAll('input[type=radio]')[1];
  rb.checked = true; rb.fire('change');                   // scope A -> B while the preview runs
  out.p1.aborted = previews[0].signal.aborted;
  previews[0].answer(); await sleep(20);                  // the old reply arrives anyway
  out.p1.after = { grid: !!prevEl.querySelector('.aib-prev-grid'), text: text(prevEl), btnDisabled: prevBtn.disabled, radios: radios(f) };
  prevBtn.click(); await sleep(20);                       // a fresh preview of scope B shows, labelled with its model
  previews[1].answer(); await sleep(20);
  out.p2 = { ids: previews[1].body.image_ids, items: prevEl.querySelectorAll('figure').length, text: text(prevEl).slice(-80), btnDisabled: prevBtn.disabled };
  f.setSource({ provider: 'mock', model: 'mock-fast', effort: 'accurate' });   // a model / effort change clears a shown preview
  out.p2.afterModel = { items: prevEl.querySelectorAll('figure').length, text: text(prevEl) };
  prevBtn.click(); await sleep(20);                       // and abandons one still running
  f.setSource({ provider: 'yolo', model: 'yolo12m', effort: 'fast' });
  previews[2].answer(); await sleep(20);
  out.p3 = { aborted: previews[2].signal.aborted, grid: !!prevEl.querySelector('.aib-prev-grid'), text: text(prevEl) };
  f.destroy();
}

// ---- 3. typing ids selects "These image ids": the estimate and Start use them
{
  const ds = { id: 'dsids', classes: ['person'] };
  const holder = { value: '' };
  let started = null;
  const f = m.batchForm({ fovea, ds, P, scopes: [scope('all', Array.from({ length: 7808 }, (_, i) => i + 1)), m.idsScope(fovea, holder)], scope: 'all',
    src: { provider: 'mock', model: 'mock-fast', effort: 'fast' }, onStarted: (r) => { started = r; } });
  await sleep(320);
  const est = f.el.querySelector('.aib-est'), area = f.el.querySelector('textarea');
  out.ids = { before: { radios: radios(f), est: text(est) } };
  area.fire('focus');                                     // tabbing through does not change the scope
  out.ids.afterFocus = radios(f);
  area.value = '5001 5002'; area.fire('input');
  out.ids.afterInput = radios(f);
  await sleep(320);
  out.ids.est = text(est);
  await f.start(); await sleep(20);
  const batch = calls.filter((c) => c[1] === '/api/ai/batch').pop();
  out.ids.batch = batch && batch[2].image_ids;
  out.ids.started = !!started;
  const ra = f.el.querySelectorAll('input[type=radio]')[0];
  ra.checked = true; ra.fire('change');
  area.fire('pointerdown');                               // clicking into the box selects it too
  out.ids.afterPointer = radios(f);
  f.destroy();
}

// ---- 4. Fill empty: the estimate says that images with a label file are skipped
{
  const ds = { id: 'dsfill', classes: ['person'] };
  const f = m.batchForm({ fovea, ds, P, scopes: [scope('a', [1, 2])], src: { provider: 'mock', model: 'mock-fast', effort: 'fast' } });
  await sleep(320);
  f.el.querySelectorAll('button').find((b) => b.getAttribute('data-value') === 'fill').click();
  await sleep(320);
  out.fill = { est: text(f.el.querySelector('.aib-est')), hint: f.el.querySelectorAll('.hint').map(text).filter((x) => /label file/.test(x)) };
  f.destroy();
}

// ---- 5. a double click (or Enter twice) on Start while the first requests are slow starts exactly one job
{
  const ds = { id: 'dsdbl', classes: ['person'] };
  let resolves = 0;
  const slow = { id: 'list', label: 'LIST', count: async () => { await sleep(80); return 7808; }, resolve: async () => { resolves++; await sleep(80); return { filters: {} }; }, sample: async () => [1] };
  const f = m.batchForm({ fovea, ds, P, scopes: [slow], src: { provider: 'mock', model: 'mock-fast' } });
  await sleep(500);
  const startBtn = f.footer()[1];
  const before = calls.filter((c) => c[1] === '/api/ai/batch').length;
  globalThis.BATCH_DELAY = 120;
  out.dbl = { enabledBefore: !startBtn.disabled };
  startBtn.click(); startBtn.click();                     // two clicks while the first is still resolving
  out.dbl.disabledDuring = startBtn.disabled;
  f.start();                                              // Enter pressed again
  const e2 = { type: 'click', detail: 2, target: startBtn, currentTarget: startBtn, stopPropagation() {}, preventDefault() {} };
  startBtn.listeners.click.forEach((fn) => fn(e2));       // the second click of a double click
  await sleep(450);
  out.dbl.posts = calls.filter((c) => c[1] === '/api/ai/batch').length - before;
  out.dbl.resolves = resolves;
  out.dbl.coolingDown = startBtn.disabled;                // inline form: a moment before it is ready again
  await sleep(1900);
  out.dbl.readyAgain = !startBtn.disabled;
  globalThis.BATCH_DELAY = 0;
  f.destroy();
}

// ---- 6. budget = what is left of the provider's 30-day cap (not the whole cap); a reached cap blocks Start
{
  const claude = { id: 'anthropic', name: 'Anthropic Claude', kind: 'cloud', needs_key: true, key: { present: true },
    capabilities: { detect: true, open_vocab: true, confidence: false }, config: { spend_cap_usd: 2.15, effort: 'medium' },
    models: [{ id: 'claude-haiku-5-5', name: 'Claude Haiku 5.5', default: true, price: { in: 1, out: 5 } }] };
  const PC = { providers: [yolo, claude], remote_ai_allowed: true };
  globalThis.CAPS = { anthropic: { cap_usd: 2.15, spent_30d: 1.98 } };
  globalThis.EST_HOOK = (body, base) => ({ ...base, usd: [0.0045 * base.images, 0.0458 * base.images], priced: true });
  const ds = { id: 'dscap', classes: ['person'] };
  const f = m.batchForm({ fovea, ds, P: PC, scopes: [scope('a', Array.from({ length: 40 }, (_, i) => i + 1))], src: { provider: 'anthropic', model: 'claude-haiku-5-5' } });
  await sleep(400);
  const budgetIn = f.el.querySelectorAll('input').find((x) => x.attrs.placeholder === 'none');
  const seg = f.el.querySelectorAll('.seg').find((x) => x.querySelectorAll('button').some((b) => b.getAttribute('data-value') === 'balanced'));
  out.cap = { budget: budgetIn.value, est: text(f.el.querySelector('.aib-est')), start: !f.footer()[1].disabled, effort: seg ? seg.value : null };
  await f.start(); await sleep(20);
  const batch = calls.filter((c) => c[1] === '/api/ai/batch').pop();
  out.cap.sent = { budget: batch[2].budget_usd, effort: batch[2].options.effort };
  f.destroy();
  globalThis.CAPS = { anthropic: { cap_usd: 1, spent_30d: 1.95 } };      // already over the cap
  const g = m.batchForm({ fovea, ds, P: { ...PC, caps: undefined }, scopes: [scope('a', [1, 2, 3])], src: { provider: 'anthropic', model: 'claude-haiku-5-5' } });
  await sleep(400);
  out.capReached = { est: text(g.el.querySelector('.aib-est')), start: !g.footer()[1].disabled };
  g.destroy();
  globalThis.CAPS = {};
  globalThis.EST_HOOK = null;
}

// ---- 7. a cloud model with no price: "cost unknown", no budget sent; with a spend cap set the run is refused up front
{
  const live = { id: 'anthropic', name: 'Anthropic Claude', kind: 'cloud', needs_key: true, key: { present: true },
    capabilities: { detect: true, open_vocab: true, confidence: false }, config: {},
    models: [{ id: 'claude-sonnet-9-9', name: 'Claude Sonnet 9 9', default: true }] };
  globalThis.EST_HOOK = (body, base) => ({ ...base, usd: [0, 0], priced: false, unpriced: ['claude-sonnet-9-9'] });
  const ds = { id: 'dsunpriced', classes: ['person'] };
  const PU = { providers: [yolo, live], remote_ai_allowed: true };
  const f = m.batchForm({ fovea, ds, P: PU, scopes: [scope('a', [1, 2, 3, 4])], src: { provider: 'anthropic', model: 'claude-sonnet-9-9' } });
  await sleep(400);
  const budgetIn = f.el.querySelectorAll('input').find((x) => x.attrs.placeholder === 'none');
  budgetIn.value = '5'; budgetIn.fire('change');
  await sleep(400);
  out.unpriced = { est: text(f.el.querySelector('.aib-est')), budgetDisabled: budgetIn.disabled, start: !f.footer()[1].disabled };
  await f.start(); await sleep(20);
  out.unpriced.sentBudget = 'budget_usd' in calls.filter((c) => c[1] === '/api/ai/batch').pop()[2];
  f.destroy();
  globalThis.CAPS = { anthropic: { cap_usd: 10, spent_30d: 1 } };
  const g = m.batchForm({ fovea, ds, P: { ...PU }, scopes: [scope('a', [1, 2])], src: { provider: 'anthropic', model: 'claude-sonnet-9-9' } });
  await sleep(400);
  out.unpricedCapped = { est: text(g.el.querySelector('.aib-est')), start: !g.footer()[1].disabled };
  g.destroy();
  globalThis.CAPS = {}; globalThis.EST_HOOK = null;
}

// ---- 8. the preview line: free and local for a local model
{
  const f = m.batchForm({ fovea, ds: { id: 'dshint', classes: ['person'] }, P, scopes: [scope('a', [1])], src: { provider: 'yolo', model: 'yolo12m' } });
  out.hintLocal = f.el.querySelectorAll('.hint').map(text).find((x) => /nothing is stored/.test(x));
  f.setSource({ provider: 'mock', model: 'mock-fast' });
  out.hintCloud = f.el.querySelectorAll('.hint').map(text).find((x) => /nothing is stored/.test(x));
  f.destroy();
}
// ---- 9. Annotate panel: a stock COCO model none of whose classes maps onto the dataset's (unnamed classes, as on
// vfp290k) — Propose is off with the reason and a "Map one" link; I says why; a mapping turns it on; removing it
// elsewhere (the batch dialog) turns it off again
{
  globalThis.PROVIDERS = { providers: [yolo], remote_ai_allowed: true };
  const ds = { id: 'dspanel', classes: ['class_0', 'class_1'] };
  const host = { fovea, dataset: ds, getItem: () => ({ id: 7, rel_path: 'train/a.jpg' }), getIndex: () => 0, getFilters: () => ({}), getSort: () => ({}),
    idsFrom: async () => [7], isEditable: () => true, flush: async () => 'clean', reload() {}, setProposals() {},
    getProposalThreshold: () => 0.25, setProposalThreshold() {}, onImageChange: () => () => {}, classNames: () => ds.classes };
  const el = h('div');
  const panel = m.mountAIPanel(el, host);
  await sleep(80);
  const st = () => ({ pick: text(el.querySelector('.aip-pick')), status: text(el.querySelector('.aip-status')),
    propose: !el.querySelector('.aip-go.btn-primary').disabled, links: el.querySelector('.aip-status').querySelectorAll('button').map(text) });
  out.panel = { initial: st() };
  await panel.propose();
  out.panel.afterI = st();
  out.panel.sentUnmapped = calls.filter((c) => c[1] === '/api/ai/propose').length;
  const sel = el.querySelectorAll('select').find((x) => x.getAttribute('aria-label') === 'Map another model class');
  sel.value = '0'; sel.fire('change');
  await sleep(80);
  out.panel.mapped = st();
  await panel.propose(); await sleep(20);
  const pc = calls.filter((c) => c[1] === '/api/ai/propose').pop();
  out.panel.sent = pc ? pc[2].options : null;
  out.panel.afterPropose = st();
  m.patchProfile(fovea, 'dspanel', { maps: {} });
  await sleep(80);
  out.panel.unmappedAgain = st();
  panel.destroy();
  globalThis.PROVIDERS = null;
}
console.log(JSON.stringify(out));
"""


def _run_node(tmp_path_factory, name, script):
    if not NODE:
        pytest.skip("node is not installed")
    d = tmp_path_factory.mktemp(name)
    (d / "views").mkdir()
    (d / "lib").mkdir()
    for f in ("ai-panel.js", "ai.js"):
        shutil.copy(STATIC / "views" / f, d / "views" / f)
    shutil.copy(STATIC / "lib" / "dom.js", d / "lib" / "dom.js")
    (d / "package.json").write_text('{"type":"module"}')
    (d / "run.mjs").write_text(script)
    p = subprocess.run([NODE, "run.mjs"], cwd=d, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    return _run_node(tmp_path_factory, "aiui", SCENARIOS)


@pytest.fixture(scope="module")
def form(tmp_path_factory):
    return _run_node(tmp_path_factory, "aiuiform", FORM_SCENARIOS)


def test_unreachable_compat_server_is_not_ready(results):
    down = results["down"]
    assert down["ok"] is False and down["unreachable"] is True and down["settings"] is True
    assert "Not reachable at 127.0.0.1:11434" in down["reason"]
    assert down["tone"] == "bad"


def test_pending_probe_reads_as_checking(results):
    c = results["checking"]
    assert c["ok"] is False and c["checking"] is True and c["tone"] == "checking" and c["isChecking"] is True
    assert results["describeChecking"] == {"ok": False, "checking": True}
    c2 = results["checking2"]
    assert c2["checking"] is True and c2["tone"] == "checking" and not c2.get("unreachable")
    assert results["downModel"].startswith("Not pulled on 127.0.0.1:11434")


def test_unpulled_models_are_skipped_and_explained(results):
    u = results["unpulled"]
    assert u["state"]["ok"] is True
    assert u["def"] == "llava:7b"                       # the configured default is not pulled: a pulled one is chosen
    assert u["problem"] == "Not pulled on the server"
    n = results["nonePulled"]
    assert n["ok"] is False and "No usable model" in n["reason"]


def test_picker_dims_unreachable_server_and_unpulled_models(results):
    rows = results["menu"]
    down = [r for r in rows if "Not reachable" in r["text"]]
    assert down and "open Settings" in down[0]["text"] and not down[0]["disabled"]       # one row leads to Settings
    listed = [r for r in rows if r["text"].startswith("qwen3-vl:8b")]
    assert listed and all(r["disabled"] for r in listed)                                  # its models are dimmed
    assert any(r["text"].startswith("qwen3-vl:8b") and "Not pulled" in r["text"] and r["disabled"] for r in rows)
    assert any(r["text"].startswith("LLaVA 7B") and not r["disabled"] for r in rows)


def test_generate_picker_lists_only_generate_capable_ready_providers(results):
    labels = " | ".join(results["genMenu"])
    assert "Claude Haiku 5.5" in labels and "gemini-x" in labels
    assert "yolo12m" not in labels and "qwen3-vl" not in labels


def test_economy_only_where_a_batch_api_exists(results):
    assert results["eco"] == {"cloud": True, "noBatch": False, "local": False, "twoStage": False}
    w = results["withSpeed"]
    assert w["speed"] == "economy" and w["options"] == {"effort": "low"}
    assert "speed" not in results["withSpeedNow"]


def test_estimate_leads_with_the_price_that_applies(results):
    assert results["priceEco"]["main"] == "≈ $0.50–1.00" and "now" in results["priceEco"]["other"]
    assert results["priceNow"]["main"] == "≈ $1.00–2.00" and "Economy" in results["priceNow"]["other"]


def test_provider_batch_progress_from_a_run(results):
    pb = results["pb"]
    assert pb["id"] == "msgbatch_01" and pb["batches"] == 2 and pb["state"] == "running"
    assert pb["total"] == 40 and pb["done"] == 10 and pb["submitted_at"] == 120 and pb["next_check_at"] == 999
    assert results["pbOpen"] == [True, False]
    pj = results["pbJob"]
    assert pj["state"] == "canceling" and pj["done"] == 20 and pj["filed"] == 7 and pj["next_check_at"] == 5
    fb = results["pbFallback"]
    assert fb["state"] == "canceling" and fb["total"] == 10 and fb["done"] == 4
    assert results["pbNone"] is None
    assert results["batchState"][:3] == ["Processing at the provider", "Cancelling at the provider", "Results ready"]
    assert results["batchState"][3] == "Weird state"


def test_suggestions_are_normalised_and_scrubbed(results):
    n = results["norm"]
    assert [i["cls"] for i in n["items"]] == [1]
    assert n["items"][0]["name"] == "fallen_person" and n["items"][0]["examples"][0]["image_id"] == 5
    assert n["usage"]["cost_usd"] == 0.01
    assert "SECRET" not in json.dumps(n["warnings"])
    assert results["normList"] == 1
    ns = results["normServer"]
    assert ns[0]["error"].startswith("No labeled boxes") and ns[0]["name"] == ""
    assert ns[1]["confidence"] == "medium" and ns[1]["warnings"] == ["a.jpg: bad"]
    assert results["conf"] == ["high", "medium", "low", "high", ""]


def test_name_problems_cover_duplicates_and_empties(results):
    p = results["problems"]
    assert p["0"].startswith("Same name as class 1") and "also suggested" in p["0"]      # person vs Person (case)
    assert p["1"].startswith("Same name as class 0")
    assert p["3"].startswith("Same name as class 2")                                      # 'car' already exists
    assert results["problemsOk"] == {}
    assert results["problemsEmpty"] == {"0": "Give it a name"}
    assert results["renamed"] == ["A", "b", "class_2", "d"]


def test_naming_estimate_is_priced_per_class(results):
    e = results["est"]
    assert e["priced"] is True and 0 < e["usd"][0] < e["usd"][1] < 0.05                 # 12 classes × 6 crops on a $1/$5 model
    assert results["estFree"]["priced"] is False and results["estFree"]["usd"] == [0, 0]


def test_fill_empty_wording_matches_the_runner(results):
    tip, hint = results["fillTip"], results["fillHint"]
    assert "without a label file" in tip and "empty label files are kept" in tip
    assert "have none" not in tip and "no boxes" not in hint
    assert "only for images without one" in hint and "including empty ones" in hint
    e = results["fillEffect"]
    assert e[0] == "1 image without a label file gets AI boxes. 1 with an empty label file is left alone (kept as deliberate background)."
    assert e[1] == ("3 images without a label file get AI boxes. 7 that already have a label file are left alone, "
                    "including 2 empty ones kept as deliberate background.")
    assert e[2] == "4 images without a label file get AI boxes."
    assert e[3] == "1 image without a label file gets AI boxes. 2 that already have a label file are left alone."


def test_never_sent_images_are_reported_not_resubmitted(results):
    s = results["sumNotSent"]
    assert s["notSent"] == 6808 and "6,808 never sent" in s["text"]
    p = results["sumPlain"]
    assert p["notSent"] == 0 and "never sent" not in p["text"]
    assert p["text"] == "1 image · 3 boxes written to 1 file"
    one, many, none = results["notSentText"]
    assert one.startswith("1 image was never sent") and "not resubmitted" in one and "billed twice" in one
    assert many.startswith("6,808 images were never sent")
    assert none == ""
    assert results["pbNotSent"] == 6808


def test_latest_only_drops_every_older_answer(results):
    lo = results["latest"]
    assert lo["firstStale"] is True and lo["firstAborted"] is True and lo["second"] is True and lo["running"] is True
    d = lo["afterDrop"]
    assert d == {"dropped": True, "second": False, "aborted": True, "running": False, "again": False}


def test_class_mapping_made_in_the_form_is_used_at_once(form):
    assert form["mapSelect"] is True
    b = form["mapBefore"]
    assert b["est"].startswith("Map a yolo12m class") and b["startDisabled"] is True
    a = form["mapAfter"]
    assert a["profile"] == {"yolo:yolo12m": {"0": 0}}
    assert a["est"].startswith("5 images · free") and a["startDisabled"] is False


def test_stale_preview_never_lands_under_new_settings(form):
    p1 = form["p1"]
    assert p1["sent"] == 1 and p1["ids"] == [1, 2, 3] and p1["btnDisabled"] is True
    assert p1["aborted"] is True                                  # the abandoned request is aborted
    after = p1["after"]
    assert after["grid"] is False and "Settings changed" in after["text"] and after["btnDisabled"] is False
    assert after["radios"] == [False, True]
    p2 = form["p2"]
    assert p2["ids"] == [4, 5, 6] and p2["items"] == 3 and "Mock fast" in p2["text"] and p2["btnDisabled"] is False
    assert p2["afterModel"] == {"items": 0, "text": ""}         # a finished preview of the old model is cleared
    p3 = form["p3"]
    assert p3["aborted"] is True and p3["grid"] is False and "Settings changed" in p3["text"]


def test_typing_ids_selects_the_ids_scope(form):
    i = form["ids"]
    assert i["before"]["radios"] == [True, False] and i["before"]["est"].startswith("7,808 images")
    assert i["afterFocus"] == [True, False]                      # keyboard focus alone changes nothing
    assert i["afterInput"] == [False, True]
    assert i["est"].startswith("2 images")
    assert i["batch"] == [5001, 5002] and i["started"] is True    # Start labels the typed ids, not all 7,808
    assert i["afterPointer"] == [False, True]


def test_fill_estimate_says_labelled_images_are_skipped(form):
    f = form["fill"]
    assert "already have a label file (even an empty one) are skipped" in f["est"]
    assert any("only for images without one" in x for x in f["hint"])


# ------------------------------------------------------------------ round 2 (review findings)

def test_effort_balanced_reaches_requests_and_follows_settings(results):
    e = results["effort"]
    assert e["values"] == ["low", "medium", "high", "low"] and e["labels"] == ["Fast", "Balanced", "Accurate"]
    assert e["defaults"] == ["balanced", "accurate", "fast", "fast"]          # Settings low / medium / high
    assert e["of"] == ["balanced", "accurate"]
    assert e["sent"] == "medium"                     # nothing chosen: the provider's Settings default (Balanced) is sent
    assert e["sentChosen"] == "low"
    # picking a model stores no effort (so a later Settings change applies) unless one was chosen for that provider
    assert e["picked"] == [{"provider": "anthropic", "model": "claude-haiku-5-5"},
                           {"provider": "anthropic", "model": "claude-haiku-5-5", "effort": "accurate"}]
    assert "effort" not in e["defaultSource"]


def test_reached_spend_cap_blocks_the_provider_with_numbers(results):
    r = results["capReached"]
    assert r["ok"] is False and r["settings"] is True
    assert r["reason"] == "The 30-day spend cap for Anthropic Claude is reached ($1.95 of $1.00)"
    assert results["capTone"] == "off"
    assert results["capMenu"] and "open Settings" in results["capMenu"][0]          # the picker marks it too
    left = results["capLeft"]
    assert left["state"] is True and left["cap"] == {"left": 0.17, "cap": 2.15, "name": "Anthropic Claude"}
    assert results["capLocal"] is None                                              # local models have no cap
    assert results["hasCap"] == [True, True, False]


def test_unpriced_cloud_cost_is_unknown_never_free(results):
    assert results["costKind"] == ["free", "unknown", "unknown", "priced", "unknown"]
    assert results["unpricedText"] == ["claude-sonnet-9-9", "Claude Sonnet 9 9"]
    u = results["untracked"]
    assert u == [True, False, False, True, False]     # Runs & cost: "not tracked" only for unpriced cloud runs


def test_job_box_shows_a_provider_wait_once(results):
    n = results["jobNote"]
    assert n[0] == "Anthropic Claude rate limited — retrying in 10 s"                 # counts and cost dropped
    assert n[1] == "Gemini provider busy — retrying in 3 s"
    assert n[2] == "" and n[3] == "Loading models…" and n[4] == ""
    assert n[5] == "Anthropic Claude rate limited — retrying in 7 s"


def test_preview_line_prices_only_cloud_models(results):
    local, cloud, flex = results["previewHint"]
    assert local.startswith("Runs on this machine;") and "price" not in local
    assert "standard price" in cloud and "Flex tier" in flex


def test_propose_reports_its_providers_wait(results):
    assert results["waitNames"] == [["Anthropic Claude"], [], ["Anthropic Claude"]]
    assert results["pickWait"] == {"reason": "Anthropic Claude rate limited", "seconds_left": 32}
    assert results["waitSentence"] == "Anthropic Claude rate limited — retrying in 32 s"
    assert results["pickWaitNone"] is None


def test_placeholders_split_by_boxes_and_named_in_text(results):
    assert results["ph"] == {"all": [0, 2, 3], "withBoxes": [0], "empty": [2, 3]}
    assert results["phNoCounts"]["withBoxes"] == [0]
    assert results["phCard"] == [0]                                                 # a prompt makes it described
    assert results["classList"] == ["a", "a and b", "a, b, c and 2 more"]


def test_stop_says_what_was_sent_and_billed(results):
    some, none = results["stopped"]
    assert some.startswith("Stopped: 2 classes were not sent (not billed)") and "billed" in some
    assert "every request had already been sent" in none


def test_random_samples_use_a_new_seed_per_run(results):
    assert results["seeds"] == {"distinct": 20, "ok": True}


def test_double_start_submits_one_job(form):
    d = form["dbl"]
    assert d["enabledBefore"] is True and d["disabledDuring"] is True
    assert d["posts"] == 1 and d["resolves"] == 1
    assert d["coolingDown"] is True and d["readyAgain"] is True


def test_budget_is_what_is_left_of_the_cap(form):
    c = form["cap"]
    assert float(c["budget"]) == 0.17 and c["start"] is True
    assert "Only $0.17 is left of the $2.15 30-day spend cap for Anthropic Claude" in c["est"] and "may stop early" in c["est"]
    assert c["effort"] == "balanced" and c["sent"] == {"budget": 0.17, "effort": "medium"}
    r = form["capReached"]
    assert r["est"].startswith("The 30-day spend cap for Anthropic Claude is reached ($1.95 of $1.00)") and r["start"] is False


def test_unpriced_model_sends_no_budget_and_is_refused_under_a_cap(form):
    u = form["unpriced"]
    assert u["est"].startswith("4 images · cost unknown (no price on file for claude-sonnet-9-9)") and "free" not in u["est"]
    assert u["budgetDisabled"] is True and u["start"] is True and u["sentBudget"] is False
    c = form["unpricedCapped"]
    assert "spend cap set in Settings could not stop it" in c["est"] and c["start"] is False


def test_preview_hint_in_the_form(form):
    assert form["hintLocal"].startswith("Runs on this machine;")
    assert form["hintCloud"].startswith("Runs now at the standard price;")


# ------------------------------------------------------------------ server hooks the AI UI relies on (round 2)
# GET /api/ai/waits (why a synchronous Propose takes long) and POST /api/ai/name-classes/cancel (Stop in the naming
# dialog skips the classes not sent yet). Small delimited hooks in fovea/api/ai.py and fovea/ai/assist.py.

def _hook_client(monkeypatch):
    import os
    os.environ.setdefault("FOVEA_DATA_DIR", str(Path(__file__).resolve().parent / ".tmp-data"))
    monkeypatch.setenv("FOVEA_AI_MOCK", "1")
    from fastapi.testclient import TestClient
    from fovea.ai import registry
    from fovea.main import app
    registry.reset()
    return TestClient(app)


def _naming_dataset(tmp_path, n=16):
    """n images, three placeholder classes, 1-3 boxes each: 3 classes x 8 crops do not fit one request (one per class)."""
    import uuid
    from PIL import Image
    from fovea import db
    from fovea.core import layout
    from fovea.core.scanner import scan_dataset
    from fovea.jobs import Job
    root = tmp_path / f"nm-{uuid.uuid4().hex[:6]}"
    for k in range(n):
        p = root / "images" / "train" / f"f{k:03d}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (160, 120), (100 + k, 90, 80)).save(p)
        lp = root / "labels" / "train" / f"f{k:03d}.txt"
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text("".join(f"{(k + j) % 3} {0.2 + 0.25 * j:.4f} 0.5 0.15 0.3\n" for j in range(3)))
    db.init_db()
    lay = layout.detect(str(root))
    ds_id = "aiui-" + uuid.uuid4().hex[:8]
    now = db.now()
    db.execute("INSERT INTO datasets(id,name,root,layout,classes,classes_source,description,status,created_at,updated_at)"
               " VALUES(?,?,?,?,?,?,?,?,?,?)", (ds_id, ds_id, lay.root, db.dumps(lay.to_dict()),
                                                db.dumps(["class_0", "class_1", "class_2"]), "user", "", "new", now, now))
    scan_dataset(ds_id, Job(id="s", kind="scan"))
    return ds_id


def test_waits_endpoint_lists_provider_waits(monkeypatch):
    import threading
    import time
    from fovea.ai import vlm
    c = _hook_client(monkeypatch)
    assert c.get("/api/ai/waits").json() == {"waits": []}
    tid = threading.get_ident() + 987654
    vlm._waits[tid] = {"reason": "Anthropic Claude rate limited", "until": time.time() + 30}
    try:
        w = c.get("/api/ai/waits").json()["waits"]
        assert w and w[0]["reason"] == "Anthropic Claude rate limited" and 28 <= w[0]["seconds_left"] <= 30
    finally:
        vlm._waits.pop(tid, None)


def test_name_classes_stop_skips_requests_not_sent(monkeypatch, tmp_path):
    import threading
    from fovea import db
    from fovea.ai import assist, registry
    c = _hook_client(monkeypatch)
    ds_id = _naming_dataset(tmp_path)
    try:
        # A Stop that arrives before its request registered still counts: nothing is sent, nothing is booked.
        assert c.post("/api/ai/name-classes/cancel", json={"token": "bad"}).status_code == 422
        assert c.post("/api/ai/name-classes/cancel", json={"token": "nm-early-stop-1"}).json() == {"cancelled": True}
        r = c.post("/api/ai/name-classes", json={"dataset_id": ds_id, "provider": "mock", "model": "mock-fast",
                                                 "per_class": 8, "cancel_token": "nm-early-stop-1"})
        assert r.status_code == 200, r.text
        items = r.json()["items"]
        assert len(items) == 3 and all(i.get("error") == "Stopped before this class was sent" for i in items)
        assert not db.query("SELECT 1 FROM ai_runs WHERE dataset_id=? AND kind='name_classes'", (ds_id,))
        # Stopped mid-run (one request at a time): the request already out finishes and is billed, the rest are skipped.
        ev = threading.Event()
        p = registry.get("mock")
        real = p.generate_json
        sent = []

        def first_then_stop(*a, **k):
            sent.append(1)
            ev.set()
            return real(*a, **k)
        monkeypatch.setattr(p, "generate_json", first_then_stop)
        monkeypatch.setattr("fovea.ai.config.provider_config", lambda pid: {"concurrency": 1})
        ds = db.query_one("SELECT * FROM datasets WHERE id=?", (ds_id,))
        out = assist.name_classes(ds, "mock", "mock-fast", None, 8, cancel=ev)
        named = [i for i in out["items"] if i.get("name")]
        stopped = [i for i in out["items"] if i.get("error") == "Stopped before this class was sent"]
        assert len(sent) == 1 and len(named) == 1 and len(stopped) == 2
        assert out["usage"]["cost_usd"] > 0
    finally:
        import time
        from fovea.jobs import jobs
        t0 = time.time()                 # let the post-scan checks finish before the dataset goes away
        while jobs.list(dataset_id=ds_id, active=True) and time.time() - t0 < 30:
            time.sleep(0.05)
        db.execute("DELETE FROM ai_runs WHERE dataset_id=?", (ds_id,))
        db.execute("DELETE FROM images WHERE dataset_id=?", (ds_id,))
        db.execute("DELETE FROM datasets WHERE id=?", (ds_id,))


# ---- review round 3: a model that cannot propose anything is not offered as ready; narrow captions; cards banner
def test_unmappable_coco_model_turns_propose_off_with_a_map_link(form):
    """Fresh profile on unnamed classes (vfp290k): yolo12m is the default and none of its classes maps. Propose is off
    and the status slot says so with "Map one" (the class map sits below the fold at 1280x720); I gives the reason
    without sending anything; a mapping turns Propose on and is sent; removing it elsewhere turns it off again."""
    p = form["panel"]
    assert p["initial"]["pick"].startswith("yolo12m")
    assert p["initial"]["propose"] is False
    assert p["initial"]["status"].startswith("Needs a class mapping") and p["initial"]["links"] == ["Map one"]
    assert "Press" not in p["initial"]["status"]
    assert p["afterI"]["status"].startswith("Map a yolo12m class to one of your classes first") and p["afterI"]["links"] == ["Map one"]
    assert p["sentUnmapped"] == 0
    assert p["mapped"]["propose"] is True and p["mapped"]["status"].startswith("Press I to propose")
    assert p["sent"] == {"class_map": {"0": 0}}
    assert p["afterPropose"]["status"].startswith("Nothing found on this image")
    assert p["unmappedAgain"]["propose"] is False and p["unmappedAgain"]["status"].startswith("Needs a class mapping")


def test_default_skips_a_closed_model_that_maps_nothing(results):
    d = results["defaults"]
    assert d["placeholders"] == "yolo:yolo12m"          # nothing can work on placeholder names: the map note says what to do
    assert d["named"] == "yoloe:yoloe-26s-seg"          # named classes no COCO class matches: the open-vocabulary model
    assert d["described"] == "yoloe:yoloe-26s-seg"      # a class card prompt counts as a name
    assert d["synonym"] == "yolo:yolo12m"               # "pedestrian" maps onto COCO person (server synonym)
    assert d["noOther"] == "yolo:yolo12m"               # never a cloud model by default
    assert results["known"] == [0, 1, None]


def test_picker_says_which_closed_models_need_a_mapping(results):
    h = results["mapHint"]
    assert any(x.startswith("yolo12m") and "needs a class mapping" in x for x in h["unmapped"])
    assert not any("needs a class mapping" in x for x in h["mapped"] + h["unknown"])
    assert not any("yoloe" in x and "needs" in x for x in h["unmapped"])


def test_request_for_a_closed_model_keeps_its_mapping_fields(results):
    r = results["req"]
    assert r["unmapped"] == {"ok": False, "fix": "map", "reason": "Map a yolo12m class to one of your classes first"}
    assert r["mapped"]["options"] == {"class_map": {"0": 1}} and "classes" not in r["mapped"]
    # a server without /api/ai/mapping: the names are resolved in the browser and sent
    assert r["old"]["options"] == {"class_map": {"0": 0}} and r["old"]["classes"] == [0]


def test_narrow_caption_path_keeps_split_and_frame_number(results):
    pl = results["pathLine"]
    k = pl["korean"]
    assert k["title"] == "images/test-1854-1860-포함/images_GOPR0812_01016.jpg" and "ai-pathline" in k["cls"]
    assert k["kids"] == [["ai-pathline-s", "test-1854-1860-포함"], ["ai-pathline-h", "images_GOPR"], ["ai-pathline-t", "0812_01016.jpg"]]
    assert pl["noSplit"]["kids"] == [["ai-pathline-s", "cam2"], ["ai-pathline-t", "frame_0042.jpg"]]   # short name: all of it stays
    assert pl["root"]["kids"] == [["ai-pathline-t", "x.jpg"]]


def _css_rule(css, selector):
    import re
    m = re.search(r"(?m)^" + re.escape(selector) + r"\s*\{([^}]*)\}", css)
    return m.group(1) if m else ""


def test_caption_and_banner_css():
    css = (STATIC / "css" / "ai.css").read_text(encoding="utf-8")
    line = _css_rule(css, ".ai-pathline")
    assert "overflow: hidden" in line and "nowrap" in line
    assert "flex: none" in _css_rule(css, ".ai-pathline-t")                     # the frame number never shrinks
    head, chip = _css_rule(css, ".ai-pathline-h"), _css_rule(css, ".ai-pathline-s")
    assert "text-overflow: ellipsis" in head and "text-overflow: ellipsis" in chip
    assert int(head.split("flex: 0 ")[1].split()[0]) > 1000 * int(chip.split("flex: 0 ")[1].split()[0])   # the head gives way first
    assert "flex: 1 1 0" in _css_rule(css, ".ai-cc-bar.callout > .grow")      # the banner text starts beside its icon
    js = (STATIC / "views" / "ai.js").read_text(encoding="utf-8")
    assert "pathLine(fovea, it" in js and "ai-ts-path" not in js


# ---------------------------------------------------------------------------------------------------- review round 4
# * Propose sends what was decided and edited on the image first (host.flush), then asks: queued decisions can never
#   land on the new proposals, and a rejection reaches the server before the run that must not repeat it.
# * One request per image: I on the next image while the previous one still waits starts its own request, and the
#   slot names the image a running request is for ("Proposing for a.jpg…"); a cap of three, said in words.
# * A cloud run that nothing would stop (no run budget, no 30-day cap) asks before it starts once the estimate is
#   large (or its cost cannot be known); a write run says it inside its one write confirmation.
# * The batch dialog repeats the estimate next to Start (its footer), so the price is in view whenever Start is.
# * The Economy price is named only where Economy can be picked (not for two-stage).
# * A dataset with no classes: Propose says so (no "Map one" that does nothing), and nothing asks the server for a
#   class mapping (it answers 422).
R4_SCENARIOS = FORM_PRELUDE + r"""
const proposes = [];
globalThis.PROPOSE_HOOK = (body, opts) => new Promise((resolve, reject) => {
  const p = { body, answer: (preds) => resolve({ preds: preds.map((x, i) => ({ id: 100 + proposes.length * 10 + i, image_id: body.image_id, state: 'pending', ...x })) }) };
  if (opts.signal) opts.signal.addEventListener('abort', () => { const e = new Error('aborted'); e.name = 'AbortError'; reject(e); });
  proposes.push(p);
});
const yoloe = { id: 'yoloe', name: 'YOLOE', kind: 'local', capabilities: { detect: true, open_vocab: true, confidence: true }, models: [{ id: 'yoloe-s', default: true }] };
const mkHost = (ds, first, extra = {}) => {
  const st = { cur: first, cb: null, flushes: 0, flushAnswer: 'clean', set: [], order: [] };
  const host = { fovea, dataset: ds, getItem: () => st.cur, getIndex: () => 0, getFilters: () => ({}), getSort: () => ({}), idsFrom: async () => [st.cur.id],
    isEditable: () => true, flush: async () => { st.flushes++; st.order.push('flush'); await sleep(5); return st.flushAnswer; }, reload() {},
    setProposals: (p) => { st.set.push(p.map((x) => x.id)); st.order.push('set'); }, getProposalThreshold: () => 0.25, setProposalThreshold() {},
    onImageChange: (cb) => { st.cb = cb; return () => {}; }, classNames: () => ds.classes, ...extra };
  st.go = (it) => { st.cur = it; st.cb && st.cb(); };
  return { host, st };
};
const panelState = (el) => ({ status: text(el.querySelector('.aip-status')).replace(/\u00a0/g, ' '), propose: !el.querySelector('.aip-go.btn-primary').disabled,
  stops: el.querySelector('.aip-status').querySelectorAll('button').filter((b) => text(b) === 'Stop').length,
  links: [...el.querySelector('.aip-status').querySelectorAll('button'), ...el.querySelector('.aip-status').querySelectorAll('a')].map(text).filter((t) => t !== 'Stop') });

// ---- 1. Propose: flush first; per-image requests; the slot names the other image; a cap of three
{
  globalThis.PROVIDERS = { providers: [yoloe], remote_ai_allowed: true };
  const ds = { id: 'dsprop', classes: ['person', 'car'] };
  const A = { id: 7, rel_path: 'train/a.jpg' }, B = { id: 8, rel_path: 'train/b.jpg' };
  const { host, st } = mkHost(ds, A);
  const el = h('div');
  const panel = m.mountAIPanel(el, host);
  await sleep(80);
  const r = { ready: panelState(el) };
  st.flushAnswer = 'conflict';
  await panel.propose();
  r.conflict = { ...panelState(el), posts: proposes.length, flushes: st.flushes };
  st.flushAnswer = 'clean';
  const origPost = api.post;
  panel.propose();
  await sleep(30);
  r.order = st.order.slice(-1).concat(proposes.length ? ['post:' + proposes[0].body.image_id] : []);
  r.flushBeforePost = st.flushes === 2 && proposes.length === 1;
  panel.propose(); await sleep(10);                                      // I again on the same image: nothing new
  r.samePosts = proposes.length;
  await sleep(450);
  r.busyA = panelState(el);
  st.go(B);                                                              // move on while A's request runs
  r.onB = panelState(el);
  panel.propose(); await sleep(30);
  r.postsAfterB = proposes.map((p) => p.body.image_id);
  await sleep(450);
  r.bothBusy = panelState(el);
  proposes[0].answer([{ cls: 0, xc: 0.2, yc: 0.2, w: 0.1, h: 0.1, conf: 0.9 }, { cls: 1, xc: 0.7, yc: 0.7, w: 0.1, h: 0.1, conf: 0.8 }]);
  await sleep(20);
  r.aDone = { ...panelState(el), set: st.set.slice() };
  proposes[1].answer([{ cls: 0, xc: 0.5, yc: 0.5, w: 0.2, h: 0.2, conf: 0.7 }]);
  await sleep(20);
  r.bDone = { ...panelState(el), set: st.set.slice() };
  // three running at once is the cap; the fourth image is told why
  for (const id of [9, 10, 11]) { st.go({ id, rel_path: `train/img${id}.jpg` }); panel.propose(); await sleep(30); }
  await sleep(450);
  st.go({ id: 12, rel_path: 'train/img12.jpg' });
  r.othersRunning = panelState(el);
  panel.propose(); await sleep(30);
  r.cap = { ...panelState(el), posts: proposes.length };
  // Stop on the named line abandons that image's request
  el.querySelector('.aip-status').querySelectorAll('button').find((b) => text(b) === 'Stop').click();
  await sleep(20);
  r.stopped = panelState(el);
  panel.destroy();
  out.propose = r;
}

// ---- 2. a dataset with no classes: Propose says so, and nothing asks the server for a class mapping
{
  globalThis.PROVIDERS = { providers: [yolo], remote_ai_allowed: true };
  const ds = { id: 'dsnocls', classes: [] };
  const { host } = mkHost(ds, { id: 1, rel_path: 'bus.jpg' });
  const before = calls.filter((c) => c[1] === '/api/ai/mapping').length, posts = proposes.length;
  const el = h('div');
  const panel = m.mountAIPanel(el, host);
  await sleep(120);
  const r = { initial: panelState(el) };
  await panel.propose(); await sleep(20);
  r.afterI = panelState(el);
  r.mappingCalls = calls.filter((c) => c[1] === '/api/ai/mapping').length - before;
  r.proposeCalls = proposes.length - posts;
  r.req = await m.requestFor(fovea, globalThis.PROVIDERS, m.profileOf('dsnocls'), { provider: 'yolo', model: 'yolo12m' }, [], { dsId: 'dsnocls' });
  r.closed = await m.closedMapping(fovea, 'dsnocls', m.profileOf('dsnocls'), yolo, 'yolo12m', []);
  r.mappingCalls2 = calls.filter((c) => c[1] === '/api/ai/mapping').length - before;
  panel.destroy();
  // the batch form on the AI tab: the reason, with where to add classes
  const f = m.batchForm({ fovea, ds, P: globalThis.PROVIDERS, scopes: [scope('a', [1, 2])], src: { provider: 'yolo', model: 'yolo12m' } });
  await sleep(350);
  r.form = { est: text(f.el.querySelector('.aib-est')), sum: text(f.summary), start: !f.footer()[1].disabled, hint: f.el.querySelectorAll('.aip-hint').map(text) };
  f.destroy();
  out.nocls = r;
}

// ---- 3. no spend limit: a large cloud run asks first; a cap, a budget or a small run does not
{
  const claude = { id: 'anthropic', name: 'Anthropic Claude', kind: 'cloud', needs_key: true, key: { present: true },
    capabilities: { detect: true, open_vocab: true, confidence: false, batch_api: true }, config: {},
    models: [{ id: 'claude-opus-5-5', name: 'Claude Opus 5.5', default: true, price: { in: 5, out: 25 } }] };
  const PC = { providers: [yolo, claude], remote_ai_allowed: true };
  globalThis.CAPS = {};
  let per = [0.00458, 0.04578];
  globalThis.EST_HOOK = (body, base) => ({ ...base, usd: [per[0] * base.images, per[1] * base.images], usd_batch: [per[0] * base.images / 2, per[1] * base.images / 2], priced: true,
    economy: { available: true }, seconds_per_image: 0.1, targets: { images: base.images, labeled: base.images, boxes: 3 * base.images, no_label_file: 0 } });
  const ds = { id: 'dsspend', classes: ['person'] };
  const all = Array.from({ length: 7808 }, (_, i) => i + 1);
  const batches = () => calls.filter((c) => c[1] === '/api/ai/batch');
  const r = {};
  const f = m.batchForm({ fovea, ds, P: PC, scopes: [scope('all', all)], src: { provider: 'anthropic', model: 'claude-opus-5-5' } });
  await sleep(400);
  r.est = text(f.el.querySelector('.aib-est'));
  r.sum = text(f.summary);
  globalThis.CONFIRMS = []; globalThis.CONFIRM_ANSWER = false;
  let n0 = batches().length;
  await f.start(); await sleep(20);
  const budgetIn = f.el.querySelectorAll('input').find((x) => x.attrs.placeholder === 'none');
  r.declined = { confirms: globalThis.CONFIRMS.map((c) => ({ title: c.title, ok: c.okLabel, cancel: c.cancelLabel, msg: typeof c.message === 'string' ? c.message : text(c.message) })),
    posts: batches().length - n0, err: text(f.el.querySelector('.aib-err')), focused: globalThis.FOCUSED === budgetIn };
  globalThis.CONFIRMS = []; globalThis.CONFIRM_ANSWER = true;
  n0 = batches().length;
  await f.start(); await sleep(1700);
  r.accepted = { confirms: globalThis.CONFIRMS.length, posts: batches().length - n0, budget: batches().length ? (batches().pop()[2].budget_usd ?? null) : null };
  budgetIn.value = '10'; budgetIn.fire('change'); await sleep(400);
  globalThis.CONFIRMS = []; n0 = batches().length;
  await f.start(); await sleep(1700);
  r.withBudget = { confirms: globalThis.CONFIRMS.length, posts: batches().length - n0, budget: batches().pop()[2].budget_usd };
  f.destroy();
  // small run (high estimate under $5): no question
  per = [0.0001, 0.0005];
  const g = m.batchForm({ fovea, ds: { id: 'dsspend2', classes: ['person'] }, P: PC, scopes: [scope('a', all.slice(0, 50))], src: { provider: 'anthropic', model: 'claude-opus-5-5' } });
  await sleep(400);
  globalThis.CONFIRMS = []; n0 = batches().length;
  await g.start(); await sleep(20);
  r.small = { confirms: globalThis.CONFIRMS.length, posts: batches().length - n0 };
  g.destroy();
  // a 30-day cap in Settings stops a run too: no question
  per = [0.00458, 0.04578];
  globalThis.CAPS = { anthropic: { cap_usd: 50, spent_30d: 0 } };
  const k = m.batchForm({ fovea, ds: { id: 'dsspend3', classes: ['person'] }, P: { ...PC }, scopes: [scope('all', all)], src: { provider: 'anthropic', model: 'claude-opus-5-5' } });
  await sleep(450);
  globalThis.CONFIRMS = []; n0 = batches().length;
  await k.start(); await sleep(20);
  r.capped = { confirms: globalThis.CONFIRMS.length, posts: batches().length - n0, budget: batches().pop()[2].budget_usd };
  k.destroy();
  globalThis.CAPS = {};
  // a write run: the spend line is inside its one write confirmation
  const w = m.batchForm({ fovea, ds: { id: 'dsspend4', classes: ['person'] }, P: PC, scopes: [scope('all', all)], src: { provider: 'anthropic', model: 'claude-opus-5-5' } });
  await sleep(400);
  w.el.querySelectorAll('button').find((b) => b.getAttribute('data-value') === 'append').click();
  await sleep(400);
  globalThis.CONFIRMS = []; globalThis.CONFIRM_ANSWER = true; n0 = batches().length;
  await w.start(); await sleep(20);
  r.write = { confirms: globalThis.CONFIRMS.map((c) => ({ title: c.title, msg: typeof c.message === 'string' ? c.message : text(c.message) })), posts: batches().length - n0 };
  w.destroy();
  // a model with no price on file: its cost cannot be known — asked from 20 images on
  globalThis.EST_HOOK = (body, base) => ({ ...base, usd: [0, 0], priced: false, unpriced: ['claude-x'] });
  const live = { ...claude, models: [{ id: 'claude-x', name: 'Claude X', default: true }] };
  const u = m.batchForm({ fovea, ds: { id: 'dsspend5', classes: ['person'] }, P: { providers: [yolo, live], remote_ai_allowed: true }, scopes: [scope('a', all.slice(0, 25))], src: { provider: 'anthropic', model: 'claude-x' } });
  await sleep(400);
  globalThis.CONFIRMS = []; globalThis.CONFIRM_ANSWER = false; n0 = batches().length;
  await u.start(); await sleep(20);
  r.unknown = { confirms: globalThis.CONFIRMS.map((c) => ({ title: c.title, ok: c.okLabel, cancel: c.cancelLabel, msg: c.message })), posts: batches().length - n0 };
  u.destroy();
  globalThis.CONFIRM_ANSWER = true; globalThis.EST_HOOK = null;
  out.spend = r;
}

// ---- 4. two-stage: no Economy price where Economy cannot be picked
{
  const claude = { id: 'anthropic', name: 'Anthropic Claude', kind: 'cloud', needs_key: true, key: { present: true },
    capabilities: { detect: true, classify: true, open_vocab: true, confidence: false, batch_api: true }, config: {},
    models: [{ id: 'claude-haiku-5-5', name: 'Claude Haiku 5.5', default: true, price: { in: 1, out: 5 } }] };
  const PT = { providers: [yolo, claude], remote_ai_allowed: true };
  m.patchProfile(fovea, 'dstwo', { twoStage: { detector: { provider: 'yolo', model: 'yolo12m', source_classes: ['person'] }, classifier: { provider: 'anthropic', model: 'claude-haiku-5-5' } } });
  globalThis.EST_HOOK = (body, base) => ({ ...base, usd: [0.00012 * base.images, 0.00155 * base.images], usd_batch: [0.00006 * base.images, 0.00078 * base.images], priced: true,
    economy: { available: false, reason: 'Two-stage runs box by box, so it has no batch mode' } });
  const f = m.batchForm({ fovea, ds: { id: 'dstwo', classes: ['fall', 'standing'] }, P: PT, scopes: [scope('a', Array.from({ length: 20 }, (_, i) => i + 1))], src: { pipeline: 'two_stage' } });
  await sleep(400);
  out.twoStage = { est: text(f.el.querySelector('.aib-est')), sum: text(f.summary) };
  f.destroy();
  globalThis.EST_HOOK = null;
  const est = { usd: [1, 2], usd_batch: [0.5, 1] };
  out.priceLine = { ok: m.priceLine(est, 'now', true), notOk: m.priceLine(est, 'now', false), serverNo: m.priceLine({ ...est, economy: { available: false } }, 'now') };
}

// ---- 6. a proposals batch started from Annotate sends the open image's edits and decisions first; a conflict stops it
{
  const order = [];
  let answer = 'clean';
  const host = { flush: async () => { order.push('flush'); await sleep(5); return answer; }, getItem: () => ({ id: 3, rel_path: 'x.jpg' }) };
  const origPost = api.post;
  api.post = (path, body, o) => { if (path === '/api/ai/batch') order.push('batch'); return origPost(path, body, o); };
  const f = m.batchForm({ fovea, ds: { id: 'dsflushb', classes: ['person'] }, P: { providers: [yoloe], remote_ai_allowed: true }, host, inModal: true, scopes: [scope('this', [3])], src: { provider: 'yoloe', model: 'yoloe-s' } });
  await sleep(350);
  answer = 'conflict';
  await f.start(); await sleep(20);
  out.batchFlush = { conflict: order.slice(), err: text(f.el.querySelector('.aib-err')) };
  answer = 'clean'; order.length = 0;
  await f.start(); await sleep(20);
  out.batchFlush.clean = order.slice();
  f.destroy();
  api.post = origPost;
}

// ---- 5. one box, not "1 boxes"
out.plural = {
  progress: m.progressText(fovea, { status: 'running', done: 1, total: 50, extra: { boxes: 1, cost_usd: 0.00013 } }),
  progressAdded: m.progressText(fovea, { status: 'running', done: 1, total: 2, extra: { added: 1 } }),
  summary: m.jobSummary(fovea, { status: 'done', total: 1, result: { images: 1, boxes: 1 } }).text,
  summary2: m.jobSummary(fovea, { status: 'done', total: 3, result: { images: 3, boxes: 2 } }).text,
};
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def r4(tmp_path_factory):
    return _run_node(tmp_path_factory, "aiuir4", R4_SCENARIOS)


def test_propose_saves_decisions_and_edits_first(r4):
    p = r4["propose"]
    assert p["ready"]["propose"] is True
    # an unsaved edit that cannot be saved (a conflict): nothing is asked, and the slot says why
    assert p["conflict"]["posts"] == 0 and p["conflict"]["flushes"] == 1
    assert p["conflict"]["status"].startswith("Save the edits on this image first")
    # the flush comes before the request
    assert p["flushBeforePost"] is True and p["order"] == ["flush", "post:7"]
    assert p["samePosts"] == 1                       # I again on the same image starts nothing new


def test_propose_runs_per_image_and_names_the_other_image(r4):
    p = r4["propose"]
    assert p["busyA"]["status"].startswith("Proposing…") and p["busyA"]["propose"] is False
    # on the next image: a named line for a.jpg, and Propose is on for this one
    assert "Proposing for a.jpg" in p["onB"]["status"] and p["onB"]["propose"] is True and p["onB"]["stops"] == 1
    assert "Press I to propose for this image too" in p["onB"]["status"]
    assert p["postsAfterB"] == [7, 8]                # I on b.jpg asked for b.jpg
    assert p["bothBusy"]["status"].startswith("Proposing…") and "Proposing for a.jpg" in p["bothBusy"]["status"]
    # a.jpg answers while b.jpg is on screen: saved for a.jpg, nothing lands on b.jpg
    assert "2 proposals saved for a.jpg" in p["aDone"]["status"] and p["aDone"]["set"] == []
    assert p["aDone"]["status"].startswith("Proposing…")      # b.jpg's own request is still running
    assert len(p["bDone"]["set"]) == 1 and len(p["bDone"]["set"][0]) == 1 and p["bDone"]["status"].startswith("1 proposal")
    # three at once is the cap, and the fourth image says so
    assert "Proposing for img11.jpg +2" in p["othersRunning"]["status"]
    assert p["cap"]["posts"] == 5 and "Already proposing for 3 images" in p["cap"]["status"]
    assert "Stopped waiting for img11.jpg" in p["stopped"]["status"]


def test_no_classes_says_so_and_asks_no_mapping(r4):
    n = r4["nocls"]
    assert n["initial"]["status"].startswith("No classes yet") and n["initial"]["propose"] is False
    assert n["initial"]["links"] == ["Add one"] and "Map one" not in n["initial"]["status"]
    assert n["afterI"]["status"].startswith("No classes yet")
    assert n["mappingCalls"] == 0 and n["mappingCalls2"] == 0 and n["proposeCalls"] == 0
    assert n["req"]["ok"] is False and n["req"]["fix"] == "classes"
    assert n["closed"]["mapped"] == 0 and n["closed"]["noClasses"] is True
    f = n["form"]
    assert f["est"].startswith("This dataset has no classes yet") and "Add classes in Overview" in f["est"]
    assert f["sum"].startswith("This dataset has no classes yet") and f["start"] is False
    assert any("Add classes in Overview" in x for x in f["hint"])


def test_large_cloud_run_without_a_cap_asks_first(r4):
    s = r4["spend"]
    assert "No spend cap" in s["est"]                                    # said up front in the estimate
    assert s["sum"].startswith("7,808 images · ≈ $35.76–357.45")         # the dialog footer's line
    d = s["declined"]
    assert len(d["confirms"]) == 1 and d["confirms"][0]["title"] == "Start a cloud run with no spend cap?"
    assert "$35.76–357.45" in d["confirms"][0]["msg"] and "7,808 images on Claude Opus 5.5" in d["confirms"][0]["msg"]
    assert d["confirms"][0]["cancel"] == "Set a cap" and d["posts"] == 0
    assert d["err"].startswith("Not started. Enter a spend cap") and d["focused"] is True
    assert s["accepted"] == {"confirms": 1, "posts": 1, "budget": None}
    assert s["withBudget"] == {"confirms": 0, "posts": 1, "budget": 10}
    assert s["small"] == {"confirms": 0, "posts": 1}
    assert s["capped"]["confirms"] == 0 and s["capped"]["budget"] == 50   # the cap's remainder is the run budget
    w = s["write"]
    assert len(w["confirms"]) == 1 and w["confirms"][0]["title"].startswith("Write AI labels to 7,808 images")
    assert "No spend cap is set: ≈ $35.76–357.45" in w["confirms"][0]["msg"] and w["posts"] == 1
    u = s["unknown"]
    assert len(u["confirms"]) == 1 and u["confirms"][0]["title"] == "Start a run whose cost is unknown?"
    assert "claude-x" in u["confirms"][0]["msg"] and u["confirms"][0]["cancel"] == "Cancel" and u["posts"] == 0


def test_two_stage_names_no_economy_price(r4):
    t = r4["twoStage"]
    assert "Economy" not in t["est"] and "Economy" not in t["sum"] and "$0.0024–0.03" in t["est"]
    pl = r4["priceLine"]
    assert pl["ok"]["other"].startswith("Economy") and pl["notOk"]["other"] is None and pl["serverNo"]["other"] is None


def test_one_box_reads_one_box(r4):
    p = r4["plural"]
    assert "1 box ·" in p["progress"] + " ·" and "1 boxes" not in p["progress"]
    assert "1 box added" in p["progressAdded"]
    assert p["summary"] == "1 image · 1 box" and p["summary2"] == "3 images · 2 boxes"


def test_batch_from_annotate_flushes_first(r4):
    b = r4["batchFlush"]
    assert b["conflict"] == ["flush"] and b["err"].startswith("The current image has unsaved or conflicting edits")
    assert b["clean"] == ["flush", "batch"]


def test_batch_dialog_footer_carries_the_estimate():
    js = (STATIC / "views" / "ai-panel.js").read_text(encoding="utf-8")
    assert "footer: () => [form.summary, ...form.footer()]" in js
    css = (STATIC / "css" / "ai.css").read_text(encoding="utf-8")
    sum_rule = _css_rule(css, ".aib-sum")
    assert "flex: 1 1 0" in sum_rule and "text-overflow: ellipsis" in sum_rule and "nowrap" in sum_rule


def test_runs_table_status_is_one_short_chip():
    js = (STATIC / "views" / "ai.js").read_text(encoding="utf-8")
    assert "`economy · ${batchStateText" not in js and "` ${duration(dur)}`" not in js     # no suffix after the chip
    assert "Took ${duration(dur)}" in js
    css = (STATIC / "css" / "ai.css").read_text(encoding="utf-8")
    assert "max-width: 200px" in _css_rule(css, ".ai-runs-src .truncate") and "white-space: normal" in _css_rule(css, ".ai-runs td.ai-runs-kind")


# ------------------------------------------------------------------------------------------- remote browsers, review links
# * Whether this browser is on the computer Fovea runs on is the SERVER's answer (GET /api/meta → server.request_is_local,
#   the address its remote-AI guard checks), not the page's host name: in Docker the host's browser opens
#   http://localhost:8000 yet reaches the container from the bridge gateway, and the server refuses cloud calls (403).
#   The host name is only the fallback when the server does not say. The answer is asked once per page.
# * Blocked is what the server blocks (api/ai.py _guard_remote): cloud providers and a local server sent the owner's
#   saved key (`guarded`); a key-less local server still works. From another machine the reason is the remote rule,
#   never "Add key in Settings…" (Settings cannot add one from there either).
# * The audit review links name order=desc, so "worst first" opens on the worst image whatever a view's default is.
REMOTE_SCENARIOS = FORM_PRELUDE + r"""
const yoloe = { id: 'yoloe', name: 'YOLOE', kind: 'local', guarded: false, capabilities: { detect: true, open_vocab: true, confidence: true }, models: [{ id: 'yoloe-s', default: true }] };
const claude = { id: 'anthropic', name: 'Anthropic Claude', kind: 'cloud', guarded: true, needs_key: true, key: { present: true, hint: '…0001' },
  capabilities: { detect: true, classify: true, generate: true, open_vocab: true }, models: [{ id: 'claude-haiku-5-5', name: 'Claude Haiku 5.5', default: true, price: { in: 1, out: 5 } }] };
const claudeNoKey = { ...claude, key: { present: false } };
const compatKeyed = { id: 'openai_compat', name: 'OpenAI-compatible', kind: 'local', guarded: true, capabilities: { detect: true, open_vocab: true }, status: { available: true }, models: [{ id: 'qwen', default: true }] };
const compatFree = { ...compatKeyed, guarded: false };
const cloudOld = { ...claude, guarded: undefined };          // an older server without the `guarded` flag: kind decides
const remoteMeta = { server: { request_is_local: false, remote_ai_allowed: false, loopback_only: false, bind_host: '0.0.0.0' } };
async function caseOf(tag, { hostname, meta, allowed = false }) {
  globalThis.location = { hostname };
  const mm = await import(`./views/ai-panel.js?${tag}`);           // a fresh module: nothing cached from another case
  const log = [];
  const fv = { api: { get: async (path) => {
    log.push(path);
    if (path === '/api/meta') { if (meta instanceof Error) throw meta; return meta; }
    if (path === '/api/ai/usage') return { caps: {} };
    if (path === '/api/ai/providers') return { providers: [yoloe, claude, compatKeyed], remote_ai_allowed: allowed };
    return {};
  } } };
  const P = await mm.loadProviders(fv, { force: true });
  const st = (p) => { const s = mm.providerState(P, p); return { ok: s.ok, reason: s.reason || '', remote: !!s.remote, settings: !!s.settings }; };
  const r = { requestIsLocal: P.request_is_local, local: mm.isLocalBrowser(P), note: mm.remoteNoteShown(P),
    claude: st(claude), claudeNoKey: st(claudeNoKey), yoloe: st(yoloe), compatKeyed: st(compatKeyed), compatFree: st(compatFree), cloudOld: st(cloudOld),
    firstUsable: (mm.firstUsable(P, mm.canClassify, { cloudFirst: true }) || {}).id || null };
  await mm.loadProviders(fv, { force: true });
  await mm.loadProviders(fv, { force: true });
  r.metaCalls = log.filter((x) => x === '/api/meta').length;
  return r;
}
out.docker = await caseOf('docker', { hostname: 'localhost', meta: remoteMeta });
out.allowed = await caseOf('allowed', { hostname: 'localhost', meta: { server: { ...remoteMeta.server, remote_ai_allowed: true } }, allowed: true });
out.lanLocal = await caseOf('lanlocal', { hostname: '192.168.0.5', meta: { server: { request_is_local: true, remote_ai_allowed: false } } });
out.fallbackRemote = await caseOf('fbremote', { hostname: '192.168.0.5', meta: Object.assign(new Error('Not Found'), { status: 404 }) });
out.fallbackLocal = await caseOf('fblocal', { hostname: 'localhost', meta: { version: '1.0' } });   // an older server: no `server`
globalThis.location = { hostname: 'localhost' };

// The Annotate panel in the Docker case: a remembered cloud source is off with the reason, and I asks nothing.
{
  const origGet = api.get;
  api.get = async (path, q) => { if (path !== '/api/meta') return origGet(path, q); calls.push(['GET', path, q]); return remoteMeta; };
  globalThis.PROVIDERS = { providers: [yoloe, claude], remote_ai_allowed: false };
  const ds = { id: 'dsremote', classes: ['person', 'car'] };
  m.patchProfile(fovea, 'dsremote', { source: { provider: 'anthropic', model: 'claude-haiku-5-5' } });
  const host = { fovea, dataset: ds, getItem: () => ({ id: 7, rel_path: 'train/a.jpg' }), getIndex: () => 0, getFilters: () => ({}), getSort: () => ({}),
    idsFrom: async () => [7], isEditable: () => true, flush: async () => 'clean', reload() {}, setProposals() {},
    getProposalThreshold: () => 0.25, setProposalThreshold() {}, onImageChange: () => () => {}, classNames: () => ds.classes };
  const el = h('div');
  const panel = m.mountAIPanel(el, host);
  await sleep(80);
  const before = calls.filter((c) => c[1] === '/api/ai/propose').length;
  const state = () => ({ status: text(el.querySelector('.aip-status')), propose: !el.querySelector('.aip-go.btn-primary').disabled });
  out.panelRemote = { initial: state() };
  await panel.propose(); await sleep(20);
  out.panelRemote.afterI = state();
  out.panelRemote.posts = calls.filter((c) => c[1] === '/api/ai/propose').length - before;
  out.panelRemote.metaCalls = calls.filter((c) => c[1] === '/api/meta').length;
  panel.destroy();
  api.get = origGet;
  globalThis.PROVIDERS = null;
}

// Review links into the audit findings
const t = await import('./views/ai.js');
out.links = {
  annotate: t.auditLink('ds1', 'annotate', { ai: 'audit' }),
  image: t.auditLink('ds1', 'annotate', { img: 42, ai: 'audit' }),
  explore: t.auditLink('ds1', 'explore', { issue: 'ai_loose' }),
};
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def remote(tmp_path_factory):
    return _run_node(tmp_path_factory, "aiuiremote", REMOTE_SCENARIOS)


CLOUD_OFF = "Cloud AI is off for browsers on other machines"


def test_server_decides_a_localhost_page_is_remote(remote):
    d = remote["docker"]
    # the address bar says localhost, the server says another machine: the server wins
    assert d["requestIsLocal"] is False and d["local"] is False and d["note"] is True
    assert d["claude"] == {"ok": False, "reason": CLOUD_OFF, "remote": True, "settings": False}
    assert d["cloudOld"]["ok"] is False and d["cloudOld"]["reason"] == CLOUD_OFF
    # the remote rule comes first: Settings cannot add a key from another machine either
    assert d["claudeNoKey"]["reason"] == CLOUD_OFF and d["claudeNoKey"]["settings"] is False
    # a local server sent the owner's key is refused too; a key-less one and Fovea's own models still work
    assert d["compatKeyed"]["ok"] is False and "saved API key" in d["compatKeyed"]["reason"]
    assert d["compatFree"]["ok"] is True and d["yoloe"]["ok"] is True
    assert d["firstUsable"] != "anthropic"
    assert d["metaCalls"] == 1                      # asked once per page, not on every provider refresh


def test_remote_ai_allowed_and_lan_local_browsers_keep_cloud(remote):
    a = remote["allowed"]
    assert a["local"] is False and a["note"] is False and a["claude"]["ok"] is True and a["compatKeyed"]["ok"] is True
    # a LAN address the server counts as local (FOVEA_LOCAL_NETS): the server's answer, not the host name
    lan = remote["lanLocal"]
    assert lan["requestIsLocal"] is True and lan["local"] is True and lan["claude"]["ok"] is True and lan["note"] is False
    assert lan["firstUsable"] == "anthropic"


def test_host_name_is_only_the_fallback(remote):
    fr = remote["fallbackRemote"]
    assert fr["requestIsLocal"] is None and fr["local"] is False and fr["claude"]["reason"] == CLOUD_OFF
    assert fr["metaCalls"] == 3                     # a failed answer is not kept: asked again on the next load
    fl = remote["fallbackLocal"]
    assert fl["requestIsLocal"] is None and fl["local"] is True and fl["claude"]["ok"] is True and fl["note"] is False


def test_annotate_panel_turns_a_remote_cloud_source_off(remote):
    p = remote["panelRemote"]
    assert p["initial"]["status"].startswith(CLOUD_OFF) and p["initial"]["propose"] is False
    assert p["afterI"]["status"].startswith(CLOUD_OFF) and p["posts"] == 0      # no request the server would refuse
    assert p["metaCalls"] == 1


def test_audit_review_links_name_the_order(remote):
    l = remote["links"]
    assert l["annotate"] == "/d/ds1/annotate?ai=audit&sort=audit&order=desc"
    assert l["image"] == "/d/ds1/annotate?img=42&ai=audit&sort=audit&order=desc"
    assert l["explore"] == "/d/ds1/explore?issue=ai_loose&sort=audit&order=desc"
    js = (STATIC / "views" / "ai.js").read_text(encoding="utf-8")
    # every audit review link goes through auditLink: none hand-built without its order
    assert "sort=audit`" not in js and "sort: 'audit' }" not in js and js.count("auditLink(ds.id") == 5


def test_remote_note_in_the_ai_tab():
    js = (STATIC / "views" / "ai.js").read_text(encoding="utf-8")
    assert "remoteNoteShown(C.P)" in js and "REMOTE_AI_NOTE" in js
    panel = (STATIC / "views" / "ai-panel.js").read_text(encoding="utf-8")
    assert "location.hostname" in panel.split("export const isLocalBrowser")[0]      # only inside the fallback
    assert "isLocalBrowser()" not in panel                                            # every caller passes P
    css = (STATIC / "css" / "ai.css").read_text(encoding="utf-8")
    assert "var(--warn-soft)" in _css_rule(css, ".ai-privacy.ai-privacy-remote")


CHIP_SCENARIOS = r"""
globalThis.location = { hostname: 'localhost' };
globalThis.navigator = globalThis.navigator || { platform: 'MacIntel' };
const m = await import('./views/ai-panel.js');
const R = (x, y, w, h) => ({ left: x, top: y, right: x + w, bottom: y + h, width: w, height: h });
const L = R(0, 0, 200, 200);
// a chip as the CSS draws it: 14 px tall, on top of its box's top-left corner
const chipOf = (B, w = 90) => ({ r: R(B.left, B.top - 14, w, 14), B });
const run = (boxes) => {
  const items = boxes.map((b) => chipOf(R(...b)));
  const moves = m.placeChips(L, items);
  return items.map((it, i) => [it.r.left + moves[i].dx, it.r.top + moves[i].dy, it.r.width, it.r.height, moves[i].dx, moves[i].dy]);
};
const out = {
  pair: run([[40, 60, 40, 80], [54, 64, 44, 72]]),
  crowd: run([0, 1, 2, 3, 4].map((i) => [40 + 12 * i, 80 + 2 * i, 36, 70])),
  topEdge: run([[10, 0, 40, 50], [120, 100, 30, 30]]),
  rightEdge: run([[170, 100, 25, 50]]),
  apart: run([[10, 50, 20, 40], [104, 50, 20, 40]]),
  noRoom: m.placeChips(R(0, 0, 90, 20), [chipOf(R(0, 4, 90, 16)), chipOf(R(0, 4, 90, 16)), chipOf(R(0, 4, 90, 16))]).length,
};
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def chips(tmp_path_factory):
    return _run_node(tmp_path_factory, "aiuichips", CHIP_SCENARIOS)


def _cover(a, b):
    return (max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
            * max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])))


def test_proposal_chips_of_overlapping_boxes_do_not_cover_each_other(chips):
    """Two-stage test and preview tiles: the score chips of overlapping boxes were drawn on top of each other, and a
    chip above a box at the top edge (or running past the right edge) was cut off by the picture."""
    for name in ("pair", "crowd", "topEdge", "rightEdge", "apart"):
        rects = chips[name]
        for i, a in enumerate(rects):
            assert a[0] >= 0 and a[1] >= 0 and a[0] + a[2] <= 200 and a[1] + a[3] <= 200, (name, i, a)
            for b in rects[i + 1:]:
                assert _cover(a, b) == 0, (name, a, b)
    assert chips["pair"][0][4:] == [0, 0]                    # the first chip keeps its place above its box
    assert chips["topEdge"][0][4:] == [0, 14]                # moved inside the box's top edge, not cut off
    assert chips["rightEdge"][0][4:] == [-65, 0]             # right-aligned with its box instead of running off
    assert [r[4:] for r in chips["apart"]] == [[0, 0], [0, 0]]
    assert chips["noRoom"] == 3                               # no free spot: every chip still gets one
    panel = (STATIC / "views" / "ai-panel.js").read_text(encoding="utf-8")
    assert "new ResizeObserver(() => { if (!layer.isConnected) { ro.disconnect(); return; } declutterChips(layer); })" in panel
