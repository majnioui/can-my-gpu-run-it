import {
  GPUS, PRECISIONS, ARCHES, GiB, evaluate, minGpus, overheadGiB, supportNote, searchGpus, customGpu,
  fmtGiB, fmtParams, fmtTokens, seqBytes,
} from './calc.js';
import { loadModel, parseRepo } from './hf.js';

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

const COUNTS = [1, 2, 4, 8, 16];
const CTX_PRESETS = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 1048576];
const EXAMPLES = [
  ['DeepSeek-V3.1', 'deepseek-ai/DeepSeek-V3.1'],
  ['Kimi K2', 'moonshotai/Kimi-K2-Instruct'],
  ['MiniMax-M2', 'MiniMaxAI/MiniMax-M2'],
  ['gpt-oss-120b', 'openai/gpt-oss-120b'],
  ['Qwen3-235B', 'Qwen/Qwen3-235B-A22B-Instruct-2507'],
  ['GLM-4.6', 'zai-org/GLM-4.6'],
  ['Qwen3-32B', 'Qwen/Qwen3-32B'],
];
const PREC_RANK = { bf16: 2, fp8: 1, int8: 1, int4: 0.5, fp4: 0.5 };
const COMMON = ['h100', 'h200', 'b200', 'b300', 'l40s', 'a100', 'mi300x', 'mi355x', 'rtxpro6000', 'rtx5090', 'rtx4090'];
const CATS = [['common', 'Common'], ['dc', 'Datacenter'], ['ws', 'Workstation'], ['consumer', 'Consumer'], ['unified', 'Unified memory'], ['all', 'All']];

const state = {
  repo: '', gpu: 'h200', count: 8, prec: 'native', kv: 2, ctx: 32768, seqs: 1, util: 0.9, overhead: null,
  custom: { mem: 48, bw: 1.0, arch: 'ada' }, cat: null,
};
let model = null;
let loadSeq = 0;

const getGpu = (id = state.gpu) => (id === 'custom' ? customGpu(state.custom) : GPUS.find((g) => g.id === id));
const gpuLabel = (g) => (g.id === 'custom' ? `${state.custom.mem} GB GPU` : `${g.name} ${g.sub.replace(/ (SXM|PCIe|unified|NVL72|Blackwell)$/, '')}`);
const precName = (id) => {
  if (id !== 'native') return PRECISIONS.find((p) => p.id === id).label;
  return model ? `${model.nativePrec.toUpperCase()} (as published)` : 'its published format';
};
const plural = (n, w) => `${n.toLocaleString()} ${w}${n === 1 ? '' : 's'}`;

function parseTokens(s) {
  const m = String(s).trim().toLowerCase().replace(/[, _]/g, '').match(/^(\d+(?:\.\d+)?)([km]?)$/);
  if (!m) return null;
  const v = parseFloat(m[1]) * (m[2] === 'k' ? 1024 : m[2] === 'm' ? 1048576 : 1);
  return v >= 1 ? Math.round(v) : null;
}

// ------------------------------------------------------------ URL state

function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  if (p.get('m')) state.repo = p.get('m');
  if (GPUS.some((g) => g.id === p.get('gpu'))) state.gpu = p.get('gpu');
  if (p.get('gpu') === 'custom') {
    state.gpu = 'custom';
    if (+p.get('cm') > 0) state.custom.mem = +p.get('cm');
    if (+p.get('cb') > 0) state.custom.bw = +p.get('cb');
    if (ARCHES.some(([a]) => a === p.get('ca'))) state.custom.arch = p.get('ca');
  }
  if (COUNTS.includes(+p.get('n'))) state.count = +p.get('n');
  if (PRECISIONS.some((x) => x.id === p.get('w'))) state.prec = p.get('w');
  if (p.get('kv') === 'fp8') state.kv = 1;
  if (+p.get('ctx') > 0) state.ctx = +p.get('ctx');
  if (+p.get('seqs') > 0) state.seqs = +p.get('seqs');
  return p.has('n');
}
function writeHash() {
  const p = new URLSearchParams();
  if (state.repo) p.set('m', state.repo);
  p.set('gpu', state.gpu);
  if (state.gpu === 'custom') { p.set('cm', state.custom.mem); p.set('cb', state.custom.bw); p.set('ca', state.custom.arch); }
  p.set('n', state.count); p.set('w', state.prec);
  if (state.kv === 1) p.set('kv', 'fp8');
  p.set('ctx', state.ctx); p.set('seqs', state.seqs);
  history.replaceState(null, '', '#' + p.toString());
}

// ------------------------------------------------------------ the sentence

function renderSentence() {
  $('modelName').textContent = model ? model.repo.split('/')[1] : 'this model';
  $('count').innerHTML = COUNTS.map((n) => `<option value="${n}"${n === state.count ? ' selected' : ''}>${n}</option>`).join('');
  $('gpuBtn').textContent = gpuLabel(getGpu());
  $('prec').innerHTML = PRECISIONS.map((p) => `<option value="${p.id}"${p.id === state.prec ? ' selected' : ''}>${esc(precName(p.id))}${p.hint ? ` (${p.hint})` : ''}</option>`).join('');
  if (document.activeElement !== $('ctx')) $('ctx').value = fmtTokens(state.ctx);
  if (document.activeElement !== $('seqs')) $('seqs').value = state.seqs;
  $('seqWord').textContent = state.seqs === 1 ? 'request' : 'requests';
  const max = model?.arch.maxPos;
  $('ctxList').innerHTML = CTX_PRESETS.filter((c) => !max || c <= max).concat(max && !CTX_PRESETS.includes(max) ? [max] : [])
    .map((c) => `<option value="${fmtTokens(c)}">`).join('');
  $('customBox').hidden = state.gpu !== 'custom';
  $('cMem').value = state.custom.mem;
  $('cBw').value = state.custom.bw;
  $('cArch').innerHTML = ARCHES.map(([a, l]) => `<option value="${a}"${a === state.custom.arch ? ' selected' : ''}>${l}</option>`).join('');
}

// ------------------------------------------------------------ results

function opts(extra = {}) {
  return {
    gpu: getGpu(), count: state.count, prec: state.prec, kvBytes: state.kv, ctx: state.ctx, seqs: state.seqs,
    util: state.util, overrideOverhead: state.overhead ?? undefined, ...extra,
  };
}

function verdict(r, gpu) {
  const n = state.count;
  const setup = `${n} × ${gpuLabel(gpu)}`;
  const s = r.status;
  const color = { yes: 'var(--fit)', tight: 'var(--tight)', partial: 'var(--tight)', no: 'var(--no)' }[s];
  const answer = { yes: 'Yes, it fits.', tight: 'It fits, barely.', partial: 'Only with a shorter context.', no: 'No, it doesn’t fit.' }[s];
  let why;
  if (s === 'no') {
    why = `The weights take ${fmtGiB(r.weightsPerGpu)} on each GPU, but only ${fmtGiB(r.perGpuUsable)} is usable on each one.`;
    const m = minGpus(model, opts());
    why += m ? ` <span class="aside">The smallest setup that works is ${m.n} × ${esc(gpuLabel(gpu))}.</span>`
      : ` <span class="aside">Even 16 of them aren’t enough in ${esc(precName(state.prec))}. Try a smaller precision or a bigger GPU below.</span>`;
  } else if (s === 'partial') {
    why = `The weights fit, but ${state.seqs > 1 ? `${state.seqs} requests of ` : ''}${fmtTokens(state.ctx)} tokens need ${fmtGiB(r.kvNeeded)} of KV cache and only ${fmtGiB(r.kvBudget)} is left.` +
      ` <span class="aside">One request can go up to ${fmtTokens(r.maxCtx)} tokens${r.maxSeqs ? `, or ${plural(r.maxSeqs, 'request')} at ${fmtTokens(state.ctx)}` : ''}.</span>`;
  } else {
    why = `${setup} holds the ${fmtGiB(r.weights)} of weights with room for ${plural(r.maxSeqs, 'request')} at ${fmtTokens(state.ctx)} tokens, or one request up to ${fmtTokens(r.maxCtx)}.` +
      (s === 'tight' ? ' <span class="aside">Less than 5% of memory is left over, so expect out-of-memory errors under real load.</span>' : '');
  }
  if (r.support === 'slow' && s !== 'no') why += ` <span class="aside">${esc(precName(state.prec))} isn’t native on this GPU, so it runs slower.</span>`;

  const cap = gpu.mem * GiB;
  const reserved = cap * (1 - state.util);
  const w = r.weightsPerGpu, o = r.overhead, k = r.kvNeeded / n;
  const used = w + o + k;
  const over = Math.max(0, used - (cap - reserved));
  const scale = Math.max(cap, used + reserved);
  const pct = (x) => `${Math.max(0, (x / scale) * 100)}%`;
  const free = Math.max(0, cap - reserved - used);
  const tps = r.decodeTps >= 100 ? Math.round(r.decodeTps) : r.decodeTps.toFixed(1);

  return `
  <section class="verdict" style="--c:${color}">
    <p class="answer">${answer}</p>
    <p class="why">${why}</p>
    <div class="mem" aria-label="Memory on each GPU">
      <div class="bar">
        <span class="w" style="width:${pct(Math.min(w, cap))}"></span>
        <span class="o" style="width:${pct(o)}"></span>
        <span class="k" style="width:${pct(Math.max(0, k - over))}"></span>
        ${over > 0 ? `<span class="x" style="width:${pct(over)}"></span>` : ''}
        <span class="r" style="width:${pct(reserved + free)}"></span>
      </div>
      <ul class="keys">
        <li><i style="background:var(--ink)"></i>Weights <b>${fmtGiB(w)}</b></li>
        <li><i style="background:#8b929c"></i>Runtime <b>${fmtGiB(o)}</b></li>
        <li><i style="background:#4f7cc4"></i>KV cache <b>${fmtGiB(k)}</b></li>
        ${over > 0 ? `<li><i style="background:var(--no)"></i>Short by <b>${fmtGiB(over)}</b></li>` : `<li>Free <b>${fmtGiB(free)}</b></li>`}
        <li>of <b>${fmtGiB(cap)}</b> per GPU</li>
      </ul>
    </div>
    ${s === 'no' ? '' : `<p class="speed">One request can generate at most <b>${tps} tokens/s</b>. That’s the memory-bandwidth limit; real speed is usually 40–70% of it.</p>`}
  </section>`;
}

function gpuTable() {
  const cur = getGpu();
  const cat = state.cat ?? 'common';
  let list = cat === 'all' ? [...GPUS] : cat === 'common' ? COMMON.map((id) => getGpu(id)) : GPUS.filter((g) => g.cat === cat);
  if (!list.some((g) => g.id === cur.id)) list.unshift(cur);
  let anySlow = false;
  const rows = list.map((g) => {
    const cells = PRECISIONS.map((p) => {
      const m = minGpus(model, opts({ gpu: g, prec: p.id }));
      if (!m) return '<td class="none">—</td>';
      const slow = m.r.support === 'slow';
      anySlow ||= slow;
      const sel = g.id === state.gpu && p.id === state.prec && m.n === state.count ? ' sel' : '';
      return `<td class="${m.r.status}${sel}"><button type="button" data-g="${g.id}" data-p="${p.id}" data-n="${m.n}" title="${m.n} × ${esc(gpuLabel(g))} in ${esc(precName(p.id))}">${m.n}${slow ? '<sup>*</sup>' : ''}</button></td>`;
    }).join('');
    return `<tr class="${g.id === cur.id ? 'cur' : ''}"><th scope="row">${esc(g.name)}<small>${esc(g.sub)}</small></th>${cells}</tr>`;
  }).join('');
  return `
  <section class="block">
    <h2>GPUs needed</h2>
    <p class="sub">The fewest GPUs of each type that run ${esc(model.repo.split('/')[1])} with ${fmtTokens(state.ctx)} tokens for ${plural(state.seqs, 'request')}. Pick a number to use that setup.</p>
    <div class="tabs" id="tabs" role="tablist">${CATS.map(([id, l]) => `<button type="button" role="tab" data-cat="${id}" class="${id === cat ? 'on' : ''}" aria-selected="${id === cat}">${l}</button>`).join('')}</div>
    <div class="table-wrap">
      <table id="gpuTable">
        <thead><tr><th>GPU</th>${PRECISIONS.map((p) => `<th>${p.id === 'native' ? `As published<br>${model.nativePrec.toUpperCase()}` : p.label}</th>`).join('')}</tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    <p class="legend">— means more than 16 GPUs.${anySlow ? ' * runs, but not natively on that GPU, so it’s slower.' : ''}</p>
  </section>`;
}

function facts() {
  const a = model.arch;
  const kvTok = seqBytes(a, 1, 2, 65536) / 65536;
  const layerParts = a.compress
    ? [...new Set(a.compress)].sort((x, y) => x - y).map((r) => `${a.compress.filter((x) => x === r).length} ${r ? `with 1/${r} compressed KV` : 'window only'}`)
    : [a.layers.full && `${a.layers.full} full attention`, a.layers.sliding && `${a.layers.sliding} sliding window (${fmtTokens(a.window)})`, a.layers.linear && `${a.layers.linear} linear attention`].filter(Boolean);
  const rows = [
    ['Parameters', `${fmtParams(model.params)}${model.paramsEstimated ? ' (estimated)' : ''}${a.isMoE && !a.activeUncertain ? `, about ${fmtParams(model.active)} active per token` : ''}`],
    a.isMoE && ['Experts', `${a.experts}, with ${a.topk} used per token`],
    ['Published weights', `${model.nativeBytes ? fmtGiB(model.nativeBytes) + ' in ' : ''}${model.nativePrec.toUpperCase()}`],
    ['Attention', a.attnKind],
    ['Layers', `${a.L}${layerParts.length > 1 || a.layers.full !== a.L ? `: ${layerParts.join(', ')}` : ''}`],
    ['KV cache per token', `${(kvTok / 1024).toFixed(1)} KiB in BF16${a.layers.full !== a.L ? ', averaged over 64K tokens' : ''}`],
    ['Longest context', `${fmtTokens(a.maxPos)} tokens`],
  ].filter(Boolean);
  return `
  <section class="block">
    <h2><a href="https://huggingface.co/${esc(model.repo)}" target="_blank" rel="noopener">${esc(model.repo)}</a></h2>
    <dl class="facts">${rows.map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join('')}</dl>
  </section>`;
}

function notes(r, gpu) {
  const a = model.arch;
  const out = [...model.warnings.map(esc)];
  const prec = state.prec === 'native' ? model.nativePrec : state.prec;
  const note = supportNote(gpu.arch, prec);
  if (note) out.push(note);
  if (state.prec !== 'native' && state.prec !== model.nativePrec) {
    out.push(PREC_RANK[state.prec] > PREC_RANK[model.nativePrec]
      ? `The published weights are already ${model.nativePrec.toUpperCase()}. Running them in ${precName(state.prec)} uses more memory without improving quality.`
      : `${precName(state.prec)} needs a quantized copy of the model (search Hugging Face for “${esc(model.repo.split('/')[1])} ${state.prec.toUpperCase()}”)${state.prec === 'fp8' ? ', or vLLM can quantize on load with <code>--quantization fp8</code>' : ''}.`);
  }
  if (a.mla && state.count > 1) out.push(`With tensor parallelism, every GPU keeps its own copy of this model’s compressed KV cache, which is what’s shown here. Data-parallel attention (SGLang <code>--enable-dp-attention</code>) fits about ${state.count}× more context.`);
  if (!a.mla && !a.compress && a.nKv < state.count) out.push(`The model has only ${a.nKv} KV heads, so across ${state.count} GPUs the KV cache is stored ${state.count / a.nKv} times.`);
  if (!r.tpValid) out.push(`${a.nHeads} attention heads don’t split evenly across ${state.count} GPUs, so tensor parallelism alone won’t load. Use pipeline parallelism or another GPU count.`);
  if (!gpu.link && gpu.cat !== 'unified' && state.count > 1) out.push(`${esc(gpu.name)} has no NVLink, so GPUs talk over PCIe. Pipeline parallelism (<code>--pipeline-parallel-size</code>) is often faster than tensor parallelism here.`);
  if (gpu.cat === 'unified') {
    out.push(`${esc(gpu.name)} shares memory with the system. ${gpu.mem.toFixed(0)} GiB is what the GPU can use by default${gpu.vendor === 'apple' ? '; raise it with <code>sudo sysctl iogpu.wired_limit_mb=…</code>' : ''}. Most people run llama.cpp, MLX, Ollama or LM Studio here. A Q4_K_M GGUF is close to INT4, and Q8_0 to INT8.`);
    if (state.count > 1) out.push('Several of these machines can only be joined over the network (exo, llama.cpp RPC), which is far slower than NVLink.');
  } else if (gpu.cat === 'consumer') {
    out.push('On consumer cards most people run llama.cpp, Ollama or LM Studio. A Q4_K_M GGUF is close to INT4, and Q8_0 to INT8. llama.cpp can spill layers into system RAM when it doesn’t quite fit, at a big speed cost.');
  }
  if (state.count === 16) out.push('16 GPUs means two servers of 8, joined by InfiniBand or RoCE.');
  if (a.compress) out.push(`DeepSeek-V4 keeps the last ${a.window} tokens in every layer, plus a compressed KV cache of the whole context in ${a.layers.full} layers. This is read from config.json; serving engines are still tuning support, so confirm the numbers on yours.`);
  else if (a.layers.sliding) out.push(`${a.layers.sliding} of ${a.L} layers only look at the last ${fmtTokens(a.window)} tokens, so their KV cache stops growing there.`);
  if (a.layers.linear) out.push(`${a.layers.linear} of ${a.L} layers use linear attention, which keeps a fixed ${fmtGiB(a.linearStateBytes * a.layers.linear)} per request instead of a cache that grows with context.`);
  if (state.ctx > a.maxPos) out.push(`You asked for more context than the model supports (${fmtTokens(a.maxPos)} tokens).`);
  if (!out.length) return '';
  return `<section class="block"><h2>Notes</h2><ul class="notes">${out.map((x) => `<li>${x}</li>`).join('')}</ul></section>`;
}

function assumptions() {
  const a = model.arch;
  return `
  <details class="assume" id="assume"${state.kv === 1 || state.util !== 0.9 || state.overhead != null ? ' open' : ''}>
    <summary>Assumptions you can change</summary>
    <div class="assume-grid">
      <label for="kv">KV cache precision</label>
      <select id="kv"><option value="2"${state.kv === 2 ? ' selected' : ''}>BF16</option><option value="1"${state.kv === 1 ? ' selected' : ''}>FP8 (half the memory)</option></select>
      <label for="util">Share of GPU memory used</label>
      <input id="util" type="number" min="0.5" max="0.99" step="0.01" value="${state.util}">
      <label for="overhead">Runtime memory per GPU, GiB</label>
      <input id="overhead" type="number" min="0" step="0.5" value="${state.overhead ?? ''}" placeholder="${overheadGiB(a, state.count).toFixed(1)} (auto)">
    </div>
  </details>`;
}

function render() {
  renderSentence();
  writeHash();
  if (!model) return;
  const gpu = getGpu();
  const r = evaluate(model, opts());
  const open = $('assume')?.open;
  $('out').innerHTML = verdict(r, gpu) + gpuTable() + facts() + notes(r, gpu) + assumptions();
  if (open) $('assume').open = true;

  $('tabs').onclick = (e) => { const c = e.target.dataset?.cat; if (c) { state.cat = c; render(); } };
  $('gpuTable').onclick = (e) => {
    const b = e.target.closest('button[data-g]');
    if (!b) return;
    Object.assign(state, { gpu: b.dataset.g, prec: b.dataset.p, count: +b.dataset.n });
    render();
    $('sentence').scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' });
  };
  $('kv').onchange = (e) => { state.kv = +e.target.value; render(); };
  $('util').onchange = (e) => { const v = +e.target.value; if (v >= 0.5 && v <= 0.99) state.util = v; render(); };
  $('overhead').onchange = (e) => { const v = e.target.value.trim(); state.overhead = v === '' || !(+v >= 0) ? null : +v; render(); };
}

function setStatus(msg, isError = false) {
  const el = $('status');
  el.hidden = !msg;
  el.className = 'status' + (isError ? ' error' : '');
  el.innerHTML = msg ?? '';
}

async function run(input, { keepCount = false } = {}) {
  const repo = parseRepo(input);
  if (!repo) { setStatus('Enter a model as organization/name, like deepseek-ai/DeepSeek-V3, or paste its Hugging Face link.', true); return; }
  const my = ++loadSeq;
  $('repo').value = repo;
  $('go').disabled = true;
  setStatus(`Reading ${esc(repo)} from Hugging Face…`);
  try {
    const m = await loadModel(repo, localStorage.getItem('hfToken') || undefined);
    if (my !== loadSeq) return;
    model = m;
    state.repo = m.repo;
    if (state.ctx > m.arch.maxPos) state.ctx = m.arch.maxPos;
    if (!keepCount) { const best = minGpus(model, opts()); if (best) state.count = best.n; }
    setStatus(null);
    render();
  } catch (e) {
    if (my !== loadSeq) return;
    model = null;
    $('out').innerHTML = '';
    render();
    setStatus(esc(e.message), true);
    if (e.gated) $('tokenBox').hidden = false;
  } finally {
    if (my === loadSeq) $('go').disabled = false;
  }
}

// ------------------------------------------------------------ wiring

$('searchForm').onsubmit = (e) => { e.preventDefault(); run($('repo').value); };
$('examples').innerHTML = 'Try ' + EXAMPLES.map(([l, m], i) => `<button type="button" data-m="${m}">${l}</button>${i < EXAMPLES.length - 2 ? ', ' : i === EXAMPLES.length - 2 ? ' or ' : ''}`).join('') + '.';
$('examples').onclick = (e) => { const m = e.target.dataset?.m; if (m) run(m); };

$('count').onchange = (e) => { state.count = +e.target.value; render(); };
$('prec').onchange = (e) => { state.prec = e.target.value; render(); };
const commitCtx = () => { const v = parseTokens($('ctx').value); if (v) state.ctx = v; $('ctx').blur(); render(); };
$('ctx').addEventListener('change', commitCtx);
$('ctx').addEventListener('keydown', (e) => { if (e.key === 'Enter') commitCtx(); });
$('seqs').addEventListener('change', () => { const v = Math.round(+$('seqs').value); if (v >= 1) state.seqs = v; render(); });

// GPU picker
const pop = $('gpuPop'), gs = $('gpuSearch'), gl = $('gpuList');
gs.placeholder = `Search ${GPUS.length} GPUs, e.g. 4090, MI300, Mac`;
let hits = [], active = 0;
function drawList() {
  const q = gs.value.trim();
  hits = searchGpus(q);
  const items = hits.map((g, i) => `<span class="opt${i === active ? ' act' : ''}${g.id === state.gpu ? ' cur' : ''}" role="option" data-i="${i}">${esc(g.name)} ${esc(g.sub)}<small>${g.mem.toFixed(0)} GiB</small></span>`);
  if (q && !hits.length) items.push(`<span class="opt none">No GPU matches “${esc(q)}”.</span>`);
  items.push(`<span class="opt add${active === hits.length ? ' act' : ''}" role="option" data-i="custom">Enter my own GPU’s specs</span>`);
  gl.innerHTML = items.join('');
  gl.querySelector('.act')?.scrollIntoView({ block: 'nearest' });
}
function openPop() { pop.hidden = false; gs.value = ''; active = Math.max(0, searchGpus('').findIndex((g) => g.id === state.gpu)); drawList(); gs.focus(); }
function closePop() { pop.hidden = true; }
function pick(i) {
  state.gpu = i === 'custom' || i === hits.length ? 'custom' : hits[i].id;
  if (state.gpu !== 'custom') closePop();
  render();
  if (state.gpu === 'custom') { drawList(); $('cMem').focus(); }
}
$('gpuBtn').onclick = () => (pop.hidden ? openPop() : closePop());
gs.addEventListener('input', () => { active = 0; drawList(); });
gs.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowDown') { active = Math.min(active + 1, hits.length); drawList(); e.preventDefault(); }
  else if (e.key === 'ArrowUp') { active = Math.max(active - 1, 0); drawList(); e.preventDefault(); }
  else if (e.key === 'Enter') { e.preventDefault(); pick(active); }
  else if (e.key === 'Escape') { closePop(); $('gpuBtn').focus(); }
});
gl.addEventListener('click', (e) => { const o = e.target.closest('[data-i]'); if (o) pick(o.dataset.i === 'custom' ? 'custom' : +o.dataset.i); });
document.addEventListener('mousedown', (e) => { if (!pop.hidden && !e.target.closest('.gpu-pick')) closePop(); });
const customChange = () => {
  const m = +$('cMem').value, b = +$('cBw').value;
  if (m > 0) state.custom.mem = m;
  if (b > 0) state.custom.bw = b;
  state.custom.arch = $('cArch').value;
  render();
};
['cMem', 'cBw', 'cArch'].forEach((id) => $(id).addEventListener('change', customChange));

// token
const token = $('token');
const syncToken = () => $('tokenToggle').classList.toggle('set', !!localStorage.getItem('hfToken'));
$('tokenToggle').onclick = () => { $('tokenBox').hidden = !$('tokenBox').hidden; if (!$('tokenBox').hidden) token.focus(); };
token.value = localStorage.getItem('hfToken') ?? '';
token.addEventListener('change', () => {
  const v = token.value.trim();
  if (v) localStorage.setItem('hfToken', v); else localStorage.removeItem('hfToken');
  syncToken();
  if (state.repo && !model) run(state.repo, { keepCount: true });
});
syncToken();

const hadCount = readHash();
render();
if (state.repo) run(state.repo, { keepCount: hadCount });
