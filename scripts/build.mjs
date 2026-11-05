// Generates the static SEO pages: /models/<slug>/, /gpus/<id>/, hubs, sitemap.xml, robots.txt.
// Run with:  node scripts/build.mjs   (needs network access to huggingface.co)
import { mkdir, writeFile, readFile, rm } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { GPUS, GiB, evaluate, minGpus, seqBytes, weightBytes, support, fmtGiB, fmtParams, fmtTokens } from '../calc.js';
import { loadModel } from '../hf.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const SITE = 'https://canmygpurunit.majnioui.me';
const TODAY = new Date().toISOString().slice(0, 10);
const PULSE = '<script defer src="https://pulse.majnioui.me/pulse.js" data-website-id="1e1dd69e-6929-4195-a18c-c4d3c46c90a1"></script>';

// repo = canonical Hugging Face repo; data = ungated mirror to read config from (for gated repos).
const MODELS = [
  { repo: 'deepseek-ai/DeepSeek-V4.1-Flash' },
  { repo: 'deepseek-ai/DeepSeek-V4-Pro' },
  { repo: 'deepseek-ai/DeepSeek-V4-Flash' },
  { repo: 'deepseek-ai/DeepSeek-V3.2' },
  { repo: 'deepseek-ai/DeepSeek-V3.1' },
  { repo: 'deepseek-ai/DeepSeek-V3' },
  { repo: 'deepseek-ai/DeepSeek-R1' },
  { repo: 'moonshotai/Kimi-K3' },
  { repo: 'moonshotai/Kimi-K2-Instruct' },
  { repo: 'zai-org/GLM-5.3' },
  { repo: 'zai-org/GLM-5.3-Flash' },
  { repo: 'zai-org/GLM-4.6' },
  { repo: 'zai-org/GLM-4.5-Air' },
  { repo: 'MiniMaxAI/MiniMax-M2.5' },
  { repo: 'MiniMaxAI/MiniMax-M2' },
  { repo: 'Qwen/Qwen3.8-2.4T-A95B' },
  { repo: 'Qwen/Qwen3.8-Flash-Next' },
  { repo: 'Qwen/Qwen3.8-27B' },
  { repo: 'Qwen/Qwen3.6-35B-A3B' },
  { repo: 'Qwen/Qwen3.5-9B' },
  { repo: 'Qwen/Qwen3-235B-A22B-Instruct-2507' },
  { repo: 'Qwen/Qwen3-Coder-480B-A35B-Instruct' },
  { repo: 'Qwen/Qwen3-Coder-30B-A3B-Instruct' },
  { repo: 'Qwen/Qwen3-Next-80B-A3B-Instruct' },
  { repo: 'Qwen/Qwen3-32B' },
  { repo: 'Qwen/Qwen3-14B' },
  { repo: 'Qwen/Qwen3-8B' },
  { repo: 'openai/gpt-oss-120b' },
  { repo: 'openai/gpt-oss-20b' },
  { repo: 'google/gemma-4-31B-it' },
  { repo: 'nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16', name: 'Nemotron 3 Ultra 550B' },
  { repo: 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16', name: 'Nemotron 3 Nano 30B' },
  { repo: 'tencent/Hy3' },
  { repo: 'meta-llama/Llama-3.3-70B-Instruct', data: 'unsloth/Llama-3.3-70B-Instruct' },
  { repo: 'meta-llama/Llama-3.1-8B-Instruct', data: 'unsloth/Llama-3.1-8B-Instruct' },
];

// GPUs that get their own page, and the ones shown in each model's table.
const GPU_PAGES = ['h100', 'h200', 'b200', 'b300', 'l40s', 'a100', 'h100pcie', 'l4', 'mi300x', 'mi355x', 'rtxpro6000', 'rtx5090', 'rtx4090', 'rtx3090', 'dgxspark', 'm3ultra512'];
const TABLE_GPUS = ['h100', 'h200', 'b200', 'b300', 'l40s', 'a100', 'mi300x', 'mi355x', 'rtxpro6000', 'rtx5090', 'rtx4090', 'dgxspark', 'm3ultra512'];
const PRECS = ['native', 'bf16', 'fp8', 'int4', 'fp4'];
const PREC_LABEL = { native: 'As published', bf16: 'BF16', fp8: 'FP8', int8: 'INT8', int4: 'INT4', fp4: 'FP4' };

const gpu = (id) => GPUS.find((g) => g.id === id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const gname = (g) => `${g.name} ${g.sub.replace(/ (SXM|PCIe|unified|NVL72|Blackwell)$/, '')}`;
const slugify = (s) => s.toLowerCase().replace(/[^a-z0-9.]+/g, '-').replace(/\./g, '-').replace(/^-|-$/g, '');
const list = (xs, last = 'and') => (xs.length < 2 ? xs.join('') : `${xs.slice(0, -1).join(', ')} ${last} ${xs.at(-1)}`);
const an = (w) => (/^[AEFHILMNORSX]/.test(w) ? 'an' : 'a');
const appLink = (m, g, n, prec, ctx) => `/#m=${encodeURIComponent(m.repo)}&gpu=${g}&n=${n}&w=${prec}&ctx=${ctx}&seqs=1`;

function shortName(entry, m) {
  if (entry.name) return entry.name;
  return entry.repo.split('/')[1].replace(/-Instruct(-\d+)?$/, '').replace(/-it$/, '').replace(/-BF16$/, '');
}

function base(m, extra = {}) {
  const ctx = Math.min(32768, m.arch.maxPos);
  return { kvBytes: 2, ctx, seqs: 1, util: 0.9, prec: 'native', count: 1, ...extra };
}
const minFor = (m, gid, prec, ctx) => minGpus(m, base(m, { gpu: gpu(gid), prec, ...(ctx ? { ctx } : {}) }));
const fitsOne = (m, gid, prec) => { const r = evaluate(m, base(m, { gpu: gpu(gid), prec, count: 1 })); return r.status === 'yes' || r.status === 'tight'; };

// ------------------------------------------------------------------ page shell

function page({ path, title, description, body, jsonld = [] }) {
  const url = SITE + path;
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(title)}</title>
<meta name="description" content="${esc(description)}">
<link rel="canonical" href="${url}">
<meta property="og:type" content="website">
<meta property="og:site_name" content="Can my GPU run it?">
<meta property="og:title" content="${esc(title)}">
<meta property="og:description" content="${esc(description)}">
<meta property="og:url" content="${url}">
<meta property="og:image" content="${SITE}/og.png">
<meta name="twitter:card" content="summary_large_image">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible+Next:wght@400;500;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="/style.css">
${PULSE}
${jsonld.map((j) => `<script type="application/ld+json">${JSON.stringify(j)}</script>`).join('\n')}
</head>
<body>
<div class="page doc">
${body}
  <footer class="foot">
    <p>Estimates assume vLLM or SGLang serving with tensor parallelism, 90% of GPU memory in use, and a BF16 KV cache. Results near the limit are borderline. Data read from Hugging Face on ${TODAY}.</p>
    <p class="credit"><a href="/">Can my GPU run it?</a> · <a href="/models/">Models</a> · <a href="/gpus/">GPUs</a> · Made by <a href="https://majnioui.me" target="_blank" rel="noopener">Moe Majnioui</a></p>
  </footer>
</div>
</body>
</html>
`;
}

const crumbs = (items) => ({
  html: `<nav class="crumbs" aria-label="Breadcrumb">${items.map(([n, p], i) => (i < items.length - 1 ? `<a href="${p}">${esc(n)}</a> / ` : `<span>${esc(n)}</span>`)).join('')}</nav>`,
  ld: {
    '@context': 'https://schema.org', '@type': 'BreadcrumbList',
    itemListElement: items.map(([n, p], i) => ({ '@type': 'ListItem', position: i + 1, name: n, item: SITE + p })),
  },
});
const faqLd = (qas) => ({
  '@context': 'https://schema.org', '@type': 'FAQPage',
  mainEntity: qas.map(([q, a]) => ({ '@type': 'Question', name: q, acceptedAnswer: { '@type': 'Answer', text: a.replace(/<[^>]+>/g, '') } })),
});

// ------------------------------------------------------------------ model pages

function modelPage(entry, m, all) {
  const a = m.arch;
  const name = entry.short;
  const ctx = Math.min(32768, a.maxPos);
  const pub = m.nativePrec.toUpperCase();
  const w = Object.fromEntries(['native', 'bf16', 'fp8', 'int4', 'fp4'].map((p) => [p, weightBytes(m, p)]));
  const kvTok = seqBytes(a, 1, 2, 65536) / 65536;
  const activeTxt = a.isMoE && !a.activeUncertain ? `, about ${fmtParams(m.active)} active per token` : '';

  // headline setups at the published precision
  const headline = ['h100', 'h200', 'b300', 'l40s'].map((id) => {
    const r = minFor(m, id, 'native');
    return r ? `${r.n} × ${gname(gpu(id))}` : `more than 16 × ${gname(gpu(id))}`;
  });
  const singles = TABLE_GPUS.filter((id) => fitsOne(m, id, 'native'));
  const cheapest = (() => {
    for (const p of ['int4', 'fp4', 'fp8']) {
      const one = TABLE_GPUS.filter((id) => fitsOne(m, id, p));
      if (one.length) return { p, one };
    }
    return null;
  })();

  let answer = `${esc(name)} has ${fmtParams(m.params)} parameters${activeTxt}. As published in ${pub} the weights take ${fmtGiB(w.native)}. With ${fmtTokens(ctx)} tokens of context it needs ${list(headline.map(esc))}.`;
  if (singles.length) answer += ` It fits on a single ${list(singles.slice(0, 4).map((id) => esc(gname(gpu(id)))), 'or')} without quantizing.`;
  else if (cheapest) answer += ` Quantized to ${PREC_LABEL[cheapest.p]}, it fits on a single ${list(cheapest.one.slice(0, 3).map((id) => esc(gname(gpu(id)))), 'or')}.`;
  else answer += ` It doesn't fit on any single GPU, even in INT4.`;

  // table
  const rows = TABLE_GPUS.map((id) => {
    const g = gpu(id);
    const cells = PRECS.map((p) => {
      const r = minFor(m, id, p);
      if (!r) return '<td class="none">—</td>';
      const slow = support(g.arch, p === 'native' ? m.nativePrec : p) === 'slow';
      return `<td class="${r.r.status}"><a href="${appLink(m, id, r.n, p, ctx)}">${r.n}${slow ? '<sup>*</sup>' : ''}</a></td>`;
    }).join('');
    return `<tr><th scope="row"><a href="/gpus/${id}/">${esc(g.name)}</a><small>${esc(g.sub)}</small></th>${cells}</tr>`;
  }).join('');

  // context on reference setups
  const refs = [];
  for (const id of ['h200', 'h100', 'b300', 'rtx4090']) {
    const r = minFor(m, id, 'native') ?? minFor(m, id, 'int4');
    if (r && !refs.some((x) => x.id === id)) refs.push({ id, n: r.n, prec: r.r.prec });
  }
  const bigCtx = [ctx, 131072, a.maxPos].filter((c, i, arr) => c <= a.maxPos && arr.indexOf(c) === i);
  const ctxRows = refs.map(({ id, n, prec }) => {
    const r = evaluate(m, base(m, { gpu: gpu(id), count: n, prec }));
    const conc = bigCtx.map((c) => { const x = evaluate(m, base(m, { gpu: gpu(id), count: n, prec, ctx: c })); return `<td>${x.maxSeqs.toLocaleString()}</td>`; }).join('');
    return `<tr><th scope="row">${n} × ${esc(gname(gpu(id)))}<small>${PREC_LABEL[prec] === 'As published' ? pub : PREC_LABEL[prec]}</small></th><td>${fmtTokens(r.maxCtx)}</td>${conc}</tr>`;
  }).join('');

  // FAQ
  const h100 = gpu('h100');
  const oneH100 = PRECS.filter((p) => p !== 'native' && fitsOne(m, 'h100', p));
  const h100min = minFor(m, 'h100', 'native');
  const r4090 = ['native', 'fp8', 'int4'].find((p) => fitsOne(m, 'rtx4090', p));
  const mac = ['native', 'fp8', 'int4'].find((p) => fitsOne(m, 'm3ultra512', p));
  const qas = [
    [`How much VRAM does ${name} need?`,
      `The weights take ${fmtGiB(w.native)} as published (${pub}), ${fmtGiB(w.bf16)} in BF16, ${fmtGiB(w.fp8)} in FP8 and ${fmtGiB(w.int4)} in INT4. On top of that, the KV cache adds about ${(kvTok / 1024).toFixed(1)} KiB per token: ${fmtGiB(seqBytes(a, 1, 2, ctx))} for one ${fmtTokens(ctx)}-token request${a.maxPos >= 131072 ? ` and ${fmtGiB(seqBytes(a, 1, 2, 131072))} at 128K` : ''}. Serving engines also need a few GiB per GPU for activations.`],
    [`Can ${name} run on a single H100?`,
      fitsOne(m, 'h100', 'native')
        ? `Yes. As published in ${pub} it fits on one H100 80 GB with ${fmtTokens(ctx)} tokens of context.`
        : oneH100.length
          ? `Not as published, but it does in ${list(oneH100.map((p) => PREC_LABEL[p]), 'or')}. As published in ${pub} it needs ${h100min ? `${h100min.n} H100s` : 'more than 16 H100s'}.`
          : `No. Even in INT4 the weights (${fmtGiB(w.int4)}) don't fit in the ~${(h100.mem * 0.9 - 2).toFixed(0)} GiB an H100 can use. As published it needs ${h100min ? `${h100min.n} H100s` : 'more than 16 H100s'}.`],
    [`Can I run ${name} on an RTX 4090 or a Mac?`,
      `${r4090 ? `On one RTX 4090 (24 GB) it fits ${r4090 === 'native' ? 'as published' : `in ${PREC_LABEL[r4090]}`} with ${fmtTokens(ctx)} of context.` : `It doesn't fit on one RTX 4090 (24 GB), even in INT4.`} ${mac ? `A Mac Studio with M3 Ultra and 512 GB runs it ${mac === 'native' ? 'as published' : `in ${PREC_LABEL[mac]}`}, for example a Q4 GGUF with llama.cpp or MLX${mac === 'native' ? '' : ''}.` : `It's too large even for a 512 GB Mac Studio in INT4.`}`],
  ];
  if (refs[0]) {
    const ref = refs[0];
    const r = evaluate(m, base(m, { gpu: gpu(ref.id), count: ref.n, prec: ref.prec }));
    qas.push([`How much context can ${name} handle on ${ref.n} × ${gname(gpu(ref.id))}?`,
      `One request can use up to ${fmtTokens(r.maxCtx)} tokens (the model's limit is ${fmtTokens(a.maxPos)}). At ${fmtTokens(ctx)} tokens per request, about ${r.maxSeqs.toLocaleString()} requests fit in memory at once.`]);
  }

  // facts + notes
  const layerTxt = a.compress
    ? `${a.L} (DeepSeek compressed attention)`
    : [a.layers.full && `${a.layers.full} full attention`, a.layers.sliding && `${a.layers.sliding} sliding window (${fmtTokens(a.window)})`, a.layers.linear && `${a.layers.linear} linear or Mamba`].filter(Boolean).join(', ');
  const facts = [
    ['Hugging Face', `<a href="https://huggingface.co/${esc(entry.repo)}" rel="noopener">${esc(entry.repo)}</a>`],
    ['Parameters', `${fmtParams(m.params)}${activeTxt}`],
    a.isMoE && ['Experts', `${a.experts}, ${a.topk} used per token`],
    ['Published weights', `${fmtGiB(w.native)} in ${pub}`],
    ['Attention', esc(a.attnKind)],
    ['Layers', esc(layerTxt || String(a.L))],
    ['KV cache per token', `${(kvTok / 1024).toFixed(1)} KiB in BF16`],
    ['Longest context', `${fmtTokens(a.maxPos)} tokens`],
  ].filter(Boolean);
  const notes = [...m.warnings.map(esc)];
  if (a.mla) notes.push('It uses multi-head latent attention, so its KV cache is far smaller than a standard model of the same size. With tensor parallelism each GPU keeps a full copy of it; data-parallel attention fits more context.');
  if (a.compress) notes.push('Every layer keeps the last 128 tokens; most also keep a compressed KV cache of the whole context, which is why long contexts stay cheap.');
  if (a.layers.linear) notes.push(`${a.layers.linear} of ${a.L} layers use linear attention or Mamba, which keep a fixed amount of memory per request instead of a cache that grows with context.`);
  if (a.layers.sliding && !a.compress) notes.push(`${a.layers.sliding} of ${a.L} layers only look at the last ${fmtTokens(a.window)} tokens, so long contexts cost less memory.`);
  if (m.nativePrec === 'fp4') notes.push('The published weights are FP4, which is native only on NVIDIA Blackwell and AMD MI350/355. Other GPUs run them through slower weight-only kernels.');
  if (m.nativePrec === 'fp8') notes.push('The published weights are FP8, native on Ada, Hopper, Blackwell and MI300 or newer. A100s run FP8 as weight-only.');

  const related = all.filter((x) => x !== entry && x.m)
    .map((x) => ({ x, d: Math.abs(Math.log(x.m.params / m.params)) + (x.repo.split('/')[0] === entry.repo.split('/')[0] ? -0.3 : 0) }))
    .sort((p, q) => p.d - q.d).slice(0, 6).map(({ x }) => `<li><a href="/models/${x.slug}/">${esc(x.short)}</a> <span class="muted">${fmtParams(x.m.params)}</span></li>`).join('');

  const path = `/models/${entry.slug}/`;
  const bc = crumbs([['Home', '/'], ['Models', '/models/'], [name, path]]);
  const best = minFor(m, 'h200', 'native');
  const description = `${name} (${fmtParams(m.params)}, ${fmtGiB(w.native)} in ${pub}) needs ${headline.slice(0, 3).join(', ')} at ${fmtTokens(ctx)} context. See GPUs needed per precision, VRAM, and max context.`;
  const body = `
  ${bc.html}
  <h1>Can my GPU run ${esc(name)}?</h1>
  <p class="lede big">${answer}</p>
  <p><a class="cta" href="${appLink(m, 'h200', best?.n ?? 8, 'native', ctx)}">Try your own GPUs and context length</a></p>

  <section class="block">
    <h2>GPUs needed for ${esc(name)}</h2>
    <p class="sub">The fewest GPUs of each type that run ${esc(name)} with ${fmtTokens(ctx)} tokens of context for one request. Select a number to open that setup.</p>
    <div class="table-wrap"><table>
      <thead><tr><th>GPU</th>${PRECS.map((p) => `<th>${p === 'native' ? `As published<br>${pub}` : PREC_LABEL[p]}</th>`).join('')}</tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
    <p class="legend">— means more than 16 GPUs. * runs, but not natively on that GPU, so it’s slower.</p>
  </section>

  ${ctxRows ? `<section class="block">
    <h2>How much context fits</h2>
    <p class="sub">The longest single request, and how many requests fit in memory at once at each context length.</p>
    <div class="table-wrap"><table>
      <thead><tr><th>Setup</th><th>Longest request</th>${bigCtx.map((c) => `<th>Requests at ${fmtTokens(c)}</th>`).join('')}</tr></thead>
      <tbody>${ctxRows}</tbody>
    </table></div>
  </section>` : ''}

  <section class="block">
    <h2>Memory needed</h2>
    <div class="table-wrap"><table class="narrow">
      <thead><tr><th>Part</th><th>Size</th></tr></thead>
      <tbody>
        ${['native', 'bf16', 'fp8', 'int4', 'fp4'].map((p) => `<tr><th scope="row">Weights, ${p === 'native' ? `as published (${pub})` : PREC_LABEL[p]}</th><td>${fmtGiB(w[p])}</td></tr>`).join('')}
        ${bigCtx.map((c) => `<tr><th scope="row">KV cache, one ${fmtTokens(c)} request</th><td>${fmtGiB(seqBytes(a, 1, 2, c))}</td></tr>`).join('')}
      </tbody>
    </table></div>
  </section>

  <section class="block">
    <h2>About ${esc(name)}</h2>
    <dl class="facts">${facts.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('')}</dl>
    ${notes.length ? `<ul class="notes spaced">${notes.map((n) => `<li>${n}</li>`).join('')}</ul>` : ''}
  </section>

  <section class="block">
    <h2>Questions</h2>
    ${qas.map(([q, ans]) => `<h3>${esc(q)}</h3><p class="qa">${esc(ans)}</p>`).join('\n    ')}
  </section>

  <section class="block">
    <h2>Similar models</h2>
    <ul class="links">${related}</ul>
  </section>`;
  return { path, html: page({ path, title: `Can my GPU run ${name}? GPUs and VRAM needed`, description, body, jsonld: [bc.ld, faqLd(qas)] }) };
}

// ------------------------------------------------------------------ GPU pages

function gpuPage(id, models) {
  const g = gpu(id);
  const label = gname(g);
  const native = ['FP8', 'FP4'].filter((p) => support(g.arch, p.toLowerCase()) === 'ok');
  const rows = models.map((e) => {
    const m = e.m;
    const ctx = Math.min(32768, m.arch.maxPos);
    const cells = ['native', 'fp8', 'int4'].map((p) => {
      const r = minFor(m, id, p);
      return r ? `<td class="${r.r.status}"><a href="${appLink(m, id, r.n, p, ctx)}">${r.n}</a></td>` : '<td class="none">—</td>';
    }).join('');
    const one = evaluate(m, base(m, { gpu: g, count: 1, prec: 'native' }));
    const oneTxt = one.status === 'no' ? '<td class="none">—</td>' : `<td>${fmtTokens(one.maxCtx)}</td>`;
    return { e, html: `<tr><th scope="row"><a href="/models/${e.slug}/">${esc(e.short)}</a><small>${fmtParams(m.params)}</small></th>${cells}${oneTxt}</tr>` };
  });
  const single = models.filter((e) => fitsOne(e.m, id, 'native'));
  const singleQ = models.filter((e) => !fitsOne(e.m, id, 'native') && fitsOne(e.m, id, 'int4'));
  const path = `/gpus/${id}/`;
  const bc = crumbs([['Home', '/'], ['GPUs', '/gpus/'], [label, path]]);
  const intro = `The ${esc(g.name)} has ${g.sub.replace(/ (SXM|PCIe|NVL72)$/, '')}${g.cat === 'unified' ? '' : ' of memory'} and ${g.bw} TB/s of bandwidth. ${native.length ? `It runs ${list(native)} natively.` : 'It has no native FP8 or FP4, so those formats run through slower weight-only kernels.'} ` +
    (single.length ? `On its own it runs ${list(single.slice(0, 6).map((e) => esc(e.short)))}${single.length > 6 ? ` and ${single.length - 6} more` : ''} as published. ` : `None of the models below fits on a single ${esc(g.name)} as published. `) +
    (singleQ.length ? `Quantized to INT4, one also runs ${list(singleQ.slice(0, 4).map((e) => esc(e.short)))}.` : '');
  const description = `Which open models fit on the ${label}: ${single.length} of ${models.length} popular models run on one as published. GPUs needed per model, precision and max context.`;
  const body = `
  ${bc.html}
  <h1>What can ${an(label)} ${esc(label)} run?</h1>
  <p class="lede big">${intro}</p>
  <p><a class="cta" href="/#gpu=${id}&n=1&w=native&ctx=32768&seqs=1">Check another model on the ${esc(g.name)}</a></p>
  <section class="block">
    <h2>How many ${esc(g.name)}s each model needs</h2>
    <p class="sub">At 32K tokens of context for one request (or the model’s limit if shorter). The last column is the longest request one ${esc(g.name)} can serve as published.</p>
    <div class="table-wrap"><table>
      <thead><tr><th>Model</th><th>As published</th><th>FP8</th><th>INT4</th><th>Context on one GPU</th></tr></thead>
      <tbody>${rows.map((r) => r.html).join('')}</tbody>
    </table></div>
    <p class="legend">— means more than 16 GPUs, or that it doesn’t fit on one.</p>
  </section>
  <section class="block">
    <h2>Other GPUs</h2>
    <ul class="links">${GPU_PAGES.filter((x) => x !== id).map((x) => `<li><a href="/gpus/${x}/">${esc(gname(gpu(x)))}</a></li>`).join('')}</ul>
  </section>`;
  return { path, html: page({ path, title: `What models can ${an(label)} ${label} run? Open LLMs that fit`, description, body, jsonld: [bc.ld] }) };
}

// ------------------------------------------------------------------ hubs, sitemap

function hubs(models) {
  const mBc = crumbs([['Home', '/'], ['Models', '/models/']]);
  const mBody = `
  ${mBc.html}
  <h1>GPU requirements for open models</h1>
  <p class="lede big">How many GPUs, how much VRAM and how much context each popular open model needs. Pick a model to see every GPU and precision.</p>
  <div class="table-wrap"><table>
    <thead><tr><th>Model</th><th>Parameters</th><th>Published weights</th><th>H100s</th><th>H200s</th><th>B300s</th></tr></thead>
    <tbody>${models.map((e) => {
      const c = (id) => { const r = minFor(e.m, id, 'native'); return r ? `<td class="${r.r.status}">${r.n}</td>` : '<td class="none">—</td>'; };
      return `<tr><th scope="row"><a href="/models/${e.slug}/">${esc(e.short)}</a></th><td>${fmtParams(e.m.params)}</td><td>${fmtGiB(weightBytes(e.m, 'native'))} ${e.m.nativePrec.toUpperCase()}</td>${c('h100')}${c('h200')}${c('b300')}</tr>`;
    }).join('')}</tbody>
  </table></div>
  <p class="legend">GPUs needed at 32K context, as published. — means more than 16.</p>`;
  const gBc = crumbs([['Home', '/'], ['GPUs', '/gpus/']]);
  const gBody = `
  ${gBc.html}
  <h1>Which models can your GPU run?</h1>
  <p class="lede big">Pick a GPU to see which open models fit on it, how many you need, and how much context you get.</p>
  <div class="table-wrap"><table>
    <thead><tr><th>GPU</th><th>Memory</th><th>Bandwidth</th><th>Models that fit on one</th></tr></thead>
    <tbody>${GPU_PAGES.map((id) => {
      const g = gpu(id);
      const n = models.filter((e) => fitsOne(e.m, id, 'native')).length;
      return `<tr><th scope="row"><a href="/gpus/${id}/">${esc(g.name)}</a><small>${esc(g.sub)}</small></th><td>${g.mem.toFixed(0)} GiB</td><td>${g.bw} TB/s</td><td>${n} of ${models.length}</td></tr>`;
    }).join('')}</tbody>
  </table></div>`;
  return [
    { path: '/models/', html: page({ path: '/models/', title: 'GPU and VRAM requirements for open LLMs', description: `GPUs, VRAM and context for ${models.length} popular open models, from DeepSeek and Kimi to Qwen, GLM and gpt-oss.`, body: mBody, jsonld: [mBc.ld] }) },
    { path: '/gpus/', html: page({ path: '/gpus/', title: 'Which open models can your GPU run?', description: 'H100, H200, B200, B300, L40S, MI300X, RTX 4090 and more: which open models fit on each GPU and how many you need.', body: gBody, jsonld: [gBc.ld] }) },
  ];
}

async function updateIndex(models) {
  const p = join(ROOT, 'index.html');
  let html = await readFile(p, 'utf8');
  const block = `<!--seo:start-->
  <section class="block browse">
    <h2>Popular models</h2>
    <ul class="links cols">${models.map((e) => `<li><a href="/models/${e.slug}/">${esc(e.short)}</a></li>`).join('')}</ul>
    <h2>GPUs</h2>
    <ul class="links cols">${GPU_PAGES.map((id) => `<li><a href="/gpus/${id}/">${esc(gname(gpu(id)))}</a></li>`).join('')}</ul>
  </section>
  <!--seo:end-->`;
  html = html.replace(/<!--seo:start-->[\s\S]*<!--seo:end-->/, block);
  await writeFile(p, html);
}

// ------------------------------------------------------------------ main

const ok = [];
for (const e of MODELS) {
  try {
    e.m = await loadModel(e.data ?? e.repo);
    e.m.repo = e.repo;
    e.short = shortName(e, e.m);
    e.slug = slugify(e.short);
    ok.push(e);
    console.log('ok  ', e.repo);
  } catch (err) {
    console.warn('skip', e.repo, err.message);
  }
}
if (new Set(ok.map((e) => e.slug)).size !== ok.length) throw new Error('duplicate slugs');

await rm(join(ROOT, 'models'), { recursive: true, force: true });
await rm(join(ROOT, 'gpus'), { recursive: true, force: true });
const pages = [
  ...ok.map((e) => modelPage(e, e.m, ok)),
  ...GPU_PAGES.map((id) => gpuPage(id, ok)),
  ...hubs(ok),
];
for (const { path, html } of pages) {
  const file = join(ROOT, path, 'index.html');
  await mkdir(dirname(file), { recursive: true });
  await writeFile(file, html);
}
await updateIndex(ok);

const urls = ['/', '/models/', '/gpus/', ...pages.filter((p) => !['/models/', '/gpus/'].includes(p.path)).map((p) => p.path)];
await writeFile(join(ROOT, 'sitemap.xml'), `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
${urls.map((u) => `  <url><loc>${SITE}${u}</loc><lastmod>${TODAY}</lastmod></url>`).join('\n')}
</urlset>
`);
await writeFile(join(ROOT, 'robots.txt'), `User-agent: *\nAllow: /\n\nSitemap: ${SITE}/sitemap.xml\n`);
console.log(`\n${pages.length} pages, ${urls.length} sitemap URLs`);
