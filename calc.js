// canirun — pure sizing logic (no DOM). Units: bytes internally, GiB for display.

export const GiB = 1024 ** 3;

// mem = usable memory (GiB, as nvidia-smi / rocm-smi report it; unified-memory devices: what the GPU may wire by default).
// bw = memory bandwidth (TB/s). cat: dc | ws | consumer | unified. link: fast GPU-GPU interconnect (NVLink / Infinity Fabric).
const G = (id, name, sub, mem, bw, arch, cat, vendor, extra = {}) => ({ id, name, sub, mem, bw, arch, cat, vendor, link: cat === 'dc', ...extra });
export const GPUS = [
  // NVIDIA datacenter
  G('h100', 'H100', '80 GB SXM', 81559 / 1024, 3.35, 'hopper', 'dc', 'nvidia', { featured: true, aliases: 'hgx dgx' }),
  G('h200', 'H200', '141 GB SXM', 143771 / 1024, 4.8, 'hopper', 'dc', 'nvidia', { featured: true, aliases: 'hgx dgx' }),
  G('l40s', 'L40S', '48 GB PCIe', 46068 / 1024, 0.864, 'ada', 'dc', 'nvidia', { featured: true, link: false }),
  G('b300', 'B300', '288 GB', 288e9 / GiB, 8.0, 'blackwell', 'dc', 'nvidia', { featured: true, aliases: 'blackwell ultra hgx dgx' }),
  G('gb300', 'GB300', '288 GB NVL72', 288e9 / GiB, 8.0, 'blackwell', 'dc', 'nvidia', { aliases: 'grace blackwell ultra nvl72' }),
  G('b200', 'B200', '180 GB', 183359 / 1024, 8.0, 'blackwell', 'dc', 'nvidia', { aliases: 'hgx dgx' }),
  G('gb200', 'GB200', '186 GB NVL72', 186, 8.0, 'blackwell', 'dc', 'nvidia', { aliases: 'grace blackwell nvl72' }),
  G('h200nvl', 'H200 NVL', '141 GB PCIe', 143771 / 1024, 4.8, 'hopper', 'dc', 'nvidia'),
  G('h100pcie', 'H100 PCIe', '80 GB', 81559 / 1024, 2.0, 'hopper', 'dc', 'nvidia', { link: false }),
  G('h100nvl', 'H100 NVL', '94 GB', 95830 / 1024, 3.9, 'hopper', 'dc', 'nvidia'),
  G('gh200-96', 'GH200', '96 GB', 97871 / 1024, 4.0, 'hopper', 'dc', 'nvidia', { aliases: 'grace hopper' }),
  G('gh200-144', 'GH200', '144 GB', 144, 4.9, 'hopper', 'dc', 'nvidia', { aliases: 'grace hopper' }),
  G('h800', 'H800', '80 GB SXM', 81559 / 1024, 3.35, 'hopper', 'dc', 'nvidia'),
  G('h20', 'H20', '96 GB', 97871 / 1024, 4.0, 'hopper', 'dc', 'nvidia'),
  G('a100', 'A100', '80 GB SXM', 80, 2.04, 'ampere', 'dc', 'nvidia'),
  G('a100-40', 'A100', '40 GB', 40, 1.56, 'ampere', 'dc', 'nvidia'),
  G('a800', 'A800', '80 GB', 80, 2.04, 'ampere', 'dc', 'nvidia'),
  G('l40', 'L40', '48 GB', 46068 / 1024, 0.864, 'ada', 'dc', 'nvidia', { link: false }),
  G('a40', 'A40', '48 GB', 46068 / 1024, 0.696, 'ampere', 'dc', 'nvidia', { link: false }),
  G('l4', 'L4', '24 GB', 23034 / 1024, 0.3, 'ada', 'dc', 'nvidia', { link: false }),
  G('a10', 'A10', '24 GB', 23028 / 1024, 0.6, 'ampere', 'dc', 'nvidia', { link: false, aliases: 'a10g aws g5' }),
  G('t4', 'T4', '16 GB', 15360 / 1024, 0.32, 'turing', 'dc', 'nvidia', { link: false, aliases: 'colab aws g4dn' }),
  G('v100', 'V100', '32 GB', 32768 / 1024, 0.9, 'turing', 'dc', 'nvidia', { aliases: 'volta' }),
  // AMD / Intel datacenter
  G('mi300x', 'MI300X', '192 GB', 192, 5.3, 'cdna3', 'dc', 'amd', { aliases: 'instinct' }),
  G('mi325x', 'MI325X', '256 GB', 256, 6.0, 'cdna3', 'dc', 'amd', { aliases: 'instinct' }),
  G('mi355x', 'MI355X', '288 GB', 288, 8.0, 'cdna4', 'dc', 'amd', { aliases: 'instinct' }),
  G('mi350x', 'MI350X', '288 GB', 288, 8.0, 'cdna4', 'dc', 'amd', { aliases: 'instinct' }),
  G('mi250x', 'MI250X', '128 GB', 128, 3.2, 'cdna2', 'dc', 'amd', { aliases: 'instinct' }),
  G('mi210', 'MI210', '64 GB', 64, 1.6, 'cdna2', 'dc', 'amd', { aliases: 'instinct' }),
  G('gaudi3', 'Gaudi 3', '128 GB', 128, 3.7, 'gaudi', 'dc', 'intel', { aliases: 'habana' }),
  G('gaudi2', 'Gaudi 2', '96 GB', 96, 2.45, 'gaudi', 'dc', 'intel', { aliases: 'habana' }),
  // Workstation
  G('rtxpro6000', 'RTX PRO 6000', '96 GB Blackwell', 97887 / 1024, 1.79, 'blackwell', 'ws', 'nvidia', { aliases: 'workstation max-q server' }),
  G('rtx6000ada', 'RTX 6000 Ada', '48 GB', 49140 / 1024, 0.96, 'ada', 'ws', 'nvidia'),
  G('rtxa6000', 'RTX A6000', '48 GB', 49140 / 1024, 0.768, 'ampere', 'ws', 'nvidia'),
  G('rtx5000ada', 'RTX 5000 Ada', '32 GB', 32760 / 1024, 0.576, 'ada', 'ws', 'nvidia'),
  G('rtx4000ada', 'RTX 4000 Ada', '20 GB', 20475 / 1024, 0.36, 'ada', 'ws', 'nvidia'),
  // Consumer
  G('rtx5090', 'RTX 5090', '32 GB', 32607 / 1024, 1.79, 'blackwell', 'consumer', 'nvidia', { aliases: 'geforce' }),
  G('rtx5080', 'RTX 5080', '16 GB', 16303 / 1024, 0.96, 'blackwell', 'consumer', 'nvidia', { aliases: 'geforce' }),
  G('rtx5070ti', 'RTX 5070 Ti', '16 GB', 16303 / 1024, 0.896, 'blackwell', 'consumer', 'nvidia', { aliases: 'geforce' }),
  G('rtx5070', 'RTX 5070', '12 GB', 12227 / 1024, 0.672, 'blackwell', 'consumer', 'nvidia', { aliases: 'geforce' }),
  G('rtx5060ti', 'RTX 5060 Ti', '16 GB', 16303 / 1024, 0.448, 'blackwell', 'consumer', 'nvidia', { aliases: 'geforce' }),
  G('rtx4090', 'RTX 4090', '24 GB', 24564 / 1024, 1.01, 'ada', 'consumer', 'nvidia', { aliases: 'geforce' }),
  G('rtx4080', 'RTX 4080 / Super', '16 GB', 16376 / 1024, 0.736, 'ada', 'consumer', 'nvidia', { aliases: 'geforce' }),
  G('rtx4070tis', 'RTX 4070 Ti Super', '16 GB', 16376 / 1024, 0.672, 'ada', 'consumer', 'nvidia', { aliases: 'geforce' }),
  G('rtx4070', 'RTX 4070', '12 GB', 12282 / 1024, 0.504, 'ada', 'consumer', 'nvidia', { aliases: 'geforce' }),
  G('rtx4060ti', 'RTX 4060 Ti', '16 GB', 16380 / 1024, 0.288, 'ada', 'consumer', 'nvidia', { aliases: 'geforce' }),
  G('rtx3090', 'RTX 3090 / Ti', '24 GB', 24576 / 1024, 0.936, 'ampere', 'consumer', 'nvidia', { aliases: 'geforce' }),
  G('rtx3080', 'RTX 3080', '10 GB', 10240 / 1024, 0.76, 'ampere', 'consumer', 'nvidia', { aliases: 'geforce' }),
  G('rtx3060', 'RTX 3060', '12 GB', 12288 / 1024, 0.36, 'ampere', 'consumer', 'nvidia', { aliases: 'geforce' }),
  G('rx7900xtx', 'RX 7900 XTX', '24 GB', 24, 0.96, 'rdna3', 'consumer', 'amd', { aliases: 'radeon' }),
  G('rx9070xt', 'RX 9070 XT', '16 GB', 16, 0.64, 'rdna4', 'consumer', 'amd', { aliases: 'radeon' }),
  // Unified memory (GPU can address most of system RAM)
  G('dgxspark', 'DGX Spark', '128 GB unified', 119, 0.273, 'blackwell', 'unified', 'nvidia', { aliases: 'gb10 ascent asus' }),
  G('aimax395', 'Ryzen AI Max+ 395', '128 GB unified', 96, 0.256, 'rdna3', 'unified', 'amd', { aliases: 'strix halo framework desktop' }),
  G('m3ultra512', 'Mac M3 Ultra', '512 GB unified', 384, 0.819, 'apple', 'unified', 'apple', { aliases: 'mac studio apple silicon' }),
  G('m3ultra256', 'Mac M3 Ultra', '256 GB unified', 192, 0.819, 'apple', 'unified', 'apple', { aliases: 'mac studio apple silicon' }),
  G('m4max128', 'Mac M4 Max', '128 GB unified', 96, 0.546, 'apple', 'unified', 'apple', { aliases: 'macbook pro mac studio apple silicon' }),
  G('m4max64', 'Mac M4 Max', '64 GB unified', 48, 0.546, 'apple', 'unified', 'apple', { aliases: 'macbook pro apple silicon' }),
  G('m4pro48', 'Mac M4 Pro', '48 GB unified', 36, 0.273, 'apple', 'unified', 'apple', { aliases: 'mac mini macbook apple silicon' }),
  G('m2ultra192', 'Mac M2 Ultra', '192 GB unified', 144, 0.8, 'apple', 'unified', 'apple', { aliases: 'mac studio apple silicon' }),
];

export const ARCHES = [
  ['blackwell', 'NVIDIA Blackwell (B200/B300, RTX 50)'], ['hopper', 'NVIDIA Hopper (H100/H200)'], ['ada', 'NVIDIA Ada (L40S, RTX 40)'],
  ['ampere', 'NVIDIA Ampere (A100, RTX 30)'], ['turing', 'NVIDIA Turing / Volta (T4, V100)'], ['cdna4', 'AMD CDNA4 (MI350/355)'],
  ['cdna3', 'AMD CDNA3 (MI300)'], ['cdna2', 'AMD CDNA2 (MI200)'], ['rdna4', 'AMD RDNA4 (RX 9000)'], ['rdna3', 'AMD RDNA3 (RX 7000)'],
  ['gaudi', 'Intel Gaudi'], ['apple', 'Apple Silicon'],
];

export function customGpu({ mem, bw, arch, count }) {
  return G('custom', 'Custom GPU', `${mem} GB`, mem * 1e9 / GiB * 0.97, bw, arch, 'dc', 'custom', { link: count !== false });
}

/** Fuzzy GPU search: every query token must prefix-match something in name/sub/aliases/vendor. */
export function searchGpus(q, list = GPUS) {
  const toks = q.toLowerCase().replace(/(\d)\s*gb/g, '$1').split(/[\s,/]+/).filter(Boolean);
  if (!toks.length) return list;
  return list
    .map((g) => {
      const hay = `${g.name} ${g.sub} ${g.aliases ?? ''} ${g.vendor} ${g.arch}`.toLowerCase();
      const compact = hay.replace(/[\s-]/g, '');
      const words = hay.split(/[\s/()-]+/);
      let score = 0;
      for (const t of toks) {
        if (words.some((w) => w === t)) score += 3;
        else if (words.some((w) => w.startsWith(t))) score += 2;
        else if (compact.includes(t)) score += 1;
        else return null;
      }
      return { g, score: score + (g.cat === 'dc' ? 0.5 : 0) };
    })
    .filter(Boolean)
    .sort((a, b) => b.score - a.score)
    .map((x) => x.g);
}

// bpp = bytes per parameter for quantized linear layers (incl. scales). Embeddings/lm_head stay BF16.
export const PRECISIONS = [
  { id: 'native', label: 'Checkpoint', short: 'native' },
  { id: 'bf16', label: 'BF16', bpp: 2 },
  { id: 'fp8',  label: 'FP8',  bpp: 1.02 },
  { id: 'int8', label: 'INT8', bpp: 1.02 },
  { id: 'int4', label: 'INT4', bpp: 0.53, hint: 'AWQ / GPTQ' },
  { id: 'fp4',  label: 'FP4',  bpp: 0.5625, hint: 'NVFP4 / MXFP4' },
];

// 'ok' = native kernels, 'slow' = runs via weight-only / dequant kernels.
const FP8_NATIVE = ['ada', 'hopper', 'blackwell', 'cdna3', 'cdna4', 'rdna4', 'gaudi'];
const FP4_NATIVE = ['blackwell', 'cdna4'];
export function support(arch, prec) {
  if (prec === 'fp8') return FP8_NATIVE.includes(arch) ? 'ok' : 'slow';
  if (prec === 'fp4') return FP4_NATIVE.includes(arch) ? 'ok' : 'slow';
  return 'ok';
}

export function supportNote(arch, prec) {
  if (support(arch, prec) === 'ok') return null;
  if (prec === 'fp8') return 'No FP8 tensor cores on this architecture. FP8 weights run as weight-only (e.g. Marlin W8A16), so throughput is lower.';
  if (prec === 'fp4') return `FP4 is native only on NVIDIA Blackwell and AMD MI350/355. Here it runs via weight-only dequant kernels${arch === 'apple' ? ' (MLX / llama.cpp)' : ''}.`;
  return null;
}

// ---------------------------------------------------------------- config parsing

const first = (...v) => v.find((x) => x !== undefined && x !== null);

/** Flatten multimodal / wrapped configs and normalise Mistral params.json. */
export function normalizeConfig(raw) {
  if (!raw) return null;
  let c = { ...raw };
  // Mistral native params.json
  if (c.dim && c.n_layers && !c.hidden_size) {
    c = {
      ...c,
      hidden_size: c.dim, num_hidden_layers: c.n_layers, num_attention_heads: c.n_heads,
      num_key_value_heads: c.n_kv_heads, intermediate_size: c.hidden_dim, model_type: c.model_type ?? 'mistral',
      max_position_embeddings: c.max_seq_len ?? c.max_position_embeddings,
    };
    if (c.moe) { c.num_local_experts = c.moe.num_experts; c.num_experts_per_tok = c.moe.num_experts_per_tok; }
  }
  const inner = first(c.text_config, c.llm_config, c.language_config, c.thinker_config?.text_config);
  if (inner && typeof inner === 'object') {
    c = { ...c, ...inner, model_type_outer: c.model_type, architectures: c.architectures, quantization_config: c.quantization_config ?? inner.quantization_config };
  }
  return c;
}

function normType(t) {
  const s = String(t).toLowerCase();
  if (s.includes('sliding')) return 'sliding';
  if (s.includes('chunk')) return 'chunked';
  if (s.includes('linear') || s.includes('mamba') || s.includes('conv') || s.includes('recurrent') || s.includes('deltanet')) return 'linear';
  if (s.includes('full') || s === 'attention' || s.includes('global')) return 'full';
  return 'full';
}

export function analyzeArch(c) {
  const L = first(c.num_hidden_layers, c.n_layer, c.num_layers, c.n_layers);
  const hidden = first(c.hidden_size, c.n_embd, c.d_model);
  const nHeads = first(c.num_attention_heads, c.n_head, c.num_heads);
  let nKv = first(c.num_key_value_heads, c.multi_query_group_num, c.n_head_kv);
  if (nKv == null) nKv = c.multi_query ? 1 : nHeads;
  const headDim = first(c.head_dim, c.attention_head_dim, hidden && nHeads ? hidden / nHeads : undefined);
  const maxPos = first(c.max_position_embeddings, c.max_seq_len, c.seq_length, c.n_positions, c.max_sequence_length, 131072);
  const vocab = c.vocab_size ?? 32000;
  const mla = c.kv_lora_rank != null;
  const window = first(c.sliding_window, c.attention_chunk_size);

  // ---- per-layer attention types
  let types = new Array(L).fill('full');
  let typeSource = 'standard';
  const compress = Array.isArray(c.compress_ratios) ? c.compress_ratios.slice(0, L) : null;
  if (compress) {
    // DeepSeek-V4: every layer keeps a sliding window; most also keep a 1/r-compressed KV of the whole context.
    types = compress.map((r) => (r > 0 ? 'full' : 'sliding'));
    typeSource = 'compress_ratios';
  } else if (Array.isArray(c.layer_types) && c.layer_types.length) {
    types = Array.from({ length: L }, (_, i) => normType(c.layer_types[i] ?? 'full'));
    typeSource = 'layer_types';
  } else if (Array.isArray(c.attn_type_list) && c.attn_type_list.length) {
    types = Array.from({ length: L }, (_, i) => (c.attn_type_list[i] === 0 ? 'linear' : 'full'));
    typeSource = 'attn_type_list';
  } else if (typeof c.hybrid_override_pattern === 'string') {
    types = [...c.hybrid_override_pattern].map((ch) => (ch === '*' ? 'full' : ch === 'M' ? 'linear' : 'none'));
    typeSource = 'hybrid_override_pattern';
  } else if (c.full_attention_interval) {
    types = Array.from({ length: L }, (_, i) => ((i + 1) % c.full_attention_interval === 0 ? 'full' : 'linear'));
    typeSource = 'full_attention_interval';
  } else if (c.attn_layer_period) {
    types = Array.from({ length: L }, (_, i) => (i % c.attn_layer_period === (c.attn_layer_offset ?? 0) ? 'full' : 'linear'));
    typeSource = 'attn_layer_period';
  } else if (c.sliding_window_pattern && c.sliding_window) {
    types = Array.from({ length: L }, (_, i) => ((i + 1) % c.sliding_window_pattern === 0 ? 'full' : 'sliding'));
    typeSource = 'sliding_window_pattern';
  } else if (c.model_type === 'gemma2' && c.sliding_window) {
    types = Array.from({ length: L }, (_, i) => (i % 2 === 0 ? 'sliding' : 'full'));
    typeSource = 'gemma2';
  } else if (c.attention_chunk_size && Array.isArray(c.no_rope_layers)) {
    types = Array.from({ length: L }, (_, i) => (c.no_rope_layers[i] ? 'chunked' : 'full'));
    typeSource = 'no_rope_layers';
  } else if (c.sliding_window && c.use_sliding_window !== false && c.sliding_window < maxPos) {
    types = new Array(L).fill('sliding');
    typeSource = 'sliding_window';
  }
  const count = (t) => types.filter((x) => x === t).length;
  const layers = {
    full: count('full'),
    sliding: count('sliding') + count('chunked'),
    linear: count('linear'),
  };

  // ---- recurrent / linear-attention state per sequence per linear layer (bytes)
  let linearStateBytes = 0;
  if (layers.linear) {
    if (c.linear_num_value_heads) {
      // Gated DeltaNet (Qwen3-Next): recurrent state + conv state
      const vH = c.linear_num_value_heads, kH = c.linear_num_key_heads ?? vH;
      const kD = c.linear_key_head_dim ?? 128, vD = c.linear_value_head_dim ?? 128;
      const conv = (2 * kH * kD + vH * vD) * ((c.linear_conv_kernel_dim ?? 4) - 1);
      linearStateBytes = (vH * kD * vD + conv) * 2;
    } else if (c.mamba_num_heads) {
      linearStateBytes = c.mamba_num_heads * (c.mamba_head_dim ?? 64) * (c.ssm_state_size ?? 128) * 2;
    } else if (c.mamba_d_state || c.mamba_expand) {
      linearStateBytes = (c.mamba_expand ?? 2) * hidden * (c.mamba_d_state ?? 16) * 2;
    } else {
      // Lightning attention (MiniMax-M1/Text-01): per-head KV outer-product state, fp32
      linearStateBytes = nHeads * headDim * headDim * 4;
    }
  }

  // ---- MoE / params
  const experts = first(c.n_routed_experts, c.num_local_experts, c.num_experts, c.moe_num_experts);
  const topk = first(c.num_experts_per_tok, c.experts_per_token, c.moe_top_k, c.top_k);
  const moeInter = first(c.moe_intermediate_size, c.expert_intermediate_size, c.intermediate_size);
  const isMoE = experts > 1 && topk > 0;
  let moeLayers = 0;
  if (isMoE) {
    moeLayers = L - (c.first_k_dense_replace ?? 0) - (Array.isArray(c.mlp_only_layers) ? c.mlp_only_layers.length : 0);
    if (c.decoder_sparse_step > 1) moeLayers = Math.floor(moeLayers / c.decoder_sparse_step);
    if (c.moe_layer_freq > 1) moeLayers = Math.floor(moeLayers / c.moe_layer_freq);
    if (c.interleave_moe_layer_step > 1) moeLayers = Math.floor(moeLayers / c.interleave_moe_layer_step);
  }
  const expertParams = isMoE ? 3 * hidden * moeInter : 0;
  const tied = c.tie_word_embeddings === true;
  const embedParams = vocab * hidden * (tied ? 1 : 2);

  let attnKind = compress
    ? `Compressed attention (window ${window} + 1/${[...new Set(compress.filter((r) => r > 0))].sort((x, y) => x - y).join(', 1/')} KV)`
    : mla ? 'MLA (compressed latent KV)' : nKv === 1 ? 'MQA' : nKv < nHeads ? `GQA (${nHeads} q / ${nKv} kv heads)` : `MHA (${nHeads} heads)`;

  return {
    L, hidden, nHeads, nKv, headDim, maxPos, vocab, mla, window, types, compress, typeSource, layers,
    linearStateBytes, isMoE, experts, topk, moeLayers, expertParams, embedParams, tied, attnKind,
    kvLoraRank: c.kv_lora_rank, qkRope: c.qk_rope_head_dim ?? 64,
    indexerDim: c.index_topk ? (c.index_head_dim ?? 128) : 0,
    intermediate: c.intermediate_size ?? 4 * hidden,
    modelType: c.model_type,
  };
}

/** Rough parameter estimate when the repo has no safetensors metadata. */
export function estimateParams(a) {
  const attn = a.mla ? a.hidden * (a.nHeads * 192 * 1.2) * 2 : a.hidden * a.headDim * (2 * a.nHeads + 2 * a.nKv);
  const denseMlp = 3 * a.hidden * a.intermediate;
  let mlp = a.L * denseMlp;
  if (a.isMoE) mlp = (a.L - a.moeLayers) * denseMlp + a.moeLayers * a.experts * a.expertParams;
  return a.embedParams + a.L * attn + mlp;
}

export function activeParams(a, total) {
  if (!a.isMoE) return total;
  return Math.max(total - (a.experts - a.topk) * a.expertParams * a.moeLayers, 0);
}

// ---------------------------------------------------------------- memory model

/** KV-cache bytes per token for ONE attention layer, summed over all TP ranks. */
export function kvBytesPerTokenLayer(a, tp, kvBytes) {
  if (a.mla) {
    // vLLM replicates the compressed latent on every TP rank (DP-attention avoids this).
    return tp * ((a.kvLoraRank + a.qkRope) * kvBytes + a.indexerDim);
  }
  const headsPerRank = Math.max(1, Math.ceil(a.nKv / tp));
  return tp * headsPerRank * a.headDim * 2 * kvBytes;
}

/** Bytes of KV + recurrent state for a single sequence of `ctx` tokens. */
export function seqBytes(a, tp, kvBytes, ctx) {
  if (a.compress) {
    // single shared latent (head_dim) per token, replicated per TP rank; indexer keys (fp8) on 1/4 layers
    const per = tp * a.headDim * kvBytes;
    let total = 0;
    for (const r of a.compress) {
      total += Math.min(ctx, a.window ?? 128) * per;
      if (r > 0) total += Math.ceil(ctx / r) * (per + (r <= 4 ? tp * a.indexerDim : 0));
    }
    return total;
  }
  const per = kvBytesPerTokenLayer(a, tp, kvBytes);
  const win = a.window ?? ctx;
  return a.layers.full * ctx * per + a.layers.sliding * Math.min(ctx, win) * per + a.layers.linear * a.linearStateBytes;
}

export function weightBytes(model, precId) {
  const { params, nativeBytes, arch: a } = model;
  if (precId === 'native') return nativeBytes ?? params * 2;
  const p = PRECISIONS.find((x) => x.id === precId);
  if (precId === 'bf16') return params * 2;
  const emb = Math.min(a.embedParams, params * 0.5);
  return (params - emb) * p.bpp + emb * 2;
}

export function overheadGiB(a, tp) {
  // CUDA context + non-torch allocations + CUDA graphs + peak activations for an 8k-token prefill chunk.
  const ffn = a.isMoE ? Math.max(a.intermediate, a.topk * (a.expertParams / (3 * a.hidden)) * 2) : a.intermediate * 2;
  const act = (8192 * (a.hidden * 4 + ffn / tp) * 2) / GiB;
  return 1.5 + act;
}

/**
 * Evaluate one configuration.
 * opts: { gpu, count, prec, kvBytes, ctx, seqs, util, overrideOverhead }
 */
export function evaluate(model, opts) {
  const { gpu, count: tp, prec, kvBytes, ctx, seqs, util } = opts;
  const a = model.arch;
  const weights = weightBytes(model, prec);
  const overhead = (opts.overrideOverhead ?? overheadGiB(a, tp)) * GiB;
  const perGpuUsable = gpu.mem * GiB * util - overhead;
  const weightsPerGpu = weights / tp;
  const kvBudget = tp * perGpuUsable - weights; // total across GPUs
  const kvNeeded = seqBytes(a, tp, kvBytes, ctx) * seqs;
  const weightsFit = weightsPerGpu <= perGpuUsable;

  // max single-sequence context (binary search; handles sliding + linear layers)
  let maxCtx = 0;
  if (kvBudget > 0) {
    let lo = 0, hi = a.maxPos;
    if (seqBytes(a, tp, kvBytes, hi) <= kvBudget) lo = hi;
    else {
      while (hi - lo > 1) {
        const mid = Math.floor((lo + hi) / 2);
        if (seqBytes(a, tp, kvBytes, mid) <= kvBudget) lo = mid; else hi = mid;
      }
    }
    maxCtx = lo;
  }
  const perSeq = seqBytes(a, tp, kvBytes, ctx);
  const maxSeqs = kvBudget > 0 && perSeq > 0 ? Math.floor(kvBudget / perSeq) : 0;

  let status;
  if (!weightsFit || kvBudget <= 0) status = 'no';
  else if (kvNeeded > kvBudget) status = 'partial'; // weights fit, but not the requested ctx × seqs
  else if ((kvBudget - kvNeeded) / (tp * gpu.mem * GiB) < 0.05) status = 'tight';
  else status = 'yes';

  const sup = prec === 'native' ? support(gpu.arch, model.nativePrec) : support(gpu.arch, prec);
  const tpValid = a.nHeads % tp === 0 && (a.mla || a.nKv % tp === 0 || tp % a.nKv === 0);

  // single-stream decode ceiling: read all active weights + this sequence's KV once per token
  const activeW = weights * (model.active / model.params);
  const decodeTps = (gpu.bw * 1e12 * tp) / (activeW + perSeq);

  return {
    tp, prec, weights, weightsPerGpu, overhead, perGpuUsable, kvBudget, kvNeeded, weightsFit,
    maxCtx, maxSeqs, status, support: sup, tpValid, decodeTps, perSeq,
    total: gpu.mem * GiB * tp,
  };
}

/** Smallest GPU count (from `counts`) that satisfies the request. */
export function minGpus(model, opts, counts = [1, 2, 4, 8, 16]) {
  for (const n of counts) {
    const r = evaluate(model, { ...opts, count: n });
    if (r.status === 'yes' || r.status === 'tight') return { n, r };
  }
  return null;
}

// ---------------------------------------------------------------- quant detection

export function detectNativePrec(cfg, safetensors) {
  if (String(cfg?.expert_dtype ?? '').includes('fp4')) return 'fp4';
  const q = cfg?.quantization_config;
  if (q) {
    const m = String(q.quant_method ?? q.quant_algo ?? '').toLowerCase();
    if (m.includes('fp4') || m.includes('mxfp4') || m.includes('nvfp4')) return 'fp4';
    if (m === 'fp8' || m.includes('fp8')) return 'fp8';
    if (['awq', 'gptq', 'bitsandbytes', 'hqq', 'exl2', 'autoround', 'auto-round'].includes(m)) return (q.bits ?? q.w_bit ?? 4) <= 4 ? 'int4' : 'int8';
    if (m === 'compressed-tensors') {
      const s = JSON.stringify(q.config_groups ?? q).toLowerCase();
      if (s.includes('"num_bits":4') && s.includes('float')) return 'fp4';
      if (s.includes('"num_bits":4')) return 'int4';
      if (s.includes('float') && s.includes('"num_bits":8')) return 'fp8';
      if (s.includes('"num_bits":8')) return 'int8';
    }
  }
  const p = safetensors?.parameters ?? {};
  const big = Object.entries(p).sort((x, y) => y[1] - x[1])[0]?.[0] ?? '';
  if (big.startsWith('F8')) return 'fp8';
  return 'bf16';
}

export const fmtGiB = (b) => {
  const g = b / GiB;
  if (g >= 1000) return `${(g / 1024).toFixed(2)} TiB`;
  if (g >= 100) return `${g.toFixed(0)} GiB`;
  if (g >= 10) return `${g.toFixed(1)} GiB`;
  return `${g.toFixed(2)} GiB`;
};

export const fmtParams = (n) => {
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(n >= 1e11 ? 0 : 1)}B`;
  return `${(n / 1e6).toFixed(0)}M`;
};

export const fmtTokens = (n) => {
  if (n >= 1048576) return `${(n / 1048576).toFixed(n % 1048576 ? 1 : 0)}M`;
  if (n >= 1024) return `${Math.floor(n / 1024)}K`;
  return `${n}`;
};
