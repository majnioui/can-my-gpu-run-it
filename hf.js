// canirun — Hugging Face Hub loader. Runs in the browser (and Node 18+ for tests).
import { normalizeConfig, analyzeArch, estimateParams, activeParams, detectNativePrec } from './calc.js';

const HUB = 'https://huggingface.co';

export function parseRepo(input) {
  let s = String(input).trim();
  s = s.replace(/^https?:\/\/(www\.)?(huggingface\.co|hf\.co)\//, '');
  s = s.replace(/\/(tree|blob|resolve)\/.*$/, '').replace(/\/+$/, '');
  const m = s.match(/^([\w.-]+\/[\w.-]+)/);
  return m ? m[1] : null;
}

async function get(url, token, asJson = true) {
  const headers = token ? { Authorization: `Bearer ${token}` } : {};
  const res = await fetch(url, { headers });
  if (!res.ok) {
    const err = new Error(`HTTP ${res.status}`);
    err.status = res.status;
    throw err;
  }
  return asJson ? res.json() : res.text();
}

/** Bytes of the weight files that would actually be loaded. */
function checkpointBytes(siblings = []) {
  const top = siblings.filter((s) => !s.rfilename.includes('/') && s.rfilename.endsWith('.safetensors') && s.size);
  const hfStyle = top.filter((s) => /^model[-.]/.test(s.rfilename));
  const files = hfStyle.length ? hfStyle : top;
  const sum = files.reduce((n, s) => n + s.size, 0);
  return sum || null;
}

async function loadConfig(repo, token) {
  try {
    return { cfg: await get(`${HUB}/${repo}/resolve/main/config.json`, token), from: repo };
  } catch (e) {
    if (e.status === 404) {
      try { return { cfg: await get(`${HUB}/${repo}/resolve/main/params.json`, token), from: repo }; } catch {}
    }
    throw e;
  }
}

export async function loadModel(input, token) {
  const repo = parseRepo(input);
  if (!repo) throw new Error('Enter a Hugging Face repo like deepseek-ai/DeepSeek-V3 or paste its URL.');

  let info;
  try {
    info = await get(`${HUB}/api/models/${repo}?blobs=true`, token);
  } catch (e) {
    if (e.status === 401 || e.status === 404) throw new Error(`Model "${repo}" not found (or it's private — add an HF token).`);
    throw new Error(`Couldn't reach Hugging Face (${e.message}).`);
  }

  const warnings = [];
  let rawCfg = null, cfgFrom = repo;
  try {
    ({ cfg: rawCfg, from: cfgFrom } = await loadConfig(repo, token));
  } catch (e) {
    // GGUF / quant repos often lack config.json — fall back to the base model.
    const base = [].concat(info.cardData?.base_model ?? []).find(Boolean);
    if (base && base !== repo) {
      try {
        ({ cfg: rawCfg, from: cfgFrom } = await loadConfig(base, token));
        warnings.push(`Architecture read from base model ${base}.`);
      } catch {}
    }
    if (!rawCfg) {
      const gated = e.status === 401 || e.status === 403;
      const err = new Error(
        gated
          ? `"${repo}" is gated — its config.json needs an HF token from an account that accepted the license. Add one under "HF token", or try an ungated mirror (e.g. unsloth/…).`
          : `Couldn't read config.json for "${repo}".`,
      );
      err.gated = gated;
      throw err;
    }
  }

  const cfg = normalizeConfig(rawCfg);
  const arch = analyzeArch(cfg);
  if (!arch.L || !arch.hidden) throw new Error(`Unrecognised architecture in ${cfgFrom}/config.json (${cfg.model_type ?? 'unknown model_type'}).`);

  let params = info.safetensors?.total ?? info.gguf?.total ?? null;
  let paramsEstimated = false;
  if (!params) {
    params = estimateParams(arch);
    paramsEstimated = true;
    warnings.push('No safetensors metadata — parameter count is estimated from config.json.');
  }

  let nativePrec = detectNativePrec(cfg, info.safetensors);
  let nativeBytes = checkpointBytes(info.siblings);
  if (!nativeBytes || String(cfg.torch_dtype ?? cfg.dtype).includes('float32')) {
    nativeBytes = null; // fp32 checkpoints are loaded as bf16
    if (nativePrec === 'bf16' && String(cfg.torch_dtype ?? cfg.dtype).includes('float32')) warnings.push('Checkpoint is FP32; servers load it as BF16.');
  }
  if (info.gguf) {
    warnings.push('GGUF repo: sizes below are for safetensors-style serving (vLLM/SGLang). For llama.cpp, use the size of the .gguf file you pick.');
    nativeBytes = null;
    nativePrec = 'bf16';
  }

  return {
    repo,
    cfgFrom,
    cfg,
    arch,
    params,
    paramsEstimated,
    active: activeParams(arch, params),
    nativePrec,
    nativeBytes,
    gated: info.gated,
    pipeline: info.pipeline_tag,
    likes: info.likes,
    downloads: info.downloads,
    warnings,
  };
}
