# Can my GPU run it?

Check whether an open model from Hugging Face fits on your GPUs, how many you need, and how much context you can serve.

Paste a model (for example `deepseek-ai/DeepSeek-V3`), then set the sentence: *Run it on 8 × H200 in FP8 with 32K tokens of context for 1 request at a time.* The page reads the model's `config.json` and safetensors metadata straight from the Hugging Face Hub (no weights are downloaded) and works out:

- **Weights:** the exact published size, or the size in BF16, FP8, INT8, INT4 or FP4
- **KV cache:** sized for the real architecture: GQA, MLA (DeepSeek, Kimi), sliding window (gpt-oss, Gemma), linear and hybrid attention (MiniMax-M1, Qwen3-Next), and DeepSeek-V4's compressed attention
- **Fit:** whether it fits, the longest context for one request, and how many requests can run at once
- **GPUs needed:** the fewest GPUs of each type for every precision, across 59 GPUs (NVIDIA, AMD, Intel Gaudi, Apple Silicon, DGX Spark), or your own GPU's specs
- **Support:** whether the GPU runs that precision natively (FP8 on Ada, Hopper and Blackwell; FP4 only on Blackwell and MI350/355)

Every result is a shareable link.

## Run locally

It's a static site with no build step and no backend.

```sh
python3 -m http.server 8765   # then open http://localhost:8765
```

It uses ES modules, so open it over HTTP, not `file://`. It deploys as-is to GitHub Pages, Cloudflare Pages or any static host.

Gated models (Llama, Gemma…) need a Hugging Face token. The token stays in the browser's `localStorage` and is sent only to huggingface.co.

## SEO pages

`scripts/build.mjs` generates static pages from live Hugging Face data:

- `/models/<model>/`: GPUs needed per GPU and precision, context limits, memory breakdown and FAQ for ~35 popular models
- `/gpus/<gpu>/`: which models fit on each of 16 popular GPUs
- `/models/`, `/gpus/`: hub pages, plus the links section on the homepage
- `sitemap.xml` and `robots.txt`

```sh
node scripts/build.mjs   # re-run to refresh data or after editing the model list
```

The generated files are committed, so the site still deploys as plain static files.

## Files

- `calc.js`: GPU specs, precision support, and the memory model (pure functions, runs in Node too)
- `hf.js`: Hugging Face loader: repo parsing, config fallbacks, quantization detection
- `app.js`: the interface and the link state
- `scripts/build.mjs`: the SEO page generator (edit `MODELS` and `GPU_PAGES` there)

## Accuracy

Estimates assume vLLM or SGLang serving with tensor parallelism. Treat results within about 5% of the limit as borderline. The decode speed shown is a memory-bandwidth ceiling; real speed is usually 40–70% of it.

---

Made by [Moe Majnioui](https://majnioui.me)
