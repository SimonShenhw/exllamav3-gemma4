# ExLlamaV3-Gemma4-Inference-Fix

> **TL;DR**: The current ExLlamaV3 main branch and existing community PRs for Gemma 4 suffer from a critical `layer_scalar` inference bug that causes garbled output or model deadlock under `torch.inference_mode()`. This repository provides a fully working fork with the fix applied, enabling correct EXL3 quantization and inference for Gemma 4 31B.

---

## The Problem

Google's [Gemma 4](https://blog.google/technology/developers/gemma-4/) (released April 2, 2026) introduces **per-layer output scalars** (`layer_scalar`) that modulate each transformer block's residual stream. These scalars range from 0.036 to 0.99, and are essential for numerical stability across Gemma 4's unusually deep 60-layer architecture.

In the reference [ExLlamaV3](https://github.com/turboderp-org/exllamav3) Gemma 4 implementation, `layer_scalar` is loaded as an `nn.Parameter`:

```python
# BROKEN: nn.Parameter triggers autograd tracking
self.layer_scalar = nn.Parameter(layer_scalar, requires_grad=False)
# ...
x = x * self.layer_scalar.to(dtype=x.dtype)  # CRASHES under inference_mode()
```

ExLlamaV3's inference engine runs under `torch.inference_mode()`, which forbids operations on autograd-tracked tensors. The result: **`RuntimeError: Inference tensors cannot be saved for backward`** on every forward pass, producing either a crash or silent garbage output.

## The Fix

Convert `layer_scalar` from `nn.Parameter` to a plain Python `float` via `.float().item()`, completely decoupling it from the autograd graph:

```python
# FIXED: plain Python scalar, zero autograd overhead
self.layer_scalar = layer_scalar.float().item()
# ...
x = x * self.layer_scalar  # works under inference_mode(), no graph tracking
```

This one-line change restores correct inference across all 60 layers. The model now produces coherent, semantically accurate output:

```
Q: What is the capital of France?
A: The capital of France is Paris.

Q: Write a Python function to check if a number is prime.
A: def is_prime(n):
       if n < 2: return False
       if n == 2: return True
       if n % 2 == 0: return False
       for i in range(3, int(n**0.5) + 1, 2):
           if n % i == 0: return False
       return True
```

---

## Credits

**The core Gemma 4 architecture adaptation for ExLlamaV3 was built by [@lesj0610](https://github.com/lesj0610).** Their `feat/gemma4-support` branch ([lesj0610/exllamav3](https://github.com/lesj0610/exllamav3/tree/feat/gemma4-support)) contributes ~2,300 lines of custom implementation including:

- `Gemma4Attention` with proper V-norm (weightless RMSNorm on value states)
- `Gemma4TransformerBlock` with integrated `layer_scalar` and per-layer-embedding support
- Manual attention fallback (`decode_flash_attn_fallback`) for head_dim > 256 with KV cache reconstruction
- K=V weight sharing for global attention layers (`v_proj=None`, `v=k`)
- Proportional RoPE (p-RoPE) with partial rotary factor

This repository applies the inference compatibility fix on top of their work and packages the result for immediate use.

---

## Quantized Model

| Variant | Size | bpw | Format | Link |
|---|---|---|---|---|
| Gemma-4-31B-it | 25 GB | 6.0 | EXL3 | [SimonShenhw/Gemma-4-31B-it-EXL3-6.0bpw](https://huggingface.co/SimonShenhw/Gemma-4-31B-it-EXL3-6.0bpw) |

> Original BF16 model: 62 GB. Compression ratio: **2.5x**. Fits on a single RTX 4090/5090 (32 GB VRAM).

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/SimonShenhw/exllamav3-gemma4.git
cd exllamav3-gemma4

# Install with CUDA support (requires CUDA Toolkit + C++ compiler)
pip install ./exllamav3 --no-build-isolation --no-deps
```

### 2. Download Quantized Weights

```bash
# From HuggingFace
huggingface-cli download SimonShenhw/Gemma-4-31B-it-EXL3-6.0bpw \
    --local-dir ./models/gemma-4-31B-it-exl3-6.0bpw
```

### 3. Run Inference

```python
from exllamav3 import Config, Model, Tokenizer
import torch

model_dir = "./models/gemma-4-31B-it-exl3-6.0bpw"

config = Config.from_directory(model_dir)
model = config.model_classes["text"](config)
model.load(device="cuda:0")  # ~25 GB VRAM
tokenizer = Tokenizer(config)

prompt = (
    "<bos><start_of_turn>user\n"
    "What is the capital of France? Answer in one sentence."
    "<end_of_turn>\n<start_of_turn>model\n"
)
ids = tokenizer.encode(prompt, encode_special_tokens=True).to("cuda:0")
params = {"attn_mode": "flash_attn_nc", "position": 0}

generated = ids.clone()
for _ in range(128):
    logits = model.forward_ls(generated, params, model.logit_layer_idx, model.modules)
    next_id = logits[0, -1].argmax().item()
    if next_id in (1, 106):  # EOS tokens
        break
    generated = torch.cat(
        [generated, torch.tensor([[next_id]], device="cuda:0")], dim=1
    )

output = tokenizer.decode(generated[0, ids.shape[1]:].unsqueeze(0))
print(output)
# >>> The capital of France is Paris.
```

---

## System Requirements

| Component | Minimum |
|---|---|
| GPU VRAM | 28 GB (6.0 bpw) |
| CUDA Toolkit | 12.8+ |
| Python | 3.10+ |
| PyTorch | 2.6+ |
| flash_attn | 2.7+ |

Tested on: RTX 5090 (32 GB), CUDA 13.0, Python 3.14, PyTorch 2.11.

---

## Architecture Overview

Gemma 4 31B introduces several features not present in Gemma 3:

| Feature | Detail | Implementation |
|---|---|---|
| **layer_scalar** | Per-layer output scaling (0.036-0.99) | `Gemma4TransformerBlock.forward()` |
| **Dual head dimensions** | Local: 256, Global: 512 | Per-layer routing in `Gemma4TextConfig` |
| **Dual KV heads** | Local: 16, Global: 4 | Per-layer `num_kv_heads` selection |
| **K=V sharing** | Global layers: no v_proj, V reuses K | `v_proj=None`, `v=k` in projection |
| **Proportional RoPE** | 25% partial rotation for global layers | `_rope_params_proportional()` |
| **V-norm** | Weightless RMSNorm on value states | `Gemma4Attention.v_norm` |
| **RMSNorm convention** | `norm(x) * weight` (no +1 bias) | `constant_bias=0.0` (vs Gemma 3's 1.0) |

---

## Known Limitations

1. **No KV cache for global layers in `flash_attn_nc` mode**: Global attention layers (head_dim=512) fall back to manual attention with full recomputation. Use the Generator API with `flash_attn` mode for cached decode.

2. **Vision model not included**: This fork focuses on text-only inference (`Gemma4ForCausalLM` / `Gemma4ForConditionalGeneration` text path). The multimodal vision encoder is not yet integrated.

3. **act_limit=100**: MLP activations are clamped during quantization calibration to prevent fp16 gelu overflow. This may marginally affect quantization quality for edge-case activations.

---

## License

This project inherits licenses from:
- [ExLlamaV3](https://github.com/turboderp-org/exllamav3) (MIT License)
- [Gemma 4](https://ai.google.dev/gemma/terms) (Gemma Terms of Use)
