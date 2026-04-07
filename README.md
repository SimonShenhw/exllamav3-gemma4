# ExLlamaV3-Gemma4-Inference-Fix

**[English](#english) | [中文](#中文)**

---

<a id="english"></a>

## English

> **TL;DR**: The current ExLlamaV3 main branch and existing community PRs for Gemma 4 suffer from a critical `layer_scalar` inference bug that causes garbled output or model deadlock under `torch.inference_mode()`. This repository provides a fully working fork with the fix applied, enabling correct EXL3 quantization and inference for Gemma 4 31B.

### The Problem

Google's [Gemma 4](https://blog.google/technology/developers/gemma-4/) (released April 2, 2026) introduces **per-layer output scalars** (`layer_scalar`) that modulate each transformer block's residual stream. These scalars range from 0.036 to 0.99, and are essential for numerical stability across Gemma 4's unusually deep 60-layer architecture.

In the reference [ExLlamaV3](https://github.com/turboderp-org/exllamav3) Gemma 4 implementation, `layer_scalar` is loaded as an `nn.Parameter`:

```python
# BROKEN: nn.Parameter triggers autograd tracking
self.layer_scalar = nn.Parameter(layer_scalar, requires_grad=False)
# ...
x = x * self.layer_scalar.to(dtype=x.dtype)  # CRASHES under inference_mode()
```

ExLlamaV3's inference engine runs under `torch.inference_mode()`, which forbids operations on autograd-tracked tensors. The result: **`RuntimeError: Inference tensors cannot be saved for backward`** on every forward pass, producing either a crash or silent garbage output.

### The Fix

Convert `layer_scalar` from `nn.Parameter` to a plain Python `float` via `.float().item()`, completely decoupling it from the autograd graph:

```python
# FIXED: plain Python scalar, zero autograd overhead
self.layer_scalar = layer_scalar.float().item()
# ...
x = x * self.layer_scalar  # works under inference_mode(), no graph tracking
```

This one-line change restores correct inference across all 60 layers:

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

### Credits

**The core Gemma 4 architecture adaptation for ExLlamaV3 was built by [@lesj0610](https://github.com/lesj0610).** Their `feat/gemma4-support` branch ([lesj0610/exllamav3](https://github.com/lesj0610/exllamav3/tree/feat/gemma4-support)) contributes ~2,300 lines of custom implementation including:

- `Gemma4Attention` with proper V-norm (weightless RMSNorm on value states)
- `Gemma4TransformerBlock` with integrated `layer_scalar` and per-layer-embedding support
- Manual attention fallback (`decode_flash_attn_fallback`) for head_dim > 256 with KV cache reconstruction
- K=V weight sharing for global attention layers (`v_proj=None`, `v=k`)
- Proportional RoPE (p-RoPE) with partial rotary factor

This repository applies the inference compatibility fix on top of their work and packages the result for immediate use.

### Quantized Model

| Variant | Size | bpw | Format | Link |
|---|---|---|---|---|
| Gemma-4-31B-it | 25 GB | 6.0 | EXL3 | [HaoweiShen/Gemma-4-31B-it-EXL3-6.0bpw](https://huggingface.co/HaoweiShen/Gemma-4-31B-it-EXL3-6.0bpw) |

> Original BF16 model: 62 GB. Compression ratio: **2.5x**. Fits on a single RTX 4090/5090 (32 GB VRAM).

### Quick Start

#### 1. Install

```bash
git clone https://github.com/SimonShenhw/exllamav3-gemma4.git
cd exllamav3-gemma4

# Install with CUDA support (requires CUDA Toolkit + C++ compiler)
pip install . --no-build-isolation --no-deps
```

#### 2. Download Quantized Weights

```bash
huggingface-cli download HaoweiShen/Gemma-4-31B-it-EXL3-6.0bpw \
    --local-dir ./models/gemma-4-31B-it-exl3-6.0bpw
```

#### 3. Run Inference

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

### System Requirements

| Component | Minimum |
|---|---|
| GPU VRAM | 28 GB (6.0 bpw) |
| CUDA Toolkit | 12.8+ |
| Python | 3.10+ |
| PyTorch | 2.6+ |
| flash_attn | 2.7+ |

Tested on: RTX 5090 (32 GB), CUDA 13.0, Python 3.14, PyTorch 2.11.

### Architecture Overview

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

### Known Limitations

1. **No KV cache for global layers in `flash_attn_nc` mode**: Global attention layers (head_dim=512) fall back to manual attention with full recomputation. Use the Generator API with `flash_attn` mode for cached decode.

2. **Vision model not included**: This fork focuses on text-only inference. The multimodal vision encoder is not yet integrated.

3. **act_limit=100**: MLP activations are clamped during quantization calibration to prevent fp16 gelu overflow. This may marginally affect quantization quality for edge-case activations.

### License

- [ExLlamaV3](https://github.com/turboderp-org/exllamav3) (MIT License)
- [Gemma 4](https://ai.google.dev/gemma/terms) (Gemma Terms of Use)

---

<a id="中文"></a>

## 中文

> **一句话概要**：当前 ExLlamaV3 主分支及社区 PR 在加载 Gemma 4 时存在 `layer_scalar` 推理 Bug，导致输出乱码或模型崩溃。本仓库彻底修复了该问题，实现了 Gemma 4 31B 的完整 EXL3 量化与推理。

### 问题描述

Google [Gemma 4](https://blog.google/technology/developers/gemma-4/)（2026 年 4 月 2 日发布）引入了**逐层输出缩放因子** (`layer_scalar`)，用于调节每个 Transformer Block 的残差流。这些标量值范围在 0.036 到 0.99 之间，对 Gemma 4 深达 60 层架构的数值稳定性至关重要。

在现有的 ExLlamaV3 Gemma 4 实现中，`layer_scalar` 被加载为 `nn.Parameter`：

```python
# 有 Bug：nn.Parameter 会触发自动求导追踪
self.layer_scalar = nn.Parameter(layer_scalar, requires_grad=False)
# ...
x = x * self.layer_scalar.to(dtype=x.dtype)  # 在 inference_mode() 下崩溃！
```

ExLlamaV3 的推理引擎在 `torch.inference_mode()` 下运行，禁止对带有自动求导追踪的张量进行操作。结果就是：**`RuntimeError: Inference tensors cannot be saved for backward`**，导致前向传播崩溃或输出乱码。

### 修复方案

将 `layer_scalar` 从 `nn.Parameter` 转换为纯 Python `float`（通过 `.float().item()`），彻底脱离自动求导图：

```python
# 修复：纯 Python 标量，零自动求导开销
self.layer_scalar = layer_scalar.float().item()
# ...
x = x * self.layer_scalar  # 在 inference_mode() 下正常工作
```

这一行改动恢复了全部 60 层的正确推理：

```
Q: 法国的首都是哪里？
A: The capital of France is Paris.

Q: 写一个 Python 函数判断素数
A: def is_prime(n):
       if n < 2: return False
       ...（完整正确的实现）

Q: 用法语、日语和西班牙语说你好
A: Bonjour / こんにちは / Hola
```

### 致谢

**ExLlamaV3 的 Gemma 4 核心架构适配由 [@lesj0610](https://github.com/lesj0610) 完成。** 其 `feat/gemma4-support` 分支 ([lesj0610/exllamav3](https://github.com/lesj0610/exllamav3/tree/feat/gemma4-support)) 贡献了约 2,300 行自定义实现，包括：

- `Gemma4Attention`：含 V-norm（无权重 RMSNorm 作用于 Value 状态）
- `Gemma4TransformerBlock`：内置 `layer_scalar` 与逐层嵌入支持
- 手动注意力回退机制 (`decode_flash_attn_fallback`)：支持 head_dim > 256 时的 KV Cache 重建
- Global 注意力层的 K=V 权重共享（`v_proj=None`, `v=k`）
- 比例旋转位置编码 (Proportional RoPE / p-RoPE)

本仓库在其工作基础上完成了推理兼容性修复，并打包为可直接使用的版本。

### 量化模型

| 模型 | 大小 | bpw | 格式 | 链接 |
|---|---|---|---|---|
| Gemma-4-31B-it | 25 GB | 6.0 | EXL3 | [HaoweiShen/Gemma-4-31B-it-EXL3-6.0bpw](https://huggingface.co/HaoweiShen/Gemma-4-31B-it-EXL3-6.0bpw) |

> 原始 BF16 模型：62 GB。压缩比：**2.5 倍**。可在单张 RTX 4090/5090（32 GB 显存）上运行。

### 快速上手

#### 1. 安装

```bash
git clone https://github.com/SimonShenhw/exllamav3-gemma4.git
cd exllamav3-gemma4

# 编译安装（需要 CUDA Toolkit + C++ 编译器）
pip install . --no-build-isolation --no-deps
```

#### 2. 下载量化权重

```bash
huggingface-cli download HaoweiShen/Gemma-4-31B-it-EXL3-6.0bpw \
    --local-dir ./models/gemma-4-31B-it-exl3-6.0bpw
```

#### 3. 运行推理

```python
from exllamav3 import Config, Model, Tokenizer
import torch

model_dir = "./models/gemma-4-31B-it-exl3-6.0bpw"

config = Config.from_directory(model_dir)
model = config.model_classes["text"](config)
model.load(device="cuda:0")  # 约 25 GB 显存
tokenizer = Tokenizer(config)

prompt = (
    "<bos><start_of_turn>user\n"
    "法国的首都是哪里？用一句话回答。"
    "<end_of_turn>\n<start_of_turn>model\n"
)
ids = tokenizer.encode(prompt, encode_special_tokens=True).to("cuda:0")
params = {"attn_mode": "flash_attn_nc", "position": 0}

generated = ids.clone()
for _ in range(128):
    logits = model.forward_ls(generated, params, model.logit_layer_idx, model.modules)
    next_id = logits[0, -1].argmax().item()
    if next_id in (1, 106):  # EOS 终止符
        break
    generated = torch.cat(
        [generated, torch.tensor([[next_id]], device="cuda:0")], dim=1
    )

output = tokenizer.decode(generated[0, ids.shape[1]:].unsqueeze(0))
print(output)
```

### 系统要求

| 组件 | 最低要求 |
|---|---|
| GPU 显存 | 28 GB（6.0 bpw） |
| CUDA Toolkit | 12.8+ |
| Python | 3.10+ |
| PyTorch | 2.6+ |
| flash_attn | 2.7+ |

测试环境：RTX 5090（32 GB）、CUDA 13.0、Python 3.14、PyTorch 2.11。

### 架构概览

Gemma 4 31B 相比 Gemma 3 引入了以下新特性：

| 特性 | 细节 | 实现位置 |
|---|---|---|
| **layer_scalar** | 逐层输出缩放（0.036-0.99） | `Gemma4TransformerBlock.forward()` |
| **双头维度** | Local: 256, Global: 512 | `Gemma4TextConfig` 逐层路由 |
| **双 KV 头数** | Local: 16, Global: 4 | 逐层 `num_kv_heads` 选择 |
| **K=V 共享** | Global 层无 v_proj，V 复用 K | 投影时 `v_proj=None`, `v=k` |
| **比例旋转编码** | Global 层仅 25% 维度旋转 | `_rope_params_proportional()` |
| **V-norm** | 无权重 RMSNorm 作用于 Value | `Gemma4Attention.v_norm` |
| **RMSNorm 约定** | `norm(x) * weight`（无 +1 偏置） | `constant_bias=0.0`（Gemma 3 为 1.0） |

### 已知限制

1. **`flash_attn_nc` 模式下 Global 层无 KV Cache**：Global 注意力层（head_dim=512）回退到手动注意力全量重算。使用 Generator API 的 `flash_attn` 模式可启用缓存解码。

2. **未包含视觉模型**：本分支专注于纯文本推理。多模态视觉编码器尚未集成。

3. **act_limit=100**：量化校准时 MLP 激活值被限制在 [-100, 100] 以防止 fp16 下 gelu 溢出，可能对极端激活值的量化质量产生微小影响。

### 许可证

- [ExLlamaV3](https://github.com/turboderp-org/exllamav3)（MIT License）
- [Gemma 4](https://ai.google.dev/gemma/terms)（Gemma Terms of Use）
