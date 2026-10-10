# Krea2 LoRA

Krea2（K2）是单流 MMDiT + Qwen3-VL-4B + Qwen-Image VAE。本仓库做 **LoRA / LoKr**，移植自 [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner)，走现有 `accelerate` + `/api/run`，不另起 Fast 式插件。

## 官方流程

在 **RAW** 上训 LoRA 或 LoKr，推理用 **Turbo**。侧栏没有全量微调入口（musubi 也没有 `krea2_train.py`）。

权重（需自行准备，部分仓库有门控）：

- DiT RAW：`krea/Krea-2-Raw` 的 `raw.safetensors`
- 推理 Turbo：`krea/Krea-2-Turbo`
- 文本：Qwen3-VL-4B-Instruct 的 **bf16** 权重——Comfy 发行的 `Comfy-Org/Krea-2` 里 `text_encoders/qwen3vl_4b_bf16.safetensors`，或 Krea-2-Raw 自带的 `text_encoder/model.safetensors`。fp8 / 量化版（如 `qwen3vl_4b_fp8_scaled`）不支持，加载时会直接报错说明
- VAE：Qwen-Image VAE（与 Anima 同款，`library/qwen_image_autoencoder_kl.py`）

默认路径：`./sd-models/krea2/`。

分词器（`vocab.json` 等）默认会向 Hugging Face 拉 `Qwen/Qwen3-VL-4B-Instruct`。国内网络超时就把这些文件放到下面任一目录（有 `vocab.json` 或 `tokenizer.json` 即可），开训会走本地、不再访问 Hub。数据集分词与文本编码器用同一份：

- `./tokenizer-cache/Qwen_Qwen3-VL-4B-Instruct/`
- 文本编码器所在目录下的 `qwen3_vl_tokenizer/`（默认即 `./sd-models/krea2/qwen3_vl_tokenizer/`）
- 文本编码器若是 diffusers 目录结构（`Krea-2-Raw/text_encoder/model.safetensors`），会直接用同级的 `Krea-2-Raw/tokenizer/`

需要：`vocab.json`、`merges.txt`、`tokenizer.json`、`tokenizer_config.json`、`special_tokens_map.json`（有 `added_tokens.json` 也一并放进去）；Krea-2-Raw 的 `tokenizer/` 只有 `tokenizer.json` + `tokenizer_config.json` 也可以。transformers 5.x 存出来的 `tokenizer_config.json` 把 `extra_special_tokens` 写成列表，本仓库的 transformers 4.51.3 原本读不了（`'list' object has no attribute 'keys'`），现在加载时自动按 `additional_special_tokens` 读取，不改动文件。

## 调度器

训练时间步：

| `timestep_sampling` | 作用 |
|---|---|
| `shift` | 固定 `discrete_flow_shift`（默认 **2.5**） |
| `krea2_shift` | 分辨率感知 μ：256px → 0.5，1280px → 1.15（`x2=6400`）。不要再配自定义固定 2.5 |
| `flux_shift` | 同族线性，但 `x2=4096`，1024px 已饱和 |
| `sigmoid` / `uniform` | musubi 可选采样 |
| `weighting_scheme` | 默认 `none` |

预览只实现 **euler**：

- RAW：约 28 步，CFG 5.5，μ 按分辨率算
- Turbo：约 8 步，CFG 1，钉死 μ=`1.15`（打开 `turbo_dit`）
- `sample_mu` 可覆盖自动 μ

## 显存

- 只支持 `mixed_precision = bf16`
- FP8 只有缩放 FP8 一条路：`fp8_base`、`fp8_scaled` 打开任一个都按缩放 FP8 训练（只开 `fp8_base` 时自动补上 `fp8_scaled`）。DiT 在加载时量化、计算时反量化回 bf16，不需要原生 FP8 张量核，30 系也能用。不再做整模强转 fp8——那会把 norm 和量化 scale 一起转坏
- `blocks_to_swap` 最大 26；与 `turbo_dit` 互斥
- 文本编码器只构建 Qwen3-VL 的语言塔（transformers 的 `Qwen3Model`，不加载用不到的视觉塔），所以直接跑在本仓库固定的 transformers 4.51.3 上，**不需要升级全局 transformers**。纯文本输入时 Qwen3-VL 的交错 M-RoPE 等同普通 RoPE，并按 musubi-tuner `b3f3d65d` 显式传入 position_ids（后缀接在 padding 之后），有效 token 的输出与 musubi-tuner 用的 Qwen3-VL 逐位一致。建议缓存文本输出。

## 数据集

沿用 kohya：`train_data_dir` + 分桶 + latent / TE 磁盘缓存。不要写 musubi 的 `dataset_config.toml`。

适配器：

- **LoRA**：`networks.lora_krea2`（圈选 `SingleStreamDiT` 下全部 Linear，默认 dim/alpha 32）
- **LoKr**：`lycoris.kohya` + `config/lycoris_krea2_preset.toml`（同样只圈 `SingleStreamDiT`；不要再列子类，避免重复包装）
