# Krea2 LoRA

Krea2（K2）是单流 MMDiT + Qwen3-VL-4B + Qwen-Image VAE。本仓库做 **LoRA / LoKr**，移植自 [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner)，走现有 `accelerate` + `/api/run`，不另起 Fast 式插件。

## 官方流程

在 **RAW** 上训 LoRA 或 LoKr，推理用 **Turbo**。侧栏没有全量微调入口（musubi 也没有 `krea2_train.py`）。

权重（需自行准备，部分仓库有门控）：

- DiT RAW：`krea/Krea-2-Raw` 的 `raw.safetensors`
- 推理 Turbo：`krea/Krea-2-Turbo`
- 文本：Comfy 发行的 Qwen3-VL-4B-Instruct
- VAE：Qwen-Image VAE（与 Anima 同款，`library/qwen_image_autoencoder_kl.py`）

默认路径：`./sd-models/krea2/`。

分词器（`vocab.json` 等）默认会向 Hugging Face 拉 `Qwen/Qwen3-VL-4B-Instruct`。国内网络超时就把这些文件放到下面任一目录（有 `vocab.json` 或 `tokenizer.json` 即可），开训会走本地、不再访问 Hub：

- `./tokenizer-cache/Qwen_Qwen3-VL-4B-Instruct/`
- `./sd-models/krea2/qwen3_vl_tokenizer/`

需要：`vocab.json`、`merges.txt`、`tokenizer.json`、`tokenizer_config.json`、`special_tokens_map.json`（有 `added_tokens.json` 也一并放进去）。

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

- `--fp8_base` + `--fp8_scaled` 必须一起用缩放 FP8
- `blocks_to_swap` 最大 26；与 `turbo_dit` 互斥
- 文本编码器需要能 import `Qwen3VLForConditionalGeneration` 的 transformers（通常 ≥ 4.57）。**本仓库不为此升级全局 transformers 4.51.3**；缺依赖时抛 `Krea2TextEncoderUnavailableError`。建议先缓存文本输出。

## 数据集

沿用 kohya：`train_data_dir` + 分桶 + latent / TE 磁盘缓存。不要写 musubi 的 `dataset_config.toml`。

适配器：

- **LoRA**：`networks.lora_krea2`（圈选 `SingleStreamDiT` 下全部 Linear，默认 dim/alpha 32）
- **LoKr**：`lycoris.kohya` + `config/lycoris_krea2_preset.toml`（同样只圈 `SingleStreamDiT`；不要再列子类，避免重复包装）
