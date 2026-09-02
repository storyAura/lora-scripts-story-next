# `vendor/lycoris/` 来源说明

本目录是 **LyCORIS 4.0.0 + 本地 overlay**，随仓库一起分发，用于覆盖 pip 安装的上游包。

## 为什么要 vendor

`requirements.txt` 里的 `lycoris-lora==4.0.0` 是上游官方包，**不包含**本项目的本地扩展算法与 Anima 适配修复。此前这些改动只存在于 `venv/Lib/site-packages/lycoris/`（被 `.gitignore` 排除），导致：

- 换机器 / 重建 venv 就全部丢失
- 对 `lycoris-lora` 执行 `--force-reinstall` 会静默退回官方版
- 需要人工在两个副本之间同步，漏同步不会有任何报错

vendor 进仓库后，改动跟着 git 走，并由脚本 + 测试保证一致。

## 安装 / 同步

创建或重装 venv 之后执行：

```bash
python scripts/sync_vendored_lycoris.py
```

该脚本把本目录覆盖到当前解释器的 `site-packages/lycoris/`。
`--check` 只报告差异不复制（退出码 1 表示有漂移）。

`tests/test_vendored_lycoris.py` 会校验已安装副本与本目录一致 —— **如果你直接改了
venv 里的 lycoris，这个测试会失败**，提醒你把改动同步回 `vendor/lycoris/` 并提交。

## 相对上游 4.0.0 的 overlay

上游 4.0.0 提供融合 kernel（Triton / TileLang / `torch.compile` / eager 自动降级，
`LYCORIS_KERNEL_BACKEND`）以及 `train_llm_adapter`。本训练器默认
`LYCORIS_KERNEL_BACKEND=auto`（能加速就加速，不行则降回 eager）；
表单可改 `torch` / `triton`。本地扩展算法不走融合核。

**仍禁止往本目录塞新算法。** 新算法走 `vendor/sd-scripts/networks/*_anima.py`。

### 本地扩展算法（仅作用于 Linear 层，Conv/Norm 层自动跳过）

| 算法 | 文件 | 说明 |
|---|---|---|
| GLoKR | `modules/glokr.py` | LoKr + GLoRA 三路径融合，详见同目录 `GLOKR.md`；GUI 已移除，仅存档加载 |
| BoKR | `modules/bokr.py` | LoKr + BoRA 双向权重解耦 |
| BoRA | `modules/bora.py` | 双向范数解耦的 LoRA 变体 |
| GloKrSora | `modules/gsokr.py` | GLoKR 的 SoRA 稀疏化变体（实验性） |
| GLoRA-BOFT | `modules/glora_boft.py` | GLoRA + 蝶形正交变换 |
| CDKA | `modules/cdka.py` | 遗留存档；GUI 走 `networks.cdka_anima` |

### Anima / kohya 补丁

- `modules/functional.py`：`compute_merged_delta`，merged forward 在 fp32 里做加减，避免 bf16 小更新被吃掉。`lokr` / `loha` / `locon` / 上游 `tlora` 的 eager 路径走此函数；`lycoris.modules.lokr.compute_merged_delta` 再导出供 `verify_vendored_lycoris` 检测是否被 pip 覆盖
- `kohya.py`：`create_network` 转发 `extra_algo_kwargs`；`LycorisNetworkKohya` 提供 `set_current_timestep` / `clear_current_timestep`
- `kohya.py` + `wrapper.py`：`exclude_name` 对 class-sweep 子层生效（`*adaln_modulation*` 必须剪掉，否则 Anima LoKr 会训到 AdaLN 调制层）
- `modules/norms.py`：`train_norm` 跳过无仿射权重的归一化层（Anima DiT `elementwise_affine=False`）
- `modules/bokr.py`：第 9 位 `use_tucker` 占位，兼容 kohya 位置传参
- `modules/gsokr.py`：`factor` / `sora_r` / `sora_epsilon` 显式转数值

## 注意

- 请勿对 `lycoris-lora` 执行 `pip install --force-reinstall` / `-I`，否则会覆盖
  本目录的改动；若已执行，重新跑一次同步脚本即可
- 上游许可证随包保留，本目录未改动许可条款
- GUI 的 T-LoRA 仍是 `networks.tlora_anima`，不是 4.0 自带的 `algo=tlora`
