# 遗留脚本（上游 / 旧版）

从仓库根目录迁入，**日常请用 `run_gui.bat` / `gui.py`**。此处脚本仅作参考或高级 CLI 用途。

| 路径 | 说明 |
|------|------|
| `cli/tagger.ps1` 等 | 秋叶系辅助脚本（打标、合并、 interrogation） |
| `cli/tensorboard.ps1` | 单独启动 TensorBoard（GUI 内已集成入口） |
| `notebooks/*.ipynb` | 上游示例笔记本，非 WebUI 主流程 |

秋叶 CLI 训练脚本已迁至 `scripts/cli/`（根目录保留转发器）。

**不可迁入 `legacy/`：**

- `start_autodl.sh` — 云镜像开机绑定（根目录）
- `run_gui.bat` / `scripts/portable/` — 整合包入口

云运维脚本在 `scripts/autodl/`。见 [docs/repo-layout.md](../docs/repo-layout.md)。
