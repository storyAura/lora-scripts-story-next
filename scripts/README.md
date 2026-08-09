# scripts/

仓库正式脚本。个人一次性脚本请放 gitignored 的 `script/`（见 `docs/repo-layout.md`）。

## 目录

| 路径 | 用途 |
|------|------|
| `cli/` | 命令行训练与 Anima Fast 安装 |
| `autodl/` | 云 GPU 运维（根目录 shim 转发到此） |
| `portable/` | 整合包启动与更新 |
| `dev/` / `stable/` | 训练脚本树（勿随意删除） |
| `maint/` | 维护者工具（发版、素材、Issue 批处理） |
| `patch-*.py` | 修改已编译的 `frontend/dist/`（见下） |

## 前端 patch（可重复用）

按需改 `frontend/dist/` 时使用，模式见 `frontend/VENDOR.md`、`AGENTS.md`。

常用：

- `patch-sidebar-nav.py` / `patch-home-portals.py` / `patch-nav-copy.py`
- `patch-anima-fast-entry.py` / `patch-anima-finetune-ui.py`
- `patch-ui-brand-version.py` / `patch-spa-frontend-cache.py`
- `spa_asset_cache.py` / `bump_spa_asset_cache_key.py`

**不要重跑** `patch-home-changelog.py`：它会用脚本内嵌的旧 changelog 覆盖较新页面。

## maint/

| 脚本 | 说明 |
|------|------|
| `process_next_trainer_assets.py` | 从 `doc/local/` 素材生成 README 封面 / favicon |
| `extract-sidebar.py` | 调试侧栏结构 |
| `close-resolved-github-issues.ps1` | 批量关闭已修复 Issue |

## 其它

- `benchmark_anima_adapters.py` / `fsdp2_frozen_base_smoke.py`：开发基准 / smoke
- `sync_vendored_lycoris.py`、`prefetch_*.py`：依赖与缓存维护
