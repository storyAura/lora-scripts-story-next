# 全项目待办优先级总表

> 维护者用 backlog 视图；**不上 README 主页传送门**。  
> **AI 建议**列供初稿参考；**团队优先级**由 @wochenlong 拍板，@niangao2331 分诊时可建议标签。  
> **只有 GitHub Issue 进入解决流程**（见 [README.md](README.md)）；本表跟踪范围与优先级，落地须有关联 Issue 或新建 Issue。

**最后更新**：2026-05-26（open Issue 列表来自 GitHub API）

**图例**

| 标记 | 含义 |
|------|------|
| **P0 / P1 / P2** | 建议优先级（见 [priority.md](priority.md) 草案） |
| **已修** | main 或近期 Release 已含修复，发版前需回归验证 |
| **流程** | 讨论/文档/分工，不直接写功能代码 |
| **草案** | 尚未正式开 GitHub Issue，或仅有 Discussion / 本地草稿 |

**速览**：[发版 TOP](#发版-top下一版-7z--release) · [排期 TOP](#排期-top接下来-12-个迭代) · [总表](#总表)

---

## 使用方式

1. 团队在 **团队优先级** 列填写：`P0` / `P1` / `P2` / `暂缓` / `不做` / `已修`。
2. 定稿后由年糕打 Issue 标签（`priority:p0` 等，标签待建）。
3. 每发一版整合包：核对 **REL-*** 与 **PKG-*** 中「已修」项，更新本表状态。

---

## 总表

| ID | 事项 | 负责人（建议） | AI 建议 | 团队优先级 | 备注 / 链接 |
|----|------|----------------|---------|------------|-------------|
| **端口与启动** |
| P-01 | 端口治理 P0 小项：端口互不抢、监控连对 API、主页 `/train-monitor`（Discussion **P0-1～3**） | @MikumikuDAIFans | P0 | **已完** | [#53](https://github.com/wochenlong/lora-scripts-next/discussions/53) |
| P-04 | 启动日志打印「访问地址清单」（**P0-4**） | @MikumikuDAIFans | P0 | | 排期 1a |
| P-05 | 浏览器自动打开路径入口，非 `:6008`（**P0-5**） | @MikumikuDAIFans | P0 | | 排期 1a |
| P-02 | 最小 `.runtime/services.json`（**P0-6**，注册表基石） | @MikumikuDAIFans | P0 | | 仍用 6006/6008/28001，不强制 281xx |
| P-07 | 监控从注册表读 `api` 地址（**P0-7**） | @MikumikuDAIFans | P0 | | P0-7 前监控仍靠 env/探测 |
| P-08 | 公共入口 28000 fallback 28001–28020（**P0-8**） | @MikumikuDAIFans | P0 | | 仅未显式指定端口时 |
| P-03 | AutoDL 显式端口占用则失败（**P0-9**） | @MikumikuDAIFans | P0 | | 排期 1c，与 P-08 同属「外部入口稳定」 |
| P-10 | Tag Editor / proxy 去除硬编码 28001（**P0-10**） | @MikumikuDAIFans | **P1** | | #53：内化端口，/registry+AutoDL 后再做 |
| P-11 | `/monitor/` 与 `/train-monitor` 别名（**P0-11**） | @MikumikuDAIFans | **P1** | | #53：同上 |
| P-12 | `/tensorboard/`、`/tagger/` 全路径反代（**P0-12**） | @MikumikuDAIFans | **P1** | | #53：同上 |
| P-13 | 内部端口迁 281xx 池（**P0-13**） | @MikumikuDAIFans | **P1** | | #53：同上 |
| P-06 | 端口规范全文：健康检查、#50 后续、插件等 | @MikumikuDAIFans | P2 | | 草案非定稿 |
| PORT-14 | PRT-007/008 验收；单测 + 手工 checklist（**P0-14**） | @MikumikuDAIFans | P0 | | 贯穿各阶段 |
| PORT-07 | `rg` 扫描用户入口裸端口（Discussion 事项 **#7**） | @wochenlong | P1 临时 | | #53：**P0 全部完成前的发布门禁**，非终态 |
| D-53 | 端口方案讨论定稿 | @wochenlong | 流程 | | [#53](https://github.com/wochenlong/lora-scripts-next/discussions/53) |
| [#50](https://github.com/wochenlong/lora-scripts-next/issues/50) | 系统性解决多服务端口冲突、404、无法连接 | @MikumikuDAIFans | P1 | | 拆为 P-01～P-06 |
| [#39](https://github.com/wochenlong/lora-scripts-next/issues/39) | 内嵌服务取代 TB/Gradio、单端口架构（大重构） | 未定 | P2 | | 与 P-06 同方向、范围更大 |
| **训练引擎（Anima / sd-scripts）** |
| [#18](https://github.com/wochenlong/lora-scripts-next/issues/18) | 混合精度疑似未生效（源码 + 整合包） | @wochenlong / @ageless-h | P0～P1 | | Issue 标题含 P0；需复现确认 |
| [#17](https://github.com/wochenlong/lora-scripts-next/issues/17) | TLoRA 训练 ModuleNotFoundError | @MikumikuDAIFans / @ageless-h | P1 | | |
| [#51](https://github.com/wochenlong/lora-scripts-next/issues/51) | Anima + Automagic/CAME 等 loss=nan | @wochenlong | P1 | | 已有缓解；环境敏感 |
| [#31](https://github.com/wochenlong/lora-scripts-next/issues/31) | 导入 sd-scripts.toml 后参数预览空白 | @wochenlong | P1 | | |
| [#29](https://github.com/wochenlong/lora-scripts-next/issues/29) | 切换 lora_type 后参数表单空白 | @wochenlong | P1 | | |
| [#25](https://github.com/wochenlong/lora-scripts-next/issues/25) | LoKr full_matrix=true 首 epoch 崩坏 | @MikumikuDAIFans | P1 | | |
| [#30](https://github.com/wochenlong/lora-scripts-next/issues/30) | Anima 训练支持 DoRA | @ageless-h | P2 | | |
| [#43](https://github.com/wochenlong/lora-scripts-next/issues/43) | 跨模型配置导入应校验类型 | @wochenlong | P1 | | |
| ENG-01 | Accelerate 续训缺 step 元数据 | @wochenlong | 已修? | | 核对 #52 是否已关闭；main 有 fallback |
| ENG-02 | Windows torch_compile / Triton 踩坑 | @wochenlong | P1 | | 整合包多已自动剔除 compile |
| **打标（新）** |
| [#40](https://github.com/wochenlong/lora-scripts-next/issues/40) | 数据集打标 backend（API、引擎扩展、测试） | @niangao2331 | P1 | | 不写 frontend/dist；API 见 `docs/api/dataset-tagging.md` |
| TAG-01 | #40 与 #41 边界、合并顺序 | @wochenlong + 年糕 | 流程 | | |
| **标签编辑 / 数据集工作区** |
| #41 | 数据集工作区（scan、caption 编辑、合一 Tab） | @wochenlong / 年糕 | P2 | | **草案**：待开 GH Issue |
| #41-draft | 在 GitHub 正式创建 Issue #41 并挂 Depends on #40 | 年糕 / @wochenlong | P1 | | 流程项 |
| **Flash Attention 2** |
| [#26](https://github.com/wochenlong/lora-scripts-next/issues/26) | 全路径支持 FA2（源码 / 整合包 / Docker） | @wochenlong | P2 | | 整合包当前不支持 FA2，见 `docs/flash-attention.md` |
| FA-01 | 源码安装 FA2 / triton / wheel 问题 | @wochenlong | P2 | | #26 子项 |
| FA-02 | 文档澄清：整合包勿安装 flash-attn | @wochenlong | P1 | | 低成本 |
| **整合包：打包 / 测试** |
| REL-01 | 发版 smoke checklist（QA） | @wochenlong | P1 高 | | 建议开 Process Issue |
| REL-02 | 打包规范 + 7z 抽查无私密 autosave | @wochenlong | P1 | | `build/整合包打包规范.md`、`risk-memo.md` |
| REL-03 | 热修后 main 与 7z 同步策略 | @wochenlong | P1 | | |
| REL-04 | Git 版整合包 Update 脚本验证 | @wochenlong | P2 | | `docs/portable-packaging-git-update.md` |
| **整合包：已知 / 历史问题** |
| PKG-01 | config/autosave 不存在 → /api/run 500 | @wochenlong | 已修 | | 发版回归 REL-01 |
| PKG-02 | 主页品牌图缺失 | @wochenlong | 已修 | | v2.5.2+ |
| PKG-03 | 主页监控链进 TensorBoard | @MikumikuDAIFans | 部分已修 | | `/train-monitor`；PRT 纳入 REL-01 |
| PKG-04 | 监控误报 6006 GUI API | @wochenlong | 已缓解 | | gui_warning |
| [#42](https://github.com/wochenlong/lora-scripts-next/issues/42) | 旧整合包 Anima 模型类型误判 | @wochenlong | P1 | | 旧包用户 |
| [#24](https://github.com/wochenlong/lora-scripts-next/issues/24) | install.ps1 内存耗尽 | @wochenlong | P1～P2 | | |
| **训练监控 / 前端** |
| [#28](https://github.com/wochenlong/lora-scripts-next/issues/28) | 训练监控 Loss 图 EMA 平滑滑块 | @wochenlong | P2 | | 体验增强 |
| UI-01 | 功能优先；UI 美化后期；SupermarKleet 前端预备役 | @SupermarKleet | P2 | | |
| **其它 open Issue** |
| [#21](https://github.com/wochenlong/lora-scripts-next/issues/21) | 默认 output_dir 不保存模型 | @wochenlong | P1 | | 误导新用户 |
| **团队 / 流程** |
| T-01 | Issue 分诊 | @niangao2331 | P1 | | |
| T-02 | PR / Issue 模板 | 年糕 / @wochenlong | P1 | | |
| T-03 | P0/P1/P2 标准定稿 | @wochenlong + 年糕 | P1 | | [priority.md](priority.md) |
| T-04 | 贡献者入门文档 | @wochenlong | P2 | | |
| T-05 | `docs/team/` 与 CONTRIBUTORS 同步 | @wochenlong | P2 | | |
| SEC-01 | 安全 / 发版门（risk-memo） | @wochenlong | P1 持续 | | [risk-memo.md](risk-memo.md) |
| DOC-01 | 用户文档总责 + 各领域 DRI 写本模块 doc | 各人 | P1 持续 | | |

---

## 两类 TOP（勿混为一谈）

| 视角 | 回答的问题 | 端口治理 |
|------|------------|----------|
| **[发版 TOP](#发版-top下一版-7z--release)** | 这一版 7z 能不能发？ | **P0 必须验**（P-01、PKG-03、REL-01） |
| **[排期 TOP](#排期-top接下来-12-个迭代)** | 接下来先做什么？ | **第一阶段主线**（P-02～P-05、#50 第一期、开发标准） |

**团队共识**（@MikumikuDAIFans 等）：**端口治理 + 开发标准** 紧急且重要；**端口未内化前不宜贸然做 #41 / 合一 UI / 废 Gradio 主路径**。  
**#40 打标 backend** 可在门禁进行中并行，但须守 G-4 硬规则；**不得**跳过注册表与外部入口稳定阶段。

```text
发版 TOP  = 这一包能不能发（端口 P0-1～3 已修 + PKG + QA）
排期 TOP  = 注册表(P0-6/7) → 外部入口稳定(P0-8/9) → 再 #40 → 端口内化(P0-10～13 作 P1) → #41
```

### Discussion #53 共识（[@MikumikuDAIFans](https://github.com/MikumikuDAIFans)，2026-05-26）

来源：[Discussion #53 回复](https://github.com/wochenlong/lora-scripts-next/discussions/53#discussioncomment-14783447)（以帖内实际链接为准）。

**待确定事项**

| 编号 | 内容 | 表态 |
|------|------|------|
| 1～6 | 单公共入口 + 路径路由终态；过渡期保留旧路径；最小注册表（可不立刻 281xx）；AutoDL 严格端口等 | **全部认同** |
| 7 | 发布门禁 `rg` 扫描裸端口 | **仅作 P0 全部完成前的临时策略**，非长期终态 |

**执行顺序（@MikumikuDAIFans 归纳）**

> 注册表先行 → 唯一外部入口稳定 → 再逐步内化子服务端口。

| 阶段 | Discussion ID | 本表 ID | 说明 |
|------|---------------|---------|------|
| 已完成 | P0-1～3 | P-01 | 端口不串台、路径入口、监控 API（可再接注册表） |
| 1a 可观测 | P0-4、P0-5 | P-04、P-05 | 启动地址清单、浏览器走路径 |
| 1b 注册表 | **P0-6、P0-7** | **P-02、P-07** | **基石**；P0-7 前仍用外部端口 + env/探测 |
| 1c 外部入口 | **P0-8、P0-9** | **P-08、P-03** | 做完后再推进「内化」 |
| 1d 内化端口 | P0-10～13 | P-10～P-13、P-11、P-12 | **建议整体降为 P1**，在注册表 + AutoDL 机制成功之后逐个做 |
| 验收 | P0-14 | PORT-14 | 贯穿各阶段 |

**与 #41 的关系**：P0-10～13 未清前，不把 #41 / Gradio 主路径下线当作 P0；与 [排期 TOP](#排期-top接下来-12-个迭代) 第三阶段一致。

---

---

## 发版 TOP（下一版 7z / Release）

**目标**：用户立刻可训练、入口不错、包内无事故。  
**发版 / QA**：@wochenlong。**端口 P0 实现**：@MikumikuDAIFans，由你 smoke 验收。

| 序 | 项 | 说明 |
|----|-----|------|
| 1 | **P-01** | 各服务默认端口互不抢；监控连主 WebUI API；主页 `/train-monitor` |
| 2 | **PKG-03 + REL-01** | smoke：起 GUI → 开训 → 监控链 → TensorBoard → 打标能开；监控不得打开 TB（PRT-007/008） |
| 3 | **PKG-01** | `config/autosave` 已修且本包已含 |
| 4 | **PKG-02** | 主页 / 引导品牌图已进包 |
| 5 | **REL-02** | 打包前 [risk-memo](risk-memo.md)：`7z` 无私密 autosave、无 `doc/` `data/` |
| 6 | **REL-03** | 热修已在 main 则重打 7z；Release 说明写清最低版本 |
| 7 | **发版说明** | 列出本版端口 / 监控 / 开训相关修复；已知未修项不写糊 |

**发版 TOP 明确不含**（避免为发版拖期）：P0-6～14 全文、#50 第一期、#40 大里程碑、#41、281xx（P-13）、#39。发版后可并行启动 **P0-4/5** 与 **P0-6/7**（排期 1a/1b）。

---

## 排期 TOP（接下来 1～2 个迭代）

> 端口子项顺序以 [#53 @MikumikuDAIFans 回复](https://github.com/wochenlong/lora-scripts-next/discussions/53) 为准。

### 第一阶段：端口架构门禁

#### 1a 可观测性（P0-4、P0-5）

| 序 | 项 | 负责人 |
|----|-----|--------|
| 1 | **P-04** 启动日志「访问地址清单」 | @MikumikuDAIFans |
| 2 | **P-05** 浏览器打开主站 + `/train-monitor` | @MikumikuDAIFans |

#### 1b 注册表基石（P0-6、P0-7）— 优先于端口内化

| 序 | 项 | 负责人 |
|----|-----|--------|
| 3 | **P-02** 最小 `.runtime/services.json` | @MikumikuDAIFans |
| 4 | **P-07** 监控优先从注册表读 `api`（此前 env/探测） | @MikumikuDAIFans |

#### 1c 唯一外部入口稳定（P0-8、P0-9）

| 序 | 项 | 负责人 |
|----|-----|--------|
| 5 | **P-08** 公共入口 fallback 28001–28020 | @MikumikuDAIFans |
| 6 | **P-03** AutoDL 显式端口占用则失败 | @MikumikuDAIFans |

#### 1d 流程与 #50 第一期

| 序 | 项 | 负责人 |
|----|-----|--------|
| 7 | **G-3：#50 第一期**（不等 P-13 / 全量反代） | @MikumikuDAIFans |
| 8 | **G-4：#53 硬规则** 写入 `docs/team/` 或对应 Issue | @wochenlong |
| 9 | **PORT-14** PRT-007/008 验收 | @MikumikuDAIFans |
| 10 | **T-03 + T-02**、**T-01** | 你 + 年糕 |

**PORT-07（`rg` 门禁）**：P0-6～9 与 P0-1～5 未完成前，作发版临时检查；**不作为 P0-14 替代**。

**G-4 硬规则（示例，以 #53 定稿为准）**

- 用户可见入口只用路径（如 `/train-monitor`），禁止 `127.0.0.1:6006|6008|28001` 作主页链接。
- 新 PR 关联 Issue；写明如何验证。
- 子服务地址从 env / services.json 读取，禁止在新代码写死 WebSocket 到 `28001`。
- 动 `gui.py`、契约路径、打包脚本须 @wochenlong review。

### 第二阶段：守规矩的功能线（与 1b/1c 可重叠）

| 序 | 项 | 负责人 | 条件 |
|----|-----|--------|------|
| 11 | **[#40](https://github.com/wochenlong/lora-scripts-next/issues/40) M1** | @niangao2331 | 守 G-4；**不等** P0-10～13 |
| 12 | **REL-01 checklist** | @wochenlong | |
| 13 | **[#18](https://github.com/wochenlong/lora-scripts-next/issues/18)** 等训练 Issue | 你 / @ageless-h | |
| 14 | **训练引擎 P1**（#17 / #51 / #25 / #31 / #29） | @MikumikuDAIFans 等 | 按 Issue 穿插 |

### 第三阶段：端口内化（#53 定为 **P1**：P0-10～13）

| 序 | 项 | 说明 |
|----|-----|------|
| 15 | **P-10** Tag Editor / proxy 去硬编码 28001 | |
| 16 | **P-11** `/monitor/` 别名 | |
| 17 | **P-12** `/tensorboard/`、`/tagger/` 反代 | |
| 18 | **P-13** 281xx 内部端口池 | |
| 19 | **P-06 / #50 后续** | 健康检查等 |

### 第四阶段：#41 与大型 UI

| 序 | 项 | 说明 |
|----|-----|------|
| 20 | **#41** 数据集合一 UI | 依赖 #40；替代 Gradio **主路径** |
| 21 | **#40 后续** + 打标 UI 接新 API | |

### 排期后置（避免埋坑）

| 项 |
|----|
| [#39](https://github.com/wochenlong/lora-scripts-next/issues/39) 内嵌服务大重构 |
| Gradio `:28001` **主路径**下线（随 #41 再 deprecate） |
| [#26](https://github.com/wochenlong/lora-scripts-next/issues/26) 整合包 FA2、[#30](https://github.com/wochenlong/lora-scripts-next/issues/30) DoRA、[#28](https://github.com/wochenlong/lora-scripts-next/issues/28) EMA |
| #41 之前侧栏大改 / 大规模 `frontend/dist` 重构 |
| UI 美化、@SupermarKleet 前端预备役（功能优先） |

### 发版 vs 排期对照

| | 端口治理 | [#40](https://github.com/wochenlong/lora-scripts-next/issues/40) 打标 | #41 标签编辑 UI |
|--|----------|----------|-----------------|
| **发版 TOP** | P0 必须验 | 不挡发版（可不进此包） | **不做** |
| **排期 TOP** | **1b/1c 注册表+外部入口（P0）** | 第二阶段（#40，守 G-4） | **第四阶段（#41）**；内化端口为 **第三阶段 P1** |

---

## 负责人负荷（参考）

| 成员 | 主负荷（AI 归纳） |
|------|------------------|
| @wochenlong | 发版 TOP、REL-*、SEC、文档总责、G-4 拍板、#18/#31/#29、发版 QA、review |
| @MikumikuDAIFans | 端口：**P0-4～9**（注册表→外部入口）→ **P1：P0-10～13**；#50 第一期；训练引擎 |
| @niangao2331 | T-01、T-02；#40 M1（守 G-4）；#41 门禁后 |
| @ageless-h | 恢复后 #18、整合包 / sd-scripts（1～2 周内不排新活） |
| @SupermarKleet | UI-01 后期；当前不挡发版 |

---

## 修订记录

| 日期 | 说明 |
|------|------|
| 2026-05-26 | 初版：端口、训练引擎、打标、#41 草案、FA2、整合包、open Issue、团队流程 |
| 2026-05-26 | 增补：发版救火 vs 架构门禁；端口治理置顶；#41 后置至门禁后 |
| 2026-05-26 | 拆分为独立「发版 TOP」「排期 TOP」三节表 + 对照表 |
| 2026-05-26 | 纳入 #53 @MikumikuDAIFans：P0-6/7 基石、P0-8/9 稳定、P0-10～13→P1、事项 7 临时 rg |
