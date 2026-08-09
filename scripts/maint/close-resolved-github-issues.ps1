# Close GitHub issues already fixed on main (v2.5.2+).
# Auth: User-level GITHUB_TOKEN (see doc/local/AGENT_INTERNAL.md), or gh auth login.
# Usage: powershell -ExecutionPolicy Bypass -File scripts/maint/close-resolved-github-issues.ps1

$ErrorActionPreference = "Stop"
$repo = "wochenlong/lora-scripts-next"

$closes = @(
    @{
        Number = 47
        Comment = @"
## 关闭说明

此问题已在 **`main`** 修复，并包含在 **v2.5.2** 变更中（提交 `72bd1e9`）。

### 原因回顾
Windows 上启用 `torch_compile` 时，`torch.compile` 默认走 `inductor` 后端，依赖 **Triton**；Triton 仅支持 Linux，便携包/源码在 Windows 上会直接崩溃。

### 修复内容
- `mikazuki/app/api.py` → `sanitize_config()`：在 Windows 检测到 `torch_compile=true` 时**自动移除**该选项及 `dynamo_backend`，并输出中英文 WARNING，避免用户无感踩坑。

### 你需要做什么
- **整合包用户**：请使用 **v2.5.2 或更新** 的 Release 7z（或 `Update-SD-Trainer.bat` 更新到含该修复的版本后重试）。
- 发布版 7z 通常无 `.git`，若 `git pull` 不可用，请直接下载新版整合包替换。

若更新后仍遇到问题，请附上 `config/autosave/*.toml` 与训练日志新开 Issue，并注明整合包版本号。
"@
    },
    @{
        Number = 46
        Comment = @"
## 关闭说明

此问题已在 **`main`** 修复，并包含在 **v2.5.2** 变更中（提交 `3f05e0f`）。

### 原因回顾
`vendor/sd-scripts/train_network.py` 基类 `assert_extra_args(self, args, train_dataset_group, val_dataset_group)` 已升级为 3 个业务参数，但 `scripts/stable/` 子类仍保留旧的 2 参数签名，导致 SDXL LoRA 开训时报：
`TypeError: SdxlNetworkTrainer.assert_extra_args() takes 3 positional arguments but 4 were given`

### 修复内容
为 `scripts/stable/` 下相关 trainer 的 `assert_extra_args` 增加 `val_dataset_group=None`，与 vendor 基类签名一致（`train_network.py`、`sdxl_train_network.py`、Textual Inversion 等）。

### 你需要做什么
请更新到 **v2.5.2+**（源码 `git pull` 或新版整合包）后重新训练 SDXL LoRA。

若仍有崩溃，请附完整 traceback 与 autosave TOML reopen 或新开 Issue。
"@
    },
    @{
        Number = 45
        Comment = @"
## 关闭说明

此问题已在 **`main`** 修复，并包含在 **v2.5.2** 变更中（提交 `3a4f82d`、`2bb4029`、`5472828` 等）。

### 原因回顾
国内网络下 `git fetch` 直连 GitHub 常被 RST（`Connection was reset`），旧版 `Update-SD-Trainer.bat` 无镜像回退与重试，一次失败即退出。

### 修复内容
`build-scripts/templates/Update-SD-Trainer.bat`（同步到整合包根目录）：
1. **多路 fetch**：直连 → `ghfast.top` → `ghproxy` → `gitmirror` 依次尝试
2. **浅克隆加深**：检测到 shallow 仓库时 `--deepen=50`，减少 fast-forward 失败
3. **子模块容错**：`dataset-tag-editor` 子模块同样支持镜像；目录已存在时跳过重复 clone
4. **失败排障**：全部失败后打印代理/VPN/手动下载 Release 等建议

详见仓库 `CHANGELOG.md` v2.5.2「整合包修复」一节。

### 你需要做什么
- 使用 **v2.5.2+** 整合包内的 `Update-SD-Trainer.bat`，或从 GitHub 拉取最新 `main` 后重新打包。
- 若四轮镜像仍失败，按脚本末尾提示检查网络/代理，或从 [Releases](https://github.com/wochenlong/lora-scripts-next/releases) 手动下载 7z。

感谢反馈，便于国内用户更新。
"@
    }
)

function Close-IssueWithComment {
    param($Number, $Comment, $Headers)
    Write-Host "Commenting #$Number ..."
    $commentBody = @{ body = $Comment.Trim() } | ConvertTo-Json -Depth 5
    $commentUri = "https://api.github.com/repos/$repo/issues/$Number/comments"
    Invoke-RestMethod -Uri $commentUri -Method POST -Headers $Headers -Body $commentBody -ContentType "application/json; charset=utf-8" | Out-Null
    Write-Host "Closing #$Number ..."
    $closeBody = @{ state = "closed"; state_reason = "completed" } | ConvertTo-Json
    $issueUri = "https://api.github.com/repos/$repo/issues/$Number"
    $result = Invoke-RestMethod -Uri $issueUri -Method PATCH -Headers $Headers -Body $closeBody -ContentType "application/json; charset=utf-8"
    Write-Host "  OK: $($result.html_url)"
}

$token = [Environment]::GetEnvironmentVariable("GITHUB_TOKEN", "User")
if (-not $token) { $token = $env:GITHUB_TOKEN }

if ($token) {
    $headers = @{
        "Authorization" = "Bearer $token"
        "Accept"        = "application/vnd.github+json"
        "X-GitHub-Api-Version" = "2022-11-28"
    }
    foreach ($item in $closes) {
        $state = Invoke-RestMethod -Uri "https://api.github.com/repos/$repo/issues/$($item.Number)" -Headers $headers
        if ($state.state -eq "closed") {
            Write-Host "#$($item.Number) already closed, skip."
            continue
        }
        Close-IssueWithComment -Number $item.Number -Comment $item.Comment -Headers $headers
    }
    Write-Host "Done (REST API)."
    exit 0
}

# Fallback: gh CLI
$gh = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\GitHub.cli_Microsoft.Winget.Source_8wekyb3d8bbwe\bin\gh.exe"
if (-not (Test-Path $gh)) {
    $cmd = Get-Command gh -ErrorAction SilentlyContinue
    if ($cmd) { $gh = $cmd.Source }
}
if (-not $gh -or -not (Test-Path $gh)) {
    Write-Error "未配置 GITHUB_TOKEN，且未找到 gh。见 doc/local/AGENT_INTERNAL.md"
}
& $gh auth status 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Error "请先: gh auth login" }

foreach ($item in $closes) {
    $n = $item.Number
    $f = New-TemporaryFile
    try {
        Set-Content -Path $f -Value $item.Comment.Trim() -Encoding UTF8
        & $gh issue comment $n --repo $repo --body-file $f.FullName
        & $gh issue close $n --repo $repo --reason completed
        Write-Host "  OK: https://github.com/$repo/issues/$n"
    } finally {
        Remove-Item $f -Force -ErrorAction SilentlyContinue
    }
}
Write-Host "Done (gh CLI)."
