#!/usr/bin/env python3
"""Rename UI Settings → Trainer Settings and expand the settings schema."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "frontend" / "dist"

SCHEMA_CODE = """Schema.intersect([
    Schema.object({
        disk_preflight_enabled: Schema.boolean().default(true).description("开训前检查磁盘空间。按本次任务估算输出权重与磁盘缓存，空间不够时拒绝开训，避免写到一半失败。估算仍偏大时可关闭"),
        tensorboard_url: Schema.string().role('folder').description("tensorboard 地址")
    }).description("训练器设置"),
    Schema.object({
        huggingface_token: Schema.string().description("Hugging Face Token。用于下载私有模型，以及把训练产物上传到 Hub。保存在本机训练器设置中"),
        huggingface_repo_id: Schema.string().description("上传到 Hugging Face Hub 的仓库 ID，例如 username/repo。留空则不自动上传"),
        huggingface_path_in_repo: Schema.string().description("仓库内保存路径（可选）"),
        huggingface_repo_visibility: Schema.union(["public", "private"]).default("private").description("Hub 仓库可见性"),
        async_upload: Schema.boolean().default(false).description("异步上传到 Hugging Face Hub，不阻塞保存"),
        save_state_to_huggingface: Schema.boolean().default(false).description("把训练状态（save_state）一并上传到 Hugging Face Hub")
    }).description("Hugging Face / Token"),
]);
"""


def patch_sidebar_sources() -> None:
    nav = ROOT / "scripts" / "patch-sidebar-nav.py"
    text = nav.read_text(encoding="utf-8")
    text = text.replace(
        '{"text":"UI 设置","link":"/other/settings.md"}',
        '{"text":"训练器设置","link":"/other/settings.md"}',
    )
    text = text.replace(
        'item("/other/settings.md", "UI 设置", "UI 设置"',
        'item("/other/settings.md", "训练器设置", "训练器设置"',
    )
    nav.write_text(text, encoding="utf-8")


def patch_settings_data() -> None:
    data_js = DIST / "assets" / "settings.html.06993f96.js"
    payload = {
        "key": "v-72e1da3e",
        "path": "/other/settings.html",
        "title": "训练器设置",
        "lang": "en-US",
        "frontmatter": {
            "type": "settings",
            "code": SCHEMA_CODE,
        },
        "excerpt": "",
        "headers": [],
        "filePathRelative": "other/settings.md",
    }
    encoded = json.dumps(payload, ensure_ascii=False)
    encoded = encoded.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    data_js.write_text(
        f"const e=JSON.parse(`{encoded}`);export{{e as data}};\n",
        encoding="utf-8",
    )


def patch_settings_vue() -> None:
    vue_js = DIST / "assets" / "settings.html.07aaabcc.js"
    text = vue_js.read_text(encoding="utf-8")
    text = text.replace("\\u8BAD\\u7EC3 UI \\u8BBE\\u7F6E", "\\u8BAD\\u7EC3\\u5668\\u8BBE\\u7F6E")
    text = text.replace("\\u8BAD\\u7EC3-ui-\\u8BBE\\u7F6E", "\\u8BAD\\u7EC3\\u5668\\u8BBE\\u7F6E")
    text = text.replace(
        "\\u4E0D\\u61C2\\u7684\\u4E0D\\u8981\\u78B0\\u8FD9\\u4E2A",
        "\\u5F00\\u5173\\u4F1A\\u5199\\u5165\\u672C\\u673A config/trainer_settings.json\\uFF0C\\u5F71\\u54CD\\u6240\\u6709\\u8BAD\\u7EC3\\u4EFB\\u52A1",
    )
    text = text.replace("训练 UI 设置", "训练器设置")
    text = text.replace("训练-ui-设置", "训练器设置")
    text = text.replace(
        "不懂的不要碰这个",
        "开关会写入本机 config/trainer_settings.json，影响所有训练任务",
    )
    vue_js.write_text(text, encoding="utf-8")


def patch_settings_html() -> None:
    html = DIST / "other" / "settings.html"
    text = html.read_text(encoding="utf-8")
    text = text.replace("训练 UI 设置", "训练器设置")
    text = text.replace("训练-ui-设置", "训练器设置")
    text = text.replace(
        "不懂的不要碰这个",
        "开关会写入本机 config/trainer_settings.json，影响所有训练任务",
    )
    html.write_text(text, encoding="utf-8")


def main() -> None:
    patch_sidebar_sources()
    runpy.run_path(str(ROOT / "scripts" / "patch-sidebar-nav.py"), run_name="__main__")
    patch_settings_data()
    patch_settings_vue()
    patch_settings_html()
    print("patched trainer settings page + sidebar")


if __name__ == "__main__":
    main()
