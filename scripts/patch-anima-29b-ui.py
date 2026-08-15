#!/usr/bin/env python3
"""Add Anima 2.9B training page under the Anima LoRA sidebar group."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from spa_asset_cache import SPA_ASSET_CACHE_KEY

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "frontend" / "dist"
ASSETS = DIST / "assets"
APP_JS = ASSETS / "app.547295de.js"
QUEUE_JS = ASSETS / "sd-trainer-queue.js"

ROUTE_KEY = "v-anima-29b"
FT_ROUTE_KEY = "v-anima-29b-ft"
DATA_JS = ASSETS / "anima-2.9b.html.data.js"
PAGE_JS = ASSETS / "anima-2.9b.html.page.js"
HTML_PATH = DIST / "lora" / "anima-2.9b.html"
FT_DATA_JS = ASSETS / "anima-2.9b-finetune.html.data.js"
FT_PAGE_JS = ASSETS / "anima-2.9b-finetune.html.page.js"
FT_HTML_PATH = DIST / "lora" / "anima-2.9b-finetune.html"

PAGE_TITLE = "Anima2.9B"
TAGLINE = "40 层 Anima 2.9B LoRA，可选只训 12 个插入层"
INTRO = "加载 40 层 2.9B 检查点。打开「只训插入层」可复现 preview-v1：冻结继承层，只训练交错插入的 12 块。"

FT_PAGE_TITLE = "Anima2.9B Finetune"
FT_TAGLINE = "40 层 Anima 2.9B 全量微调，可选只训 12 个插入层"
FT_INTRO = "更新完整 2.9B DiT 权重。打开「只训插入层」只训练交错插入的 12 块；关闭则训练全部已加载层。"


def _write_page_js(path: Path, *, h1_id: str, title: str, tagline: str, intro: str, file_label: str) -> None:
    path.write_text(
        'import{_ as s,o as t,c as o,a as e,b as a}'
        f'from"./app.547295de.js?v={SPA_ASSET_CACHE_KEY}";'
        "const _={},"
        f'c=e("h1",{{id:{json.dumps(h1_id)},tabindex:"-1"}},['
        f'e("a",{{class:"header-anchor",href:"#{h1_id}","aria-hidden":"true"}},"#"),'
        f'a(" {title}")],-1),'
        f'n=e("p",null,{json.dumps(tagline, ensure_ascii=False)},-1),'
        f'd=e("p",null,{json.dumps(intro, ensure_ascii=False)},-1),'
        "l=[c,n,d];"
        'function i(h,u){return t(),o("div",null,l)}'
        f'var p=s(_,[["render",i],["__file",{json.dumps(file_label)}]]);export{{p as default}};',
        encoding="utf-8",
    )


def write_page_assets() -> None:
    data = {
        "key": ROUTE_KEY,
        "path": "/lora/anima-2.9b.html",
        "title": PAGE_TITLE,
        "lang": "en-US",
        "frontmatter": {"example": True, "trainType": "anima-2.9b"},
        "excerpt": "",
        "headers": [],
        "filePathRelative": "lora/anima-2.9b.md",
    }
    DATA_JS.write_text(
        f"const e=JSON.parse({json.dumps(json.dumps(data, ensure_ascii=False))});export{{e as data}};",
        encoding="utf-8",
    )
    _write_page_js(
        PAGE_JS,
        h1_id="anima-29b",
        title=PAGE_TITLE,
        tagline=TAGLINE,
        intro=INTRO,
        file_label="anima-2.9b.html.vue",
    )

    src = DIST / "lora" / "sd3.html"
    html = src.read_text(encoding="utf-8")
    html = html.replace("/lora/sd3.html", "/lora/anima-2.9b.html")
    html = html.replace("sd3.html.1a4bf31e.js", "anima-2.9b.html.page.js")
    html = html.replace("sd3.html.eaeb05e1.js", "anima-2.9b.html.data.js")
    html = re.sub(
        r"<title>[^<]*</title>",
        f"<title>{PAGE_TITLE} | Next Story Trainer</title>",
        html,
        count=1,
    )
    html = re.sub(
        r"<h1[^>]*>.*?</h1>",
        f'<h1 id="anima-29b" tabindex="-1"><a class="header-anchor" href="#anima-29b" aria-hidden="true">#</a> {PAGE_TITLE}</h1>',
        html,
        count=1,
        flags=re.DOTALL,
    )
    html = re.sub(
        r"<p>Anima DiT 模型 LoRA.*?</p>",
        f"<p>{TAGLINE}</p>",
        html,
        count=1,
    )
    html = re.sub(
        r"<p>Anima DiT 训练入口.*?</p>",
        f"<p>{INTRO}</p>",
        html,
        count=1,
    )
    HTML_PATH.write_text(html, encoding="utf-8")
    print("wrote lora/anima-2.9b.html + assets")


def write_finetune_page_assets() -> None:
    data = {
        "key": FT_ROUTE_KEY,
        "path": "/lora/anima-2.9b-finetune.html",
        "title": FT_PAGE_TITLE,
        "lang": "en-US",
        "frontmatter": {"example": True, "trainType": "anima-2.9b-finetune"},
        "excerpt": "",
        "headers": [],
        "filePathRelative": "lora/anima-2.9b-finetune.md",
    }
    FT_DATA_JS.write_text(
        f"const e=JSON.parse({json.dumps(json.dumps(data, ensure_ascii=False))});export{{e as data}};",
        encoding="utf-8",
    )
    _write_page_js(
        FT_PAGE_JS,
        h1_id="anima-29b-finetune",
        title=FT_PAGE_TITLE,
        tagline=FT_TAGLINE,
        intro=FT_INTRO,
        file_label="anima-2.9b-finetune.html.vue",
    )

    src = DIST / "lora" / "anima-finetune.html"
    html = src.read_text(encoding="utf-8")
    html = html.replace("anima-finetune.html.1a4bf32e.js", "anima-2.9b-finetune.html.page.js")
    html = html.replace("anima-finetune.html.eaeb05f2.js", "anima-2.9b-finetune.html.data.js")
    html = html.replace("/lora/anima-finetune.html", "/lora/anima-2.9b-finetune.html")
    html = re.sub(
        r"<title>[^<]*</title>",
        f"<title>{FT_PAGE_TITLE} | Next Story Trainer</title>",
        html,
        count=1,
    )
    html = re.sub(
        r"<h1[^>]*>.*?</h1>",
        f'<h1 id="anima-29b-finetune" tabindex="-1"><a class="header-anchor" href="#anima-29b-finetune" aria-hidden="true">#</a> {FT_PAGE_TITLE}</h1>',
        html,
        count=1,
        flags=re.DOTALL,
    )
    html = re.sub(
        r'<p class="sd-anima-finetune-tagline">.*?</p>',
        "",
        html,
        count=1,
    )
    html = re.sub(
        r"<p>Anima DiT 全量微调.*?</p>",
        f"<p>{FT_TAGLINE}</p>",
        html,
        count=1,
    )
    html = re.sub(
        r"<p>更新完整 DiT 权重.*?</p>",
        f"<p>{FT_INTRO}</p>",
        html,
        count=1,
    )
    FT_HTML_PATH.write_text(html, encoding="utf-8")
    print("wrote lora/anima-2.9b-finetune.html + assets")


_BROKEN_PAGE_IMPORT = re.compile(
    r'"v-anima-fast":Jt\(\(\)=>wt\(\(\)=>import\("(\./anima-fast\.html\.page\.js(?:\?v=[^"]*)?)"\),'
    r'"v-anima-29b":Jt\(\(\)=>wt\(\(\)=>import\("(\./anima-2\.9b\.html\.page\.js(?:\?v=[^"]*)?)"\),\[\]\)\),'
    r"\[\]\)\),"
)
_BROKEN_DATA_IMPORT = re.compile(
    r'"v-anima-fast":\(\)=>wt\(\(\)=>import\("(\./anima-fast\.html\.data\.js(?:\?v=[^"]*)?)"\),'
    r'"v-anima-29b":\(\)=>wt\(\(\)=>import\("(\./anima-2\.9b\.html\.data\.js(?:\?v=[^"]*)?)"\),\[\]\)'
    r"\.then\(\(\{data:e\}\)=>e\),\[\]\)\.then\(\(\{data:e\}\)=>e\),?"
)
_FAST_PAGE_IMPORT = re.compile(
    r'"v-anima-fast":Jt\(\(\)=>wt\(\(\)=>import\("\./anima-fast\.html\.page\.js(?:\?v=[^"]*)?"\),\[\]\)\),'
)
_FAST_DATA_IMPORT = re.compile(
    r'"v-anima-fast":\(\)=>wt\(\(\)=>import\("\./anima-fast\.html\.data\.js(?:\?v=[^"]*)?"\),\[\]\)'
    r"\.then\(\(\{data:e\}\)=>e\),"
)


_FINETUNE_PAGE_IMPORT = re.compile(
    r'"v-a1f1ne2e":Jt\(\(\)=>wt\(\(\)=>import\("\./anima-finetune\.html\.1a4bf32e\.js(?:\?v=[^"]*)?"\),\[\]\)\),'
)
_FINETUNE_DATA_IMPORT = re.compile(
    r'"v-a1f1ne2e":\(\)=>wt\(\(\)=>import\("\./anima-finetune\.html\.eaeb05f2\.js(?:\?v=[^"]*)?"\),\[\]\)'
    r"\.then\(\(\{data:e\}\)=>e\),"
)


def _page_import() -> str:
    return (
        f'"{ROUTE_KEY}":Jt(()=>wt(()=>import('
        f'"./anima-2.9b.html.page.js?v={SPA_ASSET_CACHE_KEY}"),[])),'
    )


def _data_import() -> str:
    return (
        f'"{ROUTE_KEY}":()=>wt(()=>import('
        f'"./anima-2.9b.html.data.js?v={SPA_ASSET_CACHE_KEY}"),[]).then(({{data:e}})=>e),'
    )


def _ft_page_import() -> str:
    return (
        f'"{FT_ROUTE_KEY}":Jt(()=>wt(()=>import('
        f'"./anima-2.9b-finetune.html.page.js?v={SPA_ASSET_CACHE_KEY}"),[])),'
    )


def _ft_data_import() -> str:
    return (
        f'"{FT_ROUTE_KEY}":()=>wt(()=>import('
        f'"./anima-2.9b-finetune.html.data.js?v={SPA_ASSET_CACHE_KEY}"),[]).then(({{data:e}})=>e),'
    )


def patch_app_js() -> None:
    text = APP_JS.read_text(encoding="utf-8")

    def _fix_page(match: re.Match[str]) -> str:
        return (
            f'"v-anima-fast":Jt(()=>wt(()=>import("{match.group(1)}"),[])),'
            + _page_import()
        )

    def _fix_data(match: re.Match[str]) -> str:
        return (
            f'"v-anima-fast":()=>wt(()=>import("{match.group(1)}"),[]).then(({{data:e}})=>e),'
            + _data_import()
        )

    repaired_page, page_n = _BROKEN_PAGE_IMPORT.subn(_fix_page, text, count=1)
    if page_n:
        text = repaired_page
        print("repaired nested anima-2.9b page import in app.js")
    repaired_data, data_n = _BROKEN_DATA_IMPORT.subn(_fix_data, text, count=1)
    if data_n:
        text = repaired_data
        print("repaired nested anima-2.9b data import in app.js")

    if '"v-anima-29b":Jt(' not in text:
        match = _FAST_PAGE_IMPORT.search(text)
        if match is None:
            raise SystemExit("anima-fast page component close not found in app.js")
        text = text[: match.end()] + _page_import() + text[match.end() :]
        print("patched app.js i0 page component map")

    if '"./anima-2.9b.html.data.js' not in text:
        match = _FAST_DATA_IMPORT.search(text)
        if match is None:
            raise SystemExit("anima-fast data import close not found in app.js")
        text = text[: match.end()] + _data_import() + text[match.end() :]
        print("patched app.js route data map")

    cleaned, comma_n = re.subn(
        r'(\./anima-2\.9b\.html\.data\.js(?:\?v=[^"]*)?"\),\[\]\)\.then\(\(\{data:e\}\)=>e\),),',
        r"\1",
        text,
        count=1,
    )
    if comma_n:
        text = cleaned
        print("removed leftover comma after anima-2.9b data import")

    route_tuple = (
        f'["{ROUTE_KEY}","/lora/anima-2.9b.html",'
        f'{{title:{json.dumps(PAGE_TITLE)}}},'
        '["/lora/anima-2.9b","/lora/anima-2.9b.md"]],'
    )
    fast_tuple = (
        '["v-anima-fast","/lora/anima-fast.html",{title:"Anima LoRA \\u00b7 Fast \\u6a21\\u5f0f"},'
        '["/lora/anima-fast","/lora/anima-fast.md"]],'
    )
    if f'"{ROUTE_KEY}","/lora/anima-2.9b.html"' not in text:
        if fast_tuple not in text:
            raise SystemExit("anima-fast route tuple not found in app.js")
        text = text.replace(fast_tuple, fast_tuple + route_tuple, 1)
        print("patched app.js route tuple")

    lora_sidebar = f'{{"text":{json.dumps(PAGE_TITLE)},"link":"/lora/anima-2.9b.md"}}'
    text = text.replace(
        '{"text":"Anima2.9b训练","link":"/lora/anima-2.9b.md"}',
        lora_sidebar,
        1,
    )
    text = text.replace(
        '{"text":"2.9B 模式","link":"/lora/anima-2.9b.md"}',
        lora_sidebar,
        1,
    )
    text = text.replace(
        f'["{ROUTE_KEY}","/lora/anima-2.9b.html",{{title:"Anima2.9b\\u8BAD\\u7EC3"}}',
        f'["{ROUTE_KEY}","/lora/anima-2.9b.html",{{title:{json.dumps(PAGE_TITLE)}}}',
        1,
    )
    if lora_sidebar not in text:
        sidebar_old = '{"text":"Fast 模式","link":"/lora/anima-fast.md"}]}'
        sidebar_new = (
            '{"text":"Fast 模式","link":"/lora/anima-fast.md"},'
            f'{lora_sidebar}]}}'
        )
        if sidebar_old not in text:
            raise SystemExit("Anima LoRA Fast sidebar child not found in app.js")
        text = text.replace(sidebar_old, sidebar_new, 1)
        print("patched app.js LoRA sidebar child")

    if f'"{FT_ROUTE_KEY}":Jt(' not in text:
        match = _FINETUNE_PAGE_IMPORT.search(text)
        if match is None:
            raise SystemExit("anima-finetune page component close not found in app.js")
        text = text[: match.end()] + _ft_page_import() + text[match.end() :]
        print("patched app.js 2.9B finetune page component map")

    if '"./anima-2.9b-finetune.html.data.js' not in text:
        match = _FINETUNE_DATA_IMPORT.search(text)
        if match is None:
            raise SystemExit("anima-finetune data import close not found in app.js")
        text = text[: match.end()] + _ft_data_import() + text[match.end() :]
        print("patched app.js 2.9B finetune route data map")

    ft_route_tuple = (
        f'["{FT_ROUTE_KEY}","/lora/anima-2.9b-finetune.html",'
        f'{{title:"Anima2.9B Finetune"}},'
        '["/lora/anima-2.9b-finetune","/lora/anima-2.9b-finetune.md"]],'
    )
    finetune_tuple = (
        '["v-a1f1ne2e","/lora/anima-finetune.html",'
        '{title:"Anima \\u5168\\u91cf\\u5FAE\\u8C03 \\u4E13\\u5BB6\\u6A21\\u5F0F"},'
        '["/lora/anima-finetune","/lora/anima-finetune.md"]],'
    )
    if f'"{FT_ROUTE_KEY}","/lora/anima-2.9b-finetune.html"' not in text:
        if finetune_tuple not in text:
            raise SystemExit("anima-finetune route tuple not found in app.js")
        text = text.replace(finetune_tuple, finetune_tuple + ft_route_tuple, 1)
        print("patched app.js 2.9B finetune route tuple")

    ft_sidebar_old = (
        '{"text":"Anima Finetune","link":"/lora/anima-finetune.md"},'
        '{"text":"Stable Diffusion","link":"/dreambooth/index.md"}]}'
    )
    ft_sidebar_new = (
        '{"text":"Anima Finetune","link":"/lora/anima-finetune.md"},'
        '{"text":"Anima2.9B Finetune","link":"/lora/anima-2.9b-finetune.md"},'
        '{"text":"Stable Diffusion","link":"/dreambooth/index.md"}]}'
    )
    if '{"text":"Anima2.9B Finetune","link":"/lora/anima-2.9b-finetune.md"}' not in text:
        if ft_sidebar_old not in text:
            raise SystemExit("Anima Finetune sidebar child not found in app.js")
        text = text.replace(ft_sidebar_old, ft_sidebar_new, 1)
        print("patched app.js 全量微调 sidebar child")

    APP_JS.write_text(text, encoding="utf-8")
    print("patched app.js")


HUB_HTML = DIST / "lora" / "index.html"
HUB_JS = ASSETS / "index.html.4896b94d.js"

_HUB_LI_JS = (
    'e("li",null,[e("strong",null,"Anima"),s(" — 主推训练入口（Anima DiT）")]),'
    'e("li",null,[e("strong",null,"Flux"),s(" — Flux 模型 LoRA")]),'
    'e("li",null,[e("strong",null,"Stable Diffusion"),s(" — SD1.5 / SDXL（页顶切换训练种类，默认 SDXL）")])'
)
_HUB_LI_JS_LINKED = (
    'e("li",null,[e("a",{href:"/lora/sd3.html"},[e("strong",null,"Anima"),s(" — 主推训练入口（Anima DiT）")])]),'
    'e("li",null,[e("a",{href:"/lora/flux.html"},[e("strong",null,"Flux"),s(" — Flux 模型 LoRA")])]),'
    'e("li",null,[e("a",{href:"/lora/master.html"},[e("strong",null,"Stable Diffusion"),s(" — SD1.5 / SDXL（页顶切换训练种类，默认 SDXL）")])])'
)
_HUB_LI_HTML = (
    "<ul><li><strong>Anima</strong> — 主推训练入口（Anima DiT）</li>"
    "<li><strong>Flux</strong> — Flux 模型 LoRA</li>"
    "<li><strong>Stable Diffusion</strong> — SD1.5 / SDXL（页顶切换训练种类，默认 SDXL）</li></ul>"
)
_HUB_LI_HTML_LINKED = (
    '<ul><li><a href="/lora/sd3.html"><strong>Anima</strong> — 主推训练入口（Anima DiT）</a></li>'
    '<li><a href="/lora/flux.html"><strong>Flux</strong> — Flux 模型 LoRA</a></li>'
    '<li><a href="/lora/master.html"><strong>Stable Diffusion</strong> — SD1.5 / SDXL（页顶切换训练种类，默认 SDXL）</a></li></ul>'
)


def patch_lora_hub() -> None:
    js = HUB_JS.read_text(encoding="utf-8")
    if 'href:"/lora/sd3.html"' not in js:
        if _HUB_LI_JS not in js:
            raise SystemExit("LoRA hub JS list items not found")
        js = js.replace(_HUB_LI_JS, _HUB_LI_JS_LINKED, 1)
        HUB_JS.write_text(js, encoding="utf-8")
        print("patched LoRA hub page JS links")
    html = HUB_HTML.read_text(encoding="utf-8")
    if 'href="/lora/sd3.html"' not in html.split("theme-default-content", 1)[-1]:
        if _HUB_LI_HTML not in html:
            raise SystemExit("LoRA hub HTML list items not found")
        html = html.replace(_HUB_LI_HTML, _HUB_LI_HTML_LINKED, 1)
        HUB_HTML.write_text(html, encoding="utf-8")
        print("patched LoRA hub HTML links")


def patch_queue_js() -> None:
    text = QUEUE_JS.read_text(encoding="utf-8")
    changed = False
    lora_entry = '"anima-2.9b": { path: "/lora/anima-2.9b.html" },'
    if lora_entry not in text:
        needle = '    "anima-lora-fast": { path: "/lora/anima-fast.html" },'
        if needle not in text:
            raise SystemExit("anima-lora-fast PAGE_MAP entry not found")
        text = text.replace(needle, needle + "\n    " + lora_entry, 1)
        changed = True
        print("patched sd-trainer-queue.js PAGE_MAP anima-2.9b")
    ft_entry = '"anima-2.9b-finetune": { path: "/lora/anima-2.9b-finetune.html" },'
    if ft_entry not in text:
        needle = '    "anima-2.9b": { path: "/lora/anima-2.9b.html" },'
        if needle not in text:
            raise SystemExit("anima-2.9b PAGE_MAP entry not found")
        text = text.replace(needle, needle + "\n    " + ft_entry, 1)
        changed = True
        print("patched sd-trainer-queue.js PAGE_MAP anima-2.9b-finetune")
    if changed:
        QUEUE_JS.write_text(text, encoding="utf-8")


def main() -> None:
    write_page_assets()
    write_finetune_page_assets()
    patch_app_js()
    patch_lora_hub()
    patch_queue_js()
    print("anima-2.9b UI patch done")


if __name__ == "__main__":
    main()
