"""Process UI deliverables from doc/local into committed assets."""
from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "doc" / "local" / "Next Story Trainer"
ICON_SRC = SRC / "app-icon-mark.png"
if not ICON_SRC.is_file():
    ICON_SRC = SRC / "next-story-trainer-main-logo-transparent.png"
if not ICON_SRC.is_file():
    ICON_SRC = SRC / "logo_light.png"
HOME_SRC = SRC / "app-icon-mark.png"
if not HOME_SRC.is_file():
    HOME_SRC = SRC / "next-story-trainer-main-logo-transparent.png"
if not HOME_SRC.is_file():
    HOME_SRC = SRC / "logo_light.png"
BANNER_SRC = SRC / "changelog-banner.png"
COVER_SRC = SRC / "feature-cover.png"
if not COVER_SRC.is_file():
    COVER_SRC = SRC / "brand-banner.png"
SOCIAL_SRC = SRC / "brand-banner.png"
DIST = ROOT / "frontend" / "dist"
FAVICON_VERSION = "20260808-v2.9.5-brand-mark"


def cover_resize(img: Image.Image, size: tuple[int, int], bg: tuple[int, int, int] = (248, 244, 252)) -> Image.Image:
    tw, th = size
    src = img.convert("RGBA")
    sw, sh = src.size
    scale = max(tw / sw, th / sh)
    nw, nh = int(sw * scale), int(sh * scale)
    resized = src.resize((nw, nh), Image.Resampling.LANCZOS)
    x0 = (nw - tw) // 2
    y0 = (nh - th) // 2
    cropped = resized.crop((x0, y0, x0 + tw, y0 + th))
    out = Image.new("RGB", size, bg)
    out.paste(cropped, mask=cropped.split()[3])
    return out


def contain_on_canvas(img: Image.Image, size: tuple[int, int], bg=(0, 0, 0, 0)) -> Image.Image:
    tw, th = size
    src = img.convert("RGBA")
    sw, sh = src.size
    scale = min(tw / sw, th / sh)
    nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
    resized = src.resize((nw, nh), Image.Resampling.LANCZOS)
    out = Image.new("RGBA", size, bg)
    out.paste(resized, ((tw - nw) // 2, (th - nh) // 2), resized)
    return out


def favicon_head_html() -> str:
    v = FAVICON_VERSION
    return (
        f'    <link rel="icon" href="/favicon.ico?v={v}" sizes="any">\n'
        f'    <link rel="icon" href="/assets/icon.65fd68ba.webp?v={v}" type="image/webp" sizes="512x512">\n'
        f'    <link rel="apple-touch-icon" href="/assets/icon.png?v={v}">'
    )


def patch_dist_favicon_links() -> None:
    head = favicon_head_html()
    icon_link_re = re.compile(r'\s*<link rel="(?:icon|apple-touch-icon)"[^>]*>\s*', re.I)
    for html_path in DIST.rglob("*.html"):
        text = html_path.read_text(encoding="utf-8")
        text = icon_link_re.sub("\n", text)
        marker = '<meta name="viewport"'
        idx = text.find(marker)
        if idx < 0:
            continue
        end = text.find(">", idx) + 1
        text = text[:end] + "\n" + head + text[end:]
        html_path.write_text(text, encoding="utf-8")
    print(f"patched favicon links in {DIST.relative_to(ROOT)}/**/*.html")


def patch_home_icon_cache_buster() -> None:
    index_js = DIST / "assets" / "index.html.c6ef684b.js"
    if index_js.is_file():
        text = index_js.read_text(encoding="utf-8")
        text = re.sub(
            r'/assets/home-logo\.webp\?v=[^"\']+',
            f"/assets/home-logo.webp?v={FAVICON_VERSION}",
            text,
        )
        index_js.write_text(text, encoding="utf-8")

    index_html = DIST / "index.html"
    html = index_html.read_text(encoding="utf-8")
    html = re.sub(
        r'/assets/home-logo\.webp\?v=[^"\s]+',
        f"/assets/home-logo.webp?v={FAVICON_VERSION}",
        html,
    )
    index_html.write_text(html, encoding="utf-8")
    print("patched home logo cache buster")


def patch_train_monitor_favicon() -> None:
    path = ROOT / "train_monitor" / "index.html"
    text = path.read_text(encoding="utf-8")
    text = re.sub(r'<link rel="icon" href="[^"]*"[^>]*>', "", text)
    text = text.replace(
        "<head>",
        f'<head>\n  <link rel="icon" href="/favicon.ico?v={FAVICON_VERSION}" sizes="any">',
        1,
    )
    path.write_text(text, encoding="utf-8")
    print("patched train_monitor/index.html favicon")


def main() -> None:
    icon = Image.open(ICON_SRC)
    home = Image.open(HOME_SRC)
    banner = Image.open(BANNER_SRC)
    cover_src = Image.open(COVER_SRC)
    social_src = Image.open(SOCIAL_SRC)

    readme_dir = ROOT / "assets" / "readme"
    readme_dir.mkdir(parents=True, exist_ok=True)
    (DIST / "assets").mkdir(parents=True, exist_ok=True)

    cover = cover_resize(cover_src, (1760, 880))
    cover.save(readme_dir / "next-story-trainer-cover.png", optimize=True)

    social = cover_resize(social_src, (1200, 630))
    social.save(readme_dir / "next-story-trainer-social.png", optimize=True)

    # Monitor / committed logo: prefer light full logo if present
    light = SRC / "logo_light.png"
    if light.is_file():
        Image.open(light).convert("RGBA").save(ROOT / "assets" / "logo.png", optimize=True)
    else:
        contain_on_canvas(icon, (1024, 1024)).save(ROOT / "assets" / "logo.png", optimize=True)

    icon_1024 = contain_on_canvas(icon, (1024, 1024))
    icon_1024.save(DIST / "assets" / "icon.png", optimize=True)
    icon_1024.resize((512, 512), Image.Resampling.LANCZOS).save(
        DIST / "assets" / "icon.65fd68ba.webp", format="WEBP", quality=90, method=6
    )

    home.convert("RGBA").save(DIST / "assets" / "home-logo.webp", format="WEBP", quality=90, method=6)
    banner.convert("RGBA").save(
        DIST / "assets" / "changelog-banner.webp", format="WEBP", quality=92, method=6
    )

    ico_master = contain_on_canvas(icon, (256, 256))
    ico_sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    for dest in (ROOT / "assets" / "favicon.ico", DIST / "favicon.ico"):
        ico_master.save(dest, format="ICO", sizes=ico_sizes)

    # Do not recreate guide-mascot / next-trainer-* / anima-cover (upstream leftovers).
    for stale in (
        DIST / "assets" / "guide-mascot.webp",
        readme_dir / "next-trainer-cover.png",
        readme_dir / "next-trainer-social.png",
        readme_dir / "anima-cover.png",
        readme_dir / "logo.svg",
        ROOT / "assets" / "cover.png",
    ):
        if stale.exists():
            stale.unlink()
            print(f"removed stale {stale.relative_to(ROOT)}")

    patch_dist_favicon_links()
    patch_home_icon_cache_buster()
    patch_train_monitor_favicon()

    print("Wrote:")
    for p in [
        readme_dir / "next-story-trainer-cover.png",
        readme_dir / "next-story-trainer-social.png",
        ROOT / "assets" / "logo.png",
        DIST / "assets" / "icon.png",
        DIST / "assets" / "icon.65fd68ba.webp",
        DIST / "assets" / "home-logo.webp",
        DIST / "assets" / "changelog-banner.webp",
        ROOT / "assets" / "favicon.ico",
        DIST / "favicon.ico",
    ]:
        print(f"  {p.relative_to(ROOT)} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
