"""Frontend dist cache busting and static cache headers."""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestSpaAssetCache(unittest.TestCase):
    def test_lokr_preview_fix_uses_a_new_cache_key(self):
        from scripts.spa_asset_cache import LEGACY_SPA_ASSET_CACHE_KEYS, SPA_ASSET_CACHE_KEY

        previous_key = "20260627-config-import"
        self.assertNotEqual(SPA_ASSET_CACHE_KEY, previous_key)
        self.assertIn(previous_key, LEGACY_SPA_ASSET_CACHE_KEYS)

    def test_fast_submit_feedback_uses_a_new_cache_key(self):
        from scripts.spa_asset_cache import LEGACY_SPA_ASSET_CACHE_KEYS, SPA_ASSET_CACHE_KEY

        previous_key = "20260723-v2.9.0-lokr-preview"
        self.assertNotEqual(SPA_ASSET_CACHE_KEY, previous_key)
        self.assertIn(previous_key, LEGACY_SPA_ASSET_CACHE_KEYS)

    def test_dist_uses_current_spa_cache_key(self):
        from scripts.spa_asset_cache import LEGACY_SPA_ASSET_CACHE_KEYS, SPA_ASSET_CACHE_KEY

        dist = ROOT / "frontend" / "dist"
        offenders: list[str] = []
        for path in dist.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".html", ".js"}:
                continue
            text = path.read_text(encoding="utf-8")
            for legacy in LEGACY_SPA_ASSET_CACHE_KEYS:
                if legacy in text:
                    offenders.append(f"{path.relative_to(ROOT)} still contains {legacy}")
        self.assertFalse(
            offenders,
            "run scripts/bump_spa_asset_cache_key.py after dist patches:\n"
            + "\n".join(offenders[:20]),
        )

        app_js = dist / "assets" / "app.547295de.js"
        self.assertIn(
            SPA_ASSET_CACHE_KEY,
            app_js.read_text(encoding="utf-8"),
            "app bundle must reference current cache key",
        )

    def test_layout_import_uses_current_cache_key(self):
        from scripts.spa_asset_cache import SPA_ASSET_CACHE_KEY

        app_js = (ROOT / "frontend/dist/assets/app.547295de.js").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            f"layout.96d49288.js?v={SPA_ASSET_CACHE_KEY}",
            app_js,
        )

    def test_in_place_patched_assets_not_immutable(self):
        source = (ROOT / "mikazuki/app/application.py").read_text(encoding="utf-8")
        self.assertIn("/assets/layout.96d49288.js", source)
        self.assertIn("/assets/app.547295de.js", source)
        self.assertIn("/assets/settings.html.06993f96.js", source)
        self.assertIn("/assets/settings.html.07aaabcc.js", source)
        self.assertIn("no-cache, must-revalidate", source)


if __name__ == "__main__":
    unittest.main()
