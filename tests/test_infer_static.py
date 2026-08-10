"""Static guards for the quick-infer panel (brand injection + sidebar)."""

import unittest
from pathlib import Path

BRAND_JS = Path("frontend/dist/assets/sd-trainer-brand.js")
INFER_JS = Path("frontend/dist/assets/sd-trainer-infer.js")
APP_PY = Path("mikazuki/app/application.py")


class InferPanelStaticTests(unittest.TestCase):
    def test_brand_loads_infer_script(self):
        brand = BRAND_JS.read_text(encoding="utf-8")
        self.assertIn("/assets/sd-trainer-infer.js", brand)
        self.assertIn("sd-trainer-infer-script", brand)

    def test_infer_js_sidebar_id_and_hash(self):
        js = INFER_JS.read_text(encoding="utf-8")
        self.assertIn('NAV_ID = "sd-infer-nav"', js)
        self.assertIn('href = "#sd-infer"', js)
        self.assertIn("/api/infer/run", js)
        self.assertIn("busy_training", js)
        self.assertIn("/api/infer/lora-info", js)
        self.assertIn("autoFillFromSelectedLora", js)
        # hydration-safe: append under 训练 children, heal by content not mere id
        self.assertIn('a[href="#sd-infer"]', js)
        self.assertIn("sidebar-item-children", js)

    def test_application_serves_infer_js_no_cache(self):
        text = APP_PY.read_text(encoding="utf-8")
        self.assertIn("/assets/sd-trainer-infer.js", text)


if __name__ == "__main__":
    unittest.main()
