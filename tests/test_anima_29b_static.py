from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "mikazuki" / "schema" / "anima-2.9b.ts"
FT_SCHEMA = ROOT / "mikazuki" / "schema" / "anima-2.9b-finetune.ts"
APP_JS = ROOT / "frontend" / "dist" / "assets" / "app.547295de.js"
QUEUE_JS = ROOT / "frontend" / "dist" / "assets" / "sd-trainer-queue.js"
PAGE = ROOT / "frontend" / "dist" / "lora" / "anima-2.9b.html"
PAGE_DATA = ROOT / "frontend" / "dist" / "assets" / "anima-2.9b.html.data.js"
PAGE_JS = ROOT / "frontend" / "dist" / "assets" / "anima-2.9b.html.page.js"
FT_PAGE = ROOT / "frontend" / "dist" / "lora" / "anima-2.9b-finetune.html"
FT_PAGE_DATA = ROOT / "frontend" / "dist" / "assets" / "anima-2.9b-finetune.html.data.js"
FT_PAGE_JS = ROOT / "frontend" / "dist" / "assets" / "anima-2.9b-finetune.html.page.js"


class Anima29bSchemaStaticTests(unittest.TestCase):
    def setUp(self):
        self.schema = SCHEMA.read_text(encoding="utf-8")
        self.ft_schema = FT_SCHEMA.read_text(encoding="utf-8")

    def test_lora_schema_is_lora_only(self):
        self.assertIn('model_train_type: Schema.string().default("anima-2.9b")', self.schema)
        self.assertNotIn("anima_29b_train_mode", self.schema)
        self.assertIn("freeze_inserted_only_training: Schema.boolean().default(true)", self.schema)
        self.assertIn("lora_type:", self.schema)
        self.assertIn("Anima-2.9B-preview-v1.safetensors", self.schema)

    def test_finetune_schema_has_finetune_controls_not_lora_type(self):
        self.assertIn(
            'model_train_type: Schema.string().default("anima-2.9b-finetune")',
            self.ft_schema,
        )
        self.assertIn("freeze_inserted_only_training: Schema.boolean().default(true)", self.ft_schema)
        self.assertIn("Anima-2.9B-preview-v1.safetensors", self.ft_schema)
        self.assertIn('learning_rate: Schema.string().default("1e-5")', self.ft_schema)
        self.assertNotIn("lora_type:", self.ft_schema)
        self.assertNotIn("anima_29b_train_mode", self.ft_schema)

    def test_existing_anima_pages_keep_their_default_dit_path(self):
        sd3 = (ROOT / "mikazuki" / "schema" / "sd3-lora.ts").read_text(encoding="utf-8")
        finetune = (ROOT / "mikazuki" / "schema" / "anima-finetune.ts").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("Anima-2.9B-preview-v1.safetensors", sd3)
        self.assertNotIn("Anima-2.9B-preview-v1.safetensors", finetune)


class Anima29bFrontendStaticTests(unittest.TestCase):
    def test_dist_page_and_assets_exist(self):
        self.assertTrue(PAGE.is_file())
        self.assertTrue(PAGE_DATA.is_file())
        self.assertTrue(PAGE_JS.is_file())
        data = PAGE_DATA.read_text(encoding="utf-8")
        self.assertIn("trainType", data)
        self.assertIn("anima-2.9b", data)
        self.assertIn("/lora/anima-2.9b.html", data)

    def test_finetune_dist_page_and_assets_exist(self):
        self.assertTrue(FT_PAGE.is_file())
        self.assertTrue(FT_PAGE_DATA.is_file())
        self.assertTrue(FT_PAGE_JS.is_file())
        data = FT_PAGE_DATA.read_text(encoding="utf-8")
        self.assertIn("trainType", data)
        self.assertIn("anima-2.9b-finetune", data)
        self.assertIn("/lora/anima-2.9b-finetune.html", data)

    def test_sidebar_nav_patch_source_keeps_29b_entries(self):
        """Re-running patch-sidebar-nav.py must not wipe 2.9B from NEW_SIDEBAR_JSON."""
        src = (ROOT / "scripts" / "patch-sidebar-nav.py").read_text(encoding="utf-8")
        self.assertIn('{"text":"Anima2.9B","link":"/lora/anima-2.9b.md"}', src)
        self.assertIn(
            '{"text":"Anima2.9B Finetune","link":"/lora/anima-2.9b-finetune.md"}',
            src,
        )

    def test_app_js_registers_split_routes_and_sidebar(self):
        app = APP_JS.read_text(encoding="utf-8")
        self.assertIn('"v-anima-29b"', app)
        self.assertIn('"v-anima-29b-ft"', app)
        self.assertIn("/lora/anima-2.9b.html", app)
        self.assertIn("/lora/anima-2.9b-finetune.html", app)
        self.assertIn('{"text":"Anima2.9B","link":"/lora/anima-2.9b.md"}', app)
        self.assertIn(
            '{"text":"Fast 模式","link":"/lora/anima-fast.md"},'
            '{"text":"Anima2.9B","link":"/lora/anima-2.9b.md"}]}',
            app,
        )
        page_js = PAGE_JS.read_text(encoding="utf-8")
        self.assertIn('a(" Anima2.9B")', page_js)
        self.assertNotIn("2.9B 模式", page_js)
        html = PAGE.read_text(encoding="utf-8")
        self.assertIn("<title>Anima2.9B | Next Story Trainer</title>", html)
        self.assertIn("> Anima2.9B</h1>", html)
        self.assertIn(
            '{"text":"Anima Finetune","link":"/lora/anima-finetune.md"},'
            '{"text":"Anima2.9B Finetune","link":"/lora/anima-2.9b-finetune.md"},',
            app,
        )
        self.assertRegex(
            app,
            r'"v-anima-fast":Jt\(\(\)=>wt\(\(\)=>import\("\./anima-fast\.html\.page\.js[^"]*"\),\[\]\)\),"v-anima-29b"',
        )
        self.assertRegex(
            app,
            r'"v-a1f1ne2e":Jt\(\(\)=>wt\(\(\)=>import\("\./anima-finetune\.html\.1a4bf32e\.js[^"]*"\),\[\]\)\),"v-anima-29b-ft"',
        )

    def test_lora_hub_links_to_real_train_pages(self):
        hub_js = (ROOT / "frontend" / "dist" / "assets" / "index.html.4896b94d.js").read_text(
            encoding="utf-8"
        )
        hub_html = (ROOT / "frontend" / "dist" / "lora" / "index.html").read_text(
            encoding="utf-8"
        )
        for text in (hub_js, hub_html):
            self.assertIn("/lora/sd3.html", text)
            self.assertIn("/lora/flux.html", text)
            self.assertIn("/lora/master.html", text)
            self.assertIn("主推训练入口", text)

    def test_queue_page_map_includes_both_29b_pages(self):
        queue = QUEUE_JS.read_text(encoding="utf-8")
        self.assertIn('"anima-2.9b": { path: "/lora/anima-2.9b.html" }', queue)
        self.assertIn(
            '"anima-2.9b-finetune": { path: "/lora/anima-2.9b-finetune.html" }',
            queue,
        )
        self.assertNotIn("function ensure29bNav()", queue)
        self.assertNotIn("function ensureAnimaLoraNav()", queue)


class Anima29bSchemaConstGuardTests(unittest.TestCase):
    """Leftover required consts (except lora_type discriminators) blank the form."""

    def test_network_module_stays_a_tolerant_string(self):
        schema = SCHEMA.read_text(encoding="utf-8")
        for match in re.finditer(r"network_module:\s*Schema\.const", schema):
            self.fail(f"network_module must not be a Schema.const marker: {match.group(0)}")


if __name__ == "__main__":
    unittest.main()
