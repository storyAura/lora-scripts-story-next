from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "mikazuki" / "schema" / "krea2-lora.ts"
APP_JS = ROOT / "frontend" / "dist" / "assets" / "app.547295de.js"
PAGE = ROOT / "frontend" / "dist" / "lora" / "krea2.html"
PAGE_DATA = ROOT / "frontend" / "dist" / "assets" / "krea2.html.data.js"
PAGE_JS = ROOT / "frontend" / "dist" / "assets" / "krea2.html.page.js"
TRAINER = ROOT / "vendor" / "sd-scripts" / "krea2_train_network.py"
WRAPPER = ROOT / "scripts" / "dev" / "krea2_train_network.py"


class Krea2SchemaStaticTests(unittest.TestCase):
    def test_schema_is_lora_only_with_krea2_schedulers(self):
        schema = SCHEMA.read_text(encoding="utf-8")
        self.assertIn('model_train_type: Schema.string().default("krea2-lora")', schema)
        self.assertIn('lora_type: Schema.union(["lora", "lokr"]).default("lora")', schema)
        self.assertLess(
            schema.find("lora_type: Schema.union"),
            schema.find("pretrained_model_name_or_path"),
        )
        self.assertIn("krea2_shift", schema)
        self.assertIn("flux_shift", schema)
        self.assertIn('discrete_flow_shift: Schema.number().step(0.001).default(2.5)', schema)
        self.assertIn('network_module: Schema.string().default("networks.lora_krea2")', schema)
        self.assertIn("turbo_dit", schema)
        self.assertNotIn("krea2_train.py", schema)
        self.assertNotIn("Schema.const('krea2-lora')", schema)


class Krea2FrontendStaticTests(unittest.TestCase):
    def test_dist_page_and_assets_exist(self):
        self.assertTrue(PAGE.is_file())
        self.assertTrue(PAGE_DATA.is_file())
        self.assertTrue(PAGE_JS.is_file())
        data = PAGE_DATA.read_text(encoding="utf-8")
        self.assertIn("krea2-lora", data)
        self.assertIn("/lora/krea2.html", data)
        html = PAGE.read_text(encoding="utf-8")
        self.assertIn("<title>Krea2 | Next Story Trainer</title>", html)
        self.assertIn("Krea2</h1>", html)

    def test_sidebar_and_app_js_register_krea2_next_to_flux(self):
        src = (ROOT / "scripts" / "patch-sidebar-nav.py").read_text(encoding="utf-8")
        app = APP_JS.read_text(encoding="utf-8")
        self.assertIn('{"text":"Krea2","link":"/lora/krea2.md"}', src)
        self.assertIn('{"text":"Krea2","link":"/lora/krea2.md"}', app)
        self.assertIn('"v-krea2-lora"', app)
        self.assertIn("/lora/krea2.html", app)
        flux_idx = src.find('{"text":"Flux","link":"/lora/flux.md"}')
        krea_idx = src.find('{"text":"Krea2","link":"/lora/krea2.md"}')
        sd_idx = src.find('{"text":"Stable Diffusion","link":"/lora/master.md"}')
        self.assertTrue(0 < flux_idx < krea_idx < sd_idx)

    def test_queue_and_hub_know_krea2(self):
        queue = (ROOT / "frontend" / "dist" / "assets" / "sd-trainer-queue.js").read_text(
            encoding="utf-8"
        )
        hub = (ROOT / "frontend" / "dist" / "lora" / "index.html").read_text(encoding="utf-8")
        self.assertIn('"krea2-lora": { path: "/lora/krea2.html" }', queue)
        self.assertIn("/lora/krea2.html", hub)
        hub_js = (ROOT / "frontend" / "dist" / "assets" / "index.html.4896b94d.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('href:"/lora/krea2.html"', hub_js)
        flux_idx = hub_js.find('href:"/lora/flux.html"')
        krea_idx = hub_js.find('href:"/lora/krea2.html"')
        sd_idx = hub_js.find('href:"/lora/master.html"')
        self.assertTrue(0 < flux_idx < krea_idx < sd_idx)


class Krea2TrainerStaticTests(unittest.TestCase):
    def test_wrapper_puts_vendor_on_sys_path(self):
        wrapper = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("vendor", wrapper)
        self.assertIn("sd-scripts", wrapper)
        self.assertIn("krea2_train_network.py", wrapper)

    def test_trainer_parser_lists_all_schedulers(self):
        source = TRAINER.read_text(encoding="utf-8")
        self.assertIn("krea2_shift", source)
        self.assertIn("flux_shift", source)
        self.assertIn("sigmoid", source)
        self.assertIn("uniform", source)
        self.assertIn("discrete_flow_shift", source)
        self.assertIn("networks.lora_krea2", (ROOT / "vendor" / "sd-scripts" / "networks" / "lora_krea2.py").read_text(encoding="utf-8"))
        self.assertIn("Krea2TextEncoderUnavailableError", (ROOT / "vendor" / "sd-scripts" / "library" / "krea2_encoder.py").read_text(encoding="utf-8"))


class Krea2RoutingTests(unittest.TestCase):
    def test_trainer_mapping_routes_to_wrapper(self):
        source = (ROOT / "mikazuki" / "app" / "api.py").read_text(encoding="utf-8")
        module = ast.parse(source)
        mapping = None
        for node in module.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "trainer_mapping":
                        mapping = ast.literal_eval(node.value)
        self.assertIsNotNone(mapping)
        self.assertEqual(mapping["krea2-lora"], "./scripts/dev/krea2_train_network.py")
        self.assertNotIn("krea2-finetune", mapping)

    def test_validate_model_skips_krea2_architecture_guess(self):
        from mikazuki.utils.train_utils import validate_model

        ok, _message = validate_model("./sd-models/krea2/raw.safetensors", "krea2-lora")
        self.assertTrue(ok)

    def test_spa_cache_key_supersedes_previous_krea2_key(self):
        from scripts.spa_asset_cache import LEGACY_SPA_ASSET_CACHE_KEYS, SPA_ASSET_CACHE_KEY

        self.assertEqual(SPA_ASSET_CACHE_KEY, "20260902-krea2-hub")
        self.assertIn("20260902-krea2-lora", LEGACY_SPA_ASSET_CACHE_KEYS)

    def test_trainer_parser_accepts_krea2_shift(self):
        import sys

        vendor = str(ROOT / "vendor" / "sd-scripts")
        if vendor in sys.path:
            sys.path.remove(vendor)
        sys.path.insert(0, vendor)
        import krea2_train_network

        parser = krea2_train_network.setup_parser()
        args = parser.parse_args(["--timestep_sampling", "krea2_shift"])
        self.assertEqual(args.timestep_sampling, "krea2_shift")
        self.assertEqual(args.discrete_flow_shift, 2.5)
        self.assertEqual(args.weighting_scheme, "none")
        self.assertIn("krea2_shift", parser._option_string_actions["--timestep_sampling"].choices)


if __name__ == "__main__":
    unittest.main()
