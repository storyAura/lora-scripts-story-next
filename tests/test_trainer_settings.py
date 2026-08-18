from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from mikazuki.trainer_settings import (
    apply_trainer_settings_to_config,
    huggingface_env_overrides,
    load_trainer_settings,
    save_trainer_settings,
)


class TrainerSettingsTests(unittest.TestCase):
    def test_defaults_when_file_missing(self):
        with mock.patch.dict("os.environ", {"MIKAZUKI_TRAINER_SETTINGS": str(Path("no-such-trainer-settings.json"))}):
            self.assertTrue(load_trainer_settings()["disk_preflight_enabled"])

    def test_round_trip(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "trainer_settings.json"
            with mock.patch.dict("os.environ", {"MIKAZUKI_TRAINER_SETTINGS": str(path)}):
                saved = save_trainer_settings(
                    {
                        "disk_preflight_enabled": False,
                        "huggingface_token": "hf_test",
                        "huggingface_repo_id": "user/repo",
                        "async_upload": True,
                    }
                )
                self.assertFalse(saved["disk_preflight_enabled"])
                loaded = load_trainer_settings()
                self.assertEqual(loaded["huggingface_token"], "hf_test")
                self.assertEqual(loaded["huggingface_repo_id"], "user/repo")
                self.assertTrue(loaded["async_upload"])
                self.assertEqual(huggingface_env_overrides()["HF_TOKEN"], "hf_test")

    def test_injects_token_and_hub_fields_when_config_empty(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "trainer_settings.json"
            with mock.patch.dict("os.environ", {"MIKAZUKI_TRAINER_SETTINGS": str(path)}):
                save_trainer_settings(
                    {
                        "huggingface_token": "hf_abc",
                        "huggingface_repo_id": "me/lora",
                        "huggingface_repo_visibility": "private",
                    }
                )
                config = {"output_name": "n"}
                apply_trainer_settings_to_config(config)
                self.assertEqual(config["huggingface_token"], "hf_abc")
                self.assertEqual(config["huggingface_repo_id"], "me/lora")

                config2 = {"huggingface_token": "already", "huggingface_repo_id": "other/repo"}
                apply_trainer_settings_to_config(config2)
                self.assertEqual(config2["huggingface_token"], "already")
                self.assertEqual(config2["huggingface_repo_id"], "other/repo")

                config3 = {
                    "huggingface_token": "hf_abc",
                    "huggingface_repo_id": "me/lora",
                    "async_upload": False,
                    "save_state_to_huggingface": False,
                }
                apply_trainer_settings_to_config(config3)
                self.assertFalse(config3["async_upload"])
                self.assertFalse(config3["save_state_to_huggingface"])

    def test_does_not_inject_hub_upload_without_repo(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "trainer_settings.json"
            with mock.patch.dict("os.environ", {"MIKAZUKI_TRAINER_SETTINGS": str(path)}):
                save_trainer_settings({"huggingface_token": "hf_abc"})
                config = {}
                apply_trainer_settings_to_config(config)
                self.assertEqual(config["huggingface_token"], "hf_abc")
                self.assertNotIn("huggingface_repo_id", config)


class TrainerSettingsStaticTests(unittest.TestCase):
    def test_settings_page_schema_has_disk_toggle_and_hf_token(self):
        data = Path("frontend/dist/assets/settings.html.06993f96.js").read_text(encoding="utf-8")
        html = Path("frontend/dist/other/settings.html").read_text(encoding="utf-8")
        vue = Path("frontend/dist/assets/settings.html.07aaabcc.js").read_text(encoding="utf-8")
        self.assertIn("disk_preflight_enabled", data)
        self.assertIn("huggingface_token", data)
        self.assertIn("huggingface_repo_id", data)
        self.assertIn("save_state_to_huggingface", data)
        self.assertIn("训练器设置", data)
        self.assertIn("训练器设置", html)
        self.assertIn("\\u8BAD\\u7EC3\\u5668\\u8BBE\\u7F6E", vue)
        self.assertIn("config/trainer_settings.json", vue)
        self.assertNotIn("训练 UI 设置", html)
        self.assertNotIn("训练-ui-设置", vue)

    def test_sidebar_uses_trainer_settings_label(self):
        app = Path("frontend/dist/assets/app.547295de.js").read_text(encoding="utf-8")
        self.assertIn('"text":"训练器设置","link":"/other/settings.md"', app)
        index = Path("frontend/dist/index.html").read_text(encoding="utf-8")
        self.assertIn("训练器设置", index)
        self.assertNotIn(">UI 设置<", index)

    def test_brand_loads_settings_script_no_cache(self):
        brand = Path("frontend/dist/assets/sd-trainer-brand.js").read_text(encoding="utf-8")
        settings_js = Path("frontend/dist/assets/sd-trainer-settings.js").read_text(encoding="utf-8")
        app_py = Path("mikazuki/app/application.py").read_text(encoding="utf-8")
        self.assertIn("/assets/sd-trainer-settings.js", brand)
        self.assertIn("/api/trainer-settings", settings_js)
        self.assertIn("disk_preflight_enabled", settings_js)
        self.assertIn("/assets/sd-trainer-settings.js", app_py)


if __name__ == "__main__":
    unittest.main()
