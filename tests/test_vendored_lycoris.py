# -*- coding: utf-8 -*-
"""The vendored LyCORIS must stay in sync with the installed one.

`pip install lycoris-lora` provides upstream LyCORIS, which has none of the local
extension algos (glokr / bokr / bora / gsokr / glora_boft) nor the
Anima-specific fixes. vendor/lycoris holds the patched package so it travels with
the repo; scripts/sync_vendored_lycoris.py copies it over the installed one.

Editing only the venv copy used to silently lose the change on the next install,
so this test fails when the two drift apart.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sync_vendored_lycoris as sync


class VendoredLycorisTests(unittest.TestCase):
    def test_vendored_package_is_present_and_complete(self):
        self.assertTrue(sync.VENDORED.is_dir(), "vendor/lycoris is missing")
        files = {p.relative_to(sync.VENDORED).as_posix() for p in sync.vendored_files()}
        for required in (
            "__init__.py",
            "kohya.py",
            "wrapper.py",
            "modules/__init__.py",
            "modules/glokr.py",
            "modules/bokr.py",
            "modules/bora.py",
            "modules/gsokr.py",
            "modules/glora_boft.py",
            "modules/norms.py",
            "modules/functional.py",
            "kernels/__init__.py",
        ):
            self.assertIn(required, files, f"vendored lycoris is missing {required}")

    def test_vendored_package_carries_the_local_extensions(self):
        source = (sync.VENDORED / "kohya.py").read_text(encoding="utf-8")
        self.assertIn("extra_algo_kwargs", source, "kohya.py lost the extension kwargs forwarding")
        self.assertIn("set_current_timestep", source, "kohya.py lost the timestep hook")

        glokr = (sync.VENDORED / "modules" / "glokr.py").read_text(encoding="utf-8")
        self.assertIn("kron_rank", glokr, "glokr.py lost the multi-term Kronecker extension")
        self.assertNotIn("train_time_gates", glokr, "T-GLoKR time gates were removed; do not reintroduce")

        kohya = source
        self.assertIn("exclude_name", kohya, "kohya.py lost exclude_name in apply_preset")
        self.assertIn("target_exclude_names", kohya, "kohya.py lost exclude_name threading into create_modules_")

        lokr = (sync.VENDORED / "modules" / "lokr.py").read_text(encoding="utf-8")
        self.assertIn("compute_merged_delta", lokr, "lokr.py must re-export/use compute_merged_delta")

        modules_init = (sync.VENDORED / "modules" / "__init__.py").read_text(encoding="utf-8")
        list_start = modules_init.find("MODULE_LIST")
        self.assertGreater(list_start, 0, "modules/__init__.py lost MODULE_LIST")
        bokr_pos = modules_init.find("BokrModule", list_start)
        lokr_pos = modules_init.find("LokrModule", list_start)
        self.assertGreater(bokr_pos, 0, "MODULE_LIST must register BokrModule")
        self.assertGreater(lokr_pos, bokr_pos, "BokrModule must be listed before LokrModule")

    def test_installed_lycoris_matches_the_vendored_copy(self):
        target = sync.installed_lycoris_dir()
        if target is None:
            self.skipTest("lycoris is not installed in this interpreter")
        drifted = sync.compare(target)
        self.assertEqual(
            drifted,
            [],
            "installed lycoris differs from vendor/lycoris — run "
            "`python scripts/sync_vendored_lycoris.py` (and commit the vendored change "
            "if you edited the venv copy)",
        )


if __name__ == "__main__":
    unittest.main()
