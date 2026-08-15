"""Anima 2.9B split-page smoke: /api/run routing + wrapper rewrite (no GPU)."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from starlette.requests import Request

from mikazuki.anima_backend.adapter import INSERTED_40_BLOCK_INDICES_CSV
from mikazuki.app import api


def _request(payload: dict) -> Request:
    body = json.dumps(payload).encode("utf-8")

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request({"type": "http", "method": "POST", "path": "/api/run", "headers": []}, receive)


def _dataset_root(root: Path) -> Path:
    data_dir = root / "dataset"
    (data_dir / "1_class").mkdir(parents=True)
    return data_dir


def _base_payload(root: Path, *, train_type: str) -> dict:
    payload = {
        "model_train_type": train_type,
        "train_data_dir": str(_dataset_root(root)),
        "pretrained_model_name_or_path": str(root / "Anima-2.9B-preview-v1.safetensors"),
        "vae": str(root / "qwen_image_vae.safetensors"),
        "qwen3": str(root / "qwen_3_06b_base.safetensors"),
        "output_dir": str(root / "output"),
        "output_name": "anima-29b-smoke",
        "freeze_inserted_only_training": True,
        "attn_mode": "torch",
        "enable_preview": False,
        "optimizer_type": "AdamW8bit",
        "mixed_precision": "bf16",
        "max_train_epochs": 1,
        "train_batch_size": 1,
    }
    (root / "Anima-2.9B-preview-v1.safetensors").write_bytes(b"stub")
    (root / "qwen_image_vae.safetensors").write_bytes(b"stub")
    (root / "qwen_3_06b_base.safetensors").write_bytes(b"stub")
    return payload


def _run_submit(payload: dict, root: Path, fake_response):
    with mock.patch.object(api.os, "getcwd", return_value=str(root)), mock.patch.object(
        api.process, "run_train", return_value=fake_response
    ) as run_train:
        response = asyncio.run(api.create_toml_file(_request(payload)))
    return response, run_train


class Anima29bApiSmokeTests(unittest.TestCase):
    def test_lora_page_submits_to_network_wrapper(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            payload = _base_payload(root, train_type="anima-2.9b")
            payload["lora_type"] = "lora"
            payload["network_dim"] = 16
            payload["network_alpha"] = 16
            fake = api.APIResponseSuccess(
                message="Training started",
                data={"task_id": "29b-lora", "metadata": {"trainer_file": "./scripts/dev/anima_train_network.py"}},
            )

            response, run_train = _run_submit(payload, root, fake)

            self.assertEqual(response.status, "success", getattr(response, "message", response))
            run_train.assert_called_once()
            toml_path, trainer_file, _gpu_ids, _cpu = run_train.call_args.args
            self.assertEqual(trainer_file, "./scripts/dev/anima_train_network.py")
            text = Path(toml_path).read_text(encoding="utf-8")
            self.assertIn("freeze_inserted_only_training", text)
            self.assertNotIn("model_train_type", text)
            self.assertNotIn("anima_29b_train_mode", text)

    def test_finetune_page_submits_to_full_train_wrapper(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            payload = _base_payload(root, train_type="anima-2.9b-finetune")
            payload["learning_rate"] = "1e-5"
            fake = api.APIResponseSuccess(
                message="Training started",
                data={"task_id": "29b-ft", "metadata": {"trainer_file": "./scripts/dev/anima_train.py"}},
            )

            response, run_train = _run_submit(payload, root, fake)

            self.assertEqual(response.status, "success", getattr(response, "message", response))
            run_train.assert_called_once()
            toml_path, trainer_file, _gpu_ids, _cpu = run_train.call_args.args
            self.assertEqual(trainer_file, "./scripts/dev/anima_train.py")
            text = Path(toml_path).read_text(encoding="utf-8")
            self.assertIn("freeze_inserted_only_training", text)
            self.assertNotIn("lora_type", text)


class Anima29bWrapperSmokeTests(unittest.TestCase):
    def test_lora_wrapper_rewrites_inserted_block_indices(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "anima-29b-lora.toml"
            config_path.write_text(
                "\n".join(
                    [
                        'model_train_type = "anima-2.9b"',
                        'pretrained_model_name_or_path = "model.safetensors"',
                        'vae = "vae.safetensors"',
                        'qwen3 = "qwen3.safetensors"',
                        'lora_type = "lora"',
                        "freeze_inserted_only_training = true",
                        "network_dim = 16",
                        "network_alpha = 16",
                    ]
                ),
                encoding="utf-8",
            )
            env = os.environ.copy()
            env["ANIMA_BACKEND_WRAPPER_SMOKE"] = "1"
            result = subprocess.run(
                [sys.executable, "scripts/dev/anima_train_network.py", "--config_file", str(config_path)],
                cwd=Path.cwd(),
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Anima backend wrapper smoke OK", result.stdout)
            self.assertIn("anima_train_network.py", result.stdout)
            adapted = Path(td) / "anima-29b-lora-sd-scripts.toml"
            self.assertTrue(adapted.is_file(), result.stdout + result.stderr)
            adapted_text = adapted.read_text(encoding="utf-8")
            self.assertIn(f"train_block_indices={INSERTED_40_BLOCK_INDICES_CSV}", adapted_text)
            self.assertIn('network_module = "networks.lora_anima"', adapted_text)
            self.assertNotIn("freeze_inserted_only_training", adapted_text)
            self.assertNotIn("model_train_type", adapted_text)

    def test_finetune_wrapper_keeps_inserted_only_flag(self):
        with tempfile.TemporaryDirectory() as td:
            config_path = Path(td) / "anima-29b-finetune.toml"
            config_path.write_text(
                "\n".join(
                    [
                        'model_train_type = "anima-2.9b-finetune"',
                        'pretrained_model_name_or_path = "model.safetensors"',
                        'vae = "vae.safetensors"',
                        'qwen3 = "qwen3.safetensors"',
                        "freeze_inserted_only_training = true",
                        'learning_rate = "1e-5"',
                    ]
                ),
                encoding="utf-8",
            )
            env = os.environ.copy()
            env["ANIMA_BACKEND_WRAPPER_SMOKE"] = "1"
            result = subprocess.run(
                [sys.executable, "scripts/dev/anima_train.py", "--config_file", str(config_path)],
                cwd=Path.cwd(),
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Anima backend wrapper smoke OK", result.stdout)
            self.assertIn("anima_train.py", result.stdout)
            adapted = Path(td) / "anima-29b-finetune-sd-scripts.toml"
            self.assertTrue(adapted.is_file(), result.stdout + result.stderr)
            adapted_text = adapted.read_text(encoding="utf-8")
            self.assertIn("freeze_inserted_only_training = true", adapted_text)
            self.assertNotIn("train_block_indices", adapted_text)
            self.assertNotIn("network_module", adapted_text)


if __name__ == "__main__":
    unittest.main()
