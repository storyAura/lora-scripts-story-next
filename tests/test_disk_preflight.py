from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from mikazuki.app import api
from mikazuki.disk_preflight import (
    SKIP_ENV,
    DiskSpaceError,
    GiB,
    MiB,
    check_training_disk_space,
    estimate_training_disk_need,
    format_bytes,
    skip_disk_preflight,
)
from tests.test_standard_run_api import make_request


def test_format_bytes():
    assert format_bytes(500) == "500 B"
    assert "MB" in format_bytes(10 * MiB)
    assert "GB" in format_bytes(3 * GiB)


def test_estimate_lora_scales_with_dim_and_saves():
    small = estimate_training_disk_need(
        {
            "network_dim": 4,
            "max_train_epochs": 1,
            "save_every_n_epochs": 1,
            "resolution": "512,512",
        },
        "sdxl-lora",
        image_count=3,
    )
    large = estimate_training_disk_need(
        {
            "network_dim": 128,
            "max_train_epochs": 10,
            "save_every_n_epochs": 1,
            "resolution": "512,512",
        },
        "sdxl-lora",
        image_count=3,
    )
    assert large.output_bytes > small.output_bytes
    assert large.breakdown["checkpoint_count"] >= small.breakdown["checkpoint_count"]


def test_estimate_cache_and_multires_inflate_dataset_volume():
    base = estimate_training_disk_need(
        {
            "network_dim": 16,
            "max_train_epochs": 1,
            "save_every_n_epochs": 1,
            "resolution": "1024,1024",
            "cache_latents_to_disk": False,
            "cache_text_encoder_outputs_to_disk": False,
        },
        "anima-lora",
        image_count=10,
    )
    cached = estimate_training_disk_need(
        {
            "network_dim": 16,
            "max_train_epochs": 1,
            "save_every_n_epochs": 1,
            "resolution": "1024,1024",
            "cache_latents_to_disk": True,
            "cache_text_encoder_outputs_to_disk": True,
            "multires_per_image": True,
            "target_res": ["512", "1024"],
        },
        "anima-lora",
        image_count=10,
    )
    assert base.cache_bytes == 0
    assert cached.cache_bytes > 0
    assert cached.breakdown["multires_tiers"] == 2
    assert cached.breakdown["latents_cache_bytes"] > 0
    assert cached.breakdown["text_encoder_cache_bytes"] == 10 * 2 * MiB


def test_check_passes_when_free_space_is_enough(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out = tmp_path / "out"
    out.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    (data / "1_class").mkdir()
    (data / "1_class" / "a.png").write_bytes(b"x")

    monkeypatch.setattr(
        "mikazuki.disk_preflight.shutil.disk_usage",
        lambda _path: SimpleNamespace(total=100 * GiB, used=0, free=50 * GiB),
    )
    monkeypatch.setenv("MIKAZUKI_TRAINER_SETTINGS", str(tmp_path / "missing.json"))
    monkeypatch.delenv(SKIP_ENV, raising=False)
    need = check_training_disk_space(
        {
            "output_dir": str(out),
            "train_data_dir": str(data),
            "network_dim": 8,
            "max_train_epochs": 1,
            "save_every_n_epochs": 1,
            "cache_latents_to_disk": True,
            "resolution": "512,512",
        },
        "sdxl-lora",
        image_count=1,
        autosave_dir=str(tmp_path / "autosave"),
    )
    assert need.output_bytes > 0


def test_check_raises_when_output_volume_is_short(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out = tmp_path / "out"
    out.mkdir()

    monkeypatch.setattr(
        "mikazuki.disk_preflight.shutil.disk_usage",
        lambda _path: SimpleNamespace(total=10 * GiB, used=9 * GiB, free=200 * MiB),
    )
    monkeypatch.setenv("MIKAZUKI_TRAINER_SETTINGS", str(tmp_path / "missing.json"))
    monkeypatch.delenv(SKIP_ENV, raising=False)
    with pytest.raises(DiskSpaceError) as ctx:
        check_training_disk_space(
            {
                "output_dir": str(out),
                "train_data_dir": str(tmp_path),
                "network_dim": 64,
                "max_train_epochs": 20,
                "save_every_n_epochs": 1,
                "resolution": "1024,1024",
            },
            "anima-lora",
            image_count=50,
            autosave_dir=str(tmp_path / "autosave"),
        )
    assert ctx.value.field == "disk_space"
    assert "磁盘空间不足" in str(ctx.value)
    assert "GB" in str(ctx.value) or "MB" in str(ctx.value)
    assert ctx.value.as_dict()["paths"]


def test_skip_env_bypasses_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(SKIP_ENV, "1")
    assert skip_disk_preflight() is True
    monkeypatch.setattr(
        "mikazuki.disk_preflight.shutil.disk_usage",
        lambda _path: (_ for _ in ()).throw(AssertionError("disk_usage should not run")),
    )
    result = check_training_disk_space(
        {"output_dir": str(tmp_path), "network_dim": 64, "max_train_epochs": 99},
        "anima-lora",
        image_count=1000,
        autosave_dir=str(tmp_path),
    )
    assert result.output_bytes == 0


def test_typical_anima_lora_fits_on_3gb_free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Regression: stacked 2 GiB reserves used to reject this as ~4.4 GB."""
    out = tmp_path / "out"
    out.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    (data / "1_class").mkdir()
    (data / "1_class" / "a.png").write_bytes(b"x")

    monkeypatch.setattr(
        "mikazuki.disk_preflight.shutil.disk_usage",
        lambda _path: SimpleNamespace(total=20 * GiB, used=17 * GiB, free=3 * GiB),
    )
    monkeypatch.delenv(SKIP_ENV, raising=False)
    monkeypatch.setenv("MIKAZUKI_TRAINER_SETTINGS", str(tmp_path / "missing.json"))
    need = check_training_disk_space(
        {
            "output_dir": str(out),
            "train_data_dir": str(data),
            "network_dim": 16,
            "max_train_epochs": 10,
            "save_every_n_epochs": 2,
            "cache_latents_to_disk": True,
            "cache_text_encoder_outputs_to_disk": False,
            "freeze_inserted_only_training": True,
            "resolution": "1024,1024",
        },
        "anima-2.9b",
        image_count=40,
        autosave_dir=str(tmp_path / "autosave"),
    )
    scaled = need.output_bytes + need.cache_bytes
    assert scaled < 2 * GiB
    assert need.breakdown["checkpoint_count"] <= 6


def test_unset_save_every_does_not_assume_every_epoch():
    need = estimate_training_disk_need(
        {"network_dim": 16, "max_train_epochs": 10, "resolution": "1024,1024"},
        "anima-2.9b",
        image_count=10,
    )
    assert need.breakdown["checkpoint_count"] == 1


def test_trainer_settings_can_skip_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from mikazuki.trainer_settings import save_trainer_settings

    settings_file = tmp_path / "trainer_settings.json"
    monkeypatch.setenv("MIKAZUKI_TRAINER_SETTINGS", str(settings_file))
    monkeypatch.delenv(SKIP_ENV, raising=False)
    save_trainer_settings({"disk_preflight_enabled": False})
    monkeypatch.setattr(
        "mikazuki.disk_preflight.shutil.disk_usage",
        lambda _path: (_ for _ in ()).throw(AssertionError("disk_usage should not run")),
    )
    result = check_training_disk_space(
        {"output_dir": str(tmp_path), "network_dim": 64, "max_train_epochs": 99},
        "anima-lora",
        image_count=1000,
        autosave_dir=str(tmp_path),
    )
    assert result.output_bytes == 0


def test_cache_on_separate_volume_is_checked_apart_from_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out = tmp_path / "out"
    out.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("MIKAZUKI_TRAINER_SETTINGS", str(tmp_path / "missing.json"))
    monkeypatch.delenv(SKIP_ENV, raising=False)
    monkeypatch.setattr("mikazuki.disk_preflight.same_volume", lambda _a, _b: False)
    seen: list[str] = []

    def fake_usage(path):
        seen.append(str(path))
        return SimpleNamespace(total=100 * GiB, used=0, free=50 * GiB)

    monkeypatch.setattr("mikazuki.disk_preflight.shutil.disk_usage", fake_usage)
    check_training_disk_space(
        {
            "output_dir": str(out),
            "train_data_dir": str(data),
            "network_dim": 16,
            "max_train_epochs": 1,
            "save_every_n_epochs": 1,
            "cache_latents_to_disk": True,
            "resolution": "1024,1024",
        },
        "anima-lora",
        image_count=20,
        autosave_dir=str(tmp_path / "autosave"),
    )
    blob = " ".join(seen)
    assert str(out) in blob or str(out.resolve()) in blob
    assert str(data) in blob or str(data.resolve()) in blob


class SubmitDiskPreflightTests(unittest.TestCase):
    def test_submit_fails_before_run_train_when_disk_short(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data_dir = root / "dataset" / "1_class"
            data_dir.mkdir(parents=True)
            (data_dir / "sample.png").write_bytes(b"png")
            model_path = root / "model.safetensors"
            model_path.write_bytes(b"not-a-real-model")
            out = root / "output"
            out.mkdir()

            payload = {
                "model_train_type": "sdxl-lora",
                "train_data_dir": str(data_dir.parent),
                "pretrained_model_name_or_path": str(model_path),
                "output_dir": str(out),
                "output_name": "disk-preflight",
                "enable_preview": False,
                "network_dim": 32,
                "max_train_epochs": 10,
                "save_every_n_epochs": 1,
                "cache_latents_to_disk": False,
            }

            with mock.patch.object(api.os, "getcwd", return_value=str(root)), \
                    mock.patch.object(api, "check_training_disk_space", side_effect=DiskSpaceError(
                        "磁盘空间不足：预计约需 12.4 GB，output_dir 所在盘仅剩 3.1 GB。",
                        required_bytes=12 * GiB,
                        free_bytes=3 * GiB,
                        paths=[str(out)],
                    )), \
                    mock.patch.object(api.process, "run_train") as run_train, \
                    mock.patch.object(api.train_utils, "validate_model", return_value=(True, "ok")), \
                    mock.patch.object(api.train_utils, "validate_data_dir", return_value=True), \
                    mock.patch.object(api, "_apply_anima_training_defaults_or_fail", return_value=None):
                response = asyncio.run(api.create_toml_file(make_request(payload)))

            self.assertEqual(response.status, "fail")
            self.assertEqual(response.data["field"], "disk_space")
            self.assertIn("磁盘空间不足", response.message)
            run_train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
