"""Quick-infer API smoke (no GPU): status → lora-info → run → images → terminate."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from fastapi.testclient import TestClient
from safetensors.torch import save_file

from mikazuki.app import infer_api
from mikazuki.app.application import app


def _write_anima_lora(path: Path) -> None:
    save_file(
        {"lora_unet_dummy.weight": torch.zeros(1)},
        str(path),
        metadata={
            "ss_base_model_version": "anima",
            "ss_network_module": "networks.lora_anima",
            "ss_sd_model_name": "anima-base-v1.0.safetensors",
            "ss_vae_name": "qwen_image_vae.safetensors",
            "ss_qwen3_name": "qwen_3_06b_base.safetensors",
        },
    )


def test_infer_api_smoke_status_info_run_images_terminate(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(infer_api, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(infer_api, "_training_busy", lambda: False)
    monkeypatch.setattr(infer_api, "train_env_overrides", lambda: {})

    script = tmp_path / "vendor" / "sd-scripts" / "anima_minimal_inference.py"
    script.parent.mkdir(parents=True)
    script.write_text("# stub\n", encoding="utf-8")

    lora = tmp_path / "output" / "run" / "char.safetensors"
    lora.parent.mkdir(parents=True)
    _write_anima_lora(lora)

    dit = tmp_path / "sd-models" / "anima" / "anima-base-v1.0.safetensors"
    vae = tmp_path / "sd-models" / "anima" / "qwen_image_vae.safetensors"
    te = tmp_path / "sd-models" / "anima" / "qwen_3_06b_base.safetensors"
    dit.parent.mkdir(parents=True)
    dit.write_bytes(b"d")
    vae.write_bytes(b"v")
    te.write_bytes(b"t")

    out_dir_holder: dict[str, Path] = {}
    fake_task = SimpleNamespace(
        task_id="infer-smoke-1",
        execute=mock.Mock(),
        wait=mock.Mock(),
        process=None,
        status=SimpleNamespace(name="COMPLETED"),
        metadata={},
    )

    def create_task(cmd, env, metadata=None, cwd=None, task_id=None):
        assert "anima_minimal_inference.py" in " ".join(cmd)
        assert "--lora_weight" in cmd
        fake_task.metadata = dict(metadata or {})
        out_dir_holder["path"] = Path(fake_task.metadata["output_dir"])
        out_dir_holder["path"].mkdir(parents=True, exist_ok=True)
        (out_dir_holder["path"] / "0001.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        return fake_task

    monkeypatch.setattr(infer_api.tm, "create_task", create_task)
    monkeypatch.setattr(infer_api.tm, "terminate_task", mock.Mock())
    monkeypatch.setattr(infer_api.tm, "tasks", {"infer-smoke-1": fake_task})

    def _fake_create_task(coro):
        coro.close()
        return None

    monkeypatch.setattr(infer_api.asyncio, "create_task", _fake_create_task)
    monkeypatch.setattr(infer_api, "_infer_task", lambda: None)

    client = TestClient(app)

    status = client.get("/api/infer/status")
    assert status.status_code == 200
    body = status.json()
    assert body["status"] == "success"
    assert "busy_training" in body["data"]
    assert any(item["name"] == "char.safetensors" for item in body["data"]["recent_loras"])

    info = client.get("/api/infer/lora-info", params={"path": str(lora)})
    assert info.status_code == 200
    info_body = info.json()
    assert info_body["status"] == "success"
    assert info_body["data"]["family"] == "anima"
    assert info_body["data"]["supported"] is True
    assert Path(info_body["data"]["suggested"]["dit"]).name == "anima-base-v1.0.safetensors"

    run = client.post(
        "/api/infer/run",
        json={
            "lora_path": str(lora),
            "dit": str(dit),
            "vae": str(vae),
            "text_encoder": str(te),
            "prompt": "1girl, solo",
            "steps": 4,
            "scheduler": "simple",
            "sampler": "euler",
            "flow_shift": 5.0,
        },
    )
    assert run.status_code == 200
    run_body = run.json()
    assert run_body["status"] == "success"
    assert run_body["data"]["task_id"] == "infer-smoke-1"

    monkeypatch.setattr(infer_api, "_infer_task", lambda: fake_task)
    images = client.get("/api/infer/images/infer-smoke-1")
    assert images.status_code == 200
    images_body = images.json()
    assert images_body["status"] == "success"
    assert "0001.png" in images_body["data"]["images"]

    png = client.get("/api/infer/image/infer-smoke-1/0001.png")
    assert png.status_code == 200
    assert png.content.startswith(b"\x89PNG")

    stop = client.post("/api/infer/terminate", json={"task_id": "infer-smoke-1"})
    assert stop.status_code == 200
    assert stop.json()["status"] == "success"
    infer_api.tm.terminate_task.assert_called_once_with("infer-smoke-1")


def test_infer_frontend_smoke_assets_present():
    brand = Path("frontend/dist/assets/sd-trainer-brand.js").read_text(encoding="utf-8")
    infer = Path("frontend/dist/assets/sd-trainer-infer.js").read_text(encoding="utf-8")
    assert "/assets/sd-trainer-infer.js" in brand
    assert "/api/infer/lora-info" in infer
    assert "autoFillFromSelectedLora" in infer
    assert 'NAV_ID = "sd-infer-nav"' in infer
