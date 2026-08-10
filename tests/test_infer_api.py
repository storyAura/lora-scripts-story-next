from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from starlette.requests import Request

from mikazuki.app import infer_api


def _request(payload: dict) -> Request:
    body = json.dumps(payload).encode("utf-8")

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request({"type": "http", "method": "POST", "path": "/api/infer/run", "headers": []}, receive)


def _write_lora_with_meta(path: Path, meta: dict[str, str]) -> None:
    import torch
    from safetensors.torch import save_file

    save_file({"lora_unet_dummy.weight": torch.zeros(1)}, str(path), metadata=meta)


def test_list_recent_loras_skips_infer_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(infer_api, "_project_root", lambda: tmp_path)
    out = tmp_path / "output" / "run"
    out.mkdir(parents=True)
    (out / "char.safetensors").write_bytes(b"x")
    infer_dir = tmp_path / "output" / "infer" / "t"
    infer_dir.mkdir(parents=True)
    (infer_dir / "preview.safetensors").write_bytes(b"y")
    items = infer_api._list_recent_loras()
    names = {i["name"] for i in items}
    assert "char.safetensors" in names
    assert "preview.safetensors" not in names


def test_lora_info_detects_anima_and_suggests_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(infer_api, "_project_root", lambda: tmp_path)
    lora = tmp_path / "char.safetensors"
    _write_lora_with_meta(
        lora,
        {
            "ss_base_model_version": "anima",
            "ss_sd_model_name": "anima-base-v1.0.safetensors",
            "ss_vae_name": "qwen_image_vae.safetensors",
            "ss_qwen3_name": "qwen_3_06b_base.safetensors",
            "ss_network_module": "networks.lora_anima",
        },
    )
    dit = tmp_path / "sd-models" / "anima" / "anima-base-v1.0.safetensors"
    vae = tmp_path / "sd-models" / "anima" / "qwen_image_vae.safetensors"
    te = tmp_path / "sd-models" / "anima" / "qwen_3_06b_base.safetensors"
    dit.parent.mkdir(parents=True)
    dit.write_bytes(b"d")
    vae.write_bytes(b"v")
    te.write_bytes(b"t")

    info = infer_api._lora_info(lora)
    assert info["family"] == "anima"
    assert info["supported"] is True
    assert Path(info["suggested"]["dit"]).name == "anima-base-v1.0.safetensors"
    assert Path(info["suggested"]["vae"]).name == "qwen_image_vae.safetensors"
    assert Path(info["suggested"]["text_encoder"]).name == "qwen_3_06b_base.safetensors"


def test_lora_info_rejects_sdxl_family(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(infer_api, "_project_root", lambda: tmp_path)
    lora = tmp_path / "sdxl_smoke.safetensors"
    _write_lora_with_meta(
        lora,
        {
            "ss_base_model_version": "sdxl_base_v1-0",
            "ss_sd_model_name": "sdxl.safetensors",
            "ss_network_module": "networks.lora",
        },
    )
    info = infer_api._lora_info(lora)
    assert info["family"] == "sdxl"
    assert info["supported"] is False


def test_infer_run_rejects_when_training_busy(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(infer_api, "_training_busy", lambda: True)
    response = asyncio.run(infer_api.infer_run(_request({"prompt": "a"})))
    assert response.status == "fail"
    assert response.data["field"] == "gpu_busy"


def test_infer_run_rejects_missing_lora(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(infer_api, "_training_busy", lambda: False)
    monkeypatch.setattr(infer_api, "_infer_task", lambda: None)
    monkeypatch.setattr(infer_api, "_project_root", lambda: tmp_path)
    payload = {
        "lora_path": str(tmp_path / "missing.safetensors"),
        "dit": str(tmp_path / "dit.safetensors"),
        "vae": str(tmp_path / "vae.safetensors"),
        "text_encoder": str(tmp_path / "te"),
        "prompt": "1girl",
    }
    (tmp_path / "dit.safetensors").write_bytes(b"d")
    (tmp_path / "vae.safetensors").write_bytes(b"v")
    (tmp_path / "te").mkdir()
    response = asyncio.run(infer_api.infer_run(_request(payload)))
    assert response.status == "fail"
    assert "不存在" in response.message


def test_infer_run_starts_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(infer_api, "_training_busy", lambda: False)
    monkeypatch.setattr(infer_api, "_infer_task", lambda: None)
    monkeypatch.setattr(infer_api, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(infer_api, "_detect_lycoris", lambda _p: False)
    monkeypatch.setattr(infer_api, "train_env_overrides", lambda: {})

    script = tmp_path / "vendor" / "sd-scripts" / "anima_minimal_inference.py"
    script.parent.mkdir(parents=True)
    script.write_text("# stub\n", encoding="utf-8")
    lora = tmp_path / "a.safetensors"
    dit = tmp_path / "dit.safetensors"
    vae = tmp_path / "vae.safetensors"
    te = tmp_path / "te"
    _write_lora_with_meta(
        lora,
        {
            "ss_base_model_version": "anima",
            "ss_network_module": "networks.lora_anima",
            "ss_sd_model_name": "dit.safetensors",
        },
    )
    dit.write_bytes(b"d")
    vae.write_bytes(b"v")
    te.mkdir()

    fake_task = SimpleNamespace(task_id="infer-test-1", execute=mock.Mock(), wait=mock.Mock(), process=None)

    def create_task(cmd, env, metadata=None, cwd=None, task_id=None):
        assert "anima_minimal_inference.py" in " ".join(cmd)
        assert "--scheduler" in cmd and "beta" in cmd
        assert "--sampler" in cmd and "heun" in cmd
        assert metadata["kind"] == "infer"
        return fake_task

    monkeypatch.setattr(infer_api.tm, "create_task", create_task)

    def _fake_create_task(coro):
        coro.close()
        return None

    monkeypatch.setattr(infer_api.asyncio, "create_task", _fake_create_task)

    response = asyncio.run(
        infer_api.infer_run(
            _request(
                {
                    "lora_path": str(lora),
                    "dit": str(dit),
                    "vae": str(vae),
                    "text_encoder": str(te),
                    "prompt": "1girl, solo",
                    "steps": 20,
                    "scheduler": "beta",
                    "sampler": "heun",
                }
            )
        )
    )
    assert response.status == "success"
    assert response.data["task_id"] == "infer-test-1"
