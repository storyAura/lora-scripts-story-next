"""Persistent trainer-wide settings (disk preflight, Hugging Face token, etc.).

Stored in ``config/trainer_settings.json`` (gitignored). The settings page
syncs here so ``/api/run`` and the training queue can read the same values.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mikazuki.launch_utils import base_dir_path

SETTINGS_FILENAME = "trainer_settings.json"

# Copied into a training config when the run itself left them empty.
HF_INJECT_FIELDS = (
    "huggingface_token",
    "huggingface_repo_id",
    "huggingface_path_in_repo",
    "huggingface_repo_visibility",
    "huggingface_repo_type",
    "async_upload",
    "save_state_to_huggingface",
)

DEFAULTS: dict[str, Any] = {
    "disk_preflight_enabled": True,
    "tensorboard_url": "",
    "huggingface_token": "",
    "huggingface_repo_id": "",
    "huggingface_path_in_repo": "",
    "huggingface_repo_visibility": "private",
    "huggingface_repo_type": "model",
    "async_upload": False,
    "save_state_to_huggingface": False,
}


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def settings_path() -> Path:
    explicit = str(os.environ.get("MIKAZUKI_TRAINER_SETTINGS") or "").strip()
    if explicit:
        return Path(explicit).expanduser()
    return Path(base_dir_path()) / "config" / SETTINGS_FILENAME


def _normalize(raw: dict[str, Any] | None) -> dict[str, Any]:
    data = dict(DEFAULTS)
    if not isinstance(raw, dict):
        return data
    if "disk_preflight_enabled" in raw:
        data["disk_preflight_enabled"] = _truthy(raw.get("disk_preflight_enabled"))
    for key in (
        "tensorboard_url",
        "huggingface_token",
        "huggingface_repo_id",
        "huggingface_path_in_repo",
        "huggingface_repo_visibility",
        "huggingface_repo_type",
    ):
        if key in raw and raw[key] is not None:
            data[key] = str(raw[key]).strip()
    vis = str(data["huggingface_repo_visibility"] or "private").strip().lower()
    data["huggingface_repo_visibility"] = vis if vis in {"public", "private"} else "private"
    repo_type = str(data["huggingface_repo_type"] or "model").strip().lower()
    data["huggingface_repo_type"] = repo_type if repo_type in {"model", "dataset", "space"} else "model"
    for key in ("async_upload", "save_state_to_huggingface"):
        if key in raw:
            data[key] = _truthy(raw.get(key))
    return data


def load_trainer_settings() -> dict[str, Any]:
    path = settings_path()
    if not path.is_file():
        return dict(DEFAULTS)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return dict(DEFAULTS)
    return _normalize(payload if isinstance(payload, dict) else None)


def save_trainer_settings(raw: dict[str, Any] | None) -> dict[str, Any]:
    data = _normalize(raw)
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return data


def disk_preflight_enabled() -> bool:
    return bool(load_trainer_settings().get("disk_preflight_enabled", True))


def huggingface_token() -> str:
    return str(load_trainer_settings().get("huggingface_token") or "").strip()


def apply_trainer_settings_to_config(config: dict[str, Any]) -> None:
    """Fill empty Hugging Face upload fields from trainer-wide settings."""
    settings = load_trainer_settings()
    token = str(settings.get("huggingface_token") or "").strip()
    repo_id = str(settings.get("huggingface_repo_id") or "").strip()
    if token and not str(config.get("huggingface_token") or "").strip():
        config["huggingface_token"] = token
    if not repo_id:
        return
    for key in HF_INJECT_FIELDS:
        if key == "huggingface_token":
            continue
        incoming = config.get(key)
        empty = key not in config or incoming in (None, "")
        if empty:
            config[key] = settings[key]


def huggingface_env_overrides() -> dict[str, str]:
    """Env for training subprocesses so huggingface_hub can download with the saved token."""
    token = huggingface_token()
    if not token:
        return {}
    return {
        "HF_TOKEN": token,
        "HUGGING_FACE_HUB_TOKEN": token,
    }
