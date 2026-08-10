"""Pre-launch disk space estimates for training submit / queue launch.

Pure stdlib helpers (no torch). Conservative: prefer over-estimating so cloud
disks fail loudly with a structured message instead of Errno 28 mid-write.
"""

from __future__ import annotations

import math
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mikazuki.multires import is_multires_enabled, normalize_target_res

MiB = 1024 * 1024
GiB = 1024 * MiB

SKIP_ENV = "MIKAZUKI_SKIP_DISK_PREFLIGHT"
SAFETY_FACTOR = 1.3
VOLUME_RESERVE_BYTES = 2 * GiB
AUTOSAVE_MIN_FREE_BYTES = 500 * MiB
DEFAULT_FINETUNE_CHECKPOINT_BYTES = 15 * GiB

ANIMA_LORA_TYPES = frozenset({"anima-lora", "sd3-lora", "anima-lora-fast"})
SD_FAMILY_LORA_TYPES = frozenset({"sd-lora", "sdxl-lora", "flux-lora"})
FINETUNE_TYPES = frozenset({"anima-finetune", "sdxl-finetune", "sd-dreambooth", "flux-finetune"})


class DiskSpaceError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        required_bytes: int,
        free_bytes: int,
        paths: list[str],
        breakdown: dict[str, Any] | None = None,
    ) -> None:
        self.field = "disk_space"
        self.required_bytes = int(required_bytes)
        self.free_bytes = int(free_bytes)
        self.paths = list(paths)
        self.breakdown = dict(breakdown or {})
        super().__init__(message)

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "required_bytes": self.required_bytes,
            "free_bytes": self.free_bytes,
            "paths": self.paths,
            "breakdown": self.breakdown,
        }


@dataclass
class DiskNeedEstimate:
    """Per-volume write estimates before safety factor / reserve."""

    output_bytes: int = 0
    cache_bytes: int = 0
    logging_bytes: int = 0
    autosave_bytes: int = AUTOSAVE_MIN_FREE_BYTES
    breakdown: dict[str, int] = field(default_factory=dict)

    def with_safety(self) -> "DiskNeedEstimate":
        """Apply safety factor to variable costs; keep autosave floor."""

        def _scale(value: int) -> int:
            return int(math.ceil(value * SAFETY_FACTOR))

        scaled = DiskNeedEstimate(
            output_bytes=_scale(self.output_bytes) + VOLUME_RESERVE_BYTES,
            cache_bytes=_scale(self.cache_bytes) + (VOLUME_RESERVE_BYTES if self.cache_bytes else 0),
            logging_bytes=_scale(self.logging_bytes),
            autosave_bytes=max(self.autosave_bytes, AUTOSAVE_MIN_FREE_BYTES),
            breakdown={k: int(v) for k, v in self.breakdown.items()},
        )
        scaled.breakdown["safety_factor"] = int(SAFETY_FACTOR * 100)
        scaled.breakdown["volume_reserve_bytes"] = VOLUME_RESERVE_BYTES
        return scaled


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _int(config: Mapping[str, object], field: str, default: int) -> int:
    raw = config.get(field, default)
    if raw in (None, ""):
        return default
    try:
        return int(float(raw))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def format_bytes(num: int) -> str:
    num = max(0, int(num))
    if num >= GiB:
        return f"{num / GiB:.1f} GB"
    if num >= MiB:
        return f"{num / MiB:.0f} MB"
    return f"{num} B"


def resolve_existing_path(path: str | os.PathLike[str] | None) -> Path | None:
    """Walk parents until an existing directory is found."""
    if path is None:
        return None
    text = str(path).strip()
    if not text:
        return None
    current = Path(text).expanduser()
    try:
        current = current.resolve(strict=False)
    except OSError:
        current = Path(text).expanduser()
    for candidate in [current, *current.parents]:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    return None


def volume_free_bytes(path: str | os.PathLike[str] | None) -> int:
    existing = resolve_existing_path(path)
    if existing is None:
        existing = Path.cwd()
    usage = shutil.disk_usage(str(existing))
    return int(usage.free)


def same_volume(path_a: str | os.PathLike[str] | None, path_b: str | os.PathLike[str] | None) -> bool:
    a = resolve_existing_path(path_a)
    b = resolve_existing_path(path_b)
    if a is None or b is None:
        return False
    try:
        return a.anchor.lower() == b.anchor.lower()
    except OSError:
        return str(a.anchor).lower() == str(b.anchor).lower()


def _parse_resolution(config: Mapping[str, object]) -> tuple[int, int]:
    raw = config.get("resolution", "1024,1024")
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        try:
            return max(64, int(raw[0])), max(64, int(raw[1]))
        except (TypeError, ValueError):
            pass
    text = str(raw or "1024").strip()
    if "," in text or "x" in text.lower():
        parts = text.lower().replace("x", ",").replace("，", ",").split(",")
        try:
            w = max(64, int(float(parts[0].strip())))
            h = max(64, int(float(parts[1].strip()))) if len(parts) > 1 else w
            return w, h
        except (TypeError, ValueError):
            pass
    try:
        edge = max(64, int(float(text)))
        return edge, edge
    except (TypeError, ValueError):
        return 1024, 1024


def _checkpoint_count(config: Mapping[str, object]) -> int:
    save_every_epochs = _int(config, "save_every_n_epochs", 0)
    save_every_steps = _int(config, "save_every_n_steps", 0)
    max_epochs = _int(config, "max_train_epochs", 0)
    max_steps = _int(config, "max_train_steps", 0)

    count = 1  # final save
    if save_every_epochs > 0 and max_epochs > 0:
        count += max_epochs // save_every_epochs
    elif save_every_steps > 0 and max_steps > 0:
        count += max_steps // save_every_steps
    elif max_epochs > 0:
        # default UI often saves every 1–2 epochs; assume every epoch if unset
        count += max_epochs
    return max(1, count)


def _lora_checkpoint_bytes(model_train_type: str, network_dim: int) -> int:
    dim = max(1, network_dim)
    if model_train_type in ANIMA_LORA_TYPES:
        return int(_clamp(dim * 6 * MiB, 80 * MiB, 2 * GiB))
    if model_train_type in SD_FAMILY_LORA_TYPES or model_train_type.endswith("-lora"):
        return int(_clamp(dim * 3 * MiB, 40 * MiB, 1.5 * GiB))
    return int(_clamp(dim * 3 * MiB, 40 * MiB, 1.5 * GiB))


def _finetune_checkpoint_bytes(config: Mapping[str, object]) -> int:
    model = str(config.get("pretrained_model_name_or_path") or "").strip()
    if model and os.path.isfile(model):
        try:
            return int(os.path.getsize(model) * 1.2)
        except OSError:
            pass
    return DEFAULT_FINETUNE_CHECKPOINT_BYTES


def _multires_tier_count(config: Mapping[str, object]) -> int:
    if not is_multires_enabled(config.get("multires_per_image")):
        return 1
    try:
        tiers = normalize_target_res(config.get("target_res"))
    except ValueError:
        tiers = []
    return max(2, len(tiers)) if tiers else 2


def estimate_training_disk_need(
    config: Mapping[str, object],
    model_train_type: str,
    image_count: int,
) -> DiskNeedEstimate:
    images = max(0, int(image_count))
    saves = _checkpoint_count(config)
    breakdown: dict[str, int] = {"checkpoint_count": saves, "image_count": images}

    if model_train_type in FINETUNE_TYPES:
        one = _finetune_checkpoint_bytes(config)
        output = one * saves
        breakdown["checkpoint_bytes_each"] = one
    else:
        dim = _int(config, "network_dim", 16)
        one = _lora_checkpoint_bytes(model_train_type, dim)
        output = one * saves
        breakdown["network_dim"] = dim
        breakdown["checkpoint_bytes_each"] = one
    breakdown["output_checkpoints_bytes"] = output

    cache = 0
    w, h = _parse_resolution(config)
    tiers = _multires_tier_count(config)
    breakdown["resolution_w"] = w
    breakdown["resolution_h"] = h
    breakdown["multires_tiers"] = tiers

    if _truthy(config.get("cache_latents_to_disk")):
        # Conservative latent footprint (bf16-ish packing guard via /64 grid).
        latent = int(images * (w / 64.0) * (h / 64.0) * 16 * 4)
        latent *= tiers
        cache += latent
        breakdown["latents_cache_bytes"] = latent

    if _truthy(config.get("cache_text_encoder_outputs_to_disk")):
        per = 8 * MiB if model_train_type in ANIMA_LORA_TYPES or model_train_type == "anima-finetune" else 2 * MiB
        te = images * per
        cache += te
        breakdown["text_encoder_cache_bytes"] = te

    base_variable = output + cache
    misc = max(512 * MiB, int(base_variable * 0.10))
    breakdown["logs_samples_margin_bytes"] = misc

    return DiskNeedEstimate(
        output_bytes=output + misc,
        cache_bytes=cache,
        logging_bytes=misc if str(config.get("logging_dir") or "").strip() else 0,
        autosave_bytes=AUTOSAVE_MIN_FREE_BYTES,
        breakdown=breakdown,
    )


def skip_disk_preflight() -> bool:
    return str(os.environ.get(SKIP_ENV) or "").strip().lower() in {"1", "true", "yes", "on"}


def check_training_disk_space(
    config: Mapping[str, object],
    model_train_type: str,
    *,
    image_count: int,
    autosave_dir: str | os.PathLike[str] | None = None,
) -> DiskNeedEstimate:
    """Raise DiskSpaceError when any relevant volume is too small."""
    if skip_disk_preflight():
        return DiskNeedEstimate()

    raw = estimate_training_disk_need(config, model_train_type, image_count)
    need = raw.with_safety()

    output_dir = str(config.get("output_dir") or "").strip() or str(Path.cwd() / "output")
    logging_dir = str(config.get("logging_dir") or "").strip()
    train_data_dir = str(config.get("train_data_dir") or "").strip()
    autosave = str(autosave_dir or (Path.cwd() / "config" / "autosave"))

    cache_on_output_volume = bool(
        need.cache_bytes > 0 and train_data_dir and same_volume(train_data_dir, output_dir)
    )
    output_required = need.output_bytes + (need.cache_bytes if cache_on_output_volume else 0)

    checks: list[tuple[str, str, int]] = [
        ("output_dir", output_dir, output_required),
        ("autosave", autosave, need.autosave_bytes),
    ]
    if logging_dir and not same_volume(logging_dir, output_dir):
        checks.append(
            ("logging_dir", logging_dir, max(need.logging_bytes, 512 * MiB) + VOLUME_RESERVE_BYTES)
        )
    if need.cache_bytes > 0 and train_data_dir and not cache_on_output_volume:
        checks.append(("train_data_dir", train_data_dir, need.cache_bytes))

    for label, path, required in checks:
        free = volume_free_bytes(path)
        if free < required:
            raise DiskSpaceError(
                (
                    f"磁盘空间不足：预计约需 {format_bytes(required)}，"
                    f"{label} 所在盘（{path}）仅剩 {format_bytes(free)}。\n"
                    f"请清理旧输出/缓存，或把 output_dir 改到更大磁盘后再开训。"
                    f"（紧急跳过可设 {SKIP_ENV}=1）"
                ),
                required_bytes=required,
                free_bytes=free,
                paths=[path],
                breakdown={**need.breakdown, "failed_on": label},
            )
    return need
