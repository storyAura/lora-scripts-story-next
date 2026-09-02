"""Fold Krea2 LoRA / LoKr UI fields into sd-scripts network_args."""

from __future__ import annotations

from pathlib import Path

from mikazuki.anima_backend.adapter import (
    LOKR_FULL_MATRIX_GUARD_WARNING,
    LOKR_FULL_MATRIX_WARNING,
    LYCORIS_NETWORK_ARG_MAP,
    _is_empty_value,
    _network_args_has_truthy_arg,
    _normalize_network_args,
)
from mikazuki.utils.config_import import (
    KREA2_LORA_TYPE_BRANCH_CONSTS,
    _apply_krea2_lora_type_consts,
)

KREA2_LYCORIS_PRESET = (
    Path(__file__).resolve().parents[2] / "config" / "lycoris_krea2_preset.toml"
)

_KREA2_GUI_ONLY_KEYS = (
    "lora_type",
    "lycoris_algo",
    "lokr_factor",
    "full_matrix",
    "use_cp",
    "use_scalar",
    "decompose_both",
    "bypass_mode",
    "dora_wd",
    "conv_dim",
    "conv_alpha",
    "dropout",
    "train_norm",
    "rank_dropout",
    "module_dropout",
    "rank_dropout_scale",
)

_LYCORIS_ONLY_ARG_KEYS = frozenset({
    "algo",
    "factor",
    "preset",
    "full_matrix",
    "use_cp",
    "use_scalar",
    "decompose_both",
    "bypass_mode",
    "dora_wd",
    "conv_dim",
    "conv_alpha",
    "dropout",
    "train_norm",
    "rank_dropout",
    "module_dropout",
    "rank_dropout_scale",
    "train_llm_adapter",
})


def apply_krea2_network_configuration(config: dict) -> list[str]:
    """Force LoRA/LoKr modules and fold LyCORIS UI fields into network_args."""
    warnings: list[str] = []
    _apply_krea2_lora_type_consts(config)
    lora_type = str(config.get("lora_type") or "lora")

    if lora_type == "lokr":
        network_args = list(config.get("network_args") or [])
        has_preset = any(
            isinstance(item, str) and item.strip().startswith("preset=")
            for item in network_args
        )
        if not has_preset:
            network_args.append(f"preset={KREA2_LYCORIS_PRESET.as_posix()}")
        for ui_field, arg_key in LYCORIS_NETWORK_ARG_MAP.items():
            if ui_field == "train_llm_adapter":
                continue
            value = config.get(ui_field)
            if _is_empty_value(value):
                continue
            network_args.append(f"{arg_key}={value}")
        network_args = _normalize_network_args(network_args)
        if _network_args_has_truthy_arg(network_args, "full_matrix"):
            if _is_empty_value(config.get("scale_weight_norms")):
                config["scale_weight_norms"] = 1.0
                warnings.append(LOKR_FULL_MATRIX_GUARD_WARNING)
            else:
                warnings.append(LOKR_FULL_MATRIX_WARNING)
        config["network_args"] = network_args
    else:
        network_args = []
        for item in config.get("network_args") or []:
            if not isinstance(item, str) or "=" not in item:
                continue
            key = item.split("=", 1)[0].strip().lower()
            if key in _LYCORIS_ONLY_ARG_KEYS:
                continue
            network_args.append(item)
        network_args = _normalize_network_args(network_args)
        if network_args:
            config["network_args"] = network_args
        else:
            config.pop("network_args", None)

    for key in _KREA2_GUI_ONLY_KEYS:
        config.pop(key, None)
    return warnings


def apply_krea2_network_configuration_if_needed(config: dict, model_train_type: str) -> list[str]:
    if str(model_train_type or "").strip() != "krea2-lora":
        return []
    return apply_krea2_network_configuration(config)
