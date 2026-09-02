# Ported from kohya-ss/musubi-tuner (Apache-2.0):
# src/musubi_tuner/krea2/krea2_utils.py

"""Shared loaders / helpers for Krea 2."""

from __future__ import annotations

from typing import Optional, Union

import torch

from library.fp8_optimization_utils import apply_fp8_monkey_patch
from library.krea2_encoder import (
    QWEN3_VL_4B_INSTRUCT_REPO_ID,
    Qwen3VLConditioner,
    TextEncoderConfig,
    load_qwen3_vl_conditioner,
)
from library.krea2_models import SingleMMDiTConfig, SingleStreamDiT
from library.lora_utils import load_safetensors_with_lora_and_fp8
from library.safetensors_utils import load_safetensors
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

MODEL_VERSION_KREA2 = "krea2"
KREA2_FP8_OPTIMIZATION_TARGET_KEYS = ["blocks."]
KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS = ["mod.", "norm", "txtfusion"]

single_mmdit_large_wide = SingleMMDiTConfig(
    features=6144,
    tdim=256,
    txtdim=2560,
    heads=48,
    kvheads=12,
    multiplier=4,
    layers=28,
    patch=2,
    channels=16,
    txtheads=20,
    txtkvheads=20,
    txtlayers=12,
)


def load_krea2_dit(
    dit_path: str,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    config: SingleMMDiTConfig = single_mmdit_large_wide,
    fp8_scaled: bool = False,
    loading_device: Optional[Union[str, torch.device]] = None,
    attn_mode: str = "torch",
    split_attn: bool = False,
    lora_weights: Optional[list] = None,
    lora_multipliers: Optional[list] = None,
) -> SingleStreamDiT:
    device = torch.device(device)
    loading_device = device if loading_device is None else torch.device(loading_device)
    has_lora = lora_weights is not None and len(lora_weights) > 0

    logger.info(
        "Loading Krea 2 DiT weights from %s%s%s",
        dit_path,
        " (fp8 scaled)" if fp8_scaled else "",
        f" (+{len(lora_weights)} LoRA merged)" if has_lora else "",
    )
    with torch.device("meta"):
        dit = SingleStreamDiT(config, attn_mode=attn_mode, split_attn=split_attn)

    if fp8_scaled or has_lora:
        sd = load_safetensors_with_lora_and_fp8(
            model_files=dit_path,
            lora_weights_list=lora_weights,
            lora_multipliers=lora_multipliers,
            fp8_optimization=fp8_scaled,
            calc_device=device,
            move_to_device=(loading_device == device),
            dit_weight_dtype=None if fp8_scaled else dtype,
            target_keys=KREA2_FP8_OPTIMIZATION_TARGET_KEYS if fp8_scaled else None,
            exclude_keys=KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS if fp8_scaled else None,
        )
        if fp8_scaled:
            apply_fp8_monkey_patch(dit, sd, use_scaled_mm=False)
        if loading_device.type != "cpu":
            for key in sd.keys():
                sd[key] = sd[key].to(loading_device)
        dit.load_state_dict(sd, strict=True, assign=True)
    else:
        sd = load_safetensors(dit_path, device=loading_device, disable_mmap=True, dtype=dtype)
        dit.load_state_dict(sd, strict=True, assign=True)
    return dit


def load_krea2_text_encoder(
    path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: Union[str, torch.device] = "cpu",
    max_length: int = TextEncoderConfig.max_length,
    select_layers: tuple = TextEncoderConfig.select_layers,
    tokenizer_repo: str = QWEN3_VL_4B_INSTRUCT_REPO_ID,
) -> Qwen3VLConditioner:
    return load_qwen3_vl_conditioner(
        path,
        dtype=dtype,
        device=device,
        max_length=max_length,
        select_layers=select_layers,
        tokenizer_repo=tokenizer_repo,
    )


@torch.no_grad()
def get_krea2_prompt_embeds(encoder: Qwen3VLConditioner, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    hiddens, mask = encoder(prompts)
    return hiddens, mask.to(dtype=torch.bool)
