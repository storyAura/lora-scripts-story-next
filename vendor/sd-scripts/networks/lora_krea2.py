# networks.lora_krea2 — wrap every Linear on SingleStreamDiT (musubi default, ~264 modules).
# Reuses the Anima LoRA walker with a K2 target class.

from typing import Dict, Optional

import torch
from torch import nn

from networks import lora_anima


class LoRANetwork(lora_anima.LoRANetwork):
    ANIMA_TARGET_REPLACE_MODULE = ["SingleStreamDiT"]
    ANIMA_ADAPTER_TARGET_REPLACE_MODULE = []
    LORA_PREFIX_ANIMA = "lora_unet"


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae,
    text_encoders: list,
    unet,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    kwargs = dict(kwargs)
    kwargs["_network_factory"] = LoRANetwork
    kwargs["train_llm_adapter"] = "false"
    return lora_anima.create_network(
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )


def create_network_from_weights(
    multiplier,
    file,
    ae,
    text_encoders,
    unet,
    weights_sd: Optional[Dict[str, torch.Tensor]] = None,
    for_inference: bool = False,
    **kwargs,
):
    kwargs = dict(kwargs)
    kwargs["_network_factory"] = LoRANetwork
    kwargs["train_llm_adapter"] = "false"
    return lora_anima.create_network_from_weights(
        multiplier,
        file,
        ae,
        text_encoders,
        unet,
        weights_sd=weights_sd,
        for_inference=for_inference,
        **kwargs,
    )
