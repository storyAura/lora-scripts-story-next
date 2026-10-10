# Ported from kohya-ss/musubi-tuner (Apache-2.0):
# src/musubi_tuner/krea2/krea2_encoder.py
# Local change: only the Qwen3-VL language tower is built, as transformers' Qwen3Model,
# so the conditioner runs on the pinned transformers 4.51 (the Qwen3-VL classes need
# >= 4.57) and the unused vision tower is never loaded.

"""Krea 2 text encoder: Qwen3-VL-4B conditioner."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from accelerate import init_empty_weights
from torch import Tensor
from transformers import AutoTokenizer, Qwen2TokenizerFast, Qwen3Config, Qwen3Model

from library.safetensors_utils import load_split_weights
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

QWEN3_VL_4B_INSTRUCT_REPO_ID = "Qwen/Qwen3-VL-4B-Instruct"
QWEN3_VL_TOKENIZER_DIRNAME = QWEN3_VL_4B_INSTRUCT_REPO_ID.replace("/", "_")
_TOKENIZER_MARKERS = ("tokenizer.json", "vocab.json")

# text_config of Qwen/Qwen3-VL-4B-Instruct. Text-only input gives the interleaved M-RoPE
# identical t/h/w positions, which is plain 1D RoPE, and the deepstack visual injection
# never fires, so the language tower computes exactly what a Qwen3 model does.
QWEN3_VL_4B_TEXT_CONFIG = {
    "vocab_size": 151936,
    "hidden_size": 2560,
    "intermediate_size": 9728,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "hidden_act": "silu",
    "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-06,
    "rope_theta": 5000000,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "tie_word_embeddings": True,
    "use_sliding_window": False,
    "sliding_window": None,  # Qwen3-VL has none; 4.51 warns on any non-None value
}

# Checkpoint layouts: HF Qwen3VLForConditionalGeneration ("model.language_model."),
# the Krea-2-Raw diffusers text_encoder ("language_model.") and the Comfy single file
# ("model."). The longest prefix has to be tried first.
_LANGUAGE_MODEL_PREFIXES = ("model.language_model.", "language_model.", "model.")
_NON_TEXT_PREFIXES = ("model.visual.", "visual.", "lm_head.")
_QUANTIZED_KEY_SUFFIXES = (".weight_scale", ".scale_weight", ".comfy_quant", ".scaled_fp8")


class Krea2TextEncoderCheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be used as the Krea2 Qwen3-VL text encoder."""


@dataclass
class TextEncoderConfig:
    max_length: int = 512
    select_layers: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
    tokenizer_repo: str = QWEN3_VL_4B_INSTRUCT_REPO_ID


def extract_language_model_state_dict(sd: dict[str, Tensor]) -> dict[str, Tensor]:
    """Keep only the Qwen3-VL language tower, keyed like transformers' Qwen3Model."""
    quantized = [key for key in sd if key == "scaled_fp8" or key.endswith(_QUANTIZED_KEY_SUFFIXES)]
    if quantized:
        raise Krea2TextEncoderCheckpointError(
            f"Krea2 文本编码器不支持 fp8 / 量化版 Qwen3-VL 权重（检测到 {quantized[0]}）。"
            "请改用 bf16 版：Krea-2-Raw 的 text_encoder/model.safetensors，"
            "或 Comfy-Org/Krea-2 的 text_encoders/qwen3vl_4b_bf16.safetensors。"
        )

    converted: dict[str, Tensor] = {}
    for key, value in sd.items():
        if key.startswith(_NON_TEXT_PREFIXES):
            continue
        new_key = key
        for prefix in _LANGUAGE_MODEL_PREFIXES:
            if key.startswith(prefix):
                new_key = key[len(prefix) :]
                break
        converted[new_key] = value
    return converted


def _load_qwen3_vl_text_model(
    model_path: str,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    disable_mmap: bool = True,
) -> Qwen3Model:
    config = Qwen3Config(**QWEN3_VL_4B_TEXT_CONFIG)
    with init_empty_weights():
        model = Qwen3Model._from_config(config)

    logger.info("Loading Krea 2 text encoder (Qwen3-VL language model) weights from %s", model_path)
    sd = load_split_weights(model_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)
    sd = extract_language_model_state_dict(sd)
    info = model.load_state_dict(sd, strict=False, assign=True)
    if info.unexpected_keys or info.missing_keys:
        raise Krea2TextEncoderCheckpointError(
            f"Qwen3-VL text encoder checkpoint did not match the model: "
            f"missing={list(info.missing_keys)[:10]}, unexpected={list(info.unexpected_keys)[:10]}"
        )
    model.to(device)
    if dtype is not None:
        model.to(dtype)
    return model.eval().requires_grad_(False)


def _is_tokenizer_dir(path: Path) -> bool:
    return path.is_dir() and any((path / name).is_file() for name in _TOKENIZER_MARKERS)


def tokenizer_compat_kwargs(source: str) -> dict[str, Any]:
    """from_pretrained overrides that let transformers 4.x read a tokenizer saved by 5.x.

    transformers 5.x writes ``extra_special_tokens`` as a list (4.x calls it
    ``additional_special_tokens``), while 4.x expects a name -> token mapping there and
    fails with ``'list' object has no attribute 'keys'``. The Krea-2-Raw ``tokenizer/``
    folder ships in that form. The file on disk is left untouched.
    """
    config_path = Path(source) / "tokenizer_config.json"
    if not config_path.is_file():
        return {}
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    extra = config.get("extra_special_tokens")
    if not isinstance(extra, list):
        return {}

    additional = list(config.get("additional_special_tokens") or [])
    for token in extra:
        content = token.get("content") if isinstance(token, dict) else token
        if isinstance(content, str) and content not in additional:
            additional.append(content)
    logger.info("%s lists extra_special_tokens (transformers 5.x format); loading them as additional_special_tokens", config_path)
    return {"extra_special_tokens": {}, "additional_special_tokens": additional}


def load_qwen3_vl_tokenizer(tokenizer_class: Any, source: str, **kwargs: Any) -> Any:
    return tokenizer_class.from_pretrained(
        source, local_files_only=Path(source).is_dir(), **tokenizer_compat_kwargs(source), **kwargs
    )


def resolve_qwen3_vl_tokenizer_source(
    tokenizer_repo: str | None = None,
    tokenizer_cache_dir: str | None = None,
    model_path: str | None = None,
) -> str:
    """Prefer a local tokenizer folder so training does not hit huggingface.co."""
    candidates: list[Path] = []
    if tokenizer_repo:
        candidates.append(Path(tokenizer_repo).expanduser())
    if tokenizer_cache_dir:
        cache_root = Path(tokenizer_cache_dir).expanduser()
        candidates.append(cache_root / QWEN3_VL_TOKENIZER_DIRNAME)
        candidates.append(cache_root)
    if model_path:
        te_path = Path(model_path).expanduser()
        te_dir = te_path if te_path.is_dir() else te_path.parent
        candidates.append(te_dir / "qwen3_vl_tokenizer")
        candidates.append(te_dir / QWEN3_VL_TOKENIZER_DIRNAME)
        if te_dir.name == "text_encoder":
            # diffusers layout (krea/Krea-2-Raw): <repo>/text_encoder + <repo>/tokenizer
            candidates.append(te_dir.parent / "tokenizer")

    for path in candidates:
        if _is_tokenizer_dir(path):
            logger.info("Loading Krea 2 tokenizer from local folder %s", path)
            return str(path)

    return tokenizer_repo or QWEN3_VL_4B_INSTRUCT_REPO_ID


def load_qwen3_vl_conditioner(
    model_path: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
    max_length: int = TextEncoderConfig.max_length,
    select_layers: tuple[int, ...] = TextEncoderConfig.select_layers,
    tokenizer_repo: str = QWEN3_VL_4B_INSTRUCT_REPO_ID,
    tokenizer_cache_dir: str | None = None,
    disable_mmap: bool = True,
) -> "Qwen3VLConditioner":
    qwen = _load_qwen3_vl_text_model(model_path, dtype=dtype, device=device, disable_mmap=disable_mmap)
    source = resolve_qwen3_vl_tokenizer_source(
        tokenizer_repo=tokenizer_repo,
        tokenizer_cache_dir=tokenizer_cache_dir,
        model_path=model_path,
    )
    tokenizer = load_qwen3_vl_tokenizer(AutoTokenizer, source, max_length=max_length)
    processor = load_qwen3_vl_tokenizer(Qwen2TokenizerFast, source, max_length=max_length)
    conditioner = Qwen3VLConditioner(qwen, tokenizer, processor, max_length=max_length, select_layers=select_layers)
    return conditioner.eval().requires_grad_(False)


class Qwen3VLConditioner(torch.nn.Module):
    def __init__(
        self,
        qwen,
        tokenizer,
        processor,
        max_length: int = 512,
        select_layers: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35),
    ):
        super().__init__()
        self.qwen = qwen.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.select_layers = select_layers
        self.prompt_template_encode_prefix = (
            "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
            "texture, quantity, text, spatial relationships of the objects and background:"
            "<|im_end|>\n<|im_start|>user\n"
        )
        self.prompt_template_encode_suffix = "<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 34
        self.prompt_template_encode_suffix_start_idx = 5

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(self, text: list[str]) -> tuple[Tensor, Tensor]:
        prefix_idx = self.prompt_template_encode_start_idx
        text = [self.prompt_template_encode_prefix + item for item in text]
        suffix_text = [self.prompt_template_encode_suffix] * len(text)
        suffix_inputs = self.processor(text=suffix_text, return_tensors="pt").to(self.qwen.device, non_blocking=True)
        suffix_ids, suffix_mask = suffix_inputs["input_ids"], suffix_inputs["attention_mask"].bool()

        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                padding="max_length",
                max_length=self.max_length + prefix_idx - self.prompt_template_encode_suffix_start_idx,
                return_tensors="pt",
            ).to(self.qwen.device, non_blocking=True)
            input_ids = torch.cat([inputs["input_ids"], suffix_ids], dim=1)
            mask = torch.cat([inputs["attention_mask"].bool(), suffix_mask], dim=1)
            # The suffix follows the padding, so its rotary positions must continue from the last prompt
            # token (padded positions repeat the previous one). Qwen3-VL derived these from the mask on
            # transformers 4.x; a plain Qwen3 model uses the sequence index unless told otherwise.
            # Same positions as musubi-tuner b3f3d65d.
            position_ids = (mask.long().cumsum(-1) - 1).clamp(min=0)
            states = self.qwen(
                input_ids=input_ids,
                attention_mask=mask.long(),
                position_ids=position_ids,
                output_hidden_states=True,
                use_cache=False,
            )

        hiddens = torch.stack([states.hidden_states[i] for i in self.select_layers], dim=2)
        hiddens = hiddens[:, prefix_idx:]
        mask = mask[:, prefix_idx:]
        return hiddens, mask
