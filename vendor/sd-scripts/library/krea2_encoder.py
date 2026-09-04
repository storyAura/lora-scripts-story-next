# Ported from kohya-ss/musubi-tuner (Apache-2.0):
# src/musubi_tuner/krea2/krea2_encoder.py
# Qwen3-VL is imported lazily so transformers 4.51 still boots Anima.

"""Krea 2 text encoder: Qwen3-VL-4B conditioner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from accelerate import init_empty_weights
from torch import Tensor
from transformers import AutoTokenizer, Qwen2TokenizerFast

from library.safetensors_utils import load_split_weights
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

QWEN3_VL_4B_INSTRUCT_REPO_ID = "Qwen/Qwen3-VL-4B-Instruct"
QWEN3_VL_TOKENIZER_DIRNAME = QWEN3_VL_4B_INSTRUCT_REPO_ID.replace("/", "_")
_TOKENIZER_MARKERS = ("tokenizer.json", "vocab.json")

QWEN3_VL_4B_INSTRUCT_CONFIG = {
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "image_token_id": 151655,
    "model_type": "qwen3_vl",
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "dtype": "bfloat16",
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2560,
        "initializer_range": 0.02,
        "intermediate_size": 9728,
        "max_position_embeddings": 262144,
        "model_type": "qwen3_vl_text",
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20], "rope_type": "default"},
        "rope_theta": 5000000,
        "tie_word_embeddings": True,
        "use_cache": True,
        "vocab_size": 151936,
    },
    "tie_word_embeddings": True,
    "transformers_version": "4.57.0.dev0",
    "video_token_id": 151656,
    "vision_config": {
        "deepstack_visual_indexes": [5, 11, 17],
        "depth": 24,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1024,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "model_type": "qwen3_vl",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 2560,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "vision_end_token_id": 151653,
    "vision_start_token_id": 151652,
}


class Krea2TextEncoderUnavailableError(RuntimeError):
    """Raised when this venv's transformers cannot build Qwen3-VL."""


def require_qwen3_vl():
    try:
        from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    except ImportError as error:
        raise Krea2TextEncoderUnavailableError(
            "Krea2 文本编码器需要带 Qwen3-VL 的 transformers（通常 >= 4.57）。"
            "当前环境无法 import Qwen3VLForConditionalGeneration。"
            "请先缓存文本输出，或在确认 Anima 不受影响后再升级 transformers。"
        ) from error
    return Qwen3VLConfig, Qwen3VLForConditionalGeneration


@dataclass
class TextEncoderConfig:
    max_length: int = 512
    select_layers: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
    tokenizer_repo: str = QWEN3_VL_4B_INSTRUCT_REPO_ID


def _convert_comfyui_qwen3vl_state_dict(sd: dict[str, Tensor]) -> dict[str, Tensor]:
    converted: dict[str, Tensor] = {}
    for key, value in sd.items():
        if key.startswith("model.language_model.") or key.startswith("model.visual."):
            new_key = key
        elif key.startswith("visual."):
            new_key = "model.visual." + key[len("visual.") :]
        elif key.startswith("language_model."):
            new_key = "model." + key
        elif key.startswith("model."):
            new_key = "model.language_model." + key[len("model.") :]
        else:
            new_key = key
        converted[new_key] = value
    return converted


def _load_qwen3_vl_model(
    model_path: str,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    disable_mmap: bool = True,
):
    Qwen3VLConfig, Qwen3VLForConditionalGeneration = require_qwen3_vl()
    config = Qwen3VLConfig.from_dict(QWEN3_VL_4B_INSTRUCT_CONFIG)
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration._from_config(config)

    logger.info("Loading Krea 2 text encoder (Qwen3-VL) weights from %s", model_path)
    sd = load_split_weights(model_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)
    sd = _convert_comfyui_qwen3vl_state_dict(sd)
    info = model.load_state_dict(sd, strict=False, assign=True)
    model.tie_weights()
    unexpected = list(info.unexpected_keys)
    missing = [k for k in info.missing_keys if k != "lm_head.weight"]
    if unexpected or missing:
        raise RuntimeError(
            f"Qwen3-VL text encoder checkpoint did not match the model: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    model.to(device)
    if dtype is not None:
        model.to(dtype)
    return model.eval().requires_grad_(False)


def _is_tokenizer_dir(path: Path) -> bool:
    return path.is_dir() and any((path / name).is_file() for name in _TOKENIZER_MARKERS)


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
    qwen = _load_qwen3_vl_model(model_path, dtype=dtype, device=device, disable_mmap=disable_mmap)
    source = resolve_qwen3_vl_tokenizer_source(
        tokenizer_repo=tokenizer_repo,
        tokenizer_cache_dir=tokenizer_cache_dir,
        model_path=model_path,
    )
    local_only = Path(source).is_dir()
    tokenizer = AutoTokenizer.from_pretrained(
        source, max_length=max_length, local_files_only=local_only
    )
    processor = Qwen2TokenizerFast.from_pretrained(
        source, max_length=max_length, local_files_only=local_only
    )
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
            states = self.qwen(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)

        hiddens = torch.stack([states.hidden_states[i] for i in self.select_layers], dim=2)
        hiddens = hiddens[:, prefix_idx:]
        mask = mask[:, prefix_idx:]
        return hiddens, mask
