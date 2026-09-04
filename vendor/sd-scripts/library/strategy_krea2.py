from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import AutoTokenizer

from library import krea2_encoder, train_util
from library.strategy_base import (
    LatentsCachingStrategy,
    TextEncoderOutputsCachingStrategy,
    TextEncodingStrategy,
    TokenizeStrategy,
)
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class Krea2TokenizeStrategy(TokenizeStrategy):
    def __init__(self, tokenizer_repo: Optional[str] = None, tokenizer_cache_dir: Optional[str] = None) -> None:
        repo = tokenizer_repo or krea2_encoder.QWEN3_VL_4B_INSTRUCT_REPO_ID
        self.tokenizer = self._load_tokenizer(AutoTokenizer, repo, tokenizer_cache_dir=tokenizer_cache_dir)

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        captions = [text] if isinstance(text, str) else list(text)
        encoded = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=krea2_encoder.TextEncoderConfig.max_length,
            return_tensors="pt",
        )
        return [encoded["input_ids"], encoded["attention_mask"]]


class Krea2TextEncodingStrategy(TextEncodingStrategy):
    def encode_tokens(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], tokens: List[Any]
    ) -> List[torch.Tensor]:
        input_ids = tokens[0]
        tokenizer = tokenize_strategy.tokenizer
        captions = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        encoder = models[0]
        hiddens, mask = encoder(captions)
        return [hiddens, mask.to(dtype=torch.bool)]


class Krea2TextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    NPZ_SUFFIX = "_krea2_te.npz"

    def __init__(
        self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool, is_partial: bool = False
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial)

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return os.path.splitext(image_abs_path)[0] + self.NPZ_SUFFIX

    def is_disk_cached_outputs_expected(self, npz_path: str):
        if not self.cache_to_disk or not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True
        try:
            npz = np.load(npz_path)
            return "hiddens" in npz and "mask" in npz
        except Exception as error:
            logger.error("Error loading file: %s", npz_path)
            raise error

    def load_outputs_npz(self, npz_path: str) -> List[np.ndarray]:
        data = np.load(npz_path)
        return [data["hiddens"], data["mask"]]

    def cache_batch_outputs(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], text_encoding_strategy: TextEncodingStrategy, infos: List
    ):
        captions = [info.caption for info in infos]
        tokens = tokenize_strategy.tokenize(captions)
        with torch.no_grad():
            hiddens, mask = text_encoding_strategy.encode_tokens(tokenize_strategy, models, tokens)
        if hiddens.dtype == torch.bfloat16:
            hiddens = hiddens.float()
        hiddens_np = hiddens.cpu().numpy()
        mask_np = mask.cpu().numpy()
        for i, info in enumerate(infos):
            if self.cache_to_disk:
                np.savez(info.text_encoder_outputs_npz, hiddens=hiddens_np[i], mask=mask_np[i])
            else:
                info.text_encoder_outputs = (hiddens_np[i], mask_np[i])


# Qwen-Image VAE spatial_compression_ratio = 2 ** len(temperal_downsample) = 8.
# Cache keys are VAE latent HxW (e.g. 832x1216 → latents_104x152). Using Hunyuan's
# stride 32 looked for latents_26x38 and crashed on the first DataLoader batch.
KREA2_LATENTS_STRIDE = 8


class Krea2LatentsCachingStrategy(LatentsCachingStrategy):
    NPZ_SUFFIX = "_krea2.npz"

    def __init__(self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)

    @property
    def cache_suffix(self) -> str:
        return self.NPZ_SUFFIX

    def get_latents_npz_path(self, absolute_path: str, image_size: Tuple[int, int]) -> str:
        return os.path.splitext(absolute_path)[0] + f"_{image_size[0]:04d}x{image_size[1]:04d}" + self.NPZ_SUFFIX

    def is_disk_cached_latents_expected(self, bucket_reso: Tuple[int, int], npz_path: str, flip_aug: bool, alpha_mask: bool):
        return self._default_is_disk_cached_latents_expected(
            KREA2_LATENTS_STRIDE, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True
        )

    def load_latents_from_disk(
        self, npz_path: str, bucket_reso: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray]]:
        return self._default_load_latents_from_disk(KREA2_LATENTS_STRIDE, npz_path, bucket_reso)

    def cache_batch_latents(self, vae, image_infos: List, flip_aug: bool, alpha_mask: bool, random_crop: bool):
        def encode_by_vae(img_tensor):
            with torch.autocast(device_type=vae.device.type, dtype=vae.dtype):
                return vae.encode_pixels_to_latents(img_tensor)

        self._default_cache_batch_latents(
            encode_by_vae, vae.device, vae.dtype, image_infos, flip_aug, alpha_mask, random_crop, multi_resolution=True
        )
        if not train_util.HIGH_VRAM:
            train_util.clean_memory_on_device(vae.device)
