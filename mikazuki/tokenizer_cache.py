"""Bundled SD/SDXL tokenizer cache paths for offline portable training."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

# Matches vendor/sd-scripts/library/strategy_* TOKENIZER ids and
# strategy_base._load_tokenizer local dir naming (repo_id with "/" -> "_").
CLIP_L_TOKENIZER_HF_ID = "openai/clip-vit-large-patch14"
SDXL_TOKENIZER1_HF_ID = CLIP_L_TOKENIZER_HF_ID
SDXL_TOKENIZER2_HF_ID = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
FLUX_T5_TOKENIZER_HF_ID = "google/t5-v1_1-xxl"
QWEN3_VL_TOKENIZER_HF_ID = "Qwen/Qwen3-VL-4B-Instruct"

CLIP_TOKENIZER_FILES = (
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)

T5_TOKENIZER_FILES = (
    "spiece.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
)

T5_OPTIONAL_TOKENIZER_FILES = (
    "tokenizer.json",
)

# Back-compat alias used by prefetch / build scripts.
TOKENIZER_FILES = CLIP_TOKENIZER_FILES

TOKENIZER_FILES_BY_REPO: dict[str, tuple[str, ...]] = {
    CLIP_L_TOKENIZER_HF_ID: CLIP_TOKENIZER_FILES,
    SDXL_TOKENIZER2_HF_ID: CLIP_TOKENIZER_FILES,
    FLUX_T5_TOKENIZER_HF_ID: T5_TOKENIZER_FILES,
}

OPTIONAL_TOKENIZER_FILES_BY_REPO: dict[str, tuple[str, ...]] = {
    FLUX_T5_TOKENIZER_HF_ID: T5_OPTIONAL_TOKENIZER_FILES,
}

# Hugging Face repo id -> local cache folder name under tokenizer-cache/
BUNDLED_TOKENIZER_DIRS: dict[str, str] = {
    repo_id: repo_id.replace("/", "_") for repo_id in TOKENIZER_FILES_BY_REPO
}

# HF repo ids required per train type for offline tokenizer_cache_dir injection.
TOKENIZER_REPOS_BY_TRAIN_TYPE: dict[str, tuple[str, ...]] = {
    "sd-lora": (CLIP_L_TOKENIZER_HF_ID,),
    "sdxl-lora": (CLIP_L_TOKENIZER_HF_ID, SDXL_TOKENIZER2_HF_ID),
    "sdxl-finetune": (CLIP_L_TOKENIZER_HF_ID, SDXL_TOKENIZER2_HF_ID),
    "flux-lora": (CLIP_L_TOKENIZER_HF_ID, FLUX_T5_TOKENIZER_HF_ID),
    "flux-finetune": (CLIP_L_TOKENIZER_HF_ID, FLUX_T5_TOKENIZER_HF_ID),
    "krea2-lora": (QWEN3_VL_TOKENIZER_HF_ID,),
}

from mikazuki.china_hub import HF_TO_MODELSCOPE_REPOS

# ModelScope repo ids used when prefetching on build machines (China-friendly).
MODELSCOPE_TOKENIZER_REPOS: dict[str, str] = HF_TO_MODELSCOPE_REPOS

DEFAULT_TOKENIZER_CACHE_DIRNAME = "tokenizer-cache"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_tokenizer_cache_root(explicit: str | os.PathLike | None = None) -> Path | None:
    """Return the bundled tokenizer-cache directory when it exists."""
    if explicit:
        path = Path(explicit).expanduser()
        return path.resolve() if path.is_dir() else None

    env = (os.environ.get("MIKAZUKI_TOKENIZER_CACHE_DIR") or "").strip()
    if env:
        path = Path(env).expanduser()
        if path.is_dir():
            return path.resolve()

    candidates: list[Path] = [
        _repo_root() / DEFAULT_TOKENIZER_CACHE_DIRNAME,
    ]
    # Portable layout: parent of SD-Trainer/ holds tokenizer-cache next to tagger-models.
    candidates.append(_repo_root().parent / DEFAULT_TOKENIZER_CACHE_DIRNAME)

    for path in candidates:
        if path.is_dir():
            return path.resolve()
    return None


def tokenizer_local_dir(cache_root: Path, hf_repo_id: str) -> Path:
    folder = BUNDLED_TOKENIZER_DIRS.get(hf_repo_id, hf_repo_id.replace("/", "_"))
    return cache_root / folder


def required_tokenizer_files(repo_id: str) -> tuple[str, ...]:
    return TOKENIZER_FILES_BY_REPO.get(repo_id, CLIP_TOKENIZER_FILES)


def is_tokenizer_bundle_complete(
    cache_root: Path,
    hf_repo_ids: Iterable[str] | None = None,
) -> bool:
    ids = list(hf_repo_ids or BUNDLED_TOKENIZER_DIRS.keys())
    for repo_id in ids:
        local_dir = tokenizer_local_dir(cache_root, repo_id)
        if not all((local_dir / name).is_file() for name in required_tokenizer_files(repo_id)):
            return False
    return True


def bundled_tokenizer_cache_dir(
    *,
    explicit: str | os.PathLike | None = None,
    train_type: str | None = None,
    require_sdxl_pair: bool = True,
) -> str | None:
    """Return tokenizer_cache_dir for sd-scripts when the bundled cache is ready."""
    root = resolve_tokenizer_cache_root(explicit)
    if root is None:
        return None
    if train_type and train_type in TOKENIZER_REPOS_BY_TRAIN_TYPE:
        ids = list(TOKENIZER_REPOS_BY_TRAIN_TYPE[train_type])
    elif require_sdxl_pair:
        ids = [CLIP_L_TOKENIZER_HF_ID, SDXL_TOKENIZER2_HF_ID]
    else:
        ids = [CLIP_L_TOKENIZER_HF_ID]
    if not is_tokenizer_bundle_complete(root, ids):
        return None
    return str(root).replace("\\", "/")
