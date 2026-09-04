# -*- coding: utf-8 -*-
"""Helpers for Windows portable (embedded) Python — flash-attn/triton compatibility."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Callable, Dict, MutableMapping, Optional


def default_pytorch_alloc_conf() -> str:
    # expandable_segments is native-allocator only and ignored on Windows.
    # cudaMallocAsync needs CUDA 11.4+ and is the Linux/AutoDL default.
    if sys.platform == "win32":
        return "garbage_collection_threshold:0.8,max_split_size_mb:512"
    return "backend:cudaMallocAsync,expandable_segments:True"


def apply_pytorch_allocator_env(env: MutableMapping[str, str]) -> None:
    """Set both allocator aliases. A user-set either key wins; the other is synced."""
    alloc = env.get("PYTORCH_ALLOC_CONF")
    cuda_alloc = env.get("PYTORCH_CUDA_ALLOC_CONF")
    if alloc and cuda_alloc:
        return
    if alloc:
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", alloc)
        return
    if cuda_alloc:
        env.setdefault("PYTORCH_ALLOC_CONF", cuda_alloc)
        return
    value = default_pytorch_alloc_conf()
    env["PYTORCH_ALLOC_CONF"] = value
    env["PYTORCH_CUDA_ALLOC_CONF"] = value


def is_embedded_python(executable: Optional[str] = None) -> bool:
    exe = (executable or sys.executable).replace("\\", "/").lower()
    return "python_embeded" in exe or "python_embedded" in exe


def flash_attn_stack_usable() -> bool:
    """True when flash-attn and its triton ops import cleanly in the current interpreter."""
    try:
        import triton  # noqa: F401
        import flash_attn  # noqa: F401
        from flash_attn.ops.triton.rotary import apply_rotary  # noqa: F401
        return True
    except Exception:
        return False


def sanitize_embedded_deps(log: Optional[Callable[[str], None]] = None) -> None:
    """Remove flash-attn / triton from embedded Python if the stack cannot run."""
    if not is_embedded_python():
        return

    import importlib.util

    has_flash = importlib.util.find_spec("flash_attn") is not None
    has_triton = importlib.util.find_spec("triton") is not None
    if not has_flash and not has_triton:
        return
    if has_flash and flash_attn_stack_usable():
        return

    msg = (
        "Portable package: removing incompatible flash-attn/triton "
        "(training will use xformers or PyTorch SDPA)."
    )
    if log:
        log(msg)
    else:
        print(msg)

    subprocess.run(
        [
            sys.executable,
            "-s",
            "-m",
            "pip",
            "uninstall",
            "flash-attn",
            "flash_attn",
            "triton-windows",
            "triton",
            "-y",
        ],
        capture_output=True,
        timeout=120,
    )


def train_env_overrides() -> Dict[str, str]:
    """Environment for training subprocesses on embedded Python."""
    if not is_embedded_python():
        return {}

    overrides: Dict[str, str] = {"XFORMERS_FORCE_DISABLE_TRITON": "1"}
    if not flash_attn_stack_usable():
        overrides["TRANSFORMERS_ATTN_IMPLEMENTATION"] = "sdpa"
    return overrides
