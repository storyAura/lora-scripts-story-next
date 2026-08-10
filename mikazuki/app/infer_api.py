"""Quick Anima LoRA inference panel API (mounted under /api/infer)."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

from mikazuki.app.models import APIResponseFail, APIResponseSuccess
from mikazuki.log import log
from mikazuki.process import train_env_overrides
from mikazuki.tasks import tm
from mikazuki.train_queue import train_queue

router = APIRouter(prefix="/infer")

_MAX_RECENT_LORAS = 40
_INFER_KIND = "infer"
_SAFE_NAME = re.compile(r"^[\w.\-]+$")

_ANIMA_DEFAULTS = {
    "dit": "./sd-models/anima/anima-base-v1.0.safetensors",
    "vae": "./sd-models/anima/qwen_image_vae.safetensors",
    "text_encoder": "./sd-models/anima/qwen_3_06b_base.safetensors",
}

_FAMILY_LABELS = {
    "anima": "Anima",
    "flux": "Flux",
    "sdxl": "SDXL",
    "sd15": "SD 1.5",
    "sd3": "SD3",
    "unknown": "未知",
}


def _project_root() -> Path:
    return Path.cwd()


def _infer_root() -> Path:
    return _project_root() / "output" / "infer"


def _read_safetensors_metadata(path: Path) -> dict[str, str]:
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as handle:
            meta = handle.metadata() or {}
        return {str(k): str(v) for k, v in meta.items()}
    except Exception:  # noqa: BLE001
        return {}


def _detect_model_family(meta: dict[str, str], lora_path: Optional[Path] = None) -> str:
    blob = " ".join(
        [
            meta.get("ss_base_model_version", ""),
            meta.get("ss_sd_model_name", ""),
            meta.get("ss_network_module", ""),
            meta.get("modelspec.architecture", ""),
            meta.get("modelspec.title", ""),
            (lora_path.name if lora_path is not None else ""),
        ]
    ).lower()
    if any(token in blob for token in ("anima", "hunyuan_image", "lora_anima", "tlora_anima", "qwen_image")):
        return "anima"
    if "flux" in blob:
        return "flux"
    if "sdxl" in blob or "xl_base" in blob:
        return "sdxl"
    if "sd3" in blob or "stable_diffusion_3" in blob:
        return "sd3"
    if any(token in blob for token in ("sd15", "sd1.", "v1-5", "v1_5", "stable_diffusion_1")):
        return "sd15"
    # Anima native modules even without base_model_version
    module = meta.get("ss_network_module", "").lower()
    if module.startswith("networks.") and module.endswith("_anima"):
        return "anima"
    return "unknown"


def _find_by_basename(name: str) -> Optional[str]:
    raw = (name or "").strip()
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    basename = candidate.name
    roots = [
        _project_root() / "sd-models",
        _project_root() / "models",
    ]
    for root in roots:
        if not root.is_dir():
            continue
        try:
            hits = sorted(root.rglob(basename), key=lambda p: len(str(p)))
        except OSError:
            continue
        for hit in hits:
            if hit.is_file():
                try:
                    return str(hit.resolve())
                except OSError:
                    return str(hit)
    # Prefer project-relative default if it exists
    rel = _project_root() / raw.replace("\\", "/").lstrip("./")
    if rel.is_file():
        return str(rel.resolve())
    return None


def _suggest_base_paths(meta: dict[str, str], family: str) -> dict[str, Any]:
    dit_name = meta.get("ss_sd_model_name") or ""
    vae_name = meta.get("ss_vae_name") or ""
    te_name = meta.get("ss_qwen3_name") or ""

    dit = _find_by_basename(dit_name)
    vae = _find_by_basename(vae_name)
    text_encoder = _find_by_basename(te_name)

    notes: list[str] = []
    if family == "anima":
        if not dit:
            dit = _find_by_basename(_ANIMA_DEFAULTS["dit"]) or _ANIMA_DEFAULTS["dit"]
            if dit_name:
                notes.append(f"未在 sd-models/ 找到底模「{dit_name}」，已回退默认 Anima DiT")
            else:
                notes.append("元数据无底模名，已填入默认 Anima DiT")
        if not vae:
            vae = _find_by_basename(_ANIMA_DEFAULTS["vae"]) or _ANIMA_DEFAULTS["vae"]
            if vae_name:
                notes.append(f"未找到 VAE「{vae_name}」，已回退默认")
        if not text_encoder:
            text_encoder = _find_by_basename(_ANIMA_DEFAULTS["text_encoder"]) or _ANIMA_DEFAULTS["text_encoder"]
            if te_name:
                notes.append(f"未找到文本编码器「{te_name}」，已回退默认")
    else:
        if not dit and dit_name:
            notes.append(f"未在本地找到底模文件：{dit_name}")
        if not vae and vae_name:
            notes.append(f"未在本地找到 VAE：{vae_name}")

    return {
        "dit": dit,
        "vae": vae,
        "text_encoder": text_encoder,
        "dit_name": dit_name or None,
        "vae_name": vae_name or None,
        "text_encoder_name": te_name or None,
        "notes": notes,
    }


def _lora_info(lora_path: Path) -> dict[str, Any]:
    meta = _read_safetensors_metadata(lora_path)
    family = _detect_model_family(meta, lora_path)
    suggested = _suggest_base_paths(meta, family)
    module = meta.get("ss_network_module") or ""
    algo = ""
    try:
        args_raw = meta.get("ss_network_args") or ""
        if args_raw:
            parsed = json.loads(args_raw)
            if isinstance(parsed, dict):
                algo = str(parsed.get("algo") or "")
    except (json.JSONDecodeError, TypeError):
        pass
    lycoris = "lycoris" in module.lower() or bool(algo)
    supported = family == "anima"
    return {
        "path": str(lora_path.resolve()),
        "name": lora_path.name,
        "family": family,
        "family_label": _FAMILY_LABELS.get(family, family),
        "supported": supported,
        "lycoris": lycoris,
        "network_module": module or None,
        "network_algo": algo or None,
        "ss_sd_model_name": meta.get("ss_sd_model_name"),
        "ss_vae_name": meta.get("ss_vae_name"),
        "ss_qwen3_name": meta.get("ss_qwen3_name"),
        "ss_base_model_version": meta.get("ss_base_model_version"),
        "has_training_metadata": bool(meta.get("ss_network_module") or meta.get("ss_sd_model_name")),
        "suggested": suggested,
        "warning": None
        if supported
        else f"快速推理目前仅支持 Anima LoRA，检测到类型为 {_FAMILY_LABELS.get(family, family)}。",
    }


def _json_body_sync(raw: bytes) -> dict:
    try:
        payload = json.loads(raw or b"{}")
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _task_meta(task) -> dict:
    meta = getattr(task, "metadata", None)
    return meta if isinstance(meta, dict) else {}


def _task_occupying(task) -> bool:
    try:
        if callable(getattr(task, "occupies_slot", None)) and task.occupies_slot():
            return True
        if callable(getattr(task, "is_process_alive", None)) and task.is_process_alive():
            return True
    except Exception:  # noqa: BLE001
        return False
    return False


def _infer_task() -> Any:
    for task in tm.tasks.values():
        if not _task_occupying(task):
            continue
        if _task_meta(task).get("kind") == _INFER_KIND:
            return task
    return None


def _training_busy() -> bool:
    try:
        if any(e.get("status") == "running" for e in train_queue.entries):
            return True
    except Exception:  # noqa: BLE001
        pass
    for task in tm.tasks.values():
        if not _task_occupying(task):
            continue
        if _task_meta(task).get("kind") != _INFER_KIND:
            return True
    return False


def _list_recent_loras(limit: int = _MAX_RECENT_LORAS) -> list[dict]:
    roots = [
        _project_root() / "output",
    ]
    found: list[tuple[float, Path]] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.safetensors"):
            try:
                # skip infer outputs / obvious non-LoRA dumps
                rel = path.relative_to(_project_root()).as_posix().lower()
                if "/infer/" in f"/{rel}":
                    continue
                mtime = path.stat().st_mtime
            except OSError:
                continue
            found.append((mtime, path))
    found.sort(key=lambda item: item[0], reverse=True)
    out: list[dict] = []
    for mtime, path in found[: max(1, limit)]:
        try:
            exists = path.is_file()
            size = path.stat().st_size if exists else 0
        except OSError:
            exists = False
            size = 0
        out.append(
            {
                "path": str(path.resolve()) if exists else str(path),
                "name": path.name,
                "mtime": datetime.fromtimestamp(mtime).isoformat(timespec="seconds"),
                "size_bytes": size,
                "missing": not exists,
            }
        )
    return out


def _detect_lycoris(lora_path: Path) -> bool:
    info = _lora_info(lora_path) if lora_path.is_file() else {}
    return bool(info.get("lycoris"))


def _build_infer_command(payload: dict, out_dir: Path) -> tuple[list[str], dict[str, str], dict[str, Any]]:
    lora = Path(str(payload.get("lora_path") or "").strip()).expanduser()
    dit = Path(str(payload.get("dit") or payload.get("pretrained_model_name_or_path") or "").strip()).expanduser()
    vae = Path(str(payload.get("vae") or "").strip()).expanduser()
    text_encoder = Path(str(payload.get("text_encoder") or payload.get("qwen3") or "").strip()).expanduser()

    if not str(lora):
        raise ValueError("请选择 LoRA 权重路径")
    if not lora.is_file():
        raise FileNotFoundError(f"LoRA 文件不存在或已被删除：{lora}")
    if not dit.exists():
        raise FileNotFoundError(f"DiT / 底模路径不存在：{dit}")
    if not vae.exists():
        raise FileNotFoundError(f"VAE 路径不存在：{vae}")
    if not text_encoder.exists():
        raise FileNotFoundError(f"文本编码器（qwen3）路径不存在：{text_encoder}")

    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("请填写正面提示词")

    negative = str(payload.get("negative_prompt") or "").strip()
    width = int(payload.get("width") or 1024)
    height = int(payload.get("height") or 1024)
    steps = int(payload.get("steps") or 40)
    cfg = float(payload.get("cfg") or payload.get("guidance_scale") or 4.5)
    seed = payload.get("seed")
    flow_shift = float(payload.get("flow_shift") or 5.0)
    scheduler = str(payload.get("scheduler") or "simple").strip().lower() or "simple"
    sampler = str(payload.get("sampler") or "euler").strip().lower() or "euler"
    attn = str(payload.get("attn_mode") or "torch").strip() or "torch"
    multiplier = float(payload.get("lora_multiplier") or 1.0)
    lycoris = bool(payload.get("lycoris"))
    if payload.get("lycoris") is None:
        lycoris = _detect_lycoris(lora)

    script = _project_root() / "vendor" / "sd-scripts" / "anima_minimal_inference.py"
    if not script.is_file():
        raise FileNotFoundError(f"推理脚本缺失：{script}")

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(script),
        "--dit",
        str(dit),
        "--vae",
        str(vae),
        "--text_encoder",
        str(text_encoder),
        "--lora_weight",
        str(lora),
        "--lora_multiplier",
        str(multiplier),
        "--prompt",
        prompt,
        "--negative_prompt",
        negative,
        "--image_size",
        str(height),
        str(width),
        "--infer_steps",
        str(steps),
        "--guidance_scale",
        str(cfg),
        "--flow_shift",
        str(flow_shift),
        "--scheduler",
        scheduler,
        "--sampler",
        sampler,
        "--attn_mode",
        attn,
        "--save_path",
        str(out_dir),
        "--output_type",
        "images",
    ]
    if seed is not None and str(seed).strip() != "":
        cmd.extend(["--seed", str(int(seed))])
    if lycoris:
        cmd.append("--lycoris")
    if payload.get("fp8"):
        cmd.append("--fp8")
    if payload.get("text_encoder_cpu"):
        cmd.append("--text_encoder_cpu")

    env = os.environ.copy()
    env.update(train_env_overrides())
    project_root = str(_project_root())
    vendor = str(_project_root() / "vendor" / "sd-scripts")
    path_parts = [project_root, vendor]
    existing = env.get("PYTHONPATH", "")
    for part in existing.split(os.pathsep):
        if part and part not in path_parts:
            path_parts.append(part)
    env["PYTHONPATH"] = os.pathsep.join(path_parts)
    env["PYTHONUNBUFFERED"] = "1"

    meta = {
        "kind": _INFER_KIND,
        "lora_path": str(lora.resolve()),
        "output_dir": str(out_dir.resolve()),
        "command": cmd,
        "prompt": prompt,
    }
    return cmd, env, meta


@router.get("/status")
async def infer_status():
    infer = _infer_task()
    infer_meta = _task_meta(infer) if infer else {}
    return APIResponseSuccess(
        message=None,
        data={
            "busy_training": _training_busy(),
            "busy_infer": infer is not None,
            "task_id": getattr(infer, "task_id", None),
            "task_status": getattr(getattr(infer, "status", None), "name", None),
            "output_dir": infer_meta.get("output_dir"),
            "recent_loras": _list_recent_loras(),
            "defaults": {
                "dit": "./sd-models/anima/anima-base-v1.0.safetensors",
                "vae": "./sd-models/anima/qwen_image_vae.safetensors",
                "text_encoder": "./sd-models/anima/qwen_3_06b_base.safetensors",
                "width": 1024,
                "height": 1024,
                "steps": 40,
                "cfg": 4.5,
                "flow_shift": 5.0,
                "scheduler": "simple",
                "sampler": "euler",
            },
        },
    )


@router.get("/loras")
async def infer_loras():
    return APIResponseSuccess(message=None, data={"loras": _list_recent_loras()})


@router.get("/lora-info")
async def infer_lora_info(path: str = ""):
    lora = Path(str(path or "").strip()).expanduser()
    if not str(lora):
        return APIResponseFail(message="请提供 LoRA 路径", data={"field": "path"})
    if not lora.is_file():
        return APIResponseFail(message=f"LoRA 文件不存在：{lora}", data={"field": "path"})
    return APIResponseSuccess(message=None, data=_lora_info(lora))


@router.post("/run")
async def infer_run(request: Request):
    if _training_busy():
        return APIResponseFail(
            message="训练正在占用 GPU，快速推理已禁用。请等训练结束或终止训练后再试。",
            data={"field": "gpu_busy", "busy_training": True},
        )
    if _infer_task() is not None:
        return APIResponseFail(
            message="已有快速推理任务在运行，请等待完成或先终止。",
            data={"field": "infer_busy", "busy_infer": True},
        )

    payload = _json_body_sync(await request.body())
    lora_check = Path(str(payload.get("lora_path") or "").strip()).expanduser()
    if lora_check.is_file():
        info = _lora_info(lora_check)
        if not info.get("supported"):
            return APIResponseFail(
                message=info.get("warning") or "快速推理仅支持 Anima LoRA",
                data={"field": "family", "family": info.get("family"), "lora_info": info},
            )
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out_dir = _infer_root() / stamp
    try:
        cmd, env, meta = _build_infer_command(payload, out_dir)
    except FileNotFoundError as exc:
        return APIResponseFail(message=str(exc), data={"field": "path"})
    except (TypeError, ValueError) as exc:
        return APIResponseFail(message=str(exc), data={"field": "params"})

    task = tm.create_task(cmd, env, metadata=meta, cwd=str(_project_root()))
    if task is None:
        return APIResponseFail(
            message="无法创建推理任务：GPU 任务槽已被占用。",
            data={"field": "gpu_busy"},
        )

    def _run():
        try:
            task.execute()
            task.wait()
            rc = task.process.returncode if task.process else -1
            if rc != 0:
                log.error(f"Infer failed (exit {rc})")
                # Prefer a short OOM hint when present in the log tail.
                try:
                    from mikazuki.train_log_hub import hub

                    snap = hub.snapshot(task.task_id, from_line=0)
                    text = "\n".join(snap.get("lines") or [])[-4000:]
                    if "out of memory" in text.lower() or "cuda oom" in text.lower():
                        task.metadata = {**_task_meta(task), "error": "CUDA OOM：显存不足，请降低分辨率/步数或关闭其他占显存进程"}
                except Exception:  # noqa: BLE001
                    pass
            else:
                log.info(f"Infer finished: {out_dir}")
        except Exception as exc:  # noqa: BLE001
            log.error(f"Infer crashed: {exc}")
            task.metadata = {**_task_meta(task), "error": str(exc)}

    asyncio.create_task(asyncio.to_thread(_run))
    return APIResponseSuccess(
        message=f"快速推理已开始（{task.task_id[:8]}…）",
        data={"task_id": task.task_id, "output_dir": str(out_dir)},
    )


@router.post("/terminate")
async def infer_terminate(request: Request):
    payload = _json_body_sync(await request.body())
    task_id = str(payload.get("task_id") or "").strip()
    infer = _infer_task()
    if not task_id and infer is not None:
        task_id = infer.task_id
    if not task_id:
        return APIResponseFail(message="没有可终止的推理任务")
    task = tm.tasks.get(task_id)
    if task is None or _task_meta(task).get("kind") != _INFER_KIND:
        return APIResponseFail(message="推理任务不存在或已结束")
    tm.terminate_task(task_id)
    return APIResponseSuccess(message="已请求终止推理", data={"task_id": task_id})


@router.get("/images/{task_id}")
async def infer_images(task_id: str):
    task = tm.tasks.get(task_id)
    if task is None or _task_meta(task).get("kind") != _INFER_KIND:
        return APIResponseFail(message="推理任务不存在")
    out_dir = Path(str(_task_meta(task).get("output_dir") or ""))
    if not out_dir.is_dir():
        return APIResponseSuccess(message=None, data={"images": [], "status": getattr(task.status, "name", None)})
    names = sorted(
        [p.name for p in out_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}],
        reverse=True,
    )
    return APIResponseSuccess(
        message=None,
        data={
            "images": names,
            "status": getattr(task.status, "name", None),
            "returncode": _task_meta(task).get("returncode"),
            "error": _task_meta(task).get("error"),
        },
    )


@router.get("/image/{task_id}/{name}")
async def infer_image_file(task_id: str, name: str):
    if not _SAFE_NAME.match(name) or ".." in name or "/" in name or "\\" in name:
        return APIResponseFail(message="非法文件名")
    task = tm.tasks.get(task_id)
    if task is None or _task_meta(task).get("kind") != _INFER_KIND:
        return APIResponseFail(message="推理任务不存在")
    out_dir = Path(str(_task_meta(task).get("output_dir") or "")).resolve()
    target = (out_dir / name).resolve()
    try:
        target.relative_to(out_dir)
    except ValueError:
        return APIResponseFail(message="路径越界")
    if not target.is_file():
        return APIResponseFail(message="图片不存在或已被删除")
    return FileResponse(str(target))
