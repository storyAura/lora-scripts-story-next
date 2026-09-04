from pathlib import Path
import importlib.util

import pytest

from mikazuki.tokenizer_cache import (
    BUNDLED_TOKENIZER_DIRS,
    CLIP_TOKENIZER_FILES,
    FLUX_T5_TOKENIZER_HF_ID,
    OPTIONAL_TOKENIZER_FILES_BY_REPO,
    T5_TOKENIZER_FILES,
    TOKENIZER_FILES,
    bundled_tokenizer_cache_dir,
    is_tokenizer_bundle_complete,
    required_tokenizer_files,
    tokenizer_local_dir,
)


def test_tokenizer_local_dir_uses_underscore_folder_names():
    root = Path("/cache")
    assert tokenizer_local_dir(root, "openai/clip-vit-large-patch14") == root / "openai_clip-vit-large-patch14"
    assert (
        tokenizer_local_dir(root, "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
        == root / "laion_CLIP-ViT-bigG-14-laion2B-39B-b160k"
    )


def test_required_tokenizer_files_per_repo():
    assert required_tokenizer_files("openai/clip-vit-large-patch14") == CLIP_TOKENIZER_FILES
    assert required_tokenizer_files(FLUX_T5_TOKENIZER_HF_ID) == T5_TOKENIZER_FILES
    assert "tokenizer.json" not in T5_TOKENIZER_FILES
    assert OPTIONAL_TOKENIZER_FILES_BY_REPO[FLUX_T5_TOKENIZER_HF_ID] == ("tokenizer.json",)
    assert TOKENIZER_FILES == CLIP_TOKENIZER_FILES


def test_is_tokenizer_bundle_complete_requires_all_files(tmp_path: Path):
    root = tmp_path / "tokenizer-cache"
    for repo_id in BUNDLED_TOKENIZER_DIRS:
        local = tokenizer_local_dir(root, repo_id)
        local.mkdir(parents=True)
        for name in required_tokenizer_files(repo_id):
            (local / name).write_text("x", encoding="utf-8")
    assert is_tokenizer_bundle_complete(root)


def test_bundled_tokenizer_cache_dir_returns_none_when_incomplete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "tokenizer-cache"
    root.mkdir()
    monkeypatch.setenv("MIKAZUKI_TOKENIZER_CACHE_DIR", str(root))
    assert bundled_tokenizer_cache_dir() is None


def test_bundled_tokenizer_cache_dir_returns_path_when_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "tokenizer-cache"
    for repo_id in BUNDLED_TOKENIZER_DIRS:
        local = tokenizer_local_dir(root, repo_id)
        local.mkdir(parents=True)
        for name in required_tokenizer_files(repo_id):
            (local / name).write_text("x", encoding="utf-8")
    monkeypatch.setenv("MIKAZUKI_TOKENIZER_CACHE_DIR", str(root))
    assert bundled_tokenizer_cache_dir() == str(root).replace("\\", "/")


def test_bundled_tokenizer_cache_dir_flux_requires_t5(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "tokenizer-cache"
    for repo_id in ("openai/clip-vit-large-patch14", "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"):
        local = tokenizer_local_dir(root, repo_id)
        local.mkdir(parents=True)
        for name in required_tokenizer_files(repo_id):
            (local / name).write_text("x", encoding="utf-8")
    monkeypatch.setenv("MIKAZUKI_TOKENIZER_CACHE_DIR", str(root))
    assert bundled_tokenizer_cache_dir(train_type="sdxl-lora") == str(root).replace("\\", "/")
    assert bundled_tokenizer_cache_dir(train_type="flux-lora") is None


def test_apply_tokenizer_cache_dir_injects_for_sdxl_lora(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from mikazuki.app.api import apply_tokenizer_cache_dir

    root = tmp_path / "tokenizer-cache"
    for repo_id in ("openai/clip-vit-large-patch14", "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"):
        local = tokenizer_local_dir(root, repo_id)
        local.mkdir(parents=True)
        for name in required_tokenizer_files(repo_id):
            (local / name).write_text("x", encoding="utf-8")
    monkeypatch.setenv("MIKAZUKI_TOKENIZER_CACHE_DIR", str(root))

    config: dict = {}
    apply_tokenizer_cache_dir(config, "sdxl-lora")
    assert config["tokenizer_cache_dir"] == str(root).replace("\\", "/")


def test_apply_tokenizer_cache_dir_injects_for_krea2_lora(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from mikazuki.app.api import apply_tokenizer_cache_dir
    from mikazuki.tokenizer_cache import QWEN3_VL_TOKENIZER_HF_ID

    root = tmp_path / "tokenizer-cache"
    local = tokenizer_local_dir(root, QWEN3_VL_TOKENIZER_HF_ID)
    local.mkdir(parents=True)
    for name in required_tokenizer_files(QWEN3_VL_TOKENIZER_HF_ID):
        (local / name).write_text("x", encoding="utf-8")
    monkeypatch.setenv("MIKAZUKI_TOKENIZER_CACHE_DIR", str(root))

    config: dict = {}
    apply_tokenizer_cache_dir(config, "krea2-lora")
    assert config["tokenizer_cache_dir"] == str(root).replace("\\", "/")


def test_apply_tokenizer_cache_dir_injects_for_flux_lora(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from mikazuki.app.api import apply_tokenizer_cache_dir

    root = tmp_path / "tokenizer-cache"
    for repo_id in BUNDLED_TOKENIZER_DIRS:
        local = tokenizer_local_dir(root, repo_id)
        local.mkdir(parents=True)
        for name in required_tokenizer_files(repo_id):
            (local / name).write_text("x", encoding="utf-8")
    monkeypatch.setenv("MIKAZUKI_TOKENIZER_CACHE_DIR", str(root))

    config: dict = {}
    apply_tokenizer_cache_dir(config, "flux-lora")
    assert config["tokenizer_cache_dir"] == str(root).replace("\\", "/")


def test_prefetch_warns_when_optional_t5_tokenizer_json_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "prefetch_sdxl_tokenizer.py"
    spec = importlib.util.spec_from_file_location("prefetch_sdxl_tokenizer_test", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def fake_download(hf_repo_id: str, filename: str, dest: Path, **_kwargs):
        if hf_repo_id == FLUX_T5_TOKENIZER_HF_ID and filename == "tokenizer.json":
            raise RuntimeError("not found")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("x", encoding="utf-8")

    monkeypatch.setattr(module, "_download_tokenizer_file", fake_download)

    cache_root = tmp_path / "tokenizer-cache"
    module.ensure_sdxl_tokenizer_cache(cache_root)

    assert is_tokenizer_bundle_complete(cache_root)
    assert not (tokenizer_local_dir(cache_root, FLUX_T5_TOKENIZER_HF_ID) / "tokenizer.json").exists()
    assert "WARNING: optional tokenizer file skipped" in capsys.readouterr().err


def test_build_accelerate_train_command_uses_mirror_launch_entry(monkeypatch):
    from mikazuki import process

    monkeypatch.setattr(process, "_module_origin_under_user_site", lambda _name: False)
    args, env, _ = process.build_accelerate_train_command(
        trainer_file="./vendor/sd-scripts/sdxl_train_network.py",
        toml_path="config/autosave/test.toml",
    )
    assert "accelerate_launch.py" in args[1]
    assert env.get("PYTHONNOUSERSITE") == "1"
