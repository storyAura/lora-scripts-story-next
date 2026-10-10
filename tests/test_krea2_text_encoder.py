from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoTokenizer, Qwen2TokenizerFast, Qwen3Config, Qwen3Model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "sd-scripts"))

from library import krea2_encoder, strategy_krea2
from library.krea2_encoder import Qwen3VLConditioner

SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
TINY_TEXT_CONFIG = {
    **krea2_encoder.QWEN3_VL_4B_TEXT_CONFIG,
    "vocab_size": 64,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 8,
}


def write_tiny_tokenizer(folder: Path, *, list_form: bool) -> Path:
    """A Qwen2-style tokenizer folder; list_form mimics a tokenizer_config.json saved by transformers 5.x."""
    folder.mkdir(parents=True, exist_ok=True)
    vocab = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
    for word in ("a", "b", "c", "cat", "dog", "system", "user", "assistant"):
        vocab.setdefault(word, len(vocab))
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    tokenizer.save(str(folder / "tokenizer.json"))
    config = {
        "tokenizer_class": "Qwen2Tokenizer",
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
        "bos_token": None,
        "unk_token": None,
        "model_max_length": 262144,
    }
    if list_form:
        config.update({"backend": "tokenizers", "extra_special_tokens": ["<|im_start|>", "<|im_end|>"]})
    else:
        config["additional_special_tokens"] = ["<|im_start|>", "<|im_end|>"]
    (folder / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
    return folder


class Krea2TokenizerCompatTests(unittest.TestCase):
    def test_list_form_extra_special_tokens_become_additional_special_tokens(self):
        with tempfile.TemporaryDirectory() as td:
            folder = write_tiny_tokenizer(Path(td) / "tok", list_form=True)
            self.assertEqual(
                krea2_encoder.tokenizer_compat_kwargs(str(folder)),
                {"extra_special_tokens": {}, "additional_special_tokens": ["<|im_start|>", "<|im_end|>"]},
            )

    def test_compatible_config_is_left_alone(self):
        with tempfile.TemporaryDirectory() as td:
            folder = write_tiny_tokenizer(Path(td) / "tok", list_form=False)
            self.assertEqual(krea2_encoder.tokenizer_compat_kwargs(str(folder)), {})
            self.assertEqual(krea2_encoder.tokenizer_compat_kwargs(krea2_encoder.QWEN3_VL_4B_INSTRUCT_REPO_ID), {})

    def test_list_form_tokenizer_loads_and_keeps_special_tokens(self):
        with tempfile.TemporaryDirectory() as td:
            folder = write_tiny_tokenizer(Path(td) / "tok", list_form=True)
            original = (folder / "tokenizer_config.json").read_text(encoding="utf-8")
            for tokenizer_class in (AutoTokenizer, Qwen2TokenizerFast):
                tokenizer = krea2_encoder.load_qwen3_vl_tokenizer(tokenizer_class, str(folder), max_length=8)
                self.assertIn("<|im_start|>", tokenizer.additional_special_tokens)
                ids = tokenizer("<|im_start|>cat<|im_end|>")["input_ids"]
                self.assertEqual(ids[0], tokenizer.convert_tokens_to_ids("<|im_start|>"))
            self.assertEqual((folder / "tokenizer_config.json").read_text(encoding="utf-8"), original)


class Krea2TokenizerSourceTests(unittest.TestCase):
    def test_diffusers_layout_uses_sibling_tokenizer_folder(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "Krea-2-Raw"
            (repo / "text_encoder").mkdir(parents=True)
            write_tiny_tokenizer(repo / "tokenizer", list_form=True)
            source = krea2_encoder.resolve_qwen3_vl_tokenizer_source(
                model_path=str(repo / "text_encoder" / "model.safetensors")
            )
            self.assertEqual(Path(source), repo / "tokenizer")

    def test_sibling_tokenizer_folder_is_ignored_outside_diffusers_layout(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "krea2").mkdir()
            write_tiny_tokenizer(root / "tokenizer", list_form=False)
            source = krea2_encoder.resolve_qwen3_vl_tokenizer_source(
                model_path=str(root / "krea2" / "qwen3_vl_4b.safetensors")
            )
            self.assertEqual(source, krea2_encoder.QWEN3_VL_4B_INSTRUCT_REPO_ID)

    def test_tokenize_strategy_uses_the_text_encoder_tokenizer_without_the_hub(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "Krea-2-Raw"
            (repo / "text_encoder").mkdir(parents=True)
            write_tiny_tokenizer(repo / "tokenizer", list_form=True)
            with patch.object(strategy_krea2.Krea2TokenizeStrategy, "_load_tokenizer", side_effect=AssertionError("hub")):
                strategy = strategy_krea2.Krea2TokenizeStrategy(
                    text_encoder_path=str(repo / "text_encoder" / "model.safetensors")
                )
            self.assertEqual(Path(strategy.tokenizer.name_or_path), repo / "tokenizer")


class Krea2TextEncoderCheckpointTests(unittest.TestCase):
    def test_every_checkpoint_layout_maps_to_qwen3_keys(self):
        weight = torch.zeros(1)
        layouts = {
            "hf": {
                "model.language_model.layers.0.mlp.up_proj.weight": weight,
                "model.language_model.norm.weight": weight,
                "model.visual.blocks.0.attn.qkv.weight": weight,
                "lm_head.weight": weight,
            },
            "diffusers": {
                "language_model.layers.0.mlp.up_proj.weight": weight,
                "language_model.norm.weight": weight,
                "visual.blocks.0.attn.qkv.weight": weight,
            },
            "comfy": {
                "model.layers.0.mlp.up_proj.weight": weight,
                "model.norm.weight": weight,
                "visual.blocks.0.attn.qkv.weight": weight,
            },
        }
        for name, sd in layouts.items():
            with self.subTest(layout=name):
                self.assertEqual(
                    sorted(krea2_encoder.extract_language_model_state_dict(sd)),
                    ["layers.0.mlp.up_proj.weight", "norm.weight"],
                )

    def test_quantized_text_encoder_is_rejected_with_a_clear_error(self):
        sd = {
            "model.layers.0.mlp.up_proj.weight": torch.zeros(1),
            "model.layers.0.mlp.up_proj.weight_scale": torch.ones(1),
        }
        with self.assertRaises(krea2_encoder.Krea2TextEncoderCheckpointError) as ctx:
            krea2_encoder.extract_language_model_state_dict(sd)
        self.assertIn("qwen3vl_4b_bf16", str(ctx.exception))
        with self.assertRaises(krea2_encoder.Krea2TextEncoderCheckpointError):
            krea2_encoder.extract_language_model_state_dict({"model.norm.weight": torch.zeros(1), "scaled_fp8": torch.zeros(0)})

    def test_config_matches_qwen3_vl_4b_text_config(self):
        config = Qwen3Config(**krea2_encoder.QWEN3_VL_4B_TEXT_CONFIG)
        self.assertEqual((config.hidden_size, config.num_hidden_layers, config.head_dim), (2560, 36, 128))
        self.assertEqual((config.num_attention_heads, config.num_key_value_heads), (32, 8))
        self.assertEqual(config.rope_theta, 5000000)
        self.assertIsNone(config.sliding_window)
        self.assertIsNone(config.rope_scaling)

    def test_loads_a_diffusers_text_encoder_into_qwen3_without_the_vision_tower(self):
        torch.manual_seed(0)
        reference = Qwen3Model(Qwen3Config(**TINY_TEXT_CONFIG)).eval()
        checkpoint = {f"language_model.{key}": value.clone() for key, value in reference.state_dict().items()}
        checkpoint["visual.patch_embed.proj.weight"] = torch.zeros(2, 2)
        with tempfile.TemporaryDirectory() as td, patch.dict(krea2_encoder.QWEN3_VL_4B_TEXT_CONFIG, TINY_TEXT_CONFIG):
            path = Path(td) / "model.safetensors"
            save_file(checkpoint, str(path))
            model = krea2_encoder._load_qwen3_vl_text_model(str(path), dtype=torch.float32, device="cpu")
        self.assertIsInstance(model, Qwen3Model)
        self.assertFalse(any(param.requires_grad for param in model.parameters()))
        ids = torch.tensor([[1, 5, 7, 9]])
        with torch.no_grad():
            expected = reference(input_ids=ids, output_hidden_states=True).hidden_states
            actual = model(input_ids=ids, output_hidden_states=True).hidden_states
        for want, got in zip(expected, actual):
            torch.testing.assert_close(got, want)


class _RecordingQwen(torch.nn.Module):
    def __init__(self, hidden: int = 4, layers: int = 3):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.hidden = hidden
        self.layers = layers
        self.calls = []

    @property
    def device(self) -> torch.device:
        return self.anchor.device

    def forward(self, input_ids, attention_mask, position_ids, output_hidden_states, use_cache):
        assert use_cache is False, "the conditioner never decodes, so it must not build a KV cache"
        self.calls.append({"attention_mask": attention_mask, "position_ids": position_ids})
        batch, length = input_ids.shape
        states = [torch.full((batch, length, self.hidden), float(layer)) for layer in range(self.layers)]
        return SimpleNamespace(hidden_states=states)


class Krea2ConditionerPositionTests(unittest.TestCase):
    def test_suffix_positions_continue_after_the_prompt_not_after_the_padding(self):
        with tempfile.TemporaryDirectory() as td:
            folder = write_tiny_tokenizer(Path(td) / "tok", list_form=False)
            tokenizer = krea2_encoder.load_qwen3_vl_tokenizer(AutoTokenizer, str(folder))
            processor = krea2_encoder.load_qwen3_vl_tokenizer(Qwen2TokenizerFast, str(folder))
        qwen = _RecordingQwen()
        conditioner = Qwen3VLConditioner(qwen, tokenizer, processor, max_length=12, select_layers=(1, 2))
        conditioner.prompt_template_encode_prefix = "<|im_start|>"
        conditioner.prompt_template_encode_suffix = "<|im_end|><|im_start|>"
        conditioner.prompt_template_encode_start_idx = 1
        conditioner.prompt_template_encode_suffix_start_idx = 2

        hiddens, mask = conditioner(["cat dog", "cat"])

        call = qwen.calls[0]
        full_mask = call["attention_mask"].bool()
        positions = call["position_ids"]
        self.assertTrue(bool((~full_mask[1]).any()), "the shorter prompt must be padded before the suffix")
        for row in range(2):
            valid = full_mask[row]
            self.assertEqual(positions[row][valid].tolist(), list(range(int(valid.sum()))))
            self.assertTrue(bool((positions[row][1:] >= positions[row][:-1]).all()))
        self.assertEqual(tuple(hiddens.shape[:3]), (2, full_mask.shape[1] - 1, 2))
        self.assertEqual(tuple(mask.shape), (2, full_mask.shape[1] - 1))


if __name__ == "__main__":
    unittest.main()
