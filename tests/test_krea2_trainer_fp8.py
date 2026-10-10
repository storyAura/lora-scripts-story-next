from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "sd-scripts"))

from krea2_train_network import Krea2NetworkTrainer

from mikazuki.training_validation import TrainingConfigurationError, validate_training_configuration


def krea2_args(**overrides):
    values = {
        "text_encoder": "qwen3_vl.safetensors",
        "vae": "qwen_image_vae.safetensors",
        "mixed_precision": "bf16",
        "fp8_base": False,
        "fp8_scaled": False,
        "fp8_base_unet": False,
        "turbo_dit": False,
        "blocks_to_swap": None,
        "timestep_sampling": "shift",
        "discrete_flow_shift": 2.5,
        "cache_text_encoder_outputs": True,
        "cache_text_encoder_outputs_to_disk": True,
        "max_token_length": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def checked(args):
    Krea2NetworkTrainer().assert_extra_args(args, MagicMock(), None)
    return args


class Krea2TrainerFp8Tests(unittest.TestCase):
    def test_plain_fp8_base_runs_as_scaled_fp8(self):
        args = checked(krea2_args(fp8_base=True))
        self.assertTrue(args.fp8_scaled)
        self.assertFalse(args.fp8_base, "train_network would cast the whole quantized DiT to fp8")
        self.assertFalse(args.fp8_base_unet)

    def test_fp8_base_with_scaled_skips_the_generic_fp8_cast(self):
        args = checked(krea2_args(fp8_base=True, fp8_scaled=True))
        self.assertTrue(args.fp8_scaled)
        self.assertFalse(args.fp8_base)
        self.assertFalse(Krea2NetworkTrainer().cast_unet(args))

    def test_fp8_scaled_alone_is_accepted(self):
        args = checked(krea2_args(fp8_scaled=True))
        self.assertTrue(args.fp8_scaled)
        self.assertFalse(args.fp8_base)

    def test_bf16_without_fp8_still_casts(self):
        args = checked(krea2_args())
        self.assertFalse(args.fp8_scaled)
        self.assertTrue(Krea2NetworkTrainer().cast_unet(args))

    def test_non_bf16_mixed_precision_is_rejected(self):
        for precision in ("fp16", "no"):
            with self.subTest(mixed_precision=precision):
                with self.assertRaisesRegex(ValueError, "bf16"):
                    checked(krea2_args(mixed_precision=precision))


class Krea2GuiValidationTests(unittest.TestCase):
    BASE = {
        "pretrained_model_name_or_path": "./sd-models/krea2/raw.safetensors",
        "vae": "./sd-models/krea2/qwen_image_vae.safetensors",
        "text_encoder": "./sd-models/krea2/qwen3_vl.safetensors",
    }

    def test_fp8_scaled_no_longer_requires_fp8_base(self):
        validate_training_configuration({**self.BASE, "fp8_base": False, "fp8_scaled": True}, "krea2-lora")

    def test_fp16_mixed_precision_is_rejected(self):
        with self.assertRaises(TrainingConfigurationError) as ctx:
            validate_training_configuration({**self.BASE, "mixed_precision": "fp16"}, "krea2-lora")
        self.assertEqual(ctx.exception.field, "mixed_precision")

    def test_bf16_or_unset_mixed_precision_is_accepted(self):
        validate_training_configuration({**self.BASE, "mixed_precision": "bf16"}, "krea2-lora")
        validate_training_configuration(dict(self.BASE), "krea2-lora")


if __name__ == "__main__":
    unittest.main()
