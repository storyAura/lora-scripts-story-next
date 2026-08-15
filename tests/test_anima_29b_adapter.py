from __future__ import annotations

import unittest

from mikazuki.anima_backend.adapter import (
    INSERTED_40_BLOCK_INDICES_CSV,
    adapt_anima_config,
)
from mikazuki.training_validation import (
    TrainingConfigurationError,
    validate_training_configuration,
)


class Anima29bAdapterTests(unittest.TestCase):
    def test_lora_freeze_on_writes_train_block_indices(self):
        adapted, warnings = adapt_anima_config(
            {
                "lora_type": "lora",
                "freeze_inserted_only_training": True,
                "network_dim": 16,
                "network_alpha": 16,
            }
        )

        self.assertEqual(adapted["network_module"], "networks.lora_anima")
        self.assertIn(f"train_block_indices={INSERTED_40_BLOCK_INDICES_CSV}", adapted["network_args"])
        self.assertNotIn("freeze_inserted_only_training", adapted)
        self.assertEqual(warnings, [])

    def test_lora_freeze_respects_explicit_train_block_indices(self):
        adapted, _ = adapt_anima_config(
            {
                "lora_type": "lora",
                "freeze_inserted_only_training": True,
                "train_block_indices": "0,1,2",
            }
        )

        self.assertIn("train_block_indices=0,1,2", adapted["network_args"])
        self.assertNotIn(
            f"train_block_indices={INSERTED_40_BLOCK_INDICES_CSV}",
            adapted["network_args"],
        )

    def test_lora_freeze_off_writes_neither_flag_nor_indices(self):
        adapted, _ = adapt_anima_config(
            {
                "lora_type": "lora",
                "freeze_inserted_only_training": False,
            }
        )

        self.assertNotIn("freeze_inserted_only_training", adapted)
        self.assertTrue(
            all(
                not str(item).startswith("train_block_indices=")
                for item in adapted.get("network_args") or []
            )
        )

    def test_string_false_does_not_inject_block_indices(self):
        adapted, _ = adapt_anima_config(
            {
                "lora_type": "lora",
                "freeze_inserted_only_training": "false",
            }
        )

        self.assertTrue(
            all(
                not str(item).startswith("train_block_indices=")
                for item in adapted.get("network_args") or []
            )
        )

    def test_finetune_freeze_on_writes_the_flag(self):
        adapted, _ = adapt_anima_config(
            {
                "freeze_inserted_only_training": True,
                "learning_rate": "1e-5",
            },
            finetune=True,
        )

        self.assertIs(adapted["freeze_inserted_only_training"], True)
        self.assertNotIn("network_args", adapted)
        self.assertNotIn("train_block_indices", adapted)

    def test_finetune_freeze_off_omits_the_flag(self):
        adapted, _ = adapt_anima_config(
            {
                "freeze_inserted_only_training": False,
                "learning_rate": "1e-5",
            },
            finetune=True,
        )

        self.assertNotIn("freeze_inserted_only_training", adapted)

    def test_lycoris_plus_freeze_is_rejected(self):
        with self.assertRaises(TrainingConfigurationError) as ctx:
            adapt_anima_config(
                {
                    "lora_type": "lokr",
                    "freeze_inserted_only_training": True,
                    "lokr_factor": -1,
                }
            )
        self.assertEqual(ctx.exception.field, "freeze_inserted_only_training")

    def test_lycoris_without_freeze_still_adapts(self):
        adapted, _ = adapt_anima_config(
            {
                "lora_type": "lokr",
                "freeze_inserted_only_training": False,
                "lokr_factor": -1,
            }
        )

        self.assertEqual(adapted["network_module"], "lycoris.kohya")
        self.assertIn("algo=lokr", adapted["network_args"])

    def test_29b_finetune_rejects_frozen_base_quantization(self):
        with self.assertRaises(TrainingConfigurationError) as ctx:
            validate_training_configuration(
                {
                    "base_model_quantization": "nf4",
                },
                "anima-2.9b-finetune",
            )
        self.assertEqual(ctx.exception.field, "base_model_quantization")

    def test_29b_lora_accepts_multires(self):
        validate_training_configuration(
            {
                "anima_29b_train_mode": "lora",
                "lora_type": "lora",
                "multires_per_image": True,
                "target_res": ["512", "1024"],
            },
            "anima-2.9b",
        )


if __name__ == "__main__":
    unittest.main()
