from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "sd-scripts"))

from library.anima_block_freeze import (  # noqa: E402
    INSERTED_40_BLOCK_INDICES,
    apply_inserted_only_training_freeze,
)
from library.anima_utils import count_anima_blocks, infer_anima_num_blocks  # noqa: E402
from networks.lora_anima import parse_block_selection  # noqa: E402


class _TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)


class _TinyDiT(nn.Module):
    def __init__(self, block_count: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_TinyBlock() for _ in range(block_count))
        self.final_layer = nn.Linear(4, 4, bias=False)


def _block_keys(count: int, prefix: str = "") -> list[str]:
    return [f"{prefix}blocks.{index}.proj.weight" for index in range(count)]


class CountAnimaBlocksTests(unittest.TestCase):
    def test_counts_28_and_40_after_prefix_normalization(self):
        self.assertEqual(count_anima_blocks(_block_keys(28)), 28)
        self.assertEqual(count_anima_blocks(_block_keys(40, "net.")), 40)
        self.assertEqual(count_anima_blocks(_block_keys(40, "model.diffusion_model.")), 40)

    def test_missing_or_gapped_keys_stop_at_first_absent_index(self):
        keys = ["blocks.0.a", "blocks.1.a", "blocks.3.a"]
        self.assertEqual(count_anima_blocks(keys), 2)

    def test_infer_falls_back_to_28_when_file_is_missing(self):
        self.assertEqual(infer_anima_num_blocks("missing-checkpoint.safetensors"), 28)

    def test_infer_reads_safetensors_keys_without_requiring_full_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "anima-40.safetensors"
            tensors = {key: torch.zeros(1) for key in _block_keys(40, "net.")}
            save_file(tensors, str(path))
            self.assertEqual(infer_anima_num_blocks(str(path)), 40)

            path_28 = Path(tmp) / "anima-28.safetensors"
            save_file({key: torch.zeros(1) for key in _block_keys(28)}, str(path_28))
            self.assertEqual(infer_anima_num_blocks(str(path_28)), 28)


class InsertedOnlyFreezeTests(unittest.TestCase):
    def test_40_layer_unfreezes_only_the_12_inserted_blocks(self):
        dit = _TinyDiT(40)
        dit.requires_grad_(True)
        summary = apply_inserted_only_training_freeze(dit)

        self.assertEqual(summary["block_count"], 40)
        self.assertEqual(tuple(summary["inserted_block_indices"]), INSERTED_40_BLOCK_INDICES)
        for index, block in enumerate(dit.blocks):
            expected = index in INSERTED_40_BLOCK_INDICES
            self.assertEqual(
                all(parameter.requires_grad for parameter in block.parameters()),
                expected,
                f"block {index}",
            )
        self.assertFalse(dit.final_layer.weight.requires_grad)

    def test_28_layer_checkpoint_raises(self):
        dit = _TinyDiT(28)
        with self.assertRaisesRegex(ValueError, "28 blocks"):
            apply_inserted_only_training_freeze(dit)


class ParseBlockSelectionTests(unittest.TestCase):
    def test_csv_selects_inserted_40_indices(self):
        csv = ",".join(str(index) for index in INSERTED_40_BLOCK_INDICES)
        selected = parse_block_selection(csv, 40)
        self.assertEqual(len(selected), 40)
        self.assertEqual(sum(selected), 12)
        for index, flag in enumerate(selected):
            self.assertEqual(flag, index in INSERTED_40_BLOCK_INDICES)

    def test_all_keeps_every_block(self):
        self.assertEqual(parse_block_selection("all", 28), [True] * 28)


if __name__ == "__main__":
    unittest.main()
