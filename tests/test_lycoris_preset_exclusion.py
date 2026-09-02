"""exclude_name 必须能剪掉类圈选(target_module)递归出的子层。

2026-07-31 定案回归:lycoris_anima_preset 用类名圈 Block,旧 wrapper 的
create_modules_ 内层递归不查 TARGET_EXCLUDE_NAME,导致 adaln_modulation
调制层被 LoKr 接管,训练全局色彩崩坏。此测试守住"排除名单对类圈选生效"。

真实训练走 LycorisNetworkKohya,不是 wrapper;升级 4.0 时必须两边都剪。
"""
import unittest
from pathlib import Path

import torch.nn as nn

from lycoris.kohya import LycorisNetworkKohya
from lycoris.utils.preset import read_preset
from lycoris.wrapper import LycorisNetwork

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANIMA_PRESET = PROJECT_ROOT / "config" / "lycoris_anima_preset.toml"

_PRESET_ATTRS = (
    "ENABLE_CONV",
    "TARGET_REPLACE_MODULE",
    "TARGET_REPLACE_NAME",
    "MODULE_ALGO_MAP",
    "NAME_ALGO_MAP",
    "USE_FNMATCH",
    "TARGET_EXCLUDE_NAME",
)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(8, 8)
        self.mlp = nn.Linear(8, 8)
        self.adaln_modulation_self_attn = nn.Sequential(
            nn.SiLU(), nn.Linear(8, 24, bias=False)
        )


class TinyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()])


class LycorisPresetExclusionTests(unittest.TestCase):
    def setUp(self):
        self._snapshot = {k: getattr(LycorisNetwork, k) for k in _PRESET_ATTRS}

    def tearDown(self):
        for k, v in self._snapshot.items():
            setattr(LycorisNetwork, k, v)

    def test_exclude_name_prunes_class_swept_children(self):
        LycorisNetwork.apply_preset(
            {
                "enable_conv": False,
                "target_module": ["Block"],
                "target_name": [],
                "exclude_name": ["*adaln_modulation*"],
                "use_fnmatch": True,
            }
        )
        net = LycorisNetwork(
            TinyDiT(),
            lora_dim=4,
            alpha=4,
            network_module="lokr",
            warn_on_unmatched=False,
        )
        names = sorted(lora.lora_name for lora in net.loras)
        adaln = [n for n in names if "adaln_modulation" in n]
        self.assertFalse(adaln, f"调制层必须被排除,实际建了: {adaln}")
        self.assertEqual(
            len(names), 4, f"两个 Block 各留 self_attn+mlp 共 4 个,实际: {names}"
        )


class LLMAdapterTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(8, 8)
        self.mlp = nn.Linear(8, 8)
        self.adaln_modulation_self_attn = nn.Sequential(
            nn.SiLU(), nn.Linear(8, 24, bias=False)
        )


class TinyAnima(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()])
        self.llm_adapter = LLMAdapterTransformerBlock()


_KOHYA_PRESET_ATTRS = (
    "ENABLE_CONV",
    "UNET_TARGET_REPLACE_MODULE",
    "UNET_TARGET_REPLACE_NAME",
    "TEXT_ENCODER_TARGET_REPLACE_MODULE",
    "TEXT_ENCODER_TARGET_REPLACE_NAME",
    "MODULE_ALGO_MAP",
    "NAME_ALGO_MAP",
    "USE_FNMATCH",
    "TARGET_EXCLUDE_NAME",
)


class LycorisKohyaAnimaPresetExclusionTests(unittest.TestCase):
    def setUp(self):
        self._snapshot = {k: getattr(LycorisNetworkKohya, k) for k in _KOHYA_PRESET_ATTRS}

    def tearDown(self):
        for k, v in self._snapshot.items():
            setattr(LycorisNetworkKohya, k, v)

    def test_kohya_anima_preset_excludes_adaln_and_keeps_llm_adapter(self):
        self.assertTrue(ANIMA_PRESET.is_file(), f"missing {ANIMA_PRESET}")
        preset = read_preset(str(ANIMA_PRESET))
        self.assertIsNotNone(preset, "lycoris_anima_preset.toml failed PresetConfig validation")
        LycorisNetworkKohya.apply_preset(preset)
        net = LycorisNetworkKohya(
            None,
            TinyAnima(),
            lora_dim=4,
            alpha=4,
            network_module="lokr",
            warn_on_unmatched=False,
            train_llm_adapter=True,
        )
        names = sorted(lora.lora_name for lora in net.unet_loras)
        adaln = [n for n in names if "adaln_modulation" in n]
        self.assertFalse(adaln, f"kohya 路径必须排除调制层,实际建了: {adaln}")
        self.assertTrue(
            any("llm_adapter" in n for n in names),
            f"train_llm_adapter=true 时应保留 LLM Adapter 层,实际: {names}",
        )
        self.assertEqual(
            len(names),
            6,
            f"两个 Block + 一个 LLMAdapter 各留 self_attn+mlp 共 6 个,实际: {names}",
        )

    def test_kohya_drops_llm_adapter_when_flag_is_false(self):
        preset = read_preset(str(ANIMA_PRESET))
        self.assertIsNotNone(preset)
        LycorisNetworkKohya.apply_preset(preset)
        net = LycorisNetworkKohya(
            None,
            TinyAnima(),
            lora_dim=4,
            alpha=4,
            network_module="lokr",
            warn_on_unmatched=False,
            train_llm_adapter=False,
        )
        names = sorted(lora.lora_name for lora in net.unet_loras)
        self.assertFalse(
            any("llm_adapter" in n for n in names),
            f"train_llm_adapter=false 应摘掉 LLM Adapter,实际: {names}",
        )
        self.assertEqual(len(names), 4, f"只应留下两个 Block 的 4 层,实际: {names}")
