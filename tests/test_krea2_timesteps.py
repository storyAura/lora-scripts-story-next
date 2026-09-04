from __future__ import annotations

import math
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "sd-scripts"))

from library.flux_train_utils import get_lin_function, get_noisy_model_input_and_timesteps, time_shift
from library.krea2_sampling import krea2_shift_mu, packed_seq_len, timesteps


class Krea2TimestepNumericTests(unittest.TestCase):
    def test_resolution_anchors_match_musubi(self):
        self.assertAlmostEqual(krea2_shift_mu(256), 0.5)
        self.assertAlmostEqual(krea2_shift_mu(6400), 1.15)
        mu_1024 = krea2_shift_mu(4096)
        self.assertAlmostEqual(mu_1024, 0.90625)
        # musubi anchors exp(mu)≈2.474871 with rel_tol=1e-4 (exact exp(0.90625)≈2.475024).
        self.assertTrue(math.isclose(math.exp(mu_1024), 2.474871, rel_tol=1e-4))

    def test_1024_midpoint_matches_musubi_ratio(self):
        mu = krea2_shift_mu(4096)
        shift = math.exp(mu)
        midpoint = shift / (1.0 + shift)
        self.assertTrue(math.isclose(midpoint, 2.474871 / (1.0 + 2.474871), rel_tol=1e-4))

    def test_mu_is_monotonic_with_resolution(self):
        seq_256 = packed_seq_len(256 // 8, 256 // 8)
        seq_1024 = packed_seq_len(1024 // 8, 1024 // 8)
        seq_1280 = packed_seq_len(1280 // 8, 1280 // 8)
        self.assertEqual(seq_256, 256)
        self.assertEqual(seq_1024, 4096)
        self.assertEqual(seq_1280, 6400)
        self.assertLess(krea2_shift_mu(seq_256), krea2_shift_mu(seq_1024))
        self.assertLess(krea2_shift_mu(seq_1024), krea2_shift_mu(seq_1280))

    def test_preview_schedule_uses_krea2_x2(self):
        ts = timesteps(4096, 2, x1=256, x2=6400)
        self.assertEqual(len(ts), 3)
        self.assertGreater(ts[0], ts[1])
        self.assertGreater(ts[1], ts[-1])

    def test_train_helper_uses_x2_6400(self):
        mu = get_lin_function(x1=256, y1=0.5, x2=6400, y2=1.15)(4096)
        self.assertAlmostEqual(mu, krea2_shift_mu(4096))
        shifted = time_shift(mu, 1.0, torch.tensor([0.5]))
        self.assertAlmostEqual(float(shifted[0]), math.exp(mu) / (1.0 + math.exp(mu)), places=5)


class Krea2NoisyInputTests(unittest.TestCase):
    def test_krea2_shift_sampling_shapes(self):
        args = SimpleNamespace(
            timestep_sampling="krea2_shift",
            weighting_scheme="none",
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.0,
            sigmoid_scale=1.0,
            discrete_flow_shift=2.5,
            ip_noise_gamma=None,
            ip_noise_gamma_random_strength=False,
        )
        scheduler = MagicMock()
        scheduler.config = MagicMock()
        scheduler.config.num_train_timesteps = 1000
        latents = torch.randn(2, 16, 128, 128)
        noise = torch.randn_like(latents)
        noisy, timesteps_out, sigmas = get_noisy_model_input_and_timesteps(
            args, scheduler, latents, noise, "cpu", torch.float32
        )
        self.assertEqual(noisy.shape, latents.shape)
        self.assertEqual(timesteps_out.shape, (2,))
        self.assertEqual(sigmas.shape, (2, 1, 1, 1))

    def test_krea2_shift_1024_midpoint_matches_musubi_noisy_input(self):
        args = SimpleNamespace(
            timestep_sampling="krea2_shift",
            weighting_scheme="none",
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.0,
            sigmoid_scale=1.0,
            discrete_flow_shift=2.5,
            ip_noise_gamma=None,
            ip_noise_gamma_random_strength=False,
        )
        scheduler = MagicMock()
        scheduler.config = MagicMock()
        scheduler.config.num_train_timesteps = 1000
        latents = torch.zeros(2, 16, 128, 128)
        noise = torch.ones_like(latents)
        with patch("library.flux_train_utils.torch.randn", return_value=torch.zeros(2)):
            noisy, sampled_timesteps, sigmas = get_noisy_model_input_and_timesteps(
                args, scheduler, latents, noise, "cpu", torch.float32
            )
        expected_t = 2.474871 / (1.0 + 2.474871)
        self.assertTrue(math.isclose(float(sigmas[0, 0, 0, 0]), expected_t, rel_tol=1e-4))
        self.assertTrue(torch.allclose(noisy, torch.full_like(noisy, expected_t), atol=1e-5))
        self.assertTrue(torch.allclose(sampled_timesteps, torch.full_like(sampled_timesteps, expected_t * 1000.0), atol=1e-2))


class Krea2LoRATargetTests(unittest.TestCase):
    def test_single_stream_dit_has_264_linears(self):
        from library.krea2_models import SingleStreamDiT
        from library.krea2_utils import single_mmdit_large_wide

        with torch.device("meta"):
            dit = SingleStreamDiT(single_mmdit_large_wide)
        linear_names = [
            name for name, module in dit.named_modules() if isinstance(module, nn.Linear)
        ]
        self.assertEqual(len(linear_names), 264)
        exclude = re.compile(r".*(_modulation|_norm|_embedder|final_layer).*")
        dropped = [name for name in linear_names if exclude.fullmatch(name)]
        self.assertEqual(dropped, [])
        self.assertTrue(hasattr(dit, "device"))
        self.assertTrue(hasattr(dit, "dtype"))
        self.assertEqual(dit.device.type, "meta")


if __name__ == "__main__":
    unittest.main()
