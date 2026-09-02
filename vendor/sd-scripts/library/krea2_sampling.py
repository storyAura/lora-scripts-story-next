# Ported from kohya-ss/musubi-tuner (Apache-2.0):
# src/musubi_tuner/krea2/krea2_sampling.py

"""Functional flow-matching sampler for the K2 MMDiT."""

from __future__ import annotations

import math

import torch
from einops import rearrange, repeat
from PIL import Image
from torch import Tensor

from library.krea2_models import SingleStreamDiT
from library.krea2_utils import single_mmdit_large_wide


def gather_valid_text(txt: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    valid = [txt[i][mask[i]] for i in range(txt.shape[0])]
    max_len = max(v.shape[0] for v in valid)
    out = txt.new_zeros(txt.shape[0], max_len, txt.shape[2], txt.shape[3])
    newmask = torch.zeros(txt.shape[0], max_len, device=txt.device, dtype=torch.bool)
    for i, v in enumerate(valid):
        out[i, : v.shape[0]] = v
        newmask[i, : v.shape[0]] = True
    return out, newmask


def prepare(img: Tensor, txtlen: int, patch: int, txtmask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    b, _, h, w = img.shape
    h_, w_ = h // patch, w // patch
    imgids = torch.zeros((h_, w_, 3), device=img.device)
    imgids[..., 1] = torch.arange(h_, device=img.device)[:, None]
    imgids[..., 2] = torch.arange(w_, device=img.device)[None, :]
    imgpos = repeat(imgids, "h w three -> b (h w) three", b=b, three=3)
    imgmask = torch.ones(b, h_ * w_, device=img.device, dtype=torch.bool)
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
    txtpos = torch.zeros(b, txtlen, 3, device=img.device)
    mask = torch.cat((imgmask, txtmask), dim=1)
    pos = torch.cat((imgpos, txtpos), dim=1)
    return img, pos, mask


def timesteps(seq_len, steps, x1, x2, y1=0.5, y2=1.15, sigma=1.0, mu=None):
    ts = torch.linspace(1, 0, steps + 1)
    if mu is None:
        slope = (y2 - y1) / (x2 - x1)
        mu = slope * seq_len + (y1 - slope * x1)
    ts = math.exp(mu) / (math.exp(mu) + (1.0 / ts - 1.0) ** sigma)
    return ts.tolist()


def krea2_shift_mu(seq_len: float, x1: float = 256, y1: float = 0.5, x2: float = 6400, y2: float = 1.15) -> float:
    slope = (y2 - y1) / (x2 - x1)
    return slope * seq_len + (y1 - slope * x1)


def packed_seq_len(latent_h: int, latent_w: int, patch: int = 2) -> int:
    return (latent_h // patch) * (latent_w // patch)


def unpack(img: Tensor, packed_h: int, packed_w: int, patch: int = 2, channels: int = 16) -> Tensor:
    return rearrange(
        img,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=packed_h,
        w=packed_w,
        ph=patch,
        pw=patch,
        c=channels,
    )


def pixel_seq_len(height: int, width: int, compression: int = 8, patch: int = 2) -> int:
    return packed_seq_len(height // compression, width // compression, patch=patch)


@torch.no_grad()
def sample_euler(
    model: SingleStreamDiT,
    ae,
    txt: Tensor,
    txtmask: Tensor,
    *,
    untxt: Tensor | None = None,
    untxtmask: Tensor | None = None,
    device="cuda",
    dtype=torch.bfloat16,
    width=1024,
    height=1024,
    steps=28,
    cfg_scale=5.5,
    seed=0,
    minres=256,
    maxres=1280,
    y1=0.5,
    y2=1.15,
    mu=None,
):
    patch = model.config.patch
    compression = 2 ** len(ae.temperal_downsample)
    channels = ae.z_dim
    align = compression * patch
    width = ((width + align - 1) // align) * align
    height = ((height + align - 1) // align) * align

    n = txt.shape[0]
    cfg = cfg_scale > 1.0 and untxt is not None
    txt, txtmask = txt.to(device=device, dtype=dtype), txtmask.to(device)
    if cfg:
        untxt, untxtmask = untxt.to(device=device, dtype=dtype), untxtmask.to(device)

    noise = torch.cat(
        [
            torch.randn(
                1,
                channels,
                height // compression,
                width // compression,
                device=device,
                dtype=dtype,
                generator=torch.Generator(device=device).manual_seed(seed + i),
            )
            for i in range(n)
        ],
        dim=0,
    )
    x, pos, mask = prepare(noise, txt.shape[1], patch, txtmask)
    if cfg:
        _, unpos, unmask = prepare(noise, untxt.shape[1], patch, untxtmask)

    x1 = (minres // (compression * patch)) ** 2
    x2 = (maxres // (compression * patch)) ** 2
    ts = timesteps(x.shape[1], steps, x1, x2, y1=y1, y2=y2, mu=mu)

    img = x
    device_type = torch.device(device).type
    with torch.autocast(device_type=device_type, dtype=dtype):
        for tcurr, tprev in zip(ts[:-1], ts[1:]):
            t = torch.full((len(img),), tcurr, dtype=img.dtype, device=img.device)
            cond = model(img=img, context=txt, t=t, pos=pos, mask=mask)
            if cfg:
                uncond = model(img=img, context=untxt, t=t, pos=unpos, mask=unmask)
                v = uncond + cfg_scale * (cond - uncond)
            else:
                v = cond
            img = img + (tprev - tcurr) * v

    img = rearrange(
        img,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        ph=patch,
        pw=patch,
        h=height // (compression * patch),
        w=width // (compression * patch),
    )
    ae = ae.to(img.device)
    pixels = ae.decode_to_pixels(img.to(torch.bfloat16))
    ae = ae.to("cpu")
    pixels = ((pixels + 1.0) * 0.5).clamp(0, 1)
    pixels = rearrange(pixels * 255.0, "b c h w -> b h w c").cpu().byte().numpy()
    return [Image.fromarray(pixels[i]) for i in range(len(pixels))]


DEFAULT_PATCH = single_mmdit_large_wide.patch
