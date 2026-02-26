#!/usr/bin/env python3
"""
Widen SD1.5 latent interface from 4 -> 32 channels (method 1) and save as diffusers.

Changes:
  UNet:
    - conv_in: 4 -> 32 input channels (copy old into channels 0..3, zero rest)
    - conv_out: 4 -> 32 output channels (copy old into channels 0..3, zero rest)

  VAE (AutoencoderKL):
    - encoder.conv_out: 2*4 -> 2*32 moments, with correct mean/logvar placement
    - quant_conv:       2*4 -> 2*32, with correct mean/logvar placement
    - post_quant_conv:  4   -> 32
    - decoder.conv_in:  4   -> 32

Extra channels are initialized to "do nothing" (weights/biases zero),
and extra logvar biases are set to -30 so VAE encoding doesn't inject noise there.

This produces a normal-loadable diffusers pipeline directory.

Test load:
  pipe = StableDiffusionPipeline.from_pretrained(OUTDIR, torch_dtype=torch.float16).to("cuda")
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline


NEW_C = 32
LOGVAR_EXTRA_BIAS = -30.0  # diffusers clamps logvar; -30 ~= near-deterministic


def _new_conv_like(conv: nn.Conv2d, *, in_c: int, out_c: int) -> nn.Conv2d:
    if conv.groups != 1:
        raise ValueError(f"Unsupported: conv.groups={conv.groups} (expected 1)")
    new_conv = nn.Conv2d(
        in_channels=in_c,
        out_channels=out_c,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=(conv.bias is not None),
        padding_mode=conv.padding_mode,
    ).to(device=conv.weight.device, dtype=conv.weight.dtype)
    return new_conv


def _copy_conv_block(
    src: nn.Conv2d,
    dst: nn.Conv2d,
    *,
    src_out_slice: slice,
    src_in_slice: slice,
    dst_out_slice: slice,
    dst_in_slice: slice,
) -> None:
    with torch.no_grad():
        dst.weight[dst_out_slice, dst_in_slice, :, :].copy_(
            src.weight[src_out_slice, src_in_slice, :, :]
        )
        if src.bias is not None:
            dst.bias[dst_out_slice].copy_(src.bias[src_out_slice])


def _expand_unet_latent_io(unet, new_c: int) -> None:
    old_in = int(unet.config.in_channels)
    old_out = int(unet.config.out_channels)

    if old_in != 4 or old_out != 4:
        raise ValueError(f"Expected SD1.5 UNet 4->4, got in={old_in}, out={old_out}")

    # conv_in: (320, 4, k, k) -> (320, 32, k, k)
    old = unet.conv_in
    new = _new_conv_like(old, in_c=new_c, out_c=old.out_channels)
    with torch.no_grad():
        new.weight.zero_()
        if new.bias is not None:
            new.bias.zero_()
    _copy_conv_block(
        old, new,
        src_out_slice=slice(0, old.out_channels),
        src_in_slice=slice(0, 4),
        dst_out_slice=slice(0, old.out_channels),
        dst_in_slice=slice(0, 4),
    )
    unet.conv_in = new

    # conv_out: (4, 320, k, k) -> (32, 320, k, k)
    old = unet.conv_out
    new = _new_conv_like(old, in_c=old.in_channels, out_c=new_c)
    with torch.no_grad():
        new.weight.zero_()
        if new.bias is not None:
            new.bias.zero_()
    _copy_conv_block(
        old, new,
        src_out_slice=slice(0, 4),
        src_in_slice=slice(0, old.in_channels),
        dst_out_slice=slice(0, 4),
        dst_in_slice=slice(0, old.in_channels),
    )
    unet.conv_out = new

    unet.register_to_config(in_channels=new_c, out_channels=new_c)
    print(f"UNet latent I/O: {old_in}->{new_c}, {old_out}->{new_c}")


def _expand_vae_moments_conv(src: nn.Conv2d, *, old_c: int, new_c: int) -> nn.Conv2d:
    """
    Expand a moments conv: (2*old_c)->(2*new_c), preserving placement:
      mean:   [0 .. c-1]
      logvar: [c .. 2c-1]
    """
    if src.out_channels != 2 * old_c:
        raise ValueError(f"Expected out_channels={2*old_c}, got {src.out_channels}")

    dst = _new_conv_like(src, in_c=src.in_channels, out_c=2 * new_c)
    with torch.no_grad():
        dst.weight.zero_()
        if dst.bias is not None:
            dst.bias.zero_()
            # Make extra logvars ~ deterministic (std ~ 0) when encoding
            dst.bias[new_c : 2 * new_c].fill_(LOGVAR_EXTRA_BIAS)

    # Copy mean (old 0..old_c-1) into new mean (0..old_c-1)
    _copy_conv_block(
        src, dst,
        src_out_slice=slice(0, old_c),
        src_in_slice=slice(0, src.in_channels),
        dst_out_slice=slice(0, old_c),
        dst_in_slice=slice(0, src.in_channels),
    )
    # Copy logvar (old old_c..2*old_c-1) into new logvar (new_c..new_c+old_c-1)
    _copy_conv_block(
        src, dst,
        src_out_slice=slice(old_c, 2 * old_c),
        src_in_slice=slice(0, src.in_channels),
        dst_out_slice=slice(new_c, new_c + old_c),
        dst_in_slice=slice(0, src.in_channels),
    )
    return dst


def _expand_vae_quant_conv(src: nn.Conv2d, *, old_c: int, new_c: int) -> nn.Conv2d:
    """
    Expand quant_conv: (2*old_c)->(2*new_c) 1x1, preserving the old 2*old_c linear map
    into the correct mean/logvar slots.
    """
    if src.in_channels != 2 * old_c or src.out_channels != 2 * old_c:
        raise ValueError(
            f"Expected quant_conv {2*old_c}->{2*old_c}, got {src.in_channels}->{src.out_channels}"
        )

    dst = _new_conv_like(src, in_c=2 * new_c, out_c=2 * new_c)
    with torch.no_grad():
        dst.weight.zero_()
        if dst.bias is not None:
            dst.bias.zero_()
            dst.bias[new_c : 2 * new_c].fill_(LOGVAR_EXTRA_BIAS)

    # Indices:
    # old mean:   0..old_c-1
    # old logvar: old_c..2*old_c-1
    # new mean:   0..new_c-1
    # new logvar: new_c..2*new_c-1

    with torch.no_grad():
        # mean_out <- mean_in, logvar_in
        dst.weight[0:old_c, 0:old_c, :, :].copy_(src.weight[0:old_c, 0:old_c, :, :])
        dst.weight[0:old_c, new_c:new_c + old_c, :, :].copy_(src.weight[0:old_c, old_c:2 * old_c, :, :])

        # logvar_out <- mean_in, logvar_in
        dst.weight[new_c:new_c + old_c, 0:old_c, :, :].copy_(src.weight[old_c:2 * old_c, 0:old_c, :, :])
        dst.weight[new_c:new_c + old_c, new_c:new_c + old_c, :, :].copy_(
            src.weight[old_c:2 * old_c, old_c:2 * old_c, :, :]
        )

        if src.bias is not None:
            dst.bias[0:old_c].copy_(src.bias[0:old_c])
            dst.bias[new_c:new_c + old_c].copy_(src.bias[old_c:2 * old_c])

    return dst


def _expand_conv_in_channels_only(src: nn.Conv2d, *, new_in: int) -> nn.Conv2d:
    dst = _new_conv_like(src, in_c=new_in, out_c=src.out_channels)
    with torch.no_grad():
        dst.weight.zero_()
        if dst.bias is not None:
            dst.bias.zero_()
    # copy old into first old_in
    old_in = src.in_channels
    _copy_conv_block(
        src, dst,
        src_out_slice=slice(0, src.out_channels),
        src_in_slice=slice(0, old_in),
        dst_out_slice=slice(0, src.out_channels),
        dst_in_slice=slice(0, old_in),
    )
    return dst


def _expand_square_1x1(src: nn.Conv2d, *, old_c: int, new_c: int) -> nn.Conv2d:
    # generic (old_c->old_c) 1x1 expanded to (new_c->new_c), copying top-left block
    if src.kernel_size != (1, 1):
        raise ValueError("Expected 1x1 conv")
    if src.in_channels != old_c or src.out_channels != old_c:
        raise ValueError(f"Expected {old_c}->{old_c}, got {src.in_channels}->{src.out_channels}")

    dst = _new_conv_like(src, in_c=new_c, out_c=new_c)
    with torch.no_grad():
        dst.weight.zero_()
        if dst.bias is not None:
            dst.bias.zero_()

        dst.weight[0:old_c, 0:old_c, :, :].copy_(src.weight[0:old_c, 0:old_c, :, :])
        if src.bias is not None:
            dst.bias[0:old_c].copy_(src.bias[0:old_c])

    return dst


def _expand_vae_latent_io(vae, new_c: int) -> None:
    old_c = int(vae.config.latent_channels)
    if old_c != 4:
        raise ValueError(f"Expected SD1.5 VAE latent_channels=4, got {old_c}")

    # encoder.conv_out: moments
    vae.encoder.conv_out = _expand_vae_moments_conv(vae.encoder.conv_out, old_c=old_c, new_c=new_c)

    # quant_conv: moments mixer (1x1)
    vae.quant_conv = _expand_vae_quant_conv(vae.quant_conv, old_c=old_c, new_c=new_c)

    # post_quant_conv: latent mapper (1x1): 4->4 becomes 32->32
    vae.post_quant_conv = _expand_square_1x1(vae.post_quant_conv, old_c=old_c, new_c=new_c)

    # decoder.conv_in: consumes latent
    vae.decoder.conv_in = _expand_conv_in_channels_only(vae.decoder.conv_in, new_in=new_c)

    vae.register_to_config(latent_channels=new_c)
    print(f"VAE latent_channels: {old_c}->{new_c}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runwayml/stable-diffusion-v1-5", help="Repo id or local path")
    ap.add_argument("--out", required=True, help="Output directory for modified pipeline")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float32)

    _expand_unet_latent_io(pipe.unet, NEW_C)
    _expand_vae_latent_io(pipe.vae, NEW_C)

    pipe.save_pretrained(str(out_dir), safe_serialization=True)
    print(f"Saved modified pipeline to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
