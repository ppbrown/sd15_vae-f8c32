#!/usr/bin/env python3

"""
Takes an initialized widened SD1.5 VAE (alpha01, 32-channel) and produces two split variants:
  _lf: channels 8-31 zeroed  (only the low channels 0-7 active)
  _hf: channels 0-7 zeroed   (only the high channels 8-31 active)

Before splitting, channels 0-3 are copied to 8-11 so the HF variant
starts with the original SD1.5 reconstruction capability in those slots.
Fancy cross-channel initialization is left to prep_intermediate.py (alpha01).
"""

import copy as _copy
import jsonargparse as argparse
from pathlib import Path

import torch
from diffusers import AutoencoderKL


def _load_vae(model_path: Path, dtype: torch.dtype) -> AutoencoderKL:
    if (model_path / "vae" / "config.json").is_file():
        return AutoencoderKL.from_pretrained(str(model_path), subfolder="vae", torch_dtype=dtype)
    if (model_path / "config.json").is_file():
        return AutoencoderKL.from_pretrained(str(model_path), torch_dtype=dtype)
    raise SystemExit(f"Could not find a VAE at: {model_path} (expected config.json or vae/config.json)")


def _copy_ch_range(vae: AutoencoderKL, src_start: int, src_stop: int, dst_start: int, dst_stop: int) -> None:
    """Copy latent channel weights [src_start:src_stop] to [dst_start:dst_stop] in all relevant layers."""
    nc = int(getattr(vae.config, "latent_channels", 0))
    with torch.no_grad():
        # encoder.conv_out [2*nc, enc_in, 3, 3]: output rows = mean channels, then logvar channels
        enc_w = vae.encoder.conv_out.weight
        enc_w[dst_start:dst_stop].copy_(enc_w[src_start:src_stop])
        enc_w[nc + dst_start : nc + dst_stop].copy_(enc_w[nc + src_start : nc + src_stop])
        if vae.encoder.conv_out.bias is not None:
            b = vae.encoder.conv_out.bias
            b[dst_start:dst_stop].copy_(b[src_start:src_stop])
            b[nc + dst_start : nc + dst_stop].copy_(b[nc + src_start : nc + src_stop])

        # quant_conv [2*nc, 2*nc, 1, 1]: same layout as enc_w output dimension
        qc_w = vae.quant_conv.weight
        qc_w[dst_start:dst_stop].copy_(qc_w[src_start:src_stop])
        qc_w[nc + dst_start : nc + dst_stop].copy_(qc_w[nc + src_start : nc + src_stop])
        if vae.quant_conv.bias is not None:
            b = vae.quant_conv.bias
            b[dst_start:dst_stop].copy_(b[src_start:src_stop])
            b[nc + dst_start : nc + dst_stop].copy_(b[nc + src_start : nc + src_stop])

        # post_quant_conv [nc, nc, 1, 1]: copy output rows (one row per latent channel)
        pq_w = vae.post_quant_conv.weight
        pq_w[dst_start:dst_stop].copy_(pq_w[src_start:src_stop])
        if vae.post_quant_conv.bias is not None:
            b = vae.post_quant_conv.bias
            b[dst_start:dst_stop].copy_(b[src_start:src_stop])

        # decoder.conv_in [dec_out, nc, kH, kW]: copy input columns (one column per latent channel)
        dec_w = vae.decoder.conv_in.weight
        dec_w[:, dst_start:dst_stop].copy_(dec_w[:, src_start:src_stop])


def _zero_ch_range(vae: AutoencoderKL, zero_start: int, zero_stop: int) -> None:
    """Zero all weights associated with latent channels [zero_start:zero_stop]."""
    nc = int(getattr(vae.config, "latent_channels", 0))
    with torch.no_grad():
        # encoder.conv_out: zero output rows (mean + logvar) for suppressed channels
        enc_w = vae.encoder.conv_out.weight
        enc_w[zero_start:zero_stop].zero_()
        enc_w[nc + zero_start : nc + zero_stop].zero_()
        if vae.encoder.conv_out.bias is not None:
            b = vae.encoder.conv_out.bias
            b[zero_start:zero_stop].zero_()
            b[nc + zero_start : nc + zero_stop].zero_()

        # quant_conv: zero output rows
        qc_w = vae.quant_conv.weight
        qc_w[zero_start:zero_stop].zero_()
        qc_w[nc + zero_start : nc + zero_stop].zero_()
        if vae.quant_conv.bias is not None:
            b = vae.quant_conv.bias
            b[zero_start:zero_stop].zero_()
            b[nc + zero_start : nc + zero_stop].zero_()

        # post_quant_conv: zero output rows so suppressed channels pass nothing to the decoder
        pq_w = vae.post_quant_conv.weight
        pq_w[zero_start:zero_stop].zero_()
        if vae.post_quant_conv.bias is not None:
            b = vae.post_quant_conv.bias
            b[zero_start:zero_stop].zero_()

        # decoder.conv_in: zero input columns so decoder ignores suppressed latent channels
        dec_w = vae.decoder.conv_in.weight
        dec_w[:, zero_start:zero_stop].zero_()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Split a widened SD1.5 latent32 VAE into LF (ch0-7) and HF (ch8-31) variants."
    )
    ap.add_argument("--model", default="kl-f8ch32-alpha01",
                    help='Input full model with "vae/" subdir or VAE-only model')
    ap.add_argument("--out-lf", default="kl-f8ch32-alpha01_lf", help="Output LF VAE directory")
    ap.add_argument("--out-hf", default="kl-f8ch32-alpha01_hf", help="Output HF VAE directory")
    ap.add_argument("--split", type=int, default=8,
                    help="Channel index dividing LF (0..split-1) from HF (split..nc-1) (default: 8)")
    args = ap.parse_args()

    in_path = Path(args.model)
    lf_path = Path(args.out_lf)
    hf_path = Path(args.out_hf)

    for p in (lf_path, hf_path):
        if p.exists() and any(p.iterdir()):
            raise SystemExit(f"Refusing to write into non-empty directory: {p}")

    vae = _load_vae(in_path, dtype=torch.float32)

    nc = int(getattr(vae.config, "latent_channels", 0))
    split = int(args.split)
    old_c = split // 2  # original SD channel count; copy fills first half of HF band

    if nc <= split:
        raise SystemExit(f"VAE latent_channels={nc} must be > split={split}")

    print(f"Loaded VAE from:   {in_path}  (latent_channels={nc})")
    print(f"Copying channels 0-{old_c - 1} to {split}-{split + old_c - 1}")
    _copy_ch_range(vae, 0, old_c, split, split + old_c)

    lf_vae = _copy.deepcopy(vae)
    print(f"LF: zeroing channels {split}-{nc - 1}")
    _zero_ch_range(lf_vae, split, nc)
    lf_path.mkdir(parents=True, exist_ok=True)
    lf_vae.save_pretrained(str(lf_path), safe_serialization=True)
    print(f"Saved LF VAE to:   {lf_path}")

    hf_vae = _copy.deepcopy(vae)
    print(f"HF: zeroing channels 0-{split - 1}")
    _zero_ch_range(hf_vae, 0, split)
    hf_path.mkdir(parents=True, exist_ok=True)
    hf_vae.save_pretrained(str(hf_path), safe_serialization=True)
    print(f"Saved HF VAE to:   {hf_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
