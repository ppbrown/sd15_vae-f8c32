#!/usr/bin/env python3

"""
The initial proof-of-concept vae was expanded and created
with a bunch of zero-weight channel info.
But you cant train on that.
So this initializes those zeros to something trainable
and writes out a new vae ready for training
"""

import argparse
from pathlib import Path

import torch
from diffusers import AutoencoderKL


def _abs_sum_is_zero(t: torch.Tensor) -> bool:
    # exact zeros in your generated model should remain exact
    return float(t.detach().abs().sum().item()) == 0.0


def _all_eq(t: torch.Tensor, v: float) -> bool:
    return bool((t.detach() == v).all().item())


def _init_normal_if_zero(t: torch.Tensor, *, std: float) -> bool:
    """
    Initialize tensor t with N(0,std) only if it is currently all zeros.
    Returns True if modified.
    """
    if not _abs_sum_is_zero(t):
        return False
    with torch.no_grad():
        torch.nn.init.normal_(t, mean=0.0, std=std)
    return True


def _load_vae(model_path: Path, dtype: torch.dtype) -> AutoencoderKL:
    # Accept either:
    #   - pipeline dir containing subfolder "vae"
    #   - direct VAE dir
    if (model_path / "vae" / "config.json").is_file():
        return AutoencoderKL.from_pretrained(str(model_path), subfolder="vae", torch_dtype=dtype)
    if (model_path / "config.json").is_file():
        return AutoencoderKL.from_pretrained(str(model_path), torch_dtype=dtype)
    raise SystemExit(f"Could not find a VAE at: {model_path} (expected config.json or vae/config.json)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Prep a widened SD1.5 latent32 VAE for training by initializing only the zeroed new-channel blocks."
    )
    ap.add_argument("--model", default="intermediate-model", 
                    help='Input full model with "vae/" subdir or VAE-only model"'
                    ' (default: "intermediate-model")'
                    )
    ap.add_argument("--out", default="kl-f8ch32-alpha00", help="Output VAE-only directory (default: kl-f8ch32-alpha00)")
    ap.add_argument("--old-ch", type=int, default=4, help="Original SD latent channel count (default: 4)")
    ap.add_argument(
        "--extra-logvar-bias",
        type=float,
        default=-6.0,
        help="If extra-channel logvar bias is still -30, set it to this (default: -6.0)",
    )
    ap.add_argument(
        "--std",
        type=float,
        default=0.01,
        help="Stddev for random init of new-channel blocks (default: 0.01)",
    )
    args = ap.parse_args()

    in_path = Path(args.model)
    out_path = Path(args.out)

    if out_path.exists() and any(out_path.iterdir()):
        raise SystemExit(f"Refusing to write into non-empty directory: {out_path}")

    # Use fp32 for deterministic init and to avoid fp16 tiny-gradient issues at this stage.
    vae = _load_vae(in_path, dtype=torch.float32)

    old_c = int(args.old_ch)
    new_c = int(getattr(vae.config, "latent_channels", 0))
    if new_c <= old_c:
        raise SystemExit(f"VAE latent_channels={new_c} does not look widened beyond old_ch={old_c}")

    modified = 0

    # 1) Decoder must "see" extra latent channels: decoder.conv_in.weight[:, old_c:, :, :]
    dec_w = vae.decoder.conv_in.weight
    modified += int(_init_normal_if_zero(dec_w[:, old_c:, :, :], std=args.std))

    # 2) post_quant_conv is (latent -> latent) 1x1: initialize any blocks touching extra channels
    pq_w = vae.post_quant_conv.weight  # [new_c, new_c, 1, 1]
    modified += int(_init_normal_if_zero(pq_w[old_c:, :old_c, :, :], std=args.std))
    modified += int(_init_normal_if_zero(pq_w[:old_c, old_c:, :, :], std=args.std))
    modified += int(_init_normal_if_zero(pq_w[old_c:, old_c:, :, :], std=args.std))
    if vae.post_quant_conv.bias is not None:
        # bias is usually already 0; we only touch if zero anyway
        pq_b = vae.post_quant_conv.bias
        modified += int(_init_normal_if_zero(pq_b[old_c:], std=args.std))

    # 3) encoder.conv_out produces moments: [mean(0..c-1), logvar(c..2c-1)]
    enc = vae.encoder.conv_out
    enc_w = enc.weight  # [2*new_c, enc_in, 3, 3]
    # extra mean outs: old_c..new_c-1
    modified += int(_init_normal_if_zero(enc_w[old_c:new_c, :, :, :], std=args.std))
    # extra logvar outs: new_c+old_c .. 2*new_c-1
    modified += int(_init_normal_if_zero(enc_w[new_c + old_c : 2 * new_c, :, :, :], std=args.std))
    if enc.bias is not None:
        enc_b = enc.bias
        # extra mean bias: init if zero
        modified += int(_init_normal_if_zero(enc_b[old_c:new_c], std=args.std))
        # if extra logvar bias still pinned at -30, relax it for training
        extra_logvar_b = enc_b[new_c + old_c : 2 * new_c]
        if extra_logvar_b.numel() and _all_eq(extra_logvar_b, -30.0):
            with torch.no_grad():
                extra_logvar_b.fill_(float(args.extra_logvar_bias))
            modified += 1

    # 4) quant_conv mixes moments (2c -> 2c) 1x1
    qc = vae.quant_conv
    qc_w = qc.weight  # [2*new_c, 2*new_c, 1, 1]
    # "old" moment dims are: mean[0:old_c] and logvar[new_c:new_c+old_c]
    old_dims = list(range(0, old_c)) + list(range(new_c, new_c + old_c))
    old_dims_t = torch.tensor(old_dims, dtype=torch.long)

    # Make a boolean mask over [2*new_c] for which dims are "old"
    is_old = torch.zeros((2 * new_c,), dtype=torch.bool)
    is_old[old_dims_t] = True
    new_idx = torch.nonzero(~is_old, as_tuple=False).squeeze(1)
    old_idx = torch.nonzero(is_old, as_tuple=False).squeeze(1)

    # Initialize blocks that touch any new dims (only if currently zero)
    # (new_out, old_in), (old_out, new_in), (new_out, new_in)
    if new_idx.numel() and old_idx.numel():
        modified += int(_init_normal_if_zero(qc_w[new_idx[:, None], old_idx[None, :], :, :], std=args.std))
        modified += int(_init_normal_if_zero(qc_w[old_idx[:, None], new_idx[None, :], :, :], std=args.std))
    if new_idx.numel():
        modified += int(_init_normal_if_zero(qc_w[new_idx[:, None], new_idx[None, :], :, :], std=args.std))

    if qc.bias is not None:
        qc_b = qc.bias
        # extra mean bias (0..new_c-1), slice old_c..new_c-1
        modified += int(_init_normal_if_zero(qc_b[old_c:new_c], std=args.std))
        # relax extra logvar bias if pinned at -30
        extra_logvar_b = qc_b[new_c + old_c : 2 * new_c]
        if extra_logvar_b.numel() and _all_eq(extra_logvar_b, -30.0):
            with torch.no_grad():
                extra_logvar_b.fill_(float(args.extra_logvar_bias))
            modified += 1

    out_path.mkdir(parents=True, exist_ok=True)
    vae.save_pretrained(str(out_path), safe_serialization=True)

    print(f"Loaded VAE from: {in_path}")
    print(f"latent_channels: {old_c} -> {new_c}")
    print(f"Initialized/adjusted blocks: {modified}")
    print(f"Saved VAE-only to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
