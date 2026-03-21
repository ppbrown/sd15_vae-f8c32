#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


PATCH = 8
IMAGE_SIZE = 512
GRID = IMAGE_SIZE // PATCH

# 32 total channels:
#   24 luma (Y) low-frequency DCT coefficients
#    4 blue-difference chroma (Cb)
#    4 red-difference chroma (Cr)
Y_KEEP = 24
CB_KEEP = 4
CR_KEEP = 4
LATENT_CHANNELS = Y_KEEP + CB_KEEP + CR_KEEP

VERSION = 1


def zigzag_indices(n: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            r_start = min(s, n - 1)
            r_end = max(-1, s - n)
            for r in range(r_start, r_end, -1):
                c = s - r
                if 0 <= c < n:
                    out.append((r, c))
        else:
            c_start = min(s, n - 1)
            c_end = max(-1, s - n)
            for c in range(c_start, c_end, -1):
                r = s - c
                if 0 <= r < n:
                    out.append((r, c))
    return out


def dct_matrix(n: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=np.float32)
    scale0 = np.sqrt(1.0 / n)
    scale = np.sqrt(2.0 / n)
    for k in range(n):
        alpha = scale0 if k == 0 else scale
        for i in range(n):
            m[k, i] = alpha * np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    return m


DCT8 = dct_matrix(PATCH)
ZZ = zigzag_indices(PATCH)
Y_IDXS = ZZ[:Y_KEEP]
CB_IDXS = ZZ[:CB_KEEP]
CR_IDXS = ZZ[:CR_KEEP]


def rgb_to_ycbcr(img: np.ndarray) -> np.ndarray:
    # img: float32 in [0, 1], shape (H, W, 3)
    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]

    y = 0.299000 * r + 0.587000 * g + 0.114000 * b
    cb = -0.168736 * r - 0.331264 * g + 0.500000 * b
    cr = 0.500000 * r - 0.418688 * g - 0.081312 * b

    return np.stack([y, cb, cr], axis=-1).astype(np.float32, copy=False)


def ycbcr_to_rgb(img: np.ndarray) -> np.ndarray:
    # img: float32, shape (H, W, 3)
    y = img[..., 0]
    cb = img[..., 1]
    cr = img[..., 2]

    r = y + 1.402000 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772000 * cb

    return np.stack([r, g, b], axis=-1).astype(np.float32, copy=False)


def resize_and_center_crop(im: Image.Image, target_size: int) -> Image.Image:
    w, h = im.size
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid image size: {w}x{h}")

    # scale down (or up if needed) while preserving aspect ratio,
    # then center-crop to target_size x target_size
    scale = max(target_size / w, target_size / h)
    new_w = max(target_size, int(round(w * scale)))
    new_h = max(target_size, int(round(h * scale)))

    im = im.resize((new_w, new_h), Image.Resampling.LANCZOS)

    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    right = left + target_size
    bottom = top + target_size

    return im.crop((left, top, right, bottom))


def image_to_float_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im = resize_and_center_crop(im, IMAGE_SIZE)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def float_rgb_to_image(arr: np.ndarray) -> Image.Image:
    arr8 = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(arr8, mode="RGB")


def force_png_path(path: Path) -> Path:
    if path.suffix.lower() == ".png":
        return path
    return path.with_suffix(".png")


def save_image_lossless_png(path: Path, arr: np.ndarray) -> Path:
    out_path = force_png_path(path)
    img = float_rgb_to_image(arr)
    img.save(out_path, format="PNG", optimize=False, compress_level=0)
    return out_path


def patchify(channel: np.ndarray) -> np.ndarray:
    # channel: (512, 512) -> (64, 64, 8, 8)
    return channel.reshape(GRID, PATCH, GRID, PATCH).transpose(0, 2, 1, 3)


def unpatchify(blocks: np.ndarray) -> np.ndarray:
    # blocks: (64, 64, 8, 8) -> (512, 512)
    return blocks.transpose(0, 2, 1, 3).reshape(IMAGE_SIZE, IMAGE_SIZE)


def block_dct2(blocks: np.ndarray) -> np.ndarray:
    # blocks: (..., 8, 8)
    return (DCT8 @ blocks) @ DCT8.T


def block_idct2(coeffs: np.ndarray) -> np.ndarray:
    # coeffs: (..., 8, 8)
    return (DCT8.T @ coeffs) @ DCT8


def encode_rgb_to_latent(img_rgb: np.ndarray) -> np.ndarray:
    img_ycc = rgb_to_ycbcr(img_rgb)

    coeffs = []
    for ch in range(3):
        blocks = patchify(img_ycc[..., ch])
        dct = block_dct2(blocks)
        coeffs.append(dct)
    y_dct, cb_dct, cr_dct = coeffs

    latent = np.zeros((GRID, GRID, LATENT_CHANNELS), dtype=np.float32)

    out_ch = 0
    for r, c in Y_IDXS:
        latent[..., out_ch] = y_dct[..., r, c]
        out_ch += 1
    for r, c in CB_IDXS:
        latent[..., out_ch] = cb_dct[..., r, c]
        out_ch += 1
    for r, c in CR_IDXS:
        latent[..., out_ch] = cr_dct[..., r, c]
        out_ch += 1

    return latent


def decode_latent_to_rgb(latent: np.ndarray) -> np.ndarray:
    if latent.shape != (GRID, GRID, LATENT_CHANNELS):
        raise ValueError(
            f"latent must have shape ({GRID}, {GRID}, {LATENT_CHANNELS}); got {latent.shape}"
        )

    y_dct = np.zeros((GRID, GRID, PATCH, PATCH), dtype=np.float32)
    cb_dct = np.zeros((GRID, GRID, PATCH, PATCH), dtype=np.float32)
    cr_dct = np.zeros((GRID, GRID, PATCH, PATCH), dtype=np.float32)

    in_ch = 0
    for r, c in Y_IDXS:
        y_dct[..., r, c] = latent[..., in_ch]
        in_ch += 1
    for r, c in CB_IDXS:
        cb_dct[..., r, c] = latent[..., in_ch]
        in_ch += 1
    for r, c in CR_IDXS:
        cr_dct[..., r, c] = latent[..., in_ch]
        in_ch += 1

    y = unpatchify(block_idct2(y_dct))
    cb = unpatchify(block_idct2(cb_dct))
    cr = unpatchify(block_idct2(cr_dct))

    rgb = ycbcr_to_rgb(np.stack([y, cb, cr], axis=-1))
    return np.clip(rgb, 0.0, 1.0)


def save_latent(path: Path, latent: np.ndarray) -> None:
    np.savez_compressed(
        path,
        latent=latent.astype(np.float32, copy=False),
        version=np.int32(VERSION),
        image_size=np.int32(IMAGE_SIZE),
        patch=np.int32(PATCH),
        y_keep=np.int32(Y_KEEP),
        cb_keep=np.int32(CB_KEEP),
        cr_keep=np.int32(CR_KEEP),
        colorspace=np.array("YCbCr", dtype="<U16"),
        basis=np.array("8x8_dct_zigzag", dtype="<U32"),
    )


def load_latent(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        latent = data["latent"].astype(np.float32, copy=False)

        version = int(data["version"])
        image_size = int(data["image_size"])
        patch = int(data["patch"])
        y_keep = int(data["y_keep"])
        cb_keep = int(data["cb_keep"])
        cr_keep = int(data["cr_keep"])
        colorspace = str(data["colorspace"])
        basis = str(data["basis"])

    if version != VERSION:
        raise ValueError(f"unsupported latent version {version}; expected {VERSION}")
    if image_size != IMAGE_SIZE or patch != PATCH:
        raise ValueError(
            f"latent settings mismatch: image_size={image_size}, patch={patch}, "
            f"expected image_size={IMAGE_SIZE}, patch={PATCH}"
        )
    if y_keep != Y_KEEP or cb_keep != CB_KEEP or cr_keep != CR_KEEP:
        raise ValueError(
            f"latent channel layout mismatch: got Y/Cb/Cr={y_keep}/{cb_keep}/{cr_keep}, "
            f"expected {Y_KEEP}/{CB_KEEP}/{CR_KEEP}"
        )
    if colorspace != "YCbCr" or basis != "8x8_dct_zigzag":
        raise ValueError(
            f"latent metadata mismatch: colorspace={colorspace!r}, basis={basis!r}"
        )

    return latent


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0.0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def cmd_encode(args: argparse.Namespace) -> int:
    img = image_to_float_rgb(Path(args.input_image))
    latent = encode_rgb_to_latent(img)
    save_latent(Path(args.output_latent), latent)
    print(f"wrote latent: {args.output_latent}")
    print(f"latent shape: {latent.shape}")
    return 0


def cmd_decode(args: argparse.Namespace) -> int:
    latent = load_latent(Path(args.input_latent))
    rgb = decode_latent_to_rgb(latent)
    out_path = save_image_lossless_png(Path(args.output_image), rgb)
    print(f"wrote image: {out_path}")
    return 0


def cmd_roundtrip(args: argparse.Namespace) -> int:
    img = image_to_float_rgb(Path(args.input_image))
    latent = encode_rgb_to_latent(img)

    if args.output_latent:
        save_latent(Path(args.output_latent), latent)
        print(f"wrote latent: {args.output_latent}")

    rgb = decode_latent_to_rgb(latent)
    out_path = save_image_lossless_png(Path(args.output_image), rgb)

    print(f"wrote image: {out_path}")
    print(f"latent shape: {latent.shape}")
    print(f"psnr: {psnr(img, rgb):.4f} dB")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Synthetic 32-channel 64x64 latent codec for images. "
            "Input images are automatically scaled with preserved aspect ratio "
            "and center-cropped to 512x512 before encoding. "
            "Decoded images are always written as lossless PNG."
        )
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_encode = sub.add_parser(
        "encode",
        usage="%(prog)s encode INPUT_IMAGE OUTPUT_LATENT",
        help="encode one input image to one output latent file",
    )
    p_encode.add_argument(
        "input_image",
        metavar="INPUT_IMAGE",
        help="single input image file",
    )
    p_encode.add_argument(
        "output_latent",
        metavar="OUTPUT_LATENT",
        help="single output latent .npz file",
    )
    p_encode.set_defaults(func=cmd_encode)

    p_decode = sub.add_parser(
        "decode",
        usage="%(prog)s decode INPUT_LATENT OUTPUT_IMAGE",
        help="decode one input latent file to one output image (always saved as PNG)",
    )
    p_decode.add_argument(
        "input_latent",
        metavar="INPUT_LATENT",
        help="single input latent .npz file",
    )
    p_decode.add_argument(
        "output_image",
        metavar="OUTPUT_IMAGE",
        help="requested output image path; actual output is always PNG",
    )
    p_decode.set_defaults(func=cmd_decode)

    p_round = sub.add_parser(
        "roundtrip",
        usage="%(prog)s roundtrip INPUT_IMAGE OUTPUT_IMAGE [--latent OUTPUT_LATENT]",
        help="encode one input image and decode it to one output image (always saved as PNG)",
    )
    p_round.add_argument(
        "input_image",
        metavar="INPUT_IMAGE",
        help="single input image file",
    )
    p_round.add_argument(
        "output_image",
        metavar="OUTPUT_IMAGE",
        help="requested output reconstructed image path; actual output is always PNG",
    )
    p_round.add_argument(
        "--latent",
        dest="output_latent",
        metavar="OUTPUT_LATENT",
        help="optional output latent .npz file to save",
    )
    p_round.set_defaults(func=cmd_roundtrip)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
