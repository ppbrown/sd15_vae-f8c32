#!/usr/bin/env python3
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


MODEL_DIR = "intermediate-model"
PROMPT = "A collie dog"
STEPS = 30
OUTFILE = "collie.png"


def main() -> int:
#    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(MODEL_DIR, torch_dtype=dtype)

    # Optional: faster / commonly good scheduler choice
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    g = torch.Generator(device=device).manual_seed(1234)

    with torch.inference_mode():
        img = pipe(
            prompt=PROMPT,
            num_inference_steps=STEPS,
            guidance_scale=7.5,
            generator=g,
        ).images[0]

    img.save(OUTFILE)
    print(f"saved: {OUTFILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
