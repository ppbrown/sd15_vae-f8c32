# vae-kl-f8c32

"normal" sd and sdxl vae, is what is described as "f8c4".
8x scale, 4 channel.

latest vaes like flux2, are 32 channel.

What happens if we put a 32channel vae on sd1.5? Let's find out!

This experiment will go through multiple steps.
Step 1: Modify the sd1.5 model, and vae, so they have 32 channels instead
of just 4. Can we make it work and preserve the knowledge?
(by leaving the new extra channels zeroed)?

Answer: YES!


Run create_sd_intermediate.py to automatically pull in the old
model, and transmogrify it.
Test it with test_intermediate.py

You will get something like the included sample image, collie.png
