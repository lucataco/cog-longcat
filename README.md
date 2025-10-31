# LongCat-Video Cog Model

This directory packages Meituan's **LongCat-Video** 13.6B parameter video generator (`meituan-longcat/LongCat-Video`) for deployment on [Replicate](https://replicate.com) using [Cog](https://github.com/replicate/cog). The predictor mirrors the official Gradio demo, offering both text-to-video and image-to-video workflows with optional distillation and refinement stages for speed-versus-quality tradeoffs.

## Features

- Clones the upstream LongCat-Video repository and downloads the official weights tarball (hosted by Replicate) on first run, caching everything under `checkpoints/` for reuse.
- Supports **text-to-video** (T2V) and **image-to-video** (I2V) generation through a single Cog endpoint.
- Exposes **Distill mode** (fast, lower cost) and **Refine mode** (higher fidelity, higher VRAM) with automatic LoRA loading and cleanup.
- Fixed, production-friendly resolution choices: 480p/720p for I2V and 480×832 or 720×1280 for T2V.
- Streams the final clip to MP4 using `diffusers.export_to_video`, selecting 15 fps for base runs and 30 fps when refinement is enabled.

## Requirements

- Docker (required by Cog)
- An NVIDIA GPU with ≥48 GB VRAM is strongly recommended; refinement runs benefit from even larger memory budgets
- Fast disk & network: initial weight download is ~25 GB

## Getting Started

Install Cog (if you have not already):

```bash
curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"
chmod +x /usr/local/bin/cog
```

### Warm the Cache (optional)

The first prediction clones the LongCat repo and downloads a ~25 GB tarball of model checkpoints via `pget`. To prefetch assets:

```bash
cog predict -i prompt="A cat made of stardust walks through neon city lights"
```

All assets are stored in `checkpoints/LongCat-Video/`; subsequent runs reuse them.

## Running Predictions

### Text-to-Video

```bash
cog predict \
  -i prompt="A cinematic shot of a corgi sprinting across a snowy field" \
  -i height=480 \
  -i width=832 \
  -i use_distill=true \
  -i use_refine=false
```

- Set `use_distill=false` to run the full 50-step sampler (slower, higher quality baseline).
- Toggle `use_refine=true` for the second-stage super-resolution pass. Refinement automatically switches output FPS to 30.

### Image-to-Video

```bash
cog predict \
  -i image=@input.png \
  -i prompt="The dragon statue unfurls and takes flight" \
  -i resolution=720p \
  -i use_distill=true \
  -i use_refine=true
```

- `resolution` accepts `480p` or `720p`. The predictor resizes the conditioning image to the selected bucket before inference.
- When refinement is enabled alongside I2V, the source image is reused as conditioning for the high-resolution pass.

### Additional Parameters

- `negative_prompt` (string): Optional negative guidance, defaults to a quality-preserving preset.
- `num_frames` (int): Temporal length (defaults to 93 frames ≈ 6 s). Must satisfy LongCat’s latent divisibility constraints.
- `num_inference_steps` (int): Diffusion steps for Stage 1; defaults to 25 (auto-reduced to 16 when `use_distill=true`).
- `guidance_scale` (float): Classifier-free guidance; defaults to 7.0 (auto-reduced to 1.0 in distill mode).
- `seed` (int): Random seed; identical seeds reproduce outputs given identical settings.

Outputs are saved to a temporary MP4 and returned as the Cog artifact path.

## Project Structure

- `predict.py` – Cog predictor that initializes the LongCat pipeline, manages LoRAs, and exposes the combined T2V/I2V API.
- `requirements.txt` – Python dependencies installed inside the Cog container (CUDA-enabled PyTorch, diffusers, transformers, xFormers, etc.).
- `cog.yaml` – Runtime configuration (CUDA 12.4 base image, GPU enabled, git + pget bootstrap).
- `checkpoints/` – Populated at runtime with the LongCat Git repository and downloaded weights.

## Deploying to Replicate

After local validation:

```bash
cog login
cog push r8.im/<username>/longcat-video
```

Replace `<username>` with your Replicate handle. The push command will bundle `predict.py`, `cog.yaml`, and dependencies into a runnable model card.

## Notes

- Distill mode requires `lora/cfg_step_lora.safetensors`; refine mode requires `lora/refinement_lora.safetensors`. Both are fetched automatically—do not delete them from `checkpoints/`.
- The predictor disables LoRAs between runs and performs aggressive CUDA memory cleanup to reduce fragmentation during repeat invocations.
- For deeper customization (e.g., context parallelism or longer clips), refer to the upstream [LongCat-Video repository](https://github.com/meituan-longcat/LongCat-Video).

