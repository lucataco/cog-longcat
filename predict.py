import gc
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from cog import BasePredictor, Input, Path as CogPath
import imageio
from PIL import Image
from safetensors.torch import load_file as safetensors_load_file


CACHE_ROOT = Path(__file__).resolve().parent / "checkpoints"
REPO_URL = "https://github.com/meituan-longcat/LongCat-Video.git"
REPO_NAME = "LongCat-Video"
REPO_PATH = CACHE_ROOT / REPO_NAME
WEIGHTS_ROOT = REPO_PATH / "weights"
CHECKPOINT_DIR = WEIGHTS_ROOT / "LongCat-Video"
MODEL_URL = "https://public-weights.replicate.delivery/meituan-longcat/LongCat-Video/model.tar"

CFG_STEP_LORA_KEY = "cfg_step_lora"
REFINEMENT_LORA_KEY = "refinement_lora"

RESOLUTION_MAP = {
    "480p": (480, 832),
    "720p": (720, 1280),
}

NEGATIVE_PROMPT_DEFAULT = (
    "ugly, blurry, low quality, static, subtitles, watermark, distorted, artifacts"
)


def torch_gc() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def download_weights(url: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        ["pget", "-xf", url, str(dest)],
        close_fds=False,
    )


class Predictor(BasePredictor):
    pipe = None

    def setup(self) -> None:
        """Download weights, clone repo, and initialize LongCat-Video pipeline."""

        torch.backends.cuda.matmul.allow_tf32 = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        CACHE_ROOT.mkdir(parents=True, exist_ok=True)

        if not REPO_PATH.exists():
            subprocess.run(
                ["git", "clone", REPO_URL, str(REPO_PATH)],
                check=True,
            )

        if str(REPO_PATH) not in sys.path:
            sys.path.insert(0, str(REPO_PATH))

        from transformers import AutoTokenizer, UMT5EncoderModel

        from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
        from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
        from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
        from longcat_video.modules.scheduling_flow_match_euler_discrete import (
            FlowMatchEulerDiscreteScheduler,
        )
        from longcat_video.context_parallel import context_parallel_util

        self._dit_cls = LongCatVideoTransformer3DModel

        if not CHECKPOINT_DIR.exists():
            download_weights(MODEL_URL, WEIGHTS_ROOT)

        if not CHECKPOINT_DIR.exists():
            raise FileNotFoundError(
                "Expected LongCat-Video weights to be present after download, but the directory was not found."
            )

        dit_dir = CHECKPOINT_DIR / "dit"
        if dit_dir.exists():
            available = sorted(f.name for f in dit_dir.iterdir())
            print("[LongCat] Found DIT weight files:", available)

        cp_split_hw = context_parallel_util.get_optimal_split(1)

        tokenizer = AutoTokenizer.from_pretrained(
            str(CHECKPOINT_DIR), subfolder="tokenizer", torch_dtype=self.torch_dtype
        )
        text_encoder = UMT5EncoderModel.from_pretrained(
            str(CHECKPOINT_DIR),
            subfolder="text_encoder",
            torch_dtype=self.torch_dtype,
        )
        vae = AutoencoderKLWan.from_pretrained(
            str(CHECKPOINT_DIR), subfolder="vae", torch_dtype=self.torch_dtype
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            str(CHECKPOINT_DIR), subfolder="scheduler", torch_dtype=self.torch_dtype
        )
        dit = self._load_dit(dit_dir, cp_split_hw)

        self.pipe = LongCatVideoPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=scheduler,
            dit=dit,
        )
        self.pipe.to(self.device)

        # Preload LoRA weights for later activation
        self.cfg_step_lora_available = False
        cfg_lora_path = CHECKPOINT_DIR / "lora" / f"{CFG_STEP_LORA_KEY}.safetensors"
        if cfg_lora_path.exists():
            self.pipe.dit.load_lora(cfg_lora_path, CFG_STEP_LORA_KEY)
            self.cfg_step_lora_available = True

        self.refinement_lora_available = False
        refinement_lora_path = CHECKPOINT_DIR / "lora" / f"{REFINEMENT_LORA_KEY}.safetensors"
        if refinement_lora_path.exists():
            self.pipe.dit.load_lora(refinement_lora_path, REFINEMENT_LORA_KEY)
            self.refinement_lora_available = True

    def predict(
        self,
        prompt: str = Input(description="Describe the video you want to generate."),
        image: Optional[CogPath] = Input(
            default=None,
            description="Optional image to drive image-to-video generation.",
        ),
        negative_prompt: str = Input(
            default=NEGATIVE_PROMPT_DEFAULT,
            description="Negative prompt to avoid undesired content.",
        ),
        resolution: str = Input(
            choices=list(RESOLUTION_MAP.keys()),
            default="480p",
            description="Image-to-video target resolution.",
        ),
        height: int = Input(
            choices=[480, 720],
            default=480,
            description="Text-to-video frame height.",
        ),
        width: int = Input(
            choices=[832, 1280],
            default=832,
            description="Text-to-video frame width.",
        ),
        num_frames: int = Input(
            default=93,
            ge=33,
            le=257,
            description="Number of frames to generate. Must satisfy pipeline requirements.",
        ),
        seed: int = Input(default=42, description="Random seed for reproducibility."),
        use_distill: bool = Input(
            default=True,
            description="Use distilled weights for faster, lower-cost generation.",
        ),
        use_refine: bool = Input(
            default=False,
            description="Run the optional refinement stage for higher quality output.",
        ),
        num_inference_steps: int = Input(
            default=25,
            ge=1,
            le=200,
            description="Number of diffusion steps for the base stage.",
        ),
        guidance_scale: float = Input(
            default=7.0,
            ge=0.0,
            le=20.0,
            description="Classifier-free guidance scale for the base stage.",
        ),
    ) -> CogPath:
        if self.pipe is None:
            raise RuntimeError("Pipeline is not initialized. Call setup() first.")

        if not prompt or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        mode = "i2v" if image is not None else "t2v"

        if mode == "t2v" and (height, width) not in RESOLUTION_MAP.values():
            raise ValueError("Height and width must be one of the supported resolution pairs.")

        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))

        steps = num_inference_steps
        cfg_scale = guidance_scale
        if use_distill:
            if steps == 25:
                steps = 16
            if cfg_scale == 7.0:
                cfg_scale = 1.0

        if use_distill and not self.cfg_step_lora_available:
            raise RuntimeError("Distill mode requested but cfg_step_lora weights are unavailable.")

        if use_refine and not self.refinement_lora_available:
            raise RuntimeError("Refine mode requested but refinement LoRA weights are unavailable.")

        # Stage 1: Base generation
        stage1_output = self._generate_stage1(
            mode=mode,
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_path=image,
            resolution_key=resolution,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            use_distill=use_distill,
            generator=generator,
        )

        final_video = stage1_output
        fps = 15

        # Stage 2: Refinement (optional)
        if use_refine:
            refined_output = self._generate_refine(
                base_video=stage1_output,
                prompt=prompt,
                image_path=image if mode == "i2v" else None,
                generator=generator,
            )
            final_video = refined_output
            fps = 30

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            self._write_video(final_video, tmp.name, fps)
            video_path = tmp.name

        torch_gc()

        return CogPath(video_path)

    def _generate_stage1(
        self,
        *,
        mode: str,
        prompt: str,
        negative_prompt: Optional[str],
        image_path: Optional[CogPath],
        resolution_key: str,
        height: int,
        width: int,
        num_frames: int,
        num_inference_steps: int,
        guidance_scale: float,
        use_distill: bool,
        generator: torch.Generator,
    ) -> np.ndarray:
        if use_distill:
            self.pipe.dit.enable_loras([CFG_STEP_LORA_KEY])
        else:
            self.pipe.dit.disable_all_loras()

        try:
            if mode == "i2v":
                pil_image = Image.open(image_path).convert("RGB") if image_path else None
                output = self.pipe.generate_i2v(
                    image=pil_image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    resolution=resolution_key,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    use_distill=use_distill,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )[0]
            else:
                output = self.pipe.generate_t2v(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    use_distill=use_distill,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )[0]
        finally:
            self.pipe.dit.disable_all_loras()
            torch_gc()

        return output

    def _load_dit(self, dit_dir: Path, cp_split_hw):
        if not hasattr(self, "_dit_cls"):
            raise RuntimeError("DIT class reference not initialized.")

        config = self._dit_cls.load_config(str(CHECKPOINT_DIR), subfolder="dit")
        config = dict(config)
        config.update(
            {
                "enable_flashattn3": False,
                "enable_flashattn2": False,
                "enable_xformers": True,
                "cp_split_hw": cp_split_hw,
            }
        )

        dit = self._dit_cls.from_config(config)

        index_path = dit_dir / "diffusion_pytorch_model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"Expected safetensors index file at {index_path}, but it was not found."  # noqa: E501
            )

        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        shard_filenames = sorted(set(index_data["weight_map"].values()))
        state_dict = {}
        for shard_name in shard_filenames:
            shard_path = dit_dir / shard_name
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing shard file: {shard_path}")
            shard_state = safetensors_load_file(str(shard_path), device="cpu")
            state_dict.update(shard_state)

        missing, unexpected = dit.load_state_dict(state_dict, strict=False)
        del state_dict
        if missing:
            raise RuntimeError(f"Missing weights in DIT checkpoint: {missing}")
        if unexpected:
            raise RuntimeError(f"Unexpected weights in DIT checkpoint: {unexpected}")

        return dit

    def _write_video(self, frames: np.ndarray, path: str, fps: int) -> None:
        frames = np.clip(frames, 0.0, 1.0)
        frames_uint8 = (frames * 255).astype(np.uint8)
        if frames_uint8.ndim != 4 or frames_uint8.shape[-1] != 3:
            raise ValueError("Expected frames in shape (num_frames, H, W, 3)")

        writer = imageio.get_writer(
            path,
            fps=fps,
            codec="libx264",
            format="FFMPEG",
            output_params=["-pix_fmt", "yuv420p"],
        )
        try:
            for frame in frames_uint8:
                writer.append_data(frame)
        finally:
            writer.close()

    def _generate_refine(
        self,
        *,
        base_video: np.ndarray,
        prompt: str,
        image_path: Optional[CogPath],
        generator: torch.Generator,
    ) -> np.ndarray:
        self.pipe.dit.enable_loras([REFINEMENT_LORA_KEY])
        self.pipe.dit.enable_bsa()

        try:
            stage1_video_pil = [
                Image.fromarray((frame * 255).astype(np.uint8)) for frame in base_video
            ]
            refine_image = (
                Image.open(image_path).convert("RGB") if image_path is not None else None
            )

            output = self.pipe.generate_refine(
                image=refine_image,
                prompt=prompt,
                stage1_video=stage1_video_pil,
                num_cond_frames=1 if refine_image is not None else 0,
                num_inference_steps=50,
                generator=generator,
            )[0]
        finally:
            self.pipe.dit.disable_all_loras()
            self.pipe.dit.disable_bsa()
            torch_gc()

        return output
