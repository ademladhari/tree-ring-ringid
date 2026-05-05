"""Stable Diffusion text-to-image sampling with watermarked initial latents."""

from __future__ import annotations

from typing import Any

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

from ringid.config import WatermarkProfile
from ringid.torch_env import log_generation_torch_cuda
from ringid.watermark import WatermarkKey, inject_watermark


def load_pipeline(profile: WatermarkProfile) -> StableDiffusionPipeline:
    dtype = torch.float16 if profile.dtype == "float16" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        profile.model_id,
        torch_dtype=dtype,
        revision=profile.revision,
        safety_checker=None,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe


def _permute_bchw_hwc(t: torch.Tensor) -> torch.Tensor:
    return t[0].permute(1, 2, 0).contiguous()


def _permute_hwc_bchw(t: torch.Tensor) -> torch.Tensor:
    return t.permute(2, 0, 1).unsqueeze(0)


@torch.inference_mode()
def generate_watermarked(
    pipe: StableDiffusionPipeline,
    profile: WatermarkProfile,
    prompt: str,
    *,
    negative_prompt: str | None = None,
    height: int = 512,
    width: int = 512,
    seed: int | None = None,
    key_index: int = 0,
    generator: torch.Generator | None = None,
) -> tuple[Any, torch.Tensor, WatermarkKey]:
    """Return (PIL image), initial latents tensor (1,C,H,W), WatermarkKey."""

    images, latents, key = generate_watermarked_batch(
        pipe,
        profile,
        prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        n_images=1,
        latent_seed_base=int(seed or 0),
        watermark_seed=int(seed or 0),
        key_index=key_index,
        generator=generator,
    )
    return images[0], latents, key


@torch.inference_mode()
def generate_watermarked_batch(
    pipe: StableDiffusionPipeline,
    profile: WatermarkProfile,
    prompt: str,
    *,
    negative_prompt: str | None = None,
    height: int = 512,
    width: int = 512,
    n_images: int = 10,
    latent_seed_base: int = 0,
    watermark_seed: int = 0,
    key_index: int = 0,
    generator: torch.Generator | None = None,
) -> tuple[list[Any], torch.Tensor, WatermarkKey]:
    """Generate ``n_images`` in one UNet batch.

    - **Latent diversity:** independent initial noise via ``latent_seed_base + i``.
    - **Shared watermark pattern:** same ``watermark_seed`` for every ``inject_watermark`` (Tree-Ring_rand–style fixed key across images).
    Returns (list of PIL images, latents ``(B,C,H,W)``, reference ``WatermarkKey`` from the first slot).
    """

    if n_images < 1:
        raise ValueError("n_images must be >= 1")

    b = int(n_images)
    log_generation_torch_cuda(pipe, n_images=b)

    device = pipe.device
    dtype = pipe.unet.dtype

    if generator is not None:
        if b != 1:
            raise ValueError("A custom `generator` is only supported when `n_images == 1`.")
        generator.manual_seed(int(latent_seed_base))
        gen_list = [generator]
    else:
        gen_list = [torch.Generator(device=torch.device(device)).manual_seed(latent_seed_base + i) for i in range(b)]

    latents = pipe.prepare_latents(
        b,
        pipe.unet.config.in_channels,
        height,
        width,
        dtype,
        device,
        gen_list,
        latents=None,
    )

    slices: list[torch.Tensor] = []
    ref_key: WatermarkKey | None = None

    for i in range(b):
        hwc = latents[i].permute(1, 2, 0).contiguous().float()
        imprinted_hwc, key, _diag = inject_watermark(
            hwc,
            profile,
            generator=None,
            seed=int(watermark_seed),
            key_index=key_index,
        )
        if ref_key is None:
            ref_key = key
        elif not torch.allclose(ref_key.vector, key.vector, atol=1e-4, rtol=1e-5):
            raise RuntimeError("Watermark reference vectors differ across batch — check inject RNG reproducibility.")

        slices.append(imprinted_hwc.permute(2, 0, 1))

    latents_b = torch.stack(slices, dim=0).to(dtype=dtype)

    prompts = [prompt] * b
    negs = [negative_prompt or ""] * b

    out = pipe(
        prompt=prompts,
        negative_prompt=negs,
        height=height,
        width=width,
        num_inference_steps=profile.num_inference_steps,
        guidance_scale=profile.guidance_scale,
        latents=latents_b,
        generator=gen_list,
        eta=float(profile.scheduler_eta),
    )

    assert ref_key is not None
    return list(out.images), latents_b.detach().cpu(), ref_key
