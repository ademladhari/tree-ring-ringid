"""DDIM latent inversion aligned with diffusion-based watermark detection."""

from __future__ import annotations

from typing import Any

import torch
from diffusers import DDIMInverseScheduler, StableDiffusionPipeline

from ringid.config import WatermarkProfile


def _ddim_inverse_config_from_scheduler(scheduler: Any) -> dict[str, Any]:
    """Strip forward-only DDIM keys so `DDIMInverseScheduler.from_config` does not warn (e.g. `skip_prk_steps`)."""

    cfg = dict(scheduler.config)
    for k in (
        "skip_prk_steps",
        "thresholding",
        "dynamic_thresholding_ratio",
        "sample_max_value",
    ):
        cfg.pop(k, None)
    ts = cfg.get("timestep_spacing", "leading")
    if ts not in ("leading", "trailing"):
        cfg["timestep_spacing"] = "leading"
    return cfg


def _encode_image_latents(pipe: StableDiffusionPipeline, pil_image: Any) -> torch.Tensor:
    pv = pipe.image_processor.preprocess(pil_image).to(device=pipe.device, dtype=pipe.vae.dtype)
    enc = pipe.vae.encode(pv)
    latent_dist = enc.latent_dist
    latents = latent_dist.mode()
    return latents * pipe.vae.config.scaling_factor


@torch.inference_mode()
def invert_image_to_noise(
    pipe: StableDiffusionPipeline,
    *,
    pil_image: Any,
    prompt: str,
    negative_prompt: str | None,
    profile: WatermarkProfile,
    guidance_scale: float | None = None,
    num_inference_steps: int | None = None,
) -> torch.Tensor:
    """Invert a generated image toward initial latent noise (~x_T estimate) via `DDIMInverseScheduler`."""

    device = pipe.device
    gs = profile.guidance_scale if guidance_scale is None else guidance_scale
    steps = profile.num_inference_steps if num_inference_steps is None else num_inference_steps

    inverse_scheduler = DDIMInverseScheduler.from_config(_ddim_inverse_config_from_scheduler(pipe.scheduler))
    inverse_scheduler.set_timesteps(steps, device=device)
    timesteps = inverse_scheduler.timesteps

    latents = _encode_image_latents(pipe, pil_image)
    latents = latents.to(dtype=pipe.unet.dtype)

    do_cfg = gs > 1.0

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=(negative_prompt or "") if do_cfg else None,
    )

    encoder_hidden_states = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0) if do_cfg else prompt_embeds

    for t in timesteps:
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        latent_model_input = inverse_scheduler.scale_model_input(latent_model_input, t)

        t_int = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
        timestep = torch.full((latent_model_input.shape[0],), t_int, device=device, dtype=torch.long)

        noise_pred = pipe.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        if do_cfg:
            noise_uncond, noise_text = noise_pred.chunk(2, dim=0)
            noise_pred = noise_uncond + gs * (noise_text - noise_uncond)

        latents = inverse_scheduler.step(noise_pred.to(latents.dtype), t_int, latents).prev_sample

    return latents.detach()
