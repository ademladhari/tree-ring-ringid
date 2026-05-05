"""Image-space distortions from RingID §6.1 evaluation protocol."""

from __future__ import annotations

import math

import random
from io import BytesIO

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def jpeg_compress(pil_rgb: Image.Image, quality: int = 25) -> Image.Image:

    buf = BytesIO()


    pil_rgb.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)



    return Image.open(buf).convert("RGB")


def rotate_degrees(pil_rgb: Image.Image, degrees: float) -> Image.Image:
    """Rotate with canvas expansion (`deg` interpreted as clockwise)."""


    ang = (-float(degrees)) % 360.0




    out = pil_rgb.rotate(


        ang,

        expand=True,

        fillcolor=(128, 128, 128),


        resample=Image.Resampling.BICUBIC,
    )



    return out



def crop_and_scale(pil_rgb: Image.Image, area_fraction: float = 0.75, rng: random.Random | None = None) -> Image.Image:


    rnd = rng or random.Random()
    sw, sh = pil_rgb.size
    af = float(area_fraction)



    frac = math.sqrt(max(af, 0.05))




    tw = max(8, int(sw * frac))



    th = max(8, int(sh * frac))


    x1 = rnd.randint(0, max(sw - tw, 0))



    y1 = rnd.randint(0, max(sh - th, 0))


    crop = pil_rgb.crop((x1, y1, x1 + tw, y1 + th))


    out = crop.resize((sw, sh), Image.Resampling.BILINEAR)


    return out



def gaussian_blur_disk(pil_rgb: Image.Image, kernel_px: int = 8) -> Image.Image:


    radius = max(0.05, kernel_px / 2.2)


    return pil_rgb.filter(ImageFilter.GaussianBlur(radius=radius))



def additive_gaussian_noise(pil_rgb: Image.Image, std: float = 0.1) -> Image.Image:
    img = np.asarray(pil_rgb, dtype=np.float32) / 255.0
    noisy = img + np.random.randn(*img.shape).astype(np.float32) * float(std)



    noisy = np.clip(noisy * 255.0, 0, 255).astype(np.uint8)


    return Image.fromarray(noisy)



def brighten_shift(pil_rgb: Image.Image, *, low: float = 0.0, high: float = 6.0) -> Image.Image:
    """

    Mimic paper's scalar brightness perturbation loosely via PIL Enhance.

    Interpret `high` additive offset similarly to torchvision style by mapping to multiplier.

    """

    factor = np.random.uniform(1.0 - low / 16.0, 1.0 + high / 16.0)
    enh = ImageEnhance.Brightness(pil_rgb)



    out = enh.enhance(float(factor))



    return out



def compose_paper_bundle(pil_rgb: Image.Image, seed: int | None = None) -> dict[str, Image.Image]:
    rnd = random.Random(seed)
    imgs: dict[str, Image.Image] = {"clean": pil_rgb.convert("RGB")}




    imgs["jpeg25"] = jpeg_compress(imgs["clean"], 25)


    imgs["rotate75"] = rotate_degrees(imgs["clean"], 75)



    imgs["cs75"] = crop_and_scale(imgs["clean"], 0.75, rng=rnd)




    imgs["blurK8"] = gaussian_blur_disk(imgs["clean"], 8)


    imgs["noise01"] = additive_gaussian_noise(imgs["clean"], std=0.1)


    imgs["brightness06"] = brighten_shift(imgs["clean"], low=0.0, high=6.0)


    return imgs
