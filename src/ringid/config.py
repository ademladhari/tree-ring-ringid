"""Configuration profiles aligning with RingID paper (Tree-Ring recap + appendix A.2)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any


class FourierNorm(str, Enum):
    """FFT normalization convention (underspecified in paper — explicit here)."""

    ORTHO = "ortho"  # symmetric 1/sqrt(HW); matches torch.fft optional norm
    BACKWARD = "backward"  # PyTorch fft2/ifft2 default forward unscaled, backward 1/N
    FORWARD_SCALED = "forward_scaled"


class EmbedMode(str, Enum):
    """§4.1 vs appendix A.2 ambiguity — both supported."""

    RING_GAUSSIAN = "ring_gaussian"  # One complex Gaussian sample per radial band
    MASK_COMPLEX_GAUSSIAN = "mask_complex_gaussian"  # NC(0,N^2) independent per masked bin


class CenterConvention(str, Enum):
    """Fourier-domain ring center placement (§5.2 discusses 31 vs 32 for SD 64×64 latent)."""

    RING_ID_ALIGNED = "ring_id"  # (N/2, N/2), e.g. (32,32) for N=64
    TREE_RING_ORIGINAL = "tree_ring"  # (N//2 - 1, N//2) e.g. (31,32) per paper Fig. caption


def _fft_norm_torch_key(fn: FourierNorm) -> str | None:
    if fn == FourierNorm.ORTHO:
        return "ortho"
    if fn == FourierNorm.BACKWARD:
        return "backward"
    return None


@dataclass
class WatermarkProfile:
    """Unified hyper-parameters for imprint + detection."""

    # Latent spatial size (Stable Diffusion: 64 for 512px at 8x downsampling)
    latent_h: int = 64
    latent_w: int = 64
    latent_c: int = 4

    # Ring geometry: inclusive inner/outer radius in pixel units from center (paper Table 1: Tree-Ring 0–10; RingID often 3–14)
    ring_r_inner: int = 0
    ring_r_outer: int = 10

    # Which channels receive the ring (paper: single-channel default; RingID uses ch 3 for ring)
    ring_channels: tuple[int, ...] = (3,)
    secondary_gaussian_channels: tuple[int, ...] = ()

    center: CenterConvention = CenterConvention.RING_ID_ALIGNED
    rouder_ring: bool = True  # §5.2 “rounder ring” trajectory method

    embed_mode: EmbedMode = EmbedMode.RING_GAUSSIAN

    fft_norm: FourierNorm = FourierNorm.BACKWARD

    # Appendix A.2 variance scale NC(0, N²): N refers to spatial side length in paper notation
    @property
    def complex_normal_N(self) -> int:
        return self.latent_h  # assumes square latent; HW used if rectangular

    # RingID extras (toggle for “RingID faithful” vs “Tree-Ring baseline only”)
    use_discretization: bool = False
    discrete_alpha: float = 64.0  # paper §6.1: ±64

    lossless_ring_real_only: bool = False  # RingID §5.2: real part imprint, imag zero in spectrum

    spatial_shift: bool = False  # multiply by chessboard (-1)^(u+v) before spatial truncate (conceptual Fig. 2)
    spatial_shift_eta: float = 0.85  # §5.2 suppress peak

    extraction: str = "complex_concat"  # "complex_concat" | "real" | "magnitude"

    # Diffusion defaults from paper §6.1
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

    scheduler_eta: float = 0.0  # deterministic DDIM when 0

    # Model ids (Stable Diffusion v2 per §6.1)
    model_id: str = "sd2-community/stable-diffusion-2-1-base"
    revision: str | None = None

    dtype: str = "float16"

    extra: dict[str, Any] = field(default_factory=dict)


def profile_tree_ring_baseline(**overrides: Any) -> WatermarkProfile:
    """Baseline Tree-Ring per paper recap + Table 1 ‘0–10’ ring radius row."""
    p = WatermarkProfile(
        ring_r_inner=0,
        ring_r_outer=10,
        ring_channels=(3,),
        center=CenterConvention.RING_ID_ALIGNED,
        rouder_ring=True,
        embed_mode=EmbedMode.RING_GAUSSIAN,
        use_discretization=False,
        lossless_ring_real_only=False,
        spatial_shift=False,
        num_inference_steps=50,
    )
    _apply_overrides(p, overrides)
    return p


def profile_ringid_default(**overrides: Any) -> WatermarkProfile:
    """Default RingID setup from §6.1 (radii 3–14 on ch3, Gaussian on ch0, ordered extras)."""
    p = WatermarkProfile(
        ring_r_inner=3,
        ring_r_outer=14,
        ring_channels=(3,),
        secondary_gaussian_channels=(0,),
        center=CenterConvention.RING_ID_ALIGNED,
        rouder_ring=True,
        embed_mode=EmbedMode.RING_GAUSSIAN,
        use_discretization=True,
        discrete_alpha=64.0,
        lossless_ring_real_only=True,
        spatial_shift=True,
        spatial_shift_eta=0.85,
        num_inference_steps=50,
    )
    _apply_overrides(p, overrides)
    return p


def profile_mask_complex_gaussian(**overrides: Any) -> WatermarkProfile:
    """Appendix A.2 substitution on mask with NC(0, N²)."""
    p = profile_tree_ring_baseline()
    p.embed_mode = EmbedMode.MASK_COMPLEX_GAUSSIAN
    _apply_overrides(p, overrides)
    return p


def _apply_overrides(p: WatermarkProfile, overrides: dict[str, Any]) -> None:
    for k, v in overrides.items():
        if hasattr(p, k):
            setattr(p, k, v)
        else:
            p.extra[k] = v


def fft_norm_kwarg(profile: WatermarkProfile) -> dict[str, str]:
    nk = _fft_norm_torch_key(profile.fft_norm)
    if nk is None:
        return {}
    return {"norm": nk}


def watermark_profile_from_dict(data: dict[str, Any]) -> WatermarkProfile:
    """Rehydrate a profile saved with `profile_to_flat_dict`."""

    dd = dict(data)

    fft_norm_raw = dd.pop("fft_norm", FourierNorm.BACKWARD.value)
    fft_norm = FourierNorm(fft_norm_raw) if isinstance(fft_norm_raw, str) else FourierNorm(str(fft_norm_raw))

    embed_raw = dd.pop("embed_mode", EmbedMode.RING_GAUSSIAN.value)
    embed_mode = EmbedMode(embed_raw) if isinstance(embed_raw, str) else EmbedMode(str(embed_raw))

    center_raw = dd.pop("center", CenterConvention.RING_ID_ALIGNED.value)
    center = CenterConvention(center_raw) if isinstance(center_raw, str) else CenterConvention(str(center_raw))

    for k in ("ring_channels", "secondary_gaussian_channels"):
        if k in dd and isinstance(dd[k], list):
            dd[k] = tuple(int(x) for x in dd[k])

    wp = WatermarkProfile(fft_norm=fft_norm, embed_mode=embed_mode, center=center, extra={})
    handled = {"fft_norm", "embed_mode", "center"}
    for fd in fields(WatermarkProfile):
        if fd.name in handled:
            continue
        if fd.name == "extra":
            continue
        if fd.name in dd:
            setattr(wp, fd.name, dd[fd.name])

    for k_unknown, v in dd.items():
        if k_unknown not in {f.name for f in fields(WatermarkProfile)}:
            wp.extra[k_unknown] = v
    return wp


def profile_from_watermark_key(payload: dict[str, Any]) -> WatermarkProfile:
    """Convenience: build profile from deserialized key JSON object."""

    return watermark_profile_from_dict(payload.get("profile", {}))
