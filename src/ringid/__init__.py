"""Tree-Ring-style and RingID-style latent watermarking utilities."""

from ringid.config import (
    CenterConvention,
    EmbedMode,
    FourierNorm,
    WatermarkProfile,
    profile_mask_complex_gaussian,
    profile_ringid_default,
    profile_tree_ring_baseline,
    watermark_profile_from_dict,
)
from ringid.detect import summarize_floats, verify_images_aggregate
from ringid.sampling import generate_watermarked, generate_watermarked_batch, load_pipeline
from ringid.watermark import WatermarkKey, build_ring_shell_masks, extract_pattern, inject_watermark

__all__ = [
    "CenterConvention",
    "EmbedMode",
    "FourierNorm",
    "WatermarkProfile",
    "profile_ringid_default",
    "profile_tree_ring_baseline",
    "profile_mask_complex_gaussian",
    "watermark_profile_from_dict",
    "WatermarkKey",
    "build_ring_shell_masks",
    "inject_watermark",
    "extract_pattern",
    "generate_watermarked",
    "generate_watermarked_batch",
    "load_pipeline",
    "summarize_floats",
    "verify_images_aggregate",
]
