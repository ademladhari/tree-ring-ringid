"""Distance-based verification / identification helpers (paper §3.3: ℓ1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from ringid.config import WatermarkProfile, watermark_profile_from_dict
from ringid.inversion import invert_image_to_noise
from ringid.watermark import WatermarkKey, extract_pattern

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[misc,assignment]


def load_pil_rgb(path: Path | str):
    if Image is None:
        raise ImportError("`pillow` is required to load raster images.")
    return Image.open(Path(path)).convert("RGB")


def l1(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    a = vec_a.flatten().float()
    b = vec_b.flatten().float()

    mn = min(a.numel(), b.numel())
    if mn == 0:
        return float("nan")
    return torch.abs(a[:mn] - b[:mn]).sum().item()


def roc_auc_from_distances(d_wm: Iterable[float], d_clean: Iterable[float]) -> float:
    """ROC-AUC assuming smaller distances ⇒ watermark-like ⇒ score = -distance."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError as exc:
        raise ImportError("`scikit-learn` needed (extras: `[eval]`).") from exc

    wm = np.asarray(list(d_wm), dtype=np.float64)
    ck = np.asarray(list(d_clean), dtype=np.float64)
    preds = np.concatenate([-wm, -ck])
    labels = np.concatenate([np.ones(len(wm)), np.zeros(len(ck))])
    try:
        return float(roc_auc_score(labels, preds))
    except ValueError:
        return float("nan")


def tpr_at_fpr_from_distances(d_wm: Iterable[float], d_clean: Iterable[float], fpr: float = 0.01) -> float:
    try:
        from sklearn.metrics import roc_curve
    except ImportError as exc:

        raise ImportError("`scikit-learn` needed.") from exc



    wm = np.asarray(list(d_wm), dtype=np.float64)
    ck = np.asarray(list(d_clean), dtype=np.float64)
    preds = np.concatenate([-wm, -ck])
    labels = np.concatenate([np.ones(len(wm)), np.zeros(len(ck))])


    fp_r, tp_r, _ = roc_curve(labels, preds)
    ix = int(np.clip(np.searchsorted(fp_r, fpr), 0, len(tp_r) - 1))



    return float(tp_r[ix])




def pattern_from_noise_hwc(lat_hwc: torch.Tensor, profile: WatermarkProfile) -> torch.Tensor:



    return extract_pattern(lat_hwc, profile)["vector"]




def invert_then_pattern(
    pipe,

    *,
    pil_image,

    prompt: str,
    negative_prompt: str,

    profile: WatermarkProfile,




    guidance_scale: float | None = None,

    num_inference_steps: int | None = None,

) -> torch.Tensor:



    lat_bchw = invert_image_to_noise(

        pipe,

        pil_image=pil_image,
        prompt=prompt,
        negative_prompt=negative_prompt or "",
        profile=profile,

        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
    )


    hwc = lat_bchw[0].permute(1, 2, 0).contiguous().float().cpu()
    return extract_pattern(hwc, profile)["vector"]



def verification_distances_vs_ref(
    w_hat_vec: torch.Tensor,
    *,
    genuine_key_json: Path | str,
    null_hat_vec: torch.Tensor | None = None,
) -> dict[str, Any]:
    genuine = WatermarkKey.load_json(genuine_key_json)
    watermark_profile_from_dict(genuine.profile_dict)  # validate JSON enums / fields eagerly

    reference = genuine.vector.detach().flatten().cpu().float()
    cand = w_hat_vec.detach().flatten().cpu().float()
    out = {"d_wm_to_w": float(l1(cand, reference)), "candidate_dim": int(cand.numel()), "reference_dim": int(reference.numel())}

    out["profiles_match_dimensions"] = bool(out["candidate_dim"] == out["reference_dim"])

    if null_hat_vec is not None:
        n = null_hat_vec.flatten().cpu().float()
        out["d_wphi_to_w"] = float(l1(n, reference))

    return out


def identification_argmin(dists: dict[str | int, float]) -> tuple[str | int, float]:
    k = min(dists.keys(), key=lambda kk: dists[kk])
    return k, float(dists[k])


def summarize_floats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def verify_images_aggregate(
    pipe,
    candidate_paths: list[Path | str],
    *,
    prompt: str,
    negative_prompt: str,
    profile: WatermarkProfile,
    genuine_key_json: Path | str,
    guidance_scale: float | None = None,
    num_inference_steps: int | None = None,
    null_image_path: Path | str | None = None,
    null_prompt: str | None = None,
) -> dict[str, Any]:
    """Invert + extract each candidate; return per-image scores and aggregates for ``d_wm_to_w``."""

    null_vec: torch.Tensor | None = None
    if null_image_path is not None:
        nprompt = null_prompt or prompt
        null_vec = invert_then_pattern(
            pipe,
            pil_image=load_pil_rgb(null_image_path),
            prompt=nprompt,
            negative_prompt=negative_prompt,
            profile=profile,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )

    per_image: list[dict[str, Any]] = []
    d_wm_list: list[float] = []
    d_null_list: list[float] = []

    for p in candidate_paths:
        vec = invert_then_pattern(
            pipe,
            pil_image=load_pil_rgb(p),
            prompt=prompt,
            negative_prompt=negative_prompt,
            profile=profile,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )
        scores = verification_distances_vs_ref(vec, genuine_key_json=genuine_key_json, null_hat_vec=null_vec)
        per_image.append({"path": str(Path(p)), **scores})
        d_wm_list.append(float(scores["d_wm_to_w"]))
        if "d_wphi_to_w" in scores:
            d_null_list.append(float(scores["d_wphi_to_w"]))

    out: dict[str, Any] = {
        "per_image": per_image,
        "aggregate_d_wm_to_w": summarize_floats(d_wm_list),
    }
    if d_null_list:
        out["aggregate_d_wphi_to_w"] = summarize_floats(d_null_list)
    return out
