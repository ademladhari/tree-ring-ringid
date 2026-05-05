#!/usr/bin/env python3
"""
End-to-end verification-style evaluation (paper §3.3–4.1 flavour):

  1) Load manifest + key JSON (same prompt / profile as generation).
  2) Invert each watermarked image → spectral pattern → ℓ1 to reference key (d_wm_to_w).
  3) Generate N unwatermarked images (same prompt, no spectral imprint) → same ℓ1 to key (negative class).
  4) Summarise (mean/std/min/max), separation gap, ROC-AUC, TPR@FPR.
  5) Write readable Markdown + JSON + distance lists under --out.

Run from repo root:

  py -3 evaluation/run_pipeline.py --manifest batch_run/manifest.json --out evaluation/results
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from ringid.config import WatermarkProfile, watermark_profile_from_dict  # noqa: E402
from ringid.detect import (  # noqa: E402
    invert_then_pattern,
    load_pil_rgb,
    roc_auc_from_distances,
    summarize_floats,
    tpr_at_fpr_from_distances,
    verification_distances_vs_ref,
)
from ringid.sampling import load_pipeline  # noqa: E402
from ringid.watermark import WatermarkKey  # noqa: E402


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(p: Path, base: Path) -> Path:
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _repo_relative(s: str, base: Path) -> Path:
    """Paths in `manifest.json` from `ringid-generate` are relative to the repo cwd, not the manifest folder."""

    p = Path(s)
    if p.is_absolute():
        return p.resolve()
    return (base / p).resolve()


def _generate_clean_set(
    pipe,
    profile: WatermarkProfile,
    *,
    prompt: str,
    negative_prompt: str,
    height: int,
    width: int,
    n_clean: int,
    seed_base: int,
    out_dir: Path,
) -> list[Path]:
    device = pipe.device
    dtype = pipe.unet.dtype
    gens = [torch.Generator(device=torch.device(device)).manual_seed(seed_base + i) for i in range(n_clean)]
    latents = pipe.prepare_latents(
        n_clean,
        pipe.unet.config.in_channels,
        height,
        width,
        dtype,
        device,
        gens,
        latents=None,
    )
    prompts = [prompt] * n_clean
    negs = [negative_prompt or ""] * n_clean
    out = pipe(
        prompt=prompts,
        negative_prompt=negs,
        height=height,
        width=width,
        num_inference_steps=profile.num_inference_steps,
        guidance_scale=profile.guidance_scale,
        latents=latents,
        generator=gens,
        eta=float(profile.scheduler_eta),
    )
    paths: list[Path] = []
    for i, img in enumerate(out.images):
        p = out_dir / f"clean_{i:02d}.png"
        img.save(p)
        paths.append(p)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Watermark verification evaluation pipeline (readable reports).")
    parser.add_argument("--manifest", type=Path, default=ROOT / "batch_run" / "manifest.json")
    parser.add_argument("--wm-glob", type=str, default=None, help="Override watermarked PNG glob (default: manifest image paths).")
    parser.add_argument("--out", type=Path, default=ROOT / "evaluation" / "results")
    parser.add_argument("--n-clean", type=int, default=10, help="How many unwatermarked images to generate for the negative class.")
    parser.add_argument("--clean-seed-base", type=int, default=10_000, help="RNG base for clean latents (avoid collision with wm seeds).")
    parser.add_argument("--target-fpr", type=float, default=0.01)
    args = parser.parse_args()

    base = ROOT
    manifest_path = _resolve(args.manifest, base)
    if not manifest_path.is_file():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    man = _load_manifest(manifest_path)
    if man.get("bundle_dir") and isinstance(man.get("key_json"), str):
        bundle = (base / str(man["bundle_dir"])).resolve()
        key_path = (bundle / str(man["key_json"])).resolve()
    else:
        key_path = _repo_relative(str(man["key_json"]), base)
    if not key_path.is_file():
        print(f"ERROR: key JSON not found: {key_path}", file=sys.stderr)
        return 2

    prompt = str(man.get("prompt", ""))
    negative_prompt = str(man.get("negative_prompt", ""))
    if not prompt.strip():
        print("ERROR: manifest has empty prompt.", file=sys.stderr)
        return 2

    key_obj = WatermarkKey.load_json(key_path)
    profile: WatermarkProfile = watermark_profile_from_dict(key_obj.profile_dict)

    height = int(getattr(profile, "latent_h", 64)) * 8
    width = int(getattr(profile, "latent_w", 64)) * 8

    if args.wm_glob:
        import glob as glob_mod

        pat = args.wm_glob if Path(args.wm_glob).is_absolute() else str(base / args.wm_glob)
        wm_paths = sorted(Path(p) for p in glob_mod.glob(pat, recursive=False))
    elif man.get("bundle_dir") and man.get("images") and man["images"] and man["images"][0].get("filename") is not None:
        bundle = (base / str(man["bundle_dir"])).resolve()
        wm_paths = sorted(bundle / str(e["filename"]) for e in man.get("images", []))
    else:
        wm_paths = sorted(_repo_relative(str(entry["path"]), base) for entry in man.get("images", []))

    wm_paths = [p for p in wm_paths if p.suffix.lower() in (".png", ".jpg", ".jpeg") and p.is_file()]
    if not wm_paths:
        print(
            "ERROR: no watermarked images found. Generate them first, e.g.\n"
            "  ringid-generate --prompt \"...\" --count 10 --out-dir batch_run --latent-seed-base 0 --watermark-seed 42",
            file=sys.stderr,
        )
        return 3

    out_dir = _resolve(args.out, base)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline(profile)

    clean_paths = _generate_clean_set(
        pipe,
        profile,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        n_clean=int(args.n_clean),
        seed_base=int(args.clean_seed_base),
        out_dir=out_dir,
    )

    null_path = clean_paths[0]
    null_vec = invert_then_pattern(
        pipe,
        pil_image=load_pil_rgb(null_path),
        prompt=prompt,
        negative_prompt=negative_prompt,
        profile=profile,
    )

    wm_with_null: list[dict[str, float | str]] = []
    wm_scores: list[float] = []
    for p in wm_paths:
        vec = invert_then_pattern(
            pipe,
            pil_image=load_pil_rgb(p),
            prompt=prompt,
            negative_prompt=negative_prompt,
            profile=profile,
        )
        row = verification_distances_vs_ref(vec, genuine_key_json=key_path, null_hat_vec=null_vec)
        d_wm = float(row["d_wm_to_w"])
        wm_scores.append(d_wm)
        wm_with_null.append(
            {
                "path": str(p),
                "d_wm_to_w": d_wm,
                "d_wphi_to_w": float(row["d_wphi_to_w"]),
            }
        )

    clean_scores: list[float] = []
    for p in clean_paths:
        vec = invert_then_pattern(
            pipe,
            pil_image=load_pil_rgb(p),
            prompt=prompt,
            negative_prompt=negative_prompt,
            profile=profile,
        )
        row = verification_distances_vs_ref(vec, genuine_key_json=key_path, null_hat_vec=None)
        clean_scores.append(float(row["d_wm_to_w"]))

    agg_wm = summarize_floats(wm_scores)
    agg_clean = summarize_floats(clean_scores)
    gap = float(agg_clean["mean"] - agg_wm["mean"])

    try:
        auc = roc_auc_from_distances(wm_scores, clean_scores)
        tpr = tpr_at_fpr_from_distances(wm_scores, clean_scores, fpr=float(args.target_fpr))
    except Exception as exc:  # pragma: no cover
        auc = float("nan")
        tpr = float("nan")
        roc_err = str(exc)
    else:
        roc_err = ""

    (out_dir / "watermarked_distances.txt").write_text("\n".join(str(x) for x in wm_scores) + "\n", encoding="utf-8")
    (out_dir / "clean_distances.txt").write_text("\n".join(str(x) for x in clean_scores) + "\n", encoding="utf-8")

    metrics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "key_json": str(key_path),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "n_watermarked": len(wm_scores),
        "n_clean": len(clean_scores),
        "aggregate_watermarked_d_wm_to_w": agg_wm,
        "aggregate_clean_d_wm_to_w": agg_clean,
        "mean_gap_clean_minus_wm": gap,
        "roc_auc_neg_l1_as_score": auc,
        f"tpr_at_fpr_{args.target_fpr}": tpr,
        "roc_error": roc_err or None,
        "null_reference_image": str(null_path),
        "per_watermarked_with_null": wm_with_null,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    def _fmt(d: dict) -> str:
        return ", ".join(f"{k}={v:.4g}" if isinstance(v, float) and not math.isnan(v) else f"{k}={v}" for k, v in d.items())

    report = f"""# Watermark verification evaluation

Generated (UTC): `{metrics["generated_at_utc"]}`

## Setup

| Field | Value |
|-------|-------|
| Manifest | `{manifest_path}` |
| Key JSON | `{key_path}` |
| Prompt | {prompt!r} |
| Watermarked images | {len(wm_scores)} |
| Clean (unwatermarked) images | {len(clean_scores)} |
| Null pattern image (for `d_wphi_to_w` column) | `{null_path}` |

## What the numbers mean

- **`d_wm_to_w`**: $\\ell_1$ distance between the **recovered** spectral pattern $\\hat w$ (after DDIM inversion + FFT) and the **reference** key $w$ from `key.json`. **Smaller ⇒ closer to the imprinted key.**
- **`d_wphi_to_w`**: same $\\ell_1$, but $\\hat w$ comes from a **clean** reference image (first clean sample here), matching the paper’s “null” $\\hat w_\\emptyset$ style comparison **when you also care about separation vs a fixed null**.
- **Clean-class scores**: each unwatermarked image is inverted the same way; its $\\hat w$ is usually **farther** from $w$ than a truly watermarked image, so **larger** `d_wm_to_w` on clean images is expected if verification is informative.

## Aggregates (distance to key $w$)

**Watermarked** (`d_wm_to_w`): {_fmt(agg_wm)}

**Clean / unwatermarked** (same metric name in JSON: distance of inverted clean image to the *same* key $w$): {_fmt(agg_clean)}

**Gap** (mean clean − mean watermarked): **{gap:.4f}** (positive ⇒ clean tends to sit farther from $w$ than watermarked, as desired).

## Classifier-style summary (ROC)

Using score $= -\\ell_1$ so **smaller distance ⇒ higher score** (watermarked = positive class):

- **ROC-AUC**: **{auc if not math.isnan(auc) else "nan"}**
- **TPR @ FPR={args.target_fpr}**: **{tpr if not math.isnan(tpr) else "nan"}**
{f"- ROC note: `{roc_err}`" if roc_err else ""}

## Files written

| File | Purpose |
|------|---------|
| `metrics.json` | Machine-readable full result |
| `report.md` | This narrative |
| `watermarked_distances.txt` | One `d_wm_to_w` per line (for `ringid-eval`) |
| `clean_distances.txt` | One distance per line for clean class |
| `clean_XX.png` | Generated unwatermarked references |

## Per-image (watermarked) with null column

| path | d_wm_to_w | d_wphi_to_w |
|------|-----------|-------------|
"""
    for row in wm_with_null:
        report += f"| `{row['path']}` | {row['d_wm_to_w']:.4f} | {row['d_wphi_to_w']:.4f} |\n"

    (out_dir / "report.md").write_text(report, encoding="utf-8")

    print(json.dumps({"wrote": str(out_dir), "roc_auc": auc, "mean_gap": gap, "n_wm": len(wm_scores), "n_clean": len(clean_scores)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
