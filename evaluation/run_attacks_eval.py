#!/usr/bin/env python3
"""Attack robustness evaluation for Tree-Ring/RingID verification."""

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

from ringid.attacks import compose_paper_bundle  # noqa: E402
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


def _resolve(p: Path, base: Path) -> Path:
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _manifest_bundle(man: dict, base: Path) -> Path:
    if man.get("bundle_dir"):
        return (base / str(man["bundle_dir"])).resolve()
    return (base / "batch_run").resolve()


def _manifest_wm_paths(man: dict, base: Path) -> list[Path]:
    if man.get("bundle_dir") and man.get("images") and man["images"] and man["images"][0].get("filename") is not None:
        bundle = _manifest_bundle(man, base)
        return sorted((bundle / str(e["filename"])).resolve() for e in man.get("images", []))
    out: list[Path] = []
    for e in man.get("images", []):
        p = Path(str(e.get("path", "")))
        if not p:
            continue
        out.append((base / p).resolve() if not p.is_absolute() else p.resolve())
    return sorted(out)


def _generate_clean(pipe, profile: WatermarkProfile, prompt: str, negative_prompt: str, n: int, seed_base: int):
    import torch

    height = int(getattr(profile, "latent_h", 64)) * 8
    width = int(getattr(profile, "latent_w", 64)) * 8
    device = pipe.device
    dtype = pipe.unet.dtype
    gens = [torch.Generator(device=device).manual_seed(seed_base + i) for i in range(n)]
    latents = pipe.prepare_latents(
        n,
        pipe.unet.config.in_channels,
        height,
        width,
        dtype,
        device,
        gens,
        latents=None,
    )
    out = pipe(
        prompt=[prompt] * n,
        negative_prompt=[negative_prompt or ""] * n,
        height=height,
        width=width,
        num_inference_steps=profile.num_inference_steps,
        guidance_scale=profile.guidance_scale,
        latents=latents,
        generator=gens,
        eta=float(profile.scheduler_eta),
    )
    return out.images


def _score_image(pipe, profile: WatermarkProfile, key_path: Path, pil_img, prompt: str, negative_prompt: str) -> float:
    vec = invert_then_pattern(
        pipe,
        pil_image=pil_img,
        prompt=prompt,
        negative_prompt=negative_prompt,
        profile=profile,
    )
    row = verification_distances_vs_ref(vec, genuine_key_json=key_path)
    return float(row["d_wm_to_w"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run paper-style attack robustness evaluation and write readable report.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "batch_run" / "manifest.json")
    parser.add_argument("--out", type=Path, default=ROOT / "evaluation" / "attack_results")
    parser.add_argument("--n-clean", type=int, default=10)
    parser.add_argument("--max-wm", type=int, default=10)
    parser.add_argument("--clean-seed-base", type=int, default=20000)
    parser.add_argument("--attack-seed-base", type=int, default=1234)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    args = parser.parse_args()

    man_path = _resolve(args.manifest, ROOT)
    if not man_path.is_file():
        print(f"ERROR: manifest not found: {man_path}", file=sys.stderr)
        return 2
    man = json.loads(man_path.read_text(encoding="utf-8"))

    bundle = _manifest_bundle(man, ROOT)
    key_rel = str(man.get("key_json", "key.json"))
    key_path = (bundle / key_rel).resolve() if not Path(key_rel).is_absolute() else Path(key_rel).resolve()
    if not key_path.is_file():
        print(f"ERROR: key JSON not found: {key_path}", file=sys.stderr)
        return 2

    wm_paths = [p for p in _manifest_wm_paths(man, ROOT) if p.is_file()]
    if not wm_paths:
        print("ERROR: no watermarked images found in manifest bundle.", file=sys.stderr)
        return 3
    wm_paths = wm_paths[: max(1, int(args.max_wm))]

    prompt = str(man.get("prompt", ""))
    negative_prompt = str(man.get("negative_prompt", ""))
    if not prompt:
        print("ERROR: empty prompt in manifest.", file=sys.stderr)
        return 3

    key_obj = WatermarkKey.load_json(key_path)
    profile: WatermarkProfile = watermark_profile_from_dict(key_obj.profile_dict)
    pipe = load_pipeline(profile)

    wm_imgs = [load_pil_rgb(p) for p in wm_paths]
    clean_imgs = _generate_clean(pipe, profile, prompt, negative_prompt, int(args.n_clean), int(args.clean_seed_base))

    attack_names = ["clean", "jpeg25", "rotate75", "cs75", "blurK8", "noise01", "brightness06"]
    results: dict[str, dict] = {}

    for attack_name in attack_names:
        wm_scores: list[float] = []
        clean_scores: list[float] = []

        for i, img in enumerate(wm_imgs):
            attacked = compose_paper_bundle(img, seed=int(args.attack_seed_base) + i)[attack_name]
            wm_scores.append(_score_image(pipe, profile, key_path, attacked, prompt, negative_prompt))

        for j, img in enumerate(clean_imgs):
            attacked = compose_paper_bundle(img, seed=int(args.attack_seed_base) + 10_000 + j)[attack_name]
            clean_scores.append(_score_image(pipe, profile, key_path, attacked, prompt, negative_prompt))

        auc = roc_auc_from_distances(wm_scores, clean_scores)
        tpr = tpr_at_fpr_from_distances(wm_scores, clean_scores, fpr=float(args.target_fpr))
        agg_wm = summarize_floats(wm_scores)
        agg_clean = summarize_floats(clean_scores)
        gap = float(agg_clean["mean"] - agg_wm["mean"])

        results[attack_name] = {
            "n_wm": len(wm_scores),
            "n_clean": len(clean_scores),
            "wm_summary": agg_wm,
            "clean_summary": agg_clean,
            "mean_gap_clean_minus_wm": gap,
            "roc_auc": auc,
            f"tpr_at_fpr_{args.target_fpr}": tpr,
        }

    out_dir = _resolve(args.out, ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(man_path),
        "key_json": str(key_path),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "n_wm_used": len(wm_paths),
        "n_clean_generated": int(args.n_clean),
        "attacks": results,
    }
    (out_dir / "attack_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Attack Evaluation")
    lines.append("")
    lines.append(f"- Manifest: `{man_path}`")
    lines.append(f"- Key: `{key_path}`")
    lines.append(f"- n_wm={len(wm_paths)}, n_clean={int(args.n_clean)}")
    lines.append("")
    lines.append("| Attack | ROC-AUC | TPR@1%FPR | mean(wm d) | mean(clean d) | mean_gap(clean-wm) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for k in attack_names:
        r = results[k]
        auc = r["roc_auc"]
        tpr = r.get(f"tpr_at_fpr_{args.target_fpr}", float("nan"))
        mw = r["wm_summary"]["mean"]
        mc = r["clean_summary"]["mean"]
        gap = r["mean_gap_clean_minus_wm"]
        lines.append(
            f"| {k} | {auc:.4f} | {tpr:.4f} | {mw:.2f} | {mc:.2f} | {gap:.2f} |"
            if not math.isnan(float(auc))
            else f"| {k} | nan | nan | {mw:.2f} | {mc:.2f} | {gap:.2f} |"
        )
    lines.append("")
    lines.append("Smaller distance is better for watermarked images; a positive gap means clean images are farther from the key.")
    (out_dir / "attack_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"wrote": str(out_dir), "n_attacks": len(attack_names), "n_wm": len(wm_paths), "n_clean": int(args.n_clean)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
