"""CLI: verify watermark via latent inversion + ℓ1 (single image or batch average)."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from ringid.config import profile_ringid_default, profile_tree_ring_baseline, watermark_profile_from_dict
from ringid.detect import invert_then_pattern, load_pil_rgb, verification_distances_vs_ref, verify_images_aggregate
from ringid.sampling import load_pipeline
from ringid.watermark import WatermarkKey


def _candidate_paths(args: argparse.Namespace) -> list[str]:
    if args.candidate:
        return [args.candidate]
    if args.candidates:
        return list(args.candidates)
    if args.glob_pattern:
        found = sorted(glob.glob(args.glob_pattern, recursive=False))
        if not found:
            raise SystemExit(f"No files matched --glob {args.glob_pattern!r}")
        return found
    raise SystemExit("Provide --candidate, or --candidates …, or --glob 'path/*.png'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Invert image(s), extract spectral pattern, compare ℓ1 to key JSON.")
    parser.add_argument("--candidate", default=None, help="Single image path.")
    parser.add_argument("--candidates", nargs="+", default=None, help="Multiple image paths (batch verify).")
    parser.add_argument("--glob", dest="glob_pattern", default=None, help="Glob of images, e.g. runs/batch/img_*.png")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="With multiple images, include full per-image score dicts (larger JSON).",
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--profile", choices=("tree_ring", "ring_id", "from_key"), default="from_key")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--genuine-key-json", required=True)
    parser.add_argument("--null-image", default=None)
    parser.add_argument("--null-prompt", default=None)
    parser.add_argument("--inversion-steps", type=int, default=None)
    parser.add_argument("--inversion-guidance-scale", type=float, default=None)
    args = parser.parse_args()

    paths = _candidate_paths(args)

    wm_key_preview = WatermarkKey.load_json(args.genuine_key_json)
    if args.profile == "from_key":
        wm_profile = watermark_profile_from_dict(wm_key_preview.profile_dict)
    elif args.profile == "tree_ring":
        wm_profile = profile_tree_ring_baseline()
    else:
        wm_profile = profile_ringid_default()

    if args.model_id:
        wm_profile.model_id = args.model_id

    pipe = load_pipeline(wm_profile)

    if len(paths) == 1:
        w_hat = invert_then_pattern(
            pipe,
            pil_image=load_pil_rgb(paths[0]),
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            profile=wm_profile,
            num_inference_steps=args.inversion_steps,
            guidance_scale=args.inversion_guidance_scale,
        )
        null_vec = None
        if args.null_image:
            null_vec = invert_then_pattern(
                pipe,
                pil_image=load_pil_rgb(args.null_image),
                prompt=args.null_prompt or args.prompt,
                negative_prompt=args.negative_prompt,
                profile=wm_profile,
                num_inference_steps=args.inversion_steps,
                guidance_scale=args.inversion_guidance_scale,
            )
        scores = verification_distances_vs_ref(w_hat, genuine_key_json=args.genuine_key_json, null_hat_vec=null_vec)
        print(json.dumps(scores, indent=2))
        return

    report = verify_images_aggregate(
        pipe,
        paths,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        profile=wm_profile,
        genuine_key_json=args.genuine_key_json,
        guidance_scale=args.inversion_guidance_scale,
        num_inference_steps=args.inversion_steps,
        null_image_path=args.null_image,
        null_prompt=args.null_prompt,
    )
    slim = {
        "n_images": len(paths),
        "aggregate_d_wm_to_w": report["aggregate_d_wm_to_w"],
        "per_image": [
            {"path": row["path"], "d_wm_to_w": row["d_wm_to_w"], "d_wphi_to_w": row.get("d_wphi_to_w")}
            for row in report["per_image"]
        ],
    }
    if "aggregate_d_wphi_to_w" in report:
        slim["aggregate_d_wphi_to_w"] = report["aggregate_d_wphi_to_w"]
    if args.verbose:
        slim["per_image_full"] = report["per_image"]
    print(json.dumps(slim, indent=2))


if __name__ == "__main__":
    main()
