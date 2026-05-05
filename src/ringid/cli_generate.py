"""CLI: watermarked Stable Diffusion generation (single or batched)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ringid.config import profile_ringid_default, profile_tree_ring_baseline
from ringid.sampling import generate_watermarked, generate_watermarked_batch, load_pipeline


def _build_profile(args: argparse.Namespace):
    profile = profile_tree_ring_baseline() if args.profile == "tree_ring" else profile_ringid_default()
    profile.dtype = args.dtype
    if args.model_id:
        profile.model_id = args.model_id
    if args.steps is not None:
        profile.num_inference_steps = int(args.steps)
    if args.guidance_scale is not None:
        profile.guidance_scale = float(args.guidance_scale)
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate with Tree-Ring / RingID-style latent watermark (1 image or a batch of N in one forward pass)."
    )
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--seed", type=int, default=0, help="Legacy: when --count=1, used for latent + watermark RNG.")
    parser.add_argument("--latent-seed-base", type=int, default=None, help="Base seed for per-image latent noise (default: --seed).")
    parser.add_argument("--watermark-seed", type=int, default=None, help="Seed for spectral watermark RNG (shared across batch; default: --seed).")
    parser.add_argument("--count", type=int, default=1, help="Number of images to generate in one batched UNet run (e.g. 10).")
    parser.add_argument("--profile", choices=("tree_ring", "ring_id"), default="tree_ring")
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--out", type=str, default="wm_out.png", help="Output PNG when --count=1.")
    parser.add_argument("--key-out", type=str, default="key.json", help="Reference key JSON when --count=1.")
    parser.add_argument("--out-dir", type=str, default=None, help="Required when --count>1: directory for img_XX.png + key.json + manifest.")
    parser.add_argument("--manifest-out", type=str, default=None, help="Optional manifest path (default: <out-dir>/manifest.json when batch).")
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()

    wm_profile = _build_profile(args)
    pipe = load_pipeline(wm_profile)

    latent_base = int(args.latent_seed_base if args.latent_seed_base is not None else args.seed)
    wm_seed = int(args.watermark_seed if args.watermark_seed is not None else args.seed)
    n = int(args.count)

    if n < 1:
        raise SystemExit("--count must be >= 1")

    if n == 1:
        image, _lat, key = generate_watermarked(
            pipe,
            wm_profile,
            args.prompt,
            negative_prompt=args.negative_prompt or "",
            height=args.height,
            width=args.width,
            seed=args.seed,
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        image.save(args.out)
        key.save_json(args.key_out)
        print({"saved_png": args.out, "saved_key_json": args.key_out, "count": 1})
        return

    if not args.out_dir:
        raise SystemExit("When --count>1, pass --out-dir to store multiple PNGs and a shared key.json.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images, _lat_b, key = generate_watermarked_batch(
        pipe,
        wm_profile,
        args.prompt,
        negative_prompt=args.negative_prompt or "",
        height=args.height,
        width=args.width,
        n_images=n,
        latent_seed_base=latent_base,
        watermark_seed=wm_seed,
    )

    bundle_dir = str(Path(args.out_dir))
    manifest: dict[str, object] = {
        "count": n,
        "bundle_dir": bundle_dir,
        "latent_seed_base": latent_base,
        "watermark_seed": wm_seed,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "profile": args.profile,
        "images": [],
    }

    for i, img in enumerate(images):
        name = f"img_{i:02d}.png"
        pth = out_dir / name
        img.save(pth)
        manifest["images"].append({"index": i, "filename": name, "latent_seed": latent_base + i})

    key_path = out_dir / "key.json"
    key.save_json(key_path)
    manifest["key_json"] = "key.json"

    man_path = Path(args.manifest_out) if args.manifest_out else out_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        {
            "out_dir": str(out_dir),
            "count": n,
            "key_json": str(key_path),
            "manifest": str(man_path),
        }
    )


if __name__ == "__main__":
    main()
