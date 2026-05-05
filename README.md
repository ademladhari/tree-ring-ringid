# ringid-watermark — Tree-Ring baseline + RingID-style hooks

Faithful-but-explicit reconstruction of latent **Tree-Ring** watermarking (recap §4.1 + appendix discussion in Ci et al., RingID paper) implemented on top of **Hugging Face `diffusers`**.

This codebase does **not** aim to silently guess underspecified FFT scaling / conjugate-frequency bookkeeping—the defaults are spelled out on `WatermarkProfile` and can be swapped for experimentation.

---

## Algorithm mapping (faithfulness cheatsheet)

### Directly anchored in `paper.txt`

| Paper idea | Implemented as |
|-----------|----------------|
| Embed: FFT(latent **`x_T`**) tree-ring **`w`** in spectrum center, IFFT back, **`Re` truncation** §4.1/A.2 summary | [`inject_watermark`](src/ringid/watermark.py) fftshift-aligned ring mask • **two embed modes**: `RING_GAUSSIAN` §4.1 vs `MASK_COMPLEX_GAUSSIAN` appendix A.2 |
| Detection: inversion → FFT • ℓ¹ vs reference / null separation §3–4 | [`invert_then_pattern`](src/ringid/detect.py) + [`verification_distances_vs_ref`](src/ringid/detect.py) |
| Identification §3 | [`identification_argmin`](src/ringid/detect.py) CLI [`cli_identify.py`](src/ringid/cli_identify.py) |
| Stable Diffusion 4 ch latent, single ring channel Fig. 1 default | Defaults `latent_c = 4`, `ring_channels = (3,)` |

### Explicit implementation choices **not uniquely fixed by the excerpt**

Some items require selecting among equally plausible interpretations; they are surfaced as enums/flags:

- FFT orthogonality normalization (`WatermarkProfile.fft_norm`)
- Fourier center placement `(31, 32)` vs `(32, 32)` §5.2 commentary (`CenterConvention`)
- Exact mask construction combining rounder arcs vs filled annulus (`rouder_ring` + Euclidean shells)
- Inversion fidelity: deterministic **DDIMInverseScheduler** trajectory (recommended by Fig. 2 labels) differs from stochastic forward noising shortcuts

Review **your** academic reference implementation if you must bit-match Tree-Ring [41].

---

## Install

Requires Python ≥ 3.10, CUDA optional (CPU painfully slow).

Use a **virtual environment** so `torch` / `diffusers` stay isolated from your system Python.

**Windows (PowerShell)**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[dev,eval]"
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev,eval]"
```

After activation, `ringid-generate` and other console scripts live on that venv’s `PATH`.

### NVIDIA RTX 50-series (e.g. RTX 5060) — PyTorch **CUDA 12.8** (`cu128`)

Install **torch/torchvision from the cu128 wheel index first**, then the project **without** re-resolving `torch` from the default PyPI index (which would often install a CPU build).

**cmd.exe**

```bat
pip install -U pip
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install diffusers>=0.27 transformers>=4.35 accelerate>=0.26 pillow>=10 numpy>=1.24 safetensors>=0.4 pytest scikit-learn
pip install -e . --no-deps
```

See also [`requirements-cu128.txt`](requirements-cu128.txt) for the same recipe in comments.

When you run **`ringid-generate`** (or `generate_watermarked_batch` in Python), the first line printed is a **`[ringid] gen ...`** CUDA / GPU / batch summary so you can confirm the build matches your GPU.

---

## Quick CLI

Installed entry points:

```bash

ringid-generate --prompt "a photo of ..." --out wm.png --key-out wm_key.json



ringid-verify --candidate wm.png --prompt "same prompt text" \

  --genuine-key-json wm_key.json \

  [--null-image clean.png]


ringid-identify --candidate wm.png --prompt "..." \

  --key-json ./keys/dir



ringid-eval --watermarked-distances wm_dists.txt --clean-distances ck_dists.txt

```

`ringid-eval` expects newline-separated floats (one ℓ¹ distance per row) in each file.

### Batch: generate / verify 10 images and average metrics

**Generate 10** images in **one** `StableDiffusionPipeline` forward pass (shared spectral watermark RNG, different latent seeds `latent_seed_base + i`):

```powershell
ringid-generate --prompt "a red bicycle" --count 10 --out-dir ./batch_run `
  --latent-seed-base 0 --watermark-seed 42
```

Writes `batch_run/img_00.png` … `img_09.png`, one shared `batch_run/key.json`, and `batch_run/manifest.json`.

**Verify all 10** and print **mean / std / min / max** of `d_wm_to_w` (per-image list included):

```powershell
ringid-verify --glob "./batch_run/img_*.png" --prompt "a red bicycle" `
  --genuine-key-json ./batch_run/key.json
```

Add `--verbose` for the full per-image score dicts.

---

## Python API essentials

```python

from ringid.config import profile_tree_ring_baseline

from ringid.sampling import load_pipeline, generate_watermarked, generate_watermarked_batch
from ringid.detect import verify_images_aggregate



wm_profile = profile_tree_ring_baseline()

pipe = load_pipeline(wm_profile)

image, latent_bchw, key = generate_watermarked(

    pipe, wm_profile,

    prompt="a scenic mountain lake at dawn",

)


key.save_json("golden_key.json")


```

Batch (10 images, one UNet batch, shared watermark key):

```python

images, latents_b, key = generate_watermarked_batch(

    pipe,

    wm_profile,

    "a scenic mountain lake at dawn",

    n_images=10,

    latent_seed_base=0,

    watermark_seed=42,

)

key.save_json("golden_key.json")


report = verify_images_aggregate(

    pipe,

    [f"batch/img_{i:02d}.png" for i in range(10)],

    prompt="a scenic mountain lake at dawn",

    negative_prompt="",

    profile=wm_profile,

    genuine_key_json="golden_key.json",

)

print(report["aggregate_d_wm_to_w"])


```

Invert + spectral pattern:

```python

from ringid.detect import invert_then_pattern, verification_distances_vs_ref, load_pil_rgb



pattern = invert_then_pattern(

    pipe,

    pil_image=load_pil_rgb("wm.png"),

    prompt="a scenic mountain lake at dawn",

    negative_prompt="",

    profile=wm_profile,

)

scores = verification_distances_vs_ref(pattern, genuine_key_json="golden_key.json")

print(scores)

```

---

## RingID defaults

```python



from ringid.config import profile_ringid_default




wm_profile = profile_ringid_default()  # radii 3‑14 • ch 3 ring • ch 0 Gaussian • discretisation ± 64 hooks


```

Fully faithful RingID bookkeeping (conjugate-even/odd parity + ordered transforms) demands extra algebra—currently we ship **orthogonal toggles**: `spatial_shift`, `lossless_ring_real_only`, `use_discretization`.

---

## Image attacks (evaluation § 6 .1 knobs)

JPEG 25-like quality, ±75 ° arbitrary rotation crop/scale surrogate, Gaussian blur analogue, brightness jitter available in [`ringid.attacks`](src/ringid/attacks.py).

---

## Tests

```

pytest -q

```

---

## Citation cues

Ci et al., **RingID: Rethinking Tree‑Ring Watermarking for Enhanced Multi‑Key Identification** (arXiv:2404.14055 v3) — local copy `paper.txt`. Original Tree‑Ring watermark [41].

---

Happy watermarking 🔐
