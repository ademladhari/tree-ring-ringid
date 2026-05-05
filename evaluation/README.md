# Evaluation pipeline

`run_pipeline.py` reproduces a **paper-style verification snapshot**:

1. Reads `batch_run/manifest.json` (prompt, paths, `key.json`).
2. For each **watermarked** PNG: DDIM inversion → FFT pattern → **ℓ₁** to the reference key (`d_wm_to_w`).
3. Generates **N clean** images (same prompt, **no** tree-ring imprint) and computes the same ℓ₁ to $w$ (negative class).
4. Builds a **null** $\hat w_\emptyset$ from the first clean image and reports **`d_wphi_to_w`** next to each watermarked row.
5. Writes **`report.md`** (human-readable), **`metrics.json`**, and distance lists for `ringid-eval`.

## Run (cmd.exe, repo root)

```bat
cd /d c:\Users\ladha\Desktop\thesis\tree-ring-ringid
.venv\Scripts\activate.bat
py -3 evaluation\run_pipeline.py --manifest batch_run\manifest.json --out evaluation\results
```

Optional:

```bat
py -3 evaluation\run_pipeline.py --manifest batch_run\manifest.json --wm-glob "batch_run\img_*.png" --out evaluation\results --n-clean 10 --clean-seed-base 10000
```

Requires **watermarked PNGs** on disk (run `ringid-generate --count 10 --out-dir batch_run ...` first). New manifests include `bundle_dir` plus per-image `filename` so paths stay consistent. Older manifests with full `path` / `key_json` paths are still supported. The script saves **clean\_\*.png** under `--out` (default `evaluation/results`).

## ROC helper on the saved lists

```bat
ringid-eval --watermarked-distances evaluation\results\watermarked_distances.txt --clean-distances evaluation\results\clean_distances.txt
```
