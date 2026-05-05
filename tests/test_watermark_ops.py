import torch
from PIL import Image

from ringid.attacks import crop_and_scale, jpeg_compress
from ringid.config import profile_tree_ring_baseline, watermark_profile_from_dict
from ringid.detect import identification_argmin, l1, summarize_floats

from pathlib import Path

from ringid.watermark import (
    WatermarkKey,
    build_ring_shell_masks,
    inject_watermark,
    merge_shells,
    profile_to_flat_dict,
)


def test_ring_masks_non_empty():
    profile = profile_tree_ring_baseline()
    shells = build_ring_shell_masks(64, 64, profile)


    merged = merge_shells(shells)
    area = float(merged.sum())
    assert area > 10.0
    assert area < 0.5 * 64 * 64


def test_watermark_embed_deterministic():
    profile = profile_tree_ring_baseline()
    latent = torch.randn(64, 64, 4)

    out1, k1, _ = inject_watermark(latent.clone(), profile, generator=torch.Generator().manual_seed(123), seed=999)
    out2, k2, _ = inject_watermark(latent.clone(), profile, generator=torch.Generator().manual_seed(123), seed=999)

    assert torch.equal(out1, out2)



    assert torch.allclose(k1.vector, k2.vector)


def test_key_json_roundtrip(tmp_path):

    kk = WatermarkKey(vector=torch.arange(16).float(), profile_dict={}, seed=0)



    outp = Path(tmp_path) / "wk.json"


    kk.save_json(outp)


    kk2 = WatermarkKey.load_json(outp)



    assert torch.allclose(kk.vector, kk2.vector)


def test_profile_roundtrip():
    base = profile_tree_ring_baseline()
    restored = watermark_profile_from_dict(profile_to_flat_dict(base))
    assert restored.embed_mode.value == base.embed_mode.value




    assert restored.ring_r_outer == base.ring_r_outer


def test_l1():

    aa = torch.tensor([1.0, -2.0, 3.0])


    bb = torch.tensor([0.5, -1.0, 3.25])



    assert abs(l1(aa, bb) - 1.75) < 1e-5


def test_summarize_floats():
    s = summarize_floats([1.0, 2.0, 3.0, 4.0])
    assert s["n"] == 4 and abs(s["mean"] - 2.5) < 1e-6


def test_identification_argmin():
    kk, vv = identification_argmin({"k0": 4.5, "k1": 0.1})



    assert kk == "k1"



    assert abs(vv - 0.1) < 1e-9


def test_shared_watermark_seed_same_reference_vector():
    """Same watermark RNG seed on different latents → identical flattened reference key (batch setting)."""

    profile = profile_tree_ring_baseline()
    lat_a = torch.randn(64, 64, 4)
    lat_b = torch.randn(64, 64, 4)
    _, k_a, _ = inject_watermark(lat_a, profile, generator=None, seed=999)
    _, k_b, _ = inject_watermark(lat_b, profile, generator=None, seed=999)
    assert torch.allclose(k_a.vector, k_b.vector)


def test_attacks_resize_shape():
    im = Image.new("RGB", (128, 64), color=(127, 5, 9))





    cq = jpeg_compress(im, quality=30)


    rz = crop_and_scale(im, 0.75, rng=__import__("random").Random(0))



    assert cq.size == im.size




    assert rz.size == im.size
