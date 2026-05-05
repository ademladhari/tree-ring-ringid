"""Frequency-domain tree-ring watermark construction and latent imprinting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ringid.config import CenterConvention, EmbedMode, FourierNorm, WatermarkProfile, fft_norm_kwarg


@dataclass
class WatermarkKey:
    """Serialized reference pattern for verification / identification (flattened Fourier mask)."""

    vector: torch.Tensor  # 1d float concatenation of flattened [real,imag] spectral bins
    profile_dict: dict[str, Any]
    seed: int
    key_index: int = 0

    def save_json(self, path: str | Path) -> None:
        payload = {
            "vector": self.vector.detach().cpu().float().tolist(),
            "profile": self.profile_dict,
            "seed": self.seed,
            "key_index": self.key_index,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def load_json(path: str | Path) -> "WatermarkKey":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        vec = torch.tensor(data["vector"], dtype=torch.float32)
        return WatermarkKey(
            vector=vec,
            profile_dict=data["profile"],
            seed=data.get("seed", 0),
            key_index=data.get("key_index", 0),
        )


def fft_center_xy(h: int, w: int, center: CenterConvention) -> tuple[float, float]:
    if center == CenterConvention.RING_ID_ALIGNED:
        return float(h / 2.0), float(w / 2.0)
    return float(h / 2.0 - 1.0), float(w / 2.0)


def rounder_trajectory_shell(h: int, w: int, cy: float, cx: float, r: float, num_steps: int = 720) -> torch.Tensor:
    """§5.2: approximate a continuous circle at radius r on a low-res grid (CPU bool mask)."""

    mask = np.zeros((h, w), dtype=bool)
    theta = np.linspace(0.0, 2 * np.pi, num_steps, endpoint=False, dtype=np.float64)
    ys = cy + np.sin(theta) * r
    xs = cx + np.cos(theta) * r
    yi = np.clip(np.round(ys).astype(np.int64), 0, h - 1)
    xi = np.clip(np.round(xs).astype(np.int64), 0, w - 1)
    mask[yi, xi] = True
    return torch.from_numpy(mask)


def build_ring_shell_masks(h: int, w: int, profile: WatermarkProfile) -> list[torch.Tensor]:
    cyf, cxf = fft_center_xy(h, w, profile.center)
    hh, ww = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    hh = hh.float()
    ww = ww.float()
    dist = torch.sqrt((hh - cyf) ** 2 + (ww - cxf) ** 2)

    shells: list[torch.Tensor] = []
    cy, cx = float(cyf), float(cxf)

    for k in range(profile.ring_r_inner, profile.ring_r_outer):
        if profile.rouder_ring:
            m_rot = rounder_trajectory_shell(h, w, cy, cx, float(k + 0.5))
            m_rot |= rounder_trajectory_shell(h, w, cy, cx, float(k))
            disk_band = ((dist >= float(k)) & (dist < float(k + 1))).bool()
            m = disk_band & m_rot
        else:
            m = ((dist >= float(k)) & (dist < float(k + 1))).bool()
        shells.append(m.cpu())

    return shells


def merge_shells(shells: list[torch.Tensor]) -> torch.Tensor:
    if not shells:
        raise ValueError("no ring shells computed — widen ring_r_outer or lower ring_r_inner")
    out = shells[0].clone()
    for s in shells[1:]:
        out |= s
    return out


def chessboard_fftshift(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ii = torch.arange(h, device=device, dtype=dtype).view(h, 1)
    jj = torch.arange(w, device=device, dtype=dtype).reshape(1, w)
    return (-1.0) ** (ii + jj)


def _sample_shell_complex_vectors(
    gen: torch.Generator,
    device: torch.device,
    shells: list[torch.Tensor],
    mode: EmbedMode,
    variance_n_sq: float,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Produce per-shell complex spectral tensors (+ combined reference vector)."""

    shells_d = [s.to(device) for s in shells]
    full = merge_shells([s.cpu() for s in shells_d]).bool().to(device)

    sig = torch.sqrt(torch.tensor(max(variance_n_sq, 1e-8), device=device) / 2.0)

    if mode == EmbedMode.RING_GAUSSIAN:
        zs: list[torch.Tensor] = []
        flats: list[torch.Tensor] = []
        for s in shells_d:
            re = torch.randn((), generator=gen, device=device, dtype=torch.float32) * sig.float()
            im = torch.randn((), generator=gen, device=device, dtype=torch.float32) * sig.float()
            z = torch.complex(re, im).expand(s.shape).clone()
            z = z.masked_fill(~s, 0 + 0j)
            zs.append(z)
            flats.append(torch.cat([z.real[s].flatten(), z.imag[s].flatten()], dim=0))

        stacked = torch.cat(flats, dim=0)

    elif mode == EmbedMode.MASK_COMPLEX_GAUSSIAN:
        re = torch.randn(full.shape, generator=gen, device=device, dtype=torch.float32) * sig.float()
        im = torch.randn(full.shape, generator=gen, device=device, dtype=torch.float32) * sig.float()
        zbig = torch.complex(re, im)
        zbig = zbig.masked_fill(~full, 0 + 0j)

        zs = []
        for s in shells_d:
            z = zbig * s.float()
            zs.append(z)
        stacked = torch.cat([zbig.real[full].flatten(), zbig.imag[full].flatten()], dim=0)

    else:
        raise ValueError(mode)

    return zs, stacked


def _discretize_shells_inplace(
    zs: list[torch.Tensor],
    gen: torch.Generator,
    alpha: float,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    """§5.3: ±α spectral values per radial ring shell (Hermitian fidelity approximated via real-valued complex bins)."""

    alpha_t = torch.tensor(alpha, dtype=dtype, device=device)
    out: list[torch.Tensor] = []
    for z in zs:
        s_mask = z != 0
        if not s_mask.any():
            out.append(z)
            continue
        coin = (torch.randn((), generator=gen, device=device, dtype=dtype) >= 0).to(dtype=dtype)
        val = torch.where(coin > 0, alpha_t, -alpha_t)
        nz = torch.where(s_mask, torch.complex(val.expand_as(z.real), torch.zeros_like(z.imag)), 0 + 0j)
        out.append(nz)
    return out


def profile_to_flat_dict(profile: WatermarkProfile) -> dict[str, Any]:
    d = {k: v for k, v in vars(profile).items() if k != "extra"}
    d["fft_norm"] = profile.fft_norm.value
    d["embed_mode"] = profile.embed_mode.value
    d["center"] = profile.center.value
    if profile.extra:
        d.update(profile.extra)
    return d


def inject_watermark(
    latent_hwc: torch.Tensor,
    profile: WatermarkProfile,
    generator: torch.Generator | None = None,
    seed: int = 0,
    key_index: int = 0,
) -> tuple[torch.Tensor, WatermarkKey, dict[str, torch.Tensor]]:
    """Imprint Tree-Ring / RingID watermark into spatial **real** initial noise latent H×W×C."""
    if latent_hwc.dim() != 3:
        raise ValueError(f"Expected H,W,C tensor, got shape {tuple(latent_hwc.shape)}")
    if profile.fft_norm == FourierNorm.FORWARD_SCALED:
        raise NotImplementedError("`FourierNorm.FORWARD_SCALED` requires custom scaling hooks — not wired yet.")

    h, w, c = latent_hwc.shape

    device = latent_hwc.device
    dtype_spatial = latent_hwc.dtype

    if generator is None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
    else:

        gen = generator

    shells = build_ring_shell_masks(h, w, profile)
    full_mask = merge_shells(shells).to(device)
    nk = fft_norm_kwarg(profile)

    out = latent_hwc.clone()

    wm_flat_segments: list[torch.Tensor] = []

    variance_n_sq = float((profile.latent_h) ** 2)  # appendix uses N²; use H for latent side

    for ch_idx in sorted(set(profile.ring_channels)):
        assert 0 <= ch_idx < c, f"ring channel out of bounds: {ch_idx}"

        x = latent_hwc[:, :, ch_idx].float()

        Z = torch.fft.fftshift(torch.fft.fft2(x.to(torch.complex64), **nk))
        zs, _ = _sample_shell_complex_vectors(gen, device, shells, profile.embed_mode, variance_n_sq=variance_n_sq)

        if profile.use_discretization:
            zs = _discretize_shells_inplace(zs, gen, float(profile.discrete_alpha), device, torch.float32)

        Z_wm_shells = zs[0].clone()
        for zi in zs[1:]:
            Z_wm_shells = Z_wm_shells + zi

        if profile.lossless_ring_real_only:
            pattern_real = Z_wm_shells.real.to(Z.real.dtype)
            Z_new = torch.where(full_mask, torch.complex(pattern_real, torch.zeros_like(Z.real)), Z)
        else:
            Z_new = torch.where(full_mask, Z_wm_shells, Z)

        if profile.spatial_shift:
            H_uv = chessboard_fftshift(h, w, device, dtype=torch.float32)
            eta = torch.tensor(profile.spatial_shift_eta, dtype=torch.float32, device=device)
            Z_new = eta * torch.complex(Z_new.real * H_uv, Z_new.imag * H_uv)

        ref_parts = []
        shells_d = [s.to(Z_new.device) for s in shells]
        for sd in shells_d:
            zp = torch.where(sd, Z_new, torch.zeros_like(Z_new))
            if profile.extraction == "real":
                ref_parts.append(zp.real[sd].flatten().cpu())
            elif profile.extraction == "magnitude":
                ref_parts.append(torch.abs(zp[sd]).flatten().cpu())
            else:
                ref_parts.append(torch.cat([zp.real[sd].flatten().cpu(), zp.imag[sd].flatten().cpu()]))

        x_wm = torch.fft.ifft2(torch.fft.ifftshift(Z_new), **nk)
        spatial = x_wm.real.to(dtype_spatial)

        out[:, :, ch_idx] = spatial
        wm_flat_segments.append(torch.cat(ref_parts).float())

    aux_std = latent_hwc[:, :, sorted(set(profile.ring_channels))[0]].float().std().clamp(min=1e-3)
    for ch in sorted(set(profile.secondary_gaussian_channels)):
        sigma = latent_hwc[:, :, ch].float().std().clamp(min=aux_std.item())
        g = torch.randn((h, w), generator=gen, device=device, dtype=dtype_spatial)
        out[:, :, ch] = g * sigma.to(dtype_spatial)

    key_vec = torch.cat(wm_flat_segments, dim=0).float()

    key = WatermarkKey(
        vector=key_vec.cpu(),
        profile_dict=profile_to_flat_dict(profile),
        seed=int(seed),
        key_index=key_index,
    )

    diag = {"mask_fftshift": full_mask.detach().cpu().bool()}
    return out.detach(), key, diag


def extract_pattern(latent_hwc: torch.Tensor, profile: WatermarkProfile) -> dict[str, torch.Tensor]:
    """Extract flattened spectral bins from recovered spatial noise (matching imprint mask order)."""

    h, w, c = latent_hwc.shape
    device = latent_hwc.device

    nk = fft_norm_kwarg(profile)
    shells = build_ring_shell_masks(h, w, profile)
    full_mask = merge_shells(shells).to(device)

    parts: list[torch.Tensor] = []
    feats: dict[str, torch.Tensor] = {}

    for ch_idx in sorted(set(profile.ring_channels)):
        assert 0 <= ch_idx < c

        x = latent_hwc[:, :, ch_idx].float()
        Z = torch.fft.fftshift(torch.fft.fft2(x.to(torch.complex64), **nk))

        for s in shells:
            sm = s.to(device)
            z = torch.where(sm, Z, torch.zeros_like(Z))
            if profile.extraction == "real":
                parts.append(z.real[sm].flatten())
            elif profile.extraction == "magnitude":
                parts.append(torch.abs(z[sm]).flatten())
            else:
                parts.append(torch.cat([z.real[sm].flatten(), z.imag[sm].flatten()], dim=0))

        feats[f"fft_shifted_mag_ch{ch_idx}"] = torch.abs(Z.cpu())

    w_hat_vec = torch.cat(parts, dim=0).detach().cpu().float()

    feats["vector"] = w_hat_vec
    feats["mask_fftshift"] = full_mask.detach().cpu()
    return feats
