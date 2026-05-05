"""Torch / CUDA environment logging for generation (RTX 50-series + cu128 compatibility)."""

from __future__ import annotations

import torch


def log_generation_torch_cuda(pipe, *, n_images: int) -> None:
    """Print a single diagnostic line when starting image generation (watermarked or not)."""

    dev = pipe.device
    dev_s = str(dev)
    tv = torch.__version__
    cuda_built = getattr(torch.version, "cuda", None) or ""

    if dev.type == "cuda" and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        major, minor = torch.cuda.get_device_capability(idx)
        sm = f"sm_{major}{minor}"
        mem_gib = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        print(
            f"[ringid] gen cuda={cuda_built} torch={tv} device={dev_s!r} gpu={name!r} {sm} "
            f"vram_gib={mem_gib:.1f} batch={n_images}",
            flush=True,
        )
        # RTX 50 / Blackwell is sm_120 (12,0) — cu128 wheels are the usual fix when stock PyTorch is too old.
        if major >= 12 and cuda_built and not cuda_built.startswith("12.8"):
            print(
                "[ringid] hint: for RTX 50-series use PyTorch CUDA 12.8 wheels: "
                "pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu128",
                flush=True,
            )
    else:
        print(
            f"[ringid] gen torch={tv} device={dev_s!r} cuda_available={torch.cuda.is_available()} "
            f"torch_cuda_built={cuda_built!r} batch={n_images} (CPU is very slow)",
            flush=True,
        )
