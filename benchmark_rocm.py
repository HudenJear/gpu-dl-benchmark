#!/usr/bin/env python3
import argparse
import datetime as dt
import os
import platform
import sys
from typing import Dict, List, Tuple

import torch
import torchvision
from torch import nn
from torchvision import models

from utils.report import write_markdown
from utils.bench import (
    benchmark_model,
    benchmark_gen_model,
    benchmark_text_model,
)

# Diffusers models (optional)
try:
    from diffusers import (
        UNet2DConditionModel,
        UNet2DModel,
        AutoencoderKL,
    )
    _HAS_DIFFUSERS = True
except Exception:
    _HAS_DIFFUSERS = False

# Transformers models (optional)
try:
    from transformers import (
        BertConfig,
        BertModel,
        DistilBertConfig,
        DistilBertModel,
        GPT2Config,
        GPT2LMHeadModel,
        RobertaConfig,
        RobertaModel,
    )
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False


def get_device() -> torch.device:
    """Return ROCm-capable device; raise if not available.
    Note: PyTorch ROCm builds expose the CUDA-like API; torch.cuda.is_available() will be True on ROCm.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "ROCm is required for this benchmark. Please install a ROCm-enabled PyTorch and ensure an AMD GPU is available."
        )
    return torch.device("cuda")


def model_zoo() -> Dict[str, nn.Module]:
    """
    Vision/classification models from torchvision (no pretrained weights).
    All accept 3x224x224 inputs.
    """
    return {
        "resnet50": models.resnet50(weights=None),
        "efficientnet_v2_s": models.efficientnet_v2_s(weights=None),
        "convnext_tiny": models.convnext_tiny(weights=None),
        "swin_v2_t": models.swin_v2_t(weights=None),
        "vit_b_16": models.vit_b_16(weights=None),
    }


def gen_model_zoo(device: torch.device, image_size: int) -> Dict[str, nn.Module]:
    """
    Generative backbones via diffusers (if available):
    - UNet2DConditionModel (latent space, conditional)
    - UNet2DModel (unconditional, made smaller to fit <8GB)
    - AutoencoderKL (VAE)
    """
    if not _HAS_DIFFUSERS:
        return {}

    zoo: Dict[str, nn.Module] = {}

    unet_cond = UNet2DConditionModel(
        sample_size=image_size // 8,
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(128, 256, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        cross_attention_dim=768,
        attention_head_dim=8,
    ).to(device).eval()

    unet_uncond = UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        block_out_channels=(32, 64, 128),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
    ).to(device).eval()

    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        block_out_channels=(64, 128, 128, 256),
        down_block_types=(
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ),
        up_block_types=(
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ),
    ).to(device).eval()

    zoo = {
        "gen_unet2d_condition": unet_cond,
        "gen_unet2d": unet_uncond,
        "gen_vae_autoencoderkl": vae,
    }
    return zoo


def text_model_zoo(device: torch.device) -> Dict[str, nn.Module]:
    """Small/moderate text (language) models using Transformers configs (no downloads)."""
    if not _HAS_TRANSFORMERS:
        return {}

    zoo: Dict[str, nn.Module] = {}

    bert_cfg = BertConfig(
        vocab_size=30522,
        hidden_size=384,
        num_hidden_layers=8,
        num_attention_heads=6,
        intermediate_size=1536,
    )
    zoo["bert_small"] = BertModel(bert_cfg).to(device).eval()

    distil_cfg = DistilBertConfig(
        vocab_size=30522,
        dim=384,
        hidden_dim=1536,
        n_layers=8,
        n_heads=6,
    )
    zoo["distilbert_small"] = DistilBertModel(distil_cfg).to(device).eval()

    roberta_cfg = RobertaConfig(
        vocab_size=50265,
        hidden_size=384,
        num_hidden_layers=8,
        num_attention_heads=6,
        intermediate_size=1536,
    )
    zoo["roberta_small"] = RobertaModel(roberta_cfg).to(device).eval()

    gpt2_cfg = GPT2Config(
        vocab_size=50257,
        n_layer=8,
        n_head=6,
        n_embd=384,
        n_positions=512,
        n_ctx=512,
    )
    zoo["gpt2_small"] = GPT2LMHeadModel(gpt2_cfg).to(device).eval()

    return zoo


def format_bytes(num_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f}PB"


def get_env_info(device: torch.device) -> Dict[str, str]:
    info = {
        "python": sys.version.split(" (", 1)[0],
        "pytorch": torch.__version__,
        "torchvision": torchvision.__version__,
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "datetime_utc": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
    }
    if device.type == "cuda":
        hip_version = getattr(torch.version, "hip", None)
        info.update(
            {
                "device_name": torch.cuda.get_device_name(0),
                # Reuse the same keys used in report for compatibility
                "cuda_runtime": hip_version or "unknown",  # on ROCm builds, this is HIP version
                "driver": "ROCm",  # placeholder; driver version not exposed like NVIDIA
                "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
                "multi_gpu_count": str(torch.cuda.device_count()),
                "total_memory": format_bytes(torch.cuda.get_device_properties(0).total_memory),
                "cudnn": "MIOpen",  # analogous library on ROCm
            }
        )
    else:
        info.update(
            {
                "device_name": "CPU",
                "cuda_runtime": "N/A",
                "driver": "N/A",
                "capability": "N/A",
                "multi_gpu_count": "0",
                "total_memory": "N/A",
                "cudnn": "N/A",
            }
        )
    return info



def main():
    parser = argparse.ArgumentParser(description="ROCm GPU Benchmark for vision and language models")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size (HxW) / sequence length for text")
    parser.add_argument("--repeats", type=int, default=30, help="Number of timed iterations per model")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations per model")
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="Computation precision (autocast on GPU)",
    )
    parser.add_argument("--channels-last", action="store_true", help="Use NHWC (channels_last) memory format on GPU")
    parser.add_argument("--output", default="benchmark_results.md", help="Output Markdown file path")
    parser.add_argument("--notes", default="", help="Additional notes to include in the report")

    args = parser.parse_args()

    device = get_device()

    out_dir = os.path.join(os.getcwd(), "benchmark_results")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.output))[0]
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%SUTC")
    out_path = os.path.join(out_dir, f"{base}_{ts}.md")

    zoo = model_zoo()
    gen_zoo = gen_model_zoo(device, args.image_size) if _HAS_DIFFUSERS else {}
    text_zoo = text_model_zoo(device) if _HAS_TRANSFORMERS else {}

    model_groups: List[Tuple[Dict[str, nn.Module], str, callable]] = [
        (zoo, "vision/classification", benchmark_model),
        (gen_zoo, "vision/generation", benchmark_gen_model),
        (text_zoo, "nlp/language", benchmark_text_model),
    ]

    to_run: List[Tuple[str, str, nn.Module, callable]] = []
    for group_models, family, bench_fn in model_groups:
        if not group_models:
            continue
        for name, model in group_models.items():
            to_run.append((family, name, model, bench_fn))

    if not to_run:
        available: List[str] = []
        for group_models, _, _ in model_groups:
            available.extend(list(group_models.keys()))
        print("No valid models selected. Available:", ", ".join(available))
        sys.exit(1)

    env = get_env_info(device)

    all_results: List[Dict[str, float]] = []
    current_family = None
    for family, name, model, bench_fn in to_run:
        if family != current_family:
            print(f"\n[info] Benchmarking {family} models...")
            current_family = family
        try:
            res = bench_fn(
                name=name,
                model=model,
                device=device,
                batch_size=args.batch_size,
                image_size=args.image_size,
                repeats=args.repeats,
                warmup=args.warmup,
                precision=args.precision,
                channels_last=args.channels_last,
            )
            all_results.append(res)
            throughput = res.get("throughput_img_s", res.get("throughput_tok_s", float("nan")))
            unit = "img/s" if "throughput_img_s" in res else ("tok/s" if "throughput_tok_s" in res else "units/s")
            print(f"{name}: mean {res['latency_ms_mean']:.3f} ms, throughput {throughput:.2f} {unit}")
        except RuntimeError as e:
            print(f"Error benchmarking {name}: {e}")

    write_markdown(out_path, env, all_results, args.notes)
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
