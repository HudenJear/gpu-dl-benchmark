#!/usr/bin/env python3
import argparse
import datetime as dt
import os
import time
import platform
import sys
import time
from typing import Dict, List, Tuple

import torch
import torchvision
from torch import nn
from torchvision import models
from utils.report import write_markdown
from utils.bench import benchmark_model, benchmark_gen_model, benchmark_text_model
from diffusers import (
            UNet2DConditionModel,
            UNet2DModel,
            AutoencoderKL,
        )

# Optional: lightweight HF Transformers models for text benchmarking
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
    """Return CUDA device; raise if CUDA is unavailable."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark. Please install a CUDA-enabled PyTorch and ensure an NVIDIA GPU is available.")
    return torch.device("cuda")


def model_zoo() -> Dict[str, nn.Module]:
    """
    Select 5 representative and up-to-date torchvision models spanning CNNs and ViTs.
    We instantiate without pretrained weights to avoid downloads; we benchmark architecture/compute.
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
    Representative, more recent image generation backbones via diffusers (if available):
    - UNet2DConditionModel: Core of Stable Diffusion-style models (conditional UNet)
    - UNet2DModel: Unconditional UNet
    - AutoencoderKL: VAE used in latent diffusion (decoder/encoder)
    

    We instantiate small-ish configurations to keep forward pass reasonable. If diffusers
    is not available, this function returns an empty dict and the caller can skip.
    """
    zoo: Dict[str, nn.Module] = {}
    try:
        
        # Define lightweight configs
        unet_cond = UNet2DConditionModel(
            sample_size=image_size // 8,  # latent spatial size if simulating SD latents (assume downscale x8)
            in_channels=4,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(128, 256, 256, 512),
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            cross_attention_dim=768,
            attention_head_dim=8,
        ).to(device).eval()

        # Smaller UNet2D (unconditional) to fit <8GB: fewer blocks and channels, no attention blocks
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
            in_channels=3, out_channels=3, latent_channels=4,
            block_out_channels=(64, 128, 128, 256),
            down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
            up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        ).to(device).eval()

        zoo = {
            "gen_unet2d_condition": unet_cond,
            "gen_unet2d": unet_uncond,
            "gen_vae_autoencoderkl": vae,
        }
    except Exception as e:
        # diffusers not available or failed to instantiate; skip generative models gracefully
        print(f"[info] Skipping generative models (diffusers not available or failed to init: {e})")
    return zoo


def text_model_zoo(device: torch.device) -> Dict[str, nn.Module]:
    """
    Representative small text (language) models using Hugging Face Transformers.
    We build from configs to avoid any downloads. If transformers is not available,
    return an empty dict.
    """
    if not _HAS_TRANSFORMERS:
        return {}

    zoo: Dict[str, nn.Module] = {}

    # Moderate BERT-like encoder (fits <8GB with default settings)
    bert_cfg = BertConfig(
        vocab_size=30522,
        hidden_size=384,
        num_hidden_layers=8,
        num_attention_heads=6,
        intermediate_size=1536,
    )
    zoo["bert_small"] = BertModel(bert_cfg).to(device).eval()

    # DistilBERT-like
    distil_cfg = DistilBertConfig(
        vocab_size=30522,
        dim=384,
        hidden_dim=1536,
        n_layers=8,
        n_heads=6,
    )
    zoo["distilbert_small"] = DistilBertModel(distil_cfg).to(device).eval()

    # RoBERTa-like encoder
    roberta_cfg = RobertaConfig(
        vocab_size=50265,
        hidden_size=384,
        num_hidden_layers=8,
        num_attention_heads=6,
        intermediate_size=1536,
    )
    zoo["roberta_small"] = RobertaModel(roberta_cfg).to(device).eval()

    # Small GPT-2-like decoder LM
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
    for unit in ['B','KB','MB','GB','TB']:
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
        info.update({
            "device_name": torch.cuda.get_device_name(0),
            "cuda_runtime": torch.version.cuda or "unknown",
            "driver": torch.cuda.driver_version if hasattr(torch.cuda, 'driver_version') else "unknown",
            "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "multi_gpu_count": str(torch.cuda.device_count()),
            "total_memory": format_bytes(torch.cuda.get_device_properties(0).total_memory),
            "cudnn": torch.backends.cudnn.version() and str(torch.backends.cudnn.version()) or "unknown",
        })
    else:
        info.update({
            "device_name": "CPU",
            "cuda_runtime": "N/A",
            "driver": "N/A",
            "capability": "N/A",
            "multi_gpu_count": "0",
            "total_memory": "N/A",
            "cudnn": "N/A",
        })
    return info


def percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s)-1) * q
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[int(k)]
    d0 = s[f] * (c - k)
    d1 = s[c] * (k - f)
    return d0 + d1


 


def main():
    parser = argparse.ArgumentParser(description="CUDA-only GPU Benchmark with torchvision models")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size (HxW)")
    parser.add_argument("--repeats", type=int, default=30, help="Number of timed iterations per model")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations per model")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32", help="Computation precision (autocast on CUDA)")
    parser.add_argument("--channels-last", action="store_true", help="Use NHWC (channels_last) memory format on CUDA")
    parser.add_argument("--output", default="benchmark_results.md", help="Output Markdown file path")
    parser.add_argument("--notes", default="", help="Additional notes to include in the report")

    args = parser.parse_args()

    device = get_device()

    # Prepare unique, timestamped output path under ./benchmark_results
    out_dir = os.path.join(os.getcwd(), "benchmark_results")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.output))[0]
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%SUTC")
    out_path = os.path.join(out_dir, f"{base}_{ts}.md")

    zoo = model_zoo()
    gen_zoo = gen_model_zoo(device, args.image_size)
    text_zoo = text_model_zoo(device)

    # Register model groups with their corresponding benchmark functions.
    # This makes it easy to plug in future families (e.g., text models) with their own runner.
    model_groups: List[Tuple[Dict[str, nn.Module], str, callable]] = [
        (zoo, "vision/classification", benchmark_model),
        (gen_zoo, "vision/generation", benchmark_gen_model),
        (text_zoo, "nlp/language", benchmark_text_model),
    ]

    # Build the final list of (family, name, model, bench_fn)
    to_run: List[Tuple[str, str, nn.Module, callable]] = []
    for group_models, family, bench_fn in model_groups:
        if not group_models:
            continue
        for name, model in group_models.items():
            to_run.append((family, name, model, bench_fn))

    if not to_run:
        available = []
        for group_models, _, _ in model_groups:
            available.extend(list(group_models.keys()))
        print("No valid models selected. Available:", ", ".join(available))
        sys.exit(1)

    env = get_env_info(device)

    all_results: List[Dict[str, float]] = []
    current_family = None
    for family, name, model, bench_fn in to_run:
        if family != current_family:
            # Family header to make console output clearer
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
            # Support different throughput metrics depending on family
            throughput = res.get('throughput_img_s', res.get('throughput_tok_s', float('nan')))
            unit = 'img/s' if 'throughput_img_s' in res else ('tok/s' if 'throughput_tok_s' in res else 'units/s')
            print(f"{name}: mean {res['latency_ms_mean']:.3f} ms, throughput {throughput:.2f} {unit}")
            # Give the GPU some time to release memory between tests
            if device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            time.sleep(1.0)
        except RuntimeError as e:
            print(f"Error benchmarking {name}: {e}")

    write_markdown(out_path, env, all_results, args.notes)
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
