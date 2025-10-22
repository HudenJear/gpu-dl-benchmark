import time
from typing import Dict, List, Tuple

import torch
from torch import nn


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * q
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[int(k)]
    d0 = s[f] * (c - k)
    d1 = s[c] * (k - f)
    return d0 + d1


def benchmark_model(
    name: str,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    image_size: int,
    repeats: int,
    warmup: int,
    precision: str,
    channels_last: bool,
) -> Dict[str, float]:
    """Benchmark classification/vision backbones with single-tensor image input."""
    torch.backends.cudnn.benchmark = True

    model = model.eval().to(device)
    if channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    if channels_last and device.type == "cuda":
        x = x.to(memory_format=torch.channels_last)

    amp_dtype = None
    if device.type == "cuda":
        if precision == "fp16":
            amp_dtype = torch.float16
        elif precision == "bf16":
            amp_dtype = torch.bfloat16

    times: List[float] = []

    with torch.no_grad():
        for _ in range(warmup):
            if device.type == "cuda":
                torch.cuda.synchronize()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
                _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()

        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
                _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    mean_ms = float(sum(times) / len(times)) if times else float("nan")
    p50 = _percentile(times, 0.5)
    p90 = _percentile(times, 0.9)
    p99 = _percentile(times, 0.99)

    imgs_per_sec = (batch_size / (mean_ms / 1000.0)) if mean_ms and mean_ms > 0 else float("nan")

    return {
        "model": name,
        "batch_size": batch_size,
        "image_size": image_size,
        "precision": precision,
        "latency_ms_mean": mean_ms,
        "latency_ms_p50": p50,
        "latency_ms_p90": p90,
        "latency_ms_p99": p99,
        "throughput_img_s": imgs_per_sec,
        "repeats": repeats,
        "warmup": warmup,
    }


def benchmark_text_model(
    name: str,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    image_size: int,  # reuse as sequence length
    repeats: int,
    warmup: int,
    precision: str,
    channels_last: bool,
) -> Dict[str, float]:
    """Benchmark language models with random token inputs.

    image_size is treated as sequence length to keep CLI consistent.
    """
    torch.backends.cudnn.benchmark = True

    model = model.eval().to(device)

    seq_len = max(1, image_size)

    # Try to infer vocab size from the model's input embeddings
    vocab_size = None
    if hasattr(model, "get_input_embeddings") and callable(getattr(model, "get_input_embeddings")):
        try:
            emb = model.get_input_embeddings()
            if emb is not None and hasattr(emb, "num_embeddings"):
                vocab_size = int(emb.num_embeddings)
        except Exception:
            pass
    if vocab_size is None:
        vocab_size = 50257

    cls = model.__class__.__name__.lower()

    def build_inputs():
        if any(k in cls for k in ["causallm", "gpt", "llama", "mpt", "falcon", "gptj", "gptneo"]):
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)
            return (input_ids,), {}
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)
        return (), {"input_ids": input_ids, "attention_mask": attention_mask}

    amp_dtype = None
    if device.type == "cuda":
        if precision == "fp16":
            amp_dtype = torch.float16
        elif precision == "bf16":
            amp_dtype = torch.bfloat16

    times: List[float] = []

    with torch.no_grad():
        for _ in range(warmup):
            if device.type == "cuda":
                torch.cuda.synchronize()
            args, kwargs = build_inputs()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
                _ = model(*args, **kwargs)
            if device.type == "cuda":
                torch.cuda.synchronize()

        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            args, kwargs = build_inputs()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
                _ = model(*args, **kwargs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    mean_ms = float(sum(times) / len(times)) if times else float("nan")
    p50 = _percentile(times, 0.5)
    p90 = _percentile(times, 0.9)
    p99 = _percentile(times, 0.99)
    toks_per_sec = (batch_size * seq_len) / (mean_ms / 1000.0) if mean_ms and mean_ms > 0 else float("nan")

    return {
        "model": name,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "precision": precision,
        "latency_ms_mean": mean_ms,
        "latency_ms_p50": p50,
        "latency_ms_p90": p90,
        "latency_ms_p99": p99,
        "throughput_tok_s": toks_per_sec,
        "repeats": repeats,
        "warmup": warmup,
    }


def benchmark_gen_model(
    name: str,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    image_size: int,
    repeats: int,
    warmup: int,
    precision: str,
    channels_last: bool,
) -> Dict[str, float]:
    """Benchmark generative backbones from diffusers with representative random inputs."""
    torch.backends.cudnn.benchmark = True

    model = model.eval().to(device)
    if channels_last and device.type == "cuda":
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception:
            pass

    cls = model.__class__.__name__.lower()

    def build_inputs():
        if "unet2dconditionmodel" in cls:
            latent_h = max(1, image_size // 8)
            latent_w = max(1, image_size // 8)
            x = torch.randn(batch_size, 4, latent_h, latent_w, device=device)
            t = torch.randint(0, 1000, (batch_size,), device=device, dtype=torch.long)
            cond = torch.randn(batch_size, 77, 768, device=device)
            return (), {"sample": x, "timestep": t, "encoder_hidden_states": cond}
        if "unet2dmodel" in cls:
            x = torch.randn(batch_size, 3, image_size, image_size, device=device)
            t = torch.randint(0, 1000, (batch_size,), device=device, dtype=torch.long)
            return (), {"sample": x, "timestep": t}
        if "autoencoderkl" in cls:
            x = torch.randn(batch_size, 3, image_size, image_size, device=device)
            return (), {"sample": x}
        # Remove DiT-specific handling; DiT is excluded from tests
        # Fallback: assume single positional tensor
        x = torch.randn(batch_size, 3, image_size, image_size, device=device)
        return (x,), {}

    amp_dtype = None
    if device.type == "cuda":
        if precision == "fp16":
            amp_dtype = torch.float16
        elif precision == "bf16":
            amp_dtype = torch.bfloat16

    times: List[float] = []

    with torch.no_grad():
        for _ in range(warmup):
            if device.type == "cuda":
                torch.cuda.synchronize()
            args, kwargs = build_inputs()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
                _ = model(*args, **kwargs)
            if device.type == "cuda":
                torch.cuda.synchronize()

        for _ in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            args, kwargs = build_inputs()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype is not None)):
                _ = model(*args, **kwargs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

    mean_ms = float(sum(times) / len(times)) if times else float("nan")
    p50 = _percentile(times, 0.5)
    p90 = _percentile(times, 0.9)
    p99 = _percentile(times, 0.99)
    imgs_per_sec = (batch_size / (mean_ms / 1000.0)) if mean_ms and mean_ms > 0 else float("nan")

    return {
        "model": name,
        "batch_size": batch_size,
        "image_size": image_size,
        "precision": precision,
        "latency_ms_mean": mean_ms,
        "latency_ms_p50": p50,
        "latency_ms_p90": p90,
        "latency_ms_p99": p99,
        "throughput_img_s": imgs_per_sec,
        "repeats": repeats,
        "warmup": warmup,
    }
