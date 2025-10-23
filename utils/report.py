from typing import Dict, List


def write_markdown(
    path: str,
    env: Dict[str, str],
    results: List[Dict[str, float]],
    notes: str,
):
    lines: List[str] = []
    lines.append(f"# CUDA GPU Deep Learning Benchmark\n")
    lines.append("")
    lines.append(
        "This report benchmarks representative models (vision/classification, vision/generation, and NLP/language when available) "
        "using random input tensors. Each model was run multiple times; results below report average latency and throughput.\n"
    )

    lines.append("## Environment\n")
    lines.append("```")
    for k in [
        "datetime_utc",
        "python",
        "pytorch",
        "torchvision",
        "platform",
        "device_name",
        "cuda_runtime",
        "driver",
        "capability",
        "multi_gpu_count",
        "total_memory",
        "cudnn",
    ]:
        v = env.get(k, "")
        lines.append(f"{k}: {v}")
    lines.append("```\n")

    lines.append("## Results\n")
    # Table header (Size/Seq works for both vision and text; throughput shows unit per-row)
    lines.append(
        "| Model | Batch | Size/Seq | Precision | Mean Latency (ms) | P50 (ms) | P90 (ms) | P99 (ms) | Throughput | Repeats | Warmup |"
    )
    lines.append(
        "|:------|------:|------:|:---------:|------------------:|---------:|---------:|---------:|-------------------:|--------:|-------:|"
    )
    for r in results:
        size_or_seq = r.get('image_size', r.get('seq_len', ''))
        throughput = r.get('throughput_img_s', r.get('throughput_tok_s', float('nan')))
        unit = 'img/s' if 'throughput_img_s' in r else ('tok/s' if 'throughput_tok_s' in r else '')
        lines.append(
            f"| {r.get('model','')} | {int(r.get('batch_size', 0))} | {size_or_seq} | {r.get('precision','')} | "
            f"{float(r.get('latency_ms_mean', float('nan'))):.3f} | {float(r.get('latency_ms_p50', float('nan'))):.3f} | {float(r.get('latency_ms_p90', float('nan'))):.3f} | {float(r.get('latency_ms_p99', float('nan'))):.3f} | "
            f"{float(throughput):.2f} {unit} | {int(r.get('repeats', 0))} | {int(r.get('warmup', 0))} |"
        )
    lines.append("")

    lines.append("## Notes\n")
    if notes:
        lines.append(notes)
    else:
        lines.append(
            "- Inputs are random tensors; results measure raw forward-pass performance, not data loading or preprocessing."
        )
        lines.append("- TorchDynamo/compile is not enabled by default; add --jit to try TorchScript tracing optimization.")
        lines.append("- cudnn.benchmark is enabled for kernel auto-tuning.")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
