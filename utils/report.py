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
        "This report benchmarks 5 representative torchvision models using random input tensors. "
        "Each model was run multiple times; results below report average latency and throughput.\n"
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
    # Table header
    lines.append(
        "| Model | Batch | Size | Precision | Mean Latency (ms) | P50 (ms) | P90 (ms) | P99 (ms) | Throughput (img/s) | Repeats | Warmup |"
    )
    lines.append(
        "|:------|------:|-----:|:---------:|------------------:|---------:|---------:|---------:|-------------------:|--------:|-------:|"
    )
    for r in results:
        lines.append(
            f"| {r['model']} | {r['batch_size']} | {r['image_size']} | {r['precision']} | "
            f"{r['latency_ms_mean']:.3f} | {r['latency_ms_p50']:.3f} | {r['latency_ms_p90']:.3f} | {r['latency_ms_p99']:.3f} | "
            f"{r['throughput_img_s']:.2f} | {int(r['repeats'])} | {int(r['warmup'])} |"
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
