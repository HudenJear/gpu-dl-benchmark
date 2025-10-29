from typing import Dict, List
import os
import csv


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

    # Ranking section (tables by family like README format). We combine this run with
    # historical baselines from utils/baselines.csv for cross-GPU comparison.
    # Families: vision/classification (img/s), vision/generation (img/s), nlp/language (k tok/s)

    def _build_maps_this_run():
        cls_map = {}
        gen_map = {}
        nlp_map = {}
        for r in results:
            name = r.get('model', '')
            thr_img = r.get('throughput_img_s')
            thr_tok = r.get('throughput_tok_s')
            if thr_tok is not None:
                nlp_map[name] = float(thr_tok)
            elif thr_img is not None:
                if name.startswith('gen_'):
                    gen_map[name] = float(thr_img)
                else:
                    cls_map[name] = float(thr_img)
        return cls_map, gen_map, nlp_map

    def _load_baselines_csv() -> Dict[str, Dict[str, Dict[str, Dict[str, str]]]]:
        """
        Returns: {family: {gpu: {model: {"throughput": float, "unit": str}}}}
        """
        baseline_path = os.path.join(os.path.dirname(__file__), 'baselines.csv')
        data: Dict[str, Dict[str, Dict[str, Dict[str, str]]]] = {}
        if not os.path.isfile(baseline_path):
            return data
        try:
            with open(baseline_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fam = row.get('family', '').strip()
                    gpu = row.get('gpu', '').strip()
                    model = row.get('model', '').strip()
                    unit = row.get('unit', '').strip()
                    try:
                        thr = float(row.get('throughput', ''))
                    except Exception:
                        continue
                    data.setdefault(fam, {}).setdefault(gpu, {})[model] = {"throughput": thr, "unit": unit}
        except Exception:
            # Fail gracefully if CSV cannot be read
            return {}
        return data

    def _sorted_values(d: dict):
        return sorted(d.items(), key=lambda x: x[1], reverse=True)

    def _render_family_table(title: str, cols: list, this_run: dict, baselines: Dict[str, Dict[str, Dict[str, str]]], unit: str, scale: float = 1.0, unit_suffix: str = ''):
        lines.append(f"### {title}")
        header = "| GPU           | " + " | ".join(cols) + " |"
        lines.append(header)
        lines.append("|:--------------|" + " :-------------- |" * len(cols))
        # Current run first
        cells = []
        for c in cols:
            v = this_run.get(c)
            if v is None:
                cells.append("")
            else:
                disp = v / scale
                cells.append(f"{disp:.2f} {unit_suffix}{unit}")
        lines.append("| This Run      | " + " | ".join(cells) + " |")
        # Baseline rows
        for gpu_name, models_map in baselines.items():
            cells = []
            for c in cols:
                entry = models_map.get(c)
                if not entry:
                    cells.append("")
                    continue
                thr = float(entry.get("throughput", 0.0))
                # Units in CSV are consistent per family; for NLP we still scale
                disp = thr / scale
                cells.append(f"{disp:.2f} {unit_suffix}{unit}")
            lines.append(f"| {gpu_name:<14} | " + " | ".join(cells) + " |")
        lines.append("\n---\n")

    lines.append("## Ranking")
    cls_map, gen_map, nlp_map = _build_maps_this_run()
    baselines = _load_baselines_csv()

    # Classification models (common torchvision names we use)
    cls_cols = [
        'resnet50', 'efficientnet_v2_s', 'convnext_tiny', 'swin_v2_t', 'vit_b_16'
    ]
    _render_family_table(
        "vision/classification",
        cls_cols,
        cls_map,
        baselines.get('vision/classification', {}),
        unit="img/s",
    )

    # Generation models (our diffusers names)
    gen_cols = [
        'gen_unet2d_condition', 'gen_unet2d', 'gen_vae_autoencoderkl'
    ]
    _render_family_table(
        "vision/generation",
        gen_cols,
        gen_map,
        baselines.get('vision/generation', {}),
        unit="img/s",
    )

    # NLP models (convert to k tok/s)
    nlp_cols = [
        'bert_small', 'distilbert_small', 'roberta_small', 'gpt2_small'
    ]
    _render_family_table(
        "nlp/language",
        nlp_cols,
        nlp_map,
        baselines.get('nlp/language', {}),
        unit="tok/s",
        scale=1000.0,
        unit_suffix="k ",
    )

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
