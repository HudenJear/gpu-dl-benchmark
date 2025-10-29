# GPU Deep Learning Benchmark

This repository provides a simple, reproducible CUDA GPU benchmark that covers:
- Vision/classification (torchvision)
- Vision/generation (diffusers)
- NLP/language (transformers)

It measures forward-pass latency and throughput and writes a standardized Markdown report under `./benchmark_results`.

## Environment setup (two options)

You can either build a project-local Conda environment at `./envs/gpu-test` or use a prebuilt environment by downloading and extracting it.

### Option A: Build Conda env at ./envs/gpu-test

1) Create a clean Conda environment (Python only), then install project requirements via pip:

```
conda create -p ./envs/gpu-test python=3.12 -y
conda activate ./envs/gpu-test
pip install -r requirements.txt
```

2) (When you want to run the benchmark after installation) Activate the environment:

```
conda activate ./envs/gpu-test
```

3) (optional) Verify CUDA :

```
python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('torch=', torch.__version__)"
```

4) Run the benchmark:

```
python benchmark.py
```

If you prefer not to activate the env, you can run with:

```
conda run -p ./envs/gpu-test python benchmark.py
```

Remove the local env when done:

```
conda env remove -p ./envs/gpu-test
```

### Option B: Use a prebuilt environment (download & extract)

If your team provides a prebuilt Conda env archive (e.g., `gpu-test-conda-env.tar.gz`), you can avoid solving dependencies.

1) Download the archive to the project root (replace <URL> with your internal/release link):

```
wget -O gpu-test-conda-env.tar.gz <URL-to-prebuilt-conda-env>
```

2) Extract to `./envs/gpu-test`:

```
mkdir -p ./envs
tar -xzf gpu-test-conda-env.tar.gz -C ./envs
# After extraction, you should have: ./envs/gpu-test
```

3) Activate and run:

```
conda activate ./envs/gpu-test
python benchmark.py
```

Or run without activation:

```
conda run -p ./envs/gpu-test python benchmark.py
```

Note: This benchmark requires CUDA; CPU execution is not supported in the current version.

## Models included by default

Vision/classification (3x224x224 inputs):
- `resnet50`, `efficientnet_v2_s`, `convnext_tiny`, `swin_v2_t`, `vit_b_16`

Vision/generation (diffusers):
- `gen_unet2d_condition` (conditional UNet, latent space)
- `gen_unet2d` (smaller unconditional UNet)
- `gen_vae_autoencoderkl` (VAE)

NLP/language (transformers):
- `bert_small`, `distilbert_small`, `roberta_small`, `gpt2_small`

## Sample baseline results (by family)

Tables below are organized by task family. Rows are GPUs (currently RTX 3090 24G), and columns are test items (models). Cell values are "mean latency / throughput" with units.

### vision/classification (RTX 3090 24G)

| GPU           | resnet50       | efficientnet_v2_s | convnext_tiny  | swin_v2_t     | vit_b_16      |
|:--------------|:---------------|:------------------|:---------------|:--------------|:--------------|
| RTX 3090 24G  | 1511.41 img/s  | 1378.55 img/s     | 1077.61 img/s  | 695.98 img/s  | 430.82 img/s  |
| A800-SXM4-80GB| 3009.12 img/s  | 2276.26 img/s     | 1304.37 img/s  | 918.06 img/s  | 482.16 img/s  |
| RTX 5070Ti 16G| 1900.81 img/s  | 1969.90 img/s     | 1481.89 img/s  | 987.44 img/s  | 540.83 img/s  |
| RTX 5080 16G  | 2212.66 img/s  | 2352.21 img/s     | 1848.15 img/s  | 1195.53 img/s | 680.69 img/s  |


### vision/generation (RTX 3090 24G)

| GPU           | gen_unet2d_condition | gen_unet2d     | gen_vae_autoencoderkl |
|:--------------|:---------------------|:---------------|:----------------------|
| RTX 3090 24G  | 1571.33 img/s        | 156.64 img/s   | 181.88 img/s          |
| A800-SXM4-80GB| 2150.32 img/s        | 324.92 img/s   | 402.84 img/s          |
| RTX 5070Ti 16G| 2185.08 img/s        | 190.43 img/s   | 189.92 img/s          |
| RTX 5080 16G  | 2470.11 img/s        | 223.57 img/s   | 215.51 img/s          |


### nlp/language (RTX 3090 24G)

| GPU           | bert_small     | distilbert_small | roberta_small  | gpt2_small    |
|:--------------|:---------------|:-----------------|:---------------|:--------------|
| RTX 3090 24G  | 369.22 k tok/s | 373.86 k tok/s   | 368.85 k tok/s | 186.65 k tok/s |
| A800-SXM4-80GB| 445.85 k tok/s | 448.62 k tok/s   | 440.94 k tok/s | 216.74 k tok/s |
| RTX 5070Ti 16G| 540.91 k tok/s | 541.49 k tok/s   | 534.90 k tok/s | 244.90 k tok/s |
| RTX 5080 16G  | 717.80 k tok/s | 718.50 k tok/s   | 707.00 k tok/s | 312.77 k tok/s |


Tip: Append new GPU rows (e.g., RTX 4080, A100 40G, MI250) or extend columns with new models. You can also add columns like `Batch`, `SeqLen`, or `Precision` if you track multiple settings.

## Usage (CUDA-only)

Basic GPU run:

```
python benchmark.py --batch-size 32 --image-size 224 --precision fp32 --repeats 30 --warmup 10
```

Notes:
- `--image-size` is used as sequence length for text models.
- For lower memory usage, try `--batch-size 1` and `--precision fp16`.

Autocast precision on CUDA:

```
python benchmark.py --precision fp16
python benchmark.py --precision bf16
```

The report will be saved under `./benchmark_results/` with a timestamped filename.
python benchmark.py --jit
```

## Output

The script writes a Markdown report under `benchmark_results/` with a UTC timestamp appended to the basename for uniqueness. For example:

```
benchmark_results/benchmark_results_20251021_064643UTC.md
benchmark_results/my_report_20251021_064700UTC.md  # when using --output my_report.md
```

The report contains:
- Environment summary (device, CUDA, cuDNN, PyTorch versions)
- Textual description
- Results table with latency (mean/P50/P90/P99) and throughput

Example to customize output path and include notes:

```
python benchmark.py --output my_report.md --notes "Tested after driver update"
```

## Caveats

- Inputs are random tensors; results measure raw model forward-pass only (no data loading or preprocessing).
- cudnn.benchmark is enabled for kernel auto-tuning.
- TorchScript tracing may fail for some models; the script will automatically fall back to eager mode.
- Some models require a sufficiently recent torchvision version (e.g., Swin V2, ConvNeXt, EfficientNetV2). If you encounter import or constructor errors, upgrade torchvision.

## Troubleshooting

- If `torch.cuda.is_available()` is False, ensure you installed the CUDA-enabled PyTorch build and that your NVIDIA driver is compatible with the selected CUDA runtime (the `environment.yml` uses CUDA 12.1). See: https://pytorch.org/get-started/locally/
- On systems without NVIDIA GPUs, this CUDA-only benchmark cannot run. Install on a CUDA-capable machine.
- If `conda env create` fails due to channel priority, try adding `-c nvidia -c pytorch -c conda-forge -c defaults` explicitly to your `conda` command, or run `conda config --add channels nvidia` etc.
