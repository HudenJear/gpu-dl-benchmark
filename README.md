# GPU Deep Learning Benchmark

This repository provides a simple, reproducible CUDA GPU benchmark using 5 representative, up-to-date torchvision models and random input tensors. It measures forward-pass latency and throughput, averages results across multiple runs, and writes a standardized Markdown report.

## Models

By default, the benchmark runs the following 5 models (all accept 3x224x224 inputs):
- `resnet50`
- `efficientnet_v2_s`
- `convnext_tiny`
- `swin_v2_t`
- `vit_b_16`

You can choose a subset via `--models`.

## Requirements

- Python 3.9+
- PyTorch and torchvision with CUDA support (CUDA-only)

Install from `requirements.txt`:

```
pip install -r requirements.txt
```

Note: If you need a specific CUDA build for PyTorch/torchvision, please follow the official installation instructions: https://pytorch.org/get-started/locally/

## Quick start (Conda, CUDA-only)

The project includes a ready-to-use Conda environment file `environment.yml` so you can set up and run in an isolated env quickly.

1) Create the environment (includes PyTorch, torchvision, and CUDA 12.1 runtime):

```
conda env create -f environment.yml
```

2) Activate it:

```
conda activate gpu-bench
```

3) Verify CUDA availability (optional):

```
python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('torch=', torch.__version__)"
```

If you prefer not to activate, you can run with:

```
conda run -n gpu-bench python benchmark.py --batch-size 32
```

Note: This benchmark requires CUDA; CPU execution is not supported in the current version.

### Use a project-local Conda environment (inside this folder)

如果你希望把 Conda 环境创建在项目文件夹内（方便“一文件夹即可跑/复制”），可以使用前缀路径方式：

1) 基于 environment.yml 在本目录创建环境（路径为 ./env）

```
conda env create -p ./env -f environment.yml
```

2) 激活该本地环境：

```
conda activate ./env
```

3) 可选：不激活情况下运行命令：

```
conda run -p ./env python benchmark.py --batch-size 32
```

4) 删除本地环境：

```
conda env remove -p ./env
```

本仓库的 Makefile 已经内置了这些命令的封装：

```
make env        # 使用 environment.yml 在 ./env 下创建本地环境
make env-pip    # 仅创建最小环境并用 pip 安装 requirements.txt
make run        # 使用本地环境运行基准测试（CUDA）
make run-cpu    # 使用本地环境运行 CPU 基准
make clean-env  # 删除 ./env
```

## Usage (CUDA-only)

Basic GPU run:

```
python benchmark.py --batch-size 32 --image-size 224 --precision fp32 --repeats 30 --warmup 10 --channels-last
```

Select subset of models:

```
python benchmark.py --models resnet50 convnext_tiny
```

Try mixed-precision (autocast) on CUDA:

```
python benchmark.py --precision fp16
python benchmark.py --precision bf16
```

Enable TorchScript trace optimization:

```
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
