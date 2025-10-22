# CUDA GPU Deep Learning Benchmark


This report benchmarks 5 representative torchvision models using random input tensors. Each model was run multiple times; results below report average latency and throughput.

## Environment

```
datetime_utc: 2025-10-21 06:55:11 UTC
python: 3.12.12 | packaged by Anaconda, Inc. |
pytorch: 2.9.0+cu128
torchvision: 0.24.0+cu128
platform: Linux 5.4.0-216-generic (x86_64)
device_name: NVIDIA GeForce RTX 3090
cuda_runtime: 12.8
driver: unknown
capability: 8.6
multi_gpu_count: 8
total_memory: 23.7GB
cudnn: 91002
```

## Results

| Model | Batch | Size | Precision | Mean Latency (ms) | P50 (ms) | P90 (ms) | P99 (ms) | Throughput (img/s) | Repeats | Warmup |
|:------|------:|-----:|:---------:|------------------:|---------:|---------:|---------:|-------------------:|--------:|-------:|
| resnet50 | 32 | 224 | fp32 | 21.216 | 21.199 | 21.217 | 21.445 | 1508.32 | 30 | 10 |
| efficientnet_v2_s | 32 | 224 | fp32 | 23.240 | 23.216 | 23.305 | 23.322 | 1376.92 | 30 | 10 |
| convnext_tiny | 32 | 224 | fp32 | 29.987 | 29.978 | 30.229 | 30.336 | 1067.15 | 30 | 10 |
| swin_v2_t | 32 | 224 | fp32 | 46.732 | 46.722 | 46.837 | 46.958 | 684.76 | 30 | 10 |
| vit_b_16 | 32 | 224 | fp32 | 76.418 | 75.921 | 78.218 | 79.247 | 418.75 | 30 | 10 |

## Notes

- Inputs are random tensors; results measure raw forward-pass performance, not data loading or preprocessing.
- TorchDynamo/compile is not enabled by default; add --jit to try TorchScript tracing optimization.
- cudnn.benchmark is enabled for kernel auto-tuning.