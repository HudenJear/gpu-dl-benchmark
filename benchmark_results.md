# CUDA GPU Deep Learning Benchmark


This report benchmarks 5 representative torchvision models using random input tensors. Each model was run multiple times; results below report average latency and throughput.

## Environment

```
datetime_utc: 2025-10-21 06:44:55 UTC
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
| resnet50 | 32 | 224 | fp32 | 21.283 | 21.276 | 21.290 | 21.386 | 1503.56 | 30 | 10 |
| efficientnet_v2_s | 32 | 224 | fp32 | 23.216 | 23.207 | 23.266 | 23.271 | 1378.34 | 30 | 10 |
| convnext_tiny | 32 | 224 | fp32 | 29.929 | 29.918 | 30.198 | 30.667 | 1069.20 | 30 | 10 |
| swin_v2_t | 32 | 224 | fp32 | 46.579 | 46.577 | 46.651 | 46.693 | 687.00 | 30 | 10 |
| vit_b_16 | 32 | 224 | fp32 | 76.511 | 75.945 | 78.166 | 78.805 | 418.24 | 30 | 10 |

## Notes

- Inputs are random tensors; results measure raw forward-pass performance, not data loading or preprocessing.
- TorchDynamo/compile is not enabled by default; add --jit to try TorchScript tracing optimization.
- cudnn.benchmark is enabled for kernel auto-tuning.