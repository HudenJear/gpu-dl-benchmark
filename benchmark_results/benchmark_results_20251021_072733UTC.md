# CUDA GPU Deep Learning Benchmark


This report benchmarks 5 representative torchvision models using random input tensors. Each model was run multiple times; results below report average latency and throughput.

## Environment

```
datetime_utc: 2025-10-21 07:27:36 UTC
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
| resnet50 | 32 | 224 | fp32 | 21.276 | 21.275 | 21.284 | 21.290 | 1504.07 | 30 | 10 |
| efficientnet_v2_s | 32 | 224 | fp32 | 23.297 | 23.246 | 23.334 | 23.922 | 1373.57 | 30 | 10 |
| convnext_tiny | 32 | 224 | fp32 | 29.907 | 29.895 | 30.089 | 30.118 | 1069.97 | 30 | 10 |
| swin_v2_t | 32 | 224 | fp32 | 46.671 | 46.658 | 46.712 | 47.055 | 685.65 | 30 | 10 |
| vit_b_16 | 32 | 224 | fp32 | 76.676 | 76.112 | 78.929 | 79.134 | 417.34 | 30 | 10 |
| gen_unet2d_condition | 32 | 224 | fp32 | 20.341 | 20.324 | 20.385 | 20.723 | 1573.21 | 30 | 10 |
| gen_unet2d | 32 | 224 | fp32 | 871.259 | 871.343 | 873.680 | 874.162 | 36.73 | 30 | 10 |
| gen_vae_autoencoderkl | 32 | 224 | fp32 | 178.882 | 178.833 | 179.496 | 179.753 | 178.89 | 30 | 10 |

## Notes

- Inputs are random tensors; results measure raw forward-pass performance, not data loading or preprocessing.
- TorchDynamo/compile is not enabled by default; add --jit to try TorchScript tracing optimization.
- cudnn.benchmark is enabled for kernel auto-tuning.