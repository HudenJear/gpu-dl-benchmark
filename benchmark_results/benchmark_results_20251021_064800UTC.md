# CUDA GPU Deep Learning Benchmark


This report benchmarks 5 representative torchvision models using random input tensors. Each model was run multiple times; results below report average latency and throughput.

## Environment

```
datetime_utc: 2025-10-21 06:48:02 UTC
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
| resnet50 | 32 | 224 | fp32 | 21.285 | 21.280 | 21.323 | 21.388 | 1503.44 | 30 | 10 |
| efficientnet_v2_s | 32 | 224 | fp32 | 23.256 | 23.253 | 23.303 | 23.406 | 1375.98 | 30 | 10 |
| convnext_tiny | 32 | 224 | fp32 | 29.992 | 29.969 | 30.236 | 30.311 | 1066.95 | 30 | 10 |
| swin_v2_t | 32 | 224 | fp32 | 46.822 | 46.810 | 46.992 | 47.137 | 683.44 | 30 | 10 |
| vit_b_16 | 32 | 224 | fp32 | 76.713 | 76.008 | 78.792 | 78.888 | 417.14 | 30 | 10 |

## Notes

- Inputs are random tensors; results measure raw forward-pass performance, not data loading or preprocessing.
- TorchDynamo/compile is not enabled by default; add --jit to try TorchScript tracing optimization.
- cudnn.benchmark is enabled for kernel auto-tuning.