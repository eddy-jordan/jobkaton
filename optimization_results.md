# Optimization Results

## Objective
Convert the fine-tuned ResNet50 pneumonia classifier (Project 3) to ONNX format and measure the impact on inference latency and model size.

## Method
- Original model: PyTorch ResNet50 (`resnet50_pneumonia.pth`), fine-tuned for binary chest X-ray classification (NORMAL vs PNEUMONIA).
- Optimized model: same weights exported to ONNX (`resnet50_pneumonia.onnx`) via `torch.onnx.export`, opset 13.
- Benchmark: 50 single-image inference calls on CPU, after 5 untimed warm-up runs, using identical random input tensors for both models.
- Correctness check: outputs from both models compared directly; max absolute difference was 0.000002 (effectively identical).

## Results

| Metric | PyTorch (original) | ONNX (optimized) | Change |
|---|---|---|---|
| Avg. inference latency | 366.0 ms | 99.3 ms | 72.9% faster |
| Model file size | 90.0 MB | 89.6 MB | 0.4% |

**Summary:** Reduced inference time from 366ms to 99ms using ONNX (72.9% latency reduction), with output predictions matching the original PyTorch model.

## Notes
- Benchmarked on CPU, single-image batch size. Real-world gains would likely be larger under concurrent/batched load, which ONNX Runtime handles more efficiently than eager-mode PyTorch.
- No accuracy was lost in this conversion -- ONNX export preserves the exact learned weights; only the execution engine changes.
