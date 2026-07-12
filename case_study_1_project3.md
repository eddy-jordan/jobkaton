# Fine-Tuning ResNet50 for Pneumonia Detection: What I Learned About Overfitting, Imbalance, and Knowing When to Stop

*A technical deep dive into Project 3 of my MLOps portfolio — building a chest X-ray classifier with PyTorch and transfer learning.*

---

## The Problem

Chest X-ray classification is a genuinely hard computer vision problem to solve from scratch — medical images have subtle, high-frequency patterns that a small dataset alone can't teach a model to recognize. The [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset I used has about 5,800 images total, which sounds like a lot until you compare it to the millions of images a model like ResNet50 was originally trained on.

That gap is exactly why I chose **transfer learning** instead of training a convolutional network from scratch.

## Why Transfer Learning, and Why Freeze the Base

ResNet50, pretrained on ImageNet, has already learned to detect general visual features — edges, textures, gradients, shapes — in its early and middle layers. Those features aren't specific to cats and dogs; they're useful for almost any image, including X-rays. So instead of retraining the whole network, I froze every layer except the final fully-connected classification layer, and only trained that.

This decision had two direct consequences worth being explicit about:

1. **Much less risk of overfitting.** With only ~5,800 training images, trying to fine-tune all 25 million of ResNet50's parameters would almost certainly memorize the training set rather than generalize.
2. **Much faster training.** Only the final layer's weights needed gradient updates, which meant each epoch took a few minutes on CPU rather than hours.

The trade-off is that the frozen layers can't adapt at all to X-ray-specific patterns that differ meaningfully from natural images. In a production setting with more data and compute budget, a reasonable next step would be *unfreezing* the last few convolutional blocks and fine-tuning them at a much lower learning rate — a technique called progressive unfreezing. For this project's scope, freezing everything but the classifier head was the right call.

## The Overfitting Story, in the Actual Numbers

Here's what training looked like, epoch by epoch:

| Epoch | Train Loss | Val Loss | Learning Rate |
|---|---|---|---|
| 1 | 0.2916 | 0.6415 | 0.001 |
| 2 | 0.1863 | **0.5103** | 0.001 |
| 3 | 0.1595 | 0.6186 | 0.001 |
| 4 | 0.1521 | 0.7310 | 0.001 |
| 5 | 0.1439 | 0.5366 | 0.0005 |
| 6 | 0.1302 | 0.5874 | 0.0005 |

Look at what's happening: training loss drops steadily the entire time — the model is clearly learning *something* on every epoch. But validation loss bottoms out at **epoch 2** and never gets that low again. By epoch 4, it's noticeably worse than where it started.

This is textbook overfitting, and it's the exact reason I built early stopping into the training loop rather than just running a fixed number of epochs and taking whatever came out. My early stopping logic tracked the best validation loss seen so far and restored those exact weights — not the final epoch's weights — once four epochs passed without improvement. That meant the model I actually evaluated and saved was the epoch 2 checkpoint, not epoch 6, even though the loop kept running until epoch 6.

Without this, I'd have shipped a measurably worse model without ever knowing it, just because I happened to stop training at an arbitrary epoch count.

## The Class Imbalance I Didn't Fully Address — On Purpose

The dataset isn't balanced: pneumonia cases outnumber normal cases roughly 3-to-1 in the training set. I didn't apply class weighting or oversampling to correct this, and the final results show why that decision matters:

| | Precision | Recall |
|---|---|---|
| NORMAL | 0.89 | 0.64 |
| PNEUMONIA | 0.81 | **0.95** |

The model is clearly biased toward predicting PNEUMONIA — it catches 95% of actual pneumonia cases, but also misclassifies over a third of healthy patients as having pneumonia (85 false positives out of 234 true NORMAL cases).

Here's the honest reasoning: **in a medical screening context, this bias is arguably the right failure mode to have.** A false positive (flagging a healthy patient for a closer look) costs a follow-up exam. A false negative (missing an actual pneumonia case) costs a misdiagnosis. Those two errors are not equally expensive, and a model that leans toward over-flagging is generally safer than one that leans toward under-flagging.

That said, I want to be precise about what actually happened here: this behavior emerged *incidentally* from the class imbalance in the training data, not from a deliberate design choice to weight recall over precision. If I were taking this into an actual clinical deployment, I'd want to control this trade-off intentionally — using class weights, a custom loss function, or an explicit decision threshold tuned on a validation set — rather than relying on however the raw data distribution happens to shake out. Getting a favorable trade-off by accident isn't the same as engineering it deliberately, and I'd be upfront about that distinction with any team I was working with.

## What I'd Do Differently With More Time

- **Progressive unfreezing** of the last few ResNet blocks, with a much smaller learning rate, to let the model adapt more specifically to X-ray textures.
- **Explicit class weighting** in the loss function, so the precision/recall trade-off is a chosen parameter rather than an accident of the data.
- **Cross-validation** rather than a single train/val/test split, to get a more reliable estimate of how much the epoch-2 "best" checkpoint was luck versus signal.

## Closing Thought

The most useful part of this project wasn't getting to 83% accuracy — it was building the discipline to *watch* the validation curve instead of trusting a single final number, and being willing to say clearly which parts of the result were deliberate engineering versus a byproduct of the data. That distinction is, I think, the actual difference between running a training script and understanding a model.
