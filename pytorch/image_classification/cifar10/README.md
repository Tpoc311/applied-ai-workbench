# CIFAR-10

This was my first baseline CNN experiment based on the official PyTorch tutorial.

## Goal

Understand the full image classification workflow in PyTorch on a simple dataset:

* Dataset and data loader setup.
* CNN forward pass.
* Loss computation.
* Optimizer step.
* Evaluation on a test set.

## Experiment checklist

* [x] Reproduced the official PyTorch CIFAR-10 classification tutorial.
* [x] Trained a small CNN written from scratch.
* [x] Used this experiment to understand the end-to-end training loop for image classification.
* [x] Learned LeNet-5 architecture.
* [x] Implemented an evaluation script that computes accuracy on the test set.

## Training

```bash
python3 pytorch/image_classification/cifar10/train.py
```

## Testing

```bash
python3 pytorch/image_classification/cifar10/test.py \
  --load_model_path artifacts/models/CIFAR10/simple-net_cifar10_epoch15_batch32_lr0.002_momentum0.9.pt
```

## Results

```text
Accuracy of the network on the 10000 test images: 63 %
```

I did not tune hyperparameters in this experiment and used it mainly as a baseline for understanding the basic PyTorch
training pipeline for CNN-based image classification.

## Sources

1. [Training a Classifier](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
