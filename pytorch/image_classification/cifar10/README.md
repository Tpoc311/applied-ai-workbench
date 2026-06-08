# CIFAR-10

This is my baseline CNN experiment based on the official PyTorch tutorial.

## Goal

Understand the full image classification workflow in PyTorch on a simple dataset:

* Dataset and data loader setup.
* CNN forward pass.
* Loss computation.
* Optimizer step.
* Evaluation on a test set.

## What I did

* Reproduced the official PyTorch CIFAR-10 classification tutorial.
* Trained a small handwritten CNN.
* Used this experiment to understand the end-to-end training loop for image classification.
* Learned LeeNet (1998) architecture.

## Training

```bash
python3 pytorch/image_classification/train_simple_cnn.py
```

## Testing

```bash
python3 pytorch/image_classification/test_simple_cnn.py \
  --load_model_path artifacts/models/CIFAR10/simple-net_cifar10_epoch15_batch32_lr0.002_momentum0.9.pt
```

## Result

```text
Accuracy of the network on the 10000 test images: 63 %
```

Here I did not tune hyperparameters properly and simply kept the checkpoint with the lowest loss after 20 epochs.
Just understanding the basic PyTorch training pipeline for CNNs.

## Sources

1. [Training a Classifier](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
