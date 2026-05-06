# Image classification

This section contains my image classification experiments while studying CNNs.
The goal is to understand CNN building blocks, reproduce baseline training pipelines, and gradually move from simple
tutorial-level models to classic architectures such as AlexNet and later ResNet.

## CIFAR-10

This is my baseline CNN experiment based on the official PyTorch tutorial.

### Goal

Understand the full image classification workflow in PyTorch on a simple dataset:

* Dataset and dataloader setup.
* CNN forward pass.
* Loss computation.
* Optimizer step.
* Evaluation on a test set.

### What I did

* Reproduced the official PyTorch CIFAR-10 classification tutorial.
* Trained a small hand-written CNN.
* Used this experiment to understand the end-to-end training loop for image classification.

### Train

```bash
python3 pytorch/image_classification/train_simple_cnn.py
```

### Test

```bash
python3 pytorch/image_classification/test_simple_cnn.py \
  --load_model_path artifacts/models/CIFAR10/simple-net_cifar10_epoch15_batch32_lr0.002_momentum0.9.pt
```

### Result

```text
Accuracy of the network on the 10000 test images: 63 %
```

This experiment helped me to understand the basic PyTorch training pipeline for CNNs.
At this stage I became comfortable with tensor shapes, convolutions, pooling, and linear layers, although the
optimization process itself still felt more like a black box.

Here I did not tune hyperparameters properly and simply kept the checkpoint with the lowest loss after 20 epochs.

## AlexNet (torchvision implementation)

This experiment was my second step after the simplest CNN baseline and studying the LeNet (1998) architecture.

### Goal

Understand a classic large-scale CNN architecture and verify that I can train it myself on ImageNet.

### Hardware used

I used my PC with the next hardware for training:

* GPU - 1x Nvidia RTX 3060 12 Gb.
* CPU - AMD Ryzen 9 7900X.
* RAM - 2x32 Gb DDR5.
* SSD - Samsung 990 PRO 2 TB.
* OS - Ubuntu 24.04 LTS.

### What I did

* Studied the AlexNet architecture and its tensor flow.
* Used `torchvision.models.AlexNet` as an architecture.
* Built a train/validation pipeline for ImageNet-1000.
* Implemented preprocessing, training loop, validation loop, metrics logging, and LR scheduling.
* Implemented test script which counts accuracy of prediction on validation set.

### Train

Default training configuration is defined directly in the script arguments.

```bash
python3 pytorch/image_classification/train_alexnet.py
```

### Test

```bash
python3 pytorch/image_classification/test_alexnet.py \
  --models_dir artifacts/models/ImageNet-1000/AlexNet
```

### Result

Best validation accuracy: <b>0.5571</b> at epoch 70. After the main improvement phase, validation accuracy mostly
plateaued around <b>0.554–0.557</b>.

Training process hardware above took around <b>20 minutes</b> oer epoch and <b>30 hours</b> for all 90 epochs.

This was not a strict historical reproduction of the original 2012 AlexNet paper.
I used the torchvision implementation and focused on understanding the architecture, tensor flow, and the full training
pipeline.

This experiment helped me move from a toy CNN setup to a real ImageNet-scale training pipeline.

## Sources

1. [Training a Classifier](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
2. [ImageNet Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)](https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php)
