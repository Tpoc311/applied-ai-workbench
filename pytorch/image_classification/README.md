# Image classification

A set of CNNs used for image classification task on ILSVRC2012 dataset.

## Train

Simple CNN from PyTorch tutorial:

```bash
python3 pytorch/image_classification/train_simple_cnn.py \
  --batch_size 32 \
  --num_workers 4 \
  --epochs 15 \
  --lr 0.002
```

AlexNet:

```bash
python3 pytorch/image_classification/train_alexnet.py
```

## Test

Simple CNN from PyTorch tutorial:

```bash
python3 pytorch/image_classification/test_simple_cnn.py \
  --load_model_path simple-net_cifar10_epoch15_batch32_lr0.002_momentum0.9.pt
```

AlexNet:

```bash
python3 pytorch/image_classification/test_simple_cnn.py \
  --models_dir artifacts/models/ImageNet-1000
```

## Results

Accuracy on the test set (10,000 images): 65%. No modifications were made to the architecture, and hyperparameters
weren't extensively tuned. This is just a quick baseline to verify that everything runs end-to-end.

## Sources

1. [Training a Classifier](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
2. [ImageNet Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)](https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php)
