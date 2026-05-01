# Image classification

A simple example of CNN from PyTorch tutorial.

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
python3 pytorch/image_classification/train_alexnet.py \
  --batch_size 256 \
  --num_workers 4 \
  --epochs 50 \
  --lr 0.001
```

## Test

Simple CNN from PyTorch tutorial:

```bash
python3 pytorch/image_classification/test_simple_cnn.py \
  --load_model_path simple-net_cifar10_epoch15_batch32_lr0.002_momentum0.9.pt \
  --batch_size 32
```

## Results

Accuracy on the test set (10,000 images): 65%. No modifications were made to the architecture, and hyperparameters
weren't extensively tuned. This is just a quick baseline to verify that everything runs end-to-end.

## Sources

1. [Training a Classifier](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
