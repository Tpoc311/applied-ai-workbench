# ImageNet-1000

This experiment was my second step after the simplest CNN baseline and studying the LeNet (1998) architecture.

## Goal

Understand a classic large-scale CNN architecture and train it myself on ImageNet.

## What I did

* Researched the AlexNet and ResNet architectures, its tensor flow and key differences.
* Used `torchvision`implementations to train both of them.
* Built a train/validation pipeline for ImageNet-1000.
* Implemented preprocessing, training loop, validation loop, metrics logging, and LR scheduling.
* Implemented test script which counts accuracy of prediction on validation set.

## AlexNet

### Training

Default training configuration is defined directly in the script arguments.

```bash
python3 pytorch/image_classification/train_imagenet1000.py \
  --model AlexNet
```

Resume training example:

```bash
python3 pytorch/image_classification/train_imagenet1000.py \
  --model AlexNet \
  --resume_from artifacts/models/ImageNet-1000/AlexNet/alexnet_imagenet1000_best_batch128_lr0.01_momentum0.9.pt
```

### Testing

```bash
python3 pytorch/image_classification/test_imagenet1000.py \
  --model AlexNet \
  --models_dir artifacts/models/ImageNet-1000/AlexNet
```

## ResNet

Models available for training are presented in `src/models/imagenet1000.py` in `models_dict`. For ResNet training I used
configuration as follows:

### Training

```bash
python3 pytorch/image_classification/train_imagenet1000.py \
  --model ResNet34 \
  --batch_size 256 \
  --num_workers 6 \
  --epochs 100 \
  --lr 0.1 \
  --weight_decay 0.0001
```

More over I switched the scheduler to 'ReduceLROnPlateau' cause the [ResNet authors](https://arxiv.org/abs/1512.03385)
used the same way to reduce learning rate:

```text
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=5,
        threshold=0.001,
        threshold_mode="abs",
        min_lr=1e-5,
    )
```

I also decided to optimize not `val_loss` here, but `val_error` to make it a bit closer to authors paper.

```text
val_top1_error = 1.0 - val_top1_acc
scheduler.step(val_top1_error)
```

Resume training example:

```bash
python3 pytorch/image_classification/train_imagenet1000.py \
  --model ResNet34 \
  --resume_from artifacts/models/ImageNet-1000/ResNet/34/resnet34_imagenet1000_best_batch256_lr0.1_momentum0.9.pt
```

### Testing

```bash
python3 pytorch/image_classification/test_imagenet1000.py \
  --model ResNet34 \
  --models_dir artifacts/models/ImageNet-1000/ResNet
```

## Result

### Choose best model

<p align="center">
    <img src="images/results/top-1_acc.png" width="40%">
    <img src="images/results/top-5_acc.png" width="40%">
    <img src="images/results/losses.png" width="40%">
</p>

After approximately 47 epochs, AlexNet shows almost no improvement in `val_top1_acc`, even though `train_top1_acc`
continues to rise. Consequently, further training primarily improves performance on the training set but yields almost
no gains in generalization.

For ResNet34, the optimal checkpoint should be selected based on the maximum `val_top1_acc`; judging by the graph, this
occurs somewhere around the 45–60 epoch mark. If the values remain nearly identical after epoch 45, it is best
to choose the earliest checkpoint that achieves maximum—or near-maximum—validation accuracy.

### Training time

AlexNet took around <b>20 minutes</b> per epoch and <b>30 hours</b> for 90 epochs.

ResNet34 took around <b>60 minutes</b> per epoch and almost <b>5 days</b> for 100 epochs.

This was not a strict historical reproduction of the original papers. I used the torchvision implementation and focused
on understanding the architecture, tensor flow, and the full training pipeline.

## Sources

1. [ImageNet Large Scale Visual Recognition Challenge 2012 (ILSVRC2012)](https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php)
2. [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
3. [One weird trick for parallelizing convolutional neural networks](https://arxiv.org/abs/1404.5997)
4. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
