# Penn-Fudan Database for Pedestrian Detection

In this experiment I will do finetuning of Fast R-CNN using
[Penn-Fudan Database for Pedestrian Detection and Segmentation](https://www.cis.upenn.edu/~jshi/ped_html). The dataset
contains bounding boxes annotation and masks. Despite this I will use only bounding boxes because the experiment is
about object detection only.

## Goal

Understand basic object detection model architecture and finetune pretrained weights on Penn-Fudan dataset.

## Experiment checklist

* [x] Research the Faster R-CNN architecture.
* [ ] Understand how object detection differs from image classification: model inputs, targets, outputs, loss functions,
  and evaluation.
* [x] Understand the main parts of a two-stage detector: backbone, RPN, RoI pooling/align, classification head, and box
  regression head.
* [ ] Refresh basic object detection concepts: bounding boxes, labels, objectness score, confidence score, IoU, NMS, and
  anchors.
* [ ] Research basic object detection metrics and choose the main one for evaluation.
* [x] Understand `collate_fn` in the data loader.
* [x] Load images, bounding boxes, labels, and image IDs in the format expected by the torchvision detection model.
* [ ] Configure detection transforms and data loaders.
* [ ] Fine-tune Faster R-CNN on the Penn-Fudan dataset.
* [ ] Evaluate the model using the chosen object detection metric.
* [ ] Visualize predictions with bounding boxes, confidence scores, and ground-truth boxes.
* [ ] Analyze typical errors: missed pedestrians, false positives, duplicate boxes, and inaccurate localization.
* [ ] Write conclusions about Faster R-CNN, transfer learning, and object detection metrics.

## Training

Finetuning full model:

```bash
python3 pytorch/object_detection/penn-fudan-pedestrian/train.py \
  --lr 0.001 \
  --epochs 32 \
  --step_size 8 \
  --run_name finetune \
  --mlflow_address http://host.docker.internal:8081
```

## Evaluation

```bash
python3 pytorch/object_detection/penn-fudan-pedestrian/eval.py
```
