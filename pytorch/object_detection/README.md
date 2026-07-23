# Object Detection

This section contains my object detection experiments.

The goal is to understand the main building blocks of object detectors, reproduce training pipelines for different
datasets and models, and move from classic architectures such as Faster R-CNN to more modern approaches such as
YOLO-family models and transformer-based detectors like DETR.

## Learning focus

Object detection extends image classification by not only predicting object classes, but also localizing each object
with a bounding box.

Classical object detectors are often grouped into two main families:

* **Two-stage detectors** — first generate region proposals, then classify and refine them.
* **One-stage detectors** — predict object classes and bounding boxes directly in a single pass.

I start with two-stage detectors because they provide a clear way to understand the core detection pipeline:
backbone, feature maps, region proposals, RoI heads, class prediction, and bounding box regression.

## Available experiments

The experiments are organized by dataset and model.

* [Penn-Fudan Database for Pedestrian Detection](./penn-fudan-pedestrian) - Faster R-CNN fine-tuning experiment for
  pedestrian detection.
