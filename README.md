# Doshisha Logo Detection System

An implementation of the classic **R-CNN** object detection pipeline (Girshick et al., *Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation*, CVPR 2014), applied to detecting the **Doshisha University logo** in images.

The detector combines **Selective Search** for region proposals with a **fine-tuned ResNet-18** classifier and **Non-Maximum Suppression** for cleaning up overlapping detections.

---

## Overview

This project reproduces the core ideas of R-CNN as a two-stage detector:

1. **Region proposal** — OpenCV's Selective Search (Fast mode, `cv2.ximgproc`) generates candidate bounding boxes for each image.
2. **Region classification** — A fine-tuned **ResNet-18** (pretrained on ImageNet) classifies each proposed region as *logo* or *other*.
3. **Post-processing** — **Non-Maximum Suppression (NMS)** removes overlapping detections, keeping only the most confident boxes.

### Differences from the original paper

| | R-CNN (2014) | This project |
|---|---|---|
| Backbone | AlexNet | ResNet-18 |
| Input size | 227 × 227 | 224 × 224 |
| Classifier | Linear SVM | Fine-tuned FC head (dropout + linear) |
| Region proposals | Selective Search | Selective Search (Fast mode) |

The overall pipeline (region proposals → warp regions → CNN features → classify → NMS) stays faithful to the paper; the backbone and classification head are modernized.

---

## Pipeline

```
Input image
    │
    ▼
Selective Search (Fast)  ──►  ~thousands of region proposals
    │
    ▼
Warp each region to 224 × 224
    │
    ▼
Fine-tuned ResNet-18  ──►  score: P(logo)
    │
    ▼
Threshold (score_thresh) + Non-Maximum Suppression (iou_thresh)
    │
    ▼
Final bounding boxes
```

---

## Dataset

The model is trained on a small, manually curated dataset:

```
doshisha-logo-detection/
├── positives/    (~40 Doshisha-logo crops: wedge alone, wedge + wordmark, signs, varied sizes)
└── negatives/    (~190 "other" crops, including hard negatives like "150th" text and purple banners)
```

The train/validation split is performed **before** any augmentation or oversampling, so the validation set contains only unseen real images. This gives a more reliable measure of generalization than earlier synthetic-image experiments, where train and validation images were generated the same way.

---

## Method

**Data augmentation** (training set only): random resized crop, horizontal flip, rotation, color jitter, and random grayscale — applied to help the CNN generalize from a small dataset. Validation images are only resized and normalized.

**Model**: ResNet-18 pretrained on ImageNet. Early layers are frozen to preserve general features; only `layer4` is fine-tuned. The original classification layer is replaced with a dropout layer (p = 0.5) followed by a new linear layer with 2 outputs (*logo* / *other*).

**Training**:
- Optimizer: Adam (lr = 1e-4, weight decay = 1e-4)
- Loss: Cross-entropy
- Batch size: 32
- Max epochs: 12, with early stopping (patience = 3)

**Inference**: Selective Search generates proposals, which are scored in batches by the CNN. Proposals above `score_thresh` are kept, then NMS (`iou_thresh`) removes overlaps.

---

## Tech Stack

- **Python**, **PyTorch**, **Torchvision**
- **OpenCV** (`opencv-contrib-python` — required for Selective Search via `cv2.ximgproc`)
- **NumPy**, **Matplotlib**
- Built and run in **Google Colab** (Drive mount + file upload workflow)

---

## Getting Started

This project was built to run in **Google Colab** with a GPU runtime.

### 1. Install dependencies

Colab ships with standard OpenCV, which does **not** include Selective Search. The notebook removes it and installs the contrib version:

```bash
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless
pip install opencv-contrib-python
```

> **Important:** After installing, restart the runtime (**Runtime → Restart session**) before running the remaining cells.

### 2. Prepare your data

Upload your `positives/` and `negatives/` folders to Google Drive and update the `WORK` path in the notebook to point to your dataset folder.

### 3. Run the notebook

Execute the cells in order to:
1. Install dependencies and import libraries
2. Load and split the dataset
3. Set up augmentation and data loaders
4. Fine-tune ResNet-18
5. Run the full detection pipeline on a new uploaded test image

---

## Results

Run the final cell to upload a test image and see detections drawn with confidence scores:

```python
boxes, scores = detect(img_bgr, score_thresh=0.5, iou_thresh=0.3)
show_detections(img_bgr, boxes, scores)
```

*(Add a sample output image here to showcase your detections.)*

---

## Limitations

- Trained on a **small, manually curated dataset**, so performance is strongest on logo variants similar to the training crops.
- May struggle with heavy occlusion, extreme scale changes, or unusual backgrounds.
- Selective Search **Fast** mode trades some recall for speed compared to Quality mode.
- R-CNN scores each region proposal independently, so inference is slower than modern single-shot detectors (e.g. YOLO, SSD).

---

## Reference

R. Girshick, J. Donahue, T. Darrell, J. Malik. *Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation.* CVPR 2014.

---

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.