# 🧠 Visual Question Answering (VQA) with Multi-Modal Deep Learning

This repository presents a **multi-modal VQA system** that integrates image, object, question, and caption features to answer multiple-choice questions on abstract scene images. The final model achieves a test accuracy of **47.98%** using optimized feature fusion and training techniques.

---

## 🔍 Project Overview

- **Objective**: Build a VQA model to predict answers to questions based on abstract images using multiple data modalities.
- **Dataset**: VQA v1 Abstract Scenes Training Dataset ( first 2,000 samples used after cleaning and augmentation).
- **Dataset Source Link**:https://visualqa.org/vqa_v1_download.html
- **Answer Types**: 
  - *Yes/No*
  - *Number*
  - *Other*

---

## 🏗️ Architecture Summary

The model uses **feature fusion** from the following inputs:
- **Image Features**: `ResNet50 (Places365)` scene embeddings
- **Object Features**: `YOLOv8` object-level context
- **Textual Features**:
  - Questions → `MiniBERT` embeddings
  - Captions (given + generated) → `BLIP` + `MiniBERT`

The fused vectors are fed into a **Multilayer Perceptron (MLP)** for classification.

---

## ⚙️ Training Strategy

### ✅ Dataset Split
- **60%** Train | **30%** Validation | **10%** Test

### ✅ Preprocessing
- All feature inputs normalized
- One-hot encoding for answers
- Memory-efficient loading using `.npy` and `.h5` files

### ✅ Model Architecture
```
Input → Dense(1024, ReLU) → Dropout(0.4) 
      → Dense(512, ReLU) → Dropout(0.4)
      → Dense(256, ReLU) → Dropout(0.4)
      → Dense(128, ReLU) → Dense(18, Softmax)
```

### ✅ Optimizer & Loss
- Optimizer: `Adam` with learning rate `5e-5`
- Loss: `Categorical Crossentropy` with label smoothing = 0.1
- Mixed Precision Training: Enabled (`mixed_float16`)

---

## 📈 Experiment Timeline

| Dataset        | Features Used                         | Optimization Details                                | Accuracy   |
|----------------|----------------------------------------|----------------------------------------------------|------------|
| 20,000 images  | Image + Question + YOLO                | YOLO features only, 20 epochs                      | 23.72%     |
| 2000 images    | Image + Question + YOLO + Augmented    | Data Aug, 3-layer MLP, Dropout=0.3, 20 epochs      | 37.06%     |
| 2000 (Baseline)| Image + Caption + Question + YOLO + Aug| Caption Gen, Dropout=0.3, 20 epochs                | 45.64%     |
| 2000 (Best)    | All modalities + Augmented             | 4-layer MLP, Dropout=0.4, smoothing=0.1, 25 epochs | **48.37%** |

---

## ✅ Class Imbalance Handling
- Used **soft-matching accuracy** for evaluating close answers.
- Added **count-based numeric questions** via `Faster R-CNN` object detection.
- Performed **caption augmentation** using BLIP model.

---

## 🧪 Evaluation

- Accuracy: **47.98%**
- Metrics:
  - Confusion matrices by *answer type* (`yes/no`, `number`, `other`)
  - Specific confusion matrices for top answers in each category

---
