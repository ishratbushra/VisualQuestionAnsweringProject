# VQA Project: Multi-Modal(Scene and Object-Aware, Caption-Guided) Visual Question Answering System

## 📌 Project Overview

This project implements a Visual Question Answering (VQA) system that combines multi-modal features—scene understanding, object detection, question embedding, and captioning—to answer natural language questions about images using deep learning techniques. Built on the VQA v1 Abstract Scenes Multiple-Choice Dataset.

**Scope:**

 - **Visual Understanding:** The model should accurately extract features from abstract scene images.
  
 - **Question Understanding:** The model must interpret multiple-choice questions in natural language.
  
 - **Answer Selection** The model should select the most appropriate answer from a set of multiple-choice options.

---

## 📂 Project Dataset Info

- **Dataset**: VQA v1 Abstract Scenes Training Dataset (first 2,000 samples used after cleaning and augmentation).
- **Dataset Source Link**: https://visualqa.org/vqa_v1_download.html
- **Answer Types**:
  - *Yes/No*
  - *Number*
  - *Other*
---

## 🧠 Model Architecture

### 🔹 Image Feature Extraction

- **Scene Features**: Extracted using **ResNet50 (Places365)** model.
- **Object-aware Features**: Extracted using **YOLOv8** pre-trained on **COCO dataset** (91 class objects).

### 🔹 Textual Processing

- **Questions**: Embedded using **MiniLM (BERT variant)**.
- **Captions**: Combined **BLIP-generated** and given captions, embedded using **MiniLM**.
- **Annotations**: Label encoded and one-hot encoded for classification.

### 🔹 Data Augmentation

- Used **Faster R-CNN** to generate count-based questions (e.g., "How many people are in the image?").
- Questions types include **"how many"** based on object detection.

---

## ⯯️ Data Preprocessing

### ✅ Scene Feature Pipeline

- Resized images from `700x700 → 224x224`
- Sorted by `image_id`
- Batched using PyTorch **DataLoader**
- Used **Mixed Precision** for faster GPU-based computation
- Features saved as `image_features.npy`, `image_ids.npy`

### ✅ Object-Aware Feature Pipeline

- Resized to `640x640`, used YOLOv8 backbone without detection head
- Extracted 1024-dim object-aware feature vectors
- Missing/unreadable images receive zero vector
- Saved to `.h5` format for memory efficiency

### ✅ Question Embedding

- Questions are short, semantic (e.g., Yes/No, Count, What color...)
- Embedded using **MiniLM** from `sentence-transformers`
- Stored in `.h5` with mapping: `image_id`, `question_id`, `embedding (384-d)`

### ✅ Caption Generation

- Generated captions using **BLIP** pretrained on COCO + web data
- Parameters tuned: `Temperature`, `Top-K`, `Top-P`, `Repetition Penalty`
- Produced 10 captions/image, merged with original caption

### ✅ Annotation Encoding

- The `multiple_choice_answer` and `multiple_choices` features are combined to form the **vocabulary list** of unique answers.
- This vocabulary list is encoded using label encoder to assign a unique integer to each answer.
- The `ground_truth` / `multiple_choice_answer` is then converted into integer index and transformed into one-hot vectors which is suitable for training.
- After training, the model predicts the answer by providing probability distribution over all possible classes. The index of highest probability is selected (**argmax**) and then corresponding answer is decoded using **inverse_transform**.

---

## 🏋️ Training Process

### 📊 Dataset Split

- **Train / Validation / Test** = 60% / 30% / 10%

### 🔗 Feature Fusion

- Normalized and concatenated:
  - ResNet50 Scene Features
  - YOLOv8 Object-aware Features
  - MiniLM Question Embeddings
  - MiniLM Caption Embeddings
- Combined using `np.hstack()`

### 🧱 Model Architecture

- **MLP with 4 Dense Layers**:
  - Dense(1024, ReLU) → Dropout(0.4)
  - Dense(512, ReLU) → Dropout(0.4)
  - Dense(256, ReLU) → Dropout(0.4)
  - Dense(128, ReLU) → Dense(18, Softmax)

### 🔧 Optimization

- **Loss Function**: Categorical Crossentropy with label smoothing = 0.1
- **Optimizer**: Adam (lr = 5e-5)
- **Mixed Precision**: Enabled (`mixed_float16`)

### 📂 Vocabulary Processing

- The vocabulary list is necessary for predicting answers from a fixed set of options
- Label encoded answers into integers and applied one-hot encoding
- One-hot outputs compared against softmax probabilities for classification

---

## 📈 Model Performance & Evaluation

### ✅ Accuracy Timeline

| Dataset Size                   | Features Used                                 | Optimization                                 | Accuracy   |
| ------------------------------ | --------------------------------------------- | -------------------------------------------- | ---------- |
| 20,000 images                  | ResNet50 + Question Embedding                 | Epochs = 20                                  | ~5%        |
| 20,000 images                  | ResNet50 + Question Embedding + YOLO          | Epochs = 20                                  | 23.72%     |
| 2,000 images (Baseline Model) | ResNet50 + Question Embedding + YOLO + Caption (Given) | Epochs = 20                        | 22.00%     |
| 2,000 images                  | ResNet50 + Question Embedding + YOLO + Data Augmentation | 3 Dense Layers, Dropout=0.3, epochs=20 | 37.06%     |
| 2,000 images                  | ResNet50 + Question Embedding + YOLO + Augmentation + Caption (Given + Generated) | 3 Dense Layers, Dropout=0.3, epochs=20 | 45.64%     |
| 2,000 images (Optimized Model) | ResNet50 + Question Embedding + YOLO + Augmentation + Caption (Given + Generated) | 4 Dense Layers, Dropout=0.4, label smoothing=0.1, epochs=25 | 53.32%     |

### ✅ Confusion Matrices

- Answer Type Matrix (Yes/No, Number, Other)
- Category-specific matrices for numerical answers and top "other" answers
- Binary Matrix for Yes/No classification


---

## 🔍 Interpretability Issues & Ethical Considerations

### 🧠 Ethical and Interpretability:

#### Bias and Fairness:

- **Species Bias**: More dogs/cats than rare animals → overfitting to frequent classes.
- **Visual Bias**: Racket → girl, Frisbee → sun.
- **Role Bias in Captions**: Generated captions assume gender/roles (e.g., "a woman cooking").
- **Language Bias**: MiniBERT trained on large corpora may reflect stereotypical associations.

#### Limitation of Pre-trained Models:

- YOLOv8 pre-trained on COCO → "stool" often misclassified as "chair".
- Dataset has only 91 objects → limits rare class detection.

#### Lack of Clarity:

- Some questions are vague (e.g., "2 hamburgers and a watermelon") and lack precision.

#### Transparency:

- Generated captions were often too vague or too specific, reducing explainability.

### 🔐 Responsible Use:

- This model performs well in required tasks but not suitable for high-risk domains (medical, legal, surveillance).
- Requires human oversight when handling real-world or sensitive data.

---

## 🚀 Future Work & Next Steps

### ✅ Planned Improvements

- **Tune Captioning Module**: Use prompts (e.g., "A photo of object") with object detector to refine generated captions.
- **Enhance Feature Understanding**: Make modules more question-aware to understand color, spatial relationships.
- **Increase Dataset Size**: Focus on real-world data.
- **Compute & Memory**: Utilize high RAM/VRAM GPU/TPU to support transformer-based fusion techniques.

### ✅ Observations

- Increase Dataset size and work on real world data.

---

## 📜 License

GNU General Public License v2.0

---

## 👤 Author

- **Ishrat Jaben Bushra** – Master’s Student, Data Science and Analytics, Toronto Metropolitan University
- **Co-Author**: Aarthi Saravanan (Master’s Student, Data Science and Analytics, Toronto Metropolitan University)


