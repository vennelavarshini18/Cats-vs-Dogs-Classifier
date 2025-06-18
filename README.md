# 🐶🐱 Cats vs Dogs Classifier

A deep learning project using **TRANSFER LEARNING (Feature Extraction)** to classify images as either **Cat** or **Dog**. Built using TensorFlow, Keras, and Streamlit with a clean UI for user-friendly image uploads and real-time predictions.

---

## 📌 Features

- ✅ **Transfer Learning with MobileNetV2** for efficient and accurate predictions.
- 🖼️ Upload and classify images of cats or dogs using a sleek **Streamlit UI**.
- 📈 Improved **validation accuracy up to 95%+** with minimal overfitting.
- 🧠 Model trained on augmented dataset with real-time image transformations.
- 🧪 Includes real-time test accuracy evaluation and visualization.
- 🚀 Deployed locally via `streamlit run app.py`.

---

## 🧠 Model Architecture

- **Base Model:** MobileNetV2 (pre-trained on ImageNet)
- **Custom Layers:** Global Average Pooling, Dropout, Dense Softmax Layer
- **Loss:** Categorical Crossentropy
- **Optimizer:** Adam (lr = 1e-4)
- **Metrics:** Accuracy

---

## 📊 Model Performance

| Metric             | Value     |
|--------------------|-----------|
| Training Accuracy  | ~94+%      |
| Validation Accuracy| ~95+%      |
| Final Test Accuracy| ✅ Improved after transfer learning |
| Model Size         | Lightweight (due to MobileNetV2) |

---
