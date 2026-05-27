# 🚀 CodeAlpha Machine Learning Internship Tasks

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange?logo=tensorflow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikit-learn&logoColor=white)

This repository contains the implementation of four core Machine Learning tasks for the CodeAlpha ML Internship. The project is designed with best engineering practices, featuring modular code, proper evaluation metrics, and a fully configured virtual environment.

---

## 📂 Project Structure

- **`task1_credit_scoring.py`** — Binary classification using Logistic Regression, Decision Trees, and Random Forests on the German Credit dataset.
- **`task2_emotion_recognition_speech.py`** — Audio feature extraction (MFCCs) and classification using a Convolutional Neural Network (CNN) on the EMO-DB dataset.
- **`task3_handwritten_character_recognition.py`** — Deep Learning Computer Vision task using a CNN to classify MNIST handwritten digits (96.7% accuracy).
- **`task4_disease_prediction.py`** — Medical diagnostic modeling using SVM, XGBoost, and Logistic Regression on the Breast Cancer Wisconsin dataset (98.25% accuracy).
- **`ML-Internship-CodeAlpha.md`** — 📖 **Must Read:** A comprehensive guide breaking down the math, intuition, and algorithms behind these tasks for ML beginners.

---

## ⚙️ How to Setup and Run (Beginner Friendly)

This project uses a **Virtual Environment (`.venv312`)** to keep dependencies (like `tensorflow`, `scikit-learn`, `librosa`) isolated from your main system. 

### Step 1: Activate the Virtual Environment
Before running any script, you must activate the environment in your terminal (PowerShell recommended). 

```powershell
# Open your terminal in this project folder and run:
.venv312\Scripts\Activate.ps1
```
*(You should now see `(.venv312)` at the beginning of your terminal prompt. This means you are safely inside the environment!)*

### Step 2: Run the Tasks
Once the environment is active, you can run any of the tasks using the standard `python` command.

**▶️ Task 1: Credit Scoring**
*(Auto-downloads the German Credit dataset)*
```powershell
python task1_credit_scoring.py
```

**▶️ Task 2: Speech Emotion Recognition**
*(Requires the EMO-DB dataset placed in the `Datasets\wav` folder)*
```powershell
python task2_emotion_recognition_speech.py --data-dir "Datasets\wav" --dataset emodb --epochs 30 --batch-size 16
```

**▶️ Task 3: Handwritten Character Recognition**
*(Auto-downloads the MNIST dataset and trains a CNN in real-time)*
```powershell
python task3_handwritten_character_recognition.py --epochs 5
```

**▶️ Task 4: Disease Prediction**
*(Uses sklearn's built-in Breast Cancer dataset)*
```powershell
python task4_disease_prediction.py
```

---

## 🧠 Learning Resources
If you are learning Machine Learning, don't just run the code! Open the **[`ML-Internship-CodeAlpha.md`](./ML-Internship-CodeAlpha.md)** file included in this repository. It is a comprehensive guide written from a Senior ML Engineer's perspective, explaining:
- The intuition and math behind every algorithm used here.
- Why precision/recall matter more than accuracy in medical datasets.
- A full 6-month roadmap to becoming a solid junior ML engineer.

---
*Developed for the CodeAlpha Machine Learning Track.*
