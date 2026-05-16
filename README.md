# 🌿 PlantGuard AI — Plant Disease Recognition System

A deep learning-powered web application that detects plant diseases from leaf images in real-time. Built with TensorFlow and Streamlit.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-orange?style=flat-square&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📋 Overview

PlantGuard AI analyzes plant leaf images using a Convolutional Neural Network (CNN) trained on the **PlantVillage dataset** — 87,000+ images across **38 disease categories** covering 14 different crops.

---

## ✨ Features

- 🔍 **Real-time Disease Detection** — Upload a leaf image and get instant results
- 🎯 **38 Disease Classes** — Covers healthy and diseased states across 14 crops
- 📊 **Confidence Score** — See how confident the model is in its prediction
- 🌙 **Modern Dark UI** — Clean, responsive interface built with custom CSS
- ⚡ **Cached Model Loading** — Fast predictions with `@st.cache_resource`

---

## 🌱 Supported Crops

| Crop | Crop | Crop |
|------|------|------|
| 🍎 Apple | 🫐 Blueberry | 🍒 Cherry |
| 🌽 Corn (Maize) | 🍇 Grape | 🍊 Orange |
| 🍑 Peach | 🫑 Bell Pepper | 🥔 Potato |
| 🫐 Raspberry | 🌱 Soybean | 🎃 Squash |
| 🍓 Strawberry | 🍅 Tomato | |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Pubudugunawardhana/Plant-Disease-Prediction.git
   cd Plant-Disease-Prediction
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   ```bash
   # Windows
   .\venv\Scripts\activate

   # macOS / Linux
   source venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirement.txt
   ```

5. **Run the app**
   ```bash
   streamlit run main.py
   ```

6. **Open in browser**
   ```
   http://localhost:8501
   ```

---

## 🗂️ Project Structure

```
Plant Disease Prediction/
│
├── main.py                  # Streamlit web app (UI + prediction logic)
├── trained_model.h5         # Pre-trained CNN model (Keras HDF5 format)
├── trained_model.keras      # Pre-trained CNN model (Keras native format)
├── training_hist.json       # Training history (loss/accuracy per epoch)
├── home_page.jpeg           # Hero image for the home page
├── requirement.txt          # Python dependencies
│
├── Train_plant_disease.ipynb  # Model training notebook
└── Test_Plant_Disease.ipynb   # Model testing/evaluation notebook
```

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Architecture | Convolutional Neural Network (CNN) |
| Input Size | 128 × 128 RGB |
| Output Classes | 38 |
| Training Images | 70,295 |
| Validation Images | 17,572 |
| Test Images | 33 |
| Framework | TensorFlow / Keras |

---

## 📦 Dependencies

```
tensorflow>=2.12.0
streamlit
numpy
pillow
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.13.0
pandas==2.1.0
```

---

## 🗃️ Dataset

This project uses the **New Plant Diseases Dataset** from Kaggle — recreated using offline augmentation from the original PlantVillage dataset.

📥 [Download Dataset on Kaggle](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)

> **Note:** The dataset is **not included** in this repository due to its size (~3GB). The pre-trained model (`trained_model.h5`) is included and ready to use — no dataset needed to run the app.

---

## 🖥️ Usage

1. Launch the app and navigate to **Disease Recognition** in the sidebar
2. Upload a clear image of a plant leaf (JPG, PNG, or WebP)
3. Click **Analyze Disease**
4. View the predicted plant, condition, and confidence score

---

## ⚠️ Known Issues & Fixes

| Issue | Cause | Fix Applied |
|---|---|---|
| `tensorflow==2.10.0` not found | Python 3.11 incompatible | Updated to `tensorflow>=2.12.0` |
| `ValueError: Layer 'conv2d' expected 2 variables` | Keras 3.x format mismatch | Switched to `trained_model.h5` with `compile=False` |
| `DLL load failed` on Windows | App Control policy blocked D: drive DLL | Move venv to `C:\Users\` trusted path |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [PlantVillage Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- [TensorFlow](https://www.tensorflow.org/)
- [Streamlit](https://streamlit.io/)
