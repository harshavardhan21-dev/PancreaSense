# 🩺 PancreaSense

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.46-red)

---

> **AI-powered pancreas detection and segmentation from abdominal CT scans using CNN classification and U-Net deep learning.**

PancreaSense is a deep learning–based medical imaging application that analyzes abdominal CT scan slices to automatically detect and precisely segment the pancreas region. It combines a CNN classifier for presence detection with a U-Net model for pixel-wise segmentation, wrapped in an interactive Streamlit dashboard.

---

## 📸 Demo

> Upload a CT scan → get detection status, segmentation mask, overlay visualization — all in seconds.

---

## ✨ Features

- 🔍 **Pancreas Detection** — CNN classifier with 98.5% test accuracy determines whether a pancreas is visible in the CT slice
- 🎯 **Precise Segmentation** — U-Net model generates pixel-wise segmentation masks (Dice: 0.8142)
- 🖼️ **3-Panel Viewer** — Side-by-side display of Original CT · Segmentation Mask · Overlay
- 📊 **Quantitative Metrics** — Confidence score, segmented area (px), and region coverage (%)
- ⚡ **Real-time Analysis** — Instant results via an interactive Streamlit web app

---

## 🎯 Model Performance

### CNN Classifier (Detection)

| Metric    | Value   |
|-----------|---------|
| Accuracy  | 98.50%  |
| Precision | 97.83%  |
| Recall    | 98.04%  |

### U-Net (Segmentation)

| Metric            | Value               |
|-------------------|---------------------|
| Best Dice (val)   | 0.8142              |
| Training Epochs   | 15/20 (early stop)  |
| Input Resolution  | 256 × 256 px        |

---

## 🔬 How It Works

```
CT Image Upload
      │
      ▼
Preprocessing (grayscale → resize 256×256 → normalize)
      │
      ├──► CNN Classifier ──► Detection Score (confidence %)
      │
      └──► U-Net Model ──────► Binary Segmentation Mask
                                        │
                                        ▼
                            Overlay Generation + Area Calculation
                                        │
                                        ▼
                              Streamlit Results Dashboard
```

- Images are converted to grayscale (single-channel) to match the medical CT domain
- Both models share the same preprocessed tensor for efficiency
- Segmentation mask is thresholded at 0.5; overlays are rendered in red on the original CT
- Detection uses a combined check: classifier confidence ≥ 0.5 **and** mask area > 100 px

---

## 🗂️ Project Structure

```
PancreaSense/
├── notebooks/
│   ├── cnn_training.ipynb         # CNN classifier training
│   └── unet_training.ipynb        # U-Net segmentation training
├── src/
│   ├── classifier.py              # CNN model loader + predict_pancreas()
│   ├── segmentation.py            # U-Net loader + predict_mask() + Dice metric
│   ├── preprocess.py              # Image loading, resizing, normalization
│   └── predict.py                 # End-to-end pipeline + overlay generation
├── models/                        # Keep downloaded trained model here
├── app/                          
│   └── streamlit_app.py           # Streamlit UI
├── config.py                      # Global constants (IMG_SIZE, thresholds, paths)
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/harshavardhan21-dev/PancreaSense.git
cd PancreaSense
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up the trained models

The model files are not included in this repository due to GitHub's file size limits.

**Option A — Download pre-trained models (recommended)**

Download the pre-trained models from the link below and place them in the `models/` folder:

> 📥 **[Download pre-trained models](https://drive.google.com/drive/folders/1EdP7qkI2_wSjYmlq6VSI-143w1uIYf04?usp=sharing)**

```
models/
├── classifier.keras
└── unet.keras
```

**Option B — Train from scratch**

1. Download the dataset (see [Dataset](#-dataset) section)
2. Run `notebooks/cnn_training.ipynb` to train the CNN classifier
3. Run `notebooks/unet_training.ipynb` to train the U-Net model
4. Save the generated models into the `models/` folder

### 5. Run the application

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser and upload an abdominal CT image (PNG or JPG).

---

## ⚙️ Tech Stack

| Layer          | Technology                              |
|----------------|-----------------------------------------|
| Deep Learning  | TensorFlow 2.19 / Keras                 |
| Models         | CNN (classification), U-Net (segmentation) |
| UI             | Streamlit 1.46                          |
| Image I/O      | Pillow, OpenCV                          |
| Numerical      | NumPy, scikit-learn                     |
| Visualization  | Matplotlib                              |

---

## 📦 Requirements

```
tensorflow==2.19.0
streamlit==1.46.1
numpy==2.1.3
pandas==2.2.3
matplotlib==3.10.3
pillow==11.2.1
opencv-python==4.11.0.86
scikit-learn==1.7.0
h5py==3.14.0
kagglehub
```

---

## 📊 Dataset

**Pancreas CT Segmentation — Kaggle**

- Source: [`nandeeshhu/pancrease-ct-segmenatation`](https://www.kaggle.com/datasets/nandeeshhu/pancrease-ct-segmenatation)
- Task: Pancreas Segmentation
- Modality: CT Scan
- Organ: Pancreas
- Training Images: 5505
- Test Images: 1377
- Input Resolution: 256×256

To download the dataset:

```python
import kagglehub

path = kagglehub.dataset_download("nandeeshhu/pancrease-ct-segmenatation")
print("Path to dataset files:", path)
```

---

## 🔮 Future Scope

- [ ] DICOM image support
- [ ] 3D U-Net implementation
- [ ] Multi-organ segmentation
- [ ] Tumor detection and localization
- [ ] Cloud deployment (Streamlit Cloud / Hugging Face Spaces)
- [ ] Clinical validation studies

---

## ⚠️ Limitations

- Trained on the Pancreas-CT-SEG dataset; performance may vary on unseen datasets, scanners, or imaging conditions.
- Supports PNG and JPG formats only (no DICOM)
- Intended for research and educational purposes only and should not be used for clinical diagnosis or medical decision-making.

---

## ⚕️ Clinical Disclaimer

PancreaSense is an **AI-assisted screening tool** developed for research and educational purposes. All outputs must be reviewed by a qualified radiologist before any clinical decision-making. **This tool is not a substitute for professional medical diagnosis.**

---

## 👨‍💻 Author

**Harsha Vardhan S**

B.E. Computer Science and Engineering 
R.M.D Engineering College

[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?logo=github)](https://github.com/harshavardhan21-dev)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/harsha-vardhan-s-7019212b8/)

---

## 📄 License

This project is developed for academic and educational purposes. Contact the author for usage permissions.
