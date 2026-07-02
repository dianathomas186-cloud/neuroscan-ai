# brain tumor cnn
Brain tumor MRI classifier with Grad-CAM and RAG 
# 🧠 NeuroScan AI

**AI-Powered Brain Tumor MRI Classification with Explainable AI and Clinical Chat Assistant**

NeuroScan AI is a Flask-based web application that classifies brain MRI scans into four categories using a deep learning model, visually explains its predictions with Grad-CAM heatmaps, generates AI-assisted clinical reports using Retrieval-Augmented Generation (RAG), and lets users ask follow-up questions through an interactive chat assistant.

---

## 📋 Table of Contents

- [Features](#-features)
- [Demo](#-demo)
- [Tech Stack](#-tech-stack)
- [Model Performance](#-model-performance)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [How It Works](#-how-it-works)
- [Future Improvements](#-future-improvements)
- [Acknowledgements](#-acknowledgements)

---

## ✨ Features

- 🔍 **MRI Classification** — Detects and classifies brain tumors into four classes: **Glioma, Meningioma, Pituitary Tumor, or No Tumor**
- 🎯 **Grad-CAM Visualization** — Generates heatmaps highlighting the regions of the MRI scan that influenced the model's prediction
- 📄 **RAG-Powered Clinical Reports** — Uses a FAISS vector store and retrieval-augmented generation to produce context-aware, medically-informed report summaries
- 💬 **Interactive Chat Assistant** — Ask follow-up questions about your diagnosis, powered by the same RAG pipeline
- 📊 **Confidence Radar Chart** — Visualizes prediction confidence across all four classes using Chart.js
- ⚠️ **Confidence Threshold Warnings** — Flags low-confidence predictions so results aren't misinterpreted
- 🩺 **Severity Indicator** — Displays an at-a-glance severity/risk badge based on the prediction
- 📥 **PDF Report Download** — Export a complete diagnostic report (prediction, heatmap, confidence chart, clinical summary) as a PDF using jsPDF
- 💡 **Quick-Question Chips** — One-click common questions to speed up interaction with the chat assistant

---

## 🎥 Demo

*(Add screenshots or a GIF of the app here — result page, Grad-CAM heatmap, chat assistant, and PDF export)*

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Flask (Python) |
| **Deep Learning** | TensorFlow / Keras (ResNet50, CNN) |
| **Explainability** | Grad-CAM |
| **RAG / Retrieval** | FAISS vector store |
| **Frontend** | HTML, CSS, JavaScript |
| **Charts** | Chart.js (confidence radar chart) |
| **PDF Generation** | jsPDF |
| **Dataset Source** | Kaggle |

---

## 📈 Model Performance

| Metric | Score |
|---|---|
| **Test Accuracy** | ~73.5% |
| **ROC-AUC** | ~90.8% |

The model was trained and evaluated using transfer learning on a ResNet50 backbone, fine-tuned for the 4-class brain tumor classification task.

---

## 🗂️ Dataset

- **Source:** [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/)
- **Size:** ~10,000 MRI images
- **Classes:**
  - Glioma
  - Meningioma
  - Pituitary Tumor
  - No Tumor

---

## 📁 Project Structure

```
neuroscan-ai/
├── app.py                     # Main Flask application
├── rag.py                     # RAG pipeline (retrieval + report/chat generation)
├── predict.py                 # Model inference and Grad-CAM logic
├── knowledge_base/            # Clinical reference text used for RAG
│   ├── glioma.txt
│   ├── meningioma.txt
│   ├── notumor.txt
│   └── pituitary.txt
├── vectorstore/
│   └── db_faiss/               # FAISS vector index for RAG retrieval
├── static/                     # CSS, JS, generated images
├── templates/
│   └── result.html             # Main results UI (Grad-CAM, chart, chat, PDF export)
├── test_images/                # Sample MRI images for testing
├── requirements.txt
└── README.md
```

> **Note:** The trained model file (`.h5`) is excluded from this repository due to GitHub's 100MB file size limit. See [Installation](#-installation) below for how to obtain it.

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/dianathomas186-cloud/neuroscan-ai.git
   cd neuroscan-ai
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add the trained model**
   Since the model file exceeds GitHub's size limit, download it separately and place it in the project root:
   👉 *(Add your Google Drive / Kaggle model download link here)*

5. **Run the app**
   ```bash
   python app.py
   ```

6. Open your browser and go to:
   ```
   http://127.0.0.1:5000
   ```

---

## 🚀 Usage

1. Upload a brain MRI scan (JPEG/PNG) through the web interface
2. The model classifies the scan into one of four categories
3. View the **Grad-CAM heatmap** overlaid on the original scan to see which regions influenced the prediction
4. Review the **confidence radar chart** and **severity indicator**
5. Read the **AI-generated clinical report summary**
6. Ask the **chat assistant** follow-up questions, or use the quick-question chips
7. Download the full report as a **PDF**

---

## 🔬 How It Works

1. **Preprocessing** — The uploaded MRI image is resized and normalized to match the model's input requirements
2. **Classification** — A ResNet50-based CNN predicts the tumor class and outputs confidence scores for all four classes
3. **Explainability** — Grad-CAM computes gradients on the final convolutional layer to generate a heatmap showing which regions most influenced the prediction
4. **RAG Report Generation** — The predicted class and confidence scores are used to query a FAISS vector store containing curated clinical reference text; the retrieved context is used to generate a relevant, readable report
5. **Chat Assistant** — Follow-up questions are answered using the same RAG pipeline, grounding responses in the retrieved clinical knowledge base
6. **PDF Export** — jsPDF compiles the prediction, heatmap, chart, and report summary into a downloadable PDF

---

## 🔮 Future Improvements

- Improve model accuracy with additional data augmentation and hyperparameter tuning
- Add support for DICOM format MRI scans
- Deploy to a cloud platform (Render, AWS, or Hugging Face Spaces) for public access
- Add user authentication and scan history
- Expand the knowledge base with more detailed clinical references

---

## 🙏 Acknowledgements

- [Brain Tumor MRI Dataset — Kaggle](https://www.kaggle.com/)
- Built as a Deep Learning summative assessment project (Entri Elevate)
