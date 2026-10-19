# Multimodal Explainable AI for Early-Stage Skin Cancer Classification

This research project focuses on improving the early detection of melanoma using **Multimodal Learning** and **Explainable AI (XAI)**. By combining dermoscopic skin images with clinical patient metadata (age, sex, anatomical site), the framework aims to provide more accurate and trustworthy diagnostic assistance compared to traditional "black-box" models.

## Team Members

* **Abdelrahman Mohamed** — 202301645


* **Ahmed Mohsen** — 202301138


* **Jana Ahmed** — 202301800


* **Malak Mohamed** — 202301650



---

## Project Overview

The project is based on the **ISIC-DICM-17K dataset**, containing 17,060 balanced samples (8,530 melanoma and 8,530 non-melanoma). It explores multiple state-of-the-art architectures, ranging from **Vision Transformers (ViT)** to **Advanced CNN Ensembles**, to compare performance and clinical transparency.

### Key Features

* **Multimodal Data:** Processes both `.jpg` dermoscopic images and `.csv` clinical metadata.


* **Advanced Architectures:** Includes Swin Transformers, EfficientNetV2, and Cross-Attention modules.


* **Explainability:** Integration of Grad-CAM, SHAP, LIME, and Integrated Gradients for interpretable results.

## Implemented Architectures
The project explores multiple deep learning approaches categorized into three major groups:

### 1. Multimodal Fusion Transformers
* Vision Transformer (ViT) + Early Fusion
* Swin Transformer + Late Fusion
* Hybrid ViT-MLP
* Cross-Attention Transformer
* *Focus:* Learning relationships between image regions and clinical data.

### 2. Advanced CNN and Ensemble Models
* EfficientNet-B0 Metadata-Integrated
* ResNet-101 Majority Voting Ensemble
* MobileNetV3 Real-Time Model
* MultiExCam Hybrid Architecture
* *Focus:* Stability, ensemble robustness, and deployment efficiency.

### 3. Feature Extraction and Interpretable Networks
* Optimized DenseNet-121
* Inception-ResNet-v2
* Custom Deep Learning Architecture
* XRAI-Optimized ResNet
* *Focus:* Fine-grained lesion features and texture extraction.

## Explainable AI (XAI)
To improve interpretability and clinical trust, the project integrates:
* **Visual Explanations:** Grad-CAM, Grad-CAM++, XRAI, Integrated Gradients.
* **Model Agnostic/Feature Importance:** SHAP, LIME, Partial Dependence Plots (PDP).
* **Purpose:** Visualizing important lesion regions, model attention areas, and influential metadata features.


---

## Directory Structure

As shown in **image_831061.png**, the repository is organized as follows:

* **`EDA & preprocessing/`**: Notebooks for metadata cleaning, image processing, and data splitting.


* **`Phase 2/`**: Initial model development and individual progress reports.


* **`Phase 3/`**: Advanced multimodal architectures and final experimental setups.


* **`Final data/`**: Placeholder for processed outputs and datasets.


* **Root Files**: Includes the final research paper (`Team15_Research_Paper_Final.pdf`) and presentation (`Team15 ppt.pdf`).



---

## Setup & Execution

### 1. Prerequisites

The project is developed using **Python** and optimized for **GPU Acceleration (CUDA)** within Kaggle or Jupyter environments.

* **Core Frameworks:** PyTorch, TensorFlow/Keras.


* **Libraries:** OpenCV, Scikit-learn, NumPy, Pandas, Matplotlib, SHAP, LIME, and Captum.



### 2. Data Preparation

Before training, execute the preprocessing pipeline in the `EDA & preprocessing` folder:

1. Run `eda_processing_metadata.ipynb`.


2. Run `processing_image.ipynb`.


3. Run `final-splitted-data.ipynb` to finalize the training sets.



### 3. Training & Evaluation

Navigate to the `Phase 3` folder and select a model from a team member's directory:

* **For Transformers:** Try `02_SwinV2_GraphAttention_Fusion.ipynb` or `swin-transformer-v2-tabtransformer-cross-a.ipynb`.


* **For CNN/Hybrid Models:** Try `efficientnetv2s-crossattention-final.ipynb` or `03_ConvNeXt_FTTransformer_BilinearFusion.ipynb`.



### 4. XAI Visualizations

Explainability modules are embedded at the end of the Phase 3 notebooks. These sections generate visual heatmaps and feature importance charts to explain the model's decision-making process.

---

## Acknowledgments

Special thanks to the **International Skin Imaging Collaboration (ISIC)** for the data and the open-source contributors in the medical AI community.
