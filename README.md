# Multimodal Explainable AI for Early-Stage Skin Cancer Classification

An advanced deep learning and Explainable AI (XAI) research project focused on improving the early detection of melanoma using multimodal learning, combining dermoscopic skin images with clinical patient metadata for more accurate and trustworthy predictions.

## Team Members
* **Abdelrahman Mohamed** — 202301645
* **Ahmed Mohsen** — 202301138
* **Jana Ahmed** — 202301800
* **Malak Mohamed** — 202301650

## Project Overview
Melanoma is one of the most dangerous forms of skin cancer, but it becomes highly treatable when detected early. Traditional AI systems for skin lesion classification often operate as black-box models, providing predictions without clear explanations. In clinical environments, this lack of transparency reduces trust and limits practical adoption.

This project addresses that challenge by developing a **Multimodal Explainable AI framework** that:
* **Processes both:** Dermoscopic skin lesion images and clinical metadata (age, sex, anatomical site).
* **Performs:** Accurate melanoma classification.
* **Generates:** Interpretable visual and feature-based explanations.
* **Compares:** Multiple state-of-the-art deep learning architectures.

The project is based on the **ISIC-DICM-17K dataset**, containing 17,060 balanced melanoma and non-melanoma samples.

## Objectives
* Build high-performance multimodal skin cancer classification models.
* Compare Transformer-based and CNN-based architectures.
* Integrate Explainable AI (XAI) methods into every model.
* Improve transparency and clinical trust.
* Analyze how metadata affects classification performance.
* Produce medically interpretable visual explanations.

## Dataset: ISIC-DICM-17K
* **Total Samples:** 17,060
* **Balanced Classes:** 8,530 melanoma / 8,530 non-melanoma
* **Data Types:** Dermoscopic images (`.jpg`) and clinical metadata (`.csv`)
* **Metadata Features:** Age, Sex, Anatomical Site

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

## Research Motivation
Most previous systems rely only on images and lack explainability. Our project combines **multimodal learning** with **XAI** to create a more clinically reliable diagnostic assistant.

## Technologies Used
* **Languages & Frameworks:** Python, PyTorch, TensorFlow / Keras, OpenCV, Scikit-learn, NumPy, Pandas, Matplotlib.
* **Explainability Libraries:** SHAP, LIME, Captum.
* **Environment:** Kaggle, Jupyter Notebook, GPU Acceleration (CUDA).

## Expected Outcomes
* High-accuracy melanoma classification.
* Better multimodal feature understanding.
* Clinically interpretable predictions.
* Robust benchmarking between architectures.

## References
* Ahammed et al. (2025) — *Multimodal Skin Lesion Classification*
* Kassem et al. (2020) — *Transfer Learning for ISIC 2019*
* Nigar et al. (2022) — *LIME-based Explainable Classification*
* Raju et al. (2025) — *XAI-SkinCADx Framework*
* Aksoy et al. (2025) — *XRAI Web-Based Clinical System*

## Acknowledgments
Special thanks to the International Skin Imaging Collaboration (ISIC), research contributors in medical AI, and our course instructors.
