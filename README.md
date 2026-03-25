# Visual Field Predictor from Fundus Images

> **Status:** Ongoing Research

## 👨‍💻 Project Overview

This ongoing research project aims to predict visual field maps from standard fundus images. Our approach uses a novel hybrid deep learning architecture, **CNNMambaHybrid**, leveraging the local feature extraction of CNNs and the global sequence modeling of the Mamba architecture to capture the relationship between optic disc structure and functional vision loss. This could aid in the early detection and monitoring of ocular diseases like glaucoma.

## ✨ Key Features

- End-to-end deep learning framework for image-to-map prediction.
- Hybrid **CNNMamba** model tailored for medical imaging and sequence data.
- Multiple evaluation metrics (WingLoss, MAE, RMSE, R2) with automatic PDF report generation.

## 🛠️ Usage

### Prerequisites

```bash
pip install -r requirements.txt

