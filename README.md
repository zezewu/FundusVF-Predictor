# Visual Field Predictor from Fundus Images

> **Status:** Ongoing Research

---

## 👨‍💻 Project Overview

This ongoing research project aims to predict **visual field (VF) maps** from standard **fundus images** using deep learning techniques. The goal is to model the relationship between optic disc structure and functional vision loss, which can help support the **early detection and monitoring of ocular diseases such as glaucoma**.

To achieve this, the project introduces a novel hybrid deep learning architecture called **CNNMambaHybrid**, which combines:

* The **local feature extraction capability** of Convolutional Neural Networks (CNNs)
* The **global sequence modeling strength** of the Mamba architecture

This hybrid framework enables end-to-end prediction from retinal images to visual field maps.

---

## ✨ Key Features

* End-to-end deep learning pipeline for **fundus image → visual field map prediction**
* Hybrid **CNN + Mamba architecture** designed for medical imaging tasks
* Multiple evaluation metrics integrated:

  * WingLoss
  * MAE (Mean Absolute Error)
  * RMSE (Root Mean Square Error)
  * R² (Coefficient of Determination)
* Automatic PDF report generation for experiment tracking and evaluation

---

## 📂 Project Structure

```
project_root/
│
├── code/
│   └── train_acg.py
│
├── data/
│   └── data_ACG.csv
│
├── requirements.txt
│
└── README.md
```

---

## 🛠️ Installation

Clone the repository and install the required dependencies:

```bash
git clone <your-repository-url>
cd <your-repository-name>
pip install -r requirements.txt
```

---

## 📊 Dataset Preparation

⚠️ **Important Notice**

Due to **medical data privacy regulations**, the raw fundus images and visual field maps used in this research cannot be publicly released.

To run this project locally, please prepare your dataset following the structure below.

### Step 1: Prepare CSV File

Place your dataset CSV file (for example:

```
data_ACG.csv
```

inside the directory:

```
data/
```

### Step 2: Configure Image Paths

Ensure that the image paths specified inside the CSV file correctly point to your **local storage locations** of:

* fundus images
* corresponding visual field maps

Example structure:

```
data/
├── data_ACG.csv

images/
├── fundus_001.png
├── fundus_002.png
```

---

## 🚀 Training the Model

After preparing the dataset, start training with:

```bash
python code/train_acg.py
```

Training outputs may include:

* predicted visual field maps
* evaluation metrics
* experiment logs
* automatically generated PDF reports

---

## 📈 Evaluation Metrics

The following metrics are used to evaluate prediction performance:

| Metric   | Description                                 |
| -------- | ------------------------------------------- |
| WingLoss | Robust loss for structured prediction tasks |
| MAE      | Mean Absolute Error                         |
| RMSE     | Root Mean Square Error                      |
| R²       | Regression goodness-of-fit score            |

These metrics help measure both **pixel-level accuracy** and **overall prediction reliability**.

---

## 🧠 Model Architecture

The proposed **CNNMambaHybrid** architecture integrates:

* CNN layers for extracting **local anatomical retinal features**
* Mamba sequence modeling blocks for capturing **global structural dependencies**

This hybrid strategy improves the ability to model the relationship between:

```
optic disc morphology → functional visual field sensitivity
```

which is critical for glaucoma assessment.

---

## 📷 Example Outputs (Coming Soon)

Example visualizations will be added after the evaluation phase:

| Fundus Image | Predicted VF Map | Ground Truth VF Map |
| ------------ | ---------------- | ------------------- |
| Coming Soon  | Coming Soon      | Coming Soon         |

These examples will demonstrate the qualitative prediction performance of the model.

---

## 🔬 Current Project Status

This project is currently under **active development and evaluation**.

Preliminary experimental results are promising. More detailed:

* quantitative benchmarks
* visualization comparisons
* ablation studies
* and reproducibility details

will be released once the evaluation phase is completed.

---

## 📜 License

This project is intended for **academic research purposes only**.

If you plan to reuse this work, please cite the repository appropriately after the official release.

---

## 🤝 Contributions

Contributions, suggestions, and discussions are welcome.

Feel free to open an issue or submit a pull request if you would like to collaborate on improving this project.

---

## 📬 Contact

For academic collaboration or questions regarding implementation details, please open an issue in this repository.

Additional contact information will be provided after the project release phase.
