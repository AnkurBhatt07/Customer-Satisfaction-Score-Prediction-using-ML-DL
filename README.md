# 📊 Deep CSAT: eCommerce Customer Satisfaction Prediction

An end-to-end ML + DL project that predicts whether a support interaction ends with a **Satisfied** or **Unsatisfied** customer.

This repository includes:
- full preprocessing + feature engineering pipeline,
- model experimentation in notebooks,
- and two Flask app variants for real-time inference:
  - `app.py` → Random Forest (`rf6.joblib`)
  - `app1.py` → Deep Neural Network (`ann5_model.keras`)

---

## 🚀 Project Summary

The original CSAT target was a 1–5 scale. For business actionability, it was converted into a binary outcome:
- **1 = Satisfied**
- **0 = Unsatisfied**

The core goal is not only overall accuracy, but better detection quality for dissatisfied cases while preserving strong total performance.

---

## 🧠 Pipeline Highlights

- Missing value indicators (`*_MI`) for robust inference
- City/contact-volume binning + scaling
- Connected handling time bucketing + scaling
- Price transformation
- Text-derived sentiment + word-count features from customer remarks
- Time-gap features (`order → issue`, `issue → reply`)
- One-hot encoding for product/channel/category/supervisor/manager/shift
- Ordinal encoding + scaling for tenure
- Agent-level historical performance feature
- Saved preprocessing artifacts for production-consistent transformation

---

## 🧪 Final Chosen Models & Metrics

### 1) `rf6` (deployed in `app.py`)
Metrics from `model_training1.ipynb` (`evaluate_binary_classification` output):

**Train**
- Accuracy: **0.9142**
- Precision: **0.9572**
- Recall: **0.9417**
- F1: **0.9494**
- ROC AUC: **0.8473**

**Test**
- Accuracy: **0.8376**
- Precision: **0.9070**
- Recall: **0.9025**
- F1: **0.9048**
- ROC AUC: **0.6795**
- **Macro F1 (from classification report): 0.68**

**Test class-wise report**
- Class 0 (Dissatisfied): Precision **0.44**, Recall **0.46**, F1 **0.45**
- Class 1 (Satisfied): Precision **0.91**, Recall **0.90**, F1 **0.90**

---

### 2) `ann5` (deployed in `app1.py`)
Metrics from notebook reporting:

- Test Accuracy: **0.78** (reported)
- **Macro F1 (from classification report): 0.66**
- ROC AUC: **not reported for ann5**

**Test class-wise report**
- Class 0 (Dissatisfied): Precision **0.35**, Recall **0.61**, F1 **0.45**
- Class 1 (Satisfied): Precision **0.92**, Recall **0.81**, F1 **0.86**

---

### ✅ Selection Decision

Based on notebook conclusion and comparative metrics, **`rf6` was selected as the primary final model** because it delivered the strongest macro-F1 balance across both classes while maintaining high test accuracy and robust overall precision/recall.

---

## 📁 Key Structure

```
.
├── app.py
├── app1.py
├── templates/
│   ├── base.html
│   ├── index.html
│   └── result.html
├── notebooks/
│   ├── data_analysis1.ipynb
│   └── model_training1.ipynb
└── data_analysis1/artifacts/
	 ├── preprocess/
	 └── models/
```

---

## ▶️ Run the Apps

1. Install dependencies:
	```bash
	pip install -r requirements.txt
	```

2. Run Random Forest app:
	```bash
	python app.py
	```

3. Run ANN app:
	```bash
	python app1.py
	```

Both apps expose a form at `/` and produce prediction + confidence in `result.html`.

---

## 🛠️ Tech Stack

Python, Pandas, NumPy, Scikit-learn, TensorFlow/Keras, Flask, Joblib, TextBlob, Matplotlib, Seaborn
