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

### 1) `lr5` (deployed in `app.py`)
Metrics from `model_training1.ipynb` (`evaluate_binary_classification` output):

The lr5 Logistic Regression model was selected as the final model based on its balanced macro-level classification performance and strong minority class recall compared to other models evaluated.

Test Set Performance Metrics

Metric	| Score

Accuracy	| 0.7475

Precision (Macro) |	0.62

**Recall (Macro)** |	**0.70**

F1 Score (Macro) |	0.63

ROC AUC |	0.6984

Recall (Class 0) |	0.63
---




### ✅ Selection Decision

Based on notebook conclusion and comparative metrics, **`lr5` was selected as the primary final model** because it delivered the strongest recall for class 0 while maintaining high test accuracy.

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
