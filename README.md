# ✈️ Aircraft Delay Log Intelligence

An end-to-end NLP pipeline for classifying operational aircraft delay logs. This project uses fine-tuned transformer models to predict the **delay cause**, **operational phase**, and **predictability**, enabling structured understanding of free-text reports from airline operations.

---

## 🔍 Example

**Input**:  
`Late pushback due to hydraulic pump issue discovered during final check.`

**Output**:
- 🧠 Category: `TECHNICAL_FAILURE`
- 🧭 Phase: `final check`
- 🔮 Predictability: `UNPREDICTABLE`

---

## 🎯 Use Case

Airlines generate thousands of unstructured delay reports from pilots, crew, and ground staff. This tool transforms those into structured, actionable insights — useful for analytics, automation, and training.

---

## 🧠 Tasks

| Task | Description |
|------|-------------|
| **Text Classification** | Predicts 6 delay types (e.g., TECHNICAL_FAILURE, WEATHER) |
| **Phase Prediction** | Predicts operational phase (e.g., boarding, pushback) |
| **Predictability Estimation** | Heuristic logic: certain delays are predictable |
| **Interpretability** | Attention-based token attribution |
| **Streamlit App** | Interactive interface for real-time predictions |

---

## 🧾 Dataset

A synthetic dataset of 300+ logs was created using domain-aware GPT prompting, with the following fields:
- `log_text` (description)
- `label` (category)
- `phase` (operational step)
- `severity` (low, medium, high)

📁 File: `data/synthetic_logs.csv`

---

## 🏗️ Models

- `bert-base-uncased` fine-tuned using Hugging Face Transformers
- Separate models for:
  - Delay category: `delay_classifier/`
  - Operational phase: `delay_phase_classifier/`

---

## 🖥️ Streamlit Demo

Launch locally:

```bash
cd streamlit_app
streamlit run app.py
```

---

## 📊 Evaluation

## 📊 Evaluation Metrics

| Task                        | Metric       | Score |
|-----------------------------|--------------|-------|
| **Delay Category Classification** | Accuracy      | 1.00  |
|                              | Precision (weighted) | 1.00  |
|                              | Recall (weighted)    | 1.00  |
|                              | F1 Score (weighted)  | 1.00  |
| **Phase Prediction**         | Accuracy      | 1.00  |
|                              | Precision (weighted) | 1.00  |
|                              | Recall (weighted)    | 1.00  |
|                              | F1 Score (weighted)  | 1.00  |




## 🧩 Interpreting the Perfect Score


Although the model achieves F1 = 1.00 on delay category and phase classification, this is expected due to the structured nature of operational delay logs. In real airline operations:
Reports are written in concise, consistent templates
Specific keywords (e.g., “crew rest”, “catering truck”) directly imply certain delay types
GPT-generated synthetic data reflects this domain consistency
Thus, the task is highly learnable — and the model's performance mirrors how easily human dispatchers or operations analysts could classify these entries
---

## 🧪 Sample Logs to Test

```
Flight held at gate due to ATC departure slot congestion.
→ Category: ATC_RESTRICTION | Phase: pre-departure | UNPREDICTABLE

Fueling truck arrived late, delaying pushback.
→ LOGISTICS_ISSUE | pre-departure | PREDICTABLE

Crew rest period exceeded; replacement crew dispatched.
→ CREW_DELAY | pre-departure | PREDICTABLE
```

---

## 📂 Project Structure

```
aircraft-delay-nlp/
├── data/                 # CSV + label maps
├── model/                # Saved models
├── notebooks/            # Training + inference
├── streamlit_app/        # Streamlit interface
├── requirements.txt
└── README.md
```

---

## 🔗 GitHub

[github.com/Simo-dg/aircraft-delay-nlp](https://github.com/Simo-dg/aircraft-delay-nlp)

