import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import os

st.set_page_config(page_title="Aircraft Delay Classifier", layout="centered")
st.title("✈️ Aircraft Delay Log Classifier")
st.markdown("Predicts the delay category, operational phase, and whether the delay was predictable.")

# Load models
@st.cache_resource
def load_models():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model"))
    label_model = AutoModelForSequenceClassification.from_pretrained(os.path.join(model_path, "delay_classifier"))
    phase_model = AutoModelForSequenceClassification.from_pretrained(os.path.join(model_path, "delay_phase_classifier"))
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return label_model.eval(), phase_model.eval(), tokenizer

label_model, phase_model, tokenizer = load_models()

label_map = {
    0: "TECHNICAL_FAILURE",
    1: "CREW_DELAY",
    2: "WEATHER",
    3: "ATC_RESTRICTION",
    4: "LOGISTICS_ISSUE",
    5: "SECURITY"
}

phase_map = {
    0: "pre-departure",
    1: "boarding",
    2: "pushback",
    3: "final check",
    4: "engine start",
    5: "taxi-out"
}

def predict(text, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).squeeze().tolist()
        pred_idx = torch.argmax(logits, dim=1).item()
    return pred_idx, probs


def is_predictable(label):
    predictable = {"WEATHER", "CREW_DELAY", "LOGISTICS_ISSUE"}
    return "PREDICTABLE" if label in predictable else "UNPREDICTABLE"

text = st.text_area("✍️ Enter delay description", height=150, placeholder="e.g., Late pushback due to hydraulic pump issue discovered during final check.")

if st.button("Predict"):
    if text.strip():
        label_idx, label_probs = predict(text, label_model)
        phase_idx, _ = predict(text, phase_model)

        label = label_map[label_idx]
        phase = phase_map[phase_idx]
        predictability = is_predictable(label)

        st.markdown(f"**🧠 Category:** `{label}`")
        st.markdown(f"**🧭 Phase:** `{phase}`")
        st.markdown(f"**🔮 Predictability:** `{predictability}`")

        df_probs = pd.DataFrame({
            "Category": list(label_map.values()),
            "Confidence": [round(p, 3) for p in label_probs]
        }).sort_values("Confidence", ascending=False)
        st.bar_chart(df_probs.set_index("Category"))
    else:
        st.warning("Please enter a valid log description.")
