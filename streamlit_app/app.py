import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import os

st.set_page_config(page_title="Aircraft Delay Classifier", layout="centered")

st.title("✈️ Aircraft Delay Log Classifier")
st.markdown("Classifica automaticamente un log operativo aeroportuale in base alla descrizione scritta dall’operatore.")
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../model/delay_classifier"))

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return model.eval(), tokenizer

model, tokenizer = load_model()

label_map = {
    0: "TECHNICAL_FAILURE",
    1: "CREW_DELAY",
    2: "WEATHER",
    3: "ATC_RESTRICTION",
    4: "LOGISTICS_ISSUE",
    5: "SECURITY"
}

# Prediction function
def predict_label(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).squeeze().tolist()
        pred_idx = torch.argmax(outputs.logits, dim=1).item()
    return label_map[pred_idx], probs

# Text input
user_input = st.text_area("✍️ Inserisci descrizione ritardo", height=150, placeholder="Es: Late pushback due to hydraulic pump issue discovered during final check.")

if st.button("Classifica"):
    if user_input.strip():
        label, probs = predict_label(user_input)
        st.success(f"**🧠 Predizione:** `{label}`")
        st.subheader("📊 Confidence")
        df_probs = pd.DataFrame({
            "Classe": list(label_map.values()),
            "Probabilità": [round(p, 3) for p in probs]
        }).sort_values("Probabilità", ascending=False)
        st.bar_chart(df_probs.set_index("Classe"))
    else:
        st.warning("Inserisci una descrizione valida per procedere.")
