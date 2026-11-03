"""
Comment Toxicity Detection - Streamlit App
==========================================
Interactive web app that loads the trained Keras model and lets the user:
    1. Type a single comment and see toxicity probabilities for 6 labels
    2. Upload a CSV of comments and get bulk predictions (downloadable)
    3. View the model's evaluation metrics from training

Run from terminal (after `python toxicity.py` has been executed):
    streamlit run app.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Anchor all paths to this file's folder so the app works no matter
# which directory `streamlit run` is launched from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from toxicity import clean_text, LABEL_COLS  # reuse the same cleaning logic

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "toxicity_model.keras")
TOK_PATH = os.path.join(ARTIFACTS_DIR, "tokenizer.json")
CFG_PATH = os.path.join(ARTIFACTS_DIR, "config.json")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.csv")

st.set_page_config(page_title="Comment Toxicity Detector",
                   page_icon="⚠️", layout="wide")


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(TOK_PATH, encoding="utf-8") as f:
        tokenizer = tokenizer_from_json(f.read())
    with open(CFG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    return model, tokenizer, cfg


def predict_batch(texts, model, tokenizer, max_len):
    cleaned = [clean_text(t) for t in texts]
    seqs = pad_sequences(tokenizer.texts_to_sequences(cleaned),
                         maxlen=max_len, padding="post", truncating="post")
    return model.predict(seqs, verbose=0)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("⚠️ Comment Toxicity Detector")
st.caption("Deep-learning model (BiLSTM) that flags toxic, obscene, "
           "threatening, insulting and identity-hate comments.")

model, tokenizer, cfg = load_artifacts()
if model is None:
    st.error(
        "No trained model found. Please run **`python toxicity.py`** first "
        "to train the model and create the `artifacts/` folder."
    )
    st.stop()

MAX_LEN = cfg["max_len"]
threshold = st.sidebar.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05,
                              help="Probability above this counts as positive.")
st.sidebar.divider()
st.sidebar.markdown("**Labels predicted:**")
for lbl in LABEL_COLS:
    st.sidebar.markdown(f"- {lbl}")

tab1, tab2, tab3 = st.tabs(["🔍 Single comment",
                            "📂 Bulk CSV",
                            "📊 Model metrics"])

# ---------------------------------------------------------------------------
# Tab 1 - single comment
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Type a comment")
    text = st.text_area("Comment", height=140,
                        placeholder="Paste a comment to check ...")
    if st.button("Predict", type="primary") and text.strip():
        probs = predict_batch([text], model, tokenizer, MAX_LEN)[0]
        result = {lbl: float(p) for lbl, p in zip(LABEL_COLS, probs)}
        is_toxic = any(p >= threshold for p in probs)

        if is_toxic:
            st.error("⚠️  This comment is flagged as **TOXIC**")
        else:
            st.success("✅  This comment looks **clean**")

        # Bar chart of probabilities
        df = (pd.DataFrame({"label": LABEL_COLS, "probability": probs})
              .sort_values("probability", ascending=True))
        fig, ax = plt.subplots(figsize=(8, 3.5))
        colors = ["#ef4444" if p >= threshold else "#3b82f6"
                  for p in df["probability"]]
        ax.barh(df["label"], df["probability"], color=colors)
        ax.axvline(threshold, color="black", linestyle="--", linewidth=0.8,
                   label=f"threshold = {threshold}")
        ax.set_xlim(0, 1)
        ax.set_xlabel("predicted probability")
        ax.legend()
        st.pyplot(fig)

        with st.expander("Raw probabilities"):
            st.json({k: round(v, 4) for k, v in result.items()})

# ---------------------------------------------------------------------------
# Tab 2 - bulk CSV
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Upload a CSV")
    st.caption("CSV must contain a column named **comment_text**. "
               "Output is the same CSV with one probability column per label.")
    upload = st.file_uploader("Choose CSV", type=["csv"])
    if upload:
        df = pd.read_csv(upload)
        if "comment_text" not in df.columns:
            st.error("CSV is missing a `comment_text` column.")
        else:
            st.info(f"{len(df):,} rows received - running predictions ...")
            probs = predict_batch(df["comment_text"].astype(str).tolist(),
                                  model, tokenizer, MAX_LEN)
            for i, lbl in enumerate(LABEL_COLS):
                df[f"prob_{lbl}"] = probs[:, i].round(4)
            df["is_toxic"] = (probs >= threshold).any(axis=1)
            st.success("Done.")
            st.dataframe(df.head(50), use_container_width=True)
            st.download_button(
                "Download predictions CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="toxicity_predictions.csv",
                mime="text/csv",
            )

# ---------------------------------------------------------------------------
# Tab 3 - model metrics
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Validation-set metrics")
    if os.path.exists(METRICS_PATH):
        m = pd.read_csv(METRICS_PATH)
        st.dataframe(m, use_container_width=True)
        st.metric("Macro ROC-AUC", f"{m['roc_auc'].mean():.4f}")
        fig, ax = plt.subplots(figsize=(8, 4))
        m_sorted = m.sort_values("roc_auc")
        ax.barh(m_sorted["label"], m_sorted["roc_auc"], color="#10b981")
        ax.set_xlim(0, 1)
        ax.set_xlabel("ROC-AUC")
        ax.set_title("Per-label ROC-AUC")
        st.pyplot(fig)
    else:
        st.info("metrics.csv not found - re-run `python toxicity.py`.")

st.divider()
st.caption("Built with TensorFlow / Keras Bidirectional LSTM and Streamlit.")
