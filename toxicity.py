"""
Comment Toxicity Detection - Capstone Project
=============================================
Multi-label deep-learning model that classifies an online comment into
6 (overlapping) categories:
    toxic, severe_toxic, obscene, threat, insult, identity_hate

Dataset: Jigsaw Toxic Comment Classification Challenge (Kaggle)
    train.csv  - id, comment_text + 6 binary labels
    test.csv   - id, comment_text (no labels - the original Kaggle holdout)

Pipeline (each step is a function so the code is modular and reusable):
    1. load_data            -> read CSVs
    2. explore_data         -> shape / null / label distribution
    3. clean_text           -> lowercase, strip URLs / HTML / digits / extra ws
    4. build_tokenizer      -> Keras Tokenizer (top-N words)
    5. vectorize            -> texts -> padded integer sequences
    6. build_lstm_model     -> Embedding -> BiLSTM -> Dense -> 6 sigmoid outputs
    7. train_model          -> compile + fit with EarlyStopping
    8. evaluate_model       -> per-label accuracy / precision / recall / F1 / ROC-AUC
    9. predict_text         -> single-comment inference helper (used by Streamlit)
   10. save_artifacts       -> store model + tokenizer + config for the Streamlit app

WHY the design choices:
    - Multi-label, not multi-class -> sigmoid outputs + binary_crossentropy loss
      (a comment can be both 'toxic' AND 'obscene' at the same time)
    - Bidirectional LSTM -> reads the comment forward AND backward,
      better at catching context like "not a bad guy" vs "a bad guy"
    - Embedding layer is trained from scratch (no pretrained vectors) to keep
      the project self-contained; can be swapped for GloVe later
    - We subsample to SAMPLE_SIZE rows so it trains in a few minutes on CPU
      (the full 160k rows would take considerably longer and is not necessary
      for a capstone demo)

Run from terminal:
    pip install -r requirements.txt
    python toxicity.py            # trains the model + saves artifacts
"""

import os
import re
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report,
)

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM, Dense, Dropout, GlobalMaxPooling1D,
)
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
ARTIFACTS_DIR = "artifacts"
PLOTS_DIR = os.path.join(ARTIFACTS_DIR, "plots")

LABEL_COLS = ["toxic", "severe_toxic", "obscene",
              "threat", "insult", "identity_hate"]

# Keep training fast on CPU. Full data has ~160k rows; 50k is plenty for a
# capstone demo and trains in a few minutes on a laptop.
SAMPLE_SIZE = 50_000
RANDOM_STATE = 42

# Text -> sequence settings
MAX_VOCAB = 20_000      # keep top-N most frequent tokens
MAX_LEN = 100           # truncate / pad each comment to 100 tokens
EMBED_DIM = 64          # embedding vector size

# Training settings (kept small for CPU)
BATCH_SIZE = 256
EPOCHS = 3
VAL_SPLIT = 0.1


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
def load_data(train_path: str = TRAIN_PATH,
              test_path: str = TEST_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read train and test CSVs into DataFrames."""
    print(f"[1] Loading {train_path} ...")
    train = pd.read_csv(train_path)
    print(f"    train shape = {train.shape}")
    test = None
    if os.path.exists(test_path):
        test = pd.read_csv(test_path)
        print(f"    test  shape = {test.shape} (no labels - Kaggle holdout)")
    return train, test


# ---------------------------------------------------------------------------
# 2. Explore
# ---------------------------------------------------------------------------
def explore_data(df: pd.DataFrame) -> None:
    """Basic EDA: nulls, label balance, comment length distribution."""
    print("\n[2] Exploratory Data Analysis")
    print(f"    columns  : {list(df.columns)}")
    print(f"    nulls    : {int(df.isna().sum().sum())}")
    print(f"    duplicates: {int(df.duplicated(subset=['comment_text']).sum())}")

    # Label distribution
    print("\n    Label distribution (positive class %):")
    label_pct = (df[LABEL_COLS].mean() * 100).round(2)
    for lbl, pct in label_pct.items():
        print(f"      {lbl:<14} {pct:.2f}%")

    # Comment length stats
    lens = df["comment_text"].str.split().str.len()
    print(f"\n    comment length (in words): "
          f"mean={lens.mean():.0f}  median={lens.median():.0f}  "
          f"95th_pct={lens.quantile(0.95):.0f}  max={lens.max()}")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    # Save label distribution plot
    fig, ax = plt.subplots(figsize=(8, 4))
    label_pct.sort_values().plot(kind="barh", color="#ef4444", ax=ax)
    ax.set_xlabel("% of comments with this label")
    ax.set_title("Label distribution (multi-label)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "01_label_distribution.png"), dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# 3. Clean text
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"http\S+|www\.\S+")
_HTML_RE = re.compile(r"<.*?>")
_NON_ALPHA_RE = re.compile(r"[^a-z\s']")
_MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Lowercase, strip URLs / HTML / numbers / punctuation.
    Keep apostrophes so contractions like don't / can't survive.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def clean_dataframe(df: pd.DataFrame, col: str = "comment_text") -> pd.DataFrame:
    print("\n[3] Cleaning text ...")
    df = df.copy()
    df[col] = df[col].astype(str).apply(clean_text)
    df = df[df[col].str.len() > 0].reset_index(drop=True)
    print(f"    after cleaning shape = {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 4. Tokenizer
# ---------------------------------------------------------------------------
def build_tokenizer(texts, num_words: int = MAX_VOCAB) -> Tokenizer:
    """Fit a Keras Tokenizer on the cleaned training texts."""
    print("\n[4] Building tokenizer ...")
    tok = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tok.fit_on_texts(texts)
    print(f"    vocab size (cap) = {num_words}")
    return tok


def vectorize(texts, tokenizer: Tokenizer, max_len: int = MAX_LEN) -> np.ndarray:
    """Convert list of texts -> 2D numpy array of padded int sequences."""
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")


# ---------------------------------------------------------------------------
# 5. Model
# ---------------------------------------------------------------------------
def build_lstm_model(vocab_size: int = MAX_VOCAB,
                     embed_dim: int = EMBED_DIM,
                     max_len: int = MAX_LEN,
                     n_labels: int = len(LABEL_COLS)) -> tf.keras.Model:
    """
    Embedding -> Bidirectional LSTM -> GlobalMaxPool -> Dense -> Dropout -> Sigmoid.

    - Embedding: turns each integer token into a dense vector
    - BiLSTM: reads the sequence forward and backward to capture context
    - GlobalMaxPool: picks the strongest signal across all time steps
    - Sigmoid output (one per label) for multi-label classification
    - binary_crossentropy loss because each label is an independent yes/no
    """
    print("\n[5] Building LSTM model ...")
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_len),
        Bidirectional(LSTM(64, return_sequences=True)),
        GlobalMaxPooling1D(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(n_labels, activation="sigmoid"),
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc", multi_label=True)],
    )
    model.summary()
    return model


# ---------------------------------------------------------------------------
# 6. Train
# ---------------------------------------------------------------------------
def train_model(model, X_train, y_train, X_val, y_val):
    print("\n[6] Training ...")
    early = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early],
        verbose=2,
    )
    # Plot training curves
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history.history["loss"], label="train")
    ax[0].plot(history.history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(history.history["auc"], label="train")
    ax[1].plot(history.history["val_auc"], label="val")
    ax[1].set_title("ROC-AUC"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "02_training_curves.png"), dpi=120)
    plt.close()
    return history


# ---------------------------------------------------------------------------
# 7. Evaluate
# ---------------------------------------------------------------------------
def evaluate_model(model, X_val, y_val, threshold: float = 0.5) -> pd.DataFrame:
    """
    Per-label accuracy / precision / recall / F1 / ROC-AUC.
    Threshold = 0.5 by default to convert sigmoid probabilities to 0/1.
    """
    print("\n[7] Evaluating ...")
    probs = model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
    preds = (probs >= threshold).astype(int)

    rows = []
    for i, lbl in enumerate(LABEL_COLS):
        rows.append({
            "label": lbl,
            "accuracy":  round(accuracy_score(y_val[:, i], preds[:, i]), 4),
            "precision": round(precision_score(y_val[:, i], preds[:, i],
                                               zero_division=0), 4),
            "recall":    round(recall_score(y_val[:, i], preds[:, i],
                                            zero_division=0), 4),
            "f1":        round(f1_score(y_val[:, i], preds[:, i],
                                        zero_division=0), 4),
            "roc_auc":   round(roc_auc_score(y_val[:, i], probs[:, i]), 4),
        })
    metrics = pd.DataFrame(rows)
    print(metrics.to_string(index=False))

    macro_auc = round(metrics["roc_auc"].mean(), 4)
    print(f"\n    macro ROC-AUC = {macro_auc}")
    return metrics


# ---------------------------------------------------------------------------
# 8. Save artifacts (model + tokenizer + config)
# ---------------------------------------------------------------------------
def save_artifacts(model, tokenizer, metrics: pd.DataFrame) -> None:
    print("\n[8] Saving artifacts ...")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    model_path = os.path.join(ARTIFACTS_DIR, "toxicity_model.keras")
    model.save(model_path)

    tok_path = os.path.join(ARTIFACTS_DIR, "tokenizer.json")
    with open(tok_path, "w", encoding="utf-8") as f:
        f.write(tokenizer.to_json())

    cfg_path = os.path.join(ARTIFACTS_DIR, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({"max_len": MAX_LEN, "labels": LABEL_COLS}, f, indent=2)

    metrics.to_csv(os.path.join(ARTIFACTS_DIR, "metrics.csv"), index=False)
    print(f"    -> {model_path}")
    print(f"    -> {tok_path}")
    print(f"    -> {cfg_path}")
    print(f"    -> {ARTIFACTS_DIR}/metrics.csv")


# ---------------------------------------------------------------------------
# 9. Single-comment inference (used by Streamlit)
# ---------------------------------------------------------------------------
def predict_text(text: str, model, tokenizer, max_len: int = MAX_LEN,
                 threshold: float = 0.5) -> dict:
    """Return a dict { label: probability } and a boolean 'is_toxic_overall'."""
    cleaned = clean_text(text)
    seq = pad_sequences(tokenizer.texts_to_sequences([cleaned]),
                        maxlen=max_len, padding="post", truncating="post")
    probs = model.predict(seq, verbose=0)[0]
    out = {lbl: float(round(p, 4)) for lbl, p in zip(LABEL_COLS, probs)}
    out["is_toxic_overall"] = bool(any(probs >= threshold))
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    train, _ = load_data()
    explore_data(train)
    train = clean_dataframe(train)

    # Subsample for CPU-friendly training, but keep label balance
    if SAMPLE_SIZE and len(train) > SAMPLE_SIZE:
        train = train.sample(SAMPLE_SIZE, random_state=RANDOM_STATE) \
                     .reset_index(drop=True)
        print(f"    using random sample of {SAMPLE_SIZE} rows for CPU training")

    X_text = train["comment_text"].tolist()
    y = train[LABEL_COLS].values.astype("float32")

    X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_text, y, test_size=VAL_SPLIT, random_state=RANDOM_STATE
    )

    tokenizer = build_tokenizer(X_train_text)
    X_train = vectorize(X_train_text, tokenizer)
    X_val = vectorize(X_val_text, tokenizer)
    print(f"    X_train shape = {X_train.shape},  X_val shape = {X_val.shape}")

    model = build_lstm_model()
    train_model(model, X_train, y_train, X_val, y_val)
    metrics = evaluate_model(model, X_val, y_val)
    save_artifacts(model, tokenizer, metrics)

    # Quick sanity check
    print("\n[9] Sanity-check predictions:")
    for sample in [
        "Thanks for the helpful explanation, I really appreciate it.",
        "You are a complete idiot and I hate you so much.",
        "I think we should discuss this in more detail tomorrow.",
    ]:
        print(f"    > {sample}")
        print(f"      {predict_text(sample, model, tokenizer)}")

    print("\nDONE - check the 'artifacts/' folder.")


if __name__ == "__main__":
    main()
