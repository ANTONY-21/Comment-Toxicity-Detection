# ⚠️ Comment Toxicity Detection

Deep-learning model that reads a comment and predicts **6 toxicity labels** in one shot:
`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`.

It's a **multi-label** problem — a single comment can be `toxic` AND `obscene` AND `insult` at the same time.

---

## 1. Problem statement
Online platforms can't manually moderate millions of comments. Goal: build a deep-learning model that flags toxic comments in real time so moderators can review, filter, or warn users.

## 2. Dataset
**Jigsaw Toxic Comment Classification Challenge** (Kaggle)
- `train.csv` — 159,571 rows: `id`, `comment_text`, 6 binary labels
- `test.csv` — 153,164 rows: `id`, `comment_text` (unlabelled Kaggle holdout)

Label balance is heavily skewed (`toxic` 9.6%, `threat` only 0.3%), which is why
evaluation uses per-label precision / recall / F1 / ROC-AUC instead of plain accuracy.

Download the CSVs from Kaggle and place them next to this README.
(They are gitignored — don't commit them.)

## 3. Approach
| Step | Tool / technique |
|---|---|
| Text cleaning | lowercase, strip URLs/HTML/digits/punctuation, keep apostrophes |
| Tokenization | Keras `Tokenizer` (top 20k words) |
| Sequence building | `pad_sequences` to fixed length 100 |
| Model | `Embedding → Bidirectional LSTM → GlobalMaxPool → Dense → Dropout → 6×Sigmoid` |
| Loss | `binary_crossentropy` (multi-label) |
| Optimizer | Adam |
| Training | 3 epochs, batch 256, EarlyStopping on val loss |
| Evaluation | per-label Accuracy, Precision, Recall, F1, ROC-AUC |

## 4. Project structure
```
Comment Toxicity Detection/
├── train.csv                     # download from Kaggle (gitignored)
├── test.csv                      # download from Kaggle (gitignored)
├── toxicity.py                   # full modular pipeline (local)
├── toxicity_colab.ipynb          # same pipeline, ready for Google Colab
├── app.py                        # Streamlit app (single + bulk prediction)
├── requirements.txt
├── README.md
├── INTERVIEW_GUIDE.md            # live-evaluation study guide
└── artifacts/                    # created after training
    ├── toxicity_model.keras
    ├── tokenizer.json
    ├── config.json
    ├── metrics.csv
    └── plots/
```

## 5. How to run

### Option A — Google Colab (recommended, zero setup)
1. Open https://colab.research.google.com and upload `toxicity_colab.ipynb`
2. Run the first cell → it'll ask you to upload `train.csv`
3. Run every cell in order (~10–15 min on CPU, 1–2 min on free T4 GPU)
4. Cell 11 launches a **Gradio demo** with a public URL — use it for your LinkedIn demo video
5. Cell 12 zips and downloads the trained artifacts

### Option B — Local
```bash
pip install -r requirements.txt
python toxicity.py              # trains + saves artifacts
streamlit run app.py            # launches the Streamlit dashboard
```

## 6. Streamlit app features
- 🔍 **Single comment** — paste a comment, see a verdict + per-label probability bar chart
- 📂 **Bulk CSV** — upload a CSV with a `comment_text` column, get back a CSV with 6 probability columns + `is_toxic` flag
- 📊 **Model metrics** — per-label accuracy, precision, recall, F1, ROC-AUC (from training)

## 7. Business use cases
| Use case | How the model helps |
|---|---|
| Social media platforms | Auto-flag toxic comments in real time for moderator review |
| Online forums / communities | Filter new posts before they appear |
| Content moderation services | Scale human reviewers with an AI first-pass |
| Brand safety | Keep sponsored comments sections clean |
| E-learning | Keep student discussion forums safe |

## 8. Tech tags
`Python` `TensorFlow` `Keras` `LSTM` `Bidirectional` `NLP` `Tokenizer` `Multi-label Classification` `Streamlit` `Gradio`
