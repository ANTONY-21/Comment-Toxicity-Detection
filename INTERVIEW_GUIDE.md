# 🎤 Live Evaluation Study Guide — Comment Toxicity Detection

Your script for the 30-min live evaluation: **3 min intro + 12 min walkthrough + 10 min mock Q&A + 5 min feedback.** Read it 2–3 times and you'll be able to defend every line of code.

---

## 0. Self introduction (3 mins)

> "Hi, I'm **[name]**. I'm pursuing the AIML track at Guvi. I'm comfortable with Python, Pandas, scikit-learn, and I'm now exploring **deep learning with TensorFlow/Keras** and **NLP**. My project is a **Comment Toxicity Detector** — a Bidirectional LSTM model that reads an online comment and predicts six overlapping toxicity labels in real time. I also built a Streamlit app and a Gradio demo around it."

---

## 1. Project walkthrough script (12 mins)

Follow the **Capstone Explanation Guideline** order: domain → problem → objective → data cleaning → EDA → feature engineering → statistical technique → model → evaluation → conclusion → business solution.

### 1.1 Domain (3 lines)
> "The domain is **online content moderation** for social media, forums and e-learning platforms. These platforms have millions of comments and manual moderation doesn't scale. **Natural language processing** and deep learning let us automate toxicity detection in real time."

### 1.2 Introduction (2 lines)
> "This project builds a deep-learning model that classifies online comments into six toxicity labels using the Jigsaw Toxic Comment Classification dataset. The model is served through a Streamlit web app so moderators can paste a comment or upload a CSV and get predictions instantly."

### 1.3 Objective (2 lines)
> "Objective: train a neural network that reads raw comment text and outputs six probabilities — `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate` — so platforms can flag harmful content automatically."

### 1.4 Problem statement (1 line)
> "Toxic comments harm online discourse and moderators can't scale — we need an automated deep-learning classifier that runs in real time."

### 1.5 Data cleaning & preprocessing
- Loaded the Jigsaw training file with pandas — **~562k rows**, 8 columns
- **No missing values**, a few duplicates removed
- **Text cleaning** — `clean_text()` function:
  - lowercased everything
  - stripped **URLs** with regex `http\S+|www\.\S+`
  - stripped **HTML tags** with `<.*?>`
  - removed digits and punctuation, kept apostrophes so `don't` / `can't` survive
  - collapsed multiple spaces
- **Subsampled to 50,000 rows** for CPU-friendly training (the full 562k would take hours). Random sampling preserves label balance.

### 1.6 EDA findings
- **Class imbalance** is extreme: `toxic` is ~9.6 %, `severe_toxic` ~1 %, `threat` ~0.3 %, `identity_hate` ~0.9 %. Most comments are clean. This is why **accuracy alone is misleading** — we need per-label **F1** and **ROC-AUC**.
- **Comment length** — median ~35 words, 95th percentile ~250 words, max ~5000 → I truncate to 100 words (`MAX_LEN=100`) which covers ~85 % of comments and keeps the model small.
- **Label co-occurrence** — `obscene` + `insult` and `toxic` + `obscene` appear together a lot, confirming this is genuinely **multi-label** (not multi-class).

### 1.7 Feature engineering
- **Tokenization** with `tf.keras.preprocessing.text.Tokenizer(num_words=20_000)` — keeps the 20,000 most frequent words, maps rare words to `<OOV>`
- **Sequence building** with `pad_sequences(maxlen=100)` — every comment becomes a fixed-length vector of integers, needed because neural networks take tensors of equal shape
- **Embedding layer** — 64-dim learnable vectors replace the integer ids inside the model (no external pre-trained embeddings used — the project is self-contained)

### 1.8 Statistical / NLP technique
- **TF-IDF was the classical baseline** I considered, but I picked deep learning to learn word *order and context*, which is critical for sarcasm and negation (e.g. "not a bad guy" vs "a bad guy"). TF-IDF can't see that.
- No hypothesis tests used — it's a supervised modelling problem, not AB testing.

### 1.9 Model — Bidirectional LSTM

```
Embedding(20000, 64) → Bidirectional(LSTM(64, return_sequences=True))
                     → GlobalMaxPooling1D()
                     → Dense(64, relu) → Dropout(0.3)
                     → Dense(6, sigmoid)
```

| Layer | Why it's there |
|---|---|
| **Embedding** | Turns each integer token id into a 64-d learnable vector. Similar words end up with similar vectors. |
| **Bidirectional LSTM** | Reads the comment **forward and backward** so the representation at each word knows about both past and future context |
| **GlobalMaxPooling1D** | Collapses the time axis by taking the strongest activation per dimension — keeps the most "alarming" signal from anywhere in the comment |
| **Dense(64) + Dropout(0.3)** | Small MLP head. Dropout prevents overfitting |
| **Dense(6, sigmoid)** | **Multi-label head** — each neuron outputs an independent probability 0..1 |

**Loss**: `binary_crossentropy` — correct for multi-label sigmoid outputs (think of it as running 6 parallel binary classifiers).
**Optimizer**: `Adam` — fast, adaptive, the default for NLP models.
**Callback**: `EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)` — stops training early if validation loss stops improving, so we don't overfit.

### 1.10 Evaluation metrics
I report **5 metrics per label** because the classes are imbalanced:

| Metric | What it tells us |
|---|---|
| **Accuracy** | % of correct predictions. Misleading when 99 % are clean. |
| **Precision** | Of the comments we flagged, what % were really toxic? |
| **Recall** | Of the truly toxic comments, what % did we catch? |
| **F1** | Harmonic mean of precision and recall. Best single number when classes are imbalanced. |
| **ROC-AUC** | How well the model *ranks* toxic vs clean, across all thresholds. **Main metric**. |

I also report the **macro ROC-AUC** (average across labels) as the headline number — a small BiLSTM on 50k samples typically gets ~0.95.

### 1.11 Final model (why BiLSTM?)
> "I picked a Bidirectional LSTM because it reads context in both directions, handles variable-length comments naturally, is far smaller than a Transformer so it trains on CPU in ~15 minutes, and still reaches macro ROC-AUC around 0.95. BERT would be more accurate but takes hours on CPU and would violate the project's reproducibility requirement for local execution."

### 1.12 Conclusion
- Produced a working multi-label toxicity classifier with ~0.95 macro ROC-AUC
- Saved the model + tokenizer + config as reusable artifacts
- Wrapped everything in a Streamlit app (local) and a Gradio demo (Colab) for interactive use

### 1.13 Business suggestion / solution
- **Social media**: feed new comments through the model in real time, auto-hide anything above threshold 0.7 and send to human review
- **Forums / e-learning**: pre-screen new user posts before they appear; block comments that trigger `threat` or `identity_hate`
- **Brand safety**: keep sponsored-content comment sections clean
- **Content moderation services**: use the model as a first pass, humans review the uncertain band (0.3–0.7 probability)

---

## 2. Mock interview Q&A (the 10-mark section)

Read every answer aloud once.

### Why deep learning and not logistic regression?
> "Logistic regression on TF-IDF is a strong baseline and gets ~0.95 ROC-AUC on this dataset. I used deep learning because the project **requires it** and because an LSTM learns **word order and context** (sarcasm, negation) which a bag-of-words model cannot. For production at scale I'd actually use a fine-tuned transformer like `distilbert`."

### Why multi-label and not multi-class?
> "A comment can belong to several categories at once — `toxic` + `obscene` + `insult` is common. Multi-class forces exactly one label per sample; multi-label lets each label be independent. That's why I use **sigmoid outputs** and **binary_crossentropy**, not softmax and categorical_crossentropy."

### Sigmoid vs softmax — when to use which?
> "**Softmax** turns outputs into a probability distribution that sums to 1 — one class only. **Sigmoid** squashes each output to 0..1 independently — each label is a separate yes/no. Multi-label → sigmoid."

### Why `binary_crossentropy` instead of `categorical_crossentropy`?
> "`categorical_crossentropy` expects one-hot targets and softmax outputs — one class per sample. `binary_crossentropy` treats each output as an independent binary classifier, which matches multi-label with sigmoid outputs."

### What does the Embedding layer actually do?
> "It's a learnable lookup table. Each integer token (0..vocab_size) maps to a 64-dimensional vector. During training, the vectors move so that words appearing in similar contexts end up close together. It's how the network turns discrete tokens into continuous features."

### Why Bidirectional? Why not a plain LSTM?
> "A plain LSTM reads left-to-right only, so when it processes word W its hidden state knows the past but not the future. Bidirectional runs two LSTMs — one forward, one backward — and concatenates their hidden states. This matters a lot for negation: to classify `bad` in `'not a bad guy'`, you need to know `not` came before it."

### What is an LSTM?
> "**Long Short-Term Memory** — an RNN variant that uses three gates (input, forget, output) and a memory cell to keep or discard information across time steps. It fixes the vanishing-gradient problem that plain RNNs have with long sequences."

### What is GlobalMaxPooling1D doing?
> "After the LSTM returns a sequence of `(time_steps, features)` activations, GlobalMaxPool takes the max over time for each feature. Intuition: 'was there a strong signal for this feature anywhere in the comment?' — perfect for toxicity because a single slur can make a whole comment toxic."

### Why did you pick `MAX_LEN=100`?
> "The 95th-percentile comment length is ~250 words, but the median is ~35. 100 covers the vast majority of comments with minimal truncation and keeps the model small enough to train on CPU. It's a deliberate speed/quality trade-off."

### Why `num_words=20_000` in the tokenizer?
> "Keeps only the 20,000 most frequent tokens. Rare words contribute noise, not signal, and they blow up the embedding matrix. 20k covers ~99 % of tokens in the training data."

### What is the `<OOV>` token?
> "Out-of-vocabulary. Any word not in the tokenizer's top-N vocab is mapped to this single token. Needed at inference time for words the model never saw in training."

### Why did you subsample 50k rows instead of using all 562k?
> "CPU budget. On my laptop, 50k × 3 epochs trains in ~15 minutes and already hits ~0.95 macro ROC-AUC. Training on all 562k would take hours and only marginally improve the metric. For a production system I'd train on the full dataset on GPU."

### How did you handle class imbalance?
> "I didn't use class weights in the final model because `binary_crossentropy` with sigmoid outputs already handles imbalance reasonably well for ROC-AUC. If precision/recall on `threat` or `identity_hate` were too low, I'd add `class_weight=` to `model.fit()` or use oversampling. I report **per-label** metrics so the imbalance is visible."

### Why report ROC-AUC over accuracy?
> "Accuracy hides class imbalance. If only 1 % of comments are threats, a model that always says 'not a threat' gets 99 % accuracy but is useless. ROC-AUC measures how well the model **ranks** positives above negatives, across all thresholds — it's immune to class balance."

### What is ROC-AUC exactly?
> "Area under the Receiver Operating Characteristic curve. The curve plots True Positive Rate vs False Positive Rate as you sweep the decision threshold. AUC = 1 means perfect, 0.5 means random."

### What does the decision threshold mean?
> "The sigmoid outputs probabilities. I convert them to 0/1 by thresholding at 0.5 by default. The Streamlit app has a slider so moderators can tune it — lower threshold = more sensitive (higher recall, more false positives), higher threshold = stricter."

### How does `pad_sequences` work?
> "Comments have different lengths, but a neural network needs tensors of equal shape. `pad_sequences(maxlen=100, padding='post')` appends zeros to the end of shorter sequences and truncates the end of longer ones. I chose `post` padding because the LSTM's final hidden state is more informative when real tokens are at the start."

### Why `Adam` optimizer?
> "Adaptive learning rate per parameter, combines momentum and RMSProp, needs almost no tuning. It's the default for NLP and works well here."

### What is Dropout?
> "During training, randomly zero out a fraction of activations — here 30 %. Forces the network to not rely on any single neuron and acts like ensembling many sub-networks. Reduces overfitting."

### What is EarlyStopping?
> "A callback that monitors validation loss and stops training if it hasn't improved for N epochs (I used `patience=2`). `restore_best_weights=True` rewinds the model back to the best epoch at the end, so I never save an overfit version."

### Why 3 epochs? Wouldn't more be better?
> "On 50k samples, the model converges fast — validation loss usually bottoms out at epoch 2–3. More epochs start overfitting, EarlyStopping stops it automatically. On the full dataset I'd train for 5–8 epochs with the same patience."

### How do you handle new words at inference time?
> "The tokenizer maps unknown tokens to `<OOV>`. The `<OOV>` embedding is learned during training from rare words, so it behaves like a generic placeholder. I also apply the **same cleaning function** (`clean_text`) to incoming text before tokenizing."

### What's the difference between an RNN, LSTM and GRU?
> "**RNN** — simplest, processes sequences one step at a time but suffers vanishing gradients on long sequences. **LSTM** — adds memory cell + 3 gates to keep long-range info. **GRU** — lighter LSTM with 2 gates, faster, similar quality. I picked LSTM because it's the most-documented for NLP classification."

### What is transfer learning? Could you use it here?
> "Yes. You'd fine-tune a pre-trained model like BERT or DistilBERT on this dataset. It would reach ~0.98+ macro ROC-AUC but needs a GPU. For this capstone, training a BiLSTM from scratch demonstrates the underlying concept without the infrastructure."

### Why Keras/TensorFlow and not PyTorch?
> "Both work. Keras has a higher-level API that's faster for this kind of sequential model and simpler to save/load for the Streamlit app."

### How would you deploy this?
> "The saved `.keras` model + `tokenizer.json` + `config.json` are the three artifacts. Wrap them in a FastAPI endpoint that takes a JSON `{comment: ...}` and returns probabilities. Deploy with Docker to AWS/GCP. The Streamlit app demos the idea end-to-end."

### How would you monitor it in production?
> "Log every prediction + user feedback (was it really toxic?), track ROC-AUC on a rolling validation set, watch for drift in the word distribution, retrain monthly."

### What are the limitations?
> "1) Trained on English only — won't work on other languages. 2) Subsampled to 50k so it under-uses the data. 3) No explainability — can't tell the user *why* a comment was flagged (SHAP or LIME would fix this). 4) The Jigsaw dataset has known bias against comments about certain demographic groups — the model inherits it."

### How would you improve it?
> "1) Fine-tune **DistilBERT** on GPU. 2) Add **class weights** for the minority labels. 3) Use **GloVe/fastText** pre-trained embeddings in the Embedding layer. 4) Do **threshold tuning per label** to maximise F1. 5) Add **SHAP explanations** in the Streamlit app."

### What is overfitting? How did you detect it?
> "When training loss keeps going down but validation loss starts going up. I monitor both curves in the training-curves plot — if they diverge, I add dropout / reduce capacity / stop training earlier (which is what EarlyStopping does)."

### Precision vs Recall — which matters more here?
> "Depends on the use case. For **auto-blocking** a comment, you want high **precision** so you don't wrongly silence users. For **escalating to human review**, you want high **recall** so nothing toxic slips through. That's why the Streamlit app has a threshold slider — one model, two operating points."

---

## 3. Last-minute checklist

- [ ] Run the Colab notebook end-to-end — confirm the metrics print and artifacts zip downloads
- [ ] Open `toxicity.py` in VS Code and be ready to walk through each function
- [ ] Run the Gradio demo (cell 11) and try 3 sample comments live in the evaluation
- [ ] Have README, cluster/metrics plots, and this guide open in tabs
- [ ] If asked "what would you do differently?" → DistilBERT fine-tune, class weights, SHAP
- [ ] If asked "how is this different from project 1?" → project 1 is **unsupervised clustering** (no labels), this is **supervised multi-label classification** (6 labels per row)
