# 🧠 NLP Sentiment Analysis — LSTM vs BERT

This project compares two very different approaches to sentiment classification on real Yelp reviews:

- 🧵 A **Bidirectional LSTM built from scratch**
- 🤖 A **Fine-tuned BERT transformer**

Both models classify reviews as **Negative, Neutral, or Positive** on the same dataset — making it a direct comparison.

The goal wasn't just to get high accuracy — it was to deeply understand the NLP pipeline, from raw text preprocessing to modern transformer fine-tuning, and to understand *why* one approach outperforms the other.

---

## 📁 Repository Structure

```
nlp-sentiment-analysis/
├── classifier_using_lstm.ipynb          # Bidirectional LSTM implementation
└── sentiment_analysis_using_bert.ipynb  # Fine-tuned BERT implementation
```

---

## 🎯 What This Project Demonstrates

This project solves the same problem using two fundamentally different philosophies:

| | LSTM | BERT |
|--|------|------|
| **Approach** | Built from scratch | Transfer learning |
| **Parameters** | ~1M | ~109M |
| **Training Time** | ~5 mins | ~45 mins (GPU) |
| **Accuracy** | 68% | **78%** |
| **Macro F1** | 0.69 | **0.78** |

It demonstrates both a strong understanding of NLP fundamentals through the LSTM, and practical knowledge of modern state-of-the-art methods through BERT.

---

## 📊 Dataset

- **Source:** [Yelp Open Dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset) via Kaggle
- **Size:** 17,031 reviews after balancing
- **Classes:** Negative (1–2 stars), Neutral (3 stars), Positive (4–5 stars)
- **Balance:** 5,677 samples per class

Balancing the dataset was a deliberate choice — the raw Yelp data is heavily skewed toward 5-star reviews, which would bias the model toward positive predictions.

---

# 🧵 Part 1 — Bidirectional LSTM (Built From Scratch)

This model was built step-by-step to demonstrate full control over the NLP pipeline.

### Text Preprocessing (NLTK)
- Lowercasing, punctuation and number removal
- Tokenization using `word_tokenize`
- Stop word removal using NLTK's English corpus

### Sequence Preparation (Keras)
- Keras `Tokenizer` with vocabulary size of 10,000
- Out-of-vocabulary tokens handled with `<OOV>` token
- All sequences padded/truncated to length 50 using `pad_sequences`

### Model Architecture
```
Embedding Layer        → (vocab_size=10000, dim=64)
Bidirectional LSTM     → (units=64, return_sequences=True)
Dropout                → (rate=0.3)
Bidirectional LSTM     → (units=32)
Dropout                → (rate=0.3)
Dense                  → (units=32, activation='relu')
Output Dense           → (units=3, activation='softmax')
```

### Training
- Optimizer: Adam | Loss: Sparse Categorical Crossentropy
- Early Stopping with patience=5

### Results

**Test Accuracy: 68%**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Negative | 0.77 | 0.69 | 0.73 |
| Neutral | 0.54 | 0.71 | 0.62 |
| Positive | 0.81 | 0.64 | 0.71 |
| **Macro Avg** | **0.71** | **0.68** | **0.69** |

### What the LSTM Revealed
The model performs well on clearly positive and negative reviews but struggles with ambiguity and subtle tone. Overfitting appears after epoch 2 — train accuracy reaches 97% while validation stays at 64%. This is a known limitation of LSTMs on mid-sized datasets and directly motivated moving to BERT.

---

# 🤖 Part 2 — Fine-Tuned BERT

After observing the LSTM's limitations — overfitting, poor handling of double negatives and ambiguous phrasing — I moved to a transformer-based approach.

Unlike LSTMs which process words sequentially, BERT processes the entire sequence simultaneously using bidirectional self-attention, capturing context from both directions at every layer. This is why it handles phrases like "not bad at all" correctly where the LSTM fails.

### Tokenization (BERT WordPiece)
- Subword tokenization handles unknown words natively
- Special tokens `[CLS]` and `[SEP]` added automatically
- Sequences padded/truncated to 128 tokens

### Model Architecture
```
BERT Base (bert-base-uncased)
  └── 12 Transformer Encoder Layers
  └── 768 Hidden Dimensions
  └── 12 Attention Heads
  └── 109M Parameters
Classification Head
  └── Dense → 3 output classes (Negative, Neutral, Positive)
```

### Training Configuration
- Optimizer: AdamW (lr=2e-5)
- Scheduler: Linear warmup with decay
- Epochs: 3 | Batch Size: 32
- Hardware: T4 GPU (Google Colab)

### Results

**Test Accuracy: 78%**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Negative | 0.82 | 0.80 | 0.81 |
| Neutral | 0.67 | 0.68 | 0.68 |
| Positive | 0.84 | 0.85 | 0.84 |
| **Macro Avg** | **0.78** | **0.78** | **0.78** |

### What Changed with BERT
Validation accuracy tracked training accuracy throughout — significantly less overfitting. Only 3 epochs were needed because BERT arrives pretrained on billions of words and only needs to adapt to this specific task. The result is a 10% accuracy improvement over the LSTM on the exact same dataset.

---

# 🆚 Direct Comparison

| Metric | LSTM | BERT | Improvement |
|--------|------|------|-------------|
| Accuracy | 68% | **78%** | +10% |
| Negative F1 | 0.73 | **0.81** | +0.08 |
| Neutral F1 | 0.62 | **0.68** | +0.06 |
| Positive F1 | 0.71 | **0.84** | +0.13 |
| Macro F1 | 0.69 | **0.78** | +0.09 |

### Predictions on Tricky Reviews

| Review | LSTM | BERT |
|--------|------|------|
| "Not bad at all, quite enjoyed it" | ❌ Negative | ✅ Neutral |
| "It works, I guess" | ❌ Negative | ✅ Neutral |
| "Could have been worse I suppose" | ✅ Negative | ✅ Negative |
| "This is the worst product ever!" | ✅ Negative | ✅ Negative |
| "Really happy, great quality!" | ✅ Positive | ✅ Positive |
| "It is decent, nothing special" | ✅ Neutral | ✅ Neutral |

BERT consistently handles ambiguity better — especially neutral and double negative phrasing.

---

# 🏆 Final Verdict

For production systems, **BERT is the clear winner** — better accuracy, less overfitting, stronger contextual understanding, and faster convergence in just 3 epochs.

That said, the LSTM implementation is not without value. Building it from scratch demonstrates a genuine understanding of text preprocessing, tokenization, sequence modeling, and evaluation — all of which BERT abstracts away. Knowing both is what separates someone who can use NLP tools from someone who understands them.

---

## 🔍 Key Takeaways

- **Balanced data matters** — the raw Yelp data was skewed toward 5 stars; balancing it was essential for reliable evaluation
- **Neutral sentiment is universally hard** — both models struggled most here, which is consistent with human disagreement on neutral text
- **Transfer learning is highly data-efficient** — BERT achieved better results in 3 epochs than the LSTM achieved in 30
- **Overfitting is a real constraint** — the LSTM training curves clearly show train/val divergence, which directly informed the architectural decision to switch to BERT
- **Architecture choice matters more than hyperparameter tuning** — the biggest accuracy jump came from changing the model, not tweaking parameters

---

## ⚠️ Limitations

- Both models struggle with sarcasm and irony
- Neutral class F1 remains below 0.70 in both models — inherently ambiguous by nature
- Performance on non-Yelp reviews (different domain) may vary
- Fine-tuning on 50k+ rows would likely push BERT above 85%

---

## 🔮 Future Work

- Fine-tune RoBERTa or DeBERTa for further gains
- Train on 50k+ rows for better generalization
- Deploy as a Streamlit app with live predictions
- Add model explainability using SHAP or attention visualization
- Build a REST API for inference

---

## 🛠 Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![NLTK](https://img.shields.io/badge/NLTK-3.x-green)
![Colab](https://img.shields.io/badge/Google%20Colab-GPU-orange)
![Kaggle](https://img.shields.io/badge/Dataset-Yelp%20via%20Kaggle-lightblue)
