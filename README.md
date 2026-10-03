# Next Word Prediction using LSTM

Deep learning NLP project that predicts the next word in a sentence, trained on the full text of *Pride and Prejudice* (Project Gutenberg). Built as part of the Innomatics Research Labs capstone series.

## Overview

Given a sequence of words, the model predicts the most probable next word - the same idea behind predictive keyboards and typing assistants. An LSTM network is trained on n-gram sequences built from the novel's text.

## Results

| Metric | Training | Validation |
|---|---|---|
| Accuracy | 15.75% | 14.23% |
| Top-5 accuracy | 34.52% | 32.73% |
| Loss | 4.72 | 5.36 |
| Perplexity | - | 211.65 |

Sample generated text (seed → generated):

```
it is a truth → it is a truth of you who will not herself to another of qualities for i i believe him
mr bennet said → mr bennet said elizabeth did she been a month s uncle and is in first four miles of
```

Word choice matches the novel's vocabulary and style; short stretches are grammatical, but long-range coherence is limited by the small, single-book training corpus.

## Tech Stack

- Python, TensorFlow / Keras
- NumPy, scikit-learn, Matplotlib
- Google Colab (T4 GPU)

## Pipeline

1. **Data collection** - downloaded *Pride and Prejudice* directly from Project Gutenberg
2. **Preprocessing** - sentence splitting, cleaning, lowercasing, tokenization, n-gram sequence generation, padding
3. **Model** - `Embedding(100) → LSTM(256) → Dropout(0.3) → Dense(softmax)`
4. **Training** - Adam optimizer, sparse categorical cross-entropy, early stopping + LR scheduling
5. **Evaluation** - accuracy, top-5 accuracy, perplexity, qualitative text generation

## Repo Structure

```
├── next_word_prediction.py     # Full training script (Colab-ready, cell-by-cell)
├── Next_Word_Prediction_Documentation.docx   # Full project report
├── next_word_model.keras       # Trained model
├── tokenizer.pkl                # Fitted tokenizer
├── config.pkl                   # Model config (sequence length)
└── README.md
```

## Usage

Open `next_word_prediction.py` in Google Colab (GPU runtime recommended) and run each cell in order. The script downloads the dataset, trains the model, plots accuracy/loss, and saves the trained model + tokenizer.

To generate predictions after training:

```python
predict_next_words("elizabeth felt that she")
generate_text("mr darcy was", n_words=15)
```

## Limitations & Future Work

- Trained on a single novel - a larger/more modern corpus would improve fluency and generalization
- Vocabulary and style are specific to 19th-century English
- Could be extended with bidirectional LSTM/GRU layers, pre-trained embeddings (GloVe/Word2Vec), or a Transformer-based approach

## Author

**Sumit Mangalwedhekar** - [GitHub](https://github.com/sumit2308) · [Portfolio](https://sumit2308.github.io/portfolio)

*Built as part of the Innomatics Research Labs NLP capstone project.*
