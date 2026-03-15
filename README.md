<div align="center">

# Kaafat Job Recommendation System

### Smart Saudi Job-Matching Chatbot Built on the Jadarat Dataset

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![FAISS](https://img.shields.io/badge/FAISS-Similarity_Search-blueviolet.svg)](https://github.com/facebookresearch/faiss)
[![Gradio](https://img.shields.io/badge/Gradio-Interactive_Demo-orange.svg)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A from-scratch GPT-like Transformer language model and a TF-IDF + FAISS retrieval pipeline for generating and matching Arabic job recommendations, deployed via Gradio.

</div>

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Approach 1 — Generative Model](#approach-1--generative-model-generative-aiipynb)
- [Approach 2 — Retrieval Model](#approach-2--retrieval-model-retrivalipynb)
- [Arabic Text Preprocessing](#arabic-text-preprocessing)
- [Training Details](#training-details)
- [Evaluation Results](#evaluation-results)
- [Deployment](#deployment)
- [Getting Started](#getting-started)
- [Technical Limitations](#technical-limitations)

---

## Overview

This project tackles **Arabic job recommendation** for the Saudi labor market using two complementary approaches:

| Approach | Method | Notebook |
|----------|--------|----------|
| **Generative** | GPT-like Transformer (built from scratch) that *generates* new job descriptions | `Generative-Ai.ipynb` |
| **Retrieval** | TF-IDF vectorization + FAISS indexing that *retrieves* the most relevant existing jobs | `Retrival.ipynb` |

Both pipelines are trained on the **Jadarat dataset** (7,041 Saudi job postings) and include full Arabic text normalization and interactive Gradio deployment.

---

## Project Structure

```
Job-Matching/
├── Generative-Ai.ipynb      # Generative Transformer model (full pipeline)
├── Retrival.ipynb            # Retrieval model (TF-IDF + FAISS)
├── Jadarat_data.csv          # Dataset (7,041 Arabic job postings)
├── Generative-Demo/          # Screenshots of the generative demo
├── Retrival-Demo.png         # Screenshot of the retrieval demo
└── README.md
```

---

## Dataset

**Source:** Jadarat — Saudi Arabia's national employment platform

| Property | Detail |
|----------|--------|
| Total samples | **7,041** job postings |
| Language | Arabic |
| Key columns | Job title, City, Job description, Organization |
| Split (Generative) | 80% train (5,702) / 10% val (634) / 10% test (705) |
| Split (Retrieval) | Full dataset indexed (no train/test split needed) |

---

## Approach 1 — Generative Model (`Generative-Ai.ipynb`)

A **decoder-only Transformer** language model built entirely from scratch in PyTorch (no pretrained weights). Given a job title prompt, it *generates* a new job description.

### Pipeline Phases

| Phase | Description |
|-------|-------------|
| Phase 1 | Environment setup and imports |
| Phase 2 | Data loading and Arabic text normalization |
| Phase 3 | BPE tokenization and dataset preparation |
| Phase 4 | Transformer model architecture (from scratch) |
| Phase 5 | Demo training (quick sanity check) |
| Phase 6 | Extended full training with early stopping |
| Phase 7 | Text generation with temperature sampling |
| Phase 8 | Comprehensive testing and evaluation |
| Phase 9 | Deployment via Gradio chat interface |

### Architecture

```
Input --> Token Embedding + Positional Embedding --> 4x Transformer Blocks --> LayerNorm --> Linear --> Softmax
```

| Component | Detail |
|-----------|--------|
| Type | Decoder-only GPT (causal masked self-attention) |
| Embedding dim (`d_model`) | 256 |
| Attention heads | 8 |
| Transformer layers | 4 |
| Feed-forward dim | 512 |
| Max sequence length | 128 tokens |
| Vocab size | 5,000 (BPE) |
| Dropout | 0.1 |
| Positional encoding | Learnable |
| Total parameters | **4.7M** |

Each Transformer block contains:
- **Multi-Head Self-Attention** with causal masking
- **Residual connections** around each sub-layer
- **Layer Normalization** (post-norm)
- **Position-wise Feed-Forward Network** (Linear -> ReLU -> Dropout -> Linear)

### Tokenization

A **Byte-Pair Encoding (BPE)** tokenizer trained from scratch on the Arabic corpus using HuggingFace `tokenizers`:

- **Vocab size:** 5,000 — chosen to balance Arabic morphological richness against the small corpus size (7K samples). A larger vocabulary would cause data sparsity; a smaller one would over-segment words.
- **Special tokens:** `[PAD]`, `[UNK]`, `[BOS]`, `[EOS]`
- **UNK rate:** 0.00% (all tokens covered)

---

## Approach 2 — Retrieval Model (`Retrival.ipynb`)

A **semantic search pipeline** that finds the most relevant existing job from the dataset for a given user query and returns a formatted recommendation.

### Pipeline

```
User Query --> Arabic Normalization --> TF-IDF Vectorization --> FAISS Inner-Product Search --> Top-1 Match --> Formatted Recommendation
```

### Components

| Component | Detail |
|-----------|--------|
| **Vectorizer** | `TfidfVectorizer` with `max_features=5000` and bigrams (`ngram_range=(1, 2)`) |
| **Normalization** | L2-normalized dense vectors (`sklearn.preprocessing.normalize`) |
| **Index** | `faiss.IndexFlatIP` — exact inner-product (cosine) similarity search |
| **Retrieval** | Top-1 nearest neighbor for each query |
| **Libraries** | scikit-learn, FAISS (CPU), pandas, Gradio |

### How It Works

1. **Text preparation** — Each job posting is concatenated as: `job_title + description + city (repeated 3x for boosting) + job_title`. The city is repeated to give geographic preference more weight in similarity matching.
2. **TF-IDF encoding** — The corpus is vectorized into 5,000-dimensional sparse vectors capturing unigram and bigram importance.
3. **FAISS indexing** — All 7,041 L2-normalized vectors are added to a flat inner-product index for exact cosine search.
4. **Query matching** — A user's Arabic query is normalized, vectorized with the same TF-IDF model, and searched against the index.
5. **Recommendation** — The top match is formatted into a structured Arabic response with job title, company, city, description, and a personalized recommendation.

---

## Arabic Text Preprocessing

Both notebooks share the same Arabic normalization pipeline:

| Step | What it Does | Why |
|------|-------------|-----|
| Diacritics removal (Tashkeel) | Strips fatha, damma, kasra, shadda, sukun, tatweel | Prevents vocabulary explosion from pronunciation marks |
| Hamza normalization | Alef variants --> bare Alef | Unifies Alef variants that carry the same meaning |
| Yaa normalization | Alef Maqsura --> Yaa, Waw with Hamza --> Waw | Reduces orthographic variation |
| Symbol removal | Strips non-alphanumeric characters | Keeps only meaningful text |
| Whitespace normalization | Collapses multiple spaces | Cleans formatting artifacts |

---

## Training Details

> Training applies to the **Generative Model** only. The Retrieval Model uses unsupervised TF-IDF (no training loop).

### Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Learning rate | 3e-4 | Standard for small Transformers; paired with cosine decay |
| Batch size | 32 | Fits in Colab GPU memory while providing stable gradients |
| Epochs | 10 | Loss was still improving at epoch 10 |
| Optimizer | AdamW | Weight-decoupled Adam; better generalization |
| Weight decay | 1e-2 | Regularization to reduce overfitting |
| LR scheduler | CosineAnnealingLR | Smooth decay avoids sudden LR drops |
| Gradient clipping | max_norm = 1.0 | Prevents gradient explosion in attention layers |
| Early stopping | Patience = 3 | Monitors val loss; stops if no improvement for 3 epochs |

### Training Progression

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 1 | 5.034 | 3.236 |
| 5 | 1.499 | 1.521 |
| 10 | 1.122 | 1.390 |

The train-validation gap at epoch 10 is **0.27**, indicating mild overfitting controlled by dropout and weight decay.

---

## Evaluation Results

> Evaluation metrics apply to the **Generative Model**. The Retrieval Model is evaluated qualitatively through its Gradio demo.

### Quantitative Metrics

| Metric | Score |
|--------|-------|
| Test Perplexity | **4.33** |
| Top-1 Accuracy | **75.65%** |
| Top-5 Accuracy | **82.86%** |
| Top-10 Accuracy | **85.09%** |
| Self-BLEU-4 (diversity) | **0.231** |
| Avg 3-gram Repetition | **0.005** |

### LLM-as-Judge

An external LLM evaluator scores generated outputs on three dimensions (1-10 scale):

| Criterion | Avg Score |
|-----------|-----------|
| Coherence | 8 / 10 |
| Relevance | 9 / 10 |
| Grammar | 8 / 10 |

### Failure Modes Report

| Issue | Description | Mitigation |
|-------|-------------|------------|
| **Phrase repetition** | High-frequency phrases (e.g., document preparation) leak across job types | Repetition penalty, nucleus sampling |
| **Temperature sensitivity** | temp=1.0 produces incoherent text; temp=0.5 is more stable but repetitive | Use temp 0.6-0.8 for best balance |
| **Domain narrowness** | Trained only on Jadarat Saudi postings; cannot generalize to other domains | Expand training data |

---

## Deployment

Both models are deployed as interactive Arabic chatbots using **Gradio**:

### Generative Chatbot (`Generative-Ai.ipynb`)
- Uses `gr.ChatInterface` with OpenAI-style message format
- User types a job field in Arabic and receives a **generated** job description
- Supports temperature control for output diversity

### Retrieval Chatbot (`Retrival.ipynb`)
- Uses `gr.ChatInterface` with a public share link
- User types their qualifications, experience, and city
- Returns the **best matching existing job** with title, company, city, description, and a personalized recommendation

| Generative Demo | Retrieval Demo |
|:---:|:---:|
| ![Generative](Generative-Demo/demo.png) | ![Retrieval](Retrival-Demo.png) |

---

## Getting Started

### Prerequisites

```bash
# For the Generative Model
pip install numpy pandas scikit-learn torch tokenizers gradio matplotlib openai

# For the Retrieval Model
pip install numpy pandas scikit-learn faiss-cpu gradio matplotlib
```

### Run the Generative Model

1. Open `Generative-Ai.ipynb` in Google Colab or Jupyter
2. Upload `Jadarat_data.csv` to the working directory
3. Run all cells sequentially (phases 1-9)
4. The Gradio interface launches automatically at Phase 9

### Run the Retrieval Model

1. Open `Retrival.ipynb` in Google Colab or Jupyter
2. Upload `Jadarat_data.csv`
3. Run all cells — the TF-IDF vectorizer and FAISS index build automatically
4. The Gradio interface launches with a public share link

---

## Technical Limitations

| Limitation | Affects | Detail |
|------------|---------|--------|
| **Small dataset** | Both | 7,041 samples limits model capacity and retrieval diversity |
| **No pretrained embeddings** | Generative | Trained from scratch; Arabic-specific models (AraBERT, CAMeL) could improve quality |
| **CPU-compatible but slow** | Generative | Full training takes ~15 min on Colab GPU, significantly longer on CPU |
| **Single domain** | Both | Only Saudi job postings from Jadarat; not transferable to other markets |
| **Exact search** | Retrieval | FAISS flat index does exact search; approximate methods (IVF, HNSW) would scale better |
| **No semantic embeddings** | Retrieval | TF-IDF captures lexical overlap only; dense embeddings would improve semantic matching |

---

<div align="center">

Built with PyTorch, FAISS, and Gradio

</div>
