# Kafaat — AI-Powered Job Recommendation System

> An intelligent job recommendation system that combines two AI approaches to suggest personalized Arabic job listings, built on data from the **Jadarat** platform (7,041 job postings).

---

## 📁 Project Structure

```
├── Generative-AI.ipynb   # Generative model (GPT-like Transformer)
├── Retrival.ipynb        # Retrieval model (TF-IDF + FAISS)
└── Jadarat_data.csv      # Job listings dataset (7,041 records)
```

---

## 🧠 Model 1 — Generative AI

> **Idea:** Build a generative language model from scratch that learns from job data and produces new Arabic job recommendations.

### ⚙️ Technical Components

| Component | Details |
|-----------|---------|
| **Model** | GPT-like Transformer built with PyTorch |
| **Architecture** | 4 Transformer Blocks, 8 attention heads, d_model=256 |
| **Tokenizer** | BPE (Byte-Pair Encoding) trained on Arabic data |
| **Vocabulary Size** | 5,000 tokens |
| **Optimizer** | AdamW (lr=3e-4) with Gradient Clipping |
| **Loss Function** | CrossEntropyLoss |
| **Training** | 2 epochs, 50 batches per epoch (Demo Mode) |

### 🔄 Pipeline

```
Raw Data
    ↓
Arabic Text Normalization (remove diacritics, unify characters)
    ↓
Train BPE Tokenizer
    ↓
Build JobDataset (Input / Target sequences)
    ↓
Train JobRecommendationGPT
    ↓
Autoregressive text generation with temperature sampling
    ↓
Interactive Gradio UI
```

### 📊 Training Results

| Metric | Value |
|--------|-------|
| Training Loss (Epoch 1) | 6.44 |
| Training Loss (Epoch 2) | 4.92 |
| Test Perplexity | 63.02 |

### ⚠️ Failure Modes Report

- **Repetition:** The model may repeat sentences due to limited training data
- **Context Loss:** The model may forget early prompt details in longer generations
- **Hallucination:** May suggest unrelated skills due to vocabulary compression

### Usage

```python
# Generate a job recommendation
generate_text("Job Title: Software Developer | City:", max_tokens=60, temperature=0.7)
```

---

## 🔍 Model 2 — Retrieval AI

> **Idea:** Build a semantic search engine that retrieves the most relevant job from the Jadarat database based on the user's query.

### ⚙️ Technical Components

| Component | Details |
|-----------|---------|
| **Vectorizer** | TF-IDF (max_features=5,000, n-gram range 1-2) |
| **Search Engine** | FAISS IndexFlatIP (Inner Product similarity) |
| **Normalization** | L2 normalization on all vectors |
| **Output** | Top-1 most relevant job match |

### 🔄 Pipeline

```
User Query
    ↓
Arabic Text Normalization
    ↓
Convert query to TF-IDF vector
    ↓
Search FAISS Index (Cosine Similarity)
    ↓
Retrieve best matching job
    ↓
Build formatted recommendation
    ↓
Interactive Gradio UI
```

### 💡 Search Quality Boost

City and job title are weighted higher in the search index:

```python
df["job_text"] = (
    job_title * 1 +
    job_description * 1 +
    city * 3 +        # repeated to boost city weight
    job_title * 1     # repeated to boost title weight
)
```

### 🚀 Usage

```python
# Find the best matching job
row = retrieve_job("Software Engineer Riyadh")
print(row["المسمى الوظيفي"], row["المدينة"])
```

---

## 🆚 Model Comparison

| Criteria | Generative AI | Retrieval AI |
|----------|--------------|--------------|
| **Approach** | Generates new text | Retrieves existing records |
| **Speed** | Slower (inference) | Very fast |
| **Accuracy** | Creative, may hallucinate | Precise, from real data |
| **Training** | Requires GPU & training | No deep training needed |
| **Best Use** | Custom creative recommendations | Matching real job listings |

---

## 📦 Requirements

```bash
pip install numpy pandas scikit-learn gradio matplotlib torch tokenizers faiss-cpu
```

---

## 🗂️ Dataset

The project uses `Jadarat_data.csv` containing **7,041 job postings** from the Jadarat platform with the following fields:

| Field | Description |
|-------|-------------|
| `المسمى الوظيفي` | Job Title |
| `الوصف الوظيفي` | Job Description |
| `المدينة` | City |
| `الجهة` | Organization |

---

## 🖥️ Interactive UI

Both models run via a **Gradio Chat Interface** on Google Colab:

```python
demo.launch(share=True)  # Generates a temporary public link
```

---

## 👨‍💻 Author

**Anwar Alotaibi**  
