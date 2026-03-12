# Kafaa'at Job Matching System

## About the Project

This project is an integrated AI system aimed at guiding job seekers to discover suitable opportunities based on the "Jadarat" platform dataset (approximately 7,000 job postings).

To achieve the best possible results, **two different Artificial Intelligence models were built, tested, and compared** to select the most accurate and practical solution for the job market context.

---

## The AI Models Developed

### Model 1: Generative AI (Transformers)

* **Concept:** Building a Mini-LLM from scratch using Transformer architecture in PyTorch to autoregressively generate job recommendations word by word.
* **Technologies:** `PyTorch`, `BPE Tokenization`, `Multi-Head Attention`.
* **Mechanism:** The model was trained on the dataset's job texts to understand Arabic context and synthesize new, custom recommendations based on user input.

### Model 2: Retrieval-Based AI — *The Approved & Best Performing Model* 

* **Concept:** Instead of generating entirely new text, this model relies on semantic understanding and Vector Similarity Search to retrieve the most accurate and relevant job that actually exists in the database.
* **Technologies:** `TF-IDF Vectorizer`, `FAISS (Facebook AI Similarity Search)`, `Scikit-learn`.
* **Mechanism:** 1. Converts all job texts into numerical vectors.
2. Converts the user's input (qualifications, experience, city) into a vector.
3. Uses the ultra-fast `FAISS` library to calculate the mathematical distance (similarity) between the user's query and available jobs, extracting the closest exact match and presenting it within a professional text template.

---

## Results & Conclusion

After testing both models, **it was conclusively proven that Model 2 (Retrieval-Based using FAISS) delivered vastly superior and more practical results compared to Model 1**. The reasons for this superiority include:

1. **Factual Accuracy:** The retrieval-based model extracts real, existing jobs from the "Jadarat" database. In contrast, the Generative model suffered from "Hallucination," occasionally inventing non-existent job titles or cities.
2. **Dataset Size Constraints:** The dataset consists of roughly 7,000 rows. While this is an ideal size for TF-IDF and FAISS to operate with high efficiency, it is severely insufficient to train a Generative Transformer model from scratch to output perfectly fluent Arabic text without overfitting.
3. **Speed & Efficiency:** The approved FAISS model is significantly faster in response time and consumes far fewer computational resources (it runs highly efficiently on a standard CPU), making it ideal for real-world application and deployment.

---

## Data Preprocessing Pipeline

Both models shared a strict Arabic data cleaning pipeline to ensure high-quality input, which included:

* **Diacritics Removal:** Stripping out all Arabic diacritics (Tashkeel).
* **Hamza Normalization:** Unifying all forms of Alef (أ، إ، آ) into a single standard Alef (ا).
* **Character Normalization:** Unifying similar letters (e.g., converting ى to ي, and ؤ to و).
* **Punctuation Removal:** Stripping non-textual symbols and extra whitespaces to reduce noise and help the models focus on semantic meaning.

---

## How to Run (For the Approved FAISS Model)

1. Ensure the required libraries are installed: `numpy`, `pandas`, `scikit-learn`, `faiss-cpu`, `gradio`.
2. Place the dataset file `Jadarat_data.csv` in the correct working directory.
3. Run the cells in the provided `Jupyter Notebook` sequentially.
4. The interactive `Gradio` UI will launch in the final cell, generating a Public URL to test the job recommendation chatbot in real-time.

---
## GenAi link
https://colab.research.google.com/drive/1vJrqdgNbiNeHqzGa69crbu0DE9CGPe8m?usp=sharing
