# BioLLM

# BioLLM — Biomedical Semantic Search + RAG (Local, Reproducible)

A lightweight biomedical RAG prototype that retrieves PubMed abstracts using sentence embeddings + FAISS, then generates evidence-grounded answers locally using Ollama.

## What this project does

* **Ingests PubMed abstracts** (query-based fetch) and stores a clean corpus.
* **Builds a vector index** using **Sentence-Transformers** embeddings + **FAISS** for semantic retrieval.
* **Answers questions with citations** using a **local LLM via Ollama**, constrained to retrieved evidence.
* **Evaluates retrieval quality** using a reproducible benchmark + **Recall@K** metrics.

---

## Tech stack

* Python (3.11)
* Sentence-Transformers (`all-MiniLM-L6-v2`)
* FAISS (CPU)
* Ollama (local LLM; tested with `llama3.1:8b`)
* pandas, numpy, scikit-learn

---

## Project structure

```
BioLLM/
  scripts/
    01_fetch_pubmed.py
    02_build_index.py
    03_answer.py
    04_eval.py
  data/
    raw/           # ignored (downloaded/raw)
    processed/     # small eval CSV lives here
  indexes/         # ignored (FAISS index + docs)
  results/         # ignored by default
  .gitignore
```

---

## Setup

### 1) Create environment

```bash
conda create -n biomed-rag python=3.11 -y
conda activate biomed-rag
pip install numpy pandas tqdm scikit-learn matplotlib requests beautifulsoup4 lxml
pip install sentence-transformers faiss-cpu
```

### 2) Install + run Ollama

```bash
brew install ollama
ollama pull llama3.1:8b
ollama serve
```

---

## Run the pipeline

### 1) Fetch PubMed abstracts

```bash
python scripts/01_fetch_pubmed.py --n_docs 800
```

> If you get “No PMIDs returned”, use a broader query inside the script (or reduce filters).

### 2) Build FAISS index

```bash
python scripts/02_build_index.py
```

### 3) Ask a question (RAG answer with citations)

```bash
python scripts/03_answer.py \
  --question "What transcriptomic biomarkers are used for breast cancer subtype classification?" \
  --k 6 \
  --llm llama3.1:8b \
  --temperature 0.2 \
  --max_retries 3
```

Output is saved to:

* `results/single_answer.json`

---

## Evaluation

### Retrieval-only evaluation (Recall@K)

```bash
python scripts/04_eval.py --k 5
```

### Retrieval + answer generation (citation checks)

```bash
python scripts/04_eval.py --k 5 --run_llm
```

**Example metrics (n=140, k=5):**

* **Recall@5:** 0.8429
* **Citation present rate:** 0.9857
* **Citation valid rate (1..k):** 0.9857

> “Citation valid” checks that bracketed citations like `[1]` refer only to retrieved evidence blocks.

---

## Notes on reproducibility

* Large/generated folders (`data/raw/`, `indexes/`, `results/`) are ignored by default to keep the repo lightweight.
* Evaluation is reproducible via a fixed-seed “silver” benchmark generated from the indexed corpus.

---

## Resume alignment (what this demonstrates)

* Built a **document ingestion + embedding + indexing pipeline** over biomedical literature.
* Implemented **semantic retrieval** with FAISS + Sentence-Transformers.
* Added **evidence-grounded answering with citations** using a **local LLM** (Ollama).
* Built an **evaluation harness** measuring **Recall@K** and **citation formatting validity**.

---
