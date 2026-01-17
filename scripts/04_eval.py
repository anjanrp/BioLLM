import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
import re
import random

import faiss
from sentence_transformers import SentenceTransformer
import requests
from textwrap import shorten
from typing import Any, List, Dict


# ----------------------------
# Ollama
# ----------------------------
def ollama_chat(model: str, prompt: str) -> str:
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()["response"]


# ----------------------------
# Retrieval (cached)
# ----------------------------
def load_corpus(index_dir: Path):
    docs = pd.read_csv(index_dir / "docs.csv").fillna("")
    index = faiss.read_index(str(index_dir / "faiss.index"))
    return docs, index


def retrieve_cached(
    query: str,
    docs: pd.DataFrame,
    index,
    k: int,
    st_model: SentenceTransformer,
) -> List[Dict[str, Any]]:
    emb = st_model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.asarray(emb, dtype="float32"), k)

    hits = []
    for rank, idx in enumerate(I[0].tolist(), 1):
        row = docs.iloc[idx]
        hits.append(
            {
                "rank": rank,
                "pmid": str(row["pmid"]),
                "title": str(row["title"]),
                "abstract": str(row["abstract"]),
                "score": float(D[0][rank - 1]),
            }
        )
    return hits


# ----------------------------
# Eval helpers
# ----------------------------
def parse_relevant_pmids(x: Any) -> List[str]:
    """
    Robust parser for relevant_pmids which may come as:
      - "123|456" or "123,456"
      - int (123)
      - float (123.0) from pandas
      - empty / NaN
    Returns list of string PMIDs.
    """
    if x is None:
        return []

    # pandas can give floats for numeric columns; handle NaN
    if isinstance(x, float):
        if np.isnan(x):
            return []
        # if it's 123.0, treat as "123"
        if x.is_integer():
            return [str(int(x))]
        return [str(x)]

    # ints / numpy ints
    if isinstance(x, (int, np.integer)):
        return [str(int(x))]

    # already a list/tuple/set
    if isinstance(x, (list, tuple, set)):
        out = []
        for item in x:
            out.extend(parse_relevant_pmids(item))
        return [p for p in out if p]

    # default: string-like
    s = str(x).strip()
    if not s:
        return []

    parts = re.split(r"[,\|;\s]+", s)
    return [p for p in parts if p]


def recall_at_k(hits, relevant_pmids):
    rel = set(map(str, relevant_pmids))
    got = [h["pmid"] for h in hits]
    return 1.0 if any(p in rel for p in got) else 0.0


def citations_present(answer: str) -> int:
    return 1 if re.search(r"\[\d+\]", answer) else 0


def citations_valid(answer: str, k: int) -> int:
    nums = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]
    if not nums:
        return 0
    return 1 if all(1 <= n <= k for n in nums) else 0


# ----------------------------
# Silver eval set builder
# ----------------------------
def make_silver_eval_set(
    index_dir: Path,
    out_csv: Path,
    n_questions: int = 140,
    seed: int = 13,
):
    """
    Create a simple, reproducible benchmark from the corpus itself:
    - pick N documents
    - query = title with some tokens dropped (harder than exact title)
    - relevant_pmids = that PMID
    """
    rng = random.Random(seed)
    docs = pd.read_csv(index_dir / "docs.csv").fillna("")
    if len(docs) < n_questions:
        raise SystemExit(
            f"Need at least {n_questions} docs in {index_dir/'docs.csv'} "
            f"(found {len(docs)}). Fetch more first."
        )

    sample_idx = rng.sample(range(len(docs)), n_questions)
    rows = []
    for i, idx in enumerate(sample_idx):
        pmid = str(docs.iloc[idx]["pmid"]).strip()
        title = str(docs.iloc[idx]["title"]).strip()
        toks = re.findall(r"[A-Za-z0-9\-]+", title)

        if len(toks) >= 8:
            keep = [t for t in toks if rng.random() > 0.30]
            if len(keep) < 4:
                keep = toks[:6]
            query = " ".join(keep)
        else:
            query = title

        rows.append({"qid": f"q{i:03d}", "question": query, "relevant_pmids": pmid})

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[OK] Wrote silver eval set -> {out_csv} (n={n_questions})")


# ----------------------------
# Main
# ----------------------------
def main(
    eval_csv: Path,
    index_dir: Path,
    out_dir: Path,
    k: int,
    embed_model: str,
    llm: str,
    run_llm: bool,
    n_questions: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    if not eval_csv.exists():
        make_silver_eval_set(index_dir, eval_csv, n_questions=n_questions, seed=13)

    # Force relevant_pmids to be strings so pandas doesn’t coerce to int/float
    df = pd.read_csv(
        eval_csv,
        dtype={"qid": str, "question": str, "relevant_pmids": str},
    ).fillna("")

    # Cache heavy objects
    docs, faiss_index = load_corpus(index_dir)
    st_model = SentenceTransformer(embed_model)

    rows = []
    all_runs = []

    for row in df.itertuples(index=False):
        qid = getattr(row, "qid", "")
        q = getattr(row, "question", "")
        relevant = parse_relevant_pmids(getattr(row, "relevant_pmids", ""))

        hits = retrieve_cached(q, docs, faiss_index, k, st_model)
        r_at_k = recall_at_k(hits, relevant)

        ans = ""
        cite_ok = 0
        cite_valid = 0

        if run_llm:
            context_blocks = []
            for h in hits:
                snippet = shorten(h["abstract"], width=550, placeholder="…")
                context_blocks.append(
                    f"[{h['rank']}] PMID {h['pmid']} | {h['title']}\n{snippet}"
                )
            context = "\n\n".join(context_blocks)

            prompt = f"""You are a biomedical research assistant.
Answer the question using ONLY the provided evidence.
If evidence is insufficient, say so.
Cite sources like [1], [2] matching the evidence blocks.

Question: {q}

Evidence:
{context}
"""
            ans = ollama_chat(llm, prompt)
            cite_ok = citations_present(ans)
            cite_valid = citations_valid(ans, k)

        rows.append(
            {
                "qid": qid,
                "recall_at_k": r_at_k,
                "citations_present": cite_ok,
                "citations_valid": cite_valid,
            }
        )

        all_runs.append(
            {
                "qid": qid,
                "question": q,
                "relevant_pmids": relevant,
                "retrieval": hits,
                "answer": ans,
            }
        )

    m = pd.DataFrame(rows)
    metrics = {
        "recall@k": float(m["recall_at_k"].mean()),
        "citations_present_rate": float(m["citations_present"].mean()) if run_llm else None,
        "citations_valid_rate": float(m["citations_valid"].mean()) if run_llm else None,
        "k": k,
        "embed_model": embed_model,
        "llm": llm if run_llm else None,
        "n_questions": int(len(df)),
        "eval_csv": str(eval_csv),
    }

    (out_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    m.to_csv(out_dir / "eval_table.csv", index=False)
    (out_dir / "eval_runs.json").write_text(json.dumps(all_runs, indent=2))
    print(f"[OK] Wrote -> {out_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_csv", type=Path, default=Path("data/processed/eval_questions.csv"))
    ap.add_argument("--index_dir", type=Path, default=Path("indexes/pubmed"))
    ap.add_argument("--out_dir", type=Path, default=Path("results/eval_v1"))
    ap.add_argument("--k", type=int, default=5)  # align with resume Top-5
    ap.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--llm", type=str, default="llama3.1:8b")
    ap.add_argument("--run_llm", action="store_true", help="Also generate answers and check citation formatting")
    ap.add_argument("--n_questions", type=int, default=140, help="Only used when auto-creating eval_csv")
    args = ap.parse_args()

    main(
        args.eval_csv,
        args.index_dir,
        args.out_dir,
        args.k,
        args.embed_model,
        args.llm,
        args.run_llm,
        args.n_questions,
    )
