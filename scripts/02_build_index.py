import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json

def main(in_path: Path, out_dir: Path, model_name: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path).fillna("")
    df["text"] = ("PMID: " + df["pmid"].astype(str) + "\nTITLE: " + df["title"] + "\nABSTRACT: " + df["abstract"]).str.strip()

    emb_model = SentenceTransformer(model_name)
    texts = df["text"].tolist()
    embs = emb_model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embs = np.asarray(embs, dtype="float32")

    index = faiss.IndexFlatIP(embs.shape[1])  # cosine via normalized + inner product
    index.add(embs)

    faiss.write_index(index, str(out_dir / "faiss.index"))
    df[["pmid","title","abstract"]].to_csv(out_dir / "docs.csv", index=False)
    (out_dir / "meta.json").write_text(json.dumps({"embedding_model": model_name, "n_docs": len(df)}, indent=2))
    print(f"[OK] Index built -> {out_dir} (docs={len(df)})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", type=Path, default=Path("data/raw/pubmed_corpus.csv"))
    ap.add_argument("--out_dir", type=Path, default=Path("indexes/pubmed"))
    ap.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()
    main(args.in_path, args.out_dir, args.embed_model)
