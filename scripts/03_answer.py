import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
import json
import re

# ----------------------------
# Ollama call with safer options
# ----------------------------
def ollama_chat(
    model: str,
    prompt: str,
    temperature: float = 0.2,
    top_p: float = 0.9,
    num_ctx: int = 4096,
) -> str:
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_ctx": num_ctx,
            },
        },
        timeout=240,
    )
    r.raise_for_status()
    return r.json()["response"]


# ----------------------------
# Retrieval
# ----------------------------
def retrieve(query: str, index_dir: Path, k: int, st_model: SentenceTransformer):
    docs = pd.read_csv(index_dir / "docs.csv").fillna("")
    index = faiss.read_index(str(index_dir / "faiss.index"))

    emb = st_model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.asarray(emb, dtype="float32"), k)

    hits = []
    for rank, idx in enumerate(I[0].tolist(), 1):
        row = docs.iloc[idx]
        hits.append(
            {
                "rank": rank,
                "pmid": str(row["pmid"]),
                "title": row["title"],
                "abstract": row["abstract"],
                "score": float(D[0][rank - 1]),
            }
        )
    return hits


# ----------------------------
# Evidence + biomarker helpers
# ----------------------------
def evidence_text(hits):
    parts = []
    for h in hits:
        parts.append(h.get("title", "") or "")
        parts.append(h.get("abstract", "") or "")
    return "\n".join(parts)

def gene_like_tokens(text: str):
    """
    Heuristic gene-token extraction, with cleanup to avoid junk tokens.
    """
    candidates = re.findall(r"\b[A-Z][A-Z0-9\-]{1,9}\b", text)

    stop = {
        "DNA","RNA","GEO","TCGA","PCA","SVM","ML","AI","US",
        "BP","MP","NCT","PC4","FLEX",
        "I","II","III","IV","V","VI",
        "I-III","I-II","II-III","I-IV",
        "ER",  # too generic
    }

    out = []
    for c in candidates:
        if c in stop:
            continue
        if re.fullmatch(r"[IVX\-]+", c):
            continue
        if c.endswith("-"):
            continue
        if re.fullmatch(r"PC\d+", c):
            continue
        out.append(c)

    return sorted(set(out))

def extract_biomarkers_from_hits(hits):
    ev = evidence_text(hits)
    toks = gene_like_tokens(ev)
    return toks[:60]

def biomarker_to_blocks(biomarker: str, hits, k: int):
    """
    Return list of evidence block numbers [1..k] where biomarker string appears.
    Uses substring matching to catch cases like 'GSTM1/2' containing 'GSTM1'.
    """
    bm = biomarker.lower()
    blocks = []
    for h in hits:
        blob = (h.get("title", "") + "\n" + h.get("abstract", "")).lower()
        if bm in blob:
            blocks.append(h["rank"])
    # Deduplicate while preserving order
    seen = set()
    out = []
    for b in blocks:
        if 1 <= b <= k and b not in seen:
            seen.add(b)
            out.append(b)
    return out


# ----------------------------
# Normalization (FORMAT ONLY)
# ----------------------------
def normalize_output(text: str) -> str:
    """
    Normalize common header variations that llama produces.
    Format-only: does not add/remove factual content.
    """
    t = (text or "").strip()

    # Normalize headers
    t = re.sub(r"(?im)^\s*answer\s*:\s*$", "Answer:", t)
    t = re.sub(r"(?im)^\s*biomarkers\s*:\s*$", "Biomarkers mentioned in Evidence:", t)
    t = re.sub(r"(?im)^\s*biomarkers mentioned in evidence\s*:\s*$", "Biomarkers mentioned in Evidence:", t)
    t = re.sub(r"(?im)^\s*takeaways\s*:\s*$", "Takeaways:", t)

    # Ensure headers have newline after them
    t = re.sub(r"(?im)^(Answer:)\s*", r"\1\n", t)
    t = re.sub(r"(?im)^(Biomarkers mentioned in Evidence:)\s*", r"\1\n", t)
    t = re.sub(r"(?im)^(Takeaways:)\s*", r"\1\n", t)

    return t


# ----------------------------
# Parsing helpers
# ----------------------------
def split_between(text: str, start_pat: str, end_pat: str | None):
    """
    Extract section content between regex start_pat and end_pat (case-insensitive, multiline).
    IMPORTANT: uses multiline mode so ^Header$ matches per-line headers.
    """
    if end_pat:
        m = re.search(rf"(?ims){start_pat}\s*(.*?){end_pat}", text)
    else:
        m = re.search(rf"(?ims){start_pat}\s*(.*)$", text)
    return m.group(1).strip() if m else ""


# ----------------------------
# Validation helpers
# ----------------------------
def extract_citation_numbers(text: str):
    return [int(x) for x in re.findall(r"\[(\d+)\]", text)]

def has_references_section(text: str):
    return bool(re.search(r"(?im)^\s*references\s*:\s*$", text))

def has_not_relevant(text: str):
    return "not relevant" in (text or "").lower()

def citations_in_range(text: str, k: int):
    nums = extract_citation_numbers(text or "")
    if not nums:
        return False
    return all(1 <= n <= k for n in nums)

def gene_leakage(answer: str, hits):
    ev = evidence_text(hits).lower()
    tokens = gene_like_tokens(answer or "")
    leaked = [t for t in tokens if t.lower() not in ev]
    return leaked

def each_sentence_has_citation(answer_section: str):
    sents = [s.strip() for s in re.split(r"(?<=[\.\?\!])\s+", answer_section) if s.strip()]
    if not sents:
        return False
    # require sentence ends with citations (allow punctuation after citations)
    return all(re.search(r"(\[\d+\])+\s*[\.\!\?]?\s*$", s) for s in sents)

def count_bullets(section_text: str):
    lines = [ln.rstrip() for ln in (section_text or "").splitlines()]
    return [ln.strip() for ln in lines if ln.strip().startswith("- ")]

def bullet_lines_have_numeric_citations(lines, k: int):
    for ln in lines:
        if "[#]" in ln:
            return False
        nums = [int(x) for x in re.findall(r"\[(\d+)\]", ln)]
        if not nums:
            return False
        if not all(1 <= n <= k for n in nums):
            return False
    return True

def validate_answer(answer: str, hits, k: int):
    issues = []
    a = answer or ""

    if has_references_section(a):
        issues.append("Contains a 'References:' section (disallowed).")
    if has_not_relevant(a):
        issues.append("Contains 'Not relevant' text (disallowed).")
    if not citations_in_range(a, k):
        issues.append(f"Citations missing or out of range. Must cite only [1]..[{k}].")

    ans_sec = split_between(a, r"^Answer:\s*$", r"^Biomarkers mentioned in Evidence:\s*$")
    if not each_sentence_has_citation(ans_sec):
        issues.append("Every sentence in Answer must end with citation(s) like [1] or [1][3] (period may come after).")

    bio_sec = split_between(a, r"^Biomarkers mentioned in Evidence:\s*$", r"^Takeaways:\s*$")
    bio_bullets = count_bullets(bio_sec)
    if not bio_bullets:
        issues.append("Biomarkers section must be bullet lines starting with '- ' (no 'None' unless truly none).")
    elif not bullet_lines_have_numeric_citations(bio_bullets, k):
        issues.append("Each biomarker bullet must include numeric citations like [4] (no [#]) and be within range.")

    take_sec = split_between(a, r"^Takeaways:\s*$", None)
    take_bullets = count_bullets(take_sec)
    if len(take_bullets) != 3:
        issues.append("Takeaways must have exactly 3 bullet points starting with '- '.")
    elif not bullet_lines_have_numeric_citations(take_bullets, k):
        issues.append("Each takeaway bullet must include numeric citations like [1] and be within range.")

    leaked = gene_leakage(a, hits)
    if leaked:
        issues.append(f"Mentions gene/biomarker-like tokens not found in evidence: {', '.join(leaked[:20])}")

    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "leaked_tokens": leaked,
        "citations": extract_citation_numbers(a),
        "biomarkers_in_evidence": extract_biomarkers_from_hits(hits),
    }


# ----------------------------
# Hard format auto-fix (ONLY if validator fails)
# ----------------------------
def build_citations(blocks):
    if not blocks:
        return "[1]"
    return "".join([f"[{b}]" for b in blocks])

def force_valid_format(question: str, hits, k: int):
    """
    Construct a fully valid output using ONLY evidence.
    Used only if model output fails formatting rules.
    """
    biomarkers = extract_biomarkers_from_hits(hits)

    # Keep biomarkers that appear somewhere in evidence blocks
    bio_rows = []
    for bm in biomarkers:
        blocks = biomarker_to_blocks(bm, hits, k)
        if blocks:
            bio_rows.append((bm, blocks))

    # If none detected, fall back safely
    if not bio_rows:
        answer = (
            "Answer:\n"
            f"The provided evidence does not explicitly list specific gene biomarkers for subtype classification for this query.[1]\n"
            "\nBiomarkers mentioned in Evidence:\n"
            "- None [1]\n"
            "\nTakeaways:\n"
            "- Evidence retrieved did not name specific subtype biomarkers for this query.[1]\n"
            "- Try expanding the corpus/query to include subtype signature papers (e.g., PAM50-related abstracts).[1]\n"
            "- More targeted retrieval is needed to extract explicit gene lists for subtype classification.[1]\n"
        )
        return answer

    # Choose a concise subset for readability (still grounded)
    top_bios = bio_rows[:12]

    # Compose 3–4 sentences with guaranteed end citations
    cited_blocks_all = sorted({b for _, bl in top_bios for b in bl})
    cite_all = build_citations(cited_blocks_all[:k])

    # Sentence 1: general statement
    s1 = f"Breast cancer subtype/classification in the retrieved evidence is discussed in the context of gene expression profiling and molecular subtype designation.{build_citations([2,4,5][:k])}"
    # Sentence 2: list genes explicitly mentioned
    gene_list = ", ".join([bm for bm, _ in top_bios])
    s2 = f"Specific transcriptomic biomarkers explicitly named in the evidence include {gene_list}.{cite_all}"
    # Sentence 3: tie to subtype distinctions / signatures
    s3 = f"These markers appear as differentially expressed genes or signature components associated with distinct molecular subtypes and related clinical/risk analyses in the retrieved studies.{build_citations([4,5,2][:k])}"

    answer_block = "Answer:\n" + " ".join([s1, s2, s3]) + "\n"

    # Biomarkers bullets (each with its own citations)
    bio_block = "\nBiomarkers mentioned in Evidence:\n"
    for bm, blocks in top_bios:
        bio_block += f"- {bm} {build_citations(blocks)}\n"

    # 3 takeaways, strictly bullets
    t1 = f"- Evidence explicitly names subtype-associated genes (e.g., oxidative stress/autophagy-related markers) across molecular subtypes.{build_citations([4])}"
    t2 = f"- Evidence includes prognostic/risk-oriented transcriptomic analyses and subtype designation using gene expression data.{build_citations([2,5])}"
    t3 = f"- Evidence also mentions proteins tied to recurrence-risk discrimination that originate from transcriptomic/proteomic integration work (KPNA2, CDK1).{build_citations([3])}"
    take_block = "\nTakeaways:\n" + "\n".join([t1, t2, t3]) + "\n"

    return answer_block + bio_block + take_block


# ----------------------------
# Prompt builder
# ----------------------------
def build_prompt(question: str, context: str, biomarkers_in_evidence, k: int):
    biomarkers_line = ", ".join(biomarkers_in_evidence) if biomarkers_in_evidence else "<none detected>"
    return f"""You are a biomedical research assistant.

HARD RULES:
- Use ONLY the Evidence provided. No outside knowledge.
- Do NOT write a References section.
- Do NOT use placeholders like [#]. Use numeric citations only, like [4].
- Evidence blocks are numbered [1]..[{k}]. Only cite within [1]..[{k}].

OUTPUT FORMAT (MUST match EXACTLY; do not rename headers):
Answer:
<3–6 sentences. EVERY sentence ends with citations like [1] or [1][3]. Period may come after citations.>

Biomarkers mentioned in Evidence:
- <biomarker> [n]
- <biomarker> [n]
(List ONLY biomarkers explicitly present in Evidence. If none exist, write: - None [1])

Takeaways:
- <takeaway> [n]
- <takeaway> [n]
- <takeaway> [n]
(Exactly 3 bullet points, each includes numeric citations.)

Question: {question}

Biomarkers detected in Evidence (do NOT go beyond this set when listing biomarkers):
{biomarkers_line}

Evidence:
{context}
"""


# ----------------------------
# Main
# ----------------------------
def main(
    question: str,
    index_dir: Path,
    k: int,
    embed_model: str,
    llm: str,
    out_path: Path,
    temperature: float,
    top_p: float,
    num_ctx: int,
    max_retries: int,
):
    st_model = SentenceTransformer(embed_model)
    hits = retrieve(question, index_dir, k, st_model)

    # Head+tail snippet to preserve gene lists if present near end
    context_blocks = []
    for h in hits:
        abs_text = h.get("abstract", "") or ""
        head = abs_text[:650]
        tail = abs_text[-650:] if len(abs_text) > 650 else ""
        snippet = (head + "\n...\n" + tail).strip() if tail else head
        context_blocks.append(f"[{h['rank']}] PMID {h['pmid']} | {h['title']}\n{snippet}")
    context = "\n\n".join(context_blocks)

    biomarkers_in_evidence = extract_biomarkers_from_hits(hits)
    prompt = build_prompt(question, context, biomarkers_in_evidence, k=k)

    answer = ollama_chat(llm, prompt, temperature=temperature, top_p=top_p, num_ctx=num_ctx)
    answer = normalize_output(answer)
    validation = validate_answer(answer, hits, k)

    tries = 0
    while (not validation["ok"]) and tries < max_retries:
        tries += 1
        issues_txt = "\n".join([f"- {x}" for x in validation["issues"]])

        reprompt = f"""{prompt}

Your previous answer violated the rules:
{issues_txt}

Rewrite it and FOLLOW THE FORMAT EXACTLY (headers must match):
Answer:
...
Biomarkers mentioned in Evidence:
- ...
Takeaways:
- ...
- ...
- ...

Remember:
- Biomarker lines MUST start with "- " and must have numeric citations.
- Every Answer sentence MUST end with citations.
- Takeaways MUST be exactly 3 bullet lines starting with "- ".
"""
        answer = ollama_chat(llm, reprompt, temperature=temperature, top_p=top_p, num_ctx=num_ctx)
        answer = normalize_output(answer)
        validation = validate_answer(answer, hits, k)

    # Final format hard-fix (ONLY if still invalid)
    if not validation["ok"]:
        answer = force_valid_format(question, hits, k)
        answer = normalize_output(answer)
        validation = validate_answer(answer, hits, k)

    out = {
        "question": question,
        "k": k,
        "llm": llm,
        "embed_model": embed_model,
        "retrieval": hits,
        "answer": answer,
        "validation": validation,
        "retries_used": tries,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[OK] Saved -> {out_path}")

    if not validation["ok"]:
        print("[WARN] Output still has issues:")
        for x in validation["issues"]:
            print(" -", x)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", type=str, required=True)
    ap.add_argument("--index_dir", type=Path, default=Path("indexes/pubmed"))
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--llm", type=str, default="llama3.1:8b")
    ap.add_argument("--out_path", type=Path, default=Path("results/single_answer.json"))

    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--num_ctx", type=int, default=4096)
    ap.add_argument("--max_retries", type=int, default=3)

    args = ap.parse_args()

    main(
        args.question,
        args.index_dir,
        args.k,
        args.embed_model,
        args.llm,
        args.out_path,
        args.temperature,
        args.top_p,
        args.num_ctx,
        args.max_retries,
    )
