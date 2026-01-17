import argparse, time
from pathlib import Path
import requests
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

def esearch(term: str, retmax: int = 500, api_key: str | None = None, email: str | None = None, tool: str = "biomed-rag"):
    params = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": retmax,
        "sort": "relevance",
    }
    # NCBI best practice (optional but recommended)
    if api_key: params["api_key"] = api_key
    if email:  params["email"] = email
    if tool:   params["tool"] = tool

    r = requests.get(f"{EUTILS}/esearch.fcgi", params=params, timeout=60)
    r.raise_for_status()
    data = r.json().get("esearchresult", {})
    return data.get("idlist", [])

def efetch(pmids, api_key: str | None = None, email: str | None = None, tool: str = "biomed-rag"):
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
    if api_key: params["api_key"] = api_key
    if email:  params["email"] = email
    if tool:   params["tool"] = tool

    r = requests.get(f"{EUTILS}/efetch.fcgi", params=params, timeout=120)
    r.raise_for_status()
    return r.text

def parse_pubmed_xml(xml_text: str):
    """Robust-enough XML parsing using BeautifulSoup (lxml-xml)."""
    soup = BeautifulSoup(xml_text, "lxml-xml")
    rows = []
    for art in soup.find_all("PubmedArticle"):
        pmid_tag = art.find("PMID")
        pmid = pmid_tag.get_text(strip=True) if pmid_tag else None

        title_tag = art.find("ArticleTitle")
        title = title_tag.get_text(" ", strip=True) if title_tag else ""

        abs_texts = []
        abstract = art.find("Abstract")
        if abstract:
            for t in abstract.find_all("AbstractText"):
                abs_texts.append(t.get_text(" ", strip=True))
        abstract_joined = " ".join(abs_texts).strip()

        if pmid and (title or abstract_joined):
            rows.append({"pmid": pmid, "title": title, "abstract": abstract_joined})
    return rows

def broaden_queries(base_query: str):
    """
    If query is too strict, try a few progressively broader fallbacks.
    """
    fallbacks = [
        base_query,
        # drop some constraints
        '(breast cancer[Title/Abstract]) AND (biomarker[Title/Abstract] OR "gene expression"[Title/Abstract] OR transcriptomic[Title/Abstract])',
        '(breast cancer[Title/Abstract]) AND (biomarker[Title/Abstract] OR "gene expression"[Title/Abstract])',
        '(cancer[Title/Abstract]) AND (biomarker[Title/Abstract] OR "gene expression"[Title/Abstract])',
        'biomarker[Title/Abstract] AND "gene expression"[Title/Abstract]',
        'cancer[Title/Abstract]',
    ]
    # de-dup while preserving order
    seen, out = set(), []
    for q in fallbacks:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out

def main(out_dir: Path, query: str, n_docs: int, batch: int, sleep_s: float, api_key: str | None, email: str | None):
    out_dir.mkdir(parents=True, exist_ok=True)

    pmids = []
    used_query = None
    for q in broaden_queries(query):
        pmids = esearch(q, retmax=n_docs, api_key=api_key, email=email)
        if pmids:
            used_query = q
            break

    if not pmids:
        raise SystemExit("No PMIDs returned even after broadening. Try setting --query to something simpler like: cancer[Title/Abstract]")

    pmids = pmids[:n_docs]
    rows = []

    for i in tqdm(range(0, len(pmids), batch), desc="Fetching"):
        chunk = pmids[i:i+batch]
        xml = efetch(chunk, api_key=api_key, email=email)
        rows.extend(parse_pubmed_xml(xml))
        time.sleep(sleep_s)

    df = pd.DataFrame(rows).drop_duplicates(subset=["pmid"])
    out_path = out_dir / "pubmed_corpus.csv"
    df.to_csv(out_path, index=False)

    meta_path = out_dir / "pubmed_fetch_meta.json"
    meta_path.write_text(
        pd.Series({
            "query_used": used_query,
            "requested_n_docs": n_docs,
            "saved_n_docs": len(df)
        }).to_json(indent=2)
    )

    print(f"[OK] Query used: {used_query}")
    print(f"[OK] Saved {len(df)} docs -> {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--query", type=str, default='(breast cancer[Title/Abstract]) AND (biomarker[Title/Abstract] OR "gene expression"[Title/Abstract] OR transcriptomic[Title/Abstract])')
    ap.add_argument("--n_docs", type=int, default=800)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--sleep_s", type=float, default=0.34)  # be polite
    ap.add_argument("--api_key", type=str, default=None)
    ap.add_argument("--email", type=str, default=None)
    args = ap.parse_args()
    main(args.out_dir, args.query, args.n_docs, args.batch, args.sleep_s, args.api_key, args.email)
