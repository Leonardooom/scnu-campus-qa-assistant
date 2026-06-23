import json
from pathlib import Path

import joblib

CORPUS_CANDIDATES = [
    Path("/root/autodl-tmp/llm_workspace/day4/homework3/rag_data/scnu_rag_corpus.json"),
    Path("generated_scnu_rag/scnu_rag_corpus.json"),
]
INDEX_DIR = Path("/root/autodl-tmp/llm_workspace/day4/homework3/rag_index")
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
CHUNK_SIZE = 420
CHUNK_OVERLAP = 80


def resolve_corpus_path() -> Path:
    for candidate in CORPUS_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("cannot find scnu_rag_corpus.json in known locations")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks


def build_records(docs: list[dict]) -> list[dict]:
    records = []
    for doc in docs:
        for idx, chunk in enumerate(chunk_text(doc["content"])):
            records.append(
                {
                    "chunk_id": f"{doc['slug']}#{idx}",
                    "slug": doc["slug"],
                    "title": doc["title"],
                    "category": doc["category"],
                    "source_url": doc["source_url"],
                    "source_site": doc["source_site"],
                    "keywords": doc["keywords"],
                    "text": chunk,
                }
            )
    return records


def main() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    docs = json.loads(resolve_corpus_path().read_text(encoding="utf-8"))
    records = build_records(docs)
    assert records, "no chunks produced from corpus"

    backend = "tfidf"
    embedding_dim = None

    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(MODEL_NAME)
        texts = [f"passage: {record['title']}\n{record['text']}" for record in records]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        matrix = np.asarray(embeddings, dtype="float32")

        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        faiss.write_index(index, str(INDEX_DIR / "scnu_rag.index"))
        backend = "faiss_dense"
        embedding_dim = int(matrix.shape[1])
    except Exception as exc:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=1)
        texts = [f"{record['title']} {record['category']} {' '.join(record['keywords'])} {record['text']}" for record in records]
        matrix = vectorizer.fit_transform(texts)
        joblib.dump(vectorizer, INDEX_DIR / "tfidf_vectorizer.joblib")
        joblib.dump(matrix, INDEX_DIR / "tfidf_matrix.joblib")
        (INDEX_DIR / "tfidf_fallback_reason.txt").write_text(str(exc), encoding="utf-8")
        embedding_dim = int(matrix.shape[1])

    (INDEX_DIR / "scnu_rag_chunks.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (INDEX_DIR / "scnu_rag_index_meta.json").write_text(
        json.dumps(
            {
                "retrieval_backend": backend,
                "embedding_model": MODEL_NAME if backend == "faiss_dense" else None,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "document_count": len(docs),
                "chunk_count": len(records),
                "feature_dim": embedding_dim,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "document_count": len(docs),
                "chunk_count": len(records),
                "feature_dim": embedding_dim,
                "retrieval_backend": backend,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
