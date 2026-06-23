import json
from collections import Counter
from pathlib import Path


CORPUS_CANDIDATES = [
    Path("generated_scnu_rag/scnu_rag_corpus.json"),
    Path("/root/autodl-tmp/llm_workspace/day4/homework3/rag_data/scnu_rag_corpus.json"),
]


def resolve_corpus_path() -> Path:
    for candidate in CORPUS_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("cannot find scnu_rag_corpus.json in known locations")


def main() -> None:
    docs = json.loads(resolve_corpus_path().read_text(encoding="utf-8"))
    assert isinstance(docs, list) and docs, "corpus must be a non-empty list"

    required_fields = {
        "title",
        "category",
        "source_url",
        "source_site",
        "last_checked",
        "is_time_sensitive",
        "keywords",
        "content",
        "slug",
    }
    categories = Counter()
    domains = Counter()
    lengths = []
    bad = []

    for idx, doc in enumerate(docs):
        missing = [field for field in required_fields if field not in doc]
        if missing:
            bad.append((idx, f"missing fields: {missing}"))
            continue
        if any(not doc[field] for field in ("title", "category", "source_url", "content", "slug")):
            bad.append((idx, "empty required field"))
            continue
        if "scnu.edu.cn" not in doc["source_site"]:
            bad.append((idx, f"non-scnu domain: {doc['source_site']}"))
        categories[doc["category"]] += 1
        domains[doc["source_site"]] += 1
        lengths.append(len(doc["content"]))

    print(f"documents: {len(docs)}")
    print(f"avg_content_len: {sum(lengths) / len(lengths):.1f}")
    print(f"min_content_len: {min(lengths)}")
    print(f"max_content_len: {max(lengths)}")
    print("category_distribution:")
    for category, count in sorted(categories.items()):
        print(f"  {category}: {count}")
    print("source_domains:")
    for domain, count in sorted(domains.items()):
        print(f"  {domain}: {count}")
    print(f"errors: {len(bad)}")
    for idx, message in bad[:10]:
        print(f"  doc[{idx}]: {message}")


if __name__ == "__main__":
    main()
