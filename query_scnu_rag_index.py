import json
import sys
from pathlib import Path

import joblib


INDEX_DIR = Path("/root/autodl-tmp/llm_workspace/day4/homework3/rag_index")


def expand_query(question: str) -> str:
    extras = []
    if any(token in question for token in ["成绩", "课表", "选课", "学籍", "考试", "毕业", "学位"]):
        extras.append("教务 本科生院 成绩管理 学籍管理 选课 课表 考试 自助打印")
    if any(token in question for token in ["图书馆", "续借", "借书", "借阅", "馆藏"]):
        extras.append("图书馆 续借 借阅 借阅证 我的图书馆 馆藏查询")
    if any(token in question for token in ["校园卡", "一卡通", "门禁", "充值", "挂失", "补卡"]):
        extras.append("校园卡 一卡通 挂失 解挂 补卡 充值 服务大厅")
    if any(token in question for token in ["奖学金", "助学金", "资助", "勤工助学", "困难补助"]):
        extras.append("奖学金 学生资助 学生工作部 通知公告 评选 助学金")
    if any(token in question for token in ["宿舍", "报修", "后勤", "空调", "门锁", "维修"]):
        extras.append("宿舍 报修 后勤 维修 物业 服务电话")
    if any(token in question for token in ["就业", "实习", "宣讲会", "毕业生"]):
        extras.append("就业 创业 就业指导中心 学生服务 双选会 宣讲会")
    if any(token in question for token in ["心理", "咨询", "焦虑", "预约"]):
        extras.append("心理咨询 心理健康教育与咨询中心 预约 联系方式")
    return question if not extras else f"{question} {' '.join(extras)}"


def load_assets():
    meta = json.loads((INDEX_DIR / "scnu_rag_index_meta.json").read_text(encoding="utf-8"))
    chunks = json.loads((INDEX_DIR / "scnu_rag_chunks.json").read_text(encoding="utf-8"))
    return meta, chunks


def query_dense(question: str, top_k: int):
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    meta, chunks = load_assets()
    model = SentenceTransformer(meta["embedding_model"])
    question_vec = model.encode([f"query: {question}"], normalize_embeddings=True)
    index = faiss.read_index(str(INDEX_DIR / "scnu_rag.index"))
    scores, indices = index.search(np.asarray(question_vec, dtype="float32"), top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        item = chunks[idx].copy()
        item["score"] = float(score)
        results.append(item)
    return results


def query_tfidf(question: str, top_k: int):
    from sklearn.metrics.pairwise import cosine_similarity

    meta, chunks = load_assets()
    vectorizer = joblib.load(INDEX_DIR / "tfidf_vectorizer.joblib")
    matrix = joblib.load(INDEX_DIR / "tfidf_matrix.joblib")
    query_vec = vectorizer.transform([expand_query(question)])
    scores = cosine_similarity(query_vec, matrix).ravel()
    ranked = scores.argsort()[::-1][:top_k]
    results = []
    for idx in ranked:
        item = chunks[int(idx)].copy()
        item["score"] = float(scores[int(idx)])
        results.append(item)
    return results


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python query_scnu_rag_index.py '你的问题' [top_k]")
    question = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    meta, _ = load_assets()
    if meta["retrieval_backend"] == "faiss_dense":
        results = query_dense(question, top_k)
    else:
        results = query_tfidf(question, top_k)
    print(json.dumps({"question": question, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
