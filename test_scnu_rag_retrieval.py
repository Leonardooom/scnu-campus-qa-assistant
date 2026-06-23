import json
from pathlib import Path

import joblib
from sklearn.metrics.pairwise import cosine_similarity


INDEX_DIR = Path("/root/autodl-tmp/llm_workspace/day4/homework3/rag_index")

QUERIES = [
    "成绩查询从哪里进入",
    "图书馆图书如何续借",
    "校园卡丢失后怎么处理",
    "奖学金通知一般看哪里",
    "宿舍报修通过什么途径",
]


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
    return question if not extras else f"{question} {' '.join(extras)}"


def main() -> None:
    vectorizer = joblib.load(INDEX_DIR / "tfidf_vectorizer.joblib")
    matrix = joblib.load(INDEX_DIR / "tfidf_matrix.joblib")
    chunks = json.loads((INDEX_DIR / "scnu_rag_chunks.json").read_text(encoding="utf-8"))

    for question in QUERIES:
        query_vec = vectorizer.transform([expand_query(question)])
        scores = cosine_similarity(query_vec, matrix).ravel()
        ranked = scores.argsort()[::-1][:3]
        print(f"Q: {question}")
        for rank, idx in enumerate(ranked, start=1):
            item = chunks[int(idx)]
            print(f"  {rank}. {item['slug']} | {item['title']} | {scores[int(idx)]:.4f}")
        print()


if __name__ == "__main__":
    main()
