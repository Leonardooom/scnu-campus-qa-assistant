import argparse
import json
from collections import Counter
from pathlib import Path

REQUIRED = ("instruction", "input", "output")
BANNED = ("一般学校", "通常学校", "请自行查看官网")


def bigrams(text):
    compact = "".join(text.split())
    return {compact[i:i + 2] for i in range(max(0, len(compact) - 1))}


def similarity(a, b):
    aa, bb = bigrams(a), bigrams(b)
    return len(aa & bb) / len(aa | bb) if aa or bb else 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="training_data/scnu_campus_qa_200.json")
    parser.add_argument("--sources", default="training_data/scnu_campus_qa_sources.json")
    args = parser.parse_args()

    data = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    sources = json.loads(Path(args.sources).read_text(encoding="utf-8"))
    errors = []
    if not isinstance(data, list) or len(data) != 200:
        errors.append(f"训练数据必须恰好200条，当前{len(data) if isinstance(data, list) else '非列表'}")
    if not isinstance(sources, list) or len(sources) != 200:
        errors.append(f"来源清单必须恰好200条，当前{len(sources) if isinstance(sources, list) else '非列表'}")

    for i, row in enumerate(data, 1):
        if not isinstance(row, dict):
            errors.append(f"第{i}条不是对象")
            continue
        for field in REQUIRED:
            if not isinstance(row.get(field), str) or not row[field].strip():
                errors.append(f"第{i}条字段{field}为空或不是字符串")
        if any(term in row.get("output", "") for term in BANNED):
            errors.append(f"第{i}条含泛化占位表达")

    questions = [row.get("input", "").strip() for row in data if isinstance(row, dict)]
    answers = [row.get("output", "").strip() for row in data if isinstance(row, dict)]
    if len(set(questions)) != len(questions):
        errors.append(f"发现{len(questions) - len(set(questions))}条完全重复问题")
    if len(set(answers)) != len(answers):
        errors.append(f"发现{len(answers) - len(set(answers))}条完全重复答案")

    near = []
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            score = similarity(questions[i], questions[j])
            if score >= 0.88:
                near.append((i + 1, j + 1, round(score, 3)))
    if near:
        errors.append(f"发现{len(near)}组疑似语义近重复问题：{near[:10]}")

    category_counts = Counter(item.get("category", "未分类") for item in sources if isinstance(item, dict))
    source_questions = {item.get("question") for item in sources if isinstance(item, dict)}
    if set(questions) != source_questions:
        errors.append("来源清单与训练问题不能一一对应")
    missing_sources = sum(not str(item.get("official_source", "")).startswith(("https://", "http://")) for item in sources)
    if missing_sources:
        errors.append(f"有{missing_sources}条缺少HTTP(S)官方来源")

    lengths = [len(answer) for answer in answers]
    in_target = sum(60 <= length <= 220 for length in lengths)
    print("=" * 64)
    print(f"训练数据：{len(data)}条；唯一问题：{len(set(questions))}条")
    print(f"答案长度：平均{sum(lengths)/len(lengths):.1f}字，最短{min(lengths)}，最长{max(lengths)}")
    print(f"60~220字答案：{in_target}/{len(lengths)}（{in_target/len(lengths):.1%}）")
    print(f"来源覆盖：{len(source_questions)}/{len(questions)}；易变信息：{sum(bool(x.get('volatile')) for x in sources)}条")
    print("类别分布：")
    for category, count in category_counts.items():
        print(f"- {category}: {count}")
    print("质检结果：" + ("通过" if not errors else "未通过"))
    for error in errors:
        print(f"ERROR: {error}")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
