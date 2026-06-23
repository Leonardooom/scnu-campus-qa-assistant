import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from query_scnu_rag_index import load_assets, query_dense, query_tfidf


DEFAULT_BASE_MODEL = "/root/autodl-tmp/llm_workspace/models/Qwen/Qwen2___5-1___5B-Instruct"
DEFAULT_ADAPTER = "/root/autodl-tmp/llm_workspace/LLaMA-Factory/output/scnu_campus_lora"


def retrieve(question: str, top_k: int) -> list[dict]:
    meta, _ = load_assets()
    if meta["retrieval_backend"] == "faiss_dense":
        return query_dense(question, top_k)
    return query_tfidf(question, top_k)


def build_prompt(question: str, contexts: list[dict]) -> str:
    blocks = []
    for idx, item in enumerate(contexts, start=1):
        blocks.append(
            f"[资料{idx}] 标题：{item['title']}\n"
            f"分类：{item['category']}\n"
            f"来源：{item['source_url']}\n"
            f"内容：{item['text']}"
        )
    context_text = "\n\n".join(blocks)
    return (
        "你是华南师范大学校园智能问答助手。请优先依据提供的官方资料回答。\n"
        "如果资料给出了系统、入口、步骤、限制条件或联系方式，请直接写清楚。\n"
        "如果资料不足以支持具体事实，不要编造，请明确说明应查看对应官方入口或最新通知。\n\n"
        f"【参考资料】\n{context_text}\n\n"
        f"【学生问题】\n{question}"
    )


def generate_answer(
    question: str,
    base_model: str,
    adapter_path: str,
    top_k: int,
    max_new_tokens: int,
) -> dict:
    contexts = retrieve(question, top_k)
    prompt = build_prompt(question, contexts)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    return {"question": question, "contexts": contexts, "answer": answer}


def main() -> None:
    parser = argparse.ArgumentParser(description="Query SCNU LoRA + RAG demo")
    parser.add_argument("question", help="学生问题")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    result = generate_answer(
        question=args.question,
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
