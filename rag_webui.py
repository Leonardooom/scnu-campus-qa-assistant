import argparse
import json
from pathlib import Path

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from query_scnu_rag_index import load_assets, query_dense, query_tfidf


DEFAULT_BASE_MODEL = "/root/autodl-tmp/llm_workspace/models/Qwen/Qwen2___5-1___5B-Instruct"
DEFAULT_ADAPTER = "/root/autodl-tmp/llm_workspace/LLaMA-Factory/output/scnu_campus_lora"


class RagChatEngine:
    def __init__(self, base_model: str, adapter_path: str, max_new_tokens: int):
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.tokenizer = None
        self.model = None

    def load(self):
        if self.model is not None and self.tokenizer is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(model, self.adapter_path)
        self.model.eval()

    def retrieve(self, question: str, top_k: int) -> list[dict]:
        meta, _ = load_assets()
        if meta["retrieval_backend"] == "faiss_dense":
            return query_dense(question, top_k)
        return query_tfidf(question, top_k)

    @staticmethod
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
            "如果资料中给出了系统、入口、步骤、限制条件或联系方式，请直接写清楚。\n"
            "如果资料不足以支持具体事实，不要编造，请明确说明应查看对应官方入口或最新通知。\n\n"
            f"【参考资料】\n{context_text}\n\n"
            f"【学生问题】\n{question}"
        )

    def answer(self, question: str, top_k: int):
        self.load()
        contexts = self.retrieve(question, top_k)
        prompt = self.build_prompt(question, contexts)
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )
        answer = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()
        refs = []
        for item in contexts:
            refs.append(
                {
                    "title": item["title"],
                    "category": item["category"],
                    "source_url": item["source_url"],
                    "score": round(float(item["score"]), 4),
                    "text": item["text"],
                }
            )
        return answer, refs


def build_app(engine: RagChatEngine):
    examples = [
        "成绩查询从哪里进入？",
        "图书馆图书如何续借？",
        "校园卡丢失后怎么处理？",
        "奖学金通知一般看哪里？",
        "宿舍报修通过什么途径？",
    ]

    def chat_fn(message, history, top_k):
        if not message or not message.strip():
            return history, history, []
        answer, refs = engine.answer(message.strip(), int(top_k))
        new_history = history + [{"role": "user", "content": message.strip()}, {"role": "assistant", "content": answer}]
        return new_history, new_history, refs

    def clear_fn():
        return [], [], []

    with gr.Blocks(title="SCNU LoRA + RAG Chat") as demo:
        gr.Markdown("## 华南师范大学校园智能问答助手（LoRA + RAG）")
        gr.Markdown("左侧聊天，右侧查看检索到的官方资料片段。")
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(type="messages", height=560, label="问答")
                message = gr.Textbox(label="输入问题", placeholder="例如：校园卡丢失后怎么处理？")
                with gr.Row():
                    submit = gr.Button("发送", variant="primary")
                    clear = gr.Button("清空对话")
                gr.Examples(examples=examples, inputs=message)
            with gr.Column(scale=2):
                top_k = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="检索片段数")
                refs = gr.JSON(label="检索命中的官方资料")
        state = gr.State([])
        submit.click(chat_fn, inputs=[message, state, top_k], outputs=[chatbot, state, refs])
        message.submit(chat_fn, inputs=[message, state, top_k], outputs=[chatbot, state, refs])
        clear.click(clear_fn, outputs=[chatbot, state, refs])
    return demo


def main():
    parser = argparse.ArgumentParser(description="SCNU LoRA + RAG web UI")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7861)
    args = parser.parse_args()

    engine = RagChatEngine(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
    )
    demo = build_app(engine)
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=False)


if __name__ == "__main__":
    main()
