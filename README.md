# 华南师范大学校园智能问答助手

基于 Qwen2.5-1.5B-Instruct、LoRA 微调与 RAG 检索增强构建的课程大作业项目，面向华南师范大学学生常见办事场景，支持成绩查询、校园卡挂失、图书馆续借、奖助通知、宿舍报修等校园问答。

## 项目目标

本项目围绕“华南师范大学校园智能问答助手”展开，完成了三组实验对比：

- 基础模型：Qwen2.5-1.5B-Instruct
- LoRA 微调模型：在 200 条华师校园问答数据上进行监督微调
- LoRA + RAG：在 LoRA 基础上接入华师官方资料知识库进行检索增强

目标是比较三种方案在校园问答场景中的回答具体性、可执行性与可追溯性差异。

## 项目结构

- `build_scnu_dataset.py`：生成校园问答训练数据
- `check_scnu_dataset.py`：训练数据质检
- `build_scnu_rag_corpus.py`：整理华师官方资料库
- `check_scnu_rag_corpus.py`：RAG 语料质检
- `build_scnu_rag_index.py`：构建检索索引
- `query_scnu_rag_index.py`：命令行检索测试
- `rag_chat_demo.py`：命令行问答演示
- `rag_webui.py`：Gradio 网页问答界面
- `test_scnu_rag_retrieval.py`：RAG 检索测试脚本
- `generated_scnu_data/`：训练数据与相关产物
- `generated_scnu_rag/`：RAG 语料与索引数据
- `test result images/`：基础模型、LoRA、LoRA+RAG 的实验截图
- `实验结果对比表.md`：实验结果整理
- `模式2完整报告初稿.md`：课程报告初稿

## 数据与知识库

### 微调数据

- 数据格式：Alpaca
- 数据量：200 条
- 主题覆盖：教务学籍、考试毕业、校园卡网络、宿舍后勤、奖助、图书馆、就业、心理安全、校区生活等

### RAG 知识库

- 资料来源：华南师范大学官方站点
- 文档数量：22 篇
- 核心主题：教务、图书馆、校园卡与网络、奖助、宿舍后勤
- 检索切片：160 个知识片段
- 实际后端：字符级 TF-IDF
- 原计划方案：BGE + FAISS（因实例网络限制未落地）

## 模型与训练设置

- 基础模型：Qwen2.5-1.5B-Instruct
- 微调方法：LoRA SFT
- 训练轮数：3 epochs
- 学习率：5e-5
- LoRA rank / alpha：8 / 16
- target modules：`q_proj,v_proj`
- cutoff length：512
- batch size：4
- gradient accumulation：4
- precision：BF16
- 训练设备：RTX 4090D 24GB
- 训练主体耗时：约 27 秒

## 实验结论

三组实验表明：

- 基础模型能够回答常见校园问题，但内容偏通用，缺少华师专属入口和办理路径
- LoRA 微调后，回答结构更稳定，校园办事语气更自然
- LoRA + RAG 效果最好，能够结合华师官方资料给出系统名称、网址、电话、办理步骤和来源依据
- 对于知识库中没有明确提供的实时信息，系统会倾向保守回答，避免随意编造

## 运行方式

### 1. 构建训练数据

```bash
python build_scnu_dataset.py
python check_scnu_dataset.py
```

### 2. 构建 RAG 语料与索引

```bash
python build_scnu_rag_corpus.py
python check_scnu_rag_corpus.py
python build_scnu_rag_index.py
```

### 3. 启动 LoRA + RAG 网页问答

```bash
python rag_webui.py --server-name 0.0.0.0 --server-port 6006
```

## 实验截图说明

`test result images/` 中保存了三组模型的对比截图：

- `base model/`
- `Lora/`
- `Lora+RAG/`

可用于报告中的测试与评估章节。

## 仓库说明

本仓库上传的是课程作业的代码、数据构建脚本、知识库整理结果、实验截图与报告材料，不包含基础模型权重与 LoRA 权重文件。
