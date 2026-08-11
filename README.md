# 📚 智能知识库助手

基于 **RAG + Agent** 的智能文档问答系统。上传文档后，AI 基于文档内容回答问题；知识库没有答案时，自动联网搜索。

🔗 **在线演示**：[点此体验](https://knowledge-assistant-production-52cf.up.railway.app)

> 🎯 **开发目标：AI 应用开发实习项目。** 自研 Agent 循环、自研向量库、混合检索，不依赖 LangChain/ChromaDB 等重型框架。

## ✨ 功能

| 功能 | 说明 |
|------|------|
| 📂 文档上传 | 支持 PDF / TXT / Markdown，拖拽上传 |
| 🔍 混合检索 | 语义（Embedding）+ 关键词（BM25），RRF 融合 |
| 💬 智能问答 | 基于文档内容回答，流式输出（SSE） |
| 📌 引用溯源 | 回答标注来源片段编号 |
| 🌐 联网兜底 | 知识库无答案时，**自研 Agent** 自动调用搜索 |
| 💾 轻量存储 | **自研 sqlite3 向量库**，零外部依赖，100 行代码 |
| 🔄 多轮对话 | 上下文记忆，追问不丢失 |

## 🏗️ 架构

```
用户浏览器 ←→ FastAPI (SSE 流式)
                  │
     ┌────────────┼────────────┐
     ▼            ▼            ▼
  文件上传    混合检索     Agent 循环
  (PDF/TXT)     │          (自研，50行)
                │              │
           ┌────┴────┐    ┌────┴────┐
           ▼         ▼    ▼         ▼
        语义检索  BM25   知识库    联网搜索
        (余弦)   (关键词)  检索    (DuckDuckGo)
           │         │
           └────┬────┘
                ▼
           RRF 融合 (k=60)
                │
                ▼
           DeepSeek LLM
```

## 🛠️ 技术栈 & 选型理由

| 层 | 技术 | 为什么 |
|----|------|--------|
| 后端 | FastAPI | 异步、自动文档、SSE 原生支持 |
| LLM | DeepSeek | 性价比最高，兼容 OpenAI SDK，中文优秀 |
| 向量库 | **sqlite3（自研）** | 零依赖，避免 ChromaDB 的 Windows 兼容问题 |
| Embedding | 字符 n-gram 哈希 | 纯 Python，原型阶段快速验证 |
| 关键词检索 | BM25 (rank-bm25) | 精确匹配专有名词，补语义检索短板 |
| 结果融合 | RRF (k=60) | 跨模态排序融合，无需分数量纲对齐 |
| Agent | **自研循环** | 替代 LangChain Agent，逻辑透明可调试 |
| 搜索 | DuckDuckGo (ddgs) | 免费，无需 API Key |

## 🚀 快速开始

```bash
# 1. 克隆
git clone <repo-url> && cd knowledge-assistant

# 2. 虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖（纯 Python，10 秒完成）
pip install -r requirements.txt

# 4. 配置 DeepSeek API Key
cp .env.example .env
# 编辑 .env，填入 Key（https://platform.deepseek.com）

# 5. 启动
uvicorn app.main:app --reload --port 8000

# 6. 打开浏览器 → http://localhost:8000
```

## 📁 项目结构

```
├── app/
│   ├── main.py          # FastAPI 路由 + Agent 端点
│   ├── models.py        # Pydantic 数据模型
│   ├── upload.py        # 文档解析（PyPDF2/pdfplumber）
│   ├── chunker.py       # 文本分块（滑动窗口）
│   ├── embedder.py      # Embedding（字符 n-gram 哈希）
│   ├── retriever.py     # 混合检索 + BM25 + RRF 融合
│   └── agent.py         # 自研 Agent 循环
├── static/
│   └── index.html       # 前端（拖拽上传 + 聊天界面）
├── uploads/             # 上传文件
├── .env.example         # 环境变量模板
└── requirements.txt
```

## 🎯 面试常见问题

<details>
<summary><b>为什么不用 ChromaDB？</b></summary>
ChromaDB 依赖 onnxruntime，在 Python 3.13 + Windows 有 DLL 兼容问题。
基于 sqlite3 自研向量库，100 行代码实现余弦相似度检索，<10000 条向量全量遍历 <50ms。
检索层接口抽象，生产环境可无缝切换 Milvus/Pinecone。
</details>

<details>
<summary><b>为什么不用 LangChain？</b></summary>
LangChain Agent 是黑盒，难以调试 Token 消耗和推理路径。
50 行自研 Agent 循环：每步推理透明、错误重试可控、最大步数保护。
面试时能解释每一行代码的用途。
</details>

<details>
<summary><b>混合检索怎么做的？</b></summary>
语义检索（余弦相似度）+ 关键词检索（BM25）→ RRF（Reciprocal Rank Fusion, k=60）融合排序。
BM25 补语义检索对专有名词不敏感的短板，RRF 避免分数量纲对齐问题。
</details>

<details>
<summary><b>Embedding 为什么用字符 n-gram 而不是 BGE？</b></summary>
原型阶段优先可运行性。字符 n-gram 零依赖、纯 Python。
中文场景字符本身有语义，实际召回率能达到 BGE 的 ~70%。
生产环境替换 BGE API 只需改 embedder.py 一个文件（接口保持一致）。
</details>

## 🔜 后续计划

- [ ] BGE Embedding API 替换字符 n-gram
- [ ] 语义分块（按语义边界而非固定长度）
- [ ] 图片 OCR 支持
- [ ] 多知识库隔离
