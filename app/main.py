"""
个人知识库助手 —— RAG + Agent 混合系统
技术栈：FastAPI + DeepSeek + sqlite3 + BM25 + DuckDuckGo
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

from app.models import ChatRequest
from app.upload import save_upload
from app.chunker import chunk_text
from app.embedder import embed_texts
from app.retriever import add_chunks, search, delete_doc, get_chunk_count
from app.agent import run_agent

# ── FastAPI ─────────────────────────────────────
app = FastAPI(title="知识库助手", version="0.2.0")

static_dir = Path(__file__).parent.parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ── DeepSeek ────────────────────────────────────
deepseek = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# ── 文档列表（内存，重启后丢失但 ChromaDB 数据还在）──
documents_store: list[dict] = []

# ── RAG Prompt 模板 ─────────────────────────────
RAG_SYSTEM_PROMPT = """你是一个知识库助手。根据以下参考资料回答用户问题。

规则：
1. 如果参考资料中有答案，基于资料回答，并在末尾标注来源。
2. 如果参考资料中只有部分答案，说明哪些能找到、哪些找不到。
3. 如果参考资料中完全没有答案，明确说"资料中没有相关信息"。
4. 不要编造资料中没有的内容。
5. 保持回答简洁、准确。"""


def build_rag_prompt(query: str, chunks: list[dict]) -> tuple[str, str]:
    """
    构建 RAG Prompt。
    返回 (system_prompt, user_message)
    """
    if not chunks:
        return (
            "你是一个知识库助手。",
            f"用户问题：{query}\n\n（当前知识库中没有相关内容，请如实告知用户。）",
        )

    # 拼接参考资料
    context_parts = []
    for i, chunk in enumerate(chunks):
        source = chunk["metadata"].get("chunk_index", i)
        context_parts.append(f"[片段 {i+1}]（来源段落 {source}）\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)

    user_message = f"""参考资料：
{context}

用户问题：{query}

请基于参考资料回答（标注来源片段编号）："""

    return (RAG_SYSTEM_PROMPT, user_message)


# ── 文档上传 ────────────────────────────────────
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传文档 → 分块 → Embedding → 存 ChromaDB"""
    if not file.filename:
        raise HTTPException(400, "文件名不能为空")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".pdf", ".txt", ".md", ".markdown"):
        raise HTTPException(400, f"不支持的文件格式：{ext}")

    try:
        # 1. 保存文件 + 解析文本
        content = await file.read()
        doc_info = save_upload(content, file.filename)
        text = doc_info["text"]

        if not text or not text.strip():
            raise ValueError("文档解析后为空，可能是扫描件或加密PDF")

        # 2. 文本分块
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        doc_info["chunks"] = len(chunks)

        # 3. Embedding + 存库
        if chunks:
            chunk_texts = [c["text"] for c in chunks]
            embeddings = embed_texts(chunk_texts)
            add_chunks(doc_info["id"], chunks, embeddings)

        documents_store.append(doc_info)
        return {
            "ok": True,
            "doc": {k: v for k, v in doc_info.items() if k != "text"},
            "total_chunks_in_db": get_chunk_count(),
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.get("/api/documents")
def list_documents():
    return {"documents": documents_store, "total_chunks": get_chunk_count()}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str):
    global documents_store
    doc = next((d for d in documents_store if d["id"] == doc_id), None)
    if not doc:
        raise HTTPException(404, "文档不存在")

    # 删本地文件
    file_path = Path(doc["path"])
    if file_path.exists():
        file_path.unlink()
    # 删 ChromaDB 中的块
    delete_doc(doc_id)
    documents_store = [d for d in documents_store if d["id"] != doc_id]
    return {"ok": True}


# ── 问答（RAG 版）────────────────────────────────
@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    RAG 问答 —— 检索文档 → 拼 Prompt → LLM 流式回答
    """

    async def generate():
        try:
            # Step 1：检索相关文档块
            retrieved = search(req.message, top_k=3)

            # Step 2：构建 RAG Prompt
            system_prompt, user_message = build_rag_prompt(req.message, retrieved)

            # Step 3：流式调用 LLM
            stream = deepseek.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *req.history,
                    {"role": "user", "content": user_message},
                ],
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield f"data: {chunk.choices[0].delta.content}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Agent 问答（RAG + 联网兜底）────────────────────
@app.post("/api/chat/agent")
async def chat_agent(req: ChatRequest):
    """
    Agent 模式：LLM 自主决定从知识库检索 或 联网搜索。
    面试亮点：自研 Agent 循环，替代 LangChain。
    """

    async def generate():
        try:
            # 构建带 RAG 上下文的用户消息
            retrieved = search(req.message, top_k=3)
            if retrieved:
                context_parts = []
                for i, chunk in enumerate(retrieved):
                    context_parts.append(f"[知识库片段{i+1}] {chunk['text'][:300]}")
                context = "\n\n".join(context_parts)
                user_msg = f"用户问题：{req.message}\n\n知识库检索结果：\n{context}\n\n请判断知识库内容能否回答问题。能回答就基于知识库回答并标注来源片段编号；不能回答就调用 search_web 工具搜索网络。"
            else:
                user_msg = f"用户问题：{req.message}\n\n（知识库为空，请调用 search_web 工具搜索网络回答）"

            # 工具定义
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "search_web",
                        "description": "搜索互联网获取最新信息。当知识库中没有答案时使用。",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "搜索关键词"}
                            },
                            "required": ["query"],
                        },
                    },
                }
            ]

            # 工具执行器
            def search_web(query: str) -> str:
                try:
                    from ddgs import DDGS
                    results = list(DDGS().text(query, max_results=3))
                    if not results:
                        return "未找到搜索结果"
                    return "\n\n".join(
                        f"[搜索{i+1}] {r['title']}\n{r['body']}"
                        for i, r in enumerate(results)
                    )
                except Exception as e:
                    return f"搜索失败：{e}。请基于已有知识回答。"

            tool_handlers = {"search_web": search_web}

            # 运行 Agent
            answer, log = run_agent(
                client=deepseek,
                model=DEEPSEEK_MODEL,
                user_message=user_msg,
                tools=tools,
                tool_handlers=tool_handlers,
                history=req.history if req.history else None,
                system_prompt="你是知识库助手。优先用知识库内容回答；知识库没有答案时，调用 search_web 搜索网络。回答要标注信息来源（知识库片段编号或搜索结果编号）。",
                max_turns=2,
            )

            # 模拟流式输出
            for char in answer:
                yield f"data: {char}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── 前端 ────────────────────────────────────────
@app.get("/")
def index():
    return FileResponse(str(static_dir / "index.html"))
