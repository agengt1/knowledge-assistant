"""Pydantic 模型定义 —— 所有接口的请求/响应格式"""
from pydantic import BaseModel


class ChatRequest(BaseModel):
    """POST /chat 的请求体"""
    message: str
    history: list[dict] = []       # [{"role": "user", "content": "..."}, ...]


class ChatResponse(BaseModel):
    """非流式回答"""
    answer: str
    sources: list[dict] = []       # [{"file": "xxx.pdf", "chunk": 3, "text": "..."}]


class DocInfo(BaseModel):
    """文档信息"""
    id: str
    filename: str
    size: int
    chunks: int
    uploaded_at: str
