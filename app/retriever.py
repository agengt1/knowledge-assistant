"""
检索模块 —— 混合检索（语义 + 关键词 + RRF 融合）

面试必考：为什么混合检索？
- 语义检索（Embedding）：能理解同义词（"苹果"≈"iPhone"），但专有名词弱
- 关键词检索（BM25）：精确匹配专有名词（"DeepSeek"），但不理解语义
- RRF 融合：取两者交集，互补长短。学术上验证过比单路检索好 15-30%

面试话术：
"我用混合检索替代了单纯的语义检索——BM25 做关键词召回，
余弦相似度做语义召回，然后用 RRF（Reciprocal Rank Fusion）
合并排序。k=60 是学术界通用的平滑参数。"
"""

import json
import math
import sqlite3
from pathlib import Path

from rank_bm25 import BM25Okapi
from app.embedder import embed_query

DB_PATH = Path(__file__).parent.parent / "vectors.db"

# BM25 索引缓存（全局，增删时重建）
_bm25_index: BM25Okapi | None = None
_bm25_chunks: list[dict] = []

RRF_K = 60  # RRF 平滑参数


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            doc_id TEXT,
            chunk_index INTEGER,
            text TEXT,
            embedding TEXT,
            char_start INTEGER,
            char_end INTEGER
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON chunks(doc_id)")
    conn.commit()
    return conn


def _load_all_chunks() -> list[dict]:
    """从数据库加载所有块"""
    conn = _get_db()
    rows = conn.execute("SELECT * FROM chunks").fetchall()
    conn.close()
    return [
        {
            "id": r[0],
            "doc_id": r[1],
            "chunk_index": r[2],
            "text": r[3],
            "embedding": r[4],
            "char_start": r[5],
            "char_end": r[6],
        }
        for r in rows
    ]


def _rebuild_bm25():
    """重建 BM25 索引（文档变更后调用）"""
    global _bm25_index, _bm25_chunks
    _bm25_chunks = _load_all_chunks()
    if _bm25_chunks:
        # 分词：中文按字 + 英文按空格
        tokenized = [_tokenize(c["text"]) for c in _bm25_chunks]
        _bm25_index = BM25Okapi(tokenized)
    else:
        _bm25_index = None


def _tokenize(text: str) -> list[str]:
    """简单分词：中文单字 + 英文单词"""
    tokens = []
    word = ""
    for ch in text:
        if ch.isalpha():
            word += ch.lower()
        else:
            if word:
                tokens.append(word)
                word = ""
            if ch.strip():
                tokens.append(ch)  # 中文字符直接作为一个 token
    if word:
        tokens.append(word)
    return tokens


def add_chunks(doc_id: str, chunks: list[dict], embeddings: list[list[float]]):
    conn = _get_db()
    rows = []
    for chunk, emb in zip(chunks, embeddings):
        rows.append((
            f"{doc_id}_{chunk['index']}",
            doc_id,
            chunk["index"],
            chunk["text"],
            json.dumps(emb),
            chunk["char_start"],
            chunk["char_end"],
        ))
    conn.executemany(
        "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)", rows
    )
    conn.commit()
    conn.close()
    _rebuild_bm25()  # 重建 BM25 索引


def delete_doc(doc_id: str):
    conn = _get_db()
    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
    conn.commit()
    conn.close()
    _rebuild_bm25()


def get_chunk_count() -> int:
    conn = _get_db()
    count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.close()
    return count


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """语义检索：余弦相似度"""
    chunks = _load_all_chunks()
    if not chunks:
        return []

    query_emb = embed_query(query)
    scored = []
    for c in chunks:
        emb = json.loads(c["embedding"])
        score = _cosine_similarity(query_emb, emb)
        scored.append({
            "id": c["id"],
            "text": c["text"],
            "metadata": {
                "doc_id": c["doc_id"],
                "chunk_index": c["chunk_index"],
                "char_start": c["char_start"],
                "char_end": c["char_end"],
            },
            "score": round(score, 4),
            "source": "semantic",
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def _keyword_search(query: str, top_k: int = 10) -> list[dict]:
    """关键词检索：BM25"""
    global _bm25_index, _bm25_chunks

    if not _bm25_index or not _bm25_chunks:
        return []

    tokens = _tokenize(query)
    scores = _bm25_index.get_scores(tokens)

    # 归一化 BM25 分数到 [0, 1]
    max_score = max(scores) if max(scores) > 0 else 1.0

    scored = []
    for i, c in enumerate(_bm25_chunks):
        scored.append({
            "id": c["id"],
            "text": c["text"],
            "metadata": {
                "doc_id": c["doc_id"],
                "chunk_index": c["chunk_index"],
                "char_start": c["char_start"],
                "char_end": c["char_end"],
            },
            "score": round(scores[i] / max_score, 4),
            "source": "keyword",
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def _rrf_fusion(
    list_a: list[dict], list_b: list[dict], k: int = RRF_K, top_k: int = 5
) -> list[dict]:
    """
    Reciprocal Rank Fusion —— 融合两个排序列表。

    公式：RRF_score(d) = sum( 1 / (k + rank_i(d)) )
    其中 rank_i 是文档 d 在第 i 个列表中的排名（从 1 开始）。

    面试考点：为什么不用加权求和？
    - 两个列表的分数量纲不同（余弦相似度 vs BM25），直接加权要调参
    - RRF 只关心排名，天然跨模态，不需要归一化
    """
    scores = {}  # id -> RRF 分数
    docs = {}    # id -> 文档对象

    for lst in (list_a, list_b):
        for rank, doc in enumerate(lst, start=1):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)
            if doc_id not in docs:
                docs[doc_id] = doc

    # 按 RRF 分数排序
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    result = []
    for doc_id, rrf_score in ranked[:top_k]:
        doc = docs[doc_id].copy()
        doc["score"] = round(rrf_score, 4)
        doc["rrf_score"] = round(rrf_score, 4)
        doc["source"] = "hybrid"  # RRF 融合结果
        result.append(doc)

    return result


def search(query: str, top_k: int = 5) -> list[dict]:
    """
    混合检索入口：语义 + 关键词 → RRF 融合。

    返回 top_k 个最相关的结果。
    """
    if get_chunk_count() == 0:
        return []

    # 双路召回
    semantic_results = _semantic_search(query, top_k=10)
    keyword_results = _keyword_search(query, top_k=10)

    # RRF 融合
    return _rrf_fusion(semantic_results, keyword_results, top_k=top_k)
