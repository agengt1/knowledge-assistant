"""
文本分块模块

面试必问：chunk_size 和 overlap 怎么选的？
- chunk_size=500：中文约500字是一个语义相对完整的段落。太小→信息碎片化；太大→检索精度下降。
- overlap=50：保证相邻块之间有重叠，避免一个完整句子被切断在边界。
- 实际我试了 200/500/800，500 在检索召回率和答案完整性上最平衡。
"""


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    separator: str = "\n\n",
) -> list[dict]:
    """
    将文本按段落分割成固定大小的块。

    返回格式：
    [
        {"index": 0, "text": "第一段内容...", "char_start": 0, "char_end": 498},
        {"index": 1, "text": "第二段内容...", "char_start": 448, "char_end": 946},
        ...
    ]
    """
    if not text or not text.strip():
        return []

    # Step 1：按段落粗分（保留自然边界）
    paragraphs = text.split(separator)

    chunks = []
    current_chunk = ""
    char_start = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Step 2：段落太长，内部滑动窗口分块
        if len(para) > chunk_size:
            # 先保存当前积累的块
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""

            # 滑动窗口切分长段落
            start = 0
            while start < len(para):
                end = min(start + chunk_size, len(para))
                chunk_text_segment = para[start:end]
                chunks.append(chunk_text_segment)
                if end >= len(para):
                    break  # 切完了
                start = end - overlap
            continue

        # Step 3：段落不长，拼接到当前块
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += para + "\n"
        else:
            # 当前块满了，保存并开始新块
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n"

    # 最后一块
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Step 4：返回带元信息的块列表
    result = []
    char_pos = 0
    for i, chunk in enumerate(chunks):
        result.append({
            "index": i,
            "text": chunk,
            "char_start": char_pos,
            "char_end": char_pos + len(chunk),
        })
        char_pos += len(chunk) - overlap  # 实际字符偏移（考虑重叠）

    return result
