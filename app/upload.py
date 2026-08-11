"""文档解析模块 —— 支持 PDF、TXT、Markdown"""
import os
import uuid
from datetime import datetime
from pathlib import Path

UPLOAD_DIR = Path(__file__).parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


def parse_file(file_path: str, filename: str) -> str:
    """
    根据文件类型解析，返回纯文本。
    面试考点：为什么选 PyPDF2 而不是 pdfplumber？
        答：PyPDF2 更轻量、更快；pdfplumber 对复杂表格更好但更慢。
        对于知识库问答场景，PyPDF2 足够。
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        return _parse_pdf(file_path)
    elif ext == ".txt":
        return _parse_txt(file_path)
    elif ext in (".md", ".markdown"):
        return _parse_txt(file_path)  # Markdown 当作纯文本读，保留格式
    else:
        raise ValueError(f"不支持的文件格式：{ext}")


def _parse_pdf(file_path: str) -> str:
    """PDF 解析"""
    text_parts = []

    # 方案1：PyPDF2（轻量，大部分场景够用）
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    except Exception:
        pass

    # 如果 PyPDF2 提取不到文字，尝试 pdfplumber（对扫描件更好）
    if not text_parts or len("".join(text_parts).strip()) < 50:
        text_parts = []
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
        except Exception:
            raise RuntimeError("PDF 解析失败，文件可能为纯图片扫描件")

    return "\n\n".join(text_parts)


def _parse_txt(file_path: str) -> str:
    """纯文本/Markdown 解析"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def save_upload(file_content: bytes, filename: str) -> dict:
    """
    保存上传文件，返回文件元信息。
    面试考点：为什么用 uuid 做文件名？
        答：防止用户上传同名文件覆盖，防止路径注入攻击。
    """
    file_id = uuid.uuid4().hex[:12]
    safe_name = f"{file_id}_{filename}"
    file_path = UPLOAD_DIR / safe_name

    with open(file_path, "wb") as f:
        f.write(file_content)

    # 解析文本
    text = parse_file(str(file_path), filename)

    return {
        "id": file_id,
        "filename": filename,
        "path": str(file_path),
        "text": text,
        "size": len(file_content),
        "chunks": 0,  # Day 2 填真实值
        "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
