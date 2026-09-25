"""
Extract raw text (+ page numbers where available) from a file.
Each loader returns a list of {"text": str, "page": int | None} blocks.
"""
import re

from docx import Document
from pypdf import PdfReader


def _clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text) 
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: str) -> list[dict]:
    reader = PdfReader(path)
    blocks = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = _clean(page.extract_text() or "")
        if text:
            blocks.append({"text": text, "page": page_num})
    return blocks


def load_docx(path: str) -> list[dict]:
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = _clean("\n".join(paragraphs))
    return [{"text": text, "page": None}] if text else []


def load_txt(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = _clean(f.read())
    return [{"text": text, "page": None}] if text else []


LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_txt,
}


def load_file(path: str, suffix: str) -> list[dict]:
    loader = LOADERS.get(suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported file type: {suffix}")
    return loader(path)
