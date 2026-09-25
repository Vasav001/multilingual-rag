"""
Ingestion: traverse RAW_DOCS_DIR recursively, extract text per
file type, chunk, embed with Gemini, and upsert into Qdrant
collection. Files already ingested are skipped. Checked thru content hash.
"""
import os
from pathlib import Path

from config import settings
from ingestion.chunking import chunk_text
from ingestion.loaders import load_file
from retrieval.embedder import embed_documents
from retrieval.language import detect_language
from retrieval.qdrant_store import document_already_ingested, ensure_collection, file_hash, upsert_chunks

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".docx"}
EMBED_BATCH_SIZE = 32


def discover_files(root: str) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(p for p in root_path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)


def _batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def ingest_file(path: Path, log=print) -> dict:
    document_id = file_hash(str(path))
    if document_already_ingested(document_id):
        log(f"skip (already ingested): {path.name}")
        return {"filename": path.name, "status": "skipped", "chunks": 0}

    blocks = load_file(str(path), path.suffix)
    full_text = "\n".join(b["text"] for b in blocks)
    if not full_text.strip():
        log(f"skip (no extractable text): {path.name}")
        return {"filename": path.name, "status": "empty", "chunks": 0}

    language = detect_language(full_text[:2000])

    # Chunk each extracted block separately so page numbers stay attached,
    # then flatten into a single ordered chunk list for this document.
    all_chunks: list[dict] = []
    for block in blocks:
        for piece in chunk_text(block["text"]):
            all_chunks.append({
                "text": piece,
                "page": block.get("page"),
                "source": str(path),
                "chunk_index": len(all_chunks),
            })

    if not all_chunks:
        return {"filename": path.name, "status": "empty", "chunks": 0}

    for batch in _batched(all_chunks, EMBED_BATCH_SIZE):
        vectors = embed_documents([c["text"] for c in batch])
        upsert_chunks(
            document_id=document_id,
            filename=path.name,
            file_type=path.suffix.lower().lstrip("."),
            language=language,
            chunks=batch,
            vectors=vectors,
        )

    log(f"ingested: {path.name} ({len(all_chunks)} chunks, language={language})")
    return {"filename": path.name, "status": "ingested", "chunks": len(all_chunks), "language": language}


def run_ingestion(root: str | None = None, log=print) -> list[dict]:
    ensure_collection()
    root = root or settings.RAW_DOCS_DIR
    files = discover_files(root)
    if not files:
        log(f"No supported files found under {root}")
        return []
    return [ingest_file(p, log=log) for p in files]


if __name__ == "__main__":
    run_ingestion()
