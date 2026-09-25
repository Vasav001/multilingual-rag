"""
chunking strategy: recursive splitting on paragraph -> line -> word
boundaries, sized in tokens (tiktoken) with a configurable overlap. Works
fine on non-English/Unicode text since we never split mid-character.
"""
import tiktoken

from config import settings

_ENC = tiktoken.get_encoding("cl100k_base")

# Try these separators in order; whichever keeps chunks under budget wins.
_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_on(text: str, sep: str) -> list[str]:
    parts = text.split(sep)
    return [p + sep if i < len(parts) - 1 else p for i, p in enumerate(parts)]


def _token_len(text: str) -> int:
    return len(_ENC.encode(text))


def _recursive_split(text: str, separators: list[str], max_tokens: int) -> list[str]:
    if _token_len(text) <= max_tokens:
        return [text] if text.strip() else []
    if not separators:
        # Fall back to a hard token-based cut.
        tokens = _ENC.encode(text)
        return [_ENC.decode(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]

    sep, rest = separators[0], separators[1:]
    pieces = _split_on(text, sep)
    out: list[str] = []
    for piece in pieces:
        out.extend(_recursive_split(piece, rest, max_tokens) if _token_len(piece) > max_tokens else ([piece] if piece.strip() else []))
    return out


def _merge_with_overlap(pieces: list[str], max_tokens: int, overlap_tokens: int) -> list[str]:
    """Greedily pack small pieces back together up to max_tokens, carrying
    `overlap_tokens` worth of trailing text into the next chunk."""
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = current + piece
        if _token_len(candidate) <= max_tokens:
            current = candidate
            continue
        if current.strip():
            chunks.append(current.strip())
        # start next chunk with overlap from the end of the previous one
        overlap_text = ""
        if overlap_tokens > 0 and current:
            tokens = _ENC.encode(current)
            overlap_text = _ENC.decode(tokens[-overlap_tokens:])
        current = overlap_text + piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_text(text: str) -> list[str]:
    max_tokens = settings.CHUNK_SIZE_TOKENS
    overlap = settings.CHUNK_OVERLAP_TOKENS
    pieces = _recursive_split(text, _SEPARATORS, max_tokens)
    return _merge_with_overlap(pieces, max_tokens, overlap)
