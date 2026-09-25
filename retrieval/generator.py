"""
Output generation via Groq. 
Answers based on context only. Language aware
"""
from groq import Groq

from config import settings
from retrieval.language import language_name

_client = Groq(api_key=settings.GROQ_API_KEY)

SYSTEM_PROMPT = """You are a precise research assistant answering questions using ONLY the \
provided document excerpts (context). Rules:
1. Base your answer strictly on the context. Do not use outside knowledge and do not guess.
2. If the context does not contain enough information to answer, clearly say so in the \
requested output language (e.g. that the answer was not found in the provided documents). \
Do not make anything up.
3. Write your entire answer in the requested output language, even if the context or the \
question is in a different language.
4. Be concise and direct. You may cite the source filename/page in parentheses, e.g. (report.pdf, p.3), \
when useful. Never use any other citation marker format (no bracketed reference IDs, no \
footnote-style symbols)."""


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        loc = f", page {c['page']}" if c.get("page") else ""
        parts.append(f"[{i}] Source: {c['filename']}{loc}\n{c['text']}")
    return "\n\n".join(parts)


def generate_answer(query: str, output_lang_code: str, chunks: list[dict]) -> str:
    context = _format_context(chunks)
    lang_name = language_name(output_lang_code)
    user_prompt = (
        f"Requested output language: {lang_name}\n\n"
        f"Context:\n{context if context else '(no relevant context retrieved)'}\n\n"
        f"Question: {query}\n\n"
        f"Answer in {lang_name}."
    )
    completion = _client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return completion.choices[0].message.content
