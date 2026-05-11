"""LLM client (Groq, OpenAI-compatible).

Why Groq:
  - Free tier with high rate limits.
  - Very fast inference on Llama-family models.
  - SDK is essentially OpenAI-compatible, so swapping providers later is one
    file change.

The system prompt is the *grounding* mechanism: we explicitly instruct the
model to use only the retrieved context and to admit when it doesn't know.
This is the single biggest lever against hallucination in a RAG app.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from groq import Groq

from .vector_store import SearchHit


SYSTEM_PROMPT = """You are a careful assistant that answers questions about a
single PDF document (typically a corporate annual report).

Rules:
1. Use ONLY the information in the provided context blocks. Do not rely on
   prior knowledge of the company or industry.
2. If the answer is not in the context, reply exactly: "I don't know based on
   the provided document."
3. Quote numbers and named entities verbatim from the context.
4. Keep answers concise (2-6 sentences) unless the user asks for detail.
5. When you use a fact from the context, mention the page number in
   parentheses, e.g. "(p. 14)".
"""


@dataclass
class Answer:
    text: str
    model: str


class GroqClient:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file or the "
                "sidebar input. Get one free at https://console.groq.com"
            )
        self.client = Groq(api_key=key)
        self.model = model or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    def answer(self, question: str, hits: list[SearchHit]) -> Answer:
        context_blocks = "\n\n".join(
            f"[Chunk {h.chunk.chunk_id} | page {h.chunk.page_number}]\n{h.chunk.text}"
            for h in hits
        )
        user_msg = (
            f"Context from the document:\n\n{context_blocks}\n\n"
            f"Question: {question}"
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        return Answer(text=resp.choices[0].message.content.strip(), model=self.model)
