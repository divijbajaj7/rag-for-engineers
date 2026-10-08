"""Shared Myntra hybrid retrieval — used by M2 (demo) and M3 (chatbot UI).

The pipeline is deliberately simple, the way you'd ship a first production
retriever over a product catalogue:

    query ─┬─► Dense (semantic) top-10 ─┐
           │                            ├─► merge ─► cross-encoder rerank ─► top-k
           └─► BM25   (lexical)  top-10 ┘

    • Dense catches meaning ("comfy sneakers" ≈ "cushioned running shoes").
    • BM25 catches exact words (a brand name, "waterproof", "merino wool").
    • The cross-encoder reranker scores each (query, product) PAIR jointly and
      gives the final, most accurate ordering. No bi-encoder second pass —
      just retrieve wide, then let the reranker pick the best.

Both retrievers return top-10; we merge the unique candidates and rerank them
down to top-k (default 5) before handing them to the LLM.

Works with either OpenAI or OpenRouter — the provider/model/base URL come from
common.py, which auto-detects whichever API key is set.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    API_BASE,
    API_KEY,
    EMBED_MODEL,
    GENERATION_MODEL,
    RERANK_MODEL,
    extra_headers,
    myntra_nodes,
)

_WIDE_K = 10   # how many each retriever pulls (dense top-10, BM25 top-10)
_FINAL_K = 5   # how many survive the reranker and reach the LLM


def configure_settings():
    """Wire LlamaIndex's global embed model + LLM to the active provider."""
    from llama_index.core import Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.openai_like import OpenAILike

    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    Settings.llm = OpenAILike(
        model=GENERATION_MODEL,
        api_base=API_BASE,
        api_key=API_KEY,
        is_chat_model=True,
        is_function_calling_model=False,
        context_window=128000,
        max_tokens=1024,
    )


def make_client():
    """An OpenAI-compatible client pointed at the active provider."""
    from openai import OpenAI
    return OpenAI(base_url=API_BASE, api_key=API_KEY, default_headers=extra_headers())


class MyntraHybridRetriever:
    """Dense + BM25 → merge → cross-encoder rerank over the Myntra catalogue."""

    def __init__(self, wide_k: int = _WIDE_K, final_k: int = _FINAL_K):
        from llama_index.core import VectorStoreIndex
        from llama_index.core.retrievers import VectorIndexRetriever
        from llama_index.retrievers.bm25 import BM25Retriever
        from sentence_transformers import CrossEncoder

        self.wide_k = wide_k
        self.final_k = final_k

        print("Loading Myntra catalogue...")
        self.nodes = myntra_nodes()
        print(f"  {len(self.nodes)} products")

        print("Embedding + indexing (in-memory)...")
        self.index = VectorStoreIndex(self.nodes)

        self.dense = VectorIndexRetriever(index=self.index, similarity_top_k=wide_k)
        self.bm25 = BM25Retriever.from_defaults(nodes=self.nodes, similarity_top_k=wide_k)

        print(f"Loading cross-encoder reranker ({RERANK_MODEL})...")
        # First load downloads ~280MB from HuggingFace, then it's cached.
        self.reranker = CrossEncoder(RERANK_MODEL, max_length=512)
        print("Retriever ready.\n")

    def retrieve(self, query: str):
        """Return the final top-k products as a list of dicts.

        Each item: {"name", "text", "score"} (score = reranker relevance).
        """
        dense_hits = self.dense.retrieve(query)
        bm25_hits = self.bm25.retrieve(query)

        # Merge unique candidates (a product found by both still counts once).
        merged = {}
        for n in dense_hits + bm25_hits:
            merged[n.node_id] = n
        candidates = list(merged.values())
        if not candidates:
            return []

        # Cross-encoder rerank: score every (query, product) pair jointly.
        pairs = [[query, n.get_content()] for n in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: -x[0])

        out = []
        for score, n in ranked[: self.final_k]:
            out.append({
                "name": n.metadata.get("name", "?"),
                "text": n.get_content(),
                "score": float(score),
            })
        return out


def format_context(hits: list[dict]) -> str:
    """Render retrieved products into a numbered context block for the LLM."""
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"[{i}] {h['name']}\n{h['text']}")
    return "\n\n".join(lines)


def answer_query(client, query: str, hits: list[dict]) -> str:
    """Ground an answer in the retrieved products via a direct LLM call."""
    context = format_context(hits)
    prompt = (
        "You are a helpful shopping assistant for the Myntra store. Answer the "
        "customer's question using ONLY the products listed below. Recommend the "
        "most relevant items by name and explain why they fit. If nothing fits, "
        "say so honestly.\n\n"
        f"PRODUCTS:\n{context}\n\n"
        f"CUSTOMER QUESTION: {query}\n\nANSWER:"
    )
    r = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.2,
    )
    return (r.choices[0].message.content or "").strip()
