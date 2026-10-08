"""M2 — Hybrid RAG (dense + BM25 + reranker) over the Myntra catalogue.

The single biggest production move in RAG. If you ship one upgrade to naive
RAG, ship this.

WHY NAIVE (DENSE-ONLY) RAG STRUGGLES ON A PRODUCT CATALOGUE:
    Dense embeddings encode *meaning*, not *strings*. Ask for a specific brand,
    material, or model name and the dense vector lands near "similar-looking
    products" that never contain the exact word. BM25 scores literal term
    overlap, so it nails exact-word queries. Dense and BM25 fail on different
    query classes — production traffic needs both.

THE PIPELINE (see myntra_rag.py for the implementation):

    query ─┬─► Dense top-10 (semantic) ─┐
           │                            ├─► merge ─► cross-encoder rerank ─► top-5 ─► LLM
           └─► BM25  top-10 (lexical)  ─┘

    Retrieve wide and cheap from both retrievers, merge the unique candidates,
    then let the cross-encoder reranker (which scores each query+product pair
    jointly) pick the best 5 to hand to the LLM.

Run:
    python scripts/m2_hybrid.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OPENROUTER_BASE, check_openrouter, require_env  # noqa: E402
from myntra_rag import (  # noqa: E402
    MyntraHybridRetriever,
    answer_query,
    configure_settings,
)

# A few realistic shopping questions that exercise both semantic and lexical
# matching (materials, use-cases, and specific attributes).
DEMO_QUERIES = [
    "comfortable waterproof hiking shoes for women",
    "a warm wool sweater for winter",
    "lightweight running shoes with good cushioning",
]


def show_dense_vs_bm25(retriever: MyntraHybridRetriever, query: str) -> None:
    """Side-by-side: what dense returns vs what BM25 returns (before rerank)."""
    print(f"\n=== Dense vs BM25 for: {query!r} ===")
    print("Dense top-5 (semantic / cosine):")
    for n in retriever.dense.retrieve(query)[:5]:
        print(f"  [{n.score:.3f}] {n.metadata.get('name', '?')}")
    print("BM25 top-5 (lexical / TF-IDF):")
    for n in retriever.bm25.retrieve(query)[:5]:
        print(f"  [{n.score:.3f}] {n.metadata.get('name', '?')}")


def main() -> None:
    require_env()
    check_openrouter()

    from openai import OpenAI

    print("\n=== M2 — Build hybrid + rerank retriever over Myntra ===")
    configure_settings()
    retriever = MyntraHybridRetriever()
    client = OpenAI(base_url=OPENROUTER_BASE, api_key=os.environ["OPENROUTER_API_KEY"])

    # The teaching demo: dense and BM25 surface different products.
    show_dense_vs_bm25(retriever, DEMO_QUERIES[0])

    print("\n=== M2 — hybrid answers ===")
    for q in DEMO_QUERIES:
        hits = retriever.retrieve(q)
        answer = answer_query(client, q, hits)
        print(f"\nQ: {q}")
        print("Top products (after rerank):")
        for h in hits:
            print(f"  [{h['score']:.3f}] {h['name']}")
        print(f"A: {answer}")


if __name__ == "__main__":
    main()
