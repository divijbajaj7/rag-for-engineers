"""M0 — Naive RAG + baseline eval. Watch it fail on the golden dataset.

This is the strawman every RAG tutorial stops at:
    chunk → embed → cosine top-k → LLM

The point of M0 is NOT to ship. It is to establish a *measurable* baseline so
that M2 (hybrid) and M3 (decomposition) can prove they move a number.

We run naive RAG over the Kubernetes docs and score it against a golden
dataset with RAGAS on two metrics:
    • faithfulness     — are the answer's claims grounded in retrieved chunks?
                         (catches hallucination)
    • answer_relevancy — does the answer actually address the question asked?
                         (catches "answered a related but different question")

The headline failure (q01): "What is the default value of the --max-pods flag?"
`--max-pods` is a literal token. Dense embeddings smear it into generic
"pod limit / pod scheduling" semantics, so naive RAG retrieves topically-related
chunks that never contain the actual default (110). M2's BM25 lexical match
fixes exactly this.

Run:
    python scripts/m0_naive.py
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    API_BASE,
    API_KEY,
    DATA_DIR,
    EMBED_MODEL,
    GENERATION_MODEL,
    TEST_QUERIES,
    check_provider,
    evaluate_async,
    print_scoreboard,
    require_env,
)


def build_naive_engine():
    """Build the naive pipeline: chunk → embed → in-memory store → query engine."""
    from llama_index.core import (
        Settings,
        SimpleDirectoryReader,
        VectorStoreIndex,
    )
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.openai_like import OpenAILike

    # bge-small-en-v1.5: 384-dim, free, fast. Small index, fast cosine search.
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    # OpenAILike points LlamaIndex at any OpenAI-compatible endpoint
    # (OpenAI or OpenRouter — resolved in common.py).
    Settings.llm = OpenAILike(
        model=GENERATION_MODEL,
        api_base=API_BASE,
        api_key=API_KEY,
        is_chat_model=True,
        is_function_calling_model=False,
        context_window=128000,
        max_tokens=1024,
    )

    # 512/50 is the standard default: precise enough to retrieve, large enough
    # that semantic units survive. Overlap bridges chunk boundaries.
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    print(f"Loading docs from {DATA_DIR}...")
    docs = SimpleDirectoryReader(
        str(DATA_DIR), recursive=True, required_exts=[".md"]
    ).load_data()
    print(f"  loaded {len(docs)} markdown files")

    # from_documents chunks, embeds, and indexes into the default in-memory store.
    index = VectorStoreIndex.from_documents(docs)
    print("  index built (in-memory)")

    # Top-k=5: retrieve the 5 most similar chunks, stuff them into the prompt.
    return index, index.as_query_engine(similarity_top_k=5)


async def main() -> None:
    require_env()
    check_provider()

    print("\n=== M0 — Build naive RAG ===")
    _, engine = build_naive_engine()

    # Run the canonical queries and watch the failure modes.
    print("\n=== M0 — test queries (read the answers, they're wrong/partial) ===")
    for q in TEST_QUERIES:
        print(f"\nQ: {q}")
        r = engine.query(q)
        ans = str(r)
        print(f"A: {ans[:500]}{'...' if len(ans) > 500 else ''}")
        srcs = [n.metadata.get("file_name", "?") for n in r.source_nodes[:3]]
        print(f"   sources: {srcs}")

    # evaluate_async expects a callable returning {"answer", "contexts"}.
    def naive_query(q: str) -> dict:
        r = engine.query(q)
        return {"answer": str(r), "contexts": [n.get_content() for n in r.source_nodes]}

    # 5 golden cases — enough to expose the failure, cheap to run live in class.
    await evaluate_async(naive_query, "M0_naive", sample=5)
    print_scoreboard()


if __name__ == "__main__":
    asyncio.run(main())
