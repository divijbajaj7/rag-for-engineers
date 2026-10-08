"""M3 — Myntra chatbot UI with smart query decomposition (Streamlit).

THE ARCHITECTURE IDEA:
    A single user query often under-specifies what they want. "Benefits of
    joining this company" could mean health insurance, leave policy, learning
    budget, equity — one embedding can't point at all of them at once. The same
    is true for shopping: "something nice to wear for a winter trek" bundles
    warmth, waterproofing, and footwear into one vague sentence.

    So before retrieving, we ask the LLM to REWRITE the user's question into 3
    different, more specific sub-queries — three angles on the same intent.
    Each sub-query runs through the M2 hybrid retriever independently, we pool
    and de-duplicate all the retrieved products, and the LLM writes one combined
    answer grounded in that richer, wider context.

        user query
            │  (LLM decomposition)
            ├─► sub-query 1 ─► hybrid retrieve ─┐
            ├─► sub-query 2 ─► hybrid retrieve ─┼─► pool + dedupe ─► LLM answer
            └─► sub-query 3 ─► hybrid retrieve ─┘

    Why it helps: when your data is semantically clustered (many similar
    products, or many facets of one topic), multiple targeted queries recall
    relevant items that a single query would miss.

Run the UI:
    streamlit run scripts/m3_myntra_chatbot.py

Run once in the terminal (no UI) to sanity-check:
    python scripts/m3_myntra_chatbot.py "warm waterproof boots for a winter trek"
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import GENERATION_MODEL, OPENROUTER_BASE  # noqa: E402
from myntra_rag import (  # noqa: E402
    MyntraHybridRetriever,
    answer_query,
    configure_settings,
    format_context,
)

N_SUBQUERIES = 3

DECOMPOSE_PROMPT = (
    "You are a query-planning assistant for a Myntra shopping search engine. "
    "Rewrite the customer's request into exactly {n} distinct, more specific "
    "search queries that each capture a different angle or interpretation of "
    "what they might want. Keep each query short (a search phrase, not a "
    "sentence). Return ONLY a JSON array of {n} strings, nothing else.\n\n"
    "Customer request: {query}"
)


def make_client():
    from openai import OpenAI
    return OpenAI(base_url=OPENROUTER_BASE, api_key=os.environ["OPENROUTER_API_KEY"])


def decompose(client, query: str, n: int = N_SUBQUERIES) -> list[str]:
    """Ask the LLM to break one query into `n` targeted sub-queries."""
    r = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": DECOMPOSE_PROMPT.format(n=n, query=query)}],
        max_tokens=200,
        temperature=0.3,
    )
    raw = (r.choices[0].message.content or "").strip()
    # Strip markdown fences if the model wrapped the JSON.
    if raw.startswith("```"):
        raw = raw.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        subs = json.loads(raw)
        subs = [str(s).strip() for s in subs if str(s).strip()]
    except Exception:
        subs = []
    # Always include the original query; pad/truncate to n sub-queries.
    if not subs:
        subs = [query]
    return subs[:n]


def multi_retrieve(retriever: MyntraHybridRetriever, subqueries: list[str]):
    """Retrieve for each sub-query, then pool + de-duplicate by product name."""
    per_query = {}
    pooled: dict[str, dict] = {}
    for sq in subqueries:
        hits = retriever.retrieve(sq)
        per_query[sq] = hits
        for h in hits:
            # Keep the best (highest reranker score) copy of each product.
            if h["name"] not in pooled or h["score"] > pooled[h["name"]]["score"]:
                pooled[h["name"]] = h
    pooled_sorted = sorted(pooled.values(), key=lambda h: -h["score"])
    return per_query, pooled_sorted


def run_pipeline(client, retriever, query: str):
    """Full M3 pipeline: decompose → multi-retrieve → combined answer."""
    subqueries = decompose(client, query)
    per_query, pooled = multi_retrieve(retriever, subqueries)
    answer = answer_query(client, query, pooled)
    return subqueries, per_query, pooled, answer


# ---------------------------------------------------------------------------
# CLI fallback — `python m3_myntra_chatbot.py "your question"`
# ---------------------------------------------------------------------------
def _cli(query: str) -> None:
    from common import check_openrouter, require_env
    require_env()
    check_openrouter()
    configure_settings()
    retriever = MyntraHybridRetriever()
    client = make_client()

    subqueries, per_query, pooled, answer = run_pipeline(client, retriever, query)
    print(f"\nUser query: {query}")
    print("\nDecomposed into:")
    for s in subqueries:
        print(f"  • {s}")
    print("\nPooled products (deduped, after rerank):")
    for h in pooled:
        print(f"  [{h['score']:.3f}] {h['name']}")
    print(f"\nAnswer:\n{answer}")


# ---------------------------------------------------------------------------
# Streamlit UI — `streamlit run scripts/m3_myntra_chatbot.py`
# ---------------------------------------------------------------------------
def _streamlit() -> None:
    import streamlit as st

    st.set_page_config(page_title="Myntra RAG Chatbot (M3)", page_icon="🛍️", layout="wide")
    st.title("🛍️ Myntra RAG Chatbot")
    st.caption("M3 — smart query decomposition: your question is rewritten into "
               "3 angles, each retrieved separately, then answered together.")

    @st.cache_resource(show_spinner="Building the Myntra hybrid retriever (one-time)...")
    def _load():
        from common import require_env
        require_env()
        configure_settings()
        return MyntraHybridRetriever(), make_client()

    retriever, client = _load()

    with st.sidebar:
        st.header("How it works")
        st.markdown(
            "1. **Decompose** — the LLM rewrites your query into 3 sub-queries.\n"
            "2. **Retrieve** — each sub-query runs through dense + BM25 + reranker.\n"
            "3. **Pool** — products are merged and de-duplicated.\n"
            "4. **Answer** — the LLM responds grounded in the pooled products."
        )
        st.markdown("---")
        st.markdown("Try: *warm waterproof boots for a winter trek*, "
                    "*a gift for someone who loves running*, "
                    "*smart casual outfit for the office*.")

    query = st.chat_input("Ask about Myntra products...")
    if not query:
        st.info("Type a shopping question below to see decomposition + retrieval in action.")
        return

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Decomposing your query..."):
            subqueries = decompose(client, query)
        st.markdown("**🔎 Decomposed into 3 search angles:**")
        for s in subqueries:
            st.markdown(f"- `{s}`")

        with st.spinner("Retrieving products for each sub-query..."):
            per_query, pooled = multi_retrieve(retriever, subqueries)

        with st.expander("Retrieved products per sub-query"):
            for sq, hits in per_query.items():
                st.markdown(f"**{sq}**")
                for h in hits:
                    st.markdown(f"- `{h['score']:.2f}` {h['name']}")

        st.markdown("**🧩 Pooled, de-duplicated products:**")
        for h in pooled:
            st.markdown(f"- `{h['score']:.2f}` **{h['name']}**")

        with st.spinner("Writing the grounded answer..."):
            answer = answer_query(client, query, pooled)
        st.markdown("**💬 Answer:**")
        st.write(answer)


def _running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    if _running_in_streamlit():
        _streamlit()
    else:
        q = " ".join(sys.argv[1:]) or "warm waterproof boots for a winter trek"
        _cli(q)
else:
    # `streamlit run` imports the module (name != "__main__").
    if _running_in_streamlit():
        _streamlit()
