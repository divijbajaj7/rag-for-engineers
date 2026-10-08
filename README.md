# RAG for Engineers — 3 modules that build toward an industry RAG

A hands-on walkthrough of how three ideas stack up into a production-grade
Retrieval-Augmented Generation (RAG) system. Each module is a standalone,
end-to-end Python script you can run and read top-to-bottom.

| Module | What it shows | Corpus |
|---|---|---|
| **M0 — Naive RAG** | The strawman (`chunk → embed → top-k → LLM`) and **where it fails**, measured against a golden dataset with RAGAS (faithfulness + answer relevancy). | Kubernetes docs |
| **M2 — Hybrid RAG** | Dense (semantic) + BM25 (lexical) retrieval, merged and **reranked** by a cross-encoder before the LLM. The single biggest production upgrade. | Myntra product catalogue |
| **M3 — Myntra chatbot + smart query decomposition** | A simple **Streamlit UI** where one user query is rewritten by the LLM into 3 sub-queries, each retrieved separately, pooled, and answered together. | Myntra product catalogue |

## The story

1. **M0 — show the baseline break.** Naive dense-only RAG looks fine in a demo
   but fails on precise questions. We prove it: 5 golden questions scored with
   RAGAS. The headline failure is `--max-pods` — a literal token that dense
   embeddings smear into generic "pod limit" semantics, so the answer never
   contains the real default (110).

2. **M2 — fix retrieval with hybrid + rerank.** Dense catches *meaning*, BM25
   catches *exact words*. Run both (top-10 each) over the Myntra catalogue,
   merge the candidates, and let a cross-encoder reranker pick the final 5 for
   the LLM. No bi-encoder second pass — just retrieve wide, then rerank.

3. **M3 — decompose the query, then retrieve.** One vague query can't point at
   everything a shopper means. The LLM rewrites it into 3 targeted sub-queries,
   each goes through the M2 hybrid retriever, results are pooled and
   de-duplicated, and the LLM writes one grounded answer. Wrapped in a
   Streamlit chatbot you can play with.

Together they answer: *how do these pieces combine to build a real-world RAG?*

## Works with OpenAI **or** OpenRouter

You only need **one** API key. The code auto-detects which one you set:

- **OpenAI key** → uses OpenAI directly (`gpt-4o-mini` by default).
- **OpenRouter key** → uses OpenRouter (`claude-haiku-4.5` + `gemini-2.5-flash`,
  plus free-tier models).

Embeddings (`bge-small-en-v1.5`) and the reranker (`bge-reranker-base`) run
locally on your machine, so they never need an API key.

## Setup

```bash
cd code
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then paste ONE key (OpenAI or OpenRouter)
python download_data.py         # fetch the K8s docs used by M0
```

Edit `code/.env` and set exactly one of:

```
OPENAI_API_KEY=sk-...            # https://platform.openai.com/api-keys
# or
OPENROUTER_API_KEY=sk-or-v1-...  # https://openrouter.ai/keys
```

If you set both, add `LLM_PROVIDER=openai` (or `openrouter`) to pick one.

## Run

```bash
cd code/scripts

python m0_naive.py      # naive RAG + RAGAS eval on 5 golden questions
python m2_hybrid.py     # hybrid (dense+BM25) + reranker on Myntra
streamlit run m3_myntra_chatbot.py   # Myntra chatbot UI with query decomposition
python show_results.py  # print the scoreboard anytime
```

M3 also runs headless for a quick check:

```bash
python m3_myntra_chatbot.py "warm waterproof boots for a winter trek"
```

## Files

| File | What |
|---|---|
| `scripts/common.py` | Provider auto-detection, config, data loaders, RAGAS eval helper |
| `scripts/m0_naive.py` | Naive RAG + baseline eval (K8s) |
| `scripts/myntra_rag.py` | Shared Myntra hybrid retriever (dense + BM25 + reranker) |
| `scripts/m2_hybrid.py` | Hybrid RAG demo over the Myntra catalogue |
| `scripts/m3_myntra_chatbot.py` | Streamlit chatbot + query decomposition |
| `scripts/show_results.py` | Print the RAGAS scoreboard |
| `goldset.json` | Golden questions for M0's RAGAS eval |
| `Myntra_300_prod_catalogue.csv` | ~299 products (name + description) for M2/M3 |

## Models (defaults — override via env in `.env`)

| Role | OpenAI | OpenRouter |
|---|---|---|
| Generation | `gpt-4o-mini` | `anthropic/claude-haiku-4.5` |
| RAGAS judge | `gpt-4o-mini` | `google/gemini-2.5-flash` |
| Embeddings | `BAAI/bge-small-en-v1.5` (local) | same |
| Reranker | `BAAI/bge-reranker-base` (local) | same |

> Tip: for the cleanest RAGAS scores, use a *different* model to judge than to
> generate (OpenRouter does this by default). On OpenAI you can set
> `JUDGE_MODEL=gpt-4o` to decorrelate the judge from the generator.
