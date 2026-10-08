"""Shared utilities for the M0 / M2 / M3 modules.

Single source of truth for:
  • Provider selection — works with EITHER an OpenAI key OR an OpenRouter key
  • Model selection (swap models in one place)
  • Data loaders (K8s docs for M0, the Myntra catalogue for M2/M3)
  • The RAGAS eval helper used by M0 (faithfulness + answer_relevancy)
  • Result persistence (scripts/eval_results.json)

You don't run this file directly — the module scripts import from it.

PROVIDER AUTO-DETECTION
-----------------------
Set ONE of these in code/.env:
    OPENAI_API_KEY=sk-...          # uses OpenAI directly (gpt-4o-mini, etc.)
    OPENROUTER_API_KEY=sk-or-v1-.. # uses OpenRouter (Claude, Gemini, free models)

If both are set, LLM_PROVIDER decides (default: openrouter). You can also force
it: LLM_PROVIDER=openai. Model names and the API base URL are chosen to match
the provider automatically. Embeddings + reranker are local HuggingFace models,
so they need no API key either way.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths — resolve relative to code/ regardless of where Python is launched.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent          # the code/ directory
DATA_DIR = ROOT / "data" / "k8s-docs"                  # M0 corpus (72 md files)
GOLDSET_PATH = ROOT / "goldset.json"                   # M0 eval questions
MYNTRA_CSV = ROOT / "Myntra_300_prod_catalogue.csv"    # M2 / M3 corpus
RESULTS_PATH = ROOT / "scripts" / "eval_results.json"

# Pulls OPENAI_API_KEY / OPENROUTER_API_KEY from code/.env if present.
load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------------------
# Provider resolution — OpenAI or OpenRouter, whichever key is present.
# ---------------------------------------------------------------------------
OPENAI_BASE = "https://api.openai.com/v1"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _resolve_provider() -> tuple[str, str, str | None]:
    """Return (provider, api_base, api_key) based on env.

    Priority:
      1. Explicit LLM_PROVIDER=openai|openrouter
      2. Whichever key is set (OpenRouter wins if both are present)
    """
    forced = os.environ.get("LLM_PROVIDER", "").strip().lower()
    or_key = os.environ.get("OPENROUTER_API_KEY")
    oa_key = os.environ.get("OPENAI_API_KEY")

    if forced == "openai":
        provider = "openai"
    elif forced == "openrouter":
        provider = "openrouter"
    elif or_key:
        provider = "openrouter"
    elif oa_key:
        provider = "openai"
    else:
        provider = "openrouter"  # default; require_env() will raise a clear error

    if provider == "openai":
        return "openai", OPENAI_BASE, oa_key
    return "openrouter", OPENROUTER_BASE, or_key


PROVIDER, API_BASE, API_KEY = _resolve_provider()

# ---------------------------------------------------------------------------
# Models — chosen per provider, each overridable via env.
# To go fully free-tier on OpenRouter, set GENERATION_MODEL to a ":free" model.
# ---------------------------------------------------------------------------
if PROVIDER == "openai":
    GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "gpt-4o-mini")   # answers
    JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")             # RAGAS judge
else:
    GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "anthropic/claude-haiku-4.5")
    JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "google/gemini-2.5-flash")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # 384-dim dense embeddings, local, free
RERANK_MODEL = "BAAI/bge-reranker-base"  # cross-encoder reranker, local, CPU


def extra_headers() -> dict:
    """OpenRouter likes attribution headers; OpenAI ignores them."""
    if PROVIDER == "openrouter":
        return {"HTTP-Referer": "https://localhost/", "X-Title": "RAG for Engineers"}
    return {}


# ---------------------------------------------------------------------------
# M0 canonical test queries (Kubernetes corpus).
# Q1 is the money shot: `--max-pods` is a literal token that dense embeddings
# smear into "pod limit" semantics, so naive RAG never returns the value 110.
# ---------------------------------------------------------------------------
Q1 = "What is the default value of the --max-pods flag on the kubelet?"
Q2 = "How does an Ingress controller decide which backend Pod to forward a request to?"
Q3 = "If I want zero-downtime rolling updates for a StatefulSet with persistent storage, what features must I configure together?"
TEST_QUERIES = [Q1, Q2, Q3]

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
def require_env() -> None:
    """Fail fast if no API key is set, before anything expensive runs."""
    if not API_KEY:
        raise SystemExit(
            "No LLM API key found. Add ONE of these to code/.env:\n"
            "    OPENAI_API_KEY=sk-...            # for OpenAI\n"
            "    OPENROUTER_API_KEY=sk-or-v1-...  # for OpenRouter"
        )


def check_provider() -> None:
    """Round-trip one tiny request to confirm the provider is reachable."""
    from openai import OpenAI
    try:
        OpenAI(
            base_url=API_BASE, api_key=API_KEY, default_headers=extra_headers()
        ).chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=4,
            temperature=0,
        )
    except Exception as e:
        raise SystemExit(
            f"{PROVIDER} unreachable: {type(e).__name__}: {e}\n"
            "Confirm your API key in code/.env and that the account has credit."
        )
    print(f"Provider OK: {PROVIDER} ({API_BASE})")


# Backward-compatible alias (older imports called this name).
check_openrouter = check_provider


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_goldset() -> list[dict]:
    """Load the Kubernetes evaluation goldset (used by M0)."""
    with open(GOLDSET_PATH) as f:
        return json.load(f)["questions"]


def load_myntra_products() -> list[dict]:
    """Load the Myntra product catalogue (used by M2 and M3).

    Returns a list of {"id", "name", "description", "text"} where `text` is
    the searchable blob we embed and index: "name. description".
    The CSV has columns: index, name, description (≈299 products).
    """
    products: list[dict] = []
    with open(MYNTRA_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("name") or "").strip()
            desc = (row.get("description") or "").strip()
            if not name:
                continue
            products.append({
                "id": (row.get("") or str(len(products))).strip(),
                "name": name,
                "description": desc,
                "text": f"{name}. {desc}",
            })
    return products


def myntra_nodes():
    """Return the Myntra catalogue as LlamaIndex TextNodes (one per product)."""
    from llama_index.core.schema import TextNode
    return [
        TextNode(
            text=p["text"],
            metadata={"name": p["name"], "product_id": p["id"]},
        )
        for p in load_myntra_products()
    ]


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------
def save_scores(label: str, scores: dict) -> None:
    results: dict = {}
    if RESULTS_PATH.exists():
        try:
            results = json.loads(RESULTS_PATH.read_text())
        except Exception:
            results = {}
    results[label] = scores
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))


def print_scoreboard() -> None:
    if not RESULTS_PATH.exists():
        print("(no scoreboard yet — run m0_naive.py first)")
        return
    results = json.loads(RESULTS_PATH.read_text())
    print("\n=== Scoreboard ===")
    print(f"{'module':25s} {'faithfulness':>14s} {'answer_relevancy':>18s}")
    print("-" * 60)
    for label, scores in results.items():
        f = scores.get("faithfulness", float("nan"))
        a = scores.get("answer_relevancy", float("nan"))
        print(f"{label:25s} {f:>14.3f} {a:>18.3f}")
    print()


# ---------------------------------------------------------------------------
# RAGAS eval (used by M0) — faithfulness + answer_relevancy.
# Per-sample async to avoid RAGAS 0.4's executor recursion bug.
# On OpenRouter the judge (Gemini) is decorrelated from the generator (Haiku).
# On OpenAI both default to gpt-4o-mini; override JUDGE_MODEL to decorrelate.
# ---------------------------------------------------------------------------
async def evaluate_async(query_fn, label: str, sample: int | None = None) -> dict:
    """Score a query function against the goldset with RAGAS.

    Args:
      query_fn: callable(question:str) -> {"answer": str, "contexts": list[str]}
      label:    row key in eval_results.json (e.g. "M0_naive")
      sample:   number of questions to evaluate. None = all. M0 passes 5.

    Returns: {"faithfulness": float, "answer_relevancy": float}
    """
    import asyncio

    from openai import AsyncOpenAI
    from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, Faithfulness

    client = AsyncOpenAI(
        base_url=API_BASE,
        api_key=API_KEY,
        default_headers=extra_headers(),
    )
    llm = llm_factory(
        model=JUDGE_MODEL, provider="openai", client=client, max_tokens=8192
    )
    emb = RagasHFEmbeddings(model=EMBED_MODEL)
    faith = Faithfulness(llm=llm)
    ar = AnswerRelevancy(llm=llm, embeddings=emb)

    qs = load_goldset() if sample is None else load_goldset()[:sample]
    faith_scores: list[float] = []
    ar_scores: list[float] = []

    print(f"\n--- RAGAS eval: {label} ({len(qs)} questions) ---")
    for i, q in enumerate(qs, 1):
        if asyncio.iscoroutinefunction(query_fn):
            r = await query_fn(q["question"])
        else:
            r = query_fn(q["question"])

        ctxs = [c for c in r["contexts"] if c.strip()] or [""]

        try:
            f = float(await faith.ascore(
                user_input=q["question"], response=r["answer"], retrieved_contexts=ctxs,
            ))
        except Exception as e:
            print(f"  [{i:2d}/{len(qs)}] {q['id']} faithfulness FAILED: {type(e).__name__}: {str(e)[:120]}")
            f = float("nan")

        try:
            a = float(await ar.ascore(user_input=q["question"], response=r["answer"]))
        except Exception as e:
            print(f"  [{i:2d}/{len(qs)}] {q['id']} answer_relevancy FAILED: {type(e).__name__}: {str(e)[:120]}")
            a = float("nan")

        faith_scores.append(f)
        ar_scores.append(a)
        f_disp = f"{f:.2f}" if not math.isnan(f) else " nan"
        a_disp = f"{a:.2f}" if not math.isnan(a) else " nan"
        print(f"  [{i:2d}/{len(qs)}] {q['id']} ({q['category']:13s})  faith={f_disp}  ar={a_disp}")

    def _mean(xs: list[float]) -> float:
        clean = [x for x in xs if not math.isnan(x)]
        return sum(clean) / len(clean) if clean else float("nan")

    scores = {
        "faithfulness": _mean(faith_scores),
        "answer_relevancy": _mean(ar_scores),
    }
    print(f"\n  ⇒ {label}: faithfulness={scores['faithfulness']:.3f}  "
          f"answer_relevancy={scores['answer_relevancy']:.3f}")
    save_scores(label, scores)
    return scores
