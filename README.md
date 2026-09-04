# Agentic RAG Platform

An advanced, production-shaped Agentic RAG platform: autonomous knowledge
retrieval, evidence gathering, verification, and grounded answer generation —
not a "chat with PDF" demo.

**Status: feature-complete.** Document ingestion, chunking + embeddings,
hybrid retrieval, reranking, query planning, a bounded agentic retrieval
loop, evidence evaluation, grounded answer synthesis with citations, an
evaluation framework, observability, a full web frontend, and security/
reliability hardening are all implemented. See `docs/architecture.md` for
design decisions and rationale.

## Frontend

The full UI lives under `frontend/` — Next.js + TypeScript +
Tailwind + shadcn/ui + TanStack Query + Zustand, with all 9 sections
(Knowledge, Collections, Documents, Search, Ask, Retrieval Traces,
Evaluations, Observability, Settings) and the signature `/ask` page's
expandable retrieval trace + Developer Mode. See `docs/architecture.md`
for what was built and how it was verified.

To run it locally: start the backend (`uvicorn agentic_rag.api.main:app
--reload`), then from `frontend/`, `npm install` and `npm run dev`.

## How it works

Most "RAG" tools take a question, fetch some chunks of text that look
similar to it, and hand both to an LLM to generate an answer — with no
check on whether the retrieved text actually supports what gets said back.
This project takes a different approach: retrieval and reasoning are
treated as a small **agentic loop** that only produces an answer once it
has evidence it can actually stand behind.

1. **Understand the question.** Classify what kind of question it is
   (a simple fact lookup, a comparison, a multi-part question, something
   ambiguous or unanswerable from the corpus) and plan a retrieval
   strategy for it.
2. **Retrieve.** Search the knowledge base using a hybrid of semantic
   (embedding) search and keyword search, combined and ranked together,
   then rerank the results for precision.
3. **Judge the evidence — before writing anything.** Check whether what
   was retrieved actually answers the question, is internally consistent
   across sources, and covers the different angles the question asks
   about. If sources disagree with each other, that's flagged, not
   silently averaged. If the evidence looks thin, the system retrieves
   again with a refined strategy — inside a hard budget, so it can never
   loop forever.
4. **Answer, grounded in what was actually found.** Only once there's
   sufficient, consistent evidence does the system generate an answer —
   and every claim in it is checked against the evidence it's supposedly
   based on. A claim that isn't actually supported gets removed rather
   than shipped. If there simply isn't a good answer in the knowledge
   base, the system says so explicitly instead of guessing.

Every step along the way is visible: you can see what was retrieved, why
the system considered it sufficient (or not), what it found grounded,
and what it filtered out.

## Capabilities

- **Multi-format ingestion** — PDF, Word, Markdown, HTML, CSV, JSON, and
  plain text, all normalized into a common structure and organized into
  collections.
- **Hybrid retrieval** — semantic + keyword search, fused and reranked,
  with filtering by document, section, source, or year.
- **A bounded agentic loop** — the system can retrieve, judge, and
  refine multiple times per question, but it's hard-capped so it always
  terminates predictably rather than spinning indefinitely.
- **Contradiction-aware evidence review** — conflicting facts across
  documents are detected and either resolved by a configurable source
  priority (e.g. trust an audited report over a press release) or
  surfaced explicitly rather than papered over.
- **Answers you can verify** — every citation traces back to a real
  document and passage, and claims that aren't actually supported by the
  evidence are dropped before the answer is returned.
- **A real evaluation harness** — a baseline retrieve-then-generate
  pipeline is benchmarked side-by-side with the full agentic pipeline
  on a shared test corpus, so the difference isn't a claim, it's a
  measured result (see below).
- **Built-in observability** — every query's reasoning trace, timing,
  and decisions are inspectable, with live streaming and metrics for
  monitoring in production.
- **Security and reliability hardening** — optional authentication, rate
  limiting, prompt-injection filtering on retrieved content, request
  timeouts, and automatic retries on transient failures.
- **A full web UI** — upload documents, ask questions, inspect retrieval
  traces, and compare evaluation results, all from the browser.

See `docs/architecture.md` for how each of these is actually built and
the engineering tradeoffs behind them.

## Local setup

Requires Python 3.11, a PostgreSQL 18 instance with the `vector` extension
available, and (optionally) a Redis-compatible URL.

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"
copy .env.example .env   # then edit DATABASE_URL / REDIS_URL for your setup
alembic upgrade head
uvicorn agentic_rag.api.main:app --reload
```

Run tests: `pytest` (add `-m slow` to also run the real cross-encoder
reranker test, which downloads/loads a model on first use) · Lint:
`ruff check src tests` · Type-check: `mypy src`

A Docker Compose stack (`docker/docker-compose.yml`) is maintained for CI and
for environments that can run Docker, but is not exercised in local
day-to-day development on this machine — see `docs/architecture.md` for why.

## Baseline vs. agentic RAG

Run it yourself: `python benchmarks/run_evaluation.py --embedding local --llm mock`
(defaults shown). This rebuilds a small 15-document synthetic corpus fresh
from the real ingestion pipeline, runs both pipelines over 9 cases spanning
simple factual, comparison, temporal, analytical, aggregation, ambiguous,
unanswerable, and contradictory-evidence questions, and writes a full JSON
report to
`benchmarks/results/latest.json` (committed, from a real run — nothing
below is a fabricated number).

**Baseline**: Query → Dense Retrieval → Top-K → LLM. No planning, no hybrid
retrieval, no reranking, no evidence judgment, no citations.
**Agentic**: Query → Analyze → Plan → Hybrid Retrieval → Rerank → Evidence
Judge → Refine if necessary → Synthesize → Citation Validate.

Captured `2026-09-03`, local sentence-transformers embeddings + mock LLM:

| Metric | Baseline | Agentic |
|---|---|---|
| Recall@5 (excl. ambiguous/unanswerable) | 1.000 | 1.000 |
| Precision@5 | 0.286 | 0.286 |
| MRR | 0.929 | 0.857 |
| NDCG@5 | 0.947 | 0.903 |
| Hit Rate@5 | 1.000 | 1.000 |
| Mean latency | 0.041s | 0.090s |
| Citation precision / completeness | n/a (no citations) | 1.000 / 1.000 |

Retrieval scores are close between the two on this small corpus — both
search the same index, and the corpus isn't hard enough for reranking/
expansion/decomposition to move the needle much. **The real difference is
behavioral, not the retrieval numbers**: on the "unanswerable" case (no
relevant document exists at all), the agentic pipeline returns
`insufficient_evidence` instead of guessing; on the "contradictory_evidence"
case (two sources reporting different numbers, no way to prefer one), it
returns `conflicting_evidence` and surfaces the specific conflicting claims.
The baseline pipeline has no such option — it always emits *an* answer, with
no signal to the caller about whether the evidence actually supported it.

Generation-quality numbers from this specific run (`answer_relevance`, and
implicitly `mean_estimated_tokens`) are **not** a real quality signal —
`MockLLMProvider` excerpts evidence deterministically rather than generating
language, since no real LLM provider (Ollama, OpenAI) is available in this
environment. The retrieval-side comparison above is real (real local
embeddings, real retrieval/rerank/evidence code); the generation side needs
`--llm ollama` or `--llm openai` with real credentials to mean anything.
Three real bugs in the contradiction detector were found and fixed by
actually running this benchmark against real corpus text — see
`docs/architecture.md` for what they were.

## Design decisions

See `docs/architecture.md` for design decisions, rationale, and known
gaps/roadmap items.
