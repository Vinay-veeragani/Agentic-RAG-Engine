# Agentic RAG Platform

An advanced, production-shaped Agentic RAG platform: autonomous knowledge
retrieval, evidence gathering, verification, and grounded answer generation —
not a "chat with PDF" demo.

**Status: all 13 phases complete** (Foundation, ingestion, chunking +
embeddings, dense/sparse/hybrid retrieval, reranking, query analysis +
planning, the agentic retrieval loop, deeper evidence evaluation, answer
synthesis + citations, the evaluation framework, observability + streaming,
the full Next.js frontend, then security/reliability/performance
hardening). Everything below describes what is actually implemented today.
See `docs/architecture.md` for design decisions, rationale, and known
gaps.

## Frontend

The full spec §37 UI lives under `frontend/` — Next.js + TypeScript +
Tailwind + shadcn/ui + TanStack Query + Zustand, with all 9 sections
(Knowledge, Collections, Documents, Search, Ask, Retrieval Traces,
Evaluations, Observability, Settings) and the signature `/ask` page's
expandable retrieval trace + Developer Mode. See
`docs/architecture.md`'s "Phase 12" section for what was built, a real
base-ui-vs-Radix API gotcha it surfaced, and how it was verified.

To run it locally: start the backend (`uvicorn agentic_rag.api.main:app
--reload`), then from `frontend/`, `npm install` and `npm run dev`.

## What's implemented so far

- FastAPI app skeleton with structured JSON logging and per-request trace IDs
- Full database schema (17 tables) on PostgreSQL 18 + pgvector, via Alembic
- Domain error hierarchy covering every failure mode the system must
  eventually distinguish (`NO_KNOWLEDGE`, `INSUFFICIENT_EVIDENCE`,
  `CONFLICTING_EVIDENCE`, `BUDGET_EXCEEDED`, etc.) — errors never get
  silently converted into a fabricated success response
- `GET /health` reporting database and cache reachability independently
- Cache backend abstraction (Redis-compatible interface; falls back to an
  in-process store when no managed Redis is configured yet)
- Local object storage with path-traversal protection
- Config-driven provider selection (`LLM_PROVIDER`, `EMBEDDING_PROVIDER`,
  `RERANKER_PROVIDER`), defaulting to a mock provider so nothing requires a
  paid API key to run
- Document ingestion for PDF, DOCX, TXT, Markdown, HTML, CSV, and JSON —
  each parsed into a common structural representation (headings, paragraphs,
  tables, list items, code blocks, page numbers) via a per-format parser
  behind one interface, with re-uploads versioned rather than duplicated
  (`POST /documents`, `GET /documents`, `GET /documents/{id}`,
  `POST /collections`, `GET /collections`)
- Four chunking strategies (fixed, recursive, structural — the default, with
  parent/child chunks for oversized sections — and semantic similarity-based)
  behind one interface, plus local/mock/remote embedding providers with
  caching, indexed into pgvector via `POST /documents/{id}/ingest`
- Dense (pgvector cosine similarity), sparse (Postgres full-text search),
  and hybrid retrieval (Reciprocal Rank Fusion, independently testable as a
  pure function) with a fixed-field metadata filter (collection, document
  type, document IDs, section, heading, source, year) — `POST /search` for
  a simple ranked result list, `POST /retrieve` for the developer view with
  per-method scores and explicit strategy selection
- Reranking (mock term-overlap and a real local cross-encoder,
  `cross-encoder/ms-marco-MiniLM-L-6-v2`) narrowing a wide candidate pool
  down to a precise top-k while preserving every retrieval score alongside
  the rerank score — `POST /retrieve` with `rerank: true`
- Query understanding (classifies into simple-factual/comparison/temporal/
  analytical/multi-hop/summarization/ambiguous), retrieval planning (bounded
  strategy/expansion/decomposition/iteration decisions — always clamped to
  configured budget ceilings, never trusting the LLM alone), query
  expansion, and query decomposition, all via a shared LLM provider
  abstraction (mock, local via Ollama, or OpenAI) — `POST /query/analyze`
- A bounded agentic retrieval loop — plan → retrieve → rerank → judge
  evidence sufficiency → refine and retry if insufficient, hard-bounded by
  iteration count and retrieval-call budget (never an infinite loop by
  construction, not just by convention), ending in one of four explicit
  termination reasons — `POST /query/retrieve`
- Deeper evidence evaluation: relevance/coverage/directness scoring,
  deterministic cross-document contradiction detection (never invents a
  resolution — only resolves via a per-collection configurable source
  authority order, e.g. Annual Report > Press Release), and temporal-spread
  awareness. An unresolved contradiction stops the loop immediately with an
  explicit `conflicting_evidence` reason rather than pretending the
  evidence is fine
- Grounded answer synthesis with first-class citations that can never be
  fabricated — the LLM only ever references evidence by a small index into
  the list it was shown; real chunk/document IDs are resolved deterministically
  in code, never guessed at. Citations are validated for actual entailment
  (not just topical relatedness), and any claim that fails validation is
  removed from the final answer rather than silently left in — `POST /query`,
  the full pipeline (query → plan → retrieve → rerank → evidence → synthesis
  → citation validation)
- A real evaluation framework comparing a baseline RAG pipeline (dense
  retrieval → top-k → LLM, no planning/reranking/evidence/citations)
  against the full agentic pipeline, over a small self-contained benchmark
  corpus spanning every category the spec calls for (simple factual,
  comparison, temporal, analytical, aggregation, ambiguous, unanswerable,
  contradictory evidence) — see "Baseline vs. agentic RAG" below for real,
  captured results, not fabricated numbers
- Full observability: structured SSE streaming of the pipeline's exact
  spec-defined event sequence (`query.started` → ... → `query.completed`)
  via `POST /query/stream`, a queryable per-query trace via
  `GET /queries/{trace_id}/trace`, and Prometheus-compatible metrics
  (per-phase latency, retrieval iterations, cache hits, failures) via
  `GET /metrics` — never exposing hidden chain-of-thought, only structured
  decisions and telemetry
- Security/reliability hardening (all off/permissive by default, opt-in
  for a real deployment — see `docs/architecture.md`'s Phase 13 section):
  optional API-key auth, fixed-window rate limiting, a deterministic
  prompt-injection filter that excludes suspicious retrieved chunks from
  the synthesis prompt entirely, `max_query_latency_seconds` now actually
  enforced (504/TIMEOUT on expiry instead of a request that hangs
  forever), bounded retry-with-backoff on transient provider failures,
  configurable DB connection pool sizing, security response headers, and
  gzip compression

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

Run tests: `pytest` · Lint: `ruff check src tests` · Type-check: `mypy src`

A Docker Compose stack (`docker/docker-compose.yml`) is maintained for CI and
for environments that can run Docker, but is not exercised in local
day-to-day development on this machine — see `docs/architecture.md` for why.

## Baseline vs. agentic RAG

Run it yourself: `python benchmarks/run_evaluation.py --embedding local --llm mock`
(defaults shown). This rebuilds a small 15-document synthetic corpus fresh
from the real ingestion pipeline, runs both pipelines over 9 cases spanning
every category spec §33 asks for, and writes a full JSON report to
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

See `docs/architecture.md`.

## Implementation plan

1. ✅ Foundation + database + configuration
2. ✅ Document ingestion (PDF/DOCX/TXT/Markdown/HTML/CSV/JSON parsers)
3. ✅ Chunking + embeddings + indexing
4. ✅ Dense + sparse + hybrid retrieval, Reciprocal Rank Fusion
5. ✅ Reranking
6. ✅ Query analysis + retrieval planning
7. ✅ Agentic retrieval loop (bounded iteration, refinement)
8. ✅ Evidence evaluation + contradiction detection
9. ✅ Answer synthesis + citations + citation validation
10. ✅ Evaluation framework (baseline vs. agentic RAG benchmark)
11. ✅ Observability (structured events/metrics) + SSE streaming
12. ✅ Frontend
13. ✅ Security + reliability + performance hardening

## Limitations (current)

- API-key auth and rate limiting are implemented but off by default (see
  `docs/architecture.md`'s Phase 13 section) — a real deployment must
  explicitly set `API_KEYS` and `RATE_LIMIT_ENABLED=true`; there's still
  no `users`-table-backed identity system, key rotation, or per-key tiers
- Rate limiting is fixed-window, not sliding-window/token-bucket — a
  documented burst tradeoff at window boundaries
- Prompt-injection detection (new in Phase 13) is heuristic/regex-based,
  not semantic — a rephrased attempt that matches no listed pattern won't
  be caught
- No real managed Redis in local use yet (in-memory fallback only) —
  relevant to rate limiting now too, not only caching
- No CI pipeline yet
- No load/performance testing was run (no k6/locust harness) — Phase 13's
  pool-sizing/retry changes are reasoned improvements, not measured under
  simulated concurrent load
- No OCR — a scanned/image-only PDF parses with no extracted text rather
  than failing loudly
- Markdown tables are not specially recognized (parse as plain text)
- Token counting is an offline whitespace-based approximation, not a real
  LLM subword tokenizer (`tiktoken`'s data host is unreachable from this
  machine — see `docs/architecture.md`)
- Sparse retrieval is Postgres full-text search (`ts_rank_cd`), not true
  BM25 — no term-frequency saturation or length normalization
- No retrieval-result caching yet; every search hits Postgres directly
- No remote reranker (e.g. Cohere) — only a deterministic mock and a real
  local cross-encoder
- No Anthropic or Gemini LLM provider yet — mock, Ollama (local), and
  OpenAI only; Ollama and OpenAI are implemented but unexercised in this
  environment (no Ollama install, no API key)
- Contradiction detection only catches numeric percentage claims sharing a
  small fixed set of keywords (revenue/profit/margin/growth/decline/
  earnings/sales/income) — non-numeric semantic contradictions aren't
  detected; that would need real LLM reasoning and is a deliberate,
  documented tradeoff for something that works identically regardless of
  which LLM provider (or the mock) is configured
- Temporal awareness is informational only — evidence spanning multiple
  years is visible in the trace but not specially separated/handled
- Query decomposition retrieves subqueries independently, not as a
  dependency chain (spec's "find companies -> then look up their values"
  multi-hop pattern is not implemented)
- Nothing from the agentic loop is persisted to the database yet (no
  `GET /queries/{id}/trace`) — its full trace is only returned in the API
  response
- No conversation memory — each `/query` call runs independently; a
  follow-up like "what about 2024?" is not resolved against a prior query
- No `url` field on citations yet — nothing in ingestion captures a source
  URL
- Citation numbering doesn't deduplicate — the same chunk cited by two
  different claims gets two different citation numbers
- `citation_recall` (spec §33) isn't computed — needs a ground-truth
  "which chunks must be cited" label the benchmark fixtures don't carry
- Generation-quality benchmark numbers are plumbing-only without a real
  LLM provider configured (see "Baseline vs. agentic RAG" above)
- The benchmark corpus (15 documents) is deliberately small for a fast,
  self-contained, repo-committed run — too small to meaningfully stress
  hybrid fusion/reranking/decomposition the way a realistic corpus would
- Benchmark results aren't persisted to the database (`evaluation_datasets`/
  `evaluation_cases`/`evaluation_results` tables exist in the schema since
  Phase 1 but are unused) — only written to a JSON file
- Query traces (`GET /queries/{id}/trace`) are in-memory only, per-process,
  bounded to the 200 most recent — a restart or a second worker process
  loses/fragments trace history; the `events` table exists in the schema
  but is unused
- SSE reconnection isn't implemented — a client can send `Last-Event-ID`
  but nothing replays missed events from a persisted store
- No actual OpenTelemetry span export — "observability" today means
  structured JSON logs, structured events, and Prometheus metrics, not OTel
  traces
- No cost estimation — token/cache instruments exist but nothing converts
  them into an estimated dollar figure
- The frontend has not been visually verified in an actual browser (no
  browser-automation tool was available) — HTTP-level and type-checking
  verification only; see `docs/architecture.md`'s Phase 12 section
- No frontend automated tests (component tests, e2e)
