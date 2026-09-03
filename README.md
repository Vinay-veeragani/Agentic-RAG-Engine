# Agentic RAG Platform

An advanced, production-shaped Agentic RAG platform: autonomous knowledge
retrieval, evidence gathering, verification, and grounded answer generation —
not a "chat with PDF" demo.

**Status: Phase 7 of 13 complete** (Foundation, ingestion, chunking +
embeddings, dense/sparse/hybrid retrieval, reranking, query analysis +
planning, then the agentic retrieval loop). Everything below describes what
is actually implemented today; features from later phases (deeper evidence
evaluation + contradiction detection, answer synthesis, citations,
evaluation harness, streaming, frontend) are tracked but not yet built. See
`docs/architecture.md` for design decisions, rationale, and known gaps, and
the implementation plan below for what's next.

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
8. Evidence evaluation + contradiction detection
9. Answer synthesis + citations + citation validation
10. Evaluation framework (baseline vs. agentic RAG benchmark)
11. Observability (OpenTelemetry/metrics) + SSE streaming
12. Frontend
13. Security + reliability + performance hardening

## Limitations (current)

- No authentication/authorization yet
- No real managed Redis in local use yet (in-memory fallback only)
- No CI pipeline yet
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
- The agentic loop's evidence sufficiency check is still the lightweight
  version — no contradiction detection, source authority, or temporal
  reasoning yet (Phase 8)
- Query decomposition retrieves subqueries independently, not as a
  dependency chain (spec's "find companies -> then look up their values"
  multi-hop pattern is not implemented)
- Nothing from the agentic loop is persisted to the database yet (no
  `GET /queries/{id}/trace`) — its full trace is only returned in the API
  response
- Everything past Phase 7 (deeper evidence evaluation, answer generation,
  citations, evaluation) does not exist yet — there is no answer-generation
  endpoint to call today, only ingestion, indexing, search/retrieve/rerank,
  query analysis, the agentic retrieval loop, and `/health`
