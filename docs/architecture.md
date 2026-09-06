# Architecture

This document describes the design of the Agentic RAG platform: a
FastAPI + PostgreSQL/pgvector backend implementing a bounded, multi-step
agentic retrieval loop (query analysis → planning → hybrid retrieval →
reranking → evidence evaluation → citation-grounded synthesis), a
baseline-vs-agentic evaluation framework, full observability/streaming,
and a Next.js frontend. It covers what is implemented and, where it
matters, *why* — the tradeoffs and bugs that shaped the current design.

## Overview

The system ingests documents (PDF, DOCX, Markdown, HTML, CSV, JSON, plain
text) into collections, chunks and embeds them, and answers questions
against them through an agentic retrieval loop rather than a single
retrieve-then-generate pass. The loop plans a retrieval strategy, executes
it, evaluates whether the evidence gathered is actually sufficient (and
checks it for contradictions), refines and retries when it isn't, and only
then synthesizes an answer whose every claim is citation-checked against
the evidence that was actually retrieved. An evaluation framework compares
this pipeline against a plain dense-retrieval baseline on a small
self-contained benchmark corpus, and the whole pipeline is instrumented
with structured events, Prometheus metrics, and SSE streaming.

A recurring engineering principle throughout the codebase: **use
deterministic, testable code wherever a decision doesn't genuinely require
language understanding, and reserve LLM calls for the few places that
do** (query classification, decomposition, evidence judgment, entailment
checking, synthesis). Fusion, chunk windowing, citation-index resolution,
contradiction detection, and rate limiting are all pure/deterministic
logic for exactly this reason — it makes them independently testable and
means their correctness never depends on which model provider is
configured.

## Local development setup

Docker is not runnable on the primary development machine (insufficient
system resources). `docker/docker-compose.yml` and `docker/Dockerfile` are
maintained in the repo for CI and any environment that can run Docker, but
local day-to-day development uses:

- **PostgreSQL + pgvector**: a native Windows PostgreSQL 18 install with
  the pgvector 0.8.6 extension enabled directly (`CREATE EXTENSION
  vector;`), rather than the dockerized `pgvector/pgvector` image.
- **Redis**: intended to be a free-tier managed instance (e.g. Upstash).
  `storage/cache.py` auto-falls-back to an in-process `InMemoryCache` when
  `REDIS_URL` is unset or still the `.env.example` placeholder, so the app
  and test suite work without any external account (see Design tradeoffs
  below for the consequences of this fallback).

Everything is env-var driven (`DATABASE_URL`, `REDIS_URL`), so switching
between this local setup and the docker-compose stack is a config change,
not a code change.

Other environment-specific choices:

- **Python 3.11, not 3.12+.** The stack recommendation calls for Python
  3.12+; this machine has 3.11.9 (stable, all target dependencies have
  prebuilt wheels) and 3.14 (too new — `asyncpg`/`psycopg` wheel
  availability isn't reliable for it yet). 3.11 was chosen for dependency
  stability; nothing in the codebase depends on a 3.12-only feature.
- **Windows + asyncpg event loop policy.** Windows' default
  `ProactorEventLoop` is incompatible with asyncpg's connection teardown
  (raises on close). `agentic_rag/__init__.py` sets
  `WindowsSelectorEventLoopPolicy` at package-import time, before any event
  loop is created, for both the test suite and the running app.
- **Migrations run with a sync driver; the app runs async.** The app uses
  `asyncpg` (`postgresql+asyncpg://...`) for runtime queries. Alembic
  migrations run synchronously, so `migrations/env.py` swaps in the sync
  `psycopg` driver (`postgresql+psycopg://...`) derived from the same
  `DATABASE_URL` just for running migrations — one env var, two derived
  connection strings. Alembic's autogenerate doesn't know how to import
  `pgvector.sqlalchemy` for `Vector`-typed columns, so any future
  `alembic revision --autogenerate` touching `document_chunks.embedding`
  needs `import pgvector.sqlalchemy` added by hand to the generated
  migration — see `migrations/versions/77d4e3cf344b_initial_schema.py`.

## System architecture

- `core/config.py` — env-driven `Settings` (Pydantic Settings), including
  the agentic-loop budget ceilings (max iterations/calls/tokens/latency).
- `core/errors.py` / `core/models.py` — a full failure-mode error
  hierarchy, so every layer raises domain errors rather than ad-hoc
  exceptions.
- `storage/postgres.py`, `storage/models.py` — async engine/session
  factory and the complete schema (17 tables incl. `alembic_version`) in
  one ORM models module rather than split across `knowledge/`,
  `citations/`, `evaluation/`, etc. Several tables cross domain boundaries
  by design (e.g. `Citation` foreign-keys into both `DocumentChunk` and
  `Answer`), which would otherwise force circular imports between domain
  modules; layering repository/business logic on top of one shared models
  module avoids that. Beyond foreign keys, the schema also enforces real
  invariants at the database level, not just in application code: unique
  constraints prevent a duplicate `chunk_index` within a document version
  or the same chunk being recorded twice for one retrieval run, and check
  constraints restrict `documents.document_type` and
  `document_versions.status` to the values the code actually ever writes
  — found missing during an engineering audit, verified with regression
  tests that a bad insert genuinely raises `IntegrityError`, not just
  that the migration applied cleanly.
- `storage/cache.py` — Redis-or-in-memory cache client factory.
- `storage/vector_store.py` / `storage/object_store.py` — a `VectorStore`
  protocol (pgvector is the concrete implementation used by the
  retrievers) and an `ObjectStore` protocol with a path-traversal-safe
  local filesystem implementation.
- `api/main.py` — FastAPI app with structured JSON logging, per-request
  trace-ID middleware/header, and domain-error → HTTP translation that
  never leaks stack traces.
- `GET /health` — reports Postgres and cache reachability independently.
- `observability/tracing.py` — structlog configuration + trace-ID context
  var.

### Embedding dimension is fixed per deployment

`document_chunks.embedding` is `vector(384)` (see
`storage/models.py::EMBEDDING_DIMENSIONS`), matching the default local
embedding model family (`bge-small-en` / `all-MiniLM-L6-v2`). pgvector
columns have a static dimension — swapping to a differently-sized
embedding model requires a new migration that alters this column and a
full re-embed of existing chunks. The embedding *provider* stays
replaceable; the *dimension* is a real, unavoidable constraint of
pgvector that the provider abstraction can't hide.

## Document ingestion

- `ingestion/parsed_document.py` — the common internal representation
  every parser normalizes into (`ParsedDocument` → `DocumentElement`s),
  independent of any database model.
- Seven format-specific parsers behind one `DocumentParser` protocol
  (`ingestion/parsers/`): PDF (PyMuPDF, heading detection via relative
  font size, per-page table extraction — see below for two real bugs a
  live end-to-end test against a real 111-page lease PDF surfaced that no
  synthetic test document ever exercised), DOCX (python-docx, walks the
  body's paragraphs/tables in document order, heading detection via
  paragraph style), Markdown (`markdown-it-py` token stream — headings,
  paragraphs, list items, fenced code blocks; GFM tables are not
  specially recognized, see Design tradeoffs), HTML (BeautifulSoup,
  block-tag walk), CSV (one element per row rendered as `column: value`
  pairs), JSON (one element per record for a list-of-objects shape,
  otherwise the whole document as one element), and plain text.
- `ingestion/loaders/validation.py` — file type detection from extension
  plus a magic-byte content check for binary formats (PDF's `%PDF-`
  signature, DOCX's zip local-file-header signature) so an extension
  alone is no longer trusted for those two — a renamed file that isn't
  even a PDF/zip is rejected before it reaches a parser. Text formats
  (TXT/MD/HTML/CSV/JSON) have no reliable universal signature and aren't
  content-checked; their parsers already fail loudly on genuinely
  unparseable content. Also does path-traversal-safe filename
  sanitization and empty/oversized upload rejection — the API route
  (`api/routes/documents.py`) reads uploads in bounded chunks and aborts
  as soon as the configured size limit is crossed, rather than buffering
  an arbitrarily large upload into memory before ever checking its size.
- `ingestion/cleaners/text.py` — Unicode normalization, control-character
  stripping, whitespace cleanup, applied uniformly after parsing
  regardless of source format.
- `ingestion/pipeline.py` — orchestrates validate → detect type → parse →
  persist; re-uploading the same filename to the same collection creates
  a new `DocumentVersion` — unless the content is byte-identical to the
  latest version's (same checksum), in which case it's a genuine no-op
  (no new version row, no redundant object-store write). An earlier
  version of this computed a checksum but never actually checked it
  against anything, so every re-upload created a new version regardless
  of whether the content had changed — an engineering audit caught this;
  fixed, not just documented.
- `POST /collections`, `GET /collections`, `POST /documents` (multipart
  upload), `GET /documents`, `GET /documents/{id}` — wired to the
  pipeline above.
- `Document.source` is settable at upload (`POST /documents` accepts a
  `source` form field, e.g. `"Annual Report"`) and threaded through every
  retriever into `RetrievedCandidate.document_source`, which source
  authority resolution (see Evidence evaluation below) reads.
- `Document.document_date` is likewise settable at upload (a `document_date`
  form field) — the document's real-world date (publication/fiscal
  period), distinct from `created_at` (upload time). Deliberately
  caller-supplied rather than auto-extracted: reliably parsing a
  publication date out of arbitrary prose is a real NLP problem, not
  something a deterministic parser can do generically across seven
  formats. `MetadataFilter.year` (see Retrieval below) prefers this field
  when set.

HTML/PDF/DOCX heading and table detection are heuristic/best-effort (font
size for PDF headings, tag/style name for HTML/DOCX) — there is no
universal "this is a heading" signal in any of these formats. Caption
detection is implemented for HTML `<figcaption>` only; PDF/DOCX captions
are not distinguished from regular paragraphs.

### Two real bugs found by running the PDF parser against a real document

Every prior verification of `PdfParser` used small, clean, hand-written
test PDFs. Running it end-to-end against a real 111-page commercial lease
(via a real Groq LLM and real local embeddings, not mocks) surfaced two
genuine bugs no synthetic document ever exercised:

1. **Heading detection classified 787 of 1375 elements (57%) as
   headings** — including full paragraphs up to 6,219 characters long.
   The cause: `max_size` (the block's largest font size, used to decide
   "is this a heading") was computed from *every* span in a text block,
   including whitespace-only ones — a stray blank run with an unrelated,
   larger font size (a real, if odd, PDF-authoring artifact) silently
   inflated a whole paragraph's `max_size` even though no visible text
   was actually that size. Separately, a paragraph containing just one
   bolded/larger word (common in legal documents — defined terms are
   often emphasized) got its *entire* text reclassified as a heading,
   since the check only looked at the block's max font size, never how
   much of the block was actually that size. Fixed by (a) only
   considering spans with real (non-whitespace) text when computing
   `max_size`, and (b) requiring a heading to be short (≤150 characters)
   *and* have at least 60% of its actual characters at the oversized
   font — not just contain one oversized word. Dropped the false-positive
   rate from 787 to 22 headings on the same document; the 22 that remain
   are genuine section titles and exhibit headers (plus a few short,
   genuinely garbled fragments from a scanned signature page — a source-
   document limitation, not a parser bug). Regression tests
   (`tests/unit/test_parsers_binary.py`) construct PDFs with each specific
   defect (a whitespace-only oversized span; a long paragraph with one
   oversized word) via PyMuPDF's own HTML-box insertion, so both are
   proven fixed, not just asserted against this one real document.
2. **Smart quotes/dashes came back as mojibake** (`Landlord's` →
   `Landlordâ€™s`) — the classic UTF-8-decoded-as-cp1252 double-encoding
   bug: this specific PDF's real UTF-8 bytes for `'` (`E2 80 99`) got
   decoded one byte at a time as Windows-1252 somewhere in extraction,
   producing three separate mojibake characters instead of one real one.
   `ingestion/cleaners/text.py::clean_text` (applied uniformly to every
   parser, not just PDF) now detects this specific pattern and repairs it
   by round-tripping through cp1252 bytes — but only when the tell-tale
   marker is present *and* doing so actually reduces it; ordinary text
   (including genuine non-English text) is always returned untouched,
   verified by tests that assert non-mojibake text round-trips as-is.

Not fixed, and likely not safely fixable: the same PDF also has real
extraction-level corruption from what looks like a broken font/ToUnicode
CMap for certain letter pairs — "Term" comes back as "Tenn", "current
forms" as "cun-ent fonns". This isn't a decoding bug like the mojibake
above (there's no reversible transform to undo); the embedded font itself
maps the wrong Unicode codepoints to those glyphs in the source PDF, so
PyMuPDF is returning exactly what the file says those characters are. The
only real fix is OCR (render the page to an image, re-read via a vision
model or Tesseract, bypassing the broken embedded font entirely) — a
genuinely larger feature, not a parser tweak, and consistent with the
existing "no OCR" tradeoff already noted for scanned/image-only PDFs.

Also found in that same live test: the local cross-encoder reranker's
first-ever request paid the full model download/load cost (~10+ seconds)
inline, since the model loads lazily on first use. `api/dependencies/
reranker.py::warm_up_default_reranker()` is now called once at app
startup (`api/main.py`'s `lifespan`) when the configured reranker
supports it, so that cost is paid before any request needs it — a no-op
for `MockReranker`, which has nothing to warm up.

## Chunking & embeddings

- `chunking/tokenization.py` — a fully offline, dependency-free tokenizer
  (whitespace-vs-non-whitespace runs), used instead of `tiktoken`.
  `tiktoken` needs to download its BPE merge table from a Microsoft
  blob-storage host on first use; that host is unreachable from this
  machine's Python specifically (DNS resolution fails for it even though
  PyPI and every other host used elsewhere — Hugging Face Hub included —
  resolve fine). Rather than have chunk-token-budgeting depend on a
  network fetch, token counting is a small rule-based splitter: exactly
  reversible (`"".join(encode(text)) == text`), deterministic, but an
  approximation of any real LLM's actual subword tokenizer, not an exact
  match. Swapping `tiktoken` back in if it becomes reachable is contained
  to this one file.
- Four chunkers behind one `Chunker` protocol (`chunking/`):
  `FixedSizeChunker` (pure token-count sliding window, ignores structure
  — the baseline), `RecursiveChunker` (accumulates whole elements up to
  the token budget, recursively splits an oversized single element on a
  separator hierarchy), `StructuralChunker` (the default — groups
  elements into contiguous same-heading runs and never merges across a
  heading boundary; an oversized section gets one parent chunk covering
  the whole section plus child chunks from `RecursiveChunker`), and
  `SemanticChunker` (embeds sentences and breaks on a cosine-similarity
  drop *or* the token budget, whichever comes first — the one chunker
  that genuinely needs an embedding call rather than pure deterministic
  logic).
- Parent/child chunk resolution (`chunking/pipeline.py`): chunks are
  flushed once to get real database-assigned UUIDs, *then*
  `parent_chunk_id` is resolved and a second flush persists it —
  SQLAlchemy applies a column's Python-side `default=` callable at flush
  time, not at object construction, so resolving parent references before
  the first flush silently produced `NULL` (`chunk.id` reads back `None`
  until flushed). Caught by an integration test asserting actual
  persisted `parent_chunk_id` values, not just the in-memory
  `ChunkCandidate.parent_index`.
- `embeddings/` — `EmbeddingProvider` protocol; `LocalEmbeddingProvider`
  (sentence-transformers `all-MiniLM-L6-v2`, 384-dim, CPU, runs in a
  thread via `asyncio.to_thread` since the library is synchronous);
  `MockEmbeddingProvider` (deterministic hash-seeded unit vectors, no
  model/network — default for tests and local dev); `OpenAIEmbeddingProvider`
  (a plain `httpx` call to OpenAI's REST endpoint rather than pulling in
  the `openai` SDK for one call site; requests `dimensions=384` explicitly
  via OpenAI's Matryoshka truncation support so it matches the fixed
  pgvector column width regardless of which provider is configured).
- `embeddings/cache.py` — `CachedEmbeddingProvider` wraps any provider
  behind the shared cache client, keyed by model name + text hash, so
  switching models never serves another model's stale vector.
- `POST /documents/{id}/ingest` — chunks + embeds + persists
  `DocumentChunk` rows for a document version (deliberately separate from
  `POST /documents`, which only parses + stores metadata); accepts
  optional per-call overrides for strategy/chunk size/overlap/similarity
  threshold, defaulting to platform config.

`SemanticChunker`'s sentence splitting is a simple punctuation-based
regex, not a real sentence-boundary detector — it mis-splits on
abbreviations ("e.g.", "Dr.") the way most lightweight splitters do.

## Retrieval (dense/sparse/hybrid + RRF)

- `document_chunks.content_tsv` — a generated (`GENERATED ALWAYS AS ...
  STORED`) `tsvector` column plus a GIN index, added by a hand-written
  migration (`938c2d23d9a7`) rather than `alembic revision --autogenerate`,
  since autogenerate doesn't reliably reproduce Postgres's computed-column
  syntax. Verified with `alembic check` reporting no drift between the
  ORM model and the applied schema.
- `retrieval/dense.py` — `DenseRetriever`: pgvector cosine-distance search
  (`1 - distance` recovers similarity for the normalized vectors every
  embedding provider here produces), optional `score_threshold`,
  filterable.
- `retrieval/sparse.py` — `SparseRetriever`: PostgreSQL full-text search
  via `plainto_tsquery` + `ts_rank_cd` over `content_tsv`.
  `plainto_tsquery` (not `to_tsquery`) treats arbitrary user input as
  plain text rather than tsquery syntax, so operators/parentheses/quotes
  in a query can't break or redefine the search — exercised directly with
  adversarial input, not just assumed safe. This is Postgres full-text
  search, not literal BM25 (see Design tradeoffs).
- `retrieval/filters.py` — `MetadataRetriever` (filter-only, no query)
  plus `build_filter_conditions()`, shared by every retriever. Filters are
  a fixed field list (collection, document type, document IDs, section,
  heading, source, year) rather than an open-ended key/value language —
  "arbitrary safe metadata filters" is satisfied by that closed field
  list, not by sanitizing arbitrary filter expressions. `year` filters on
  `Document.document_date` (a caller-supplied real-world date, set at
  upload time — see Document ingestion) when one was provided, falling
  back to `created_at` (upload time) only when it wasn't. An earlier
  version of this filter used `created_at` unconditionally, which an
  engineering audit correctly flagged as silently wrong for anything but
  a freshly-published document — fixed, not just documented as a gap.
- `retrieval/fusion.py` — `reciprocal_rank_fusion()`: a pure function over
  ranked ID lists with zero DB/embedding dependency, so fusion is
  independently testable with unit tests that never touch a database. RRF
  was chosen over a learned fusion model deliberately: it needs no
  training data, behaves predictably and is trivial to reason about
  (score = sum of `1/(k+rank)` across the lists a candidate appears in),
  and is a well-established baseline for combining ranked lists from
  heterogeneous retrieval signals.
- `retrieval/hybrid.py` — `HybridRetriever`: pulls a wider candidate pool
  (default 30) from dense and sparse independently, fuses via RRF, and
  returns the top-k with dense/sparse/fusion scores all preserved per
  candidate — scores are never collapsed into one number, so the caller
  can always see how a result was actually found.
- `POST /search` (simple, one score per result) and `POST /retrieve`
  (developer/debug view: per-method scores, explicit strategy selection
  among dense/sparse/hybrid).

The full-text search config is hardcoded to `"english"`, not yet
per-collection configurable. Retrieval-run persistence
(`retrieval_runs`/`retrieved_chunks` tables) and result caching are
addressed by the agentic loop and observability layers described below,
not by the retrievers themselves.

## Reranking

- `retrieval/reranking.py` — `Reranker` protocol: `rerank(query,
  candidates, top_k)` sets `rerank_score` on each candidate and returns
  the top-k by that score, leaving `dense_score`/`sparse_score`/
  `fusion_score` untouched. `MockReranker` (deterministic query/content
  term-overlap, no model or network) and `LocalCrossEncoderReranker`
  (sentence-transformers `CrossEncoder`,
  `cross-encoder/ms-marco-MiniLM-L-6-v2`, CPU, no API key) sit behind it.
  Every other reranking test exercises `MockReranker` only, so
  `tests/unit/test_reranking.py::test_local_cross_encoder_reranker_ranks_relevant_content_first`
  runs the real model against a genuinely relevant vs. irrelevant pair and
  asserts the correct ordering — real, automated proof reranking helps,
  not just that it runs without error. It's marked `slow` (downloads/loads
  a real model) and excluded from the default `pytest` run; run it with
  `pytest -m slow`.
- `retrieval/reranking.py`'s `rerank_with_fallback()` wraps every call to
  `reranker.rerank()` (both `POST /retrieve` and the agentic loop's
  `RetrievalAgent`): if the reranker raises — a model that fails to load,
  or errors mid-inference — the query still succeeds, falling back to the
  input order (already fusion/retrieval-scored) truncated to `top_k`,
  logged and counted via `agentic_rag_rerank_failures_total` rather than
  failing the whole request over a non-critical enhancement step.
- `POST /retrieve` accepts `rerank`/`rerank_top_k` fields. When
  `rerank=true`, retrieval first fetches `candidate_pool_size` candidates
  (top 20-30) and the reranker narrows that down to `rerank_top_k` (top
  5-10 evidence chunks); `rank` is reassigned to reflect the post-rerank
  order.

No remote reranker (e.g. Cohere Rerank) is implemented — only mock and
local. The `Reranker` protocol is the same shape a remote provider would
implement (mirroring how `OpenAIEmbeddingProvider` slots into
`EmbeddingProvider`), but adding one wasn't done speculatively without a
concrete need.

## Query understanding & planning

- `generation/llm.py` — `LLMProvider` protocol: `complete()` (raw text,
  mirroring every real chat API) plus `complete_structured()`, provided
  by default via `BaseLLMProvider` in terms of `complete()` —
  schema-instructed prompting (the pydantic JSON Schema is embedded in
  the system prompt), JSON extraction (handles a bare object, a markdown
  code fence, or prose wrapping one), pydantic validation, and one retry
  with the validation error fed back to the model before raising
  `ModelProviderError`. This is the first LLM-reasoning component in the
  system — query classification/expansion/decomposition are exactly the
  kind of language-understanding task deterministic rules handle poorly
  in general, unlike RRF or chunk windowing.
- Four real providers: `MockLLMProvider` (below), `OllamaLLMProvider`
  (local, `/api/chat` with `format: "json"`, no API key — implemented but
  not exercised by any test, since Ollama isn't installed/running on this
  machine), `OpenAILLMProvider` and `GroqLLMProvider` (both plain `httpx`
  calls against `OpenAICompatibleLLMProvider`, parameterized by base URL
  + default model rather than one class per vendor — OpenAI, Groq, and a
  growing list of others all expose the same `/chat/completions`
  request/response shape). `GroqLLMProvider` was actually exercised
  end-to-end against a real 111-page PDF with a real API key (see
  Document ingestion above for what that surfaced) — correct query
  classification, well-reasoned evidence-sufficiency judgments, grounded
  synthesis, and, critically, a correct `insufficient_evidence` refusal
  on an out-of-scope question rather than a fabricated answer. Anthropic
  and Gemini are not implemented — the same `LLMProvider` protocol would
  support them, but adding unexercised provider code with no way to
  verify it wasn't done speculatively.
- `generation/mock.py` — `MockLLMProvider.complete_structured` does not
  go through `BaseLLMProvider`'s wrapper; it directly introspects the
  *requested pydantic schema type* (via `model_fields`) and fills each
  field deterministically — keyword/pattern heuristics for fields whose
  name signals intent (`query_type`, `decompose`, `subqueries`,
  `reasoning`, ...), generic rules by Python type otherwise (bool/int/
  float/str/Enum/`list[str]`/`list[BaseModel]`/nested `BaseModel`). This
  is real architectural investment, not a shortcut: every agent built on
  `complete_structured` (evidence evaluation, citation validation, answer
  synthesis) gets a working offline mock for free, the same role
  `MockEmbeddingProvider`/`MockReranker` play for their layers. Query
  heuristics operate on text extracted from a `"Query: <text>"` line in
  the prompt specifically, not the raw prompt blob — otherwise the
  planner's prompt (which appends a JSON classification blob after the
  query) would feed unrelated JSON text into the same keyword rules; this
  was a real bug caught while manually exercising the mock end-to-end,
  fixed before it reached tests.
- `agents/query_analyzer.py` — `QueryAnalyzer` (classifies into
  `QueryType`, returns structured `is_ambiguous`/`is_answerable`/
  `reasoning`, never free-form planning text) and `QueryExpander`
  (proposes up to 5 phrasing variants; only invoked when the plan says
  to — expansion is never applied blindly to every query).
- `agents/planner.py` — `RetrievalPlanner` (decides strategy,
  `expand_query`, `decompose`, `max_iterations`, `top_k`, reusing
  `retrieval.base.MetadataFilter` directly rather than a parallel schema)
  and `QueryDecomposer` (splits a complex query into up to 8 standalone
  subqueries). The planner's `max_iterations`/`top_k` are always clamped
  to configured ceilings after the LLM (or mock) call — the planner must
  be bounded and validated in code, not just prompted for; covered by a
  unit test that sets a ceiling of 1 and asserts the mock's default of 3
  gets clamped down regardless.
- `POST /query/analyze` — a preview endpoint exposing analysis + plan +
  conditional expansion/decomposition, ahead of the full agentic loop
  that actually executes a plan like this (see below).

`MockLLMProvider`'s heuristics are tuned to this codebase's actual prompt
schemas (`QueryAnalysis`, `RetrievalPlan`, `QueryExpansion`,
`QueryDecomposition`) and field-naming conventions — not a generic
pydantic-schema-to-plausible-data filler for arbitrary external schemas,
though the type-based fallback rules do generalize. `QueryDecomposer`'s
mock-path splitting (`" and "`/comma) is a naive heuristic, not real
dependency-aware decomposition.

## The agentic retrieval loop

- `agents/retrieval_agent.py` — `RetrievalAgent.retrieve(queries, plan)`:
  dispatches to dense/sparse/hybrid per the plan's strategy, and supports
  multiple query variants at once (used when the plan enabled expansion)
  by retrieving each variant independently and fusing the per-variant
  rankings with the same `reciprocal_rank_fusion` that combines dense +
  sparse, applied one level up. Always reranks down to a final evidence
  count (default 8, within the suggested 5-10) and reassigns `rank`
  post-fusion/rerank.
- `agents/research_agent.py` — `AgenticRetrievalLoop.run()`: the bounded
  plan → retrieve → rerank → evaluate → refine-if-insufficient loop.
  Hard-bounded *by construction*: the loop body is a `for iteration in
  range(1, plan.max_iterations + 1)`, and `max_iterations` is already
  clamped to `settings.max_retrieval_iterations` by `RetrievalPlanner`
  before this ever runs — a bounded `for` loop cannot run forever
  regardless of what any LLM (or the mock) proposes.
  `settings.max_retrieval_calls` is checked independently every
  iteration, since query decomposition can turn one iteration into
  several retrieval calls. Query refinement between iterations is
  deterministic (`original_query + " " + missing_information`) rather
  than another LLM call — a defensible choice, and one fewer place a bad
  structured-output response could break the loop. Every run ends in
  exactly one `TerminationReason` (`sufficient_evidence` /
  `max_iterations_reached` / `max_retrieval_calls_reached` /
  `no_evidence_found` / `conflicting_evidence`, added with contradiction
  detection — see Evidence evaluation below).
- The API layer passes a caller-supplied `collection_id` in, and it wins
  over whatever the plan guessed (an LLM/mock has no way to know a real
  collection ID) — verified end-to-end that retrieval stays correctly
  scoped to the given collection and returns `no_evidence_found` for an
  empty or wrong one.
- `POST /query/retrieve` — a preview endpoint (like `/query/analyze`)
  returning the full structured trace and final evidence, no synthesized
  answer.
- `tests/integration/test_research_agent.py::test_loop_refines_query_and_succeeds_on_second_iteration`
  proves the actual "insufficient → refine → retry → sufficient"
  transition end to end — the two extremes (instant success, permanent
  failure) were covered elsewhere, but nothing previously asserted the
  middle case the whole design rests on. `MockLLMProvider`'s own
  sufficiency heuristic (query/evidence term overlap) can't reliably
  demonstrate this on demand, since its deterministic query refinement
  echoes the original query rather than adding genuinely new search terms
  — so the test uses a thin wrapper forcing exactly one insufficient
  judgment before switching to sufficient, while every other structured
  call (analysis, planning, expansion) still runs through the real mock
  unchanged. Confirms a real second retrieval fires with a genuinely
  different, refined query, and both iterations show up in the trace.

### Two real bugs found by running the loop, not by design review

Both were invisible in review and only surfaced by actually running the
loop against real Postgres before writing any test:

1. `asyncio.gather()` across query variants / decomposed subqueries, all
   sharing one `AsyncSession`, threw `InvalidRequestError: This session
   is provisioning a new connection; concurrent operations are not
   permitted` — SQLAlchemy's async session is not safe for concurrent use
   from multiple coroutines. Fixed by retrieving sequentially instead.
   The broader design principle (execute concurrently *where safe*) still
   holds; this is the "where safe" carve-out, not a shortcut around it.
2. A decomposed query's subqueries could burn through
   `max_retrieval_calls` within a single outer iteration, since the
   budget check only ran once per iteration, not once per subquery. Fixed
   by truncating the subquery list to the remaining budget before
   retrieving.

Query decomposition treats subqueries as independent parallel retrievals,
not a dependency graph — genuinely useful for a query like "what are the
revenue and profit figures", where both halves are directly searchable,
but not for a query where the second step's evidence is only findable via
an entity the *first* step's evidence reveals (a real multi-hop shape,
e.g. "who is the CEO, and where did they work before"). `agents/multi_hop
.py` closes exactly that gap for a two-hop chain: when the query
classifies as multi-hop and decomposition yields at least two subqueries,
`AgenticRetrievalLoop._retrieve_chained()` runs hop one, has an LLM
extract the single bridging entity from hop one's evidence, deterministically
folds that entity into hop two's query text, then runs hop two against the
*resolved* query — not run independently and merged by ranking, which
cannot find evidence keyed by a name the original query never used.
Proven with a real, deliberately adversarial test
(`tests/integration/test_multi_hop.py`): a control case shows the naive,
unchained search for hop two's literal text is provably unable to find the
answer (Postgres full-text search on an empty, all-stopword tsquery
matches nothing), while the chained path does find it. A genuinely longer
dependency chain (three or more hops) is still out of scope — that belongs
to a full Deep Research–style system, a different project. The loop
always reranks (no way to skip it) and evidence count is a fixed default
(8) rather than plan-configurable.

## Evidence evaluation & contradiction detection

`agents/evidence_agent.py`'s `EvidenceAgent.evaluate()` splits by how much
genuine reasoning each piece needs:

- **Relevance/coverage/directness** are LLM-judged 0.0-1.0 fields on
  `EvidenceAssessment`, alongside `sufficient`/`reason`/
  `missing_information`. No candidates always short-circuits to
  `sufficient=False` without an LLM call.
- **Contradiction detection** is fully deterministic: regex extracts
  `(metric keyword, percentage)` pairs (e.g. "revenue ... 4%") from
  evidence content; two different values for the same keyword from
  *different* documents is a `Contradiction`. This does not attempt
  general semantic contradiction detection (two sources disagreeing in
  prose with no shared number) — that would need real LLM reasoning and
  is a documented tradeoff (see Design tradeoffs). What it does catch
  works identically for every provider including the mock, since it's
  not an LLM call at all.
- **Source authority**: `Collection.source_authority_config` (a schema
  column that existed from the start but was unused until this feature)
  holds a per-collection `{"order": [...]}` list of source labels,
  most-to-least authoritative — configurable via `POST /collections
  {"source_authority_order": [...]}`, never hardcoded as universally
  correct. A `Contradiction.resolution` is set *only* when the two
  sources' configured ranks actually differ — a provenance-based
  preference, explicitly not a claim about which content is factually
  correct. Equal or unclassified ranks leave `resolution=None`,
  surfacing the conflict as-is rather than inventing one (if the system
  cannot resolve the conflict, it says so explicitly rather than guessing).
- **Temporal awareness**: regex year extraction across evidence content,
  surfaced as `years_referenced`/`spans_multiple_periods` — informational
  for now, not a hard block on mixing periods.

`AgenticRetrievalLoop` builds the evidence agent per run, loading the
target collection's authority config first. An **unresolved**
contradiction ends the run immediately with
`TerminationReason.CONFLICTING_EVIDENCE` — refining the search query
can't fix two sources genuinely disagreeing, so continuing would just
burn iteration budget pretending the problem is retrieval quality. A
contradiction the authority policy *does* resolve does not block the
loop; it's still surfaced in the iteration trace either way.

### A real bug found by exercising authority resolution end-to-end

`retrieval/hybrid.py`'s `_merge()` (written before `document_source`
existed) rebuilt each `RetrievedCandidate` field-by-field and simply
didn't include it, so hybrid retrieval silently dropped source
information that dense and sparse retrieval both populated correctly.
Caught because a same-scenario authority-resolution test failed with
`resolution=None` when it should have resolved — invisible without
actually running the full retrieve→evaluate path; a unit test of
`_merge` in isolation wouldn't have caught it either, since it would need
to know to check this specific field. Fixed, with a regression test
asserting `document_source` survives hybrid fusion.

### Three real bugs found by running the evaluation benchmark

Every one of these was invisible until real corpus text hit the
contradiction detector — exactly the value a real (if small) benchmark
run provides over unit tests written against hand-picked examples:

1. **The metric-pattern regex only matched a literal `%` symbol.** The
   benchmark corpus (and most real prose) spells out "percent." The
   known contradictory-evidence case silently failed to be detected.
   Fixed by accepting `%`, `percent`, and `per cent`.
2. **Different time periods were flagged as contradictions.** A 2023
   margin of 22% and a 2024 margin of 25% in two different documents were
   flagged as conflicting — they're just different years. Fixed by
   extracting each candidate's referenced years and skipping the flag
   when both candidates reference years and those years don't overlap. A
   follow-on discovery: a document that mentions its own year *and* a
   comparison year ("a 25 percent increase over 2023") pollutes its
   year-set — worked around by rewording the corpus to keep each
   year-specific document self-contained, since disambiguating "which
   year this specific number is about" within one chunk is a harder
   problem than this feature set out to solve.
3. **Contradiction detection considered the entire evidence list,
   unscoped by relevance.** With a small corpus and a padded-out top-8
   evidence pool, an unrelated conflict ranked 7th of 8 was flagging
   *every* query that happened to retrieve it — including queries with
   nothing to do with that conflict. Fixed by scoping the check to the
   top 4 candidates by rerank rank, on the reasoning that a low-relevance
   chunk that only made it into the pool to fill out top_k shouldn't be
   able to veto an otherwise-clean answer. This is a real architectural
   finding, not a corpus-tuning trick: a production corpus large enough
   that top_k is a small fraction of it would show this less often, but
   the *unscoped* check was still wrong in principle regardless of corpus
   size.

Contradiction detection only catches numeric claims sharing one of a
small fixed set of keywords (revenue/profit/margin/growth/decline/
earnings/sales/income) framed as a percentage — non-numeric semantic
contradictions and contradictions in absolute figures are not detected,
a real limitation of the deterministic-only approach, traded for it
working identically and reliably regardless of which LLM provider (or
the mock) is configured. `Contradiction.resolution` text is a fixed
template, not model-generated — deliberate (never invent a resolution),
but means the same two-source conflict always produces identically
worded resolution text.

## Answer synthesis & citations

**The key design decision here**: the LLM is never asked to produce a
real citation ID (chunk/document UUID) — only a small 1-based index
(`[1]`, `[2]`, ...) into the evidence list it was shown in the prompt.
`citations/resolver.py::resolve_citations()` is the *only* place those
indices become real IDs, by looking them up in the same Python evidence
list the prompt was built from. An index outside that list's actual
range is silently dropped, never guessed at. This makes "never fabricate
a citation" a structural guarantee rather than a prompt instruction
hoping the model complies — the same pattern used for chunking/embedding
indices elsewhere, applied here to the highest-stakes case in the whole
system.

- `agents/synthesis_agent.py` — `SynthesisAgent`: given a query and
  evidence, produces discrete claims each carrying its own
  `evidence_indices`. No evidence at all is a deterministic short-circuit
  (`insufficient_evidence=True`, no LLM call) — nothing to synthesize
  from, matching `EvidenceAgent`'s and `MetadataFilter`'s established
  pattern.
- `agents/citation_agent.py` — `CitationAgent`: checks whether a claim's
  *cited* evidence actually entails it (not merely relates to the same
  topic) — a genuine language-understanding judgment, so it goes through
  the LLM. A claim with zero citations is trivially unsupported without
  needing a model call.
- `agents/verifier.py` — `AnswerVerifier` ("groundedness"): assembles the
  final answer from *only* the claims whose citations passed entailment
  validation — literally dropping unsupported ones and rejoining the
  rest, rather than asking an LLM to "edit" its own prior answer. If
  every claim gets dropped, the whole answer becomes
  `AnswerStatus.INSUFFICIENT_EVIDENCE`, not an empty-but-"grounded"
  answer.
- `citations/formatter.py` — deterministic display formatting (e.g.
  `"[1] Annual Report, page 42, Revenue Recognition"`, falling back to
  filename when no source/page/section are set).
- `citations/validator.py` — `citation_precision` (fraction of proposed
  citations that were actually entailed) and `citation_completeness`
  (fraction of claims that ended up with a validated citation), pure
  arithmetic over already-computed results. `citation_recall` needs a
  ground-truth relevant-chunk set from an evaluation dataset and isn't
  computable from one live query — that's the evaluation framework's job
  (see below).
- `POST /query` — the system's primary endpoint: runs the full agentic
  retrieval loop, then — unless it ended in `CONFLICTING_EVIDENCE` or
  `NO_EVIDENCE_FOUND`, in which case synthesis is skipped entirely rather
  than attempting to paper over the problem — synthesizes and
  citation-validates an answer. `/query/analyze` and `/query/retrieve`
  remain as lower-level preview endpoints (useful for the frontend's
  retrieval-trace view). Verified against a running server: a real query
  against real indexed content produced a grounded answer with
  `citation_completeness`/`citation_precision` both 1.0 and a correctly
  page/section/source-labeled citation.

### A real bug found by exercising the pipeline before writing any test

The mock LLM's generic string-fallback (used for the synthesized claim's
`text` field) built the claim from the *query* text with added filler
words ("mock value for: ..."). When that same claim text was later fed
back into citation validation as the thing being checked against
evidence, the filler words diluted the lexical-overlap ratio below the
entailment threshold — so even directly-relevant evidence got its claim
rejected as unsupported. Fixed by deriving the mock's claim text from the
*evidence* it was given instead of the query, which is also more
semantically honest: a claim quoting its source evidence should of
course be entailed by it.

Citation entailment validation is a single LLM call per claim with no
retry-with-different-evidence — if evidence exists but doesn't quite
support a claim, the claim is dropped, not repaired or re-retrieved for.
No `url` field is ever populated on a citation — nothing in the
ingestion pipeline captures a source URL yet. `POST /query` always runs
the full loop from scratch; there is no conversation memory linking a
follow-up query to a prior one. Citation formatting always numbers
citations `[1]`, `[2]`, ... in the order claims were validated, with no
deduplication if two different claims cite the same underlying chunk (it
appears twice, with two different numbers).

## Evaluation framework

- `evaluation/datasets.py` — a small, self-contained benchmark corpus (16
  synthetic documents: 9 finance-related plus 8 topic-distinct
  distractors, plus a genuine two-hop pair) and 10 cases covering simple
  factual (x2), comparison, temporal, analytical, aggregation, ambiguous,
  unanswerable, contradictory-evidence, and a real multi-hop case (`cfo_2024
  .txt` answers "who", `cfo_prior_role.txt` answers "what did they do
  before" — but only by name, which a plain independent-subquery search
  can't find at all; see Query understanding & planning below for why
  this specifically requires the dependency-chained retrieval path, not
  just decomposition). `build_benchmark_corpus()` ingests + indexes this
  corpus fresh through the *real* pipeline and resolves each case's
  ground-truth relevant documents from the real `Document.id` values that
  ingestion just created — nothing here is a hardcoded chunk ID.
- `evaluation/retrieval.py` — Recall@K/Precision@K/MRR/NDCG/Hit Rate as
  pure functions, unit-tested against hand-computed expected values.
- `evaluation/baseline.py` — a literal baseline pipeline: Query → Dense
  Retrieval → Top-K → LLM. No planning, no reranking, no evidence
  judgment, no citations, no bounded refinement.
- `evaluation/generation.py` — `GenerationJudge.judge_answer_relevance()`,
  the one genuinely new LLM-judge call this framework adds.
  `faithfulness` and `context_relevance` are *not* separately re-judged —
  they're exactly what `citation_precision` and
  `EvidenceAssessment.relevance` already measure, and recomputing them
  with a second judge would be redundant, not more rigorous.
- `evaluation/citations.py` — aggregates per-case `CitationQualityMetrics`
  into corpus-wide means. `citation_recall` needs a ground-truth "which
  chunks must be cited" label this benchmark's fixtures don't carry (only
  document-level relevance, for retrieval metrics) — not computed, a
  documented gap rather than a silent approximation.
- `evaluation/runner.py::run_benchmark()` — runs both pipelines over
  every case and assembles a `BenchmarkReport`. Retrieval-metric *means*
  exclude the ambiguous/unanswerable cases (they test correct
  abstention, not ranking quality) — each case's own numbers are still
  visible individually.
- `benchmarks/run_evaluation.py` — an actually-runnable CLI script
  (`python benchmarks/run_evaluation.py --embedding local --llm mock`)
  producing a printed comparison table and a JSON report
  (`benchmarks/results/latest.json`, committed as evidence of a real
  run). Defaults to the *local* sentence-transformers embedding provider
  rather than the mock one specifically so retrieval metrics measure real
  retrieval quality, not noise; also defaults `--reranker` to `mock` so
  the committed report reruns fast with no model download. Passing
  `--reranker local` exercises the real cross-encoder end to end — doing
  so surfaced a real interaction worth disclosing rather than hiding:
  on this corpus it flips several cases (including a plain
  `simple_factual` revenue lookup) from `grounded` to `conflicting_evidence`,
  because reordering candidates changes which chunks land in the
  contradiction detector's top-4 rank-scoped window (see Evidence
  evaluation above), incidentally pulling in an unrelated numeric claim
  from elsewhere in the corpus. Not fixed here — it's a genuine
  precision limit of keyword-scoped contradiction detection interacting
  with reranking, not a reranker bug, and a good target for follow-up
  work rather than a claim to paper over.

### Real benchmark results (captured 2026-09-04, `--embedding local --llm mock`)

| Metric | Baseline | Agentic |
|---|---|---|
| Recall@5 (excl. ambiguous/unanswerable) | 0.938 | 1.000 |
| Precision@5 | 0.275 | 0.300 |
| MRR | 0.938 | 0.875 |
| NDCG@5 | 0.906 | 0.900 |
| Hit Rate@5 | 1.000 | 1.000 |
| Mean latency | 0.052s | 0.082s |
| Citation precision | n/a (no citations) | 1.000 |
| Citation completeness | n/a | 1.000 |

Recall now genuinely diverges, driven entirely by the `multi_hop` case
(see The agentic retrieval loop and Query understanding & planning
above): the baseline's single dense-retrieval pass finds only the
document naming the CFO (recall 0.5 on that case), never the second
document describing their prior role — findable only by that name, which
the query itself never uses. The agentic pipeline's dependency-chained
retrieval extracts the name from the first hop's evidence and resolves
the second hop's search against it, finding both (recall 1.0 on that
case). Every other case scores similarly between the two pipelines —
expected, since both search the same indexed content and the corpus
isn't hard enough for reranking/expansion to visibly move the needle on
a single-hop question. **Beyond that measured difference, the rest is
behavioral, not a retrieval score**: for the "unanswerable" case (a query
with no relevant document at all), the agentic pipeline correctly
returns `insufficient_evidence` rather than a guess; for the
"contradictory_evidence" case (two sources reporting different numbers
with no way to prefer one), it correctly returns `conflicting_evidence`
and surfaces the specific conflicting claims rather than confidently
reporting one arbitrary number. The baseline pipeline has no such option
— it always produces *an* answer, with no signal to the caller about
whether the underlying evidence actually supported it.

`mean_answer_relevance` and other generation-quality numbers from this
specific run are not meaningful measures of real generation quality: the
mock LLM's answers are lexical excerpts of evidence, not created
language, since no real LLM provider (Ollama, OpenAI) was available in
this environment. The retrieval-metric comparison, by contrast, is real
and provider-independent (driven by real local embeddings and real
retrieval code). `estimated_tokens` only counts the final answer text via
this repo's own offline tokenizer — not prompt/context tokens actually
sent to an LLM, and not a real provider-reported usage figure; it's a
rough, consistent proxy, not a cost estimate. The 16-document corpus is
deliberately small for a fast, self-contained, repo-committed benchmark —
large enough to demonstrate the qualitative baseline-vs-agentic
differences above, but too small to meaningfully stress hybrid fusion,
reranking, or decomposition the way a realistic multi-thousand-chunk
corpus would. Benchmark results are written to a JSON file; the corpus is
rebuilt fresh on every run rather than persisted to the
`evaluation_datasets`/`evaluation_cases`/`evaluation_results` tables.

## Observability & streaming

- `observability/events.py` — `EventType` (13 event names spanning the
  full pipeline: `query.started`, `query.analyzed`, `plan.created`,
  `retrieval.started`, `retrieval.completed`, `reranking.started`,
  `reranking.completed`, `evidence.evaluated`, `retrieval.refined`,
  `generation.started`, `citation.validation.started`, `query.completed`,
  `query.failed`), `Event` (event ID, query/trace ID, timestamp, type,
  structured payload — nothing else, never hidden chain-of-thought), and
  `EventEmitter`, which is dual-purpose: it always appends to an
  in-memory list (read back after a non-streaming call completes) and,
  when constructed with `queue=True`, also pushes onto an `asyncio.Queue`
  a concurrent SSE endpoint drains live.
- `observability/metrics.py` — a small, deliberately narrow set of
  Prometheus instruments, one per thing actually worth tracking:
  per-phase latency histograms (query/retrieval/rerank/generation),
  retrieval-iteration and estimated-token histograms, cache hit/miss and
  query-failure counters. Exposed via `GET /metrics` in standard
  Prometheus text exposition format.
- Every phase of the pipeline is instrumented for real, not just
  logged-and-forgotten: `RetrievalAgent.retrieve()` returns a
  `RetrievalOutcome` carrying separately-measured retrieval and rerank
  latency and optionally emits `retrieval.started/completed`/
  `reranking.started/completed`; `AgenticRetrievalLoop` emits
  `query.started/analyzed`, `plan.created`, `evidence.evaluated`,
  `retrieval.refined`, records `QUERY_LATENCY_SECONDS`/
  `RETRIEVAL_ITERATIONS`, and accumulates per-phase latency onto both
  `IterationTrace` and the final `AgenticRetrievalResult`; `AnswerVerifier`
  emits `generation.started`/`citation.validation.started` and records
  its own latency separately. `embeddings/cache.py` records real cache
  hit/miss counts.
- `POST /query/stream` — SSE version of `POST /query`: the same
  `_run_query_pipeline()` helper runs as a background `asyncio.Task`
  while the route drains `EventEmitter`'s queue and yields each event as
  a `text/event-stream` frame (`id:`/`event:`/`data:` lines) as it
  happens.
- `GET /queries/{trace_id}/trace` — backed by `TraceStore`: process-local,
  in-memory only, bounded to the 200 most recent traces (LRU eviction) —
  see Design tradeoffs. Both `POST /query` and `POST /query/retrieve`
  store their emitted events here so a trace is queryable immediately
  after the call that produced it returns.
- `GET /metrics` — Prometheus text exposition format via
  `prometheus_client.generate_latest()`.

### Two real design corrections made while wiring this up

1. `query.completed` was initially emitted inside
   `AgenticRetrievalLoop.run()` itself — but that loop is *also* the
   first stage of the full `POST /query` pipeline (synthesis + citation
   validation follow it there), so a caller using the full pipeline would
   see `query.completed` fire before synthesis even started. Fixed by
   moving that emission to whichever caller actually finishes the whole
   pipeline (`POST /query/retrieve` right after the loop, `POST /query`
   after synthesis+validation) — `run()` itself only records its own
   metrics and emits `query.failed` on an internal exception.
2. `main.py`'s existing middleware already calls `bind_trace_id()` once
   per request (used for the `x-trace-id` response header) — the route
   code was calling `bind_trace_id()` *again*, generating a second,
   different trace ID than the one already in the response header, so a
   client comparing `X-Trace-Id` against the response body's `trace_id`
   field would see two different values. Fixed by having routes read the
   already-bound ID via `get_trace_id()` instead of rebinding.

A pre-existing test-isolation bug was also caught while running the full
suite after wiring this in: several retrieval tests queried without a
`collection_id` filter, silently relying on being the only data in a
small local database. Months of accumulated manual smoke-testing and
benchmark runs against the same shared local Postgres had grown the
database large enough that unrelated documents started outranking the
tests' own fixtures — invisible until it actually happened. Fixed by
scoping every affected test to its own collection, which is also simply
the correct way to write an isolated retrieval test regardless of
database size.

SSE reconnection is not implemented: a client may send `Last-Event-ID`,
but nothing replays events emitted before a reconnect from a persisted
store — only `GET /queries/{trace_id}/trace` can be polled afterward, and
only if the client already knows its trace ID. OpenTelemetry spans (as
opposed to structured JSON logs + these events) are not implemented —
what exists today is structured logging plus this event system, not
actual OTel span export. No cost estimation exists — cache hit/miss and
token-count instruments exist, but nothing converts them into an
estimated dollar cost per query.

## Frontend

Next.js 16.3.4 (App Router, Turbopack) + React + TypeScript + Tailwind
CSS v4 + shadcn/ui + TanStack Query v5 + Zustand, under `frontend/`.

Backend additions made to support it: `CORSMiddleware` registered with a
configurable `cors_allow_origins` setting (default
`http://localhost:3000`); `GET /collections/{id}`; `GET /settings`
(non-secret config only — provider names and budget ceilings, never
secrets); `GET /evaluations/latest` and `/evaluations/latest/summary`
(raw pass-through of the benchmark report); `QueryResponse` now also
returns `analysis` and `plan` so the frontend's retrieval-trace view
doesn't have to re-derive them.

Nine pages: `/` (Knowledge dashboard), `/collections` (list + create,
including `source_authority_order`), `/documents` (per-collection upload
+ explicit ingest trigger, showing real `DocumentIndexResponse` chunk
counts), `/search` (direct hybrid retrieval with per-result score
breakdowns, no agentic loop), `/ask` (the signature feature, below),
`/traces` (look up a trace ID and render its raw structured event
timeline), `/evaluations` (the real baseline-vs-agentic benchmark,
rendered as a comparison table — no fabricated numbers, it fetches the
same `benchmarks/results/latest.json` the backend serves),
`/observability` (live health + raw `/metrics` text), `/settings`
(read-only server config + the Developer Mode toggle).

`/ask` renders the answer, `StatusBadge` (`AnswerStatus` /
`TerminationReason`), `CitationList`, and an accordion `RetrievalTrace`
covering every stage the backend actually returns structured data for —
query classification, retrieval plan, then per-iteration search → hybrid
fusion → reranking → evidence evaluation (including any detected
contradictions and their resolution) → refinement, then final
synthesis/citation-validation metrics. There is no chain-of-thought
section because the backend never produces one to show.

**Developer Mode** (Zustand + `persist`, toggled from `/settings` or the
header): reveals trace ID, termination reason, strategy, top K, max
iterations, iterations used, retrieval+rerank latency, the three
configured provider names (fetched from `/settings`), and a collapsible
raw JSON dump of the full response.

**SSE**: `api.query.stream()` posts to `/query/stream` and parses `data:
` lines off a raw `ReadableStream` reader rather than `EventSource`,
since the endpoint is POST-based and `EventSource` only supports GET.

Shared infrastructure: `lib/types.ts` (a hand-written mirror of the
backend's Pydantic response schemas — no live-server codegen, since the
backend is still evolving), `lib/api.ts` (typed fetch wrapper +
`ApiError`), a module-level singleton `QueryClient` per current official
Next.js TanStack Query guidance, and a `CollectionSelect` shared across
Documents/Search/Ask.

### A real, non-obvious discovery from building this

shadcn/ui's CLI-scaffolded `Button` and `Accordion` in this project's
installed version are built on **base-ui**, not Radix — its API differs
in two places this project hit immediately: `Button` has no `asChild`
prop (base-ui uses a `render={<Link .../>}` prop instead), and
`Accordion.Root` has no `type="multiple"` prop (it takes a plain
`multiple` boolean, with `defaultValue`/`value` as an array either way).
Both were caught by `next build`'s TypeScript pass, not by inspection —
worth calling out because assuming Radix's API here would have shipped
silently broken interactivity.

### Verification performed

`npm run build` compiles and full-project type-checks cleanly, statically
generating all 9 routes plus `/`. `npx eslint .` is clean. An end-to-end
HTTP-level check (no browser-automation tool was available) started the
real FastAPI backend and the Next.js dev server together, confirmed all 9
pages return `200` with the expected server-rendered markup, and
confirmed `/health`, `/collections`, `/settings`, and
`/evaluations/latest` return correctly-shaped JSON against a real local
database with real ingested data. This does not substitute for actually
clicking through the UI in a browser, which is still an outstanding
verification step. There are no frontend automated tests (component
tests, Playwright/e2e) yet. The evaluation and trace pages assume the
artifacts already exist (`benchmarks/results/latest.json`, a known trace
ID) — there's no UI affordance yet to *trigger* a new benchmark run or
discover trace IDs other than pasting one from an Ask response. Document
upload/ingest is a manual two-step (`upload` then `Ingest` button) with
no progress indicator for large files.

## Security & reliability hardening

Every addition here follows the same "off/permissive by default, opt-in
for a real deployment" pattern the rest of the codebase already uses for
provider selection: nothing extra is required to run locally.

- **Optional API-key auth** (`security/auth.py`): `Settings.api_keys` is
  empty by default — auth disabled, every route open. Configure one or
  more keys and every route except `GET /health` and `GET /metrics`
  requires `Authorization: Bearer <key>` or `X-API-Key: <key>`, enforced
  by a single middleware in `api/main.py` rather than a per-route
  dependency, so a new route can't accidentally ship unauthenticated.
  `is_valid_api_key()` compares a candidate against every configured key
  with `hmac.compare_digest` (constant-time per candidate) rather than
  Python's `in` operator, which would otherwise leak timing information
  proportional to how many leading characters a guess shares with a real
  key — caught during an engineering audit; the key list is small enough
  that checking every entry costs nothing.
- **Rate limiting** (`security/rate_limit.py`): a fixed-window limiter
  (`Settings.rate_limit_enabled`, off by default) keyed by API key when
  present, else client IP, on top of the existing `CacheClient`
  abstraction (works identically against `InMemoryCache` or real Redis).
  Fixed-window is a deliberate simplicity tradeoff over a sliding window
  or token bucket — documented in the module docstring, along with the
  up-to-2x-burst-at-a-window-boundary behavior that comes with it. It
  defaults to **off** specifically because the existing integration test
  suite makes far more than 120 requests/minute against a shared
  in-memory rate-limit bucket when run in one process — caught by
  actually running the full suite with it enabled by default, not by
  inspection, and is why the default flipped to off rather than the test
  suite being reshaped around it.
- **Prompt-injection filtering** (`security/prompt_injection.py`): a
  bounded, deterministic regex sweep (same style as the contradiction
  detector) for common injection phrasings ("ignore previous
  instructions", "you are now", `<|im_start|>` delimiters, etc.) run over
  every retrieved chunk in `agents/verifier.py` before it reaches a
  synthesis/citation prompt. A match doesn't prove intent, but is
  conservative enough that the chunk is excluded from evidence entirely
  — logged and counted
  (`agentic_rag_prompt_injection_flagged_total`), never silently
  included. This is the first thing in the codebase that actually acts
  on "retrieved documents are untrusted data" for adversarial content
  embedded *inside* a chunk, rather than just at the metadata-filter
  layer.
- **`max_query_latency_seconds` is enforced**: `POST /query`, `POST
  /query/retrieve`, and `POST /query/stream` all wrap the pipeline in
  `asyncio.wait_for`, raising `QueryTimeoutError`
  (`FailureMode.TIMEOUT` → 504) on expiry rather than letting a hung
  provider call hang the request forever.
- **Bounded retry with backoff for transient provider failures**
  (`core/retry.py`): `OpenAICompatibleLLMProvider` (so both
  `OpenAILLMProvider` and `GroqLLMProvider`), `OpenAIEmbeddingProvider`,
  and `OllamaLLMProvider` retry a connection error/timeout or a retryable 5xx
  up to 3 attempts with exponential backoff before surfacing
  `ModelProviderError` — never retries a 4xx, since a client error can't
  succeed differently on a second attempt. This complements (doesn't
  replace) `generation/llm.py`'s one-retry-on-malformed-JSON logic, which
  is a different failure mode (bad output, not a failed request).
- **DB connection pool sizing is configurable**
  (`Settings.db_pool_size`/`db_max_overflow`, defaults 5/10) instead of
  SQLAlchemy's hardcoded defaults — `pool_pre_ping=True` was already set.
- **Security response headers** (`X-Content-Type-Options: nosniff`,
  `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, plus HSTS when
  `app_env=production`) and **gzip response compression**
  (`GZipMiddleware`, 1KB threshold) on every response.
- `/settings` reports `auth_enabled`, `rate_limit_enabled`, and the
  configured rate-limit window/threshold (never the actual API keys) —
  the frontend's Settings page displays this security posture.
- Two `FailureMode` values (`RATE_LIMITED` → 429, `UNAUTHORIZED` → 401)
  and their error classes, following the same pattern every other
  failure mode already uses.

Verification: `pip-audit` (backend) and `npm audit --omit=dev` (frontend)
both report zero known vulnerabilities in current dependencies.

## Design tradeoffs & roadmap

The choices below were made deliberately, favoring simplicity, testability,
and provider-independence over completeness. They're listed here as the
honest tradeoffs they are, not gaps to apologize for.

- **In-memory cache and trace storage.** `InMemoryCache` (the default
  when `REDIS_URL` is unset) and `TraceStore` are both process-local: they
  don't persist across restarts or coordinate across multiple worker
  processes. This is a real constraint for any deployment that needs
  cross-process shared caching or durable trace history — the fix is
  operational (point `REDIS_URL` at a real instance, persist `TraceStore`
  to the existing `events` table) rather than architectural, since both
  already sit behind an abstraction that a real backend can drop into.
  For rate limiting specifically, this stopped being just a documented
  caveat: an engineering audit correctly pointed out that a naive
  multi-worker deployment with `RATE_LIMIT_ENABLED=true` and no real
  Redis would silently multiply the effective limit by the worker count
  (each worker enforcing its own independent counter) rather than
  actually enforcing it. `api/main.py`'s `validate_runtime_config()` now
  fails app startup loudly for exactly that combination — `WORKERS` (a
  new setting, must match the real `--workers N`) greater than 1, rate
  limiting enabled, and no real `REDIS_URL` — while a single worker (the
  common local/dev case) is left alone, since its in-memory cache *is*
  the whole process and nothing is actually broken there. `GET /settings`
  also exposes the resolved `cache_backend` and `workers` so this is
  visible, not just enforced.
- **Sparse retrieval uses Postgres full-text search, not true BM25.**
  `ts_rank_cd` doesn't implement BM25's term-frequency saturation or
  document-length normalization. A real BM25 implementation (e.g. via the
  ParadeDB/`pg_search` extension) is a reasonable future upgrade rather
  than a correctness bug in the current design — full-text search was
  chosen to avoid an extra extension dependency for the sparse-retrieval
  role RRF fusion already tolerates being approximate in.
  Result caching for `/search`/`/retrieve` is likewise deferred until a
  deployment's latency/cost profile actually calls for it.
  
- **Fixed-window rate limiting over sliding-window/token-bucket.** Simpler
  to implement and reason about, at the cost of allowing up to 2x burst
  traffic at a window boundary — an accepted tradeoff for a first
  implementation, documented in the module docstring.
- **Regex-based prompt-injection detection, not semantic.** Bounded and
  deterministic, consistent with how contradiction detection is scoped,
  but a rephrased injection attempt that doesn't match a listed pattern
  won't be caught. `tests/adversarial/test_prompt_injection.py` documents
  this honestly with `xfail(strict=True)` cases (a reworded instruction,
  a non-English translation) that currently do bypass detection — proof
  of the gap's real boundary, not just a docstring claim. Patterns cover
  instruction-override, system-prompt probing, jailbreak phrasing,
  refusal-injection ("do not answer..."), and exfiltration-style
  ("send/forward this data...") attempts. A semantic/model-based detector
  is the natural next step if adversarial content becomes a real concern
  in a given deployment.
- **Deterministic, keyword-scoped contradiction detection.** Only numeric
  claims sharing one of a small fixed keyword set, framed as a
  percentage, are checked. This trades recall (it misses non-numeric or
  absolute-figure contradictions) for something that works identically
  regardless of which LLM provider is configured and needs no model call
  to run.
- **No OCR.** A scanned/image-only PDF parses with zero text elements
  rather than failing loudly. A useful follow-up would be flagging
  documents with suspiciously few extracted characters relative to page
  count rather than silently succeeding; full OCR support is future work.
  A related, real case found via a live end-to-end test (see Document
  ingestion above): a PDF whose embedded font has a broken ToUnicode
  mapping for certain letter pairs extracts *wrong* text with no error at
  all ("Term" as "Tenn") — OCR is the only real fix for that case too,
  since the corruption is in what the source file itself claims those
  glyphs are, not in how it's decoded.
- **Multi-hop chaining is bounded to exactly two hops.** `agents/multi_hop
  .py` (see The agentic retrieval loop above) closes the real two-hop
  gap — an entity extracted from hop one's evidence resolves hop two's
  query — but a genuinely longer dependency chain (three-plus hops,
  branching sub-questions) is a substantially larger undertaking
  (effectively a different class of system) and stays out of scope here.
- **API-key auth has no scoping, rotation, or persistence layer.** Keys
  live in `Settings.api_keys` (env-configured) only — sufficient for a
  single-tenant deployment, but a multi-tenant or higher-security
  deployment would need a `users`-table-backed identity system.
  Authentication and rate limiting are both implemented but **off by
  default**; a real deployment must explicitly configure `API_KEYS` and
  `RATE_LIMIT_ENABLED=true`.
- **Structured logs + a custom event system instead of OpenTelemetry
  spans.** Covers the same observability need with less integration
  surface for now; wiring an actual OTel exporter is a natural extension
  once a deployment needs to feed a broader tracing backend. No cost
  estimation is derived from the token/cache metrics that already exist.
- **GFM Markdown tables are not specially recognized** — a markdown
  table parses as plain paragraph text; adding the `mdit-py-plugins`
  table plugin is a small, deferred addition rather than a design
  problem. A block tag nested inside another matched block tag in HTML
  (e.g. a `<p>` inside a `<li>`) is emitted as two separate elements
  rather than merged.
- **No CI pipeline yet and no load/performance testing harness** (no
  k6/locust). Hardening work (connection pooling, retry/backoff) is
  reasoned engineering, verified locally and by the test suite, rather
  than validated under simulated concurrent load — a CI gate and a load
  test are the natural next additions before a production rollout.
