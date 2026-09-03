# Architecture

Status: Phase 5 (Reranking) complete, on top of Phase 1 (Foundation),
Phase 2 (Ingestion), Phase 3 (Chunking + embeddings + indexing), and
Phase 4 (Dense + sparse + hybrid retrieval, RRF). This document will grow
with each phase; it only describes what is actually implemented.

## Design decisions

### Local development without Docker

Docker is not runnable on the primary development machine (insufficient system
resources). `docker/docker-compose.yml` and `docker/Dockerfile` are still
maintained in the repo — for CI and for any environment that *can* run
Docker — but local day-to-day development uses:

- **PostgreSQL + pgvector**: a native Windows PostgreSQL 18 install with the
  pgvector 0.8.6 extension enabled directly (`CREATE EXTENSION vector;`),
  rather than the dockerized `pgvector/pgvector` image.
- **Redis**: intended to be a free-tier managed instance (e.g. Upstash), but
  no account exists yet. `storage/cache.py` auto-falls-back to an in-process
  `InMemoryCache` when `REDIS_URL` is unset or still the `.env.example`
  placeholder, so the app and test suite work without any external account.
  This is a genuine limitation for anything requiring cross-process
  coordination — see "Known limitations" below.

Everything is env-var driven (`DATABASE_URL`, `REDIS_URL`), so switching
between this local setup and the docker-compose stack is a config change, not
a code change.

### Single ORM models module

All SQLAlchemy tables (`storage/models.py`) live in one file rather than
split across `knowledge/`, `citations/`, `evaluation/`, etc. Domain modules
added in later phases will import ORM classes from here and layer
repository/business logic on top, rather than each owning its own table
definitions. Rationale: several tables cross domain boundaries by design
(e.g. `Citation` FKs into both `DocumentChunk` and `Answer`), which would
otherwise force circular imports between domain modules.

### Embedding dimension is fixed per deployment

`document_chunks.embedding` is `vector(384)` (see
`storage/models.py::EMBEDDING_DIMENSIONS`), matching the default local
embedding model family (e.g. `bge-small-en` / `all-MiniLM-L6-v2`, added in
the embeddings phase). pgvector columns have a static dimension — swapping to
a differently-sized embedding model requires a new migration that alters this
column and a full re-embed of existing chunks. The embedding *provider* stays
replaceable (spec principle: replaceable embedding models); the *dimension*
is a real, unavoidable constraint of pgvector, not something the provider
abstraction can hide.

### Migrations run with a sync driver; the app runs async

The app uses `asyncpg` (`postgresql+asyncpg://...`) for runtime queries.
Alembic migrations run synchronously, so `migrations/env.py` swaps in the
sync `psycopg` driver (`postgresql+psycopg://...`) derived from the same
`DATABASE_URL` just for running migrations. No separate migration-only URL is
configured — one env var, two derived connection strings.

Known quirk: Alembic's autogenerate does not know how to import
`pgvector.sqlalchemy` for `Vector`-typed columns. Any future
`alembic revision --autogenerate` touching `document_chunks.embedding` (or
another vector column) needs `import pgvector.sqlalchemy` added by hand to
the generated migration file before it will run — see
`migrations/versions/77d4e3cf344b_initial_schema.py` for the pattern.

### Windows + asyncpg event loop policy

Windows' default `ProactorEventLoop` is incompatible with asyncpg's
connection teardown (raises on close). `agentic_rag/__init__.py` sets
`WindowsSelectorEventLoopPolicy` at package-import time, before any event
loop is created — this runs for both the test suite and the running app.

### Python 3.11, not 3.12+

The stack recommendation calls for Python 3.12+. This machine has 3.11.9
(stable, all target dependencies have prebuilt wheels) and 3.14 (too new —
`asyncpg`/`psycopg` wheel availability is not yet reliable for it). 3.11 was
chosen for dependency stability; nothing in the codebase depends on a
3.12-only feature.

## What's implemented (Phase 1)

- `core/config.py` — env-driven `Settings` (Pydantic Settings), including the
  agentic-loop budget ceilings (max iterations/calls/tokens/latency) required
  by spec §1 even though nothing enforces them yet (added with the agentic
  retrieval loop in a later phase).
- `core/errors.py` / `core/models.py` — the full failure-mode error hierarchy
  (spec §35), so later phases raise these rather than inventing new ad-hoc
  exceptions.
- `storage/postgres.py`, `storage/models.py` — async engine/session factory
  and the complete schema from spec §28 (17 tables incl. `alembic_version`),
  migrated and verified against the real local Postgres.
- `storage/cache.py` — Redis-or-in-memory cache client factory (see above).
- `storage/vector_store.py` — `VectorStore` Protocol only; no implementation
  yet (lands with dense retrieval).
- `storage/object_store.py` — `ObjectStore` Protocol + a path-traversal-safe
  local filesystem implementation.
- `api/main.py` — FastAPI app with structured JSON logging, per-request
  trace-ID middleware/header, and domain-error → HTTP translation that never
  leaks stack traces.
- `GET /health` — reports Postgres and cache reachability independently.
- `observability/tracing.py` — structlog configuration + trace-ID context var.
- `docker/` — maintained, not run locally (see above).
- Tests: 6 passing (2 unit, 4 integration against the real local Postgres +
  FastAPI app). `ruff check` and `mypy --strict` both clean.

## What's implemented (Phase 2)

- `ingestion/parsed_document.py` — the common internal representation every
  parser normalizes into (`ParsedDocument` → `DocumentElement`s), independent
  of any database model, per spec §5.
- Seven format-specific parsers behind one `DocumentParser` protocol
  (`ingestion/parsers/`): PDF (PyMuPDF, heading detection via relative font
  size, per-page table extraction), DOCX (python-docx, walks the body's
  paragraphs/tables in document order, heading detection via paragraph
  style), Markdown (`markdown-it-py` token stream — headings, paragraphs,
  list items, fenced code blocks; GFM tables are a known gap, see below),
  HTML (BeautifulSoup, block-tag walk), CSV (one element per row rendered as
  `column: value` pairs), JSON (one element per record for a list-of-objects
  shape, otherwise the whole document as one element), and plain text.
- `ingestion/loaders/validation.py` — file type detection from extension,
  path-traversal-safe filename sanitization, empty/oversized upload
  rejection (spec §36).
- `ingestion/cleaners/text.py` — Unicode normalization, control-character
  stripping, whitespace cleanup, applied uniformly after parsing regardless
  of source format.
- `ingestion/pipeline.py` — orchestrates validate → detect type → parse →
  persist; re-uploading the same filename to the same collection creates a
  new `DocumentVersion` rather than a duplicate `Document`.
- `POST /collections`, `GET /collections`, `POST /documents` (multipart
  upload), `GET /documents`, `GET /documents/{id}` — wired to the pipeline
  above; verified against the real local Postgres and by an actual HTTP
  upload to a running server (not only the test client).
- Tests: 39 passing (unit: parsers incl. two using real generated PDF/DOCX
  bytes, validation, config; integration: pipeline + API against real
  Postgres; adversarial: malformed PDF/DOCX/JSON, path traversal, oversized
  upload, unsupported type, empty file). `ruff` and `mypy --strict` clean.

### Known gaps from Phase 2

- Markdown tables (GFM extension) are not specially recognized — no table
  plugin is installed for `markdown-it-py`; a markdown table parses as plain
  paragraph text. Adding proper support means adding the `mdit-py-plugins`
  dependency, deferred until something actually needs it.
- HTML/PDF/DOCX table and heading detection are heuristic/best-effort (font
  size for PDF headings, tag/style name for HTML/DOCX) — there is no
  universal "this is a heading" signal in any of these formats.
- A block tag nested inside another matched block tag in HTML (e.g. a `<p>`
  inside a `<li>`) is emitted as two separate elements rather than merged.
- No OCR — a scanned/image-only PDF will parse with zero text elements
  rather than failing loudly; this is worth a follow-up validation (e.g.
  flag documents with suspiciously few extracted characters relative to page
  count) rather than silent success, but is not implemented yet.
- Caption detection (spec §5) is only implemented for HTML `<figcaption>`;
  PDF/DOCX captions are not distinguished from regular paragraphs.

## What's implemented (Phase 3)

- `chunking/tokenization.py` — a fully offline, dependency-free tokenizer
  (whitespace-vs-non-whitespace runs), used instead of `tiktoken`. `tiktoken`
  needs to download its BPE merge table from a Microsoft blob-storage host on
  first use; that host turned out to be unreachable from this machine's
  Python specifically (DNS resolution fails for it even though PyPI and every
  other host used so far — Hugging Face Hub included — resolve fine). Rather
  than have chunk-token-budgeting depend on a network fetch, token counting
  is a small rule-based splitter: exactly reversible
  (`"".join(encode(text)) == text`), deterministic, but an approximation of
  any real LLM's actual subword tokenizer, not an exact match. If `tiktoken`
  becomes reachable in some environment this runs in, swapping it back in is
  contained to this one file.
- Four chunkers behind one `Chunker` protocol (`chunking/`): `FixedSizeChunker`
  (pure token-count sliding window, ignores structure — the baseline),
  `RecursiveChunker` (accumulates whole elements up to the token budget,
  recursively splits an oversized single element on a separator hierarchy),
  `StructuralChunker` (the default — groups elements into contiguous
  same-heading runs and never merges across a heading boundary; an oversized
  section gets one parent chunk covering the whole section plus child chunks
  from `RecursiveChunker`), and `SemanticChunker` (embeds sentences and
  breaks on a cosine-similarity drop *or* the token budget, whichever comes
  first — the one chunker that genuinely needs an embedding call rather than
  pure deterministic logic, per engineering principle #2).
- Parent/child chunk resolution (`chunking/pipeline.py`): chunks are flushed
  once to get real database-assigned UUIDs, *then* `parent_chunk_id` is
  resolved and a second flush persists it — SQLAlchemy applies a column's
  Python-side `default=` callable at flush time, not at object construction,
  so resolving parent references before the first flush silently produced
  `NULL` (`chunk.id` reads back `None` until flushed). Caught by an
  integration test asserting actual persisted `parent_chunk_id` values, not
  just the in-memory `ChunkCandidate.parent_index`.
- `embeddings/` — `EmbeddingProvider` protocol; `LocalEmbeddingProvider`
  (sentence-transformers `all-MiniLM-L6-v2`, 384-dim, CPU, runs in a thread
  via `asyncio.to_thread` since the library is synchronous); `MockEmbeddingProvider`
  (deterministic hash-seeded unit vectors, no model/network — default for
  tests and local dev); `OpenAIEmbeddingProvider` (plain `httpx` call to
  OpenAI's REST endpoint rather than pulling in the `openai` SDK for one call
  site; requests `dimensions=384` explicitly via OpenAI's Matryoshka
  truncation support so it matches the fixed pgvector column width regardless
  of which provider is configured).
- `embeddings/cache.py` — `CachedEmbeddingProvider` wraps any provider behind
  the shared cache client (Redis or in-memory), keyed by model name + text
  hash, so switching models never serves another model's stale vector.
- `POST /documents/{id}/ingest` — chunks + embeds + persists `DocumentChunk`
  rows for a document version (deliberately separate from `POST /documents`,
  which only parses + stores metadata); accepts optional per-call overrides
  for strategy/chunk size/overlap/similarity threshold, defaulting to the
  platform config. Verified against the real local Postgres+pgvector and via
  an actual HTTP request to a running server.
- Tests: 58 passing total (added: unit tests for all four chunkers including
  parent/child structure and oversized-element splitting, embedding provider
  determinism/caching; integration tests for the persisted chunk+embedding
  round trip through pgvector and the `/ingest` API). `ruff` and
  `mypy --strict` both clean.

### Known gaps from Phase 3

- `SemanticChunker`'s sentence splitting is a simple punctuation-based regex,
  not a real sentence boundary detector — it will mis-split on abbreviations
  ("e.g.", "Dr.") the way most lightweight splitters do.
- No embedding-model-dimension migration tooling — switching
  `EMBEDDING_DIMENSIONS` (e.g. to use a different local or remote model)
  still means writing a new Alembic migration and re-embedding by hand; see
  the note on this in the Phase 1 section above.
- `OpenAIEmbeddingProvider` is implemented but unexercised by any test that
  calls the real OpenAI API (no key configured in this environment) — its
  request/response shape is verified against OpenAI's documented API, not
  against a live call.

## What's implemented (Phase 4)

- `document_chunks.content_tsv` — a generated (`GENERATED ALWAYS AS ... STORED`)
  `tsvector` column plus a GIN index, added by hand-written migration
  `938c2d23d9a7` rather than `alembic revision --autogenerate` (autogenerate
  doesn't reliably reproduce Postgres's computed-column syntax). Verified
  with `alembic check` reporting no drift between the ORM model and the
  applied schema.
- `retrieval/dense.py` — `DenseRetriever`: pgvector cosine-distance search
  (`1 - distance` recovers similarity for the normalized vectors every
  embedding provider here produces), optional `score_threshold`, filterable.
- `retrieval/sparse.py` — `SparseRetriever`: PostgreSQL full-text search via
  `plainto_tsquery` + `ts_rank_cd` over `content_tsv`. `plainto_tsquery`
  (not `to_tsquery`) treats arbitrary user input as plain text rather than
  tsquery syntax, so operators/parentheses/quotes in a query can't break or
  redefine the search — exercised directly with adversarial input, not just
  assumed safe. This is Postgres full-text search, not literal BM25 — see
  the Phase 4 gap note below.
- `retrieval/filters.py` — `MetadataRetriever` (filter-only, no query) plus
  `build_filter_conditions()`, shared by every retriever. Filters are a fixed
  field list (collection, document type, document IDs, section, heading,
  source, year) rather than an open-ended key/value language — spec §9's
  "arbitrary safe metadata filters" is satisfied by that closed field list,
  not by sanitizing arbitrary filter expressions.
- `retrieval/fusion.py` — `reciprocal_rank_fusion()`: a pure function over
  ranked ID lists with zero DB/embedding dependency, per spec §9 "make
  fusion independently testable." Covered by unit tests that never touch a
  database.
- `retrieval/hybrid.py` — `HybridRetriever`: pulls a wider candidate pool
  (default 30) from dense and sparse independently, fuses via RRF, returns
  the top-k with dense/sparse/fusion scores all preserved per candidate
  (never collapsed into one number) — spec §14 provenance requirement,
  even though reranking itself is Phase 5.
- `POST /search` (simple, one score per result) and `POST /retrieve`
  (developer/debug view: per-method scores, explicit strategy selection
  among dense/sparse/hybrid) — spec §29. Verified against the real local
  Postgres+pgvector and via actual HTTP requests to a running server.
- Tests: 85 passing total (added: pure RRF unit tests, filter-condition
  unit tests, integration tests for all four retrievers and both API
  routes against real Postgres, adversarial tests for tsquery-breaking
  characters and SQL-injection-shaped query text). `ruff` and
  `mypy --strict` both clean.

### Known gaps from Phase 4

- Sparse retrieval is Postgres full-text search (`ts_rank_cd`), which is
  *not* true BM25 — no term-frequency saturation or document-length
  normalization the way BM25 defines it. Documented here rather than
  claimed as BM25-equivalent; a real BM25 implementation (e.g. via the
  ParadeDB/`pg_search` extension) is future work, not something this phase
  attempted.
- No result caching yet (spec §32) — every `/search`/`/retrieve` call hits
  Postgres directly. Retrieval-result caching is deferred to whichever
  phase actually needs the latency/cost savings.
- The full-text search config is hardcoded to `"english"` — not yet
  per-collection configurable.
- Persistence of retrieval runs (`retrieval_runs`/`retrieved_chunks` tables,
  already in the schema since Phase 1) is not wired up yet — these
  retrievers are called directly, not yet through the query/plan/trace
  machinery that Phase 6/7 (query planning, the agentic retrieval loop)
  will add.

## What's implemented (Phase 5)

- `retrieval/reranking.py` — `Reranker` protocol: `rerank(query, candidates,
  top_k)` sets `rerank_score` on each candidate and returns the top-k by
  that score, leaving `dense_score`/`sparse_score`/`fusion_score` untouched
  — never collapsing scores into one number (spec §14 provenance).
  `MockReranker` (deterministic query/content term-overlap, no model or
  network — same role as `MockEmbeddingProvider`) and
  `LocalCrossEncoderReranker` (sentence-transformers `CrossEncoder`,
  `cross-encoder/ms-marco-MiniLM-L-6-v2`, CPU, no API key) sit behind it.
  Verified with a real (non-mocked) model download and inference call: for
  the query "What happened to revenue?" the cross-encoder scored a revenue
  passage at ~-0.001 and an unrelated weather passage at ~-11.2 — a large,
  correctly-ordered separation from a genuine semantic signal, not just
  plumbing that runs without error.
- `POST /retrieve` gained `rerank`/`rerank_top_k` fields. When
  `rerank=true`, retrieval first fetches `candidate_pool_size` candidates
  (the "top 20-30" from spec §14) and the reranker narrows that down to
  `rerank_top_k` (the "top 5-10 evidence chunks"); `rank` is reassigned to
  reflect the post-rerank order, since the pre-rerank rank is stale once
  results are reordered.
- Tests: 93 passing total (added: unit tests for both rerankers including
  empty-query/empty-candidates edges; integration tests for `/retrieve` with
  `rerank=true/false` against real Postgres). `ruff` and `mypy --strict`
  both clean.

### Known gaps from Phase 5

- No remote reranker (e.g. Cohere Rerank) — only mock and local. The
  `Reranker` protocol is the same shape a remote provider would implement
  (mirroring how `OpenAIEmbeddingProvider` slots into `EmbeddingProvider` in
  Phase 3), but adding one wasn't done speculatively without a concrete need.
- `MockReranker`'s term-overlap scoring is intentionally not semantically
  meaningful — same caveat as `MockEmbeddingProvider`.

## Known limitations

- No CI pipeline yet.
- No real Redis instance in use locally — `InMemoryCache` does not persist
  across restarts or coordinate across processes, so anything built on top of
  it (rate limiting, cross-worker cache) will not behave correctly under
  multiple app processes until a real Redis is configured.
- No authentication/authorization implemented — `security/` is empty; user
  identity in the schema (`users` table) exists but nothing populates or
  checks it yet.
- OpenTelemetry/Prometheus exporters are not wired up yet — only structured
  JSON logs and a trace-ID today. Full observability lands in its own phase.
