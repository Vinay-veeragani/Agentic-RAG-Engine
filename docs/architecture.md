# Architecture

Status: Phase 11 (Observability + streaming) complete, on top of Phase 1
(Foundation), Phase 2 (Ingestion), Phase 3 (Chunking + embeddings +
indexing), Phase 4 (Dense + sparse + hybrid retrieval, RRF), Phase 5
(Reranking), Phase 6 (Query analysis + retrieval planning), Phase 7
(Agentic retrieval loop), Phase 8 (Evidence evaluation + contradiction
detection), Phase 9 (Answer synthesis + citations + citation validation),
and Phase 10 (Evaluation framework). This document will grow with each
phase; it only describes what is actually implemented.

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

## What's implemented (Phase 6)

- `generation/llm.py` — `LLMProvider` protocol: `complete()` (raw text,
  mirroring every real chat API) plus `complete_structured()`, provided by
  default via `BaseLLMProvider` in terms of `complete()` — schema-instructed
  prompting (the pydantic JSON Schema is embedded in the system prompt),
  JSON extraction (handles a bare object, a markdown code fence, or prose
  wrapping one), pydantic validation, and **one retry** with the validation
  error fed back to the model before raising `ModelProviderError`. This is
  the first LLM-reasoning component in the codebase — introduced here
  because query classification/expansion/decomposition are exactly the kind
  of language-understanding task deterministic rules handle poorly in
  general (engineering principle #2), unlike RRF or chunk windowing.
- Three providers: `MockLLMProvider` (see below), `OllamaLLMProvider`
  (local, `/api/chat` with `format: "json"`, no API key — implemented but
  **not exercised by any test**: Ollama isn't installed/running on this
  machine), `OpenAILLMProvider` (plain `httpx` call, same "implemented but
  unexercised without a key" status as `OpenAIEmbeddingProvider`). Anthropic
  and Gemini are not implemented — the same `LLMProvider` protocol would
  support them, but adding unexercised provider code with no way to verify
  it wasn't done speculatively.
- `generation/mock.py` — `MockLLMProvider.complete_structured` does not go
  through `BaseLLMProvider`'s wrapper; it directly introspects the
  *requested pydantic schema type* (via `model_fields`) and fills each field
  deterministically — keyword/pattern heuristics for fields whose name
  signals intent (`query_type`, `decompose`, `subqueries`, `reasoning`, ...),
  generic rules by Python type otherwise (bool/int/float/str/Enum/
  `list[str]`/`list[BaseModel]`/nested `BaseModel`). This is real
  architectural investment, not a shortcut: every future agent built on
  `complete_structured` (evidence evaluation, citation validation, answer
  synthesis in later phases) gets a working offline mock for free, the same
  role `MockEmbeddingProvider`/`MockReranker` play for their layers. Query
  heuristics operate on text extracted from a `"Query: <text>"` line in the
  prompt specifically — not the raw prompt blob — so that e.g. the
  planner's prompt (which appends a JSON classification blob after the
  query) doesn't feed unrelated JSON text into the same keyword rules; this
  was a real bug caught while manually exercising the mock end-to-end,
  fixed before it reached tests.
- `agents/query_analyzer.py` — `QueryAnalyzer` (spec §10: classifies into
  `QueryType`, returns structured `is_ambiguous`/`is_answerable`/`reasoning`,
  never free-form planning text) and `QueryExpander` (spec §11: proposes up
  to 5 phrasing variants; only invoked when the plan says to — "do not
  blindly expand every query").
- `agents/planner.py` — `RetrievalPlanner` (spec §13: decides strategy,
  `expand_query`, `decompose`, `max_iterations`, `top_k`, reusing
  `retrieval.base.MetadataFilter` directly rather than a parallel schema)
  and `QueryDecomposer` (spec §12: splits a complex query into up to 8
  standalone subqueries). **The planner's `max_iterations`/`top_k` are
  always clamped to configured ceilings after the LLM (or mock) call** —
  spec §13 "the planner must be bounded and validated," enforced in code,
  not just prompted for; covered by a unit test that sets a ceiling of 1 and
  asserts the mock's default of 3 gets clamped down regardless.
- `POST /query/analyze` — a preview endpoint (not one of spec §29's listed
  routes) exposing analysis + plan + conditional expansion/decomposition
  ahead of the full agentic loop that will actually execute a plan like this
  in Phase 7. Verified via an actual HTTP request to a running server.
- Tests: 119 passing total (added: `BaseLLMProvider` retry-then-raise
  behavior against a scripted fake provider; `MockLLMProvider` schema-fill
  correctness across query-type heuristics, nested/list fields, and
  determinism; the four agent classes; the API route; an empty-query
  rejection). `ruff` and `mypy --strict` both clean.

### Known gaps from Phase 6

- No Anthropic or Gemini `LLMProvider` implementation yet (see above).
- `MockLLMProvider`'s heuristics are tuned to this codebase's actual prompt
  schemas (`QueryAnalysis`, `RetrievalPlan`, `QueryExpansion`,
  `QueryDecomposition`) and field-naming conventions — it is not a generic
  pydantic-schema-to-plausible-data filler for arbitrary external schemas,
  though the type-based fallback rules (Enum/bool/int/float/str/list/nested
  model) do generalize.
- `QueryDecomposer`'s mock-path splitting (`" and "`/comma) is a naive
  heuristic, not real dependency-aware decomposition — multi-hop subquery
  *dependencies* (spec §17: "which companies... **and** what were the
  acquisition values") are explicitly Phase 7's concern, not implemented
  here.
- Nothing from this phase is wired into the retrieval endpoints yet —
  `/query/analyze` is a standalone preview; the agentic loop that actually
  executes a plan against `/retrieve` is Phase 7.

## What's implemented (Phase 7)

- `agents/evidence_agent.py` — `EvidenceAgent.assess(query, candidates)` ->
  `EvidenceAssessment{sufficient, reason, missing_information}` (spec §15's
  lightweight version — the fuller judgment with source authority,
  directness, temporal correctness, and contradiction detection is Phase
  8's scope; building that here would get ahead of the phase meant to
  implement it). No candidates always short-circuits to `sufficient=False`
  without an LLM call.
- `agents/retrieval_agent.py` — `RetrievalAgent.retrieve(queries, plan)`:
  dispatches to dense/sparse/hybrid per the plan's strategy, and — new here
  — supports *multiple query variants at once* (used when the plan enabled
  expansion) by retrieving each variant independently and fusing the
  per-variant rankings with the same `reciprocal_rank_fusion` that combines
  dense+sparse in Phase 4, applied one level up. Always reranks down to a
  final evidence count (default 8, within spec's suggested 5-10) and
  reassigns `rank` post-fusion/rerank.
- `agents/research_agent.py` — `AgenticRetrievalLoop.run()`: the bounded
  plan -> retrieve -> rerank -> evaluate -> refine-if-insufficient loop
  (spec §16). Hard-bounded *by construction*: the loop body is a
  `for iteration in range(1, plan.max_iterations + 1)`, and
  `max_iterations` is already clamped to `settings.max_retrieval_iterations`
  by `RetrievalPlanner` (Phase 6) before this ever runs — a bounded `for`
  loop cannot run forever regardless of what any LLM (or the mock)
  proposes. `settings.max_retrieval_calls` is checked independently every
  iteration, since query decomposition (spec §12/§17) can turn one
  iteration into several retrieval calls. Query refinement between
  iterations is deterministic (`original_query + " " + missing_information`)
  rather than another LLM call — a defensible principle-#1 choice, and one
  fewer place a bad structured-output response could break the loop.
  Every run ends in exactly one `TerminationReason`
  (`sufficient_evidence` / `max_iterations_reached` /
  `max_retrieval_calls_reached` / `no_evidence_found`).
- **Two real bugs found and fixed while manually exercising the loop against
  real Postgres before writing any test** (both are exactly the kind of
  thing that would have been invisible in a design review, only visible by
  running it):
  1. `asyncio.gather()` across query variants / decomposed subqueries, all
     sharing one `AsyncSession`, threw
     `InvalidRequestError: This session is provisioning a new connection;
     concurrent operations are not permitted` — SQLAlchemy's async session
     is not safe for concurrent use from multiple coroutines. Fixed by
     retrieving sequentially instead. Spec §12 itself says "execute
     concurrently *where safe*" — this is the "where safe" carve-out, not
     a shortcut around the spec.
  2. A decomposed query's subqueries could burn through
     `max_retrieval_calls` within a single outer iteration, since the
     budget check only ran once per iteration, not once per subquery.
     Fixed by truncating the subquery list to the remaining budget before
     retrieving.
- `agents/planner.py`'s `MetadataFilter` reuse pays off here: the API layer
  passes a caller-supplied `collection_id` in, and it wins over whatever
  the plan guessed (the LLM/mock has no way to know a real collection ID) —
  verified end-to-end that retrieval stays correctly scoped to the given
  collection and returns `NO_EVIDENCE_FOUND` for an empty or wrong one.
- `generation/mock.py` gained an `EvidenceAssessment`-aware path: it now
  extracts an `"Evidence:\n..."` block from the prompt (alongside the
  existing `"Query: ..."` extraction) and judges sufficiency by lexical
  overlap between query and evidence terms — enough to make "insufficient
  evidence -> refine -> retry" and "sufficient evidence -> stop"
  genuinely exercisable offline, not just structurally plausible.
- `POST /query/retrieve` — a preview endpoint (like `/query/analyze`, not
  one of spec §29's listed routes) returning the full structured trace and
  final evidence, no synthesized answer yet. Verified against a running
  server: querying "Why did revenue decline?" against an indexed document
  produced a correct one-iteration `sufficient_evidence` trace with real
  dense/sparse/fusion/rerank scores on the returned evidence.
- Tests: 136 passing total (added: evidence agent sufficiency heuristics;
  pure fusion/refinement/dedup helpers; five loop-termination scenarios
  against real Postgres — sufficient evidence, iteration budget exhausted,
  empty collection, cross-collection scoping, retrieval-call budget
  exhausted mid-decomposition; the API route). `ruff` and `mypy --strict`
  both clean.

### Known gaps from Phase 7

- Nothing from this phase persists to the `queries`/`query_plans`/
  `retrieval_runs`/`evidence_items` tables yet (all present in the schema
  since Phase 1) — the loop returns its full trace in-memory/over the API
  response, but there is no `GET /queries/{id}/trace` to replay a past run.
  Wiring that up is deferred to the observability phase, where the
  telemetry/tracing story is actually being built out.
- Query decomposition treats subqueries as independent parallel retrievals,
  not a dependency graph — spec §17's example ("identify acquisitions ->
  identify target companies -> retrieve acquisition values -> ...", where
  each step's output feeds the next step's query) is not implemented; that
  is genuinely hard (dependency-aware multi-hop chaining) and was
  explicitly not attempted here, per the instruction not to turn this
  project into #5 (the future Deep Research system) prematurely.
- Evidence sufficiency is still the lightweight version — no contradiction
  detection, source authority weighting, or temporal-correctness checking
  yet (Phase 8).
- The loop always reranks (no way to skip it) and evidence count is a fixed
  default (8) rather than plan-configurable.

## What's implemented (Phase 8)

- `agents/evidence_agent.py` grew from Phase 7's lightweight sufficiency
  check into the fuller spec §15 judgment, split by how much genuine
  reasoning each piece needs (principle #1/#2):
  - **Relevance/coverage/directness** (spec §15): added as LLM-judged
    0.0-1.0 fields on `EvidenceAssessment`, alongside the existing
    sufficient/reason/missing_information.
  - **Contradiction detection** (spec §18): fully deterministic — regex
    extracts `(metric keyword, percentage)` pairs (e.g. "revenue ... 4%")
    from evidence content; two different values for the same keyword from
    *different* documents is a `Contradiction`. This does not attempt
    general semantic contradiction detection (two sources disagreeing in
    prose with no shared number) — that would need real LLM reasoning and
    is a documented gap. What it does catch matches this spec section's own
    worked example ("Revenue declined 4%.") exactly, and works identically
    for every provider including the mock, since it's not an LLM call at all.
  - **Source authority** (spec §20): `Collection.source_authority_config`
    (a schema column that existed since Phase 1 but was unused until now)
    holds a per-collection `{"order": [...]}` list of source labels,
    most-to-least authoritative — configurable via
    `POST /collections {"source_authority_order": [...]}`, never hardcoded
    as universally correct. A `Contradiction.resolution` is set *only* when
    the two sources' configured ranks actually differ — a
    provenance-based preference, explicitly not a claim about which
    content is factually correct. Equal or unclassified ranks leave
    `resolution=None`, surfacing the conflict as-is rather than inventing
    one (spec §18: "if the system cannot resolve the conflict, tell the
    user explicitly").
  - **Temporal awareness** (spec §19): regex year extraction across
    evidence content, surfaced as `years_referenced`/`spans_multiple_periods`
    — informational for now, not a hard block on mixing periods.
  - `Document.source` (existing column, previously always `None`) is now
    settable at upload (`POST /documents` gained a `source` form field,
    e.g. `"Annual Report"`) and threaded through every retriever into
    `RetrievedCandidate.document_source`, which authority resolution reads.
- `agents/research_agent.py`: the loop now calls `EvidenceAgent.evaluate()`
  (not the old `assess()`) and builds the evidence agent *per run*, loading
  the target collection's authority config first. An **unresolved**
  contradiction ends the run immediately with the new
  `TerminationReason.CONFLICTING_EVIDENCE` — refining the search query
  cannot fix two sources genuinely disagreeing, so continuing would just
  burn iteration budget pretending the problem is retrieval quality. A
  contradiction the authority policy *does* resolve does not block the
  loop; it's still surfaced in the iteration trace either way.
- **A real bug found and fixed by manually exercising authority resolution
  end-to-end before trusting the feature**: `retrieval/hybrid.py`'s
  `_merge()` (written in Phase 4, before `document_source` existed)
  rebuilt each `RetrievedCandidate` field-by-field and simply didn't
  include it, so hybrid retrieval silently dropped source information dense
  and sparse retrieval both populated correctly. Caught because a
  same-scenario authority-resolution test failed with `resolution=None`
  when it should have resolved — invisible without actually running the
  full retrieve→evaluate path, not something a unit test of `_merge` in
  isolation would have caught either, since that test would need to know
  to check this specific field. Fixed, and a regression test now asserts
  `document_source` survives hybrid fusion specifically.
- Tests: 147 passing total (added: contradiction detection — cross-document
  numeric conflicts, same-document exclusion, authority-based resolution,
  unclassified sources; relevance/coverage/directness scoring; year
  extraction; five new loop/API scenarios for conflicting vs.
  authority-resolved evidence; the hybrid-fusion regression test). `ruff`
  and `mypy --strict` both clean.

### Known gaps from Phase 8

- Contradiction detection only catches numeric claims sharing one of a
  small fixed set of keywords (revenue/profit/margin/growth/decline/
  earnings/sales/income) framed as a percentage. Non-numeric semantic
  contradictions, contradictions in absolute figures (not just
  percentages), and contradictions using unlisted keywords are not
  detected — a real limitation of the deterministic-only approach, traded
  for it working identically and reliably regardless of which LLM
  provider (or the mock) is configured.
- Temporal awareness is informational only — a query spanning multiple
  years doesn't get special handling beyond being visible in the trace;
  spec §19's stronger requirement ("must produce temporally separated
  evidence") isn't implemented.
- `Contradiction.resolution` text is a fixed template, not model-generated
  — deliberate (spec §18 says never invent a resolution), but means the
  same two-source conflict always produces identically-worded resolution
  text.

## What's implemented (Phase 9)

- **The key design decision this phase**: the LLM is never asked to
  produce a real citation ID (chunk/document UUID) — only a small 1-based
  index (`[1]`, `[2]`, ...) into the evidence list it was shown in the
  prompt. `citations/resolver.py::resolve_citations()` is the *only* place
  those indices become real IDs, by looking them up in the same Python
  evidence list the prompt was built from. An index outside that list's
  actual range is silently dropped, never guessed at. This makes "never
  fabricate a citation" (spec §22) an structural guarantee rather than a
  prompt instruction hoping the model complies — the same pattern already
  used for chunking/embedding indices in earlier phases, applied here to
  the highest-stakes case in the whole system.
- `agents/synthesis_agent.py` — `SynthesisAgent` (spec §21): given a query
  and evidence, produces discrete claims each carrying its own
  `evidence_indices`. No evidence at all is a deterministic short-circuit
  (`insufficient_evidence=True`, no LLM call) — nothing to synthesize from,
  matching `EvidenceAgent`'s and `MetadataFilter`'s established pattern.
- `agents/citation_agent.py` — `CitationAgent` (spec §23): checks whether a
  claim's *cited* evidence actually entails it (not merely relates to the
  same topic) — a genuine language-understanding judgment, so it goes
  through the LLM. A claim with zero citations is trivially unsupported
  without needing a model call.
- `agents/verifier.py` — `AnswerVerifier` (spec §24, "groundedness"):
  assembles the final answer from *only* the claims whose citations passed
  entailment validation — literally dropping unsupported ones and rejoining
  the rest, rather than asking an LLM to "edit" its own prior answer. If
  every claim gets dropped, the whole answer becomes
  `AnswerStatus.INSUFFICIENT_EVIDENCE`, not an empty-but-"grounded" answer.
- `citations/formatter.py` — deterministic display formatting matching
  spec §22's example exactly (`"[1] Annual Report, page 42, Revenue
  Recognition"`, falling back to filename when no source/page/section are
  set).
- `citations/validator.py` — `citation_precision` (fraction of proposed
  citations that were actually entailed) and `citation_completeness`
  (fraction of claims that ended up with a validated citation), pure
  arithmetic over already-computed results. `citation_recall` (spec §33)
  needs a ground-truth relevant-chunk set from an evaluation dataset and
  isn't computable from one live query — that's Phase 10's job.
- `POST /query` — spec §29's actual named endpoint, finally implemented for
  real: runs the full agentic retrieval loop, then — unless it ended in
  `CONFLICTING_EVIDENCE` or `NO_EVIDENCE_FOUND`, in which case synthesis is
  skipped entirely rather than attempting to paper over the problem —
  synthesizes and citation-validates an answer. `/query/analyze` and
  `/query/retrieve` remain as lower-level preview endpoints (useful for a
  future Developer Mode UI showing the full pipeline breakdown). Verified
  against a running server: a real query against real indexed content
  produced a grounded answer with `citation_completeness`/
  `citation_precision` both 1.0 and a correctly page/section/source-labeled
  citation.
- **A real bug found by manually exercising the pipeline before writing any
  test**: the mock LLM's generic string-fallback (used for the synthesized
  claim's `text` field) built the claim from the *query* text with added
  filler words ("mock value for: ..."). When that same claim text was later
  fed back into citation validation as the thing being checked against
  evidence, the filler words diluted the lexical-overlap ratio below the
  entailment threshold — so even directly-relevant evidence got its claim
  rejected as unsupported. Fixed by deriving the mock's claim text from the
  *evidence* it was given instead of the query, which is also more
  semantically honest: a claim quoting its source evidence should of course
  be entailed by it.
- Tests: 167 passing total (added: citation resolution/formatting/metrics
  as pure functions; synthesis/citation-agent/verifier unit tests including
  a scripted-fake-LLM test proving unsupported claims actually get removed
  from the final answer; five `/query` integration scenarios — grounded,
  no evidence, conflicting evidence, insufficient evidence, empty-query
  rejection). `ruff` and `mypy --strict` both clean.

### Known gaps from Phase 9

- Citation entailment validation is a single LLM call per claim with no
  retry-with-different-evidence — if evidence exists but doesn't quite
  support a claim, the claim is dropped, not repaired or re-retrieved for.
- No `url` field is ever populated on a citation (spec §22 lists it as
  optional/conditional) — nothing in the ingestion pipeline captures a
  source URL yet.
- `POST /query` always runs the full loop from scratch; there is no
  conversation memory yet linking a follow-up query to a prior one (spec
  §26 — not attempted this phase).
- Citation formatting always numbers citations `[1]`, `[2]`, ... in the
  order claims were validated, with no deduplication if two different
  claims cite the same underlying chunk (it will appear twice, with two
  different numbers).

## What's implemented (Phase 10)

- `evaluation/datasets.py` — a small, self-contained benchmark corpus (15
  synthetic documents: 7 finance-related plus 8 topic-distinct distractors)
  and 9 cases covering every category spec §33 lists (simple factual x2,
  multi-hop via comparison, comparison, temporal, analytical, aggregation,
  ambiguous, unanswerable, contradictory evidence). `build_benchmark_corpus()`
  ingests+indexes this corpus fresh through the *real* pipeline and resolves
  each case's ground-truth relevant documents from the real `Document.id`
  values that ingestion just created — nothing here is a hardcoded chunk ID.
- `evaluation/retrieval.py` — Recall@K/Precision@K/MRR/NDCG/Hit Rate as pure
  functions, unit-tested against hand-computed expected values.
- `evaluation/baseline.py` — the literal spec §34 baseline: Query -> Dense
  Retrieval -> Top-K -> LLM. No planning, no reranking, no evidence
  judgment, no citations, no bounded refinement.
- `evaluation/generation.py` — `GenerationJudge.judge_answer_relevance()`,
  the one genuinely new LLM-judge call this phase adds. `faithfulness` and
  `context_relevance` are *not* separately re-judged — they're exactly what
  citation_precision (Phase 9) and `EvidenceAssessment.relevance` (Phase 8)
  already measure, and recomputing them with a second judge would be
  redundant, not more rigorous.
- `evaluation/citations.py` — aggregates per-case `CitationQualityMetrics`
  into corpus-wide means. `citation_recall` (spec §33) needs a ground-truth
  "which chunks must be cited" label this benchmark's fixtures don't carry
  (only document-level relevance, for retrieval metrics) — not computed,
  a documented gap rather than a silent approximation.
- `evaluation/runner.py::run_benchmark()` — runs both pipelines over every
  case and assembles a `BenchmarkReport`. Retrieval-metric *means* exclude
  the ambiguous/unanswerable cases (they test correct abstention, not
  ranking quality) — each case's own numbers are still visible individually.
- `benchmarks/run_evaluation.py` — an actually-runnable CLI script
  (`python benchmarks/run_evaluation.py --embedding local --llm mock`)
  producing a printed comparison table and a JSON report
  (`benchmarks/results/latest.json`, committed as evidence of a real run —
  see numbers below). Defaults to the *local* sentence-transformers
  embedding provider rather than the mock one specifically so retrieval
  metrics measure real retrieval quality, not noise.

### Three real bugs found by actually running the benchmark, not by writing tests in isolation

Every one of these was invisible until real corpus text hit the Phase 8
contradiction detector — exactly the value a real (if small) benchmark run
provides over unit tests written against hand-picked examples:

1. **The metric-pattern regex only matched a literal `%` symbol.** This
   repo's own benchmark corpus (and most real prose) spells out "percent."
   The known contradictory-evidence case silently failed to be detected.
   Fixed by accepting `%`, `percent`, and `per cent`.
2. **Different time periods were flagged as contradictions.** A 2023 margin
   of 22% and a 2024 margin of 25% in two different documents were flagged
   as conflicting — they're just different years (spec §19's exact
   concern). Fixed by extracting each candidate's referenced years and
   skipping the flag when both candidates reference years and those years
   don't overlap. A follow-on discovery: a document that mentions its own
   year *and* a comparison year ("a 25 percent increase over 2023") pollutes
   its year-set — worked around by rewording the corpus to keep each
   year-specific document self-contained, since disambiguating "which year
   this specific number is about" within one chunk is a harder problem than
   this phase set out to solve.
3. **Contradiction detection considered the entire evidence list, unscoped
   by relevance.** With a small corpus and a padded-out top-8 evidence
   pool, an unrelated conflict ranked 7th of 8 was flagging *every* query
   that happened to retrieve it — including queries with nothing to do
   with that conflict. Fixed by scoping the check to the top 4 candidates
   by rerank rank, on the reasoning that a low-relevance chunk that only
   made it into the pool to fill out top_k shouldn't be able to veto an
   otherwise-clean answer. This is a real architectural finding, not a
   corpus-tuning trick: a production corpus large enough that top_k is a
   small fraction of it would show this less often, but the *unscoped*
   check was still wrong in principle regardless of corpus size.

### Real benchmark results (captured `2026-09-03`, `--embedding local --llm mock`)

| Metric | Baseline | Agentic |
|---|---|---|
| Recall@5 (excl. ambiguous/unanswerable) | 1.000 | 1.000 |
| Precision@5 | 0.286 | 0.286 |
| MRR | 0.929 | 0.857 |
| NDCG@5 | 0.947 | 0.903 |
| Hit Rate@5 | 1.000 | 1.000 |
| Mean latency | 0.041s | 0.090s |
| Citation precision | n/a (no citations) | 1.000 |
| Citation completeness | n/a | 1.000 |

Retrieval metrics are nearly identical between the two pipelines on this
small, low-ambiguity corpus — expected, since both ultimately search the
same indexed content, and the corpus isn't hard enough for reranking/
expansion/decomposition to visibly move the needle. **The qualitative
difference is where the two pipelines diverge in behavior, not their
retrieval scores**: for the "unanswerable" case (a query with no relevant
document at all), the agentic pipeline correctly returns
`insufficient_evidence` rather than a guess; for the "contradictory_evidence"
case (two sources reporting different numbers with no way to prefer one),
it correctly returns `conflicting_evidence` and surfaces the specific
conflicting claims rather than confidently reporting one arbitrary number.
The baseline pipeline has no such option — it always produces *an* answer,
with no signal to the caller about whether the underlying evidence
actually supported it. `mean_answer_relevance` and generation-quality
numbers from this specific run are not meaningful measures of real
generation quality: the mock LLM's answers are lexical excerpts of
evidence, not created language, since no real LLM provider (Ollama, OpenAI)
is available in this environment — see below.

### Known gaps from Phase 10

- **Generation-quality metrics from this environment's benchmark run are
  plumbing-only, not a real quality signal.** `MockLLMProvider` doesn't
  generate language — it excerpts evidence text deterministically. A
  meaningful faithfulness/answer-relevance/generation-quality comparison
  needs a real LLM provider (`--llm ollama` once Ollama is installed, or
  `--llm openai` with a key); the retrieval-metric comparison is real and
  provider-independent (driven by real local embeddings + real retrieval
  code), but the generation side of this specific captured run is not
  evidence of real answer quality.
- `citation_recall` is not computed (see above).
- `estimated_tokens` only counts the final answer text via this repo's own
  offline tokenizer — not the prompt/context tokens actually sent to an
  LLM, and not a real provider-reported usage figure. It is a rough,
  consistent proxy, not a cost estimate.
- The 15-document corpus is deliberately small for a fast, self-contained,
  repo-committed benchmark. It is large enough to demonstrate the
  qualitative baseline-vs-agentic differences above, but a corpus this
  small cannot meaningfully stress hybrid fusion, reranking, or
  decomposition the way a realistic multi-thousand-chunk corpus would.
- Nothing here persists to the `evaluation_datasets`/`evaluation_cases`/
  `evaluation_results` tables (present in the schema since Phase 1) — the
  benchmark corpus is rebuilt fresh on every run and results are only
  written to a JSON file, not the database.

## What's implemented (Phase 11)

- `observability/events.py` — `EventType` (spec §30's exact 13 event names:
  `query.started`, `query.analyzed`, `plan.created`, `retrieval.started`,
  `retrieval.completed`, `reranking.started`, `reranking.completed`,
  `evidence.evaluated`, `retrieval.refined`, `generation.started`,
  `citation.validation.started`, `query.completed`, `query.failed`),
  `Event` (event ID, query/trace ID, timestamp, type, structured payload —
  nothing else, never hidden chain-of-thought), and `EventEmitter`, which is
  dual-purpose: it always appends to an in-memory list (read back after a
  non-streaming call completes) and, when constructed with `queue=True`,
  also pushes onto an `asyncio.Queue` a concurrent SSE endpoint drains live.
- `observability/metrics.py` — a small, deliberately narrow set of
  Prometheus instruments, one per thing spec §31 actually asks to track:
  per-phase latency histograms (query/retrieval/rerank/generation),
  retrieval-iteration and estimated-token histograms, cache hit/miss and
  query-failure counters. Exposed via `GET /metrics` in standard Prometheus
  text exposition format — scrapeable without a custom exporter.
- Every phase of the pipeline is instrumented for real, not just
  logged-and-forgotten: `RetrievalAgent.retrieve()` now returns a
  `RetrievalOutcome` carrying separately-measured retrieval and rerank
  latency (not just candidates) and optionally emits
  `retrieval.started/completed`/`reranking.started/completed`;
  `AgenticRetrievalLoop` emits `query.started/analyzed`, `plan.created`,
  `evidence.evaluated`, `retrieval.refined`, records
  `QUERY_LATENCY_SECONDS`/`RETRIEVAL_ITERATIONS`, and accumulates per-phase
  latency onto both `IterationTrace` and the final
  `AgenticRetrievalResult`; `AnswerVerifier` emits `generation.started`/
  `citation.validation.started` and records its own latency separately.
  `embeddings/cache.py` records real cache hit/miss counts (previously
  untracked since Phase 3).
- **A real design correction made while wiring this up**: `query.completed`
  was initially emitted inside `AgenticRetrievalLoop.run()` itself — but
  that loop is *also* the first stage of the full `POST /query` pipeline
  (synthesis + citation validation follow it there), so a caller using the
  full pipeline would see `query.completed` fire before synthesis even
  started. Fixed by moving that emission to whichever caller actually
  finishes the whole pipeline (`POST /query/retrieve` right after the loop,
  `POST /query` after synthesis+validation) — `run()` itself only records
  its own metrics and emits `query.failed` on an internal exception.
- **Another real bug caught while wiring the emitter through the routes**:
  `main.py`'s existing middleware already calls `bind_trace_id()` once per
  request (used for the `x-trace-id` response header) — the new route code
  was calling `bind_trace_id()` *again*, generating a second, different
  trace ID than the one already in the response header, so a client
  comparing `X-Trace-Id` against the response body's `trace_id` field would
  see two different values. Fixed by having routes read the
  already-bound ID via `get_trace_id()` instead of rebinding.
- `POST /query/stream` — SSE version of `POST /query`: the same
  `_run_query_pipeline()` helper runs as a background `asyncio.Task` while
  the route drains `EventEmitter`'s queue and yields each event as a
  `text/event-stream` frame (`id:`/`event:`/`data:` lines) as it happens.
  Verified against a running server: a real query produced the exact spec
  §30 event sequence in order, ending in `query.completed`.
- `GET /queries/{trace_id}/trace` — spec §29's trace endpoint, backed by
  `TraceStore`: process-local, in-memory only, bounded to the 200 most
  recent traces (LRU eviction) — not persisted across restarts or shared
  across worker processes, a documented gap rather than a silent
  approximation. Both `POST /query` and `POST /query/retrieve` store their
  emitted events here so a trace is queryable immediately after the call
  that produced it returns.
- `GET /metrics` — Prometheus text exposition format via
  `prometheus_client.generate_latest()`.
- **Found and fixed a real, pre-existing test-isolation bug while running
  the full suite after this phase's changes**: several retrieval tests
  queried without a `collection_id` filter, silently relying on being the
  only data in a small local database. Months of accumulated manual
  smoke-testing and benchmark runs against the same shared local Postgres
  (documented throughout this file) had grown the database large enough
  that unrelated documents started outranking the tests' own fixtures —
  invisible until it actually happened. Fixed by scoping every affected
  test to its own collection, which is also simply the correct way to
  write an isolated retrieval test regardless of database size.
- Tests: 199 passing total (added: `EventEmitter`/`TraceStore` behavior,
  Prometheus metrics rendering, full SSE event-sequence assertions against
  a real running pipeline, trace round-trip, 404 for an unknown trace,
  `/metrics` content type). `ruff` and `mypy --strict` both clean.

### Known gaps from Phase 11

- `TraceStore` is in-memory and per-process — restarting the app, or
  running more than one worker process, loses/fragments trace history. A
  real implementation would persist to the `events` table (present in the
  schema since Phase 1, still unused).
- SSE reconnection is not implemented: a client may send `Last-Event-ID`,
  but nothing replays events emitted before a reconnect from a persisted
  store — only `GET /queries/{trace_id}/trace` can be polled afterward,
  and only if the client already knows its trace ID.
- OpenTelemetry spans (as opposed to structured JSON logs + these events)
  are not implemented — spec §31 asks for "OpenTelemetry-compatible
  tracing," and what exists today is structured logging plus this event
  system, not actual OTel span export.
- No cost estimation — cache hit/miss and token-count instruments exist,
  but nothing converts them into an estimated dollar cost per query.

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
