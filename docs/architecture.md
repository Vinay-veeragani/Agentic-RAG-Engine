# Architecture

Status: Phase 2 (Document ingestion) complete, on top of Phase 1 (Foundation +
database + configuration). This document will grow with each phase; it only
describes what is actually implemented.

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
