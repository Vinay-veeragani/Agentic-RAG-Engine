# Agentic RAG Engine

### Production-Grade Autonomous Retrieval, Evidence Verification & Grounded Generation

> An advanced Agentic RAG system that autonomously plans retrieval, performs hybrid search, reranks evidence, detects insufficiency and contradictions, iteratively retrieves missing information, and generates citation-grounded answers with verifiable evidence.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-green.svg)](#)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-blue.svg)](#)
[![Redis](https://img.shields.io/badge/Redis-coordination-red.svg)](#)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](#)

---

## Why This Project?

Most RAG systems follow a relatively simple pipeline:

```text
User Question
      ↓
Embedding
      ↓
Vector Search
      ↓
Top-K Chunks
      ↓
LLM
      ↓
Answer
```

That approach works for simple questions, but it becomes unreliable when questions require:

* multiple pieces of evidence
* comparisons across documents
* temporal reasoning
* conflicting sources
* incomplete retrieval
* source-quality assessment
* precise citations
* iterative investigation

This project treats retrieval as an **autonomous evidence-gathering problem** rather than a single search operation.

```text
                    ┌──────────────────────┐
                    │      User Query      │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │   Query Analyzer     │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Retrieval Planner    │
                    └──────────┬───────────┘
                               ↓
              ┌──────────────────────────────────┐
              │        Hybrid Retrieval          │
              │                                  │
              │ Dense │ Sparse │ Metadata        │
              └────────────────┬─────────────────┘
                               ↓
                    ┌──────────────────────┐
                    │      Reranker        │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │    Evidence Judge    │
                    └──────────┬───────────┘
                               ↓
                     ┌─────────┴─────────┐
                     │                   │
                Insufficient          Sufficient
                     │                   │
                     ↓                   ↓
              Query Refinement      Synthesis
                     │                   │
                     └──→ Retrieve ←─────┘
                                         ↓
                              Citation Validation
                                         ↓
                              Grounded Answer
```

The key idea is simple:

> **The system does not generate an answer merely because it found some chunks. It first determines whether the available evidence is sufficient.**

---

# Core Capabilities

## 🧠 Agentic Retrieval

The system can dynamically decide:

* what retrieval strategy to use
* whether a question needs decomposition
* whether query expansion is useful
* whether more evidence is required
* whether another retrieval iteration should be performed
* when enough evidence has been collected

The retrieval loop is bounded by configurable limits for:

* iterations
* retrieval calls
* tokens
* latency

Cost per query is estimated and reported for observability, but is not itself
an enforced stopping condition today — the loop terminates on iteration/call/
token/latency budgets. This prevents uncontrolled autonomous loops.

---

## 🔎 Hybrid Retrieval

Instead of relying exclusively on vector similarity, the engine combines multiple retrieval signals:

```text
                  Query
                    │
          ┌─────────┼─────────┐
          ↓         ↓         ↓
       Dense      Sparse    Metadata
       Search     Search     Filters
          │         │         │
          └─────────┼─────────┘
                    ↓
             Result Fusion
                    ↓
                  RRF
                    ↓
               Reranking
```

Supported retrieval mechanisms include:

* dense vector retrieval (pgvector cosine similarity, HNSW index)
* PostgreSQL full-text search (`ts_rank_cd`, not literal BM25, same idea)
* metadata filtering
* hybrid retrieval
* Reciprocal Rank Fusion
* cross-encoder reranking

The architecture keeps retrieval deterministic and separates retrieval mechanics from LLM reasoning.

---

# 🔁 Agentic Retrieval Loop

A major component of the system is the bounded retrieval loop.

```text
Query
  ↓
Analyze
  ↓
Plan
  ↓
Retrieve
  ↓
Rerank
  ↓
Evaluate Evidence
  │
  ├── Sufficient ─────→ Generate
  │
  └── Insufficient
            ↓
       Refine Query
            ↓
       Retrieve Again
```

The system does not blindly retrieve more documents.

The evidence evaluation step determines whether another retrieval iteration is justified.

---

# 🧩 Query Understanding

Queries are classified before retrieval.

Examples include:

* simple factual
* multi-hop
* comparison
* temporal
* analytical
* ambiguous
* potentially unanswerable

The planner can select different retrieval strategies based on the query.

For example:

```text
"Who is the CEO of Company X?"

→ Direct retrieval
```

while:

```text
"How did Company X's operating margin change
between FY2023 and FY2025, and what caused the change?"

→ Decompose
→ Retrieve multiple evidence sets
→ Apply temporal filtering
→ Compare evidence
→ Verify
→ Synthesize
```

---

# 📚 Document Intelligence

The ingestion pipeline supports multiple document formats.

Current parsers include:

* PDF
* DOCX
* TXT
* Markdown
* HTML
* CSV
* JSON

Documents are normalized into a common internal representation while preserving useful structural information.

Where available, the system preserves:

* page numbers
* headings
* sections
* paragraphs
* source metadata
* document dates
* chunk relationships (parent/child)

---

# ✂️ Intelligent Chunking

The system supports multiple chunking strategies, selectable per collection:

* **structural (default)** — groups by heading, never splits a section; oversized sections get a full parent chunk plus recursively-split child chunks
* **recursive** — packs whole elements up to a token budget, carries the tail forward as overlap
* **semantic** — splits into sentences, embeds each one, merges consecutive sentences until similarity drops below a threshold
* **fixed** — plain fixed-size windows, the simple baseline

Defaults: 400 tokens/chunk, 50 token overlap, 20 token minimum. Chunking
configuration is retained per document version so indexing decisions remain
reproducible.

```text
Document
   │
   ├── Section
   │     ├── Paragraph
   │     └── Paragraph
   │
   └── Section
         ├── Paragraph
         └── Paragraph
```

This provides considerably richer retrieval context than treating every document as an unstructured text blob.

---

# ⚖️ Evidence Evaluation

Retrieval is followed by evidence assessment.

Evidence can be evaluated based on factors such as:

* relevance
* coverage
* directness
* source authority
* temporal correctness
* contradiction
* support for the requested claim

The system can explicitly distinguish between:

```text
Evidence is sufficient
Evidence is insufficient
Evidence is conflicting
Knowledge is unavailable
```

This allows the system to refuse unsupported answers rather than forcing the LLM to produce one.

---

# ⚔️ Contradiction Detection

Different documents may provide conflicting information.

The engine detects contradictions across retrieved evidence using deterministic
matching (same metric, different numbers, non-overlapping source documents,
same reporting period) and applies configurable source-authority policies —
contradiction detection itself is not an LLM guess.

Example:

```text
Source A
Revenue: $42M
Published: 2025

Source B
Revenue: $48M
Published: 2026
```

Instead of blindly selecting whichever chunk has the highest similarity score, the system can consider:

* source authority
* document metadata
* temporal context
* evidence relationships

If the conflict cannot be resolved reliably, the system exposes the conflict instead of hiding it.

---

# ⏳ Temporal Awareness

Documents often contain information that changes over time.

The system preserves temporal metadata such as:

* publication date
* reporting period
* fiscal year
* source date

This helps prevent retrieval systems from treating information from different periods as interchangeable facts.

---

# 📌 Citation Grounding

Citations are a first-class part of the architecture.

The answer generation process does not allow the LLM to freely invent arbitrary citation identifiers.

Instead:

```text
Retrieved Evidence
       ↓
Evidence IDs
       ↓
LLM generates claim + evidence references
       ↓
Deterministic citation resolution
       ↓
Citation validation
       ↓
Final answer
```

Citation references are resolved against the actual evidence available to the system.

The citation layer can validate:

* citation existence
* claim support
* citation precision
* citation completeness

Unsupported claims can be removed rather than presented as grounded facts.

---

# 🛡️ Prompt Injection Defense

Retrieved documents are treated as **untrusted data**, not instructions.

The system attempts to detect instruction-like content inside retrieved material before it reaches the generation context.

The security boundary is:

```text
System Instructions
        ↑
        │
Trusted Application Logic
        ↑
        │
Retrieved Documents
        │
        └── UNTRUSTED DATA
```

A document cannot override the system's instructions simply because it contains text such as:

```text
Ignore previous instructions...
Reveal system prompt...
Call this tool...
```

Security controls include:

* prompt-injection detection
* document isolation
* metadata filtering
* path traversal protection
* input validation
* bounded execution
* configurable budgets
* timeout handling

---

# 🗃️ Knowledge Collections

Knowledge is organized into isolated collections.

```text
Collection
   │
   ├── Documents
   │      │
   │      ├── Versions
   │      │
   │      └── Metadata
   │
   └── Chunks
          │
          └── Embeddings
```

Collection-level configuration can control:

* embedding model
* chunking configuration
* metadata filters
* source authority

Collection isolation is enforced at query/retrieval boundaries.

---

# 🏗️ Architecture

```text
┌───────────────────────────────────────────────────────┐
│                       Client                          │
│             REST API / Streaming / UI                │
└───────────────────────────┬───────────────────────────┘
                            ↓
┌───────────────────────────────────────────────────────┐
│                    RAG Orchestrator                   │
│                                                       │
│ Query Analysis → Planning → Retrieval → Verification │
└───────────────────────────┬───────────────────────────┘
                            ↓
┌───────────────────────────────────────────────────────┐
│                       Agents                          │
│                                                       │
│ Query │ Planner │ Retrieval │ Evidence │ Synthesis   │
│ Citation │ Verification                              │
└───────────────────────────┬───────────────────────────┘
                            ↓
┌───────────────────────────────────────────────────────┐
│                    Retrieval Layer                    │
│                                                       │
│ Dense │ Sparse │ Metadata │ Hybrid │ RRF │ Reranker  │
└───────────────────────────┬───────────────────────────┘
                            ↓
┌───────────────────────────────────────────────────────┐
│                    Knowledge Layer                    │
│                                                       │
│ Documents │ Chunks │ Embeddings │ Sources │ Versions │
└───────────────────────────┬───────────────────────────┘
                            ↓
┌───────────────────────────────────────────────────────┐
│                     Storage                           │
│                                                       │
│              PostgreSQL + pgvector                    │
│                    Redis                              │
└───────────────────────────────────────────────────────┘
```

---

# 🧱 Design Philosophy

A central architectural principle is:

> **LLMs reason. Deterministic software executes and enforces.**

### LLM responsibilities

The model handles:

* query understanding
* query planning
* decomposition
* query refinement
* evidence assessment
* synthesis
* claim extraction

### Deterministic software responsibilities

The application handles:

* database operations
* vector search
* keyword search
* rank fusion
* filtering
* citation resolution
* validation
* permissions
* budgets
* timeouts
* caching
* event handling
* persistence

This prevents the LLM from becoming an uncontrolled source of system behavior.

---

# 💾 Storage

The system uses:

### PostgreSQL

Used as the primary durable data store for:

* collections
* documents
* document versions
* chunks
* embeddings
* metadata
* query records
* retrieval information

### pgvector

Used for vector similarity search with an HNSW index.

### Redis

Used where coordination and ephemeral infrastructure are appropriate, including:

* caching
* rate limiting
* streaming/event infrastructure

If no Redis URL is configured, the system automatically falls back to an
in-process cache — useful for local development without external services.

---

# 🚀 API

The system exposes a FastAPI service.

Representative endpoints:

```text
POST   /collections
GET    /collections
GET    /collections/{id}

POST   /documents
GET    /documents
GET    /documents/{id}
POST   /documents/{id}/ingest

POST   /search
POST   /retrieve
POST   /query/analyze

POST   /query
POST   /query/stream

GET    /queries/{id}
GET    /queries/{id}/trace

GET    /evaluations/latest
GET    /evaluations/latest/summary

GET    /health
GET    /metrics
```

---

# 🌊 Streaming

Long-running retrieval operations can be streamed through SSE.

Example event lifecycle:

```text
query.started
      ↓
query.analyzed
      ↓
plan.created
      ↓
retrieval.started
      ↓
retrieval.completed
      ↓
reranking.started
      ↓
reranking.completed
      ↓
evidence.evaluated
      ↓
retrieval.refined
      ↓
generation.started
      ↓
citation.validation.started
      ↓
query.completed
```

This makes retrieval behavior observable rather than hiding everything behind a single API response.

---

# 🔬 Evaluation Framework

The project includes an evaluation harness designed to compare traditional RAG against Agentic RAG.

Evaluation categories include:

* factual questions
* multi-hop questions
* comparisons
* temporal questions
* analytical questions
* ambiguous questions
* unanswerable questions
* contradictory evidence

### Retrieval metrics

* Recall@K
* Precision@K
* MRR
* NDCG@K
* Hit Rate@K

### Generation metrics

* faithfulness
* answer relevance
* context relevance

### Citation metrics

* citation precision
* citation completeness

(`citation recall` would require a hand-labeled ground-truth "which chunks are
actually relevant" set that this project doesn't maintain — the code
documents this explicitly rather than faking the metric.)

### System metrics

* latency
* estimated token usage
* estimated cost
* retrieval iterations
* retrieval calls

---

# 📊 Baseline vs Agentic RAG

One of the project's goals is not simply to claim that Agentic RAG is better.

The repository contains separate execution paths for:

```text
Baseline RAG
    vs
Agentic RAG
```

This allows the system to measure whether additional reasoning actually improves retrieval and answer quality.

Benchmark results are generated from the evaluation suite rather than hard-coded claims. Run it yourself:

```bash
python benchmarks/run_evaluation.py --embedding local --llm mock
```

See `benchmarks/` for reproducible evaluation scenarios and committed results
(`benchmarks/results/latest.json`), and `docs/architecture.md` for a full
write-up of a real run's numbers and what they mean.

---

# 🔭 Observability

Every query can be traced through the retrieval pipeline.

Useful telemetry includes:

* query ID
* trace ID
* retrieval strategy
* retrieved documents
* retrieved chunks
* retrieval scores
* reranking scores
* evidence decisions
* iteration count
* latency
* estimated token usage and cost
* errors
* cache behavior

The UI exposes this as a retrieval/debugging trace rather than only showing the final answer.

---

# 🖥️ Knowledge Intelligence UI

The frontend is designed as a **Knowledge Intelligence Workspace**, not a generic chatbot.

The primary workflow is:

```text
Ask
 ↓
Understand
 ↓
Retrieve
 ↓
Verify
 ↓
Answer
```

The UI exposes:

* collections
* documents
* search
* questions
* answers
* citations
* source documents
* evidence signals
* retrieval traces
* evaluation results
* observability

### Developer Mode

For debugging and engineering investigation, the interface can expose:

* query ID
* trace ID
* retrieval strategy
* top-K configuration
* embedding model
* reranker
* iteration count
* token usage
* latency
* structured events
* raw execution metadata

Hidden chain-of-thought is never exposed.

---

# 🧪 Testing

The repository contains extensive automated tests covering:

### Unit tests

* chunking
* embeddings
* retrieval
* RRF
* filtering
* ranking
* citation resolution
* citation validation
* security
* query planning

### Integration tests

* PostgreSQL
* pgvector
* ingestion
* retrieval
* evidence evaluation
* contradiction detection
* collection isolation

Tests run against a dedicated, disposable `agentic_rag_test` database (never
your real dev database) with all providers forced to mock — see
`docs/architecture.md`'s "Test isolation" section.

### End-to-end scenarios

Representative flows cover:

```text
Document
   ↓
Ingestion
   ↓
Chunking
   ↓
Embedding
   ↓
Indexing
   ↓
Query
   ↓
Retrieval
   ↓
Evidence
   ↓
Generation
   ↓
Citation Validation
   ↓
Grounded Answer
```

Adversarial scenarios include:

* prompt injection
* conflicting sources
* insufficient evidence
* malformed documents
* temporal conflicts
* retrieval failures
* budget exhaustion

---

# ⚡ Reliability & Safety

The system is intentionally bounded.

Configurable limits exist for:

* maximum retrieval iterations
* retrieval calls
* tokens
* latency
* document size
* request execution timeouts

Explicit failure states include:

```text
NO_KNOWLEDGE
INSUFFICIENT_EVIDENCE
CONFLICTING_EVIDENCE
RETRIEVAL_ERROR
MODEL_ERROR
TIMEOUT
BUDGET_EXCEEDED
INVALID_DOCUMENT
UNSUPPORTED_FILE_TYPE
PROMPT_INJECTION_DETECTED
```

The objective is graceful failure rather than hallucinated success.

---

# 📁 Project Structure

```text
Agentic-RAG-Engine/
│
├── src/
│   └── agentic_rag/
│       ├── agents/
│       ├── api/
│       ├── chunking/
│       ├── citations/
│       ├── core/
│       ├── embeddings/
│       ├── evaluation/
│       ├── generation/
│       ├── ingestion/
│       ├── knowledge/
│       ├── observability/
│       ├── retrieval/
│       ├── security/
│       └── storage/
│
├── tests/
├── benchmarks/
├── docs/
├── examples/
├── migrations/
├── docker/
│   ├── docker-compose.yml
│   └── Dockerfile
│
├── frontend/
├── alembic.ini
├── pyproject.toml
├── LICENSE
└── README.md
```

---

# 🚀 Quick Start

## 1. Clone

```bash
git clone https://github.com/Vinay-veeragani/Agentic-RAG-Engine.git
cd Agentic-RAG-Engine
```

## 2. Configure environment

```bash
cp .env.example .env
```

Configure the required database and model/provider settings.

## 3. Start infrastructure

A Docker Compose stack is provided for environments that can run Docker
(also what CI uses):

```bash
docker compose -f docker/docker-compose.yml up -d
```

If you're on a native local install instead (e.g. a native Windows
PostgreSQL 18 + pgvector install and a managed Redis URL — see
`docs/architecture.md`'s "Local development setup"), skip this step and just
point `DATABASE_URL` / `REDIS_URL` in `.env` at your existing instances.

## 4. Install dependencies

```bash
pip install -e ".[dev]"
```

## 5. Run migrations

```bash
alembic upgrade head
```

One-time setup for the test suite (a separate, disposable database so
running `pytest` never touches your real data — see
`docs/architecture.md`'s "Test isolation" section):

```sql
CREATE DATABASE agentic_rag_test OWNER agentic_rag_app;
\c agentic_rag_test
CREATE EXTENSION vector;
```

```bash
DATABASE_URL=postgresql+asyncpg://agentic_rag_app:<password>@localhost:5432/agentic_rag_test alembic upgrade head
```

## 6. Start the API

```bash
uvicorn agentic_rag.api.main:app --reload
```

## 7. Start the frontend (optional)

```bash
cd frontend
npm install
npm run dev
```

The API will then be available at `http://localhost:8000` and the UI at `http://localhost:3000`.

---

# 🧪 Run Tests

```bash
pytest
```

Add `-m slow` to also run the real cross-encoder reranker test (downloads/loads a model on first use).

Type checking:

```bash
mypy src
```

Linting:

```bash
ruff check src tests
```

---

# 🔍 Example Query Flow

Suppose the knowledge base contains annual reports from several years.

Question:

```text
How did Company X's operating margin change
between FY2023 and FY2025, and what factors
contributed to the change?
```

The system can execute:

```text
1. Analyze question
        ↓
2. Detect temporal + analytical query
        ↓
3. Decompose into sub-questions
        ↓
4. Retrieve FY2023 evidence
        ↓
5. Retrieve FY2025 evidence
        ↓
6. Rerank evidence
        ↓
7. Evaluate coverage
        ↓
8. Detect conflicting evidence if present
        ↓
9. Retrieve additional evidence if necessary
        ↓
10. Synthesize supported claims
        ↓
11. Validate citations
        ↓
12. Return grounded answer
```

The important part is that the system can **recognize when the first retrieval pass is not enough**.

---

# 🧠 Why Not Just Use More Context?

A common approach to improving RAG is:

```text
Retrieve more chunks
        ↓
Put everything into context
        ↓
Ask the LLM
```

This project takes a different approach.

More context does not automatically mean better evidence.

Instead, the system attempts to answer:

> **Which evidence is actually necessary to support the answer?**

This makes retrieval, verification, and citation quality explicit parts of the system.

---

# 🏛️ Key Engineering Decisions

### PostgreSQL + pgvector

Keeps relational metadata and vector search close together while reducing infrastructure complexity for the core system.

### Hybrid Retrieval

Dense retrieval captures semantic similarity while sparse retrieval provides lexical matching for exact terms, identifiers, and domain-specific language.

### RRF

Rank fusion combines retrieval strategies without requiring the scores from different retrievers to be directly comparable.

### Cross-Encoder Reranking

Retrieval produces a candidate set; reranking performs more expensive relevance evaluation only on the smaller candidate pool.

### Bounded Agentic Loop

Autonomous retrieval without hard limits can become expensive and unpredictable. The loop is therefore explicitly bounded.

### Deterministic Citation Resolution

Citation IDs are resolved against actual evidence rather than trusting the model to invent source references.

### Evidence Before Generation

The generator receives validated evidence rather than raw retrieval results whenever possible.

---

# 🔐 Security Model

The system follows a defense-in-depth approach.

```text
Input Validation
      ↓
Document Validation
      ↓
Collection Isolation
      ↓
Metadata Filtering
      ↓
Prompt Injection Detection
      ↓
Evidence Validation
      ↓
Citation Validation
      ↓
Grounded Generation
```

Retrieved content is always considered untrusted.

---

# 📈 Performance Model

The architecture separates inexpensive deterministic operations from expensive model operations.

```text
Cheap / Deterministic
──────────────────────
Metadata filtering
Sparse search
Vector search
RRF
Validation
Caching
Citation resolution

Expensive / Model-based
───────────────────────
Query planning
Decomposition
Evidence reasoning
Reranking
Generation
```

This allows expensive reasoning to be introduced only where it provides value.

---

# 🧭 Roadmap

Potential future extensions include:

* additional vector-store backends
* stronger semantic retrieval
* learned retrieval policies
* richer temporal reasoning
* advanced multi-hop retrieval
* larger evaluation datasets
* distributed execution
* additional local rerankers
* multimodal document retrieval
* web and external-source research
* knowledge graphs

---

# ⚠️ What This Project Is Not

This repository is intentionally focused.

It is **not**:

* a generic ChatGPT clone
* a simple "chat with PDF" application
* a hosted SaaS product
* a replacement for every commercial RAG platform
* an unrestricted autonomous agent
* a benchmark claiming universal superiority of Agentic RAG

It is a **production-shaped reference implementation for autonomous retrieval, evidence verification, and grounded generation**.

---

# 🎯 Design Goals

The project prioritizes:

```text
Correctness
    >
Uncontrolled autonomy

Evidence
    >
Model confidence

Deterministic execution
    >
LLM-controlled infrastructure

Bounded reasoning
    >
Infinite agent loops

Verifiable citations
    >
Generated citations

Measured evaluation
    >
Marketing claims
```

---

# 📚 Documentation

Detailed documentation is available under:

```text
docs/
```

See `docs/architecture.md` for the full design rationale, engineering
tradeoffs, and write-ups of real bugs found and fixed by running this system
against a real LLM and a real document.

---

# 🤝 Contributing

Contributions, discussions, architectural suggestions, and improvements are welcome.

Before submitting a major change, please consider:

1. Does it preserve deterministic execution boundaries?
2. Does it introduce an unnecessary LLM dependency?
3. Can the behavior be evaluated?
4. Can the behavior be tested?
5. Does it preserve evidence provenance?
6. Does it introduce an unbounded execution path?

---

# 📄 License

MIT License — see `LICENSE`.

---

## Final Thought

Retrieval-Augmented Generation is often presented as:

> **Search → Context → LLM**

Real-world knowledge systems are more complicated.

They need to determine:

* what to search for
* how to search
* whether the retrieved evidence is sufficient
* whether sources disagree
* whether information is temporally valid
* whether another retrieval iteration is necessary
* whether the final claims are actually supported
* and whether every citation can be traced back to real evidence

This project explores that layer between **retrieval and trustworthy generation**.

> **Don't just retrieve context. Build evidence. Verify it. Then generate.**
