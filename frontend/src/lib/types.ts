// Mirrors the backend's Pydantic response schemas (src/agentic_rag/api/schemas/*).
// Kept hand-written rather than codegen'd — the backend is still evolving
// phase by phase, and a generator would need to run against a live server
// to stay in sync anyway. Field names match exactly so no mapping layer
// is needed between fetch() and the UI.

export interface Collection {
  id: string;
  name: string;
  description: string | null;
  source_authority_config: Record<string, unknown>;
  created_at: string;
}

export interface DocumentSummary {
  id: string;
  collection_id: string;
  title: string | null;
  source: string | null;
  document_date: string | null;
  filename: string;
  document_type: string;
  checksum: string;
  created_at: string;
}

export interface DocumentVersionSummary {
  id: string;
  version_number: number;
  status: string;
  created_at: string;
}

export interface DocumentDetail extends DocumentSummary {
  versions: DocumentVersionSummary[];
}

export interface ElementTypeCount {
  element_type: string;
  count: number;
}

export interface DocumentIngestResponse {
  document: DocumentSummary;
  version: DocumentVersionSummary;
  element_count: number;
  page_count: number | null;
  element_type_counts: ElementTypeCount[];
}

export interface DocumentIndexResponse {
  document_id: string;
  version_id: string;
  version_number: number;
  strategy: string;
  embedding_model: string;
  embedding_dimensions: number;
  chunk_count: number;
  parent_chunk_count: number;
}

export interface MetadataFilter {
  collection_id?: string | null;
  document_type?: string | null;
  document_ids?: string[] | null;
  section?: string | null;
  heading?: string | null;
  source?: string | null;
  year?: number | null;
}

export interface SearchResultItem {
  chunk_id: string;
  document_id: string;
  document_filename: string;
  document_title: string | null;
  page: number | null;
  section: string | null;
  heading: string | null;
  content: string;
  score: number;
}

export interface SearchResponse {
  query: string;
  results: SearchResultItem[];
}

export interface RetrievedCandidateResponse {
  chunk_id: string;
  document_id: string;
  document_filename: string;
  document_title: string | null;
  page: number | null;
  section: string | null;
  heading: string | null;
  content: string;
  dense_score: number | null;
  sparse_score: number | null;
  fusion_score: number | null;
  rerank_score: number | null;
  rank: number | null;
}

export interface RetrieveResponse {
  query: string;
  strategy: string;
  results: RetrievedCandidateResponse[];
}

export interface QueryAnalysis {
  query_type: string;
  is_ambiguous: boolean;
  is_answerable: boolean;
  reasoning: string;
}

export interface RetrievalPlan {
  strategy: string;
  expand_query: boolean;
  decompose: boolean;
  max_iterations: number;
  top_k: number;
  filters: MetadataFilter;
}

export interface QueryAnalyzeResponse {
  query: string;
  analysis: QueryAnalysis;
  plan: RetrievalPlan;
  expanded_queries: string[] | null;
  subqueries: string[] | null;
}

export interface ContradictionResponse {
  claim_a: string;
  claim_b: string;
  document_a: string;
  document_b: string;
  chunk_id_a: string;
  chunk_id_b: string;
  resolution: string | null;
}

export interface IterationTraceResponse {
  iteration: number;
  queries_used: string[];
  retrieval_strategy: string;
  candidates_retrieved: number;
  sufficient: boolean;
  reason: string;
  missing_information: string[];
  contradictions: ContradictionResponse[];
  years_referenced: number[];
  spans_multiple_periods: boolean;
  retrieval_latency_seconds: number;
  rerank_latency_seconds: number;
}

export type TerminationReason =
  | "sufficient_evidence"
  | "max_iterations_reached"
  | "max_retrieval_calls_reached"
  | "no_evidence_found"
  | "conflicting_evidence";

export interface AgenticRetrieveResponse {
  query: string;
  trace_id: string;
  analysis: QueryAnalysis;
  plan: RetrievalPlan;
  iterations: IterationTraceResponse[];
  termination_reason: TerminationReason;
  evidence: RetrievedCandidateResponse[];
}

export type AnswerStatus =
  | "grounded"
  | "insufficient_evidence"
  | "conflicting_evidence"
  | "no_evidence_found";

export interface CitationResponse {
  label: string;
  claim: string;
  chunk_id: string;
  document_id: string;
  document_filename: string;
  page: number | null;
  section: string | null;
  source: string | null;
  evidence_score: number | null;
}

export interface QueryResponse {
  query: string;
  trace_id: string;
  analysis: QueryAnalysis;
  plan: RetrievalPlan;
  status: AnswerStatus;
  answer: string | null;
  citations: CitationResponse[];
  citation_completeness: number | null;
  citation_precision: number | null;
  termination_reason: TerminationReason;
  iterations: IterationTraceResponse[];
}

export interface StreamEvent {
  event_id: string;
  query_id: string;
  timestamp: string;
  event_type: string;
  payload: Record<string, unknown>;
}

export interface TraceResponse {
  trace_id: string;
  events: StreamEvent[];
}

export interface SettingsResponse {
  app_env: string;
  llm_provider: string;
  embedding_provider: string;
  reranker_provider: string;
  max_retrieval_iterations: number;
  max_retrieval_calls: number;
  max_tokens_per_query: number;
  max_query_latency_seconds: number;
  max_upload_size_bytes: number;
  auth_enabled: boolean;
  rate_limit_enabled: boolean;
  rate_limit_requests_per_window: number;
  rate_limit_window_seconds: number;
  workers: number;
  cache_backend: string;
}

export interface HealthResponse {
  status: string;
  database: string;
  cache: string;
}

export interface EvaluationCaseResult {
  query: string;
  category: string;
  relevant_document_count: number;
  baseline: EvaluationPipelineResult;
  agentic: EvaluationPipelineResult;
}

export interface EvaluationPipelineResult {
  answer: string | null;
  latency_seconds: number;
  estimated_tokens: number;
  retrieval: {
    recall: number;
    precision: number;
    mrr: number;
    ndcg: number;
    hit_rate: number;
  };
  answer_relevance: number | null;
  status: string | null;
  iterations: number | null;
  citation_metrics: {
    claims_total: number;
    claims_supported: number;
    citations_total: number;
    citations_entailed: number;
  } | null;
}

export interface EvaluationSummary {
  mean_recall: number;
  mean_precision: number;
  mean_mrr: number;
  mean_ndcg: number;
  mean_hit_rate: number;
  mean_latency_seconds: number;
  mean_estimated_tokens: number;
  mean_answer_relevance: number | null;
  citation_metrics: {
    mean_precision: number;
    mean_completeness: number;
    cases_with_citations: number;
    total_cases: number;
  } | null;
}

export interface EvaluationReport {
  generated_at: string;
  embedding_provider: string;
  llm_provider: string;
  cases: EvaluationCaseResult[];
  baseline_summary: EvaluationSummary;
  agentic_summary: EvaluationSummary;
}

export interface ApiErrorBody {
  code: string;
  message: string;
  trace_id: string | null;
}
