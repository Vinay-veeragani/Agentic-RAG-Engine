import type {
  AgenticRetrieveResponse,
  ApiErrorBody,
  Collection,
  DocumentDetail,
  DocumentIndexResponse,
  DocumentIngestResponse,
  DocumentSummary,
  EvaluationReport,
  HealthResponse,
  MetadataFilter,
  QueryAnalyzeResponse,
  QueryResponse,
  RetrieveResponse,
  SearchResponse,
  SettingsResponse,
  StreamEvent,
  TraceResponse,
} from "@/lib/types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public body: ApiErrorBody | null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    let body: ApiErrorBody | null = null;
    try {
      body = await response.json();
    } catch {
      // response wasn't JSON — leave body null, message below still helps
    }
    throw new ApiError(
      body?.message ?? `Request to ${path} failed with ${response.status}`,
      response.status,
      body,
    );
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

function toJsonBody(value: unknown): string {
  return JSON.stringify(value);
}

export const api = {
  health: (): Promise<HealthResponse> => request("/health"),

  collections: {
    list: (): Promise<Collection[]> => request("/collections"),
    get: (id: string): Promise<Collection> => request(`/collections/${id}`),
    create: (body: {
      name: string;
      description?: string;
      source_authority_order?: string[];
    }): Promise<Collection> =>
      request("/collections", { method: "POST", body: toJsonBody(body) }),
  },

  documents: {
    list: (collectionId?: string): Promise<DocumentSummary[]> =>
      request(
        collectionId ? `/documents?collection_id=${collectionId}` : "/documents",
      ),
    get: (id: string): Promise<DocumentDetail> => request(`/documents/${id}`),
    upload: async (
      collectionId: string,
      file: File,
      opts?: { title?: string; source?: string },
    ): Promise<DocumentIngestResponse> => {
      const form = new FormData();
      form.append("collection_id", collectionId);
      form.append("file", file);
      if (opts?.title) form.append("title", opts.title);
      if (opts?.source) form.append("source", opts.source);
      const response = await fetch(`${API_URL}/documents`, { method: "POST", body: form });
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new ApiError(body?.message ?? "Upload failed", response.status, body);
      }
      return response.json();
    },
    ingest: (
      id: string,
      overrides?: Record<string, unknown>,
    ): Promise<DocumentIndexResponse> =>
      request(`/documents/${id}/ingest`, {
        method: "POST",
        body: toJsonBody(overrides ?? {}),
      }),
  },

  search: (body: {
    query: string;
    top_k?: number;
    filters?: MetadataFilter;
  }): Promise<SearchResponse> =>
    request("/search", { method: "POST", body: toJsonBody(body) }),

  retrieve: (body: {
    query: string;
    strategy?: string;
    top_k?: number;
    candidate_pool_size?: number;
    filters?: MetadataFilter;
    rerank?: boolean;
    rerank_top_k?: number;
  }): Promise<RetrieveResponse> =>
    request("/retrieve", { method: "POST", body: toJsonBody(body) }),

  query: {
    analyze: (query: string): Promise<QueryAnalyzeResponse> =>
      request("/query/analyze", { method: "POST", body: toJsonBody({ query }) }),
    agenticRetrieve: (
      query: string,
      collectionId?: string,
    ): Promise<AgenticRetrieveResponse> =>
      request("/query/retrieve", {
        method: "POST",
        body: toJsonBody({ query, collection_id: collectionId }),
      }),
    ask: (query: string, collectionId?: string): Promise<QueryResponse> =>
      request("/query", {
        method: "POST",
        body: toJsonBody({ query, collection_id: collectionId }),
      }),
    /** Streams the same pipeline as `ask`, invoking `onEvent` as each
     * structured event (spec §30) arrives. Resolves once the stream ends. */
    stream: async (
      query: string,
      collectionId: string | undefined,
      onEvent: (event: StreamEvent) => void,
      signal?: AbortSignal,
    ): Promise<void> => {
      const response = await fetch(`${API_URL}/query/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: toJsonBody({ query, collection_id: collectionId }),
        signal,
      });
      if (!response.ok || !response.body) {
        throw new ApiError(`Stream request failed with ${response.status}`, response.status, null);
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() ?? "";
        for (const block of blocks) {
          const dataLine = block.split("\n").find((line) => line.startsWith("data: "));
          if (dataLine) onEvent(JSON.parse(dataLine.slice("data: ".length)));
        }
      }
    },
  },

  trace: (traceId: string): Promise<TraceResponse> => request(`/queries/${traceId}/trace`),

  settings: (): Promise<SettingsResponse> => request("/settings"),

  evaluations: {
    latest: (): Promise<EvaluationReport> => request("/evaluations/latest"),
  },

  metricsUrl: `${API_URL}/metrics`,
};
