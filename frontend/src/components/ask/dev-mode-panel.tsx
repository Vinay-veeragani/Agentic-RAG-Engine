"use client";

import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { api } from "@/lib/api";
import type { QueryResponse } from "@/lib/types";

/** Developer Mode (spec §37): query ID, trace ID, configured providers, top
 * K, retrieval strategy, iteration count, latency, and the raw JSON
 * response — never hidden chain-of-thought, since none is produced. */
export function DevModePanel({ response }: { response: QueryResponse }) {
  const { data: settings } = useQuery({ queryKey: ["settings"], queryFn: api.settings });

  const totalRetrievalLatency = response.iterations.reduce(
    (sum, it) => sum + it.retrieval_latency_seconds + it.rerank_latency_seconds,
    0,
  );

  return (
    <Card className="border-dashed">
      <CardHeader>
        <CardTitle className="text-sm font-mono">Developer Mode</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-xs font-mono">
        <div className="grid grid-cols-2 gap-x-4 gap-y-1 sm:grid-cols-3">
          <DevField label="trace_id" value={response.trace_id} />
          <DevField label="termination_reason" value={response.termination_reason} />
          <DevField label="strategy" value={response.plan.strategy} />
          <DevField label="top_k" value={String(response.plan.top_k)} />
          <DevField label="max_iterations" value={String(response.plan.max_iterations)} />
          <DevField label="iterations_used" value={String(response.iterations.length)} />
          <DevField
            label="retrieval+rerank latency"
            value={`${(totalRetrievalLatency * 1000).toFixed(0)} ms`}
          />
          <DevField label="llm_provider" value={settings?.llm_provider ?? "…"} />
          <DevField label="embedding_provider" value={settings?.embedding_provider ?? "…"} />
          <DevField label="reranker_provider" value={settings?.reranker_provider ?? "…"} />
        </div>
        <details>
          <summary className="cursor-pointer text-muted-foreground">Raw JSON response</summary>
          <pre className="mt-2 max-h-96 overflow-auto rounded-md bg-muted p-3">
            {JSON.stringify(response, null, 2)}
          </pre>
        </details>
      </CardContent>
    </Card>
  );
}

function DevField({ label, value }: { label: string; value: string }) {
  return (
    <div className="truncate">
      <span className="text-muted-foreground">{label}=</span>
      <span>{value}</span>
    </div>
  );
}
