"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { api, ApiError } from "@/lib/api";
import { useAppStore } from "@/lib/store";
import { CollectionSelect } from "@/components/collection-select";
import { StatusBadge } from "@/components/ask/status-badge";
import { CitationList } from "@/components/ask/citation-list";
import { RetrievalTrace } from "@/components/ask/retrieval-trace";
import { DevModePanel } from "@/components/ask/dev-mode-panel";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";

export default function AskPage() {
  const [query, setQuery] = useState("");
  const [collectionId, setCollectionId] = useState<string | null>(null);
  const developerMode = useAppStore((s) => s.developerMode);

  const ask = useMutation({
    mutationFn: () => api.query.ask(query, collectionId ?? undefined),
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Ask</h1>
        <p className="text-muted-foreground">
          The full agentic pipeline: query analysis, bounded iterative retrieval, evidence
          verification, grounded synthesis, and validated citations — with the entire trace
          visible below the answer, never a hidden chain-of-thought.
        </p>
      </div>

      <Card>
        <CardContent className="space-y-3 pt-6">
          <Textarea
            placeholder="Ask a question grounded in your ingested documents…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            rows={3}
          />
          <div className="flex items-center justify-between gap-3">
            <CollectionSelect value={collectionId} onChange={setCollectionId} />
            <Button onClick={() => ask.mutate()} disabled={!query || ask.isPending}>
              {ask.isPending ? "Thinking…" : "Ask"}
            </Button>
          </div>
          {ask.isError && (
            <Alert variant="destructive">
              <AlertDescription>
                {ask.error instanceof ApiError ? ask.error.message : "Query failed."}
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {ask.data && (
        <div className="space-y-4">
          <Card>
            <CardContent className="space-y-3 pt-6">
              <div className="flex items-center justify-between">
                <StatusBadge status={ask.data.status} />
                <span className="text-xs text-muted-foreground">
                  trace {ask.data.trace_id}
                </span>
              </div>
              <p className="whitespace-pre-wrap text-sm leading-relaxed">
                {ask.data.answer ?? "No answer could be grounded in the available evidence."}
              </p>
              <CitationList citations={ask.data.citations} />
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-6">
              <RetrievalTrace
                analysis={ask.data.analysis}
                plan={ask.data.plan}
                iterations={ask.data.iterations}
                citationCompleteness={ask.data.citation_completeness}
                citationPrecision={ask.data.citation_precision}
              />
            </CardContent>
          </Card>

          {developerMode && <DevModePanel response={ask.data} />}
        </div>
      )}
    </div>
  );
}
