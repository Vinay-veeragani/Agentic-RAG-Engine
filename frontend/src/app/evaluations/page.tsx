"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { EvaluationSummary } from "@/lib/types";

const METRICS: { key: keyof EvaluationSummary; label: string }[] = [
  { key: "mean_recall", label: "Recall" },
  { key: "mean_precision", label: "Precision" },
  { key: "mean_mrr", label: "MRR" },
  { key: "mean_ndcg", label: "NDCG" },
  { key: "mean_hit_rate", label: "Hit rate" },
  { key: "mean_latency_seconds", label: "Latency (s)" },
  { key: "mean_estimated_tokens", label: "Tokens" },
  { key: "mean_answer_relevance", label: "Answer relevance" },
];

export default function EvaluationsPage() {
  const { data: report, isPending, isError } = useQuery({
    queryKey: ["evaluations"],
    queryFn: api.evaluations.latest,
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Evaluations</h1>
        <p className="text-muted-foreground">
          Real benchmark results: baseline single-shot retrieval vs. the full
          agentic pipeline, over a fixed evaluation corpus — no fabricated numbers.
        </p>
      </div>

      {isPending && <p className="text-sm text-muted-foreground">Loading…</p>}
      {isError && (
        <p className="text-sm text-muted-foreground">
          No evaluation report found yet. Run the benchmark script to generate one.
        </p>
      )}

      {report && (
        <>
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Summary: baseline vs. agentic</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Metric</TableHead>
                    <TableHead>Baseline</TableHead>
                    <TableHead>Agentic</TableHead>
                    <TableHead>Δ</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {METRICS.map(({ key, label }) => {
                    const b = report.baseline_summary[key];
                    const a = report.agentic_summary[key];
                    if (b === null || a === null) return null;
                    const delta = (a as number) - (b as number);
                    return (
                      <TableRow key={key}>
                        <TableCell className="font-medium">{label}</TableCell>
                        <TableCell>{(b as number).toFixed(3)}</TableCell>
                        <TableCell>{(a as number).toFixed(3)}</TableCell>
                        <TableCell
                          className={delta >= 0 ? "text-emerald-600" : "text-red-600"}
                        >
                          {delta >= 0 ? "+" : ""}
                          {delta.toFixed(3)}
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
              <p className="mt-3 text-xs text-muted-foreground">
                Generated {new Date(report.generated_at).toLocaleString()} · embedding:{" "}
                {report.embedding_provider} · llm: {report.llm_provider}
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">Per-case results</CardTitle>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Query</TableHead>
                    <TableHead>Category</TableHead>
                    <TableHead>Baseline recall</TableHead>
                    <TableHead>Agentic recall</TableHead>
                    <TableHead>Agentic status</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {report.cases.map((c, i) => (
                    <TableRow key={i}>
                      <TableCell className="max-w-xs truncate">{c.query}</TableCell>
                      <TableCell className="text-muted-foreground">{c.category}</TableCell>
                      <TableCell>{c.baseline.retrieval.recall.toFixed(2)}</TableCell>
                      <TableCell>{c.agentic.retrieval.recall.toFixed(2)}</TableCell>
                      <TableCell className="text-muted-foreground">
                        {c.agentic.status ?? "—"}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
