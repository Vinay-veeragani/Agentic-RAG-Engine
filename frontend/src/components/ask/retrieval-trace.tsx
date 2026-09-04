import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import type {
  IterationTraceResponse,
  QueryAnalysis,
  RetrievalPlan,
} from "@/lib/types";

/** The signature "expand the pipeline" view: Query classification
 * → Retrieval plan → Search → Hybrid fusion → Reranking → Evidence
 * evaluation → Retrieval refinement → Final evidence → Answer synthesis →
 * Citation validation. Renders only the structured decisions the backend
 * actually returns — never a hidden chain-of-thought, because none exists. */
export function RetrievalTrace({
  analysis,
  plan,
  iterations,
  citationCompleteness,
  citationPrecision,
}: {
  analysis: QueryAnalysis;
  plan: RetrievalPlan;
  iterations: IterationTraceResponse[];
  citationCompleteness: number | null;
  citationPrecision: number | null;
}) {
  return (
    <Accordion multiple className="w-full" defaultValue={["plan"]}>
      <AccordionItem value="classification">
        <AccordionTrigger>Query classification</AccordionTrigger>
        <AccordionContent className="space-y-2 text-sm">
          <Field label="Type" value={analysis.query_type} />
          <Field label="Ambiguous" value={analysis.is_ambiguous ? "yes" : "no"} />
          <Field label="Answerable" value={analysis.is_answerable ? "yes" : "no"} />
          <Field label="Reasoning" value={analysis.reasoning} />
        </AccordionContent>
      </AccordionItem>

      <AccordionItem value="plan">
        <AccordionTrigger>Retrieval plan</AccordionTrigger>
        <AccordionContent className="space-y-2 text-sm">
          <Field label="Strategy" value={plan.strategy} />
          <Field label="Expand query" value={plan.expand_query ? "yes" : "no"} />
          <Field label="Decompose" value={plan.decompose ? "yes" : "no"} />
          <Field label="Max iterations" value={String(plan.max_iterations)} />
          <Field label="Top K" value={String(plan.top_k)} />
        </AccordionContent>
      </AccordionItem>

      {iterations.map((iteration) => (
        <AccordionItem key={iteration.iteration} value={`iteration-${iteration.iteration}`}>
          <AccordionTrigger>
            Iteration {iteration.iteration}: search → fusion → reranking → evidence
          </AccordionTrigger>
          <AccordionContent className="space-y-4 text-sm">
            <section className="space-y-1">
              <h4 className="font-medium">Search</h4>
              <Field label="Queries used" value={iteration.queries_used.join(" · ")} />
              <Field label="Strategy" value={iteration.retrieval_strategy} />
              <Field
                label="Candidates retrieved"
                value={String(iteration.candidates_retrieved)}
              />
              <Field
                label="Retrieval latency"
                value={`${(iteration.retrieval_latency_seconds * 1000).toFixed(0)} ms`}
              />
            </section>

            <section className="space-y-1">
              <h4 className="font-medium">Reranking</h4>
              <Field
                label="Rerank latency"
                value={`${(iteration.rerank_latency_seconds * 1000).toFixed(0)} ms`}
              />
            </section>

            <section className="space-y-1">
              <h4 className="font-medium">Evidence evaluation</h4>
              <Field label="Sufficient" value={iteration.sufficient ? "yes" : "no"} />
              <Field label="Reason" value={iteration.reason} />
              {iteration.missing_information.length > 0 && (
                <Field
                  label="Missing information"
                  value={iteration.missing_information.join("; ")}
                />
              )}
              {iteration.years_referenced.length > 0 && (
                <Field
                  label="Years referenced"
                  value={iteration.years_referenced.join(", ")}
                />
              )}
              {iteration.contradictions.length > 0 && (
                <div className="space-y-1">
                  <span className="text-muted-foreground">Contradictions:</span>
                  {iteration.contradictions.map((c, i) => (
                    <div key={i} className="rounded-md border border-red-500/30 bg-red-500/5 p-2">
                      <p>
                        <strong>{c.document_a}</strong>: {c.claim_a} vs.{" "}
                        <strong>{c.document_b}</strong>: {c.claim_b}
                      </p>
                      <p className="mt-1 text-xs text-muted-foreground">
                        {c.resolution ?? "Unresolved — no configured authority order distinguishes these sources."}
                      </p>
                    </div>
                  ))}
                </div>
              )}
              {!iteration.sufficient && iteration.iteration < iterations.length && (
                <Badge variant="outline" className="mt-1">
                  retrieval refined for next iteration
                </Badge>
              )}
            </section>
          </AccordionContent>
        </AccordionItem>
      ))}

      <AccordionItem value="synthesis">
        <AccordionTrigger>Answer synthesis &amp; citation validation</AccordionTrigger>
        <AccordionContent className="space-y-2 text-sm">
          <Field
            label="Citation completeness"
            value={citationCompleteness !== null ? citationCompleteness.toFixed(2) : "n/a"}
          />
          <Field
            label="Citation precision"
            value={citationPrecision !== null ? citationPrecision.toFixed(2) : "n/a"}
          />
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex gap-2">
      <span className="min-w-40 shrink-0 text-muted-foreground">{label}</span>
      <span>{value}</span>
    </div>
  );
}
