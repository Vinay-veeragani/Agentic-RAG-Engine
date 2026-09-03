import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { AnswerStatus, TerminationReason } from "@/lib/types";

const STATUS_STYLES: Record<string, string> = {
  grounded: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400 border-emerald-500/30",
  sufficient_evidence:
    "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400 border-emerald-500/30",
  insufficient_evidence: "bg-amber-500/15 text-amber-700 dark:text-amber-400 border-amber-500/30",
  max_iterations_reached: "bg-amber-500/15 text-amber-700 dark:text-amber-400 border-amber-500/30",
  max_retrieval_calls_reached:
    "bg-amber-500/15 text-amber-700 dark:text-amber-400 border-amber-500/30",
  conflicting_evidence: "bg-red-500/15 text-red-700 dark:text-red-400 border-red-500/30",
  no_evidence_found: "bg-red-500/15 text-red-700 dark:text-red-400 border-red-500/30",
};

const STATUS_LABELS: Record<string, string> = {
  grounded: "Grounded",
  sufficient_evidence: "Sufficient evidence",
  insufficient_evidence: "Insufficient evidence",
  max_iterations_reached: "Max iterations reached",
  max_retrieval_calls_reached: "Retrieval budget exhausted",
  conflicting_evidence: "Conflicting evidence",
  no_evidence_found: "No evidence found",
};

export function StatusBadge({ status }: { status: AnswerStatus | TerminationReason }) {
  return (
    <Badge variant="outline" className={cn("font-medium", STATUS_STYLES[status])}>
      {STATUS_LABELS[status] ?? status}
    </Badge>
  );
}
