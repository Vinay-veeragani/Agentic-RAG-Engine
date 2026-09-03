import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { CitationResponse } from "@/lib/types";

export function CitationList({ citations }: { citations: CitationResponse[] }) {
  if (citations.length === 0) return null;

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-medium text-muted-foreground">Sources</h3>
      <div className="space-y-2">
        {citations.map((citation, i) => (
          <Card key={`${citation.chunk_id}-${i}`} className="border-muted-foreground/20">
            <CardContent className="flex items-start justify-between gap-4 py-3">
              <div>
                <p className="text-sm font-medium">{citation.label}</p>
                <p className="mt-1 text-sm text-muted-foreground">&ldquo;{citation.claim}&rdquo;</p>
              </div>
              {citation.evidence_score !== null && (
                <Badge variant="secondary" className="shrink-0">
                  score {citation.evidence_score.toFixed(2)}
                </Badge>
              )}
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
