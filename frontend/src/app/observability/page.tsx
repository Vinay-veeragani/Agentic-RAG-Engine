"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

export default function ObservabilityPage() {
  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 15_000,
  });

  const { data: metricsText } = useQuery({
    queryKey: ["metrics"],
    queryFn: async () => {
      const res = await fetch(api.metricsUrl);
      return res.text();
    },
    refetchInterval: 15_000,
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Observability</h1>
        <p className="text-muted-foreground">
          Live system health and Prometheus-compatible metrics exported by the backend.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Health</CardTitle>
        </CardHeader>
        <CardContent className="flex gap-3">
          <Badge variant={health?.status === "ok" ? "secondary" : "destructive"}>
            api: {health?.status ?? "…"}
          </Badge>
          <Badge variant={health?.database === "ok" ? "secondary" : "destructive"}>
            database: {health?.database ?? "…"}
          </Badge>
          <Badge variant={health?.cache === "ok" ? "secondary" : "destructive"}>
            cache: {health?.cache ?? "…"}
          </Badge>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Metrics (/metrics)</CardTitle>
        </CardHeader>
        <CardContent>
          <pre className="max-h-[32rem] overflow-auto rounded-md bg-muted p-3 text-xs">
            {metricsText ?? "Loading…"}
          </pre>
        </CardContent>
      </Card>
    </div>
  );
}
