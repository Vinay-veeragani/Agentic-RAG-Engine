"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { api, ApiError } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";

export default function TracesPage() {
  const [traceId, setTraceId] = useState("");

  const fetchTrace = useMutation({
    mutationFn: () => api.trace(traceId),
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Retrieval Traces</h1>
        <p className="text-muted-foreground">
          Look up the raw structured event timeline for any query by trace ID — the
          same events an SSE stream would have delivered live.
        </p>
      </div>

      <Card>
        <CardContent className="flex items-end gap-3 pt-6">
          <div className="min-w-64 flex-1 space-y-1.5">
            <Label htmlFor="trace">Trace ID</Label>
            <Input
              id="trace"
              value={traceId}
              onChange={(e) => setTraceId(e.target.value)}
              placeholder="e.g. the trace_id from an Ask response"
              onKeyDown={(e) => {
                if (e.key === "Enter" && traceId) fetchTrace.mutate();
              }}
            />
          </div>
          <Button onClick={() => fetchTrace.mutate()} disabled={!traceId || fetchTrace.isPending}>
            {fetchTrace.isPending ? "Loading…" : "Load trace"}
          </Button>
        </CardContent>
      </Card>

      {fetchTrace.isError && (
        <Alert variant="destructive">
          <AlertDescription>
            {fetchTrace.error instanceof ApiError ? fetchTrace.error.message : "Trace not found."}
          </AlertDescription>
        </Alert>
      )}

      {fetchTrace.data && (
        <div className="space-y-2">
          {fetchTrace.data.events.map((event) => (
            <Card key={event.event_id}>
              <CardContent className="space-y-2 py-3">
                <div className="flex items-center justify-between">
                  <Badge variant="outline" className="font-mono">
                    {event.event_type}
                  </Badge>
                  <span className="text-xs text-muted-foreground">
                    {new Date(event.timestamp).toLocaleTimeString()}
                  </span>
                </div>
                <pre className="max-h-64 overflow-auto rounded-md bg-muted p-3 text-xs">
                  {JSON.stringify(event.payload, null, 2)}
                </pre>
              </CardContent>
            </Card>
          ))}
          {fetchTrace.data.events.length === 0 && (
            <p className="text-sm text-muted-foreground">No events recorded for this trace.</p>
          )}
        </div>
      )}
    </div>
  );
}
