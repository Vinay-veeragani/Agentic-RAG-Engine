"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { api, ApiError } from "@/lib/api";
import { CollectionSelect } from "@/components/collection-select";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [collectionId, setCollectionId] = useState<string | null>(null);

  const search = useMutation({
    mutationFn: () => {
      if (!collectionId) throw new Error("Select a collection first.");
      return api.search({ query, collection_id: collectionId, top_k: topK });
    },
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Search</h1>
        <p className="text-muted-foreground">
          Direct hybrid dense + sparse retrieval, with full per-result score breakdowns — no
          agentic loop, no synthesis.
        </p>
      </div>

      <Card>
        <CardContent className="space-y-3 pt-6">
          <div className="flex flex-wrap items-end gap-3">
            <div className="min-w-64 flex-1 space-y-1.5">
              <Label htmlFor="query">Query</Label>
              <Input
                id="query"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && query && collectionId) search.mutate();
                }}
              />
            </div>
            <div className="space-y-1.5">
              <Label>Collection</Label>
              <CollectionSelect
                value={collectionId}
                onChange={setCollectionId}
                placeholder="Select a collection"
              />
            </div>
            <div className="w-24 space-y-1.5">
              <Label htmlFor="topk">Top K</Label>
              <Input
                id="topk"
                type="number"
                min={1}
                max={50}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
              />
            </div>
            <Button
              onClick={() => search.mutate()}
              disabled={!query || !collectionId || search.isPending}
            >
              {search.isPending ? "Searching…" : "Search"}
            </Button>
          </div>
          {search.isError && (
            <Alert variant="destructive">
              <AlertDescription>
                {search.error instanceof ApiError ? search.error.message : "Search failed."}
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {search.data && (
        <div className="space-y-3">
          {search.data.results.map((r) => (
            <Card key={r.chunk_id}>
              <CardContent className="space-y-2 pt-6">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">
                    {r.document_title ?? r.document_filename}
                  </span>
                  <Badge variant="secondary">score {r.score.toFixed(3)}</Badge>
                </div>
                <p className="text-xs text-muted-foreground">
                  {[r.section, r.heading, r.page !== null ? `page ${r.page}` : null]
                    .filter(Boolean)
                    .join(" · ") || "—"}
                </p>
                <p className="text-sm leading-relaxed">{r.content}</p>
              </CardContent>
            </Card>
          ))}
          {search.data.results.length === 0 && (
            <p className="text-sm text-muted-foreground">No results.</p>
          )}
        </div>
      )}
    </div>
  );
}
