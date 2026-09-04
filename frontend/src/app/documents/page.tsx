"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "@/lib/api";
import { CollectionSelect } from "@/components/collection-select";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { DocumentIndexResponse } from "@/lib/types";

export default function DocumentsPage() {
  const queryClient = useQueryClient();
  const [collectionId, setCollectionId] = useState<string | null>(null);
  const [title, setTitle] = useState("");
  const [source, setSource] = useState("");
  const [documentDate, setDocumentDate] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [indexResults, setIndexResults] = useState<Record<string, DocumentIndexResponse>>({});

  const { data: documents, isPending } = useQuery({
    queryKey: ["documents", collectionId],
    queryFn: () => api.documents.list(collectionId ?? undefined),
  });

  const upload = useMutation({
    mutationFn: async () => {
      if (!file || !collectionId) throw new Error("Select a collection and a file first.");
      return api.documents.upload(collectionId, file, {
        title: title || undefined,
        source: source || undefined,
        documentDate: documentDate || undefined,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      setTitle("");
      setSource("");
      setDocumentDate("");
      setFile(null);
    },
  });

  const ingest = useMutation({
    mutationFn: (documentId: string) => api.documents.ingest(documentId),
    onSuccess: (result, documentId) => {
      setIndexResults((prev) => ({ ...prev, [documentId]: result }));
      queryClient.invalidateQueries({ queryKey: ["documents"] });
    },
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Documents</h1>
          <p className="text-muted-foreground">
            Upload source documents, then trigger ingestion (parse → chunk → embed → index).
          </p>
        </div>
        <CollectionSelect value={collectionId} onChange={setCollectionId} />
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Upload document</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid gap-3 sm:grid-cols-4">
            <div className="space-y-1.5">
              <Label htmlFor="title">Title (optional)</Label>
              <Input id="title" value={title} onChange={(e) => setTitle(e.target.value)} />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="source">Source (optional)</Label>
              <Input
                id="source"
                placeholder="annual report"
                value={source}
                onChange={(e) => setSource(e.target.value)}
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="document-date">Document date (optional)</Label>
              <Input
                id="document-date"
                type="date"
                value={documentDate}
                onChange={(e) => setDocumentDate(e.target.value)}
              />
              <p className="text-xs text-muted-foreground">
                The document&apos;s real date, not today&apos;s upload date.
              </p>
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="file">File</Label>
              <Input
                id="file"
                type="file"
                accept=".pdf,.docx,.txt,.md,.html,.csv,.json"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
            </div>
          </div>
          {!collectionId && (
            <p className="text-sm text-muted-foreground">Select a collection above to enable upload.</p>
          )}
          {upload.isError && (
            <Alert variant="destructive">
              <AlertDescription>
                {upload.error instanceof ApiError ? upload.error.message : "Upload failed."}
              </AlertDescription>
            </Alert>
          )}
          <Button
            onClick={() => upload.mutate()}
            disabled={!file || !collectionId || upload.isPending}
          >
            {upload.isPending ? "Uploading…" : "Upload"}
          </Button>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="pt-6">
          {isPending ? (
            <p className="text-sm text-muted-foreground">Loading…</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Title</TableHead>
                  <TableHead>Source</TableHead>
                  <TableHead>Date</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Ingest</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {documents?.map((doc) => {
                  const result = indexResults[doc.id];
                  return (
                    <TableRow key={doc.id}>
                      <TableCell className="font-medium">{doc.title ?? doc.filename}</TableCell>
                      <TableCell className="text-muted-foreground">{doc.source ?? "—"}</TableCell>
                      <TableCell className="text-muted-foreground">
                        {doc.document_date ?? "—"}
                      </TableCell>
                      <TableCell>
                        <Badge variant="outline">{doc.document_type}</Badge>
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <Button
                            size="sm"
                            variant="secondary"
                            onClick={() => ingest.mutate(doc.id)}
                            disabled={ingest.isPending && ingest.variables === doc.id}
                          >
                            {ingest.isPending && ingest.variables === doc.id
                              ? "Ingesting…"
                              : "Ingest"}
                          </Button>
                          {result && (
                            <span className="text-xs text-muted-foreground">
                              {result.chunk_count} chunks indexed
                            </span>
                          )}
                        </div>
                      </TableCell>
                    </TableRow>
                  );
                })}
                {documents?.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={5} className="text-center text-muted-foreground">
                      No documents in this scope yet.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
