"use client";

import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Alert, AlertDescription } from "@/components/ui/alert";

export default function CollectionsPage() {
  const queryClient = useQueryClient();
  const { data: collections, isPending } = useQuery({
    queryKey: ["collections"],
    queryFn: api.collections.list,
  });

  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [authorityOrder, setAuthorityOrder] = useState("");

  const createCollection = useMutation({
    mutationFn: () =>
      api.collections.create({
        name,
        description: description || undefined,
        source_authority_order: authorityOrder
          ? authorityOrder.split(",").map((s) => s.trim().toLowerCase()).filter(Boolean)
          : undefined,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["collections"] });
      setName("");
      setDescription("");
      setAuthorityOrder("");
    },
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Collections</h1>
        <p className="text-muted-foreground">
          A collection groups documents with shared retrieval and source-authority
          configuration (spec §27/§20).
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">New collection</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid gap-3 sm:grid-cols-3">
            <div className="space-y-1.5">
              <Label htmlFor="name">Name</Label>
              <Input id="name" value={name} onChange={(e) => setName(e.target.value)} />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="description">Description</Label>
              <Input
                id="description"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
              />
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="authority">
                Source authority order (comma-separated, most authoritative first)
              </Label>
              <Input
                id="authority"
                placeholder="annual report, press release"
                value={authorityOrder}
                onChange={(e) => setAuthorityOrder(e.target.value)}
              />
            </div>
          </div>
          {createCollection.isError && (
            <Alert variant="destructive">
              <AlertDescription>
                {createCollection.error instanceof ApiError
                  ? createCollection.error.message
                  : "Failed to create collection."}
              </AlertDescription>
            </Alert>
          )}
          <Button
            onClick={() => createCollection.mutate()}
            disabled={!name || createCollection.isPending}
          >
            {createCollection.isPending ? "Creating…" : "Create collection"}
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
                  <TableHead>Name</TableHead>
                  <TableHead>Description</TableHead>
                  <TableHead>Source authority order</TableHead>
                  <TableHead>Created</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {collections?.map((c) => (
                  <TableRow key={c.id}>
                    <TableCell className="font-medium">{c.name}</TableCell>
                    <TableCell className="text-muted-foreground">
                      {c.description ?? "—"}
                    </TableCell>
                    <TableCell className="text-muted-foreground">
                      {Array.isArray(c.source_authority_config?.order)
                        ? (c.source_authority_config.order as string[]).join(" > ")
                        : "default"}
                    </TableCell>
                    <TableCell className="text-muted-foreground">
                      {new Date(c.created_at).toLocaleString()}
                    </TableCell>
                  </TableRow>
                ))}
                {collections?.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={4} className="text-center text-muted-foreground">
                      No collections yet.
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
