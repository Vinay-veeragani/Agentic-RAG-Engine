"use client";

import Link from "next/link";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { FolderKanban, MessageSquareText, Search } from "lucide-react";

export default function KnowledgePage() {
  const { data: collections } = useQuery({
    queryKey: ["collections"],
    queryFn: api.collections.list,
  });
  const { data: documents } = useQuery({
    queryKey: ["documents"],
    queryFn: () => api.documents.list(),
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Knowledge</h1>
        <p className="text-muted-foreground">
          Autonomous knowledge retrieval, evidence gathering, verification, and grounded answer
          generation — not a chat-with-PDF demo.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Collections
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-semibold">{collections?.length ?? "…"}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Documents</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-semibold">{documents?.length ?? "…"}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">Pipeline</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm">
              Query → Analyze → Plan → Hybrid Retrieval → Rerank → Evidence Judge → Refine →
              Synthesize → Validate Citations
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <QuickLink
          href="/documents"
          icon={FolderKanban}
          title="Ingest documents"
          description="Upload PDF, DOCX, TXT, Markdown, HTML, CSV, or JSON into a collection."
        />
        <QuickLink
          href="/search"
          icon={Search}
          title="Search"
          description="Run hybrid dense + sparse retrieval directly, with full score breakdowns."
        />
        <QuickLink
          href="/ask"
          icon={MessageSquareText}
          title="Ask"
          description="The full agentic pipeline: a grounded, cited answer with an expandable trace."
        />
      </div>
    </div>
  );
}

function QuickLink({
  href,
  icon: Icon,
  title,
  description,
}: {
  href: string;
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  description: string;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Icon className="h-4 w-4 text-primary" />
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-sm text-muted-foreground">{description}</p>
        <Button size="sm" variant="secondary" nativeButton={false} render={<Link href={href} />}>
          Open
        </Button>
      </CardContent>
    </Card>
  );
}
