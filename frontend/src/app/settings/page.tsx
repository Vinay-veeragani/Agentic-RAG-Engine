"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useAppStore } from "@/lib/store";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";

export default function SettingsPage() {
  const { data: settings } = useQuery({ queryKey: ["settings"], queryFn: api.settings });
  const developerMode = useAppStore((s) => s.developerMode);
  const toggleDeveloperMode = useAppStore((s) => s.toggleDeveloperMode);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Settings</h1>
        <p className="text-muted-foreground">
          Read-only server configuration and client-side preferences. Secrets are never exposed
          by the API.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Client preferences</CardTitle>
        </CardHeader>
        <CardContent className="flex items-center justify-between">
          <div>
            <Label htmlFor="dev-mode">Developer Mode</Label>
            <p className="text-sm text-muted-foreground">
              Show query/trace IDs, provider names, iteration counts, latency, and raw JSON on
              the Ask page.
            </p>
          </div>
          <Switch id="dev-mode" checked={developerMode} onCheckedChange={toggleDeveloperMode} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Server configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-1 text-sm">
          {settings ? (
            <>
              <Row label="Environment" value={settings.app_env} />
              <Row label="LLM provider" value={settings.llm_provider} />
              <Row label="Embedding provider" value={settings.embedding_provider} />
              <Row label="Reranker provider" value={settings.reranker_provider} />
              <Separator className="my-2" />
              <Row label="Max retrieval iterations" value={String(settings.max_retrieval_iterations)} />
              <Row label="Max retrieval calls" value={String(settings.max_retrieval_calls)} />
              <Row label="Max tokens per query" value={String(settings.max_tokens_per_query)} />
              <Row label="Max query latency (s)" value={String(settings.max_query_latency_seconds)} />
              <Row
                label="Max upload size"
                value={`${(settings.max_upload_size_bytes / (1024 * 1024)).toFixed(0)} MB`}
              />
              <Separator className="my-2" />
              <Row label="API-key auth" value={settings.auth_enabled ? "enabled" : "disabled"} />
              <Row
                label="Rate limiting"
                value={
                  settings.rate_limit_enabled
                    ? `enabled (${settings.rate_limit_requests_per_window} req / ${settings.rate_limit_window_seconds}s)`
                    : "disabled"
                }
              />
              <Row label="Cache backend" value={settings.cache_backend} />
              <Row label="Workers" value={String(settings.workers)} />
            </>
          ) : (
            <p className="text-muted-foreground">Loading…</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between py-1">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-mono">{value}</span>
    </div>
  );
}
