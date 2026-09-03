"use client";

import { useQuery } from "@tanstack/react-query";
import { Code2 } from "lucide-react";
import { api } from "@/lib/api";
import { useAppStore } from "@/lib/store";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";

export function Header() {
  const developerMode = useAppStore((s) => s.developerMode);
  const toggleDeveloperMode = useAppStore((s) => s.toggleDeveloperMode);

  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 15_000,
  });

  const healthy = health?.status === "ok";

  return (
    <header className="flex h-14 items-center justify-between border-b border-border px-6">
      <div className="flex items-center gap-2">
        <span
          className={`h-2 w-2 rounded-full ${healthy ? "bg-emerald-500" : "bg-amber-500"}`}
        />
        <span className="text-sm text-muted-foreground">
          {health ? (healthy ? "Connected" : "Degraded") : "Connecting..."}
          {health && (
            <span className="ml-2">
              db: <Badge variant="outline">{health.database}</Badge> cache:{" "}
              <Badge variant="outline">{health.cache}</Badge>
            </span>
          )}
        </span>
      </div>
      <div className="flex items-center gap-2">
        <Code2 className="h-4 w-4 text-muted-foreground" />
        <Label htmlFor="developer-mode" className="text-sm text-muted-foreground">
          Developer Mode
        </Label>
        <Switch id="developer-mode" checked={developerMode} onCheckedChange={toggleDeveloperMode} />
      </div>
    </header>
  );
}
