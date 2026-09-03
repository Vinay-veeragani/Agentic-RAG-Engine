"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  BookOpen,
  FlaskConical,
  FolderKanban,
  Gauge,
  MessageSquareText,
  Search,
  Settings,
  Waypoints,
} from "lucide-react";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { href: "/", label: "Knowledge", icon: BookOpen },
  { href: "/collections", label: "Collections", icon: FolderKanban },
  { href: "/documents", label: "Documents", icon: FolderKanban },
  { href: "/search", label: "Search", icon: Search },
  { href: "/ask", label: "Ask", icon: MessageSquareText },
  { href: "/traces", label: "Retrieval Traces", icon: Waypoints },
  { href: "/evaluations", label: "Evaluations", icon: FlaskConical },
  { href: "/observability", label: "Observability", icon: Activity },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="flex h-full w-60 flex-col border-r border-border bg-sidebar text-sidebar-foreground">
      <div className="flex items-center gap-2 px-4 py-4">
        <Gauge className="h-5 w-5 text-primary" />
        <span className="font-semibold tracking-tight">Agentic RAG</span>
      </div>
      <nav className="flex-1 space-y-0.5 px-2">
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
          const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors",
                active
                  ? "bg-primary/10 text-primary font-medium"
                  : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
              )}
            >
              <Icon className="h-4 w-4" />
              {label}
            </Link>
          );
        })}
      </nav>
      <div className="px-4 py-3 text-xs text-muted-foreground">
        Advanced Agentic RAG Platform
      </div>
    </aside>
  );
}
