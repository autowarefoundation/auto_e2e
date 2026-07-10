"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Boxes,
  Brain,
  Clapperboard,
  Database,
  Home,
  Workflow,
} from "lucide-react";

import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { href: "/", label: "Home", icon: Home },
  { href: "/datasets", label: "Datasets", icon: Database },
  { href: "/reasoning-labels", label: "Reasoning Labels", icon: Brain },
  { href: "/models", label: "Models", icon: Boxes },
  { href: "/runs", label: "Runs", icon: Workflow },
  { href: "/scenes", label: "Scenes", icon: Clapperboard },
] as const;

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed inset-y-0 left-0 z-40 hidden w-56 flex-col border-r border-slate-800 bg-slate-950 md:flex">
      <div className="flex h-14 items-center border-b border-slate-800 px-4">
        <Link href="/" className="text-sm font-semibold tracking-tight">
          DataModelConsole
        </Link>
      </div>
      <nav className="flex-1 space-y-1 p-3">
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
          const active =
            href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-2.5 rounded-md px-3 py-2 text-sm transition-colors",
                active
                  ? "bg-slate-800 text-slate-50"
                  : "text-slate-400 hover:bg-slate-900 hover:text-slate-200",
              )}
            >
              <Icon className="size-4" />
              {label}
            </Link>
          );
        })}
      </nav>
      <div className="border-t border-slate-800 p-4 text-xs text-slate-500">
        Phase 1 — read-only
      </div>
    </aside>
  );
}
