"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { Menu, X } from "lucide-react";

import { NAV_ITEMS, navItemActive } from "@/components/sidebar";
import { cn } from "@/lib/utils";

export function Header() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  // Close the drawer on navigation so it doesn't linger over the new page.
  useEffect(() => {
    setOpen(false);
  }, [pathname]);

  return (
    <>
      <header className="sticky top-0 z-30 flex h-14 items-center border-b border-slate-800 bg-slate-950/80 px-6 backdrop-blur">
        <button
          type="button"
          onClick={() => setOpen(true)}
          className="mr-3 -ml-2 rounded-md p-1.5 text-slate-400 hover:bg-slate-900 hover:text-slate-200 md:hidden"
          aria-label="Open navigation"
        >
          <Menu className="size-5" />
        </button>
        <h1 className="text-sm font-semibold tracking-tight">
          DataModelConsole
        </h1>
        <span className="ml-3 rounded-full border border-slate-700 px-2 py-0.5 text-[10px] uppercase tracking-wider text-slate-400">
          auto-e2e platform
        </span>
      </header>

      {/* Mobile navigation drawer, rendered as a sibling OUTSIDE <header>: the
          header's backdrop-blur makes it the containing block for any fixed
          descendant, so a drawer inside it would resolve `fixed inset-0`
          against the 56px header instead of the viewport. */}
      {open && (
        <div className="fixed inset-0 z-50 md:hidden">
          <div
            className="absolute inset-0 bg-slate-950/70"
            onClick={() => setOpen(false)}
            aria-hidden
          />
          <nav className="absolute inset-y-0 left-0 flex w-64 flex-col border-r border-slate-800 bg-slate-950 p-3">
            <div className="mb-2 flex items-center justify-between px-1">
              <span className="text-sm font-semibold tracking-tight">
                DataModelConsole
              </span>
              <button
                type="button"
                onClick={() => setOpen(false)}
                className="rounded-md p-1.5 text-slate-400 hover:bg-slate-900 hover:text-slate-200"
                aria-label="Close navigation"
              >
                <X className="size-5" />
              </button>
            </div>
            <div className="space-y-1">
              {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
                const active = navItemActive(pathname, href);
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
            </div>
          </nav>
        </div>
      )}
    </>
  );
}
