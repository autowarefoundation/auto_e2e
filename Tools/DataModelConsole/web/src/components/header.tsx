"use client";

import { Dialog } from "@base-ui/react/dialog";
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
    <Dialog.Root open={open} onOpenChange={setOpen}>
      <header className="sticky top-0 z-30 flex h-14 items-center border-b border-slate-800 bg-slate-950/80 px-6 backdrop-blur">
        <Dialog.Trigger
          className="mr-3 -ml-2 rounded-md p-1.5 text-slate-400 hover:bg-slate-900 hover:text-slate-200 md:hidden"
          aria-label="Open navigation"
        >
          <Menu className="size-5" />
        </Dialog.Trigger>
        <h1 className="text-sm font-semibold tracking-tight">
          DataModelConsole
        </h1>
        <span className="ml-3 rounded-full border border-slate-700 px-2 py-0.5 text-[10px] uppercase tracking-wider text-slate-400">
          auto-e2e platform
        </span>
      </header>

      <Dialog.Portal>
        <Dialog.Backdrop className="fixed inset-0 z-50 bg-slate-950/70 md:hidden" />
        <Dialog.Popup
          aria-modal="true"
          className="fixed inset-y-0 left-0 z-50 flex w-64 flex-col border-r border-slate-800 bg-slate-950 p-3 md:hidden"
        >
          <nav aria-label="Primary navigation" className="flex min-h-0 flex-1 flex-col">
            <div className="mb-2 flex items-center justify-between px-1">
              <Dialog.Title className="text-sm font-semibold tracking-tight">
                DataModelConsole
              </Dialog.Title>
              <Dialog.Close
                className="rounded-md p-1.5 text-slate-400 hover:bg-slate-900 hover:text-slate-200"
                aria-label="Close navigation"
              >
                <X className="size-5" />
              </Dialog.Close>
            </div>
            <div className="space-y-1">
              {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
                const active = navItemActive(pathname, href);
                return (
                  <Link
                    key={href}
                    href={href}
                    aria-current={active ? "page" : undefined}
                    onClick={() => setOpen(false)}
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
        </Dialog.Popup>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
