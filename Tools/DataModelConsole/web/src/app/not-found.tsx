import Link from "next/link";
import { Home } from "lucide-react";

import { buttonVariants } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

// Themed 404 page: Next renders this for unmatched routes and notFound().
export default function NotFound() {
  return (
    <div className="flex min-h-[60vh] items-center justify-center">
      <Card className="border-slate-800 bg-slate-950/50">
        <CardContent className="flex flex-col items-center gap-3 px-10 py-8 text-center">
          <p className="font-mono text-5xl font-semibold text-slate-200">404</p>
          <p className="text-sm text-slate-400">
            This page could not be found.
          </p>
          <Link
            href="/"
            className={cn(buttonVariants({ variant: "outline", size: "sm" }))}
          >
            <Home className="size-3.5" />
            Back home
          </Link>
        </CardContent>
      </Card>
    </div>
  );
}
