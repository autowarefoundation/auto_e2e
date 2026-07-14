import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { FlytePhase } from "@/types";

type StatusTone = "success" | "warning" | "error" | "info" | "neutral";

const TONE_CLASSES: Record<StatusTone, string> = {
  success: "border-green-500/40 bg-green-500/15 text-green-500",
  warning: "border-amber-500/40 bg-amber-500/15 text-amber-500",
  error: "border-red-500/40 bg-red-500/15 text-red-500",
  info: "border-blue-500/40 bg-blue-500/15 text-blue-500",
  neutral: "border-slate-500/40 bg-slate-500/15 text-slate-400",
};

export function flytePhaseTone(phase: FlytePhase): StatusTone {
  switch (phase) {
    case "SUCCEEDED":
      return "success";
    case "RUNNING":
    case "SUCCEEDING":
    case "QUEUED":
      return "warning";
    case "FAILED":
    case "FAILING":
    case "ABORTED":
    case "ABORTING":
    case "TIMED_OUT":
      return "error";
    default:
      return "neutral";
  }
}

export function mlflowStatusTone(status: string): StatusTone {
  switch (status) {
    case "FINISHED":
      return "success";
    case "RUNNING":
    case "SCHEDULED":
      return "warning";
    case "FAILED":
    case "KILLED":
      return "error";
    default:
      return "neutral";
  }
}

export function StatusBadge({
  label,
  tone,
}: {
  label: string;
  tone: StatusTone;
}) {
  return (
    <Badge variant="outline" className={cn("font-mono", TONE_CLASSES[tone])}>
      {label}
    </Badge>
  );
}
