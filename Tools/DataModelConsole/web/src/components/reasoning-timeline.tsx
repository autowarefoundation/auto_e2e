import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import type { ReasoningHorizon, ReasoningLabelRecord } from "@/types";

// One tag group inside a horizon card (skips empty groups).
function TagGroup({ title, tags }: { title: string; tags: string[] }) {
  if (tags.length === 0) return null;
  return (
    <div>
      <p className="mb-1 text-[10px] uppercase tracking-wider text-slate-500">
        {title}
      </p>
      <div className="flex flex-wrap gap-1">
        {tags.map((tag) => (
          <Badge key={tag} variant="secondary" className="text-[10px]">
            {tag}
          </Badge>
        ))}
      </div>
    </div>
  );
}

// Scalar response fields become single-tag groups.
function responseTags(h: ReasoningHorizon): [string, string[]][] {
  return [
    ["relation", h.relation_to_ego ? [h.relation_to_ego] : []],
    ["hazard event", h.hazard_event ?? []],
    ["cause", h.cause ?? []],
    ["longitudinal", h.longitudinal_response ? [h.longitudinal_response] : []],
    ["lateral", h.lateral_response ? [h.lateral_response] : []],
    ["tactical", h.tactical_response ? [h.tactical_response] : []],
    ["rule", h.rule_response ? [h.rule_response] : []],
  ];
}

// 5-horizon timeline view of a reasoning label record (v2 compositional
// action-relevant ontology).
export function ReasoningTimeline({ label }: { label: ReasoningLabelRecord }) {
  const horizons = [...(label.horizons ?? [])].sort(
    (a, b) => a.horizon_sec - b.horizon_sec,
  );

  // v2 producer writes teacher_model/teacher_provider and dataset_name; fall
  // back to the v1 short fields so older labels still surface provenance.
  const teacher = label.teacher_model ?? label.teacher_provider ?? label.teacher;
  const dataset = label.dataset_name ?? label.dataset;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2 text-xs text-slate-400">
        <span className="font-mono">{label.sample_id}</span>
        {dataset && (
          <>
            <Separator orientation="vertical" className="h-3" />
            <span>dataset: {dataset}</span>
          </>
        )}
        {teacher && (
          <>
            <Separator orientation="vertical" className="h-3" />
            <span>teacher: {teacher}</span>
          </>
        )}
        {label.abstained && (
          <>
            <Separator orientation="vertical" className="h-3" />
            <span className="text-amber-500">
              abstained{label.teacher_error ? `: ${label.teacher_error}` : ""}
            </span>
          </>
        )}
        {label.prompt_version && (
          <>
            <Separator orientation="vertical" className="h-3" />
            <span>prompt: {label.prompt_version}</span>
          </>
        )}
        {label.schema_version && (
          <>
            <Separator orientation="vertical" className="h-3" />
            <span>schema: {label.schema_version}</span>
          </>
        )}
      </div>
      <div className="grid gap-3 lg:grid-cols-5">
        {horizons.map((h) => (
          <div
            key={h.horizon_sec}
            className="rounded-lg border border-slate-800 bg-slate-900/50 p-3"
          >
            <div className="mb-2 flex items-center justify-between">
              <span className="font-mono text-sm font-semibold text-blue-500">
                t+{h.horizon_sec}s
              </span>
              {h.confidence !== undefined && (
                <span className="text-xs text-slate-500">
                  conf {h.confidence.toFixed(2)}
                </span>
              )}
            </div>
            <div className="space-y-2">
              {responseTags(h).map(([title, tags]) => (
                <TagGroup key={title} title={title} tags={tags} />
              ))}
            </div>
            {h.evidence && (
              <p className="mt-2 border-t border-slate-800 pt-2 text-xs leading-relaxed text-slate-400">
                {h.evidence}
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
