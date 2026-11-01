"use client";

// Form for an ODD query: groups joined by AND/OR at the top, conditions joined
// by AND/OR inside each group. A condition picks a label key, "is" or "is
// not", one or more values (IN), and optional refinements (minimum interval
// length, minimum confidence, sources). Incomplete conditions stay in the
// draft and are left out of the search until they have a key and a value.

import { useState } from "react";
import { Plus, SlidersHorizontal, Trash2, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  MAX_CONDITIONS,
  MAX_GROUPS,
  conditionCount,
  groupAccent,
  newConditionID,
  keyLabel,
  namespaceLabel,
  namespaceOf,
  valueColor,
  valueLabel,
} from "@/lib/odd";
import { cn } from "@/lib/utils";
import type {
  ODDFacet,
  ODDOp,
  ODDQuery,
  ODDQueryCondition,
  ODDQueryGroup,
} from "@/types";

export function emptyCondition(key = "", id = newConditionID()): ODDQueryCondition {
  return { key, values: [], id };
}

function OpToggle({
  op,
  onChange,
  label,
}: {
  op: ODDOp;
  onChange: (op: ODDOp) => void;
  label: string;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex rounded-md bg-slate-900 p-0.5 ring-1 ring-slate-700">
      {(["and", "or"] as const).map((value) => (
        <button
          key={value}
          type="button"
          role="radio"
          aria-checked={op === value}
          onClick={() => onChange(value)}
          className={cn(
            "rounded px-2.5 py-0.5 text-[11px] font-semibold tracking-wide transition-colors",
            op === value ? "bg-slate-200 text-slate-900" : "text-slate-400 hover:text-slate-200",
          )}
        >
          {value === "and" ? "ALL" : "ANY"}
        </button>
      ))}
    </div>
  );
}

function ConditionEditor({
  condition,
  vocabulary,
  byKey,
  onChange,
  onRemove,
  index,
}: {
  condition: ODDQueryCondition;
  vocabulary: ODDFacet[];
  byKey: Map<string, ODDFacet>;
  onChange: (c: ODDQueryCondition) => void;
  onRemove: () => void;
  index: string;
}) {
  const facet = byKey.get(condition.key);
  const refined = Boolean(
    condition.min_duration_s || condition.min_confidence || condition.sources?.length,
  );
  const [open, setOpen] = useState(refined);
  const namespaces = [...new Set(vocabulary.map((f) => namespaceOf(f.key)))];
  const toggle = (list: string[] | undefined, item: string) =>
    list?.includes(item) ? list.filter((x) => x !== item) : [...(list ?? []), item];

  return (
    <div
      className="rounded-lg border border-slate-800 bg-slate-950/60 p-3"
      role="group"
      aria-label={`Condition ${index}`}
    >
      <div className="flex flex-wrap items-center gap-2">
        <select
          value={condition.key}
          onChange={(e) => onChange(emptyCondition(e.target.value, condition.id))}
          aria-label={`Condition ${index} label`}
          className="h-8 w-full min-w-0 rounded-md border border-slate-700 bg-slate-900 px-2 text-sm text-slate-100 sm:w-auto sm:max-w-xs sm:flex-1"
        >
          <option value="">Choose a label…</option>
          {namespaces.map((namespace) => (
            <optgroup key={namespace} label={namespaceLabel(namespace)}>
              {vocabulary
                .filter((f) => namespaceOf(f.key) === namespace)
                .map((f) => (
                  <option key={f.key} value={f.key}>
                    {keyLabel(f.key)}
                  </option>
                ))}
            </optgroup>
          ))}
        </select>
        <div role="radiogroup" aria-label={`Condition ${index} polarity`} className="inline-flex rounded-md bg-slate-900 p-0.5 ring-1 ring-slate-700">
          {[false, true].map((not) => (
            <button
              key={String(not)}
              type="button"
              role="radio"
              aria-checked={Boolean(condition.not) === not}
              onClick={() => onChange({ ...condition, not: not || undefined })}
              className={cn(
                "rounded px-2 py-0.5 text-[11px] font-medium transition-colors",
                Boolean(condition.not) === not
                  ? not
                    ? "bg-rose-500/90 text-white"
                    : "bg-slate-200 text-slate-900"
                  : "text-slate-400 hover:text-slate-200",
              )}
            >
              {not ? "is not" : "is"}
            </button>
          ))}
        </div>
        <div className="ml-auto flex items-center gap-1">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            aria-expanded={open}
            aria-label={`Refine condition ${index}`}
            onClick={() => setOpen(!open)}
            className={cn(refined && "text-sky-300")}
          >
            <SlidersHorizontal className="size-3.5" />
            <span className="hidden sm:inline">Refine</span>
          </Button>
          <Button type="button" variant="ghost" size="sm" aria-label={`Remove condition ${index}`} onClick={onRemove}>
            <X className="size-3.5" />
          </Button>
        </div>
      </div>

      {facet ? (
        <div className="mt-2.5 flex flex-wrap gap-1.5" aria-label={`Condition ${index} values`}>
          {facet.values.map(({ value, scenes }) => {
            const selected = condition.values.includes(value);
            return (
              <button
                key={value}
                type="button"
                aria-pressed={selected}
                onClick={() => onChange({ ...condition, values: toggle(condition.values, value) })}
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs transition-colors",
                  selected
                    ? "border-slate-300 bg-slate-100 text-slate-900"
                    : "border-slate-700 bg-slate-900/60 text-slate-300 hover:border-slate-500",
                )}
              >
                <span
                  className="inline-block size-2 rounded-full"
                  style={{ background: valueColor(byKey, condition.key, value) }}
                />
                {valueLabel(value)}
                <span className="tabular-nums text-slate-500">
                  {scenes}
                </span>
              </button>
            );
          })}
        </div>
      ) : (
        <p className="mt-2 text-xs text-slate-500">Pick a label to see its values.</p>
      )}
      {facet && condition.values.length === 0 && (
        <p className="mt-1.5 text-[11px] text-amber-300/80">Select at least one value to apply this condition.</p>
      )}

      {open && facet && (
        <div className="mt-3 grid gap-3 rounded-md bg-slate-900/60 p-3 sm:grid-cols-3">
          <label className="space-y-1 text-[11px] text-slate-400">
            <span>Interval lasts at least</span>
            <span className="flex items-center gap-1.5">
              {/* Uncontrolled: a controlled number input would erase the
                  partial "0" of "0.5" and turn it into 5. */}
              <input
                type="number"
                min={0}
                max={3600}
                step={0.5}
                defaultValue={condition.min_duration_s ?? ""}
                placeholder="0"
                onChange={(e) => {
                  const v = e.target.valueAsNumber;
                  onChange({ ...condition, min_duration_s: v > 0 ? v : undefined });
                }}
                className="h-7 w-20 rounded border border-slate-700 bg-slate-950 px-2 text-xs text-slate-100"
              />
              <span>s</span>
            </span>
          </label>
          <label className="space-y-1 text-[11px] text-slate-400">
            <span>
              Confidence at least{" "}
              <span className="tabular-nums text-slate-200">
                {Math.round((condition.min_confidence ?? 0) * 100)}%
              </span>
            </span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={condition.min_confidence ?? 0}
              onChange={(e) => {
                const v = Number(e.target.value);
                onChange({ ...condition, min_confidence: v > 0 ? v : undefined });
              }}
              className="w-full accent-sky-400"
            />
          </label>
          <div className="space-y-1 text-[11px] text-slate-400">
            <span>From sources</span>
            <div className="flex flex-wrap gap-1">
              {facet.sources.map((source) => {
                const selected = condition.sources?.includes(source) ?? false;
                return (
                  <button
                    key={source}
                    type="button"
                    aria-pressed={selected}
                    onClick={() => {
                      const sources = toggle(condition.sources, source);
                      onChange({ ...condition, sources: sources.length ? sources : undefined });
                    }}
                    className={cn(
                      "rounded border px-1.5 py-0.5 font-mono text-[10px]",
                      selected
                        ? "border-sky-400 bg-sky-400/15 text-sky-200"
                        : "border-slate-700 text-slate-400 hover:border-slate-500",
                    )}
                  >
                    {source}
                  </button>
                );
              })}
              {!condition.sources?.length && <span className="text-slate-600">any</span>}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export function QueryBuilder({
  draft,
  onChange,
  vocabulary,
}: {
  draft: ODDQuery;
  onChange: (q: ODDQuery) => void;
  vocabulary: ODDFacet[];
}) {
  const byKey = new Map(vocabulary.map((f) => [f.key, f]));
  const total = conditionCount(draft);
  const setGroup = (index: number, group: ODDQueryGroup | null) => {
    const groups = draft.groups.slice();
    if (group === null || group.conditions.length === 0) groups.splice(index, 1);
    else groups[index] = group;
    onChange({ ...draft, groups });
  };

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2 text-sm text-slate-300">
        <span>Match scenes where</span>
        <OpToggle op={draft.op} onChange={(op) => onChange({ ...draft, op })} label="Join groups with" />
        <span>of these groups hold</span>
      </div>

      {draft.groups.length === 0 && (
        <div className="rounded-lg border border-dashed border-slate-700 p-6 text-center text-sm text-slate-500">
          No conditions yet. Every labelled scene is shown. Add a group, or click a value in the
          label explorer.
        </div>
      )}

      {draft.groups.map((group, g) => (
        <div
          key={g}
          role="group"
          aria-label={`Group ${g + 1}`}
          className="relative rounded-xl border border-slate-800 bg-slate-900/30 p-3 pl-4"
        >
          <span
            aria-hidden
            className="absolute inset-y-3 left-0 w-1 rounded-full"
            style={{ background: groupAccent(g) }}
          />
          <div className="mb-2 flex flex-wrap items-center gap-2 text-xs text-slate-400">
            <span className="font-semibold text-slate-200">Group {g + 1}</span>
            <OpToggle op={group.op} onChange={(op) => setGroup(g, { ...group, op })} label={`Join group ${g + 1} with`} />
            <span>of</span>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="ml-auto"
              aria-label={`Remove group ${g + 1}`}
              onClick={() => setGroup(g, null)}
            >
              <Trash2 className="size-3.5" />
            </Button>
          </div>
          <div className="space-y-2">
            {group.conditions.map((condition, c) => (
              <div key={condition.id ?? `${g}-${c}`}>
                {c > 0 && (
                  <div className="my-1 pl-2 text-[10px] font-semibold tracking-widest text-slate-500">
                    {group.op === "and" ? "AND" : "OR"}
                  </div>
                )}
                <ConditionEditor
                  index={`${g + 1}.${c + 1}`}
                  condition={condition}
                  vocabulary={vocabulary}
                  byKey={byKey}
                  onChange={(next) => {
                    const conditions = group.conditions.slice();
                    conditions[c] = next;
                    setGroup(g, { ...group, conditions });
                  }}
                  onRemove={() =>
                    setGroup(g, { ...group, conditions: group.conditions.filter((_, i) => i !== c) })
                  }
                />
              </div>
            ))}
          </div>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="mt-2"
            disabled={total >= MAX_CONDITIONS}
            onClick={() => setGroup(g, { ...group, conditions: [...group.conditions, emptyCondition()] })}
          >
            <Plus className="size-3.5" /> Add condition
          </Button>
        </div>
      ))}

      <div className="flex flex-wrap items-center gap-2">
        <Button
          type="button"
          variant="secondary"
          size="sm"
          disabled={draft.groups.length >= MAX_GROUPS || total >= MAX_CONDITIONS}
          onClick={() =>
            onChange({ ...draft, groups: [...draft.groups, { op: "and", conditions: [emptyCondition()] }] })
          }
        >
          <Plus className="size-3.5" /> Add group
        </Button>
        {draft.groups.length > 0 && (
          <Button type="button" variant="ghost" size="sm" onClick={() => onChange({ op: "and", groups: [] })}>
            Clear all
          </Button>
        )}
        <span className="ml-auto text-[11px] tabular-nums text-slate-500">
          {total}/{MAX_CONDITIONS} conditions · {draft.groups.length}/{MAX_GROUPS} groups
        </span>
      </div>
    </div>
  );
}
