// Shared helpers for the ODD Search page: query parsing and limits (the same
// contract as the Go ValidateODDQuery), label names, and value colours.

import type {
  ODDFacet,
  ODDOp,
  ODDQuery,
  ODDQueryCondition,
  ODDQueryGroup,
} from "@/types";

// Limits from api/internal/service/odd_query.go.
export const MAX_GROUPS = 8;
export const MAX_CONDITIONS = 24;
const MAX_VALUES = 32;
const MAX_SOURCES = 8;
const MAX_DURATION_S = 3600;

const KEY = /^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$/;
const VALUE = /^[a-z0-9][a-z0-9_]*$/;

const validKey = (key: unknown): key is string =>
  typeof key === "string" && key.length <= 128 && KEY.test(key);
const validValue = (value: unknown): value is string =>
  typeof value === "string" && value.length <= 64 && VALUE.test(value);
const validOp = (op: unknown): op is ODDOp => op === "and" || op === "or";

export const EMPTY_QUERY: ODDQuery = { op: "and", groups: [] };

function cleanCondition(raw: unknown): ODDQueryCondition | null {
  if (!raw || typeof raw !== "object") return null;
  const c = raw as Record<string, unknown>;
  if (!validKey(c.key) || !Array.isArray(c.values)) return null;
  const values = [...new Set(c.values.filter(validValue))].slice(0, MAX_VALUES);
  if (values.length === 0) return null;
  const condition: ODDQueryCondition = { key: c.key, values };
  if (c.not === true) condition.not = true;
  const duration = Number(c.min_duration_s);
  if (Number.isFinite(duration) && duration > 0) {
    condition.min_duration_s = Math.min(duration, MAX_DURATION_S);
  }
  const confidence = Number(c.min_confidence);
  if (Number.isFinite(confidence) && confidence > 0) {
    condition.min_confidence = Math.min(confidence, 1);
  }
  if (Array.isArray(c.sources)) {
    const sources = [...new Set(c.sources.filter(validValue))].slice(0, MAX_SOURCES);
    if (sources.length > 0) condition.sources = sources;
  }
  return condition;
}

// cleanQuery keeps the executable part of a query: complete conditions
// (a key and at least one value) within the API limits. A hand-edited URL
// therefore degrades to a smaller query instead of a failed search.
export function cleanQuery(raw: unknown): ODDQuery {
  if (!raw || typeof raw !== "object") return EMPTY_QUERY;
  const q = raw as Record<string, unknown>;
  let budget = MAX_CONDITIONS;
  const groups: ODDQueryGroup[] = [];
  for (const rawGroup of Array.isArray(q.groups) ? q.groups : []) {
    if (groups.length >= MAX_GROUPS || budget === 0) break;
    const g = (rawGroup ?? {}) as Record<string, unknown>;
    const conditions = (Array.isArray(g.conditions) ? g.conditions : [])
      .map(cleanCondition)
      .filter((c): c is ODDQueryCondition => c !== null)
      .slice(0, budget);
    if (conditions.length === 0) continue;
    budget -= conditions.length;
    groups.push({ op: validOp(g.op) ? g.op : "and", conditions });
  }
  return { op: validOp(q.op) ? q.op : "and", groups };
}

export function parseQueryParam(raw: string | null): ODDQuery {
  if (!raw) return EMPTY_QUERY;
  try {
    return cleanQuery(JSON.parse(raw));
  } catch {
    return EMPTY_QUERY;
  }
}

export function conditionCount(query: ODDQuery): number {
  return query.groups.reduce((n, g) => n + g.conditions.length, 0);
}

const NAMESPACES: Record<string, string> = {
  odd: "Operating domain",
  event: "Events",
  perception: "Perception",
};

export function namespaceOf(key: string): string {
  return key.split(".", 1)[0];
}

export function namespaceLabel(namespace: string): string {
  return NAMESPACES[namespace] ?? namespace;
}

// keyLabel turns "odd.road.junction_type" into "Road junction type".
export function keyLabel(key: string): string {
  const words = key.split(".").slice(1).join(" ").replaceAll("_", " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

export function valueLabel(value: string): string {
  return value.replaceAll("_", " ");
}

// Dark-surface categorical slots in fixed order (validated: lightness band,
// chroma floor, adjacent CVD ΔE >= 8.4, normal-vision ΔE >= 19.3, >= 3:1 on
// #0f172a). Bars also carry their value as text, so colour is never the
// only cue.
const SLOTS = [
  "#3987e5",
  "#d95926",
  "#199e70",
  "#c98500",
  "#d55181",
  "#008300",
  "#9085e9",
  "#e66767",
];
export const OTHER_COLOR = "#64748b";

// groupAccent marks query group i with the i-th categorical slot.
export function groupAccent(i: number): string {
  return SLOTS[i % SLOTS.length];
}

// valueColor follows the value, not its rank in a result: the slot is the
// value's position in the key's vocabulary over every labelled scene, so a
// filter never repaints a value. Past eight values a key folds to grey.
export function valueColor(
  vocabulary: Map<string, ODDFacet>,
  key: string,
  value: string,
): string {
  const index = vocabulary.get(key)?.values.findIndex((v) => v.value === value) ?? -1;
  return index >= 0 && index < SLOTS.length ? SLOTS[index] : OTHER_COLOR;
}

export function seconds(ns: number, digits = 1): string {
  return `${(ns / 1e9).toFixed(digits)} s`;
}

// describeQuery renders a query as one readable line.
export function describeQuery(query: ODDQuery): string {
  if (query.groups.length === 0) return "Every labelled scene";
  const joinWord = (op: ODDOp) => (op === "and" ? " AND " : " OR ");
  return query.groups
    .map((group) => {
      const inner = group.conditions
        .map((c) => {
          const values =
            c.values.length === 1
              ? valueLabel(c.values[0])
              : `any of ${c.values.map(valueLabel).join(", ")}`;
          const refine = [
            c.min_duration_s ? `≥ ${c.min_duration_s} s` : "",
            c.min_confidence ? `conf ≥ ${c.min_confidence}` : "",
            c.sources?.length ? `from ${c.sources.join("/")}` : "",
          ].filter(Boolean);
          return `${keyLabel(c.key)} ${c.not ? "is not" : "is"} ${values}${
            refine.length ? ` (${refine.join(", ")})` : ""
          }`;
        })
        .join(joinWord(group.op));
      return query.groups.length > 1 && group.conditions.length > 1
        ? `(${inner})`
        : inner;
    })
    .join(joinWord(query.op));
}
