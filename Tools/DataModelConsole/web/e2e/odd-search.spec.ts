import { expect, test, type Page, type Route } from "@playwright/test";

const S = 1_000_000_000;

function fulfillJSON(route: Route, body: unknown, status = 200) {
  return route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

const info = {
  dataset: "kitscenes",
  version: "v3.7",
  label_version: "v3.1",
  labelset_id: "oddls-0123456789abcdef0123456789abcdef",
  manifest_sha256: "a".repeat(64),
  quality: {
    structural_status: "passed",
    audit_status: "pending_human_audit",
    certification_status: "experimental",
  },
  labeled_scenes: 404,
};

const vocabulary = [
  {
    key: "odd.environment.day_phase",
    values: [
      { value: "day", scenes: 380 },
      { value: "dusk", scenes: 20 },
    ],
    sources: ["vlm"],
  },
  {
    key: "odd.road.context",
    values: [
      { value: "urban", scenes: 299 },
      { value: "rural", scenes: 223 },
    ],
    sources: ["map_route", "vlm"],
  },
  {
    key: "perception.image.blur",
    values: [{ value: "none", scenes: 400 }],
    sources: ["image_qc"],
  },
];

const urbanLane = {
  key: "odd.road.context",
  intervals: [
    { start_ns: 2 * S, end_ns: 9 * S, values: ["urban"], source: "map_route", confidence: 0.9 },
  ],
};

function scene(uid: string, shard: string, lanes: unknown[]) {
  return { scene_uid: uid, shard, duration_ns: 20 * S, distance_m: 120.4, lanes };
}

function searchResponse(q: { groups?: { conditions: { key: string }[] }[] }) {
  const filtered = (q.groups ?? []).length > 0;
  return {
    ...info,
    query: q,
    vocabulary,
    facets: vocabulary,
    scenes: filtered
      ? [scene("0a1b2c3d-0000-0000-0000-000000000001", "part-aaaa-train-000000.tar", [urbanLane])]
      : [
          scene("0a1b2c3d-0000-0000-0000-000000000001", "part-aaaa-train-000000.tar", []),
          scene("0a1b2c3d-0000-0000-0000-000000000002", "part-bbbb-train-000000.tar", []),
        ],
    page: { limit: 20, offset: 0, total: filtered ? 1 : 2, more: false },
  };
}

async function mockAPI(page: Page, searches: URL[]) {
  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/api/v1/datasets") {
      return fulfillJSON(route, {
        datasets: [
          { name: "kitscenes-val", version: "v3.7", prefix: "kitscenes-val/v3.7/shards/" },
          { name: "kitscenes", version: "v3.7", prefix: "kitscenes/v3.7/shards/" },
        ],
      });
    }
    if (url.pathname === "/api/v1/odd/scenes") {
      searches.push(url);
      if (url.searchParams.get("dataset") !== "kitscenes") {
        return fulfillJSON(route, { error: "no ODD labels for this dataset version", code: "NOT_FOUND" }, 404);
      }
      return fulfillJSON(route, searchResponse(JSON.parse(url.searchParams.get("q") ?? "{}")));
    }
    if (url.pathname.startsWith("/api/v1/odd/scenes/")) {
      return fulfillJSON(route, {
        ...info,
        scene: scene("0a1b2c3d-0000-0000-0000-000000000001", "part-aaaa-train-000000.tar", [
          {
            key: "odd.environment.day_phase",
            intervals: [{ start_ns: 0, end_ns: 20 * S, values: ["day"], source: "vlm", confidence: 0.8 }],
          },
          urbanLane,
          {
            key: "perception.image.blur",
            intervals: [
              { start_ns: 0, end_ns: 5 * S, values: ["none"], source: "image_qc", confidence: 0.99, camera_id: "front" },
              { start_ns: 1 * S, end_ns: 6 * S, values: ["none"], source: "image_qc", confidence: 0.99, camera_id: "rear" },
            ],
          },
        ]),
      });
    }
    return fulfillJSON(route, { error: "unmocked", code: "NOT_FOUND" }, 404);
  });
}

function lastQuery(searches: URL[]) {
  return JSON.parse(searches.at(-1)?.searchParams.get("q") ?? "null");
}

test("ODD search adds a value from the label explorer and shows its intervals", async ({ page }) => {
  const searches: URL[] = [];
  await mockAPI(page, searches);
  await page.goto("/odd");

  await expect(page.getByText("of 404 scenes match.")).toBeVisible();
  await expect(page.getByText("kitscenes v3.1")).toBeVisible();
  await expect(page.getByText("experimental")).toBeVisible();
  expect(searches[0].searchParams.get("version")).toBe("v3.7");

  await page.getByRole("button", { name: "Add Road context is urban, 299 scenes" }).click();
  await expect(page).toHaveURL(/q=/);
  await expect.poll(() => lastQuery(searches)).toEqual({
    op: "and",
    groups: [{ op: "and", conditions: [{ key: "odd.road.context", values: ["urban"] }] }],
  });
  const bar = page.getByRole("button", { name: /^Road context: urban, 2\.0 s to 9\.0 s, map_route/ });
  await expect(bar).toBeVisible();
  await bar.hover();
  await expect(page.getByRole("tooltip")).toContainText("Confidence");
  await expect(page.getByRole("tooltip")).toContainText("90%");
  await expect(page.getByRole("link", { name: "Open part-aaaa-train-000000.tar in the player" })).toHaveAttribute(
    "href",
    "/scenes/kitscenes/part-aaaa-train-000000.tar/0?version=v3.7",
  );
});

test("ODD query builder sends groups, NOT, IN and refinements", async ({ page }) => {
  const searches: URL[] = [];
  await mockAPI(page, searches);
  await page.goto("/odd?dataset=kitscenes");
  await expect(page.getByText("of 404 scenes match.")).toBeVisible();

  await page.getByRole("button", { name: "Add group" }).click();
  await page.getByLabel("Condition 1.1 label").selectOption("odd.road.context");
  const values = page.getByLabel("Condition 1.1 values");
  await values.getByRole("button", { name: /urban/ }).click();
  await values.getByRole("button", { name: /rural/ }).click();
  await page.getByRole("radiogroup", { name: "Condition 1.1 polarity" }).getByRole("radio", { name: "is not" }).click();
  await page.getByRole("button", { name: "Refine condition 1.1" }).click();
  // Typed key by key: a partial "0" must not be erased mid-entry.
  await page.getByRole("spinbutton").pressSequentially("0.5");
  await page.getByRole("button", { name: "vlm", exact: true }).click();

  await page.getByRole("button", { name: "Add condition" }).click();
  await page.getByLabel("Condition 1.2 label").selectOption("odd.environment.day_phase");
  await page.getByLabel("Condition 1.2 values").getByRole("button", { name: /dusk/ }).click();
  await page.getByRole("radiogroup", { name: "Join group 1 with" }).getByRole("radio", { name: "ANY" }).click();

  await expect.poll(() => lastQuery(searches)).toEqual({
    op: "and",
    groups: [
      {
        op: "or",
        conditions: [
          { key: "odd.road.context", values: ["urban", "rural"], not: true, min_duration_s: 0.5, sources: ["vlm"] },
          { key: "odd.environment.day_phase", values: ["dusk"] },
        ],
      },
    ],
  });
  await expect(page.getByText("2/24 conditions · 1/8 groups")).toBeVisible();
});

test("ODD search leaves an incomplete condition out of the query", async ({ page }) => {
  const searches: URL[] = [];
  await mockAPI(page, searches);
  await page.goto("/odd?dataset=kitscenes");
  await expect(page.getByText("of 404 scenes match.")).toBeVisible();

  await page.getByRole("button", { name: "Add group" }).click();
  await page.getByLabel("Condition 1.1 label").selectOption("odd.road.context");
  await expect(page.getByText("Select at least one value to apply this condition.")).toBeVisible();
  await page.waitForTimeout(600);
  expect(searches.every((url) => url.searchParams.get("q") === JSON.stringify({ op: "and", groups: [] }))).toBe(true);
});

test("ODD scene expands to every label grouped by namespace", async ({ page }) => {
  const searches: URL[] = [];
  await mockAPI(page, searches);
  await page.goto("/odd?dataset=kitscenes");
  const card = page.getByRole("article", { name: "Scene part-aaaa-train-000000.tar" });
  await card.getByRole("button", { name: "All labels" }).click();

  await expect(card.getByRole("region", { name: "Operating domain" })).toBeVisible();
  await expect(card.getByRole("region", { name: "Perception" })).toBeVisible();
  // Two overlapping per-camera intervals stack instead of hiding each other.
  const blur = card.getByRole("button", { name: /^Image blur: none/ });
  await expect(blur).toHaveCount(2);
  const [front, rear] = await Promise.all([blur.nth(0).boundingBox(), blur.nth(1).boundingBox()]);
  expect(front && rear && rear.y > front.y).toBe(true);
});

test("ODD search says so when a dataset has no labels and drops a malformed URL query", async ({ page }) => {
  const searches: URL[] = [];
  await mockAPI(page, searches);
  await page.goto("/odd?dataset=kitscenes-val");
  await expect(page.getByText("No ODD labels for KITScenes Val v3.7.")).toBeVisible();

  await page.goto(
    "/odd?dataset=kitscenes&q=" +
      encodeURIComponent(JSON.stringify({ op: "xor", groups: [{ op: "and", conditions: [{ key: "Road", values: ["x"] }] }] })),
  );
  await expect(page.getByText("of 404 scenes match.")).toBeVisible();
  expect(lastQuery(searches)).toEqual({ op: "and", groups: [] });
});
