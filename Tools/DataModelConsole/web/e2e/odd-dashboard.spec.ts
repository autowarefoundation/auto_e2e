import { expect, test, type Page, type Route } from "@playwright/test";

const PIXEL = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAEAAAAAkAQMAAAADwq7RAAAAIGNIUk0AAHomAACAhAAA+gAAAIDoAAB1MAAA6mAAADqYAAAXcJy6UTwAAAAGUExURTNBVf///753ZLcAAAABYktHRAH/Ai3eAAAAB3RJTUUH6gcPAQU1u04EUwAAAA1JREFUGNNjYBgFlAIAAUQAAS6fR94AAAAASUVORK5CYII=",
  "base64",
);

function fulfillJSON(route: Route, body: unknown) {
  return route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

const ontology = {
  schema_version: "odd_ontology_registry_v1",
  ontology_version: "odd_ontology_v1",
  ontology_sha256: "a".repeat(64),
  statuses: ["valid", "unavailable", "not_observable", "ambiguous"],
  sources: ["map_route", "gnss_ins", "vlm", "image_qc", "fusion"],
  labels: [
    {
      key: "odd.road.context",
      namespace: "odd",
      display_name: "Road Context",
      description: "Functional character of the road surroundings.",
      cardinality: "single",
      values: [{ value: "urban" }, { value: "suburban" }],
      primary_sources: ["map_route", "vlm"],
      backends: ["deterministic", "openai_compatible"],
      subject: "scene",
      temporal_scope: "interval",
      quality_tier: "experimental",
    },
  ],
};

const statistics = {
  schema_version: "odd_statistics_v1",
  labelset_id: "oddls-test",
  scene_count: 2,
  scene_duration_ns: 200,
  keys: [
    {
      key: "odd.road.context",
      namespace: "odd",
      valid_scene_count: 2,
      eligible_scene_count: 2,
      observable_scene_coverage: 1,
      eligible_duration_ns: 200,
      valid_duration_ns: 200,
      status_scene_counts: { valid: 2 },
      status_duration_ns: { valid: 200 },
      source_scene_counts: { vlm: 2 },
      values: [
        {
          value: "urban",
          scene_count: 1,
          scene_ratio: 0.5,
          duration_ns: 100,
          duration_ratio: 0.5,
        },
        {
          value: "suburban",
          scene_count: 0,
          scene_ratio: 0,
          duration_ns: 0,
          duration_ratio: 0,
        },
      ],
    },
  ],
};

async function installODDDashboardRoutes(page: Page) {
  await page.route("**/api/v1/odd/**", (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/api/v1/odd/ontology") {
      return fulfillJSON(route, ontology);
    }
    if (url.pathname === "/api/v1/odd/statistics") {
      return fulfillJSON(route, statistics);
    }
    if (url.pathname === "/api/v1/odd/scenes/search") {
      return fulfillJSON(route, {
        dataset: "kitscenes",
        version: "v3.0",
        labelset_id: "oddls-test",
        scenes: [
          {
            scene_uid: "scene-1",
            shard_name: "scene-1.tar",
            start_timestamp_ns: 0,
            end_timestamp_ns: 100,
            distance_m: 42,
            observations: [],
          },
        ],
        total: 1,
        limit: 100,
        offset: 0,
        more: false,
      });
    }
    return route.fulfill({ status: 404, body: "not mocked" });
  });
}

test("ODD Dashboard exposes composition, search, and zero-count ontology", async ({
  page,
}) => {
  await installODDDashboardRoutes(page);
  await page.goto("/odd");

  await expect(page.getByRole("heading", { name: "ODD Dashboard" })).toBeVisible();
  await expect(page.getByText("Observable 100%")).toBeVisible();
  await page.getByRole("button", { name: /^urban / }).click();
  await expect(page.getByText("1 matching scenes")).toBeVisible();
  await expect(page.getByRole("link", { name: /scene-1/ })).toBeVisible();

  await page.getByRole("button", { name: "Ontology" }).click();
  await page.getByText("odd.road.context", { exact: true }).click();
  await expect(page.getByText("suburban · 0")).toBeVisible();
});

test("ODD Dashboard remains horizontally contained on mobile", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await installODDDashboardRoutes(page);
  await page.goto("/odd");
  await expect(page.getByRole("heading", { name: "ODD Dashboard" })).toBeVisible();

  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth - window.innerWidth,
    ),
  ).toBe(0);
});

test("Scene places current ODD labels directly below Reasoning", async ({
  page,
}) => {
  await page.route("**/api/v1/**", (route) => {
    const url = new URL(route.request().url());
    if (
      url.pathname ===
      "/api/v1/datasets/kitscenes/shards/scene-1.tar/index"
    ) {
      return fulfillJSON(route, {
        fps: 10,
        version: "v3.0",
        shard: "scene-1.tar",
        samples: [
          {
            key: "sample-1",
            sample_uid: "sample-1",
            split_group_uid: "scene-1",
            split_bucket: 1,
            episode_id: "scene-1",
            frame_idx: 0,
            trip_frame: 0,
            members: { "cam_0.jpg": { offset: 512, size: 200 } },
            ego_now: [5, 0, 0, 0],
            ego_history: Array.from({ length: 64 }, () => [5, 0, 0, 0]).flat(),
            ego_future: Array.from({ length: 64 }, () => [0, 0]).flat(),
            has_reasoning: false,
          },
        ],
      });
    }
    if (url.pathname === "/api/v1/datasets/kitscenes/shards") {
      return fulfillJSON(route, {
        dataset: "kitscenes",
        shards: [
          {
            name: "scene-1.tar",
            key: "kitscenes/v3.0/shards/scene-1.tar",
            size_bytes: 1000,
            last_modified: "2026-07-28T00:00:00Z",
          },
        ],
        page: { limit: 1000, offset: 0, total: 1, more: false },
      });
    }
    if (url.pathname === "/api/v1/odd/scenes/scene-1") {
      return fulfillJSON(route, {
        scene_uid: "scene-1",
        dataset_name: "kitscenes",
        dataset_version: "v3.0",
        start_timestamp_ns: 0,
        end_timestamp_ns: 1_000_000_000,
        distance_m: 42,
        observations: [
          {
            observation_uid: "oddobs-1",
            scene_uid: "scene-1",
            key: "odd.road.context",
            status: "valid",
            values: ["urban"],
            confidence: 0.9,
            source: "vlm",
            start_timestamp_ns: 0,
            end_timestamp_ns: 1_000_000_000,
            measurements: {},
            provenance: {},
          },
        ],
      });
    }
    if (url.pathname.includes("/image/cam_0")) {
      return route.fulfill({
        status: 200,
        contentType: "image/png",
        body: PIXEL,
      });
    }
    if (url.pathname.endsWith("/overlay-models")) {
      return fulfillJSON(route, {
        dataset: "kitscenes",
        version: "v3.0",
        shard: "scene-1.tar",
        models: [],
      });
    }
    return route.fulfill({ status: 404, body: "not published" });
  });

  await page.goto("/scenes/kitscenes/scene-1.tar/0?version=v3.0");
  await expect(page.getByText("urban", { exact: true })).toBeVisible();

  const ordered = await page
    .locator('[aria-label^="Episode player"]')
    .evaluate((player) => {
      const reasoning = Array.from(player.querySelectorAll("p")).find(
        (element) => element.textContent?.trim() === "Reasoning label",
      );
      const odd = Array.from(player.querySelectorAll("p")).find(
        (element) => element.textContent?.trim() === "ODD Labels",
      );
      return Boolean(
        reasoning &&
          odd &&
          reasoning.compareDocumentPosition(odd) &
            Node.DOCUMENT_POSITION_FOLLOWING,
      );
    });
  expect(ordered).toBe(true);
});
