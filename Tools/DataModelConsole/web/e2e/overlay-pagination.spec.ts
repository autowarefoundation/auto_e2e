import {
  expect,
  test,
  type Page as PlaywrightPage,
  type Route,
} from "@playwright/test";

const SHARD = "train-000000.tar";
const SCENE_URL = `/scenes/catalog/${SHARD}/0?version=v2.1`;
const PAGE_TOKEN = "f".repeat(64);
const MAX_DATA_PAGES = 20;
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

function model(id: string, version: number) {
  return {
    model_artifact_id: id,
    registered_model_name: "auto-e2e-driving-policy",
    model_version: version,
    run_id: `run-${id}`,
    model_name: `Model ${id}`,
    eval_ade: 1,
    eval_fde: 2,
    val_fraction: 0.2,
    overlay_schema: "v1",
    sample_count: 1,
  };
}

async function installSceneRoutes(
  page: PlaywrightPage,
  overlayModels: (route: Route, url: URL) => Promise<void>,
) {
  await page.route("**/api/v1/**", (route) => {
    const url = new URL(route.request().url());
    if (
      url.pathname ===
      `/api/v1/datasets/catalog/shards/${SHARD}/overlay-models`
    ) {
      return overlayModels(route, url);
    }
    if (
      url.pathname === `/api/v1/datasets/catalog/shards/${SHARD}/index`
    ) {
      return fulfillJSON(route, {
        fps: 10,
        version: "v2.1",
        shard: SHARD,
        samples: [
          {
            key: "s00000000",
            sample_uid: "catalog-v1-e000001-f000000",
            split_group_uid: "catalog-episode-1",
            split_bucket: 9,
            episode_id: "episode-1",
            frame_idx: 0,
            trip_frame: 0,
            members: {
              "cam_0.jpg": { offset: 512, size: 200 },
            },
            ego_now: [5, 0, 0, 0],
            ego_history: Array.from({ length: 64 }, () => [5, 0, 0, 0]).flat(),
            ego_future: Array.from({ length: 64 }, () => [0, 0]).flat(),
            has_reasoning: false,
          },
        ],
      });
    }
    if (url.pathname === "/api/v1/datasets/catalog/shards") {
      return fulfillJSON(route, {
        dataset: "catalog",
        shards: [
          {
            name: SHARD,
            key: `catalog/v2.1/shards/${SHARD}`,
            size_bytes: 1000,
            last_modified: "2026-07-16T00:00:00Z",
          },
        ],
        page: { limit: 1000, offset: 0, total: 1, more: false },
      });
    }
    if (url.pathname.endsWith("/rig-projection")) {
      return route.fulfill({ status: 404, body: "not published" });
    }
    if (url.pathname.endsWith("/blob")) {
      return route.fulfill({ status: 403, body: "private range" });
    }
    if (url.pathname.includes("/image/cam_0")) {
      return route.fulfill({
        status: 200,
        contentType: "image/png",
        body: PIXEL,
      });
    }
    return route.fulfill({ status: 404, body: "not mocked" });
  });
}

test("overlay catalog merges, deduplicates, and globally sorts two pages", async ({
  page,
}) => {
  const requests: URL[] = [];
  await installSceneRoutes(page, (route, url) => {
    requests.push(url);
    const token = url.searchParams.get("page_token");
    if (token === PAGE_TOKEN) {
      return fulfillJSON(route, {
        dataset: "catalog",
        version: "v2.1",
        shard: SHARD,
        models: [
          model("model-c", 9),
          model("model-a", 7),
          model("model-d", 4),
        ],
      });
    }
    return fulfillJSON(route, {
      dataset: "catalog",
      version: "v2.1",
      shard: SHARD,
      models: [model("model-b", 7), model("model-d", 4)],
      next_page_token: PAGE_TOKEN,
    });
  });

  await page.goto(SCENE_URL);

  const select = page.locator("#trajectory-model");
  await expect(select.locator("option")).toHaveCount(4);
  await expect(select).toHaveValue("model-c");
  expect(
    await select.locator("option").evaluateAll((options) =>
      options.map((option) => (option as HTMLOptionElement).value),
    ),
  ).toEqual(["model-c", "model-a", "model-b", "model-d"]);

  expect(requests).toHaveLength(2);
  expect(requests.map((url) => url.searchParams.get("limit"))).toEqual([
    "100",
    "100",
  ]);
  expect(requests.map((url) => url.searchParams.get("page_token"))).toEqual([
    null,
    PAGE_TOKEN,
  ]);
  expect(requests[1].toString()).toContain(
    new URLSearchParams({ page_token: PAGE_TOKEN }).toString(),
  );
});

test("overlay catalog rejects a token cycle without another request", async ({
  page,
}) => {
  const tokens: Array<string | null> = [];
  await installSceneRoutes(page, (route, url) => {
    const token = url.searchParams.get("page_token");
    tokens.push(token);
    return fulfillJSON(route, {
      dataset: "catalog",
      version: "v2.1",
      shard: SHARD,
      models: [model(token ? "model-b" : "model-a", 1)],
      next_page_token: "cycle-token",
    });
  });

  await page.goto(SCENE_URL);
  await expect(
    page.getByText("Trajectory model catalog unavailable"),
  ).toBeVisible();
  expect(tokens).toEqual([null, "cycle-token"]);

  await page.waitForTimeout(200);
  expect(tokens).toEqual([null, "cycle-token"]);
});

test("overlay catalog permits one empty terminal probe after 20 pages", async ({
  page,
}) => {
  const tokens: Array<string | null> = [];
  const tokenForPage = (pageNumber: number) =>
    pageNumber.toString(16).padStart(64, "0");

  await installSceneRoutes(page, (route, url) => {
    const token = url.searchParams.get("page_token");
    tokens.push(token);
    const pageNumber = token === null ? 0 : Number.parseInt(token, 16);
    if (pageNumber === MAX_DATA_PAGES) {
      return fulfillJSON(route, {
        dataset: "catalog",
        version: "v2.1",
        shard: SHARD,
        models: [],
      });
    }
    return fulfillJSON(route, {
      dataset: "catalog",
      version: "v2.1",
      shard: SHARD,
      models: [model(tokenForPage(pageNumber + 1), pageNumber + 1)],
      next_page_token: tokenForPage(pageNumber + 1),
    });
  });

  await page.goto(SCENE_URL);

  const select = page.locator("#trajectory-model");
  await expect(select.locator("option")).toHaveCount(MAX_DATA_PAGES);
  await expect(select).toHaveValue(tokenForPage(MAX_DATA_PAGES));
  expect(tokens).toHaveLength(MAX_DATA_PAGES + 1);
  expect(tokens[0]).toBeNull();
  expect(tokens.at(-1)).toBe(tokenForPage(MAX_DATA_PAGES));
});
