import { expect, test, type Route } from "@playwright/test";

function fulfillJSON(route: Route, body: unknown) {
  return route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

function shard(name: string, dataset: string, version: string) {
  return {
    name,
    key: `${dataset}/${version}/shards/${name}`,
    size_bytes: 512,
    last_modified: "2026-07-15T00:00:00Z",
  };
}

function datasetVersion(version: string) {
  return {
    version,
    total_samples: 10,
    shards: 1,
    episodes: 1,
    num_views: 1,
    has_map: false,
    has_world_model: false,
    has_gps: false,
    size_bytes: 1024,
    has_manifest: true,
  };
}

test("Scene locator pins KITScenes version and ignores stale shards", async ({
  page,
}) => {
  const kitscenesShards = Array.from({ length: 533 }, (_, index) =>
    shard(
      `train-${String(index).padStart(6, "0")}.tar`,
      "kitscenes",
      "v2.2",
    ),
  );
  let l2dRequested = false;
  const shardRequests: Array<{ dataset: string; version: string | null }> = [];

  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/api/v1/datasets") {
      return fulfillJSON(route, {
        datasets: [
          {
            name: "l2d",
            version: "v2.0",
            prefix: "l2d/v2.0/shards/",
          },
          {
            name: "kitscenes",
            version: "v2.2",
            prefix: "kitscenes/v2.2/shards/",
          },
        ],
      });
    }
    if (url.pathname === "/api/v1/datasets/l2d/versions") {
      return fulfillJSON(route, {
        dataset: "l2d",
        versions: [datasetVersion("v2.0")],
      });
    }
    if (url.pathname === "/api/v1/datasets/kitscenes/versions") {
      return fulfillJSON(route, {
        dataset: "kitscenes",
        versions: [
          datasetVersion("v2.2"),
          datasetVersion("v2.1"),
        ],
      });
    }

    const match = url.pathname.match(
      /^\/api\/v1\/datasets\/([^/]+)\/shards$/,
    );
    if (match) {
      const dataset = decodeURIComponent(match[1]);
      const version = url.searchParams.get("version");
      shardRequests.push({ dataset, version });
      if (dataset === "l2d") {
        l2dRequested = true;
        return fulfillJSON(route, {
          dataset,
          shards: [shard("l2d-stale.tar", "l2d", "v2.0")],
          page: { limit: 1000, offset: 0, total: 1, more: false },
        });
      }
      return fulfillJSON(route, {
        dataset,
        shards: kitscenesShards,
        page: {
          limit: 1000,
          offset: 0,
          total: kitscenesShards.length,
          more: false,
        },
      });
    }

    return route.fulfill({ status: 404, body: "not mocked" });
  });

  await page.goto("/scenes");
  await expect(page.getByLabel("Dataset", { exact: true })).toHaveValue(
    "kitscenes",
  );
  await expect(page.getByLabel("Dataset version")).toHaveValue("v2.2");

  const options = page.locator("#scene-shard-options option");
  await expect(options).toHaveCount(533);
  expect(shardRequests).toContainEqual({
    dataset: "kitscenes",
    version: "v2.2",
  });
  await page.waitForTimeout(100);
  expect(l2dRequested).toBe(false);
  await expect(
    page.locator('#scene-shard-options option[value="l2d-stale.tar"]'),
  ).toHaveCount(0);

  await page.getByLabel("Shard").fill("not-published.tar");
  await expect(
    page.getByRole("button", { name: "Open", exact: true }),
  ).toBeDisabled();

  await page.getByLabel("Shard").fill("train-000532.tar");
  await page.getByLabel("Frame index").fill("64");
  await page.reload();
  await expect(page.getByLabel("Dataset", { exact: true })).toHaveValue(
    "kitscenes",
  );
  await expect(page.getByLabel("Dataset version")).toHaveValue("v2.2");
  await expect(page.getByLabel("Shard")).toHaveValue("train-000532.tar");
  await expect(page.getByLabel("Frame index")).toHaveValue("64");
  await page.getByRole("button", { name: "Open", exact: true }).click();
  await expect(page).toHaveURL(
    /\/scenes\/kitscenes\/train-000532\.tar\/64\?version=v2\.2$/,
  );
});

test("Scene locator restores a historical version with Back and Forward", async ({
  page,
}) => {
  await page.route("**/api/v1/**", (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/api/v1/datasets") {
      return fulfillJSON(route, {
        datasets: [
          {
            name: "kitscenes",
            version: "v2.2",
            prefix: "kitscenes/v2.2/shards/",
          },
        ],
      });
    }
    if (url.pathname === "/api/v1/datasets/kitscenes/versions") {
      return fulfillJSON(route, {
        dataset: "kitscenes",
        versions: [
          datasetVersion("v2.2"),
          datasetVersion("v2.1"),
        ],
      });
    }
    if (url.pathname === "/api/v1/datasets/kitscenes/shards") {
      const version = url.searchParams.get("version") ?? "";
      return fulfillJSON(route, {
        dataset: "kitscenes",
        shards: [shard(`${version}.tar`, "kitscenes", version)],
        page: { limit: 1000, offset: 0, total: 1, more: false },
      });
    }
    return route.fulfill({ status: 404, body: "not mocked" });
  });

  await page.goto("/scenes?dataset=kitscenes&version=v2.1");
  await expect(page.getByLabel("Dataset version")).toHaveValue("v2.1");
  await expect(
    page.locator('#scene-shard-options option[value="v2.1.tar"]'),
  ).toHaveCount(1);

  await page.getByLabel("Dataset version").selectOption("v2.2");
  await expect(page).toHaveURL(/version=v2\.2/);
  await page.goBack();
  await expect(page.getByLabel("Dataset version")).toHaveValue("v2.1");
  await page.goForward();
  await expect(page.getByLabel("Dataset version")).toHaveValue("v2.2");
});
