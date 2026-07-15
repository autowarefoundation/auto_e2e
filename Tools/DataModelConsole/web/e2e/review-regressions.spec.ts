import { expect, test, type Page, type Route } from "@playwright/test";

function fulfillJSON(route: Route, body: unknown) {
  return route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

function version(version: string) {
  return {
    version,
    total_samples: 2,
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

function shard(name: string) {
  return {
    name,
    key: `review/v2.1/shards/${name}`,
    size_bytes: 512,
    last_modified: "2026-07-15T00:00:00Z",
  };
}

function sample(key: string) {
  return {
    key,
    members: [
      {
        name: `${key}.cam_0.jpg`,
        size_bytes: 128,
        offset: 512,
      },
    ],
  };
}

async function installCatalogRoutes(
  page: Page,
  delayedPath: "/shards" | "/samples",
) {
  let releaseDelayed: (() => void) | undefined;
  const delayed = new Promise<void>((resolve) => {
    releaseDelayed = resolve;
  });
  let delayedRequested = false;
  const imageRequests: string[] = [];

  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    const selectedVersion = url.searchParams.get("version");
    const offset = Number(url.searchParams.get("offset") ?? "0");

    if (path === "/api/v1/datasets/review/versions") {
      return fulfillJSON(route, {
        dataset: "review",
        versions: [version("v2.2"), version("v2.1")],
      });
    }
    if (path === "/api/v1/reasoning-labels/prompt-versions") {
      return fulfillJSON(route, { prompt_versions: [] });
    }
    if (path === "/api/v1/datasets/review/shards") {
      if (
        delayedPath === "/shards" &&
        selectedVersion === "v2.2" &&
        offset > 0
      ) {
        delayedRequested = true;
        await delayed;
        return fulfillJSON(route, {
          dataset: "review",
          shards: [shard("v22-late.tar")],
          page: { limit: 50, offset, total: 2, more: false },
        });
      }
      const name =
        selectedVersion === "v2.1" ? "v21-only.tar" : "v22-first.tar";
      return fulfillJSON(route, {
        dataset: "review",
        shards: [shard(name)],
        page: {
          limit: 50,
          offset: 0,
          total: selectedVersion === "v2.2" ? 2 : 1,
          more: selectedVersion === "v2.2",
        },
      });
    }
    if (
      path ===
      "/api/v1/datasets/review/shards/train-000000.tar/samples"
    ) {
      if (
        delayedPath === "/samples" &&
        selectedVersion === "v2.2" &&
        offset > 0
      ) {
        delayedRequested = true;
        await delayed;
        return fulfillJSON(route, {
          dataset: "review",
          shard: "train-000000.tar",
          samples: [sample("v22-late")],
          page: { limit: 60, offset, total: 2, more: false },
        });
      }
      const key = selectedVersion === "v2.1" ? "v21-only" : "v22-first";
      return fulfillJSON(route, {
        dataset: "review",
        shard: "train-000000.tar",
        samples: [sample(key)],
        page: {
          limit: 60,
          offset: 0,
          total: selectedVersion === "v2.2" ? 2 : 1,
          more: selectedVersion === "v2.2",
        },
      });
    }
    if (path.includes("/image/")) {
      imageRequests.push(route.request().url());
      return route.fulfill({ status: 404 });
    }
    return route.fulfill({ status: 404, body: "not mocked" });
  });

  return {
    release: () => releaseDelayed?.(),
    requested: () => delayedRequested,
    imageRequests: () => [...imageRequests],
  };
}

test("dataset pagination ignores a response from the previously selected version", async ({
  page,
}) => {
  const delayed = await installCatalogRoutes(page, "/shards");
  await page.goto("/datasets/review?version=v2.2");
  await expect(page.getByText("v22-first.tar")).toBeVisible();

  await page.getByRole("button", { name: "Load more" }).click();
  await expect.poll(delayed.requested).toBe(true);
  await page.getByLabel("Dataset version").selectOption("v2.1");
  await expect(page.getByText("v21-only.tar")).toBeVisible();

  delayed.release();
  await page.waitForTimeout(100);
  await expect(page.getByText("v22-late.tar")).toHaveCount(0);
});

test("sample pagination ignores a response from the previously selected version", async ({
  page,
}) => {
  const delayed = await installCatalogRoutes(page, "/samples");
  await page.goto(
    "/datasets/review/shards/train-000000.tar?version=v2.2",
  );
  await expect(page.getByText("v22-first", { exact: true })).toBeVisible();

  await page.getByRole("button", { name: /Load more/ }).click();
  await expect.poll(delayed.requested).toBe(true);
  await page.evaluate(() => {
    window.history.pushState(
      null,
      "",
      "/datasets/review/shards/train-000000.tar?version=v2.1",
    );
    window.dispatchEvent(new PopStateEvent("popstate"));
  });
  await expect(page.getByText("v21-only", { exact: true })).toBeVisible();

  delayed.release();
  await page.waitForTimeout(100);
  await expect(page.getByText("v22-late", { exact: true })).toHaveCount(0);
});

test("sample thumbnails request only their bounded tar member ranges", async ({
  page,
}) => {
  const routes = await installCatalogRoutes(page, "/samples");
  await page.goto(
    "/datasets/review/shards/train-000000.tar?version=v2.1",
  );
  await expect(page.getByText("v21-only", { exact: true })).toBeVisible();
  await expect.poll(() => routes.imageRequests().length).toBe(1);

  const requestURL = new URL(routes.imageRequests()[0]);
  expect(requestURL.searchParams.get("offset")).toBe("512");
  expect(requestURL.searchParams.get("size")).toBe("128");
});
