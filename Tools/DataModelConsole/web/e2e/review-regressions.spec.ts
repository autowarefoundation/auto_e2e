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

test("dataset shards wait for version resolution and version history is navigable", async ({
  page,
}) => {
  let releaseVersions!: () => void;
  const versionsGate = new Promise<void>((resolve) => {
    releaseVersions = resolve;
  });
  const shardVersions: Array<string | null> = [];

  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/api/v1/datasets/review/versions") {
      await versionsGate;
      return fulfillJSON(route, {
        dataset: "review",
        versions: [version("v2.2"), version("v2.1")],
      });
    }
    if (url.pathname === "/api/v1/datasets/review/shards") {
      const selectedVersion = url.searchParams.get("version");
      shardVersions.push(selectedVersion);
      return fulfillJSON(route, {
        dataset: "review",
        shards: [shard(`${selectedVersion}.tar`)],
        page: { limit: 50, offset: 0, total: 1, more: false },
      });
    }
    if (url.pathname === "/api/v1/reasoning-labels/prompt-versions") {
      return fulfillJSON(route, { prompt_versions: [] });
    }
    return route.fulfill({ status: 404, body: "not mocked" });
  });

  await page.goto("/datasets/review?version=v2.1");
  await page.waitForTimeout(100);
  expect(shardVersions).toEqual([]);

  releaseVersions();
  await expect(page.getByText("v2.1.tar")).toBeVisible();
  expect(shardVersions).toEqual(["v2.1"]);

  await page.getByLabel("Dataset version").selectOption("v2.2");
  await expect(page.getByText("v2.2.tar")).toBeVisible();
  await expect(page).toHaveURL(/version=v2\.2$/);

  await page.goBack();
  await expect(page).toHaveURL(/version=v2\.1$/);
  await expect(page.getByText("v2.1.tar")).toBeVisible();
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

test("scene locator paginates through the complete shard publication", async ({
  page,
}) => {
  const allShards = Array.from({ length: 533 }, (_, index) =>
    shard(`train-${String(index).padStart(6, "0")}.tar`),
  );
  await page.route("**/api/v1/**", (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/api/v1/datasets") {
      return fulfillJSON(route, {
        datasets: [
          {
            name: "l2d",
            version: "v2.1",
            prefix: "l2d/v2.1/shards/",
          },
        ],
      });
    }
    if (url.pathname === "/api/v1/datasets/l2d/versions") {
      return fulfillJSON(route, {
        dataset: "l2d",
        versions: [version("v2.1")],
      });
    }
    if (url.pathname === "/api/v1/datasets/l2d/shards") {
      const offset = Number(url.searchParams.get("offset") ?? "0");
      const pageSize = 200;
      const items = allShards.slice(offset, offset + pageSize);
      return fulfillJSON(route, {
        dataset: "l2d",
        shards: items,
        page: {
          limit: pageSize,
          offset,
          total: allShards.length,
          more: offset + items.length < allShards.length,
        },
      });
    }
    return route.fulfill({ status: 404, body: "not mocked" });
  });

  await page.goto("/scenes");
  const options = page.locator("#scene-shard-options option");
  await expect(options).toHaveCount(533);
  await expect(
    page.locator(
      '#scene-shard-options option[value="train-000532.tar"]',
    ),
  ).toHaveCount(1);
});

test("reasoning label reads retain the selected dataset version", async ({
  page,
}) => {
  const teacher = "cHJvdmlkZXIAbW9kZWw";
  let labelRequest = "";
  await page.route("**/api/v1/**", (route) => {
    const url = new URL(route.request().url());
    if (
      url.pathname ===
      "/api/v1/datasets/review/shards/train-000000.tar/samples/sample"
    ) {
      return fulfillJSON(route, {
        key: "sample",
        episode_id: "episode",
        frame_idx: 0,
        meta: {},
        cameras: [],
        ego_history: [],
        ego_future: [],
      });
    }
    if (
      url.pathname ===
      "/api/v1/datasets/review/shards/train-000000.tar/index"
    ) {
      return fulfillJSON(route, {
        fps: 10,
        version: "v2.0",
        shard: "train-000000.tar",
        samples: [],
      });
    }
    if (
      url.pathname === "/api/v1/reasoning-labels/review/sample"
    ) {
      labelRequest = route.request().url();
      return route.fulfill({ status: 404, body: "no label" });
    }
    return route.fulfill({ status: 404, body: "not mocked" });
  });

  await page.goto(
    `/datasets/review/shards/train-000000.tar/samples/sample?version=v2.0&teacher=${teacher}&prompt_version=p1`,
  );
  await expect(page.getByText("sample", { exact: true }).first()).toBeVisible();
  await expect.poll(() => labelRequest).not.toBe("");

  const query = new URL(labelRequest).searchParams;
  expect(query.get("prompt_version")).toBe("p1");
  expect(query.get("version")).toBe("v2.0");
  expect(query.get("teacher")).toBe(teacher);
});

test("reasoning prompt discovery follows the selected dataset version", async ({
  page,
}) => {
  const teacherOld = "cHJvdmlkZXIAbW9kZWwtb2xk";
  const teacherNew = "cHJvdmlkZXIAbW9kZWwtbmV3";
  const promptRequests: Array<string | null> = [];
  const statsTeachers: Array<string | null> = [];
  await page.route("**/api/v1/**", (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/api/v1/datasets") {
      return fulfillJSON(route, {
        datasets: [
          {
            name: "review",
            version: "v2.1",
            prefix: "review/v2.1/shards/",
          },
        ],
      });
    }
    if (url.pathname === "/api/v1/datasets/review/versions") {
      return fulfillJSON(route, {
        dataset: "review",
        versions: [version("v2.1"), version("v2.0")],
      });
    }
    if (url.pathname === "/api/v1/reasoning-labels/prompt-versions") {
      const selectedVersion = url.searchParams.get("version");
      promptRequests.push(selectedVersion);
      const prompt =
        selectedVersion === "v2.0" ? "prompt-old" : "prompt-new";
      const teacher =
        selectedVersion === "v2.0" ? teacherOld : teacherNew;
      return fulfillJSON(route, {
        dataset: "review",
        prompt_versions: [
          {
            teacher,
            teacher_provider: "provider",
            teacher_model:
              selectedVersion === "v2.0" ? "model-old" : "model-new",
            prompt_version: prompt,
            count: 2,
          },
        ],
      });
    }
    if (url.pathname === "/api/v1/reasoning-labels/stats-detail") {
      const selectedVersion = url.searchParams.get("version") ?? "";
      statsTeachers.push(url.searchParams.get("teacher"));
      return fulfillJSON(route, {
        dataset: "review",
        version: selectedVersion,
        teacher: url.searchParams.get("teacher"),
        teacher_provider: "provider",
        teacher_model:
          selectedVersion === "v2.0" ? "model-old" : "model-new",
        prompt_version: url.searchParams.get("prompt_version"),
        computed_at: "2026-07-15T00:00:00Z",
        cached: true,
        stats: {
          n_labels: 2,
          horizon_count: 10,
          by_field: {},
          confidence_histogram: [],
        },
      });
    }
    return route.fulfill({ status: 404, body: "not mocked" });
  });

  await page.goto(
    `/reasoning-labels?dataset=review&version=v2.0&teacher=${teacherOld}&prompt_version=prompt-old`,
  );
  await expect(page.locator("#rl-prompt")).toHaveValue(
    JSON.stringify([teacherOld, "prompt-old"]),
  );
  expect(promptRequests).toContain("v2.0");
  await expect.poll(() => statsTeachers).toContain(teacherOld);

  await page.locator("#rl-version").selectOption("v2.1");
  await expect(page.locator("#rl-prompt")).toHaveValue(
    JSON.stringify([teacherNew, "prompt-new"]),
  );
  expect(promptRequests).toContain("v2.1");
  await expect.poll(() => statsTeachers).toContain(teacherNew);
});
