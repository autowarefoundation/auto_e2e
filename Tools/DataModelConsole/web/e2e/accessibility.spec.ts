import { expect, test, type Page, type Route } from "@playwright/test";

const REASONING_URL =
  "/reasoning-labels?dataset=review&version=v2.1&teacher=teacher-id&prompt_version=prompt-v1";
const PIXEL = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAEAAAAAkAQMAAAADwq7RAAAAIGNIUk0AAHomAACAhAAA+gAAAIDoAAB1MAAA6mAAADqYAAAXcJy6UTwAAAAGUExURTNBVf///753ZLcAAAABYktHRAH/Ai3eAAAAB3RJTUUH6gcPAQU1u04EUwAAAA1JREFUGNNjYBgFlAIAAUQAAS6fR94AAAAldEVYdGRhdGU6Y3JlYXRlADIwMjYtMDctMTVUMDE6MDU6NTMrMDA6MDCLG6dUAAAAJXRFWHRkYXRlOm1vZGlmeQAyMDI2LTA3LTE1VDAxOjA1OjUzKzAwOjAw+kYf6AAAACh0RVh0ZGF0ZTp0aW1lc3RhbXAAMjAyNi0wNy0xNVQwMTowNTo1MyswMDowMK1TPjcAAAAASUVORK5CYII=",
  "base64",
);

function fulfillJSON(route: Route, body: unknown) {
  return route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

function deferred() {
  let release!: () => void;
  const promise = new Promise<void>((resolve) => {
    release = resolve;
  });
  return { promise, release };
}

async function installReasoningRoutes(
  page: Page,
  options: {
    delayStats?: boolean;
    delayScenes?: boolean;
    statsError?: boolean;
  } = {},
) {
  const statsGate = deferred();
  const scenesGate = deferred();

  await page.route("**/api/v1/**", async (route) => {
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
        versions: [
          {
            version: "v2.1",
            total_samples: 3,
            shards: 1,
            episodes: 1,
            num_views: 1,
            has_map: false,
            has_world_model: false,
            has_gps: false,
            size_bytes: 1024,
            has_manifest: true,
          },
        ],
      });
    }
    if (url.pathname === "/api/v1/reasoning-labels/prompt-versions") {
      return fulfillJSON(route, {
        dataset: "review",
        prompt_versions: [
          {
            teacher: "teacher-id",
            teacher_provider: "mock",
            teacher_model: "mock-teacher",
            prompt_version: "prompt-v1",
            count: 3,
          },
        ],
      });
    }
    if (url.pathname === "/api/v1/reasoning-labels/stats-detail") {
      if (options.statsError) {
        return route.fulfill({ status: 500, body: "stats failed" });
      }
      if (options.delayStats) await statsGate.promise;
      return fulfillJSON(route, {
        dataset: "review",
        version: "v2.1",
        teacher: "teacher-id",
        teacher_provider: "mock",
        teacher_model: "mock-teacher",
        prompt_version: "prompt-v1",
        computed_at: "2026-07-16T00:00:00Z",
        cached: true,
        stats: {
          n_labels: 3,
          horizon_count: 15,
          by_field: {
            lateral_response: {
              keep_lane: 2,
              yield: 1,
            },
          },
          confidence_histogram: [
            { bucket: "0.8-0.9", count: 1 },
            { bucket: "0.9-1.0", count: 2 },
          ],
        },
      });
    }
    if (url.pathname === "/api/v1/scenes/search") {
      if (options.delayScenes) await scenesGate.promise;
      return fulfillJSON(route, {
        dataset: "review",
        version: "v2.1",
        teacher: "teacher-id",
        prompt_version: "prompt-v1",
        field: url.searchParams.get("field"),
        value: url.searchParams.get("value"),
        scenes: [],
        total: 0,
        available: 0,
        truncated: false,
      });
    }
    return route.fulfill({ status: 404, body: "not mocked" });
  });

  return {
    releaseStats: statsGate.release,
    releaseScenes: scenesGate.release,
  };
}

async function installPlayerRoutes(page: Page) {
  const samples = Array.from({ length: 20 }, (_, index) => ({
    key: `s${String(index).padStart(8, "0")}`,
    sample_uid: `a11y-v1-e000001-f${String(index).padStart(6, "0")}`,
    split_group_uid: "a11y-episode-1",
    split_bucket: 9,
    episode_id: "episode-1",
    frame_idx: index,
    trip_frame: index,
    members: {
      "cam_0.jpg": { offset: 512 + index * 1024, size: 200 },
    },
    ego_now: [5, 0, 0, 0],
    ego_history: Array.from({ length: 64 }, () => [5, 0, 0, 0]).flat(),
    ego_future: Array.from({ length: 64 }, () => [0, 0]).flat(),
    has_reasoning: false,
  }));

  await page.route("**/api/v1/**", (route) => {
    const url = new URL(route.request().url());

    if (
      url.pathname ===
      "/api/v1/datasets/a11y/shards/train-000000.tar/index"
    ) {
      return fulfillJSON(route, {
        fps: 10,
        version: "v2.1",
        shard: "train-000000.tar",
        samples,
      });
    }
    if (url.pathname === "/api/v1/datasets/a11y/shards") {
      return fulfillJSON(route, {
        dataset: "a11y",
        shards: [
          {
            name: "train-000000.tar",
            key: "a11y/v2.1/shards/train-000000.tar",
            size_bytes: 1000,
            last_modified: "2026-07-16T00:00:00Z",
          },
        ],
        page: { limit: 200, offset: 0, total: 1, more: false },
      });
    }
    if (url.pathname.endsWith("/overlay-models")) {
      return fulfillJSON(route, {
        dataset: "a11y",
        version: "v2.1",
        shard: "train-000000.tar",
        models: [],
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

test("skip link, current navigation, and mobile dialog preserve keyboard focus", async ({
  page,
}) => {
  await installReasoningRoutes(page);
  await page.setViewportSize({ width: 320, height: 800 });
  await page.goto(REASONING_URL);
  await expect(page.getByRole("heading", { name: /Reasoning Labels/ })).toBeVisible();

  const skipLink = page.getByRole("link", { name: "Skip to main content" });
  await page.keyboard.press("Tab");
  await expect(skipLink).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.locator("#main-content")).toBeFocused();

  const trigger = page.getByRole("button", { name: "Open navigation" });
  await trigger.click();
  const dialog = page.getByRole("dialog", { name: "DataModelConsole" });
  await expect(dialog).toHaveAttribute("aria-modal", "true");
  await expect(
    dialog.getByRole("link", { name: "Reasoning Labels" }),
  ).toHaveAttribute("aria-current", "page");
  await expect(
    dialog.getByRole("button", { name: "Close navigation" }),
  ).toBeFocused();

  for (let index = 0; index < 10; index++) {
    await page.keyboard.press("Tab");
    await expect
      .poll(() =>
        dialog.evaluate((element) =>
          element.contains(document.activeElement),
        ),
      )
      .toBe(true);
  }

  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  await expect(trigger).toBeFocused();

  await page.setViewportSize({ width: 1024, height: 800 });
  await expect(
    page.getByRole("navigation", { name: "Primary navigation" }).getByRole(
      "link",
      { name: "Reasoning Labels" },
    ),
  ).toHaveAttribute("aria-current", "page");
});

test("reasoning chart and scene dialog remain accessible at 320px", async ({
  page,
}) => {
  const routes = await installReasoningRoutes(page, {
    delayStats: true,
    delayScenes: true,
  });
  await page.setViewportSize({ width: 320, height: 800 });
  await page.goto(REASONING_URL);

  await expect(
    page.getByRole("status").filter({ hasText: "Computing statistics" }),
  ).toBeVisible();
  routes.releaseStats();

  const chartButton = page.locator('button[title^="keep_lane:"]');
  await expect(chartButton).toBeVisible();
  const chart = chartButton.locator("xpath=ancestor::*[@data-slot='card']");
  const geometry = await chart.evaluate((element) => {
    const card = element.getBoundingClientRect();
    const clipped = Array.from(element.querySelectorAll("button")).some(
      (button) => {
        const rect = button.getBoundingClientRect();
        return rect.left < card.left - 1 || rect.right > card.right + 1;
      },
    );
    return {
      clientWidth: element.clientWidth,
      scrollWidth: element.scrollWidth,
      clipped,
    };
  });
  expect(geometry.scrollWidth).toBeLessThanOrEqual(geometry.clientWidth + 1);
  expect(geometry.clipped).toBe(false);

  await chartButton.click();
  const dialog = page.getByRole("dialog", {
    name: /lateral_response.*keep_lane/,
  });
  await expect(dialog).toHaveAttribute("aria-modal", "true");
  await expect(dialog.getByRole("button", { name: "Close" })).toBeFocused();
  await expect(
    dialog.getByRole("status", { name: "Loading matching scenes" }),
  ).toBeVisible();

  await page.keyboard.press("Tab");
  await expect
    .poll(() =>
      dialog.evaluate((element) => element.contains(document.activeElement)),
    )
    .toBe(true);
  routes.releaseScenes();
  await expect(
    dialog.getByRole("status").filter({ hasText: "No matching scenes." }),
  ).toBeVisible();

  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  await expect(chartButton).toBeFocused();
});

test("errors are announced as alerts", async ({ page }) => {
  await installReasoningRoutes(page, { statsError: true });
  await page.goto(REASONING_URL);

  await expect(
    page.getByRole("alert").filter({ hasText: "Failed to load data." }),
  ).toBeVisible();
});

test("Space does not toggle playback from interactive or editable focus", async ({
  page,
}) => {
  await installPlayerRoutes(page);
  await page.goto(
    "/scenes/a11y/train-000000.tar/0?version=v2.1",
    { waitUntil: "domcontentloaded" },
  );
  await expect(
    page.locator('[aria-label^="Episode player"]'),
  ).toBeVisible();

  const shortcuts = page.getByRole("button", {
    name: "Keyboard shortcuts",
  });
  await shortcuts.focus();
  await page.keyboard.press("Space");
  await expect(page.getByText("play / pause", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Play", exact: true })).toBeVisible();

  await page.getByRole("button", { name: /close \(esc\)/i }).click();
  await page.evaluate(() => {
    const editor = document.createElement("div");
    editor.contentEditable = "true";
    editor.dataset.testid = "editor";
    const child = document.createElement("span");
    child.tabIndex = 0;
    child.dataset.testid = "editor-child";
    child.textContent = "Editable";
    editor.append(child);
    document.body.append(editor);
  });
  await page.getByTestId("editor-child").focus();
  await page.keyboard.press("Space");
  await expect(page.getByRole("button", { name: "Play", exact: true })).toBeVisible();
});
