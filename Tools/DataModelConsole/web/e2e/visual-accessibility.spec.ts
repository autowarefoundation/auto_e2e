import { expect, test, type Page, type Route } from "@playwright/test";

const PIXEL = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAEAAAAAkAQMAAAADwq7RAAAAIGNIUk0AAHomAACAhAAA+gAAAIDoAAB1MAAA6mAAADqYAAAXcJy6UTwAAAAGUExURTNBVf///753ZLcAAAABYktHRAH/Ai3eAAAAB3RJTUUH6gcPAQU1u04EUwAAAA1JREFUGNNjYBgFlAIAAUQAAS6fR94AAAAldEVYdGRhdGU6Y3JlYXRlADIwMjYtMDctMTVUMDE6MDU6NTMrMDA6MDCLG6dUAAAAJXRFWHRkYXRlOm1vZGlmeQAyMDI2LTA3LTE1VDAxOjA1OjUzKzAwOjAw+kYf6AAAACh0RVh0ZGF0ZTp0aW1lc3RhbXAAMjAyNi0wNy0xNVQwMTowNTo1MyswMDowMK1TPjcAAAAASUVORK5CYII=",
  "base64",
);
const CAMERAS = Array.from({ length: 7 }, (_, index) => `cam_${index}`);

function fulfillJSON(route: Route, body: unknown) {
  return route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

function episodePath(): Buffer {
  const body = Buffer.alloc(2 * 32);
  for (let index = 0; index < 2; index++) {
    const offset = index * 32;
    body.writeDoubleLE(49, offset);
    body.writeDoubleLE(11 + index * 0.0001, offset + 8);
    body.writeDoubleLE(90, offset + 16);
    body.writeDoubleLE(1_700_000_000 + index * 0.1, offset + 24);
  }
  return body;
}

function egoHistory(speed: number): number[] {
  return Array.from({ length: 64 }, () => [speed, 0, 0, 0]).flat();
}

function egoFuture(): number[] {
  return Array.from({ length: 64 }, () => [0.05, 0]).flat();
}

async function mockConsole(page: Page) {
  await page.route("https://tile.openstreetmap.org/**", (route) =>
    route.fulfill({ status: 200, contentType: "image/png", body: PIXEL }),
  );
  await page.route("**/api/v1/**", (route) => {
    const path = new URL(route.request().url()).pathname;

    if (path === "/api/v1/datasets") {
      return fulfillJSON(route, {
        datasets: [
          {
            name: "kitscenes",
            version: "v2.1",
            prefix: "kitscenes/v2.1/shards/",
          },
        ],
      });
    }
    if (path === "/api/v1/datasets/kitscenes/versions") {
      return fulfillJSON(route, {
        dataset: "kitscenes",
        versions: [
          {
            version: "v2.1",
            total_samples: 2,
            shards: 1,
            episodes: 1,
            num_views: 7,
            has_map: true,
            has_world_model: false,
            has_gps: true,
            size_bytes: 1024,
            has_manifest: true,
          },
        ],
      });
    }
    if (path === "/api/v1/datasets/kitscenes/geo-stats") {
      return fulfillJSON(route, {
        dataset: "kitscenes",
        version: "v2.1",
        summary: {
          bbox: [10.99, 48.99, 11.02, 49.02],
          episode_count: 5,
          path_point_count: 400,
          sample_pose_count: 120,
          privacy: {
            k_anonymity: 5,
            endpoint_exclusion_frames: 10,
            heatmap_grid_degrees: 0.01,
          },
        },
        heatmap_url:
          "/api/v1/datasets/kitscenes/geo/heatmap?version=v2.1",
        n_samples: 120,
      });
    }
    if (path === "/api/v1/datasets/kitscenes/geo/heatmap") {
      return fulfillJSON(route, {
        type: "FeatureCollection",
        features: [
          {
            type: "Feature",
            geometry: { type: "Point", coordinates: [11, 49] },
            properties: { sample_count: 80, episode_count: 5 },
          },
          {
            type: "Feature",
            geometry: { type: "Point", coordinates: [11.01, 49.01] },
            properties: { sample_count: 40, episode_count: 5 },
          },
        ],
      });
    }
    if (path === "/api/v1/datasets/kitscenes/shards") {
      return fulfillJSON(route, {
        dataset: "kitscenes",
        shards: [
          {
            name: "train-000000.tar",
            key: "kitscenes/v2.1/shards/train-000000.tar",
            size_bytes: 1024,
            last_modified: "2026-07-16T00:00:00Z",
          },
        ],
        page: { limit: 1000, offset: 0, total: 1, more: false },
      });
    }
    if (path.endsWith("/index")) {
      return fulfillJSON(route, {
        fps: 10,
        version: "v2.1",
        shard: "train-000000.tar",
        samples: [0, 1].map((frame) => ({
          key: `s${String(frame).padStart(8, "0")}`,
          sample_uid: `kitscenes-v1-scene-0042-f${String(frame).padStart(6, "0")}`,
          split_group_uid: "kitscenes-scene-0042",
          split_bucket: 1,
          episode_id: "scene-0042",
          frame_idx: frame,
          trip_frame: 64 + frame,
          members: Object.fromEntries(
            CAMERAS.map((camera, index) => [
              `${camera}.jpg`,
              {
                offset: frame * 10_000 + index * 512,
                size: PIXEL.length,
              },
            ]),
          ),
          ego_now: [8 + frame, 0.05, 0, 0],
          ego_history: egoHistory(8 + frame),
          ego_future: egoFuture(),
          pose_current: {
            latitude_deg: 49,
            longitude_deg: 11 + frame * 0.00001,
            heading_deg_cw_from_north: 90,
            timestamp_ns: "1700000000000000000",
            gps_accuracy_m: null,
          },
          has_reasoning: false,
        })),
      });
    }
    if (path.endsWith("/overlay-models")) {
      return fulfillJSON(route, {
        dataset: "kitscenes",
        version: "v2.1",
        shard: "train-000000.tar",
        models: [],
      });
    }
    if (path.endsWith("/rig-projection")) {
      return route.fulfill({ status: 404, body: "not published" });
    }
    if (path.endsWith("/geo/episodes/scene-0042")) {
      return route.fulfill({
        status: 200,
        contentType: "application/octet-stream",
        body: episodePath(),
      });
    }
    if (path.endsWith("/blob")) {
      return route.fulfill({ status: 403, body: "private range" });
    }
    if (path.includes("/image/cam_")) {
      return route.fulfill({
        status: 200,
        contentType: "image/png",
        body: PIXEL,
      });
    }
    return route.fulfill({ status: 404, body: "not mocked" });
  });
}

test("map exposes features and supports keyboard panning", async ({ page }) => {
  await mockConsole(page);
  await page.goto("/geo?dataset=kitscenes&version=v2.1");

  const map = page.getByRole("region", {
    name: "Aggregate geographic dataset coverage",
  });
  await expect(map).toBeVisible();
  await expect(map).toHaveAttribute(
    "aria-keyshortcuts",
    "ArrowUp ArrowDown ArrowLeft ArrowRight",
  );

  const markers = page.getByRole("list", { name: "Map markers" });
  await expect(markers.getByRole("listitem")).toHaveCount(2);
  await expect(markers).toContainText("80 samples | 5 episodes");
  await expect(markers).toContainText("latitude 49.00000");

  const firstMarker = map.locator("svg circle").first();
  const markerX = () =>
    firstMarker.evaluate((element) => Number(element.getAttribute("cx")));
  const before = await markerX();

  await map.focus();
  await page.keyboard.press("ArrowRight");

  await expect(map).toBeFocused();
  await expect.poll(markerX).toBeLessThan(before - 50);
});

test("camera and player controls remain usable at 320px", async (
  { page },
  testInfo,
) => {
  await mockConsole(page);
  await page.setViewportSize({ width: 320, height: 844 });
  await page.goto(
    "/scenes/kitscenes/train-000000.tar/0?version=v2.1&mode=focus",
  );

  await expect(
    page.locator('[aria-label^="Episode player"]'),
  ).toBeVisible();
  const filmstrip = page.getByRole("group", { name: "Camera filmstrip" });
  const cameraButtons = filmstrip.getByRole("button");
  await expect(cameraButtons).toHaveCount(7);

  const frontCenter = filmstrip.getByRole("button", {
    name: "front-center camera",
  });
  const ringFront = filmstrip.getByRole("button", {
    name: "ring-front camera",
  });
  await expect(frontCenter).toHaveAttribute("aria-pressed", "true");
  await expect(ringFront).toHaveAttribute("aria-pressed", "false");
  expect(await frontCenter.evaluate((element) => element.tagName)).toBe(
    "BUTTON",
  );

  await ringFront.focus();
  await page.keyboard.press("Enter");
  await expect(ringFront).toHaveAttribute("aria-pressed", "true");
  await expect(frontCenter).toHaveAttribute("aria-pressed", "false");

  const filmstripMetrics = await filmstrip.evaluate((element) => ({
    clientWidth: element.clientWidth,
    scrollWidth: element.scrollWidth,
  }));
  expect(filmstripMetrics.scrollWidth).toBeGreaterThan(
    filmstripMetrics.clientWidth,
  );

  const tileMetrics = await cameraButtons.evaluateAll((buttons) =>
    buttons.map((button) => {
      const buttonBox = button.getBoundingClientRect();
      const label = Array.from(button.querySelectorAll("span")).find(
        (element) => element.classList.contains("bottom-0"),
      );
      const labelBox = label?.getBoundingClientRect();
      return {
        width: buttonBox.width,
        height: buttonBox.height,
        labelFits:
          labelBox !== undefined &&
          labelBox.left >= buttonBox.left &&
          labelBox.right <= buttonBox.right,
      };
    }),
  );
  for (const metric of tileMetrics) {
    expect(metric.width).toBeGreaterThanOrEqual(112);
    expect(metric.height).toBeGreaterThanOrEqual(63);
    expect(metric.labelFits).toBe(true);
  }

  const timeline = page.getByRole("slider", { name: "Timeline" });
  await timeline.focus();
  await page.keyboard.press("ArrowRight");
  await expect(timeline).toHaveAttribute("aria-valuenow", "1");

  const sceneMap = page.getByRole("region", {
    name: "Driven route with recorded and predicted trajectories",
  });
  await expect(sceneMap).toBeVisible();
  const paths = page.getByRole("list", { name: "Map paths" });
  await expect(paths).toContainText("recorded future path, 64 points");

  await sceneMap.focus();
  await page.keyboard.press("ArrowLeft");
  await expect(timeline).toHaveAttribute("aria-valuenow", "1");

  const documentOverflow = await page.evaluate(
    () => document.documentElement.scrollWidth - window.innerWidth,
  );
  expect(documentOverflow).toBeLessThanOrEqual(1);
  await page.screenshot({
    path: testInfo.outputPath("player-accessibility-320.png"),
    fullPage: true,
  });
});
