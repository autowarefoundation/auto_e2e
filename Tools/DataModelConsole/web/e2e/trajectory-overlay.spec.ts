import { createHash } from "node:crypto";

import { expect, test } from "@playwright/test";

const MODEL_ID = "a".repeat(64);
const SAMPLE_UIDS = [
  "kitscenes-v1-scene-0042-f000064",
  "kitscenes-v1-scene-0042-f000065",
  "kitscenes-v1-scene-0042-f000066",
];
const SAMPLE_KEYS = ["s00000064", "s00000065", "s00000066"];
const KITSCENES_FRONT_MATRIX = [
  [
    128.75527954101562,
    -131.19908142089844,
    1.0061875581741333,
    -52.390167236328125,
  ],
  [
    127.52409362792969,
    -0.9081462025642395,
    -249.18359375,
    -149.383544921875,
  ],
  [
    0.9999749660491943,
    0.0070677390322089195,
    0.0003683842078316957,
    -0.4213404357433319,
  ],
];
const PIXEL = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAEAAAAAkAQMAAAADwq7RAAAAIGNIUk0AAHomAACAhAAA+gAAAIDoAAB1MAAA6mAAADqYAAAXcJy6UTwAAAAGUExURTNBVf///753ZLcAAAABYktHRAH/Ai3eAAAAB3RJTUUH6gcPAQU1u04EUwAAAA1JREFUGNNjYBgFlAIAAUQAAS6fR94AAAAldEVYdGRhdGU6Y3JlYXRlADIwMjYtMDctMTVUMDE6MDU6NTMrMDA6MDCLG6dUAAAAJXRFWHRkYXRlOm1vZGlmeQAyMDI2LTA3LTE1VDAxOjA1OjUzKzAwOjAw+kYf6AAAACh0RVh0ZGF0ZTp0aW1lc3RhbXAAMjAyNi0wNy0xNVQwMTowNTo1MyswMDowMK1TPjcAAAAASUVORK5CYII=",
  "base64",
);
const NEXT_PIXEL = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAEAAAAAkAQMAAAADwq7RAAAAIGNIUk0AAHomAACAhAAA+gAAAIDoAAB1MAAA6mAAADqYAAAXcJy6UTwAAAAGUExURdwmJv///wv9ac8AAAABYktHRAH/Ai3eAAAAB3RJTUUH6gcPDxE6D8HjnQAAAA1JREFUGNNjYBgFlAIAAUQAAS6fR94AAAAldEVYdGRhdGU6YXRlADIwMjYtMDctMTVUMTU6MTc6NTgrMDA6MDAR0XezAAAAJXRFWHRkYXRlOm1vZGlmeQAyMDI2LTA3LTE1VDE1OjE3OjU4KzAwOjAwYIzPDwAAACh0RVh0ZGF0ZTp0aW1lc3RhbXAAMjAyNi0wNy0xNVQxNToxNzo1OCswMDowMDeZ7tAAAAAASUVORK5CYII=",
  "base64",
);

function uidHash(uid: string): bigint {
  return createHash("sha256").update(uid).digest().readBigUInt64LE(0);
}

function overlayBody(): Buffer {
  const sampleCount = SAMPLE_UIDS.length;
  const seedCount = 3;
  const horizon = 64;
  const headerBytes = 20;
  const seedsBytes = seedCount * 8;
  const directoryBytes = sampleCount * 12;
  const controlsBytes = sampleCount * seedCount * horizon * 2 * 4;
  const body = Buffer.alloc(
    headerBytes + seedsBytes + directoryBytes + controlsBytes + sampleCount * 4,
  );
  body.write("AOVL", 0, "ascii");
  body.writeUInt16LE(1, 4);
  body.writeUInt16LE(0, 6);
  body.writeUInt32LE(sampleCount, 8);
  body.writeUInt16LE(seedCount, 12);
  body.writeUInt16LE(horizon, 14);
  body.writeUInt16LE(2, 16);
  body.writeUInt16LE(0, 18);

  [0, 1, 2].forEach((seed, index) => {
    body.writeBigInt64LE(BigInt(seed), headerBytes + index * 8);
  });
  const directory = SAMPLE_UIDS.map((uid, row) => ({
    hash: uidHash(uid),
    row,
  })).sort((a, b) => (a.hash < b.hash ? -1 : 1));
  let cursor = headerBytes + seedsBytes;
  for (const entry of directory) {
    body.writeBigUInt64LE(entry.hash, cursor);
    body.writeUInt32LE(entry.row, cursor + 8);
    cursor += 12;
  }

  const controlsOffset = headerBytes + seedsBytes + directoryBytes;
  for (let row = 0; row < sampleCount; row++) {
    for (let seed = 0; seed < seedCount; seed++) {
      for (let step = 0; step < horizon; step++) {
        const index = ((row * seedCount + seed) * horizon + step) * 2;
        body.writeFloatLE(0.05, controlsOffset + index * 4);
        body.writeFloatLE(
          (seed - 1) * 0.012,
          controlsOffset + (index + 1) * 4,
        );
      }
    }
  }
  const speedsOffset = controlsOffset + controlsBytes;
  for (let row = 0; row < sampleCount; row++) {
    body.writeFloatLE(8 + row, speedsOffset + row * 4);
  }
  return body;
}

function episodePath(): Buffer {
  const body = Buffer.alloc(80 * 32);
  for (let index = 0; index < 80; index++) {
    const offset = index * 32;
    body.writeDoubleLE(49, offset);
    body.writeDoubleLE(11 + index * 0.00001, offset + 8);
    body.writeDoubleLE(90, offset + 16);
    body.writeDoubleLE(1_700_000_000 + index * 0.1, offset + 24);
  }
  return body;
}

function egoHistory(speed: number): number[] {
  return Array.from({ length: 64 }, () => [speed, 0, 0, 0]).flat();
}

function egoFuture(): number[] {
  return Array.from({ length: 64 }, () => [0.05, 0.01]).flat();
}

test("trajectory overlays and geographic views honor production contracts", async ({
  page,
}, testInfo) => {
  let blobRequests = 0;
  let directImageRequests = 0;
  let frameOneImageAttempts = 0;
  let frameOneImagesAvailable = false;
  let rigRequestPath = "";
  const consoleErrors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => consoleErrors.push(error.message));

  await page.route("https://tile.openstreetmap.org/**", (route) =>
    route.fulfill({ status: 200, contentType: "image/png", body: PIXEL }),
  );
  await page.route("**/api/v1/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    const json = (body: unknown) =>
      route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(body),
      });

    if (path === "/api/v1/datasets") {
      return json({
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
      return json({
        dataset: "kitscenes",
        versions: [
          {
            version: "v2.1",
            total_samples: 3,
            shards: 1,
            episodes: 1,
            num_views: 2,
            has_map: true,
            has_world_model: false,
            has_gps: true,
            size_bytes: 1000,
            has_manifest: true,
          },
        ],
      });
    }
    if (path.endsWith("/geo-stats")) {
      return json({
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
        computed_at: "2026-07-15T00:00:00Z",
      });
    }
    if (path.endsWith("/geo/heatmap")) {
      return json({
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
    if (path.endsWith("/shards")) {
      return json({
        dataset: "kitscenes",
        shards: [
          {
            name: "train-000000.tar",
            key: "kitscenes/v2.1/shards/train-000000.tar",
            size_bytes: 1000,
            last_modified: "2026-07-15T00:00:00Z",
          },
        ],
        page: { limit: 200, offset: 0, total: 1, more: false },
      });
    }
    if (path.endsWith("/index")) {
      return json({
        fps: 10,
        version: "v2.1",
        shard: "train-000000.tar",
        blob_ranges_allowed: false,
        samples: SAMPLE_UIDS.map((sampleUID, index) => ({
          key: SAMPLE_KEYS[index],
          sample_uid: sampleUID,
          split_group_uid: "kitscenes-scene-0042",
          split_bucket: 1,
          episode_id: "scene-0042",
          frame_idx: 64 + index,
          trip_frame: 64 + index,
          members: {
            "cam_0.jpg": { offset: index * 10_000 + 512, size: 200 },
            "cam_1.jpg": { offset: index * 10_000 + 2048, size: 200 },
          },
          ego_now: [8 + index, 0.05, 0, 0],
          ego_history: egoHistory(8 + index),
          ego_future: egoFuture(),
          pose_current: {
            latitude_deg: 49,
            longitude_deg: 11 + index * 0.00001,
            heading_deg_cw_from_north: 90,
            timestamp_ns: "1700000000000000000",
            gps_accuracy_m: null,
          },
          has_reasoning: false,
        })),
      });
    }
    if (path.endsWith("/overlay-models")) {
      return json({
        dataset: "kitscenes",
        version: "v2.1",
        shard: "train-000000.tar",
        models: [
          {
            model_artifact_id: MODEL_ID,
            registered_model_name: "auto-e2e-driving-policy",
            model_version: 30,
            run_id: "run-30",
            model_name: "ConvNeXt-T",
            eval_ade: 1.25,
            eval_fde: 2.5,
            val_fraction: 0.3,
            overlay_schema: "v1",
            sample_count: 3,
          },
        ],
      });
    }
    if (path.endsWith(`/overlays/${MODEL_ID}`)) {
      return route.fulfill({
        status: 200,
        contentType: "application/vnd.auto-e2e.overlay",
        body: overlayBody(),
      });
    }
    if (path.endsWith("/rig-projection")) {
      rigRequestPath = path;
      return json({
        schema_version: "v1",
        dataset: "kitscenes",
        geometry_type: "pinhole",
        image_size: 256,
        projection: {
          type: "pinhole",
          matrix: [KITSCENES_FRONT_MATRIX, KITSCENES_FRONT_MATRIX],
        },
      });
    }
    if (path.endsWith("/geo/episodes/scene-0042")) {
      return route.fulfill({
        status: 200,
        contentType: "application/octet-stream",
        body: episodePath(),
      });
    }
    if (path.endsWith("/blob")) {
      blobRequests++;
      return route.fulfill({ status: 403, body: "private range" });
    }
    if (path.includes("/image/cam_")) {
      directImageRequests++;
      if (path.includes(`/samples/${SAMPLE_KEYS[1]}/`)) {
        frameOneImageAttempts++;
        if (!frameOneImagesAvailable) {
          return route.fulfill({
            status: 200,
            contentType: "image/png",
            body: Buffer.from("not an image"),
          });
        }
        return route.fulfill({
          status: 200,
          contentType: "image/png",
          body: NEXT_PIXEL,
        });
      }
      return route.fulfill({
        status: 200,
        contentType: "image/png",
        body: PIXEL,
      });
    }
    return route.fulfill({ status: 404, body: "not mocked" });
  });

  await page.goto(
    "/scenes/kitscenes/train-000000.tar/0?version=v2.1",
    { waitUntil: "networkidle" },
  );
  await expect(page.locator("#trajectory-model")).toHaveValue(MODEL_ID);
  await expect(page.getByText("3 seeds | median")).toBeVisible();
  await expect(page.getByText("episode/clip hold-out")).toBeVisible();
  await expect(page.getByText("Scene map")).toBeVisible();
  expect(rigRequestPath).toBe(
    "/api/v1/datasets/kitscenes/shards/train-000000.tar/rig-projection",
  );
  await expect(
    page.locator('svg path[stroke="#6ee7b7"]').first(),
  ).toHaveAttribute("d", /^M/);
  await expect(page.getByLabel("Camera trajectory legend")).toContainText(
    "Ground truth",
  );
  await expect(page.getByLabel("Camera trajectory legend")).toContainText(
    "Prediction",
  );
  await expect.poll(() => directImageRequests).toBeGreaterThan(0);
  expect(blobRequests).toBe(0);

  const overlayPixels = await page
    .locator("canvas[aria-hidden='true']")
    .evaluateAll((elements) => {
      let nonEmptyCanvases = 0;
      let groundTruthPixels = 0;
      let predictionPixels = 0;
      for (const element of elements) {
        const canvas = element as HTMLCanvasElement;
        if (canvas.width === 0 || canvas.height === 0) continue;
        const context = canvas.getContext("2d");
        if (!context) continue;
        const data = context.getImageData(
          0,
          0,
          canvas.width,
          canvas.height,
        ).data;
        let nonEmpty = false;
        for (let offset = 0; offset < data.length; offset += 4) {
          const red = data[offset];
          const green = data[offset + 1];
          const blue = data[offset + 2];
          const alpha = data[offset + 3];
          if (alpha === 0) continue;
          nonEmpty = true;
          if (green > red * 1.2 && green > blue * 1.2) {
            predictionPixels++;
          }
          if (blue > red * 1.2 && blue > green * 1.2) {
            groundTruthPixels++;
          }
        }
        if (nonEmpty) nonEmptyCanvases++;
      }
      return { nonEmptyCanvases, groundTruthPixels, predictionPixels };
    });
  expect(overlayPixels.nonEmptyCanvases).toBeGreaterThan(0);
  expect(overlayPixels.groundTruthPixels).toBeGreaterThan(0);
  expect(overlayPixels.predictionPixels).toBeGreaterThan(0);

  const imageCanvas = page.locator("canvas:not([aria-hidden])").first();
  const overlayCanvas = page.locator("canvas[aria-hidden='true']").first();
  const cameraPixel = () =>
    imageCanvas.evaluate((canvas) =>
      Array.from(
        (canvas as HTMLCanvasElement)
          .getContext("2d")!
          .getImageData(0, 0, 1, 1).data,
      ),
    );
  const overlaySnapshot = () =>
    overlayCanvas.evaluate((canvas) =>
      (canvas as HTMLCanvasElement).toDataURL(),
    );

  await expect.poll(cameraPixel).toEqual([51, 65, 85, 255]);
  const initialOverlay = await overlaySnapshot();
  const attemptsBeforeSeek = frameOneImageAttempts;
  const timeline = page.getByRole("slider", { name: "Timeline" });
  await timeline.focus();
  await page.keyboard.press("ArrowRight");
  await expect(timeline).toHaveAttribute("aria-valuenow", "1");
  await expect
    .poll(() => frameOneImageAttempts)
    .toBeGreaterThan(attemptsBeforeSeek);
  await page.waitForTimeout(100);

  // While frame 1 is unavailable, retain frame 0 as an atomic image/overlay
  // pair instead of painting frame 1's paths over frame 0's image.
  expect(await cameraPixel()).toEqual([51, 65, 85, 255]);
  expect(await overlaySnapshot()).toBe(initialOverlay);

  frameOneImagesAvailable = true;
  await expect
    .poll(cameraPixel, { timeout: 5_000 })
    .toEqual([220, 38, 38, 255]);
  await expect
    .poll(overlaySnapshot, { timeout: 5_000 })
    .not.toBe(initialOverlay);

  await page.screenshot({
    path: testInfo.outputPath("scene-overlay-desktop.png"),
    fullPage: true,
  });

  await page
    .getByRole("checkbox", { name: "Display-limited" })
    .check();
  await expect
    .poll(
      () =>
        new URL(page.url()).searchParams.get("prediction_mode"),
    )
    .toBe("display-limited");

  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByText("Scene map")).toBeVisible();
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - window.innerWidth,
  );
  expect(overflow).toBeLessThanOrEqual(1);
  await page.screenshot({
    path: testInfo.outputPath("scene-overlay-mobile.png"),
    fullPage: true,
  });

  await page.goto("/geo", { waitUntil: "networkidle" });
  await expect(page.getByText("Geographic coverage")).toBeVisible();
  await expect(page.getByText("2 published cells")).toBeVisible();
  await expect(page.locator('svg circle[fill="#10b981"]')).toHaveCount(2);
  await page.screenshot({
    path: testInfo.outputPath("geo-coverage.png"),
    fullPage: true,
  });

  expect(consoleErrors, consoleErrors.join("\n")).toHaveLength(0);
});
