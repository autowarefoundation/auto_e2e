import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import path from "node:path";

import {
  expect,
  test,
  type Locator,
  type Page,
} from "@playwright/test";

const DATASET = "kitscenes";
const VERSION = "v3.3";
const SHARD = "scene-a-train-000000.tar";
const AUTOE2E_WEIGHT =
  "e41978b037986bc874ec9eec0aebf732c096ae5bfa64cee9c5fb3a4168e87a01";
const AUTOE2E_MODEL =
  "0db1fed95d588e17c6589ef543fe4724ff130cddbe369c36e2c4615920fa375e";
const BEVFORMER_WEIGHT =
  "5585bc4d3ff8b396928cb92d91f773a2c57a81258f83cab0c668ebb2eb9d3307";
const BEVFORMER_MODEL =
  "31d54e3e882c65e0a58036b4bd3f1c1868d9eef09a78ee2f2c9ffecb022cc20d";
const HENET_WEIGHT =
  "c8cc1c2bb5d00e73a744e0f29b08085bc111a25bf4bd0478d8110b1a8db6d220";
const HENET_MODEL =
  "7315c3b08377bef6e8f67e58438103022dd4655a593b29b2fef4e6c8b2762b6b";
const SAMPLE_UIDS = [
  "kitscenes-v1-scene-a-f000000",
  "kitscenes-v1-scene-a-f000001",
] as const;
const HEIGHT = 450;
const WIDTH = 300;
const CLASS_COUNT = 8;

function hashWords(value: string) {
  const digest = createHash("sha256").update(value).digest();
  return {
    high: digest.readUInt32LE(4),
    low: digest.readUInt32LE(0),
  };
}

function paintRectangle(
  target: Uint8Array,
  sampleRow: number,
  classIndex: number,
  minRow: number,
  maxRow: number,
  minCol: number,
  maxCol: number,
  value: number,
) {
  const scaledMinRow = Math.round((minRow / 96) * HEIGHT);
  const scaledMaxRow = Math.min(
    HEIGHT - 1,
    Math.round(((maxRow + 1) / 96) * HEIGHT) - 1,
  );
  const scaledMinCol = Math.round((minCol / 64) * WIDTH);
  const scaledMaxCol = Math.min(
    WIDTH - 1,
    Math.round(((maxCol + 1) / 64) * WIDTH) - 1,
  );
  for (
    let rasterRow = scaledMinRow;
    rasterRow <= scaledMaxRow;
    rasterRow++
  ) {
    for (
      let rasterCol = scaledMinCol;
      rasterCol <= scaledMaxCol;
      rasterCol++
    ) {
      const index =
        (((sampleRow * CLASS_COUNT + classIndex) * HEIGHT + rasterRow) *
          WIDTH) +
        rasterCol;
      target[index] = value;
    }
  }
}

function occupancyBody(kind: "autoe2e" | "bevformer" | "henet"): Buffer {
  const sampleCount = SAMPLE_UIDS.length;
  const cellCount = sampleCount * CLASS_COUNT * HEIGHT * WIDTH;
  const hasTeacher = kind === "autoe2e";
  const validByteCount = Math.ceil(cellCount / 8);
  const byteLength =
    20 +
    sampleCount * 12 +
    cellCount +
    (hasTeacher ? cellCount + validByteCount : 0);
  const output = new Uint8Array(byteLength);
  const view = new DataView(output.buffer);
  output.set([65, 83, 79, 67], 0);
  view.setUint16(4, 1, true);
  view.setUint16(6, hasTeacher ? 1 : 0, true);
  view.setUint32(8, sampleCount, true);
  view.setUint16(12, CLASS_COUNT, true);
  view.setUint16(14, HEIGHT, true);
  view.setUint16(16, WIDTH, true);

  const directory = SAMPLE_UIDS.map((sampleUID, row) => ({
    ...hashWords(sampleUID),
    row,
  })).sort(
    (left, right) =>
      left.high - right.high || left.low - right.low,
  );
  let cursor = 20;
  for (const entry of directory) {
    view.setUint32(cursor, entry.low, true);
    view.setUint32(cursor + 4, entry.high, true);
    view.setUint32(cursor + 8, entry.row, true);
    cursor += 12;
  }

  const prediction = new Uint8Array(output.buffer, cursor, cellCount);
  if (kind === "autoe2e") {
    for (let sampleRow = 0; sampleRow < sampleCount; sampleRow++) {
      paintRectangle(prediction, sampleRow, 0, 3, 93, 8, 55, 154);
      paintRectangle(prediction, sampleRow, 2, 36, 55, 8, 55, 186);
      for (let row = 8; row < 90; row += 14) {
        paintRectangle(prediction, sampleRow, 1, row, row + 5, 26, 27, 226);
        paintRectangle(prediction, sampleRow, 1, row, row + 5, 37, 38, 226);
      }
      paintRectangle(prediction, sampleRow, 3, 44, 48, 13, 50, 238);
      paintRectangle(prediction, sampleRow, 4, 58, 59, 10, 53, 246);
      paintRectangle(
        prediction,
        sampleRow,
        5,
        63 - sampleRow * 5,
        69 - sampleRow * 5,
        30,
        34,
        238,
      );
      paintRectangle(prediction, sampleRow, 5, 38, 44, 45, 49, 220);
      paintRectangle(prediction, sampleRow, 6, 55, 57, 20, 22, 230);
      paintRectangle(prediction, sampleRow, 7, 48, 52, 10, 14, 218);
    }
  } else if (kind === "bevformer") {
    for (let sampleRow = 0; sampleRow < sampleCount; sampleRow++) {
      paintRectangle(prediction, sampleRow, 5, 67, 75, 27, 33, 244);
      paintRectangle(prediction, sampleRow, 5, 42, 50, 43, 49, 226);
      paintRectangle(prediction, sampleRow, 6, 58, 60, 18, 20, 235);
      paintRectangle(prediction, sampleRow, 7, 35, 41, 11, 17, 214);
    }
  } else {
    for (let sampleRow = 0; sampleRow < sampleCount; sampleRow++) {
      paintRectangle(prediction, sampleRow, 0, 3, 93, 8, 55, 192);
      for (let row = 12; row < 90; row += 16) {
        paintRectangle(prediction, sampleRow, 1, row, row + 2, 26, 27, 238);
        paintRectangle(prediction, sampleRow, 1, row, row + 2, 37, 38, 238);
      }
      paintRectangle(
        prediction,
        sampleRow,
        5,
        60 - sampleRow * 5,
        66 - sampleRow * 5,
        30,
        34,
        246,
      );
    }
  }
  cursor += cellCount;

  if (hasTeacher) {
    const teacher = new Uint8Array(output.buffer, cursor, cellCount);
    for (let sampleRow = 0; sampleRow < sampleCount; sampleRow++) {
      teacher.set(
        prediction.subarray(
          sampleRow * CLASS_COUNT * HEIGHT * WIDTH,
          (sampleRow + 1) * CLASS_COUNT * HEIGHT * WIDTH,
        ),
        sampleRow * CLASS_COUNT * HEIGHT * WIDTH,
      );
      paintRectangle(teacher, sampleRow, 5, 63, 69, 30, 34, 0);
      paintRectangle(teacher, sampleRow, 5, 61, 67, 32, 36, 255);
      paintRectangle(teacher, sampleRow, 6, 55, 57, 20, 22, 0);
      paintRectangle(teacher, sampleRow, 6, 52, 54, 23, 25, 255);
    }
    cursor += cellCount;
    new Uint8Array(output.buffer, cursor, validByteCount).fill(255);
  }
  return Buffer.from(output);
}

function model(
  modelArtifactID: string,
  overrides: Record<string, unknown>,
) {
  return {
    model_artifact_id: modelArtifactID,
    display_name: "AutoE2E Reactive BEV segmentation",
    model_family: "AutoE2E Reactive",
    artifact_kind: "native-semantic-occupancy",
    artifact_schema: "v1",
    created_at: "2026-08-19T14:00:00Z",
    dataset_manifest_sha256: "a".repeat(64),
    geometry_id: "autoe2e-bev-450x300-0p4m-v1",
    taxonomy_version: "autoe2e-bev-semantic-v1",
    head_version: "bev-segmentation-head-v1",
    input_contract: "autoe2e-packed-calibrated-camera-v1",
    supported_classes: [
      "drivable_area",
      "lane_boundary",
      "intersection",
      "crosswalk",
      "stop_line",
      "vehicle",
      "vulnerable_road_user",
      "other_obstacle",
    ],
    teacher_available: true,
    limitations: [
      "Predictions use the native head without viewer-side correction.",
    ],
    model_source: {
      code_license_spdx: "Apache-2.0",
      config: "embedded-checkpoint-config",
      license_spdx: "NOASSERTION",
      repository: "https://github.com/autowarefoundation/auto_e2e",
      repository_revision: "c".repeat(40),
      training_data_license_spdx: "NOASSERTION",
      weight_sha256: AUTOE2E_WEIGHT,
      weight_source_url: `urn:sha256:${AUTOE2E_WEIGHT}`,
    },
    producer_config: {
      deterministic_algorithms: true,
      probability_encoding: "uint8-rint-gzip-level-6-v1",
    },
    sample_count: 2,
    shard_count: 1,
    shard_sample_count: 2,
    ...overrides,
  };
}

async function installOccupancyMocks(page: Page) {
  const cameraImage = readFileSync(
    path.join(
      process.cwd(),
      "public/assets/semantic-occupancy/kenney-car-kit/Textures/colormap.png",
    ),
  );
  const members = Object.fromEntries(
    Array.from({ length: 6 }, (_, camera) => [
      `cam_${camera}.jpg`,
      { offset: camera * cameraImage.length, size: cameraImage.length },
    ]),
  );
  const samples = SAMPLE_UIDS.map((sampleUID, frameIndex) => ({
    key: sampleUID,
    sample_uid: sampleUID,
    split_group_uid: "scene-a",
    split_bucket: 0,
    episode_id: "scene-a",
    frame_idx: frameIndex,
    trip_frame: frameIndex,
    members,
    ego_now: [8.5, 0.1, 0, 0],
    ego_history: [],
    ego_future: [],
    has_reasoning: false,
  }));
  const autoe2eBody = occupancyBody("autoe2e");
  const bevformerBody = occupancyBody("bevformer");
  const henetBody = occupancyBody("henet");
  const models = [
    model(AUTOE2E_MODEL, {}),
    model(BEVFORMER_MODEL, {
      display_name: "BEVFormer V2 R50 t8 detection footprints",
      model_family: "BEVFormer V2",
      artifact_kind: "detection-derived-occupancy",
      head_version: "bevformer-v2-r50-t8-box-raster-v1",
      input_contract:
        "kitscenes-packed-256-square-six-camera-to-bevformer-640x256-v1",
      supported_classes: [
        "other_obstacle",
        "vehicle",
        "vulnerable_road_user",
      ],
      teacher_available: false,
      limitations: [
        "Official BEVFormer V2 publishes 3-D detection, not BEV segmentation.",
        "Road and map classes are unsupported and remain empty.",
        "External weight license is not separately stated; redistribution is disabled.",
      ],
      model_source: {
        code_license_spdx: "Apache-2.0",
        config: "bevformerv2-r50-t8-24ep.py",
        license_spdx: "NOASSERTION",
        repository: "https://github.com/fundamentalvision/BEVFormer",
        repository_revision:
          "66b65f3a1f58caf0507cb2a971b9c0e7f842376c",
        training_data_license_spdx: "CC-BY-NC-SA-4.0",
        weight_sha256: BEVFORMER_WEIGHT,
        weight_source_url:
          "https://drive.google.com/drive/folders/1Ml_usx5BNx43CFH1Di2OTazuzSyAlBto",
      },
      producer_config: {
        deterministic_algorithms: true,
        score_threshold: 0.2,
      },
    }),
    model(HENET_MODEL, {
      display_name: "HENet BEV segmentation",
      model_family: "HENet",
      head_version: "henet-det-bevseg-v1",
      input_contract:
        "kitscenes-packed-256-square-six-camera-to-henet-short-640x1152-long-256x704-v1",
      supported_classes: [
        "drivable_area",
        "lane_area",
        "vehicle",
      ],
      teacher_available: false,
      limitations: [
        "HENet divider predictions are shown as lane dividers, not lane surfaces.",
        "HENet is only free for academic research purposes and needs author authorization for commerce.",
      ],
      model_source: {
        code_license_spdx: "LicenseRef-HENet-Research-Only",
        config: "henet_det_bevseg.py",
        license_spdx: "LicenseRef-HENet-Research-Only",
        repository: "https://github.com/VDIGPKU/HENet",
        repository_revision:
          "29ca81dd109cabe0a0c53ee354c4a74ad1559740",
        training_data_license_spdx: "NOASSERTION",
        weight_sha256: HENET_WEIGHT,
        weight_source_url:
          "https://drive.google.com/drive/folders/1AYajSnL1JLrFOTHcpjX7WlRRLXUxQhwV",
      },
      producer_config: {
        input_height: 640,
        input_width: 1152,
        temporal_boundary_policy: "repeat-current-frame-v1",
      },
    }),
  ];

  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    const pathname = url.pathname;
    if (pathname === "/api/v1/datasets") {
      await route.fulfill({
        json: {
          datasets: [
            {
              name: DATASET,
              version: VERSION,
              prefix: `${DATASET}/${VERSION}/shards/`,
            },
          ],
        },
      });
      return;
    }
    if (pathname === `/api/v1/datasets/${DATASET}/versions`) {
      await route.fulfill({
        json: {
          dataset: DATASET,
          versions: [
            {
              version: VERSION,
              total_samples: samples.length,
              shards: 1,
              episodes: 1,
              num_views: 6,
              has_map: true,
              has_world_model: false,
              has_gps: true,
              size_bytes: 2048,
              has_manifest: true,
            },
          ],
        },
      });
      return;
    }
    if (pathname === `/api/v1/datasets/${DATASET}/shards`) {
      await route.fulfill({
        json: {
          dataset: DATASET,
          shards: [
            {
              name: SHARD,
              key: `${DATASET}/${VERSION}/shards/${SHARD}`,
              size_bytes: 2048,
              last_modified: "2026-08-19T14:00:00Z",
            },
          ],
          page: {
            limit: 1000,
            offset: 0,
            total: 1,
            more: false,
          },
        },
      });
      return;
    }
    if (
      pathname ===
      `/api/v1/datasets/${DATASET}/shards/${SHARD}/index`
    ) {
      await route.fulfill({
        json: {
          fps: 10,
          version: VERSION,
          shard: SHARD,
          blob_ranges_allowed: true,
          samples,
        },
      });
      return;
    }
    if (
      pathname ===
      `/api/v1/datasets/${DATASET}/shards/${SHARD}/semantic-occupancy-models`
    ) {
      await route.fulfill({
        json: {
          dataset: DATASET,
          version: VERSION,
          shard: SHARD,
          models,
        },
      });
      return;
    }
    if (pathname.includes("/semantic-occupancy/")) {
      await route.fulfill({
        body: pathname.endsWith(BEVFORMER_MODEL)
          ? bevformerBody
          : pathname.endsWith(HENET_MODEL)
            ? henetBody
            : autoe2eBody,
        contentType: "application/vnd.auto-e2e.semantic-occupancy",
      });
      return;
    }
    if (pathname.includes("/image/cam_")) {
      await route.fulfill({
        body: cameraImage,
        contentType: "image/png",
      });
      return;
    }
    await route.fulfill({ status: 404, body: "unmocked API route" });
  });
}

async function webGLPaintState(canvas: Locator) {
  return canvas.evaluate((element) => {
    const target = element as HTMLCanvasElement;
    const gl =
      target.getContext("webgl2") ?? target.getContext("webgl");
    if (!gl) {
      return {
        colorfulSamples: 0,
        hash: 0,
        height: 0,
        paintedSamples: 0,
        variance: 0,
        width: 0,
      };
    }
    gl.finish();
    const width = gl.drawingBufferWidth;
    const height = gl.drawingBufferHeight;
    const pixels = new Uint8Array(width * height * 4);
    gl.readPixels(
      0,
      0,
      width,
      height,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      pixels,
    );
    const background = [pixels[0], pixels[1], pixels[2]];
    let colorfulSamples = 0;
    let hash = 2166136261;
    let luminanceSquaredSum = 0;
    let luminanceSum = 0;
    let paintedSamples = 0;
    let samples = 0;
    for (let offset = 0; offset < pixels.length; offset += 16) {
      const red = pixels[offset];
      const green = pixels[offset + 1];
      const blue = pixels[offset + 2];
      const difference =
        Math.abs(red - background[0]) +
        Math.abs(green - background[1]) +
        Math.abs(blue - background[2]);
      if (difference > 18) paintedSamples++;
      if (Math.max(red, green, blue) - Math.min(red, green, blue) > 12) {
        colorfulSamples++;
      }
      const luminance = (red + green + blue) / 3;
      luminanceSum += luminance;
      luminanceSquaredSum += luminance * luminance;
      hash = Math.imul(hash ^ red, 16777619);
      hash = Math.imul(hash ^ green, 16777619);
      hash = Math.imul(hash ^ blue, 16777619);
      samples++;
    }
    const mean = luminanceSum / samples;
    return {
      colorfulSamples,
      hash: hash >>> 0,
      height,
      paintedSamples,
      variance: luminanceSquaredSum / samples - mean * mean,
      width,
    };
  });
}

async function openOccupancy(
  page: Page,
  viewport: { width: number; height: number },
) {
  const errors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") {
      errors.push(`console: ${message.text()}`);
    }
  });
  page.on("pageerror", (error) => {
    errors.push(`page: ${error.message}`);
  });
  await installOccupancyMocks(page);
  await page.setViewportSize(viewport);
  await page.goto("/occupancy", { waitUntil: "domcontentloaded" });
  await expect(
    page.getByLabel("Occupancy model", { exact: true }),
  ).toHaveValue(
    AUTOE2E_MODEL,
    { timeout: 20_000 },
  );
  const region = page.getByRole("region", {
    name: "3D semantic occupancy",
  });
  const canvas = region.locator("canvas").first();
  await expect(canvas).toBeVisible({ timeout: 20_000 });
  await canvas.scrollIntoViewIfNeeded();
  await expect(canvas).toHaveAttribute("data-engine", /three\.js/, {
    timeout: 30_000,
  });
  await expect
    .poll(async () => (await webGLPaintState(canvas)).paintedSamples, {
      timeout: 30_000,
    })
    .toBeGreaterThan(500);
  const cameraEvidence = page.getByRole("group", {
    name: "Camera evidence mosaic",
  });
  await cameraEvidence.scrollIntoViewIfNeeded();
  await expect(cameraEvidence.locator('img[alt^="cam_"]')).toHaveCount(6);
  await expect
    .poll(() =>
      cameraEvidence.locator('img[alt^="cam_"]').evaluateAll((images) =>
        images.every(
          (image) => (image as HTMLImageElement).naturalWidth > 0,
        ),
      ),
    )
    .toBe(true);
  return { canvas, errors, region };
}

async function prepareFullPageScreenshot(page: Page) {
  await page.addStyleTag({
    content: `
      header.sticky { position: static !important; }
      nextjs-portal { display: none !important; }
    `,
  });
  await page.evaluate(() => window.scrollTo(0, 0));
}

test("uses the shared 3D renderer for real occupancy selections", async ({
  page,
}, testInfo) => {
  testInfo.setTimeout(180_000);
  const { canvas, errors, region } = await openOccupancy(page, {
    width: 1440,
    height: 1000,
  });
  const orbitPaint = await webGLPaintState(canvas);
  expect(orbitPaint.width).toBeGreaterThan(700);
  expect(orbitPaint.height).toBeGreaterThan(500);
  expect(orbitPaint.colorfulSamples).toBeGreaterThan(300);
  expect(orbitPaint.variance).toBeGreaterThan(12);
  await expect(
    region.getByRole("slider", {
      name: "Semantic confidence threshold",
    }),
  ).toHaveValue("0.2");
  await expect(region).toContainText("4 objects");
  await expect(region).not.toContainText("/56 objects");
  const cameraLayout = await page
    .getByRole("group", { name: "Camera evidence mosaic" })
    .locator("figure")
    .evaluateAll((figures) =>
      figures.map((figure) => ({
        col: figure.style.gridColumn,
        label: figure.querySelector("figcaption")?.textContent?.trim(),
        row: figure.style.gridRow,
      })),
    );
  expect(cameraLayout).toEqual([
    { col: "2", label: "front-center", row: "1" },
    { col: "1", label: "front-left", row: "1" },
    { col: "3", label: "front-right", row: "1" },
    { col: "2", label: "rear", row: "2" },
    { col: "1", label: "rear-left", row: "2" },
    { col: "3", label: "rear-right", row: "2" },
  ]);
  const provenanceFollowsRenderer = await region.evaluate((renderer) => {
    const provenance = document.querySelector(
      '[aria-label="Occupancy model provenance"]',
    );
    return Boolean(
      provenance &&
        renderer.compareDocumentPosition(provenance) &
          Node.DOCUMENT_POSITION_FOLLOWING,
    );
  });
  expect(provenanceFollowsRenderer).toBe(true);

  await region.getByRole("button", { name: "Top view" }).click();
  await page.waitForTimeout(850);
  const topPaint = await webGLPaintState(canvas);
  expect(topPaint.hash).not.toBe(orbitPaint.hash);

  await region.getByRole("tab", { name: "Teacher" }).click();
  await page.waitForTimeout(250);
  const teacherPaint = await webGLPaintState(canvas);
  expect(teacherPaint.hash).not.toBe(topPaint.hash);

  await page.getByLabel("Next occupancy frame").click();
  await page.waitForTimeout(250);
  await expect(page.getByText(SAMPLE_UIDS[1])).toBeVisible();
  const nextFramePaint = await webGLPaintState(canvas);
  expect(nextFramePaint.hash).not.toBe(teacherPaint.hash);

  await page
    .getByLabel("Occupancy model", { exact: true })
    .selectOption(BEVFORMER_MODEL);
  await expect(
    region.getByRole("tab", { name: "Teacher" }),
  ).toBeDisabled();
  await expect(
    page.getByText(
      "Official BEVFormer V2 publishes 3-D detection, not BEV segmentation.",
    ),
  ).toBeVisible();
  await expect(page.getByText("CC-BY-NC-SA-4.0")).toBeVisible();
  await expect(
    page.getByRole("link", { name: "source" }),
  ).toHaveAttribute(
    "href",
    "https://drive.google.com/drive/folders/1Ml_usx5BNx43CFH1Di2OTazuzSyAlBto",
  );
  await expect(page.getByText(/"score_threshold":0.2/)).toBeVisible();
  await expect
    .poll(async () => (await webGLPaintState(canvas)).hash, {
      timeout: 20_000,
    })
    .not.toBe(nextFramePaint.hash);

  await page
    .getByLabel("Occupancy model", { exact: true })
    .selectOption(HENET_MODEL);
  await expect(region.getByText("Lane divider", { exact: true })).toBeVisible();
  await expect(
    page.getByText(
      "HENet is only free for academic research purposes and needs author authorization for commerce.",
    ),
  ).toBeVisible();
  await expect(
    page.getByText("LicenseRef-HENet-Research-Only").first(),
  ).toBeVisible();
  await expect(
    page.getByRole("link", { name: "source" }),
  ).toHaveAttribute(
    "href",
    "https://drive.google.com/drive/folders/1AYajSnL1JLrFOTHcpjX7WlRRLXUxQhwV",
  );

  const layout = await page.evaluate(() => ({
    horizontalOverflow:
      document.documentElement.scrollWidth -
      document.documentElement.clientWidth,
  }));
  expect(layout.horizontalOverflow).toBe(0);
  await canvas.evaluate((element) =>
    element.scrollIntoView({ block: "center", inline: "nearest" }),
  );
  await page.waitForTimeout(1_000);
  await page.screenshot({
    animations: "disabled",
    path: testInfo.outputPath("occupancy-dashboard-desktop.png"),
  });
  expect(errors).toEqual([]);
});

test("keeps the occupancy workflow usable on mobile", async ({
  page,
}, testInfo) => {
  testInfo.setTimeout(90_000);
  const { canvas, errors, region } = await openOccupancy(page, {
    width: 390,
    height: 844,
  });
  const orbitPaint = await webGLPaintState(canvas);
  expect(orbitPaint.width).toBeGreaterThan(300);
  expect(orbitPaint.height).toBeGreaterThan(350);
  expect(orbitPaint.colorfulSamples).toBeGreaterThan(120);

  await region.getByRole("button", { name: "Ego view" }).click();
  await page.waitForTimeout(850);
  const egoPaint = await webGLPaintState(canvas);
  expect(egoPaint.hash).not.toBe(orbitPaint.hash);

  const layout = await page.evaluate(() => {
    const viewportWidth = document.documentElement.clientWidth;
    const region = document.querySelector(
      '[aria-label="3D semantic occupancy"]',
    );
    const camera = region?.querySelector(
      '[role="group"][aria-label="Semantic occupancy camera"]',
    );
    const canvas = region?.querySelector("canvas");
    if (!region || !camera || !canvas) return null;
    const regionRect = region.getBoundingClientRect();
    const cameraRect = camera.getBoundingClientRect();
    const canvasRect = canvas.getBoundingClientRect();
    return {
      cameraInside:
        cameraRect.left >= canvasRect.left &&
        cameraRect.right <= canvasRect.right &&
        cameraRect.top >= canvasRect.top &&
        cameraRect.bottom <= canvasRect.bottom,
      horizontalOverflow:
        document.documentElement.scrollWidth - viewportWidth,
      regionInside:
        regionRect.left >= 0 && regionRect.right <= viewportWidth,
    };
  });
  expect(layout).toEqual({
    cameraInside: true,
    horizontalOverflow: 0,
    regionInside: true,
  });
  await prepareFullPageScreenshot(page);
  await page.screenshot({
    fullPage: true,
    path: testInfo.outputPath("occupancy-dashboard-mobile.png"),
  });
  expect(errors).toEqual([]);
});
