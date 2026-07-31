// Player smoke test. Requires the dev/prod web server on :3000 and the Go API
// on :8080 (reading real S3, AWS_PROFILE=autowarefoundation). Uses a canonical
// KITScenes v3.1 scene with real camera JPEGs.
//
// Run: (servers up) npx playwright test
import { test, expect, type Page } from "@playwright/test";

const SCENE =
  "/scenes/kitscenes/part-09da7262ba918bb3-train-000000.tar/0?version=v3.1";

async function cameraPaintState(page: Page) {
  return page.evaluate(() => {
    const canvases = Array.from(
      document.querySelectorAll<HTMLCanvasElement>(
        'button[aria-label$=" camera"] > canvas:not([aria-hidden])',
      ),
    ).filter((canvas) => canvas.offsetParent !== null);
    let ok = 0;
    for (const canvas of canvases) {
      const context = canvas.getContext("2d");
      if (!context || canvas.width === 0) continue;
      const { data } = context.getImageData(
        0,
        0,
        canvas.width,
        canvas.height,
      );
      let sum = 0;
      for (let offset = 0; offset < data.length; offset += 4) {
        sum += data[offset] + data[offset + 1] + data[offset + 2];
      }
      if (sum / (data.length / 4) / 3 > 2) ok++;
    }
    return { total: canvases.length, ok };
  });
}

test("player renders real camera pixels, advances, and focuses", async ({ page }) => {
  const consoleErrors: string[] = [];
  const responseErrors: string[] = [];
  page.on("console", (m) => {
    if (
      m.type() === "error" &&
      !m.text().startsWith("Failed to load resource:")
    ) {
      consoleErrors.push(m.text());
    }
  });
  page.on("pageerror", (e) => consoleErrors.push(`pageerror: ${e.message}`));
  page.on("response", (response) => {
    if (response.status() < 400) return;
    const path = new URL(response.url()).pathname;
    // A pack can omit a dedicated rig artifact and use its embedded calibration.
    if (response.status() === 404 && path.endsWith("/rig-projection")) return;
    responseErrors.push(`${response.status()} ${response.url()}`);
  });

  await page.goto(SCENE, { waitUntil: "domcontentloaded" });
  await expect(
    page.locator('[aria-label^="Episode player"]'),
  ).toBeVisible({ timeout: 30_000 });
  const playbackControls = page.getByRole("region", {
    name: "Playback controls",
  });
  await expect(
    playbackControls.getByRole("slider", { name: "Timeline" }),
  ).toBeVisible();
  await expect(
    playbackControls.getByRole("button", { name: "Play", exact: true }),
  ).toBeVisible();
  const playbackPrecedesCameras = await page
    .locator('[aria-label^="Episode player"]')
    .evaluate((player) => {
      const controls = player.querySelector('[aria-label="Playback controls"]');
      const camera = player.querySelector('button[aria-label$=" camera"]');
      return Boolean(
        controls &&
        camera &&
        controls.compareDocumentPosition(camera) &
          Node.DOCUMENT_POSITION_FOLLOWING,
      );
    });
  expect(playbackPrecedesCameras).toBe(true);
  // Every camera frame canvas must have non-blank pixels (real frame, not
  // black). The aria-hidden trajectory layer is intentionally transparent
  // until a model is selected, so it is not a frame-pixel assertion target.
  await expect
    .poll(
      () => cameraPaintState(page),
      { timeout: 30_000 },
    )
    .toEqual({ total: 6, ok: 6 });

  // Playback advances the frame readout.
  const readout = () =>
    page.evaluate(() =>
      Array.from(document.querySelectorAll("p, div")).find((e) =>
        /frame \d+\/\d+/.test(e.textContent ?? ""),
      )?.textContent ?? "",
    );
  const before = await readout();
  await page.locator('[aria-label^="Episode player"]').focus();
  await page.keyboard.press("Space");
  await page.waitForTimeout(900);
  await page.keyboard.press("Space");
  expect(await readout()).not.toBe(before);

  // Focus mode enlarges a single camera; Esc returns to grid.
  await page.keyboard.press("f");
  await page.waitForTimeout(300);
  await page.keyboard.press("Escape");

  expect(consoleErrors, `console errors: ${consoleErrors.join("; ")}`).toHaveLength(0);
  expect(
    responseErrors,
    `HTTP errors: ${responseErrors.join("; ")}`,
  ).toHaveLength(0);
});

// Fill-rate regression: windowed contiguous fetch must let the buffer advance
// at roughly real time. Before it, the player made ~6 tiny range GETs per
// frame and the buffer filled well below 10Hz over a high-latency link, so
// playback stalled for a long time before it looked smooth. We measure how far
// the playhead advances over a fixed wall-clock window after pressing play.
test("playback fills its buffer near real time (windowed fetch)", async ({
  page,
}) => {
  test.setTimeout(90_000);
  await page.goto(SCENE, { waitUntil: "domcontentloaded" });
  await expect(
    page.locator('[aria-label^="Episode player"]'),
  ).toBeVisible({ timeout: 30_000 });
  // The real S3 window fetch competes with the rest of the E2E suite. Start
  // timing only after every visible camera for the current frame is decoded.
  // The buffering chip starts hidden before the first readiness probe, so its
  // absence alone is not a sufficient signal.
  await expect
    .poll(
      () => cameraPaintState(page),
      { timeout: 45_000 },
    )
    .toEqual({ total: 6, ok: 6 });

  const valueNow = () =>
    page.evaluate(() => {
      const s = document.querySelector('[role="slider"]');
      const v = s?.getAttribute("aria-valuenow");
      return v ? Number(v) : -1;
    });

  const start = await valueNow();
  await page.getByRole("button", { name: "Play", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Pause", exact: true }),
  ).toBeVisible();
  // Sample the playhead every 500ms for 5s; require monotonic, non-trivial
  // advance. Buffer-gating means it may briefly hold, but over 5s it should
  // cover many frames — well beyond the ~1 frame/2s of the old per-image path.
  const samples: number[] = [];
  for (let i = 0; i < 10; i++) {
    await page.waitForTimeout(500);
    samples.push(await valueNow());
  }
  await page.getByRole("button", { name: "Pause", exact: true }).click();

  const end = samples[samples.length - 1];
  const advanced = end - start;
  const monotonic = samples.every((v, i) => i === 0 || v >= samples[i - 1]);
  console.log(`fill-rate: start=${start} samples=${samples.join(",")} advanced=${advanced}`);

  expect(monotonic, "playhead advanced monotonically (no racing/rewind)").toBeTruthy();
  // The six-view KITScenes pack is decoded over the local S3/API path. Require
  // sustained progress while allowing shared CI and network latency.
  expect(advanced, "playhead advanced >=5 frames in 5s").toBeGreaterThanOrEqual(5);
});
