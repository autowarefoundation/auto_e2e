// Player smoke test. Requires the dev/prod web server on :3000 and the Go API
// on :8080 (reading real S3, AWS_PROFILE=autowarefoundation). Uses the NVIDIA
// scene because its camera JPEGs are real; the L2D shard's camera frames are
// known-black stale data.
//
// Run: (servers up) npx playwright test
import { test, expect } from "@playwright/test";

const SCENE = "/scenes/nvidia_av/train-000000.tar/0";

test("player renders real camera pixels, advances, and focuses", async ({ page }) => {
  const consoleErrors: string[] = [];
  page.on("console", (m) => {
    if (m.type() === "error") consoleErrors.push(m.text());
  });
  page.on("pageerror", (e) => consoleErrors.push(`pageerror: ${e.message}`));

  await page.goto(SCENE, { waitUntil: "networkidle" });
  // Let the FrameStore fetch + canvases paint.
  await page.waitForTimeout(3500);

  // Every camera canvas must have non-blank pixels (real frame, not black).
  const painted = await page.evaluate(() => {
    const canvases = Array.from(document.querySelectorAll("canvas"));
    let ok = 0;
    for (const c of canvases) {
      const ctx = c.getContext("2d");
      if (!ctx || c.width === 0) continue;
      const { data } = ctx.getImageData(0, 0, c.width, c.height);
      let sum = 0;
      for (let i = 0; i < data.length; i += 4) sum += data[i] + data[i + 1] + data[i + 2];
      if (sum / (data.length / 4) / 3 > 2) ok++;
    }
    return { total: canvases.length, ok };
  });
  expect(painted.total).toBeGreaterThanOrEqual(7);
  expect(painted.ok).toBe(painted.total);

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
});

// Fill-rate regression: windowed contiguous fetch must let the buffer advance
// at roughly real time. Before it, the player made ~6 tiny range GETs per
// frame and the buffer filled well below 10Hz over a high-latency link, so
// playback stalled for a long time before it looked smooth. We measure how far
// the playhead advances over a fixed wall-clock window after pressing play.
test("playback fills its buffer near real time (windowed fetch)", async ({
  page,
}) => {
  await page.goto(SCENE, { waitUntil: "networkidle" });
  // Warm the first window so the first frame is decodable, then play.
  await page.waitForTimeout(3000);
  await page.locator('[aria-label^="Episode player"]').focus();

  const valueNow = () =>
    page.evaluate(() => {
      const s = document.querySelector('[role="slider"]');
      const v = s?.getAttribute("aria-valuenow");
      return v ? Number(v) : -1;
    });

  const start = await valueNow();
  await page.keyboard.press("Space");
  // Sample the playhead every 500ms for 5s; require monotonic, non-trivial
  // advance. Buffer-gating means it may briefly hold, but over 5s it should
  // cover many frames — well beyond the ~1 frame/2s of the old per-image path.
  const samples: number[] = [];
  for (let i = 0; i < 10; i++) {
    await page.waitForTimeout(500);
    samples.push(await valueNow());
  }
  await page.keyboard.press("Space");

  const end = samples[samples.length - 1];
  const advanced = end - start;
  const monotonic = samples.every((v, i) => i === 0 || v >= samples[i - 1]);
  console.log(`fill-rate: start=${start} samples=${samples.join(",")} advanced=${advanced}`);

  expect(monotonic, "playhead advanced monotonically (no racing/rewind)").toBeTruthy();
  // Over 5s of wall clock, expect at least 20 frames (2 fps) advanced — a low
  // bar the old path failed and the windowed path clears with large margin.
  expect(advanced, "playhead advanced ≥20 frames in 5s").toBeGreaterThanOrEqual(20);
});
