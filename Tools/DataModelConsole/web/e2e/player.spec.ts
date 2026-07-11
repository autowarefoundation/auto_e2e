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
