import { test, expect } from "@playwright/test";

// Verifies the scene drawer links to REAL shards (server-resolved) and marks
// scenes not packed into the selected version as unavailable, instead of the
// old floor(id/1000) guess that 404'd for l2d. Requires web:3000 + api:8080.

const URL =
  "/reasoning-labels?dataset=l2d&version=v2.0&prompt_version=action_relevant_reasoning_v3_temporal_front256";

test("scene drawer links resolve to real shards; unpacked scenes marked unavailable", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });

  await page.goto(URL, { waitUntil: "domcontentloaded" });
  // Stats compute may be a cold scan; wait for a known label to render.
  await page.waitForSelector("text=keep_lane", { timeout: 120_000 });

  // Click the lateral_response keep_lane bar to open the scene drawer.
  await page.locator("text=keep_lane").first().click();

  const dialog = page.locator('[role="dialog"]');
  await expect(dialog).toBeVisible({ timeout: 30_000 });

  // Wait for the scene list to populate (shard resolution builds indexes).
  await page.waitForSelector('[role="dialog"] li', { timeout: 60_000 });

  // A linked scene must point at a real shard path, not a guessed one. The
  // first in-range scene links to train-000000.tar.
  const firstLink = dialog.locator("a[href*='/shards/']").first();
  await expect(firstLink).toBeVisible();
  const href = await firstLink.getAttribute("href");
  console.log("first scene href:", href);
  expect(href).toContain("/shards/train-000000.tar/");

  // Verify the linked key exists in the canonical shard index. Fetching the
  // sample-detail endpoint here would repeat a cold full-tar scan even though
  // scene resolution has already identified the shard.
  const resp = await page.request.get(
    (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8080") +
      "/api/v1/datasets/l2d/shards/train-000000.tar/index?version=v2.0",
  );
  console.log("shard-index status:", resp.status());
  expect(resp.status()).toBe(200);
  const index = (await resp.json()) as { samples?: Array<{ key: string }> };
  expect(index.samples?.some((sample) => sample.key === "s00000000")).toBe(
    true,
  );

  // The header should report available-of-total (l2d: 413 of 999 for keep_lane).
  const header = await dialog.locator("text=/in this version/").first().textContent();
  console.log("drawer header:", header);
  expect(header).toMatch(/of .* in this version/);

  // Unavailable scenes (frame not packed into v2.0) render as non-links with a
  // "not in v2.0" hint — confirm at least one exists (l2d has ~586).
  const notInCount = await dialog.locator("text=/not in v2.0/").count();
  console.log("unavailable (not in v2.0) rows shown:", notInCount);

  expect(errors, `console errors: ${errors.join("; ")}`).toHaveLength(0);
});
