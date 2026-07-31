import { test, expect } from "@playwright/test";

// Verifies carried reasoning labels resolve to real shards in the latest pack
// instead of retaining source-version shard coordinates.

const PAGE_URL =
  "/reasoning-labels?dataset=kitscenes&version=v3.1&prompt_version=action_relevant_reasoning_v3_temporal_front256";

test("carried reasoning labels resolve to real shards in the latest pack", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });

  await page.goto(PAGE_URL, { waitUntil: "domcontentloaded" });
  // Stats compute may be a cold scan; wait for a known label to render.
  await page.waitForSelector("text=keep_lane", { timeout: 120_000 });
  await expect(page.getByText(/Compatible labels from v3\.0/)).toBeVisible();

  // Click the lateral_response keep_lane bar to open the scene drawer.
  await page.locator("text=keep_lane").first().click();

  const dialog = page.locator('[role="dialog"]');
  await expect(dialog).toBeVisible({ timeout: 30_000 });

  // Wait for the scene list to populate (shard resolution builds indexes).
  await page.waitForSelector('[role="dialog"] li', { timeout: 60_000 });

  // A linked scene must use the latest pack coordinate, not the artifact
  // source version's shard name.
  const firstLink = dialog.locator("a[href*='/shards/']").first();
  await expect(firstLink).toBeVisible();
  const href = await firstLink.getAttribute("href");
  console.log("first scene href:", href);
  expect(href).toContain("/datasets/kitscenes/shards/");
  expect(href).toContain("version=v3.1");

  const linkedURL = new URL(href!, "http://localhost");
  const path = linkedURL.pathname.split("/");
  const shard = decodeURIComponent(path[4]);
  const sampleKey = decodeURIComponent(path[6]);
  const resp = await page.request.get(
    (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8080") +
      `/api/v1/datasets/kitscenes/shards/${encodeURIComponent(shard)}/index?version=v3.1`,
  );
  console.log("shard-index status:", resp.status());
  expect(resp.status()).toBe(200);
  const index = (await resp.json()) as { samples?: Array<{ key: string }> };
  expect(index.samples?.some((sample) => sample.key === sampleKey)).toBe(true);

  // The header reports source labels that still resolve in the latest pack.
  const header = await dialog.locator("text=/in this version/").first().textContent();
  console.log("drawer header:", header);
  expect(header).toMatch(/of .* in this version/);

  expect(errors, `console errors: ${errors.join("; ")}`).toHaveLength(0);
});
