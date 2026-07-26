import { expect, test, type Route } from "@playwright/test";

function fulfillJSON(route: Route, body: unknown) {
  return route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

function execution(id: string) {
  return {
    execution_id: id,
    workflow_name: "full_run",
    phase: "SUCCEEDED",
    started_at: "2026-07-15T00:00:00Z",
    duration_s: 12,
    inputs: {},
    outputs: {},
    nodes: [],
  };
}

test("Runs follows the Flyte continuation token", async ({ page }) => {
  const tokens: string[] = [];
  await page.route("**/api/v1/flyte/executions**", (route) => {
    const url = new URL(route.request().url());
    const token = url.searchParams.get("token") ?? "";
    tokens.push(token);
    return token === "flyte-2"
      ? fulfillJSON(route, { items: [execution("exec-2")] })
      : fulfillJSON(route, {
          items: [execution("exec-1")],
          next_page_token: "flyte-2",
        });
  });

  await page.goto("/runs");
  await expect(page.getByRole("link", { name: "exec-1" })).toBeVisible();
  await page.getByRole("button", { name: "Load more executions" }).click();
  await expect(page.getByRole("link", { name: "exec-2" })).toBeVisible();
  expect(tokens).toEqual(["", "flyte-2"]);
});

test("a rapid double request loads one Flyte continuation page", async ({
  page,
}) => {
  let secondPageRequests = 0;
  let releaseSecondPage!: () => void;
  const secondPageGate = new Promise<void>((resolve) => {
    releaseSecondPage = resolve;
  });

  await page.route("**/api/v1/flyte/executions**", async (route) => {
    const url = new URL(route.request().url());
    if (url.searchParams.get("token") === "flyte-2") {
      secondPageRequests += 1;
      await secondPageGate;
      return fulfillJSON(route, { items: [execution("exec-2")] });
    }
    return fulfillJSON(route, {
      items: [execution("exec-1")],
      next_page_token: "flyte-2",
    });
  });

  await page.goto("/runs");
  const loadMore = page.getByRole("button", {
    name: "Load more executions",
  });
  await expect(loadMore).toBeEnabled();
  await loadMore.evaluate((button: HTMLButtonElement) => {
    button.click();
    button.click();
  });
  await expect.poll(() => secondPageRequests).toBe(1);
  releaseSecondPage();
  await expect(page.getByRole("link", { name: "exec-2" })).toBeVisible();
  expect(secondPageRequests).toBe(1);
});
