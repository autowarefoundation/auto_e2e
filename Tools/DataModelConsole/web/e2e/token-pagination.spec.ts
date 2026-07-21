import { expect, test, type Route } from "@playwright/test";

function fulfillJSON(route: Route, body: unknown) {
  return route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

function experiment(id: string) {
  return {
    experiment_id: id,
    name: `Experiment ${id}`,
    artifact_location: `s3://models/${id}`,
    lifecycle_stage: "active",
    run_count: 0,
    last_update_time: 1_750_000_000_000,
  };
}

function run(experimentId: string, id: string) {
  return {
    run_id: id,
    run_name: `Run ${id}`,
    experiment_id: experimentId,
    status: "FINISHED",
    start_time: 1_750_000_000_000,
    end_time: 1_750_000_001_000,
    params: {},
    metrics: {},
  };
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

test("Models follows experiment and run continuation tokens", async ({
  page,
}) => {
  const experimentTokens: string[] = [];
  const runTokens: string[] = [];

  await page.route("**/api/v1/mlflow/**", (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/api/v1/mlflow/experiments") {
      const token = url.searchParams.get("page_token") ?? "";
      experimentTokens.push(token);
      return token === "experiments-2"
        ? fulfillJSON(route, { items: [experiment("b")] })
        : fulfillJSON(route, {
            items: [experiment("a")],
            next_page_token: "experiments-2",
          });
    }

    const match = url.pathname.match(
      /^\/api\/v1\/mlflow\/experiments\/([^/]+)\/runs$/,
    );
    if (!match) return route.fulfill({ status: 404 });
    const experimentId = decodeURIComponent(match[1]);
    const token = url.searchParams.get("page_token") ?? "";
    runTokens.push(token);
    return token === "runs-2"
      ? fulfillJSON(route, {
          items: [run(experimentId, "b-2")],
        })
      : fulfillJSON(route, {
          items: [run(experimentId, "b-1")],
          next_page_token: "runs-2",
        });
  });

  await page.goto("/models?experiment=b");
  await expect(
    page.getByRole("button", { name: /Experiment b/ }),
  ).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByText("Run b-1")).toBeVisible();
  expect(experimentTokens).toEqual(["", "experiments-2"]);

  await page.getByRole("button", { name: "Load more runs" }).click();
  await expect(page.getByText("Run b-2")).toBeVisible();
  expect(runTokens.at(-1)).toBe("runs-2");
  expect(runTokens.filter((token) => token === "runs-2")).toHaveLength(1);
  expect(runTokens.slice(0, -1).every((token) => token === "")).toBe(true);
});

test("Models drops a delayed run page after experiment navigation", async ({
  page,
}) => {
  let releaseStale!: () => void;
  const staleGate = new Promise<void>((resolve) => {
    releaseStale = resolve;
  });
  let staleRequested = false;

  await page.route("**/api/v1/mlflow/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/api/v1/mlflow/experiments") {
      return fulfillJSON(route, {
        items: [experiment("a"), experiment("b")],
      });
    }

    const match = url.pathname.match(
      /^\/api\/v1\/mlflow\/experiments\/([^/]+)\/runs$/,
    );
    if (!match) return route.fulfill({ status: 404 });
    const experimentId = decodeURIComponent(match[1]);
    const token = url.searchParams.get("page_token") ?? "";
    if (experimentId === "a" && token === "a-2") {
      staleRequested = true;
      await staleGate;
      return fulfillJSON(route, {
        items: [run("a", "stale-a-2")],
      });
    }
    if (experimentId === "a") {
      return fulfillJSON(route, {
        items: [run("a", "a-1")],
        next_page_token: "a-2",
      });
    }
    return fulfillJSON(route, {
      items: [run("b", "current-b-1")],
    });
  });

  await page.goto("/models?experiment=a");
  await expect(page.getByText("Run a-1")).toBeVisible();
  await page.getByRole("button", { name: "Load more runs" }).click();
  await expect.poll(() => staleRequested).toBe(true);

  await page.getByRole("button", { name: /Experiment b/ }).click();
  await expect(page.getByText("Run current-b-1")).toBeVisible();
  releaseStale();

  await page.waitForTimeout(100);
  await expect(page.getByText("Run stale-a-2")).toHaveCount(0);
  await expect(page.getByText("Run current-b-1")).toBeVisible();
});

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

test("pagination rejects token cycles and deduplicates entity IDs", async ({
  page,
}) => {
  const tokens: string[] = [];
  await page.route("**/api/v1/mlflow/**", (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/api/v1/mlflow/experiments") {
      const token = url.searchParams.get("page_token") ?? "";
      tokens.push(token);
      if (token === "a") {
        return fulfillJSON(route, {
          items: [experiment("a"), experiment("b")],
          next_page_token: "b",
        });
      }
      if (token === "b") {
        return fulfillJSON(route, {
          items: [experiment("c")],
          next_page_token: "a",
        });
      }
      return fulfillJSON(route, {
        items: [experiment("a")],
        next_page_token: "a",
      });
    }
    return fulfillJSON(route, { items: [] });
  });

  await page.goto("/models?experiment=missing");
  await expect
    .poll(() => tokens, { timeout: 5_000 })
    .toEqual(["", "a", "b"]);
  await expect(
    page.getByText("Upstream pagination token entered a cycle."),
  ).toHaveCount(1);
  await expect(
    page.getByRole("button", { name: /Experiment a/ }),
  ).toHaveCount(1);
  await expect(
    page.getByRole("button", { name: /Experiment b/ }),
  ).toHaveCount(1);
  await expect(
    page.getByRole("button", { name: /Experiment c/ }),
  ).toHaveCount(0);
  await page.waitForTimeout(100);
  expect(tokens).toEqual(["", "a", "b"]);
});

test("a rapid double request loads one continuation page", async ({ page }) => {
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

test("new frontend accepts the legacy array response during rollout", async ({
  page,
}) => {
  await page.route("**/api/v1/mlflow/**", (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/api/v1/mlflow/experiments") {
      return fulfillJSON(route, [experiment("legacy")]);
    }
    return fulfillJSON(route, [run("legacy", "legacy-run")]);
  });

  await page.goto("/models");
  await expect(
    page.getByRole("button", { name: /Experiment legacy/ }),
  ).toHaveAttribute("aria-pressed", "true");
  await expect(page.getByText("Run legacy-run")).toBeVisible();
});
