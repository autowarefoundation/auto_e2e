import { expect, test, type Route } from "@playwright/test";

const EXPERIMENT_A = "experiment-a";
const EXPERIMENT_B = "experiment-b";

type Phase = "initial" | "select-b" | "back-a" | "forward-b";

function fulfillJSON(route: Route, body: unknown) {
  return route.fulfill({
    status: 200,
    contentType: "application/json",
    headers: {
      "access-control-allow-origin": "*",
      "cache-control": "no-store",
    },
    body: JSON.stringify(body),
  });
}

function experiment(experimentId: string, name: string) {
  return {
    experiment_id: experimentId,
    name,
    artifact_location: `s3://models/${experimentId}`,
    lifecycle_stage: "active",
    run_count: 1,
    last_update_time: 1_750_000_000_000,
  };
}

function run(experimentId: string, runId: string, runName: string) {
  return {
    run_id: runId,
    run_name: runName,
    experiment_id: experimentId,
    status: "FINISHED",
    start_time: 1_750_000_000_000,
    end_time: 1_750_000_001_000,
    params: {},
    metrics: {},
  };
}

test("experiment selection follows URL history and ignores a stale run response", async ({
  page,
}) => {
  let phase: Phase = "initial";
  let releaseInitialA: () => void = () => {};
  const initialAGate = new Promise<void>((resolve) => {
    releaseInitialA = resolve;
  });
  let initialARequests = 0;
  let initialAResponses = 0;
  const runRequests: Array<{
    experiment: string;
    urlExperiment: string | null;
    phase: Phase;
  }> = [];

  await page.route("**/api/v1/mlflow/**", async (route) => {
    const requestURL = new URL(route.request().url());
    if (requestURL.pathname === "/api/v1/mlflow/experiments") {
      return fulfillJSON(route, {
        items: [
          experiment(EXPERIMENT_A, "Experiment A"),
          experiment(EXPERIMENT_B, "Experiment B"),
        ],
      });
    }

    const match = requestURL.pathname.match(
      /^\/api\/v1\/mlflow\/experiments\/([^/]+)\/runs$/,
    );
    if (!match) {
      return route.fulfill({ status: 404, body: "not mocked" });
    }

    const experimentId = decodeURIComponent(match[1]);
    const requestPhase = phase;
    runRequests.push({
      experiment: experimentId,
      urlExperiment: new URL(page.url()).searchParams.get("experiment"),
      phase: requestPhase,
    });

    if (experimentId === EXPERIMENT_A && requestPhase === "initial") {
      initialARequests++;
      await initialAGate;
      await fulfillJSON(route, {
        items: [run(EXPERIMENT_A, "run-a-stale", "Stale A run")],
      });
      initialAResponses++;
      return;
    }
    if (experimentId === EXPERIMENT_A && requestPhase === "back-a") {
      return fulfillJSON(route, {
        items: [run(EXPERIMENT_A, "run-a-current", "Current A run")],
      });
    }
    if (experimentId === EXPERIMENT_B && requestPhase === "select-b") {
      return fulfillJSON(route, {
        items: [run(EXPERIMENT_B, "run-b-selected", "Selected B run")],
      });
    }
    if (experimentId === EXPERIMENT_B && requestPhase === "forward-b") {
      return fulfillJSON(route, {
        items: [run(EXPERIMENT_B, "run-b-forward", "Forward B run")],
      });
    }

    return fulfillJSON(route, {
      items: [run(experimentId, "run-unexpected", "Unexpected run")],
    });
  });

  await page.goto("/models?experiment=missing");
  await expect(page).toHaveURL(
    new RegExp(`/models\\?experiment=${EXPERIMENT_A}$`),
  );

  const experimentA = page.getByRole("button", { name: /Experiment A/ });
  const experimentB = page.getByRole("button", { name: /Experiment B/ });
  const runsTitle = page
    .locator('[data-slot="card-title"]')
    .filter({ hasText: "Runs" });

  await expect(experimentA).toHaveAttribute("aria-pressed", "true");
  await expect(runsTitle).toContainText("Experiment A");
  await expect.poll(() => initialARequests).toBeGreaterThan(0);

  phase = "select-b";
  await experimentB.click();
  await expect(page).toHaveURL(
    new RegExp(`/models\\?experiment=${EXPERIMENT_B}$`),
  );
  await expect(experimentB).toHaveAttribute("aria-pressed", "true");
  await expect(runsTitle).toContainText("Experiment B");
  await expect(page.getByText("Selected B run")).toBeVisible();

  releaseInitialA();
  await expect
    .poll(() => initialARequests - initialAResponses)
    .toBe(0);
  await page.waitForTimeout(100);
  await expect(page.getByText("Stale A run")).toHaveCount(0);
  await expect(page.getByText("Selected B run")).toBeVisible();

  phase = "back-a";
  await page.goBack();
  await expect(page).toHaveURL(
    new RegExp(`/models\\?experiment=${EXPERIMENT_A}$`),
  );
  await expect(experimentA).toHaveAttribute("aria-pressed", "true");
  await expect(runsTitle).toContainText("Experiment A");
  await expect(page.getByText("Current A run")).toBeVisible();

  phase = "forward-b";
  await page.goForward();
  await expect(page).toHaveURL(
    new RegExp(`/models\\?experiment=${EXPERIMENT_B}$`),
  );
  await expect(experimentB).toHaveAttribute("aria-pressed", "true");
  await expect(runsTitle).toContainText("Experiment B");
  await expect(page.getByText("Forward B run")).toBeVisible();

  expect(
    runRequests.every(
      ({ experiment: requested, urlExperiment }) =>
        requested === urlExperiment,
    ),
    JSON.stringify(runRequests),
  ).toBe(true);
  expect(runRequests).toEqual(
    expect.arrayContaining([
      expect.objectContaining({
        experiment: EXPERIMENT_A,
        phase: "initial",
      }),
      expect.objectContaining({
        experiment: EXPERIMENT_B,
        phase: "select-b",
      }),
      expect.objectContaining({
        experiment: EXPERIMENT_A,
        phase: "back-a",
      }),
      expect.objectContaining({
        experiment: EXPERIMENT_B,
        phase: "forward-b",
      }),
    ]),
  );
});
