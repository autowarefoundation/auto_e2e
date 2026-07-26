import { expect, test, type Page, type Route } from "@playwright/test";

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

const DAY_MS = 24 * 60 * 60 * 1000;
const NOW = Date.now();

const experiments = [
  {
    run_id: "run-eval",
    run_name: "route-on-evaluation",
    experiment_id: "11",
    experiment_name: "auto-e2e",
    mlflow_status: "FINISHED",
    start_time: NOW - DAY_MS,
    end_time: NOW - DAY_MS + 600_000,
    dataset: "s3://auto-e2e/PhysicalAI-AV",
    dataset_version: "v3.0",
    data_fingerprint: "fingerprint-v3-full",
    validation_scope: "full",
    validation_split_id: "split-v3-full",
    backbone: "AutoE2E",
    fusion_mode: "cross-attention",
    route_conditioning: true,
    seed: "42",
    epochs: "10",
    epochs_completed: "10",
    lineage_status: "complete",
    primary_execution_id: "flyte-eval",
    primary_execution_url:
      "https://flyte.example/console/projects/auto-e2e/domains/development/executions/flyte-eval",
    mlflow_url: "https://mlflow.example/#/experiments/11/runs/run-eval",
    evaluation: { ade: 3.51344, fde: 10.18399, gate_pass: true },
    validation: { ade: 3.8395, fde: 10.9898 },
    train_execution: {
      execution_id: "flyte-train-eval",
      workflow_name: "full_run",
      phase: "SUCCEEDED",
      started_at: "2026-07-15T00:00:00Z",
      duration_s: 3600,
    },
    eval_execution: {
      execution_id: "flyte-eval",
      workflow_name: "evaluate_model",
      phase: "SUCCEEDED",
      started_at: "2026-07-15T02:00:00Z",
      duration_s: 900,
    },
    model_versions: [
      {
        name: "AutoE2E",
        version: "42",
        role: "best",
        status: "READY",
        url: "https://mlflow.example/#/models/AutoE2E/versions/42",
      },
    ],
    params: {
      "train/lr": "0.0003",
      "train/batch_size": "16",
      "train/weight_decay": "0.01",
      "ctx/train_docker_image": "training@sha256:111111111111",
    },
    tags: {
      "ctx/eval_docker_image": "evaluation@sha256:222222222222",
      "best_checkpoint_sha256": "aaaaaaaaaaaaaaaa",
      "final_checkpoint_sha256": "bbbbbbbbbbbbbbbb",
    },
    metrics: {
      "eval/navigation/collision_rate": 0.025,
      "eval/navigation/route_completion": 0.91,
    },
  },
  {
    run_id: "run-validation",
    run_name: "route-off-validation",
    experiment_id: "11",
    experiment_name: "auto-e2e",
    mlflow_status: "FINISHED",
    start_time: NOW - 3 * DAY_MS,
    end_time: NOW - 3 * DAY_MS + 600_000,
    dataset: "s3://auto-e2e/PhysicalAI-AV",
    dataset_version: "v3.0",
    data_fingerprint: "fingerprint-v3-full",
    validation_scope: "full",
    validation_split_id: "split-v3-full",
    backbone: "AutoE2E",
    fusion_mode: "cross-attention",
    route_conditioning: false,
    seed: "42",
    epochs: "10",
    epochs_completed: "10",
    lineage_status: "complete",
    primary_execution_id: "flyte-validation",
    primary_execution_url:
      "https://flyte.example/console/projects/auto-e2e/domains/development/executions/flyte-validation",
    mlflow_url:
      "https://mlflow.example/#/experiments/11/runs/run-validation",
    validation: { ade: 3.9049, fde: 11.4284 },
    train_execution: {
      execution_id: "flyte-validation",
      workflow_name: "full_run",
      phase: "SUCCEEDED",
      started_at: "2026-07-14T00:00:00Z",
      duration_s: 3600,
    },
    model_versions: [],
    params: { "train/lr": "0.0003" },
    tags: {},
    metrics: {},
  },
  {
    run_id: "run-failed",
    run_name: "kitscenes-smoke-failed",
    experiment_id: "11",
    experiment_name: "auto-e2e",
    mlflow_status: "FAILED",
    start_time: NOW - 20 * DAY_MS,
    end_time: NOW - 20 * DAY_MS + 100_000,
    dataset: "s3://auto-e2e/KITScenes",
    dataset_version: "v2.1",
    data_fingerprint: "fingerprint-kitscenes-smoke",
    validation_scope: "subset",
    validation_split_id: "split-kitscenes-smoke",
    backbone: "AutoE2E",
    route_conditioning: true,
    seed: "7",
    epochs: "2",
    lineage_status: "partial",
    primary_execution_id: "flyte-failed",
    primary_execution_url:
      "https://flyte.example/console/projects/auto-e2e/domains/development/executions/flyte-failed",
    mlflow_url: "https://mlflow.example/#/experiments/11/runs/run-failed",
    train_execution: {
      execution_id: "flyte-failed",
      workflow_name: "full_run",
      phase: "FAILED",
      started_at: "2026-07-13T00:00:00Z",
      duration_s: 100,
    },
    model_versions: [],
    params: {},
    tags: {},
    metrics: {},
  },
  {
    run_id: "run-full-failed",
    run_name: "kitscenes-full-failed",
    experiment_id: "11",
    experiment_name: "auto-e2e",
    mlflow_status: "FAILED",
    start_time: NOW - 20 * DAY_MS,
    end_time: NOW - 20 * DAY_MS + 100_000,
    dataset: "s3://auto-e2e/KITScenes",
    dataset_version: "v3.0",
    data_fingerprint: "fingerprint-kitscenes-full",
    validation_scope: "full",
    validation_split_id: "split-kitscenes-full",
    backbone: "AutoE2E",
    route_conditioning: true,
    seed: "7",
    epochs: "2",
    lineage_status: "partial",
    primary_execution_id: "flyte-full-failed",
    primary_execution_url:
      "https://flyte.example/console/projects/auto-e2e/domains/development/executions/flyte-full-failed",
    mlflow_url: "https://mlflow.example/#/experiments/11/runs/run-full-failed",
    train_execution: {
      execution_id: "flyte-full-failed",
      workflow_name: "full_run",
      phase: "FAILED",
      started_at: "2026-07-13T00:00:00Z",
      duration_s: 100,
    },
    model_versions: [],
    params: {},
    tags: {},
    metrics: {},
  },
  {
    run_id: "run-unlinked",
    run_name: "legacy-l2d-run",
    experiment_id: "9",
    experiment_name: "legacy",
    mlflow_status: "FINISHED",
    start_time: NOW - 120 * DAY_MS,
    end_time: NOW - 120 * DAY_MS + 100_000,
    dataset: "s3://auto-e2e/L2D",
    dataset_version: "v1.0",
    validation_split_id: "split-l2d-smoke",
    backbone: "legacy",
    lineage_status: "missing",
    mlflow_url: "https://mlflow.example/#/experiments/9/runs/run-unlinked",
    validation: { ade: 5.25, fde: 15.5 },
    model_versions: [],
    params: {},
    tags: {},
    metrics: {},
  },
];

async function mockExperiments(page: Page) {
  await page.route("**/api/v1/experiments", (route) =>
    fulfillJSON(route, {
      generated_at: "2026-07-26T00:00:00Z",
      summary: {
        total: 5,
        running: 1,
        failed: 2,
        evaluated: 1,
        registered: 1,
        unlinked: 3,
      },
      experiments,
    }),
  );
}

test("shows joined results, source links, and complete run details", async ({
  page,
}) => {
  await mockExperiments(page);
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto("/experiments");

  await expect(
    page.getByRole("heading", { name: "Experiments", exact: true }),
  ).toBeVisible();
  await expect(page.getByText("3 results", { exact: true })).toBeVisible();
  await expect(
    page.getByRole("img", { name: "evaluation ADE trend" }),
  ).toBeVisible();

  const table = page.getByRole("table");
  await expect(
    page.getByRole("combobox", { name: "Filter by validation scope" }),
  ).toHaveCount(0);
  await expect(table).not.toContainText("flyte-failed");
  await expect(table).not.toContainText("run-unlinked");
  const evaluationRow = table
    .getByRole("row")
    .filter({ hasText: "flyte-eval" });
  const validationRow = table
    .getByRole("row")
    .filter({ hasText: "flyte-validation" });

  await expect(evaluationRow).toContainText("Eval");
  await expect(evaluationRow).toContainText("ADE3.513 m");
  await expect(evaluationRow).toContainText("FDE10.184 m");
  await expect(validationRow).toContainText("Val");
  await expect(validationRow).toContainText("ADE3.905 m");
  await expect(validationRow).toContainText("FDE11.428 m");
  await expect(validationRow).toContainText("EVAL PENDING");

  await expect(evaluationRow.getByRole("link", { name: /Flyte/ })).toHaveAttribute(
    "href",
    experiments[0].primary_execution_url,
  );
  await expect(
    evaluationRow.getByRole("link", { name: /MLflow/ }),
  ).toHaveAttribute("href", experiments[0].mlflow_url);
  await expect(evaluationRow.getByRole("link", { name: /v42/ })).toHaveAttribute(
    "href",
    experiments[0].model_versions[0].url,
  );

  await evaluationRow.click();
  await expect(page).toHaveURL(/\/experiments\?run=run-eval$/);
  const details = page.getByRole("dialog", { name: "PhysicalAI AV v3.0" });
  await expect(details).toContainText("Evaluation");
  await expect(details).toContainText("ADE 3.513 m");
  await expect(details).toContainText("Validation");
  await expect(details).toContainText("ADE 3.840 m");
  await expect(details).toContainText("collision_rate");
  await expect(details).toContainText("Quality gate PASS");
  await expect(details).toContainText("What changed");
  await expect(details).toContainText("OfftoOn");
  await expect(details).toContainText("Training pipeline");
  await expect(details).toContainText("Evaluation");
  await expect(details).toContainText("training@sha256:111111111111");
  await expect(details).toContainText("aaaaaaaaaaaaaaaa");
  await expect(details).toContainText("flyte-eval");
  await expect(details).toContainText("run-eval");
  await expect(details).toContainText("v42 · best");
  await expect(
    details.getByRole("link", { name: "Open run artifacts in MLflow" }),
  ).toHaveAttribute("href", experiments[0].mlflow_url);

  await details
    .getByRole("button", { name: "Close experiment details" })
    .click();
  await expect(page).toHaveURL(/\/experiments$/);
});

test("filters experiments by dataset, status, and identifiers", async ({
  page,
}) => {
  await mockExperiments(page);
  await page.goto("/experiments");
  await expect(page.getByText("3 results", { exact: true })).toBeVisible();

  await page
    .getByRole("combobox", { name: "Filter by dataset", exact: true })
    .selectOption("s3://auto-e2e/PhysicalAI-AV");
  await expect(page.getByText("2 results", { exact: true })).toBeVisible();
  await expect(page.getByRole("table")).not.toContainText("KITScenes");

  await page
    .getByRole("combobox", { name: "Filter by status" })
    .selectOption("succeeded");
  await expect(page.getByText("1 results", { exact: true })).toBeVisible();
  await expect(page.getByRole("table")).toContainText("flyte-eval");
  await expect(page.getByRole("table")).not.toContainText("flyte-validation");

  await page.getByRole("button", { name: "Reset filters" }).click();
  await page
    .getByRole("searchbox", { name: "Search experiments" })
    .fill("flyte-full-failed");
  await expect(page.getByText("1 results", { exact: true })).toBeVisible();
  await expect(page.getByRole("table")).toContainText("KITScenes");
  await expect(page.getByRole("table")).toContainText("FAILED");

  await page.getByRole("button", { name: "Reset filters" }).click();
  await page
    .getByRole("combobox", { name: "Filter by status" })
    .selectOption("unlinked");
  await expect(page.getByText("1 results", { exact: true })).toBeVisible();
  await expect(page.getByRole("table")).toContainText("PARTIAL");
  await expect(page.getByRole("table")).toContainText("flyte-full-failed");
  await expect(page.getByRole("table")).not.toContainText("run-unlinked");
});

test("filters experiments by a recent time window", async ({ page }) => {
  await mockExperiments(page);
  await page.goto("/experiments");

  await page
    .getByRole("combobox", { name: "Filter by period" })
    .selectOption("7");
  await expect(page.getByText("2 results", { exact: true })).toBeVisible();
  await expect(page.getByRole("table")).toContainText("flyte-eval");
  await expect(page.getByRole("table")).toContainText("flyte-validation");
  await expect(page.getByRole("table")).not.toContainText("flyte-failed");

  await page
    .getByRole("combobox", { name: "Filter by period" })
    .selectOption("30");
  await expect(page.getByText("3 results", { exact: true })).toBeVisible();
  await expect(page.getByRole("table")).toContainText("flyte-full-failed");
});

test("warns when selected experiment results are not comparable", async ({
  page,
}) => {
  await mockExperiments(page);
  await page.goto("/experiments");

  await page.getByRole("checkbox", { name: "Compare route-on-evaluation" }).check();
  await page
    .getByRole("checkbox", { name: "Compare kitscenes-full-failed" })
    .check();
  await expect(page.getByText("2/3 selected", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Compare", exact: true }).click();

  const comparison = page.getByRole("dialog", { name: "Compare experiments" });
  await expect(comparison).toContainText("route-on-evaluation");
  await expect(comparison).toContainText("kitscenes-full-failed");
  await expect(comparison).toContainText(
    "Dataset fingerprint or validation split differs. Scores are not directly comparable.",
  );
  await expect(comparison).toContainText("Eval ADE");
  await expect(comparison).toContainText("3.513 m");
});

test("mobile experiment list and details do not overflow the viewport", async ({
  page,
}) => {
  await mockExperiments(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/experiments");

  await expect(page.locator("article")).toHaveCount(3);
  await expect(page.getByRole("table")).toBeHidden();
  await expect(
    page.locator("article").filter({ hasText: "flyte-validation" }),
  ).toContainText("Val");
  await expect
    .poll(() =>
      page.evaluate(
        () => document.documentElement.scrollWidth <= document.documentElement.clientWidth,
      ),
    )
    .toBe(true);

  await page.locator("article").filter({ hasText: "flyte-eval" }).click();
  const details = page.getByRole("dialog", { name: "PhysicalAI AV v3.0" });
  await expect(details).toBeVisible();
  await expect
    .poll(() =>
      page.evaluate(
        () => document.documentElement.scrollWidth <= document.documentElement.clientWidth,
      ),
    )
    .toBe(true);
});

test("legacy models URL redirects to the experiments workspace", async ({
  page,
}) => {
  await mockExperiments(page);
  await page.goto("/models");
  await expect(page).toHaveURL(/\/experiments$/);
  await expect(
    page.getByRole("heading", { name: "Experiments", exact: true }),
  ).toBeVisible();
});
