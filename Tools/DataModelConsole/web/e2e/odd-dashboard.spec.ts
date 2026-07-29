import { expect, test, type Page, type Route } from "@playwright/test";

const PIXEL = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAEAAAAAkAQMAAAADwq7RAAAAIGNIUk0AAHomAACAhAAA+gAAAIDoAAB1MAAA6mAAADqYAAAXcJy6UTwAAAAGUExURTNBVf///753ZLcAAAABYktHRAH/Ai3eAAAAB3RJTUUH6gcPAQU1u04EUwAAAA1JREFUGNNjYBgFlAIAAUQAAS6fR94AAAAASUVORK5CYII=",
  "base64",
);

function fulfillJSON(route: Route, body: unknown) {
  return route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

const ontology = {
  schema_version: "odd_ontology_registry_v1",
  ontology_version: "odd_ontology_v1",
  ontology_sha256: "a".repeat(64),
  dataset_name: "kitscenes",
  dataset_version: "v3.0",
  labelset_id: "oddls-test",
  statuses: ["valid", "unavailable", "not_observable", "ambiguous"],
  sources: ["map_route", "gnss_ins", "vlm", "image_qc", "fusion"],
  labels: [
    {
      key: "odd.road.context",
      namespace: "odd",
      display_name: "Road Context",
      description: "Functional character of the road surroundings.",
      cardinality: "single",
      values: [{ value: "urban" }, { value: "suburban" }],
      primary_sources: ["map_route", "vlm"],
      backends: ["deterministic", "openai_compatible"],
      subject: "scene",
      temporal_scope: "interval",
      quality_tier: "experimental",
      dataset_support: {
        support_state: "supported_experimental",
        quality_tier: "experimental",
        valid_scene_count: 2,
        eligible_scene_count: 2,
        observable_scene_coverage: 1,
        attempted_count: 2,
        successful_count: 2,
      },
    },
  ],
};

const confidence = {
  observation_count: 2,
  duration_weighted_mean: 0.9,
  p10: 0.8,
  p50: 0.9,
  p90: 0.95,
  bins: [],
};

const interval = {
  lower: 0.1,
  upper: 0.9,
  method: "wilson_scene_95",
};

const statistics = {
  schema_version: "odd_statistics_v2",
  labelset_id: "oddls-test",
  scene_count: 2,
  scene_duration_ns: 200_000_000_000,
  scene_distance_m: 2_000,
  distance_weighting: {
    method_scene_counts: { duration_proportional: 2 },
    normalization: "per_scene_recorded_distance",
  },
  cooccurrences: { minimum_overlap_ns: 100_000_000, odd_pairs: [], odd_event: [] },
  keys: [
    {
      key: "odd.road.context",
      namespace: "odd",
      quality_tier: "experimental",
      valid_scene_count: 2,
      eligible_scene_count: 2,
      observable_scene_coverage: 1,
      eligible_duration_ns: 200_000_000_000,
      valid_duration_ns: 200_000_000_000,
      observable_duration_coverage: 1,
      eligible_distance_m: 2_000,
      valid_distance_m: 2_000,
      observable_distance_coverage: 1,
      valid_interval_count: 2,
      attempted_count: 2,
      successful_count: 2,
      conflict_count: 0,
      status_scene_counts: {
        valid: 2,
        unavailable: 0,
        not_observable: 0,
        ambiguous: 0,
      },
      status_duration_ns: {
        valid: 200_000_000_000,
        unavailable: 0,
        not_observable: 0,
        ambiguous: 0,
      },
      status_distance_m: {
        valid: 2_000,
        unavailable: 0,
        not_observable: 0,
        ambiguous: 0,
      },
      source_scene_counts: { vlm: 2 },
      source_duration_ns: { vlm: 200_000_000_000 },
      source_distance_m: { vlm: 2_000 },
      confidence,
      values: [
        {
          value: "urban",
          scene_count: 1,
          scene_ratio: 0.5,
          scene_ratio_ci95: interval,
          duration_ns: 100_000_000_000,
          duration_ratio: 0.5,
          duration_ratio_ci95: interval,
          distance_m: 1_000,
          distance_ratio: 0.5,
          distance_ratio_ci95: interval,
          valid_interval_count: 1,
          event_instance_count: 0,
          confidence,
        },
        {
          value: "suburban",
          scene_count: 0,
          scene_ratio: 0,
          scene_ratio_ci95: interval,
          duration_ns: 0,
          duration_ratio: 0,
          duration_ratio_ci95: interval,
          distance_m: 0,
          distance_ratio: 0,
          distance_ratio_ci95: interval,
          valid_interval_count: 0,
          event_instance_count: 0,
          confidence: { ...confidence, observation_count: 0 },
        },
      ],
    },
  ],
};

const labelsets = {
  dataset: "kitscenes",
  version: "v3.0",
  state: "ready",
  labelsets: [
    {
      schema_version: "odd_labelset_manifest_v1",
      status: "ready",
      labelset_id: "oddls-test",
      dataset_name: "kitscenes",
      dataset_version: "v3.0",
      dataset_manifest_uri: "s3://datasets/kitscenes/v3.0/manifest.json",
      dataset_manifest_sha256: "b".repeat(64),
      ontology_version: "odd_ontology_v1",
      ontology_sha256: "a".repeat(64),
      labeler_version: "odd_dataset_labeler_v1",
      labeler_image_digest: "c".repeat(64),
      labeler_source_revision: "d".repeat(40),
      publication_scope: "full",
      expected_scene_count: 2,
      scene_count: 2,
      openai_compatible: {
        model: "road-observer",
        model_revision: "revision-1",
        sampling: {
          regular_interval_s: 1,
          maximum_anchors: 128,
          trigger_context_s: 1,
        },
      },
      quality: {
        schema_version: "odd_quality_v1",
        structural_status: "passed",
        audit_status: "pending_human_audit",
        certification_status: "experimental",
      },
      artifacts: {
        ontology: {
          key: "odd/ontology.json",
          sha256: "a".repeat(64),
          byte_size: 100,
        },
        observations_parquet: {
          key: "odd/observations.parquet",
          sha256: "e".repeat(64),
          byte_size: 200,
          authoritative: true,
        },
      },
    },
  ],
};

const metricValues = {
  ade_1s_m: 1.1,
  ade_2s_m: 2.2,
  ade_3s_m: 3.3,
  ade_horizon_m: 4.4,
  fde_horizon_m: 10,
  acceleration_mae: 0.2,
  curvature_mae: 0.01,
};

const modelMetrics = {
  dataset: "kitscenes",
  version: "v3.0",
  labelset_id: "oddls-test",
  labelset_manifest_sha256: "f".repeat(64),
  projections: [
    {
      schema_version: "odd_model_metric_projection_v1",
      status: "ready",
      projection_id: "1".repeat(64),
      projection_policy_version: "odd_interval_projection_v1",
      metric_policy_version: "control_displacement_seed_mean_v1",
      frequency_hz: 10,
      horizon_steps: 64,
      horizon_seconds: 6.4,
      observation_join: "start <= anchor < end",
      event_join:
        "event_start < anchor + model_horizon and event_end > anchor",
      seed_aggregation: "arithmetic_mean",
      sample_uid_digest: "2".repeat(64),
      sample_count: 3820,
      scene_count: 40,
      samples_with_observations: 3800,
      samples_with_events: 120,
      overall: {
        sample_count: 3820,
        scene_count: 40,
        metrics: metricValues,
      },
      slices: [
        {
          kind: "observation",
          key: "odd.road.context",
          value: "urban",
          status: "valid",
          sample_count: 1200,
          scene_count: 30,
          metrics: {
            ...metricValues,
            ade_3s_m: 4.3,
            ade_horizon_m: 5.4,
            fde_horizon_m: 12,
          },
        },
        {
          kind: "observation",
          key: "odd.road.context",
          value: "suburban",
          status: "valid",
          sample_count: 500,
          scene_count: 15,
          metrics: {
            ...metricValues,
            ade_3s_m: 2.3,
            ade_horizon_m: 3.4,
            fde_horizon_m: 8,
          },
        },
      ],
      model: {
        artifact_sha256: "3".repeat(64),
        registered_model_name: "auto-e2e-driving-policy",
        model_version: 42,
        run_id: "run-42",
      },
      evaluation_dataset: {
        dataset: "kitscenes",
        version: "v2.2",
        manifest_uri: "s3://datasets/kitscenes/v2.2/manifest.json",
        manifest_sha256: "4".repeat(64),
        overlay_manifest_key: "overlays/model/manifest.json",
        overlay_manifest_sha256: "5".repeat(64),
        overlay_cache_identity: "6".repeat(64),
      },
      labelset: {
        dataset: "kitscenes",
        version: "v3.0",
        labelset_id: "oddls-test",
        manifest_key: "kitscenes/v3.0/odd/labelsets/oddls-test/manifest.json",
        manifest_sha256: "f".repeat(64),
        dataset_manifest_sha256: "b".repeat(64),
      },
      validation: {
        strategy: "exact_group_fraction",
        split_id: "split-v1",
        group_count: 40,
        sample_count: 3820,
        sample_uid_digest: "2".repeat(64),
      },
    },
  ],
};

const oddExecutions = [
  {
    execution_id: "odd-current",
    workflow_name: "odd-dataset-labeler",
    phase: "SUCCEEDED",
    started_at: "2026-07-29T00:00:00Z",
    duration_s: 3600,
  },
  {
    execution_id: "odd-previous",
    workflow_name: "wf_generate_odd_labelset",
    phase: "SUCCEEDED",
    started_at: "2026-07-28T00:00:00Z",
    duration_s: 7200,
  },
  {
    execution_id: "training-run",
    workflow_name: "wf_train_il",
    phase: "RUNNING",
    started_at: "2026-07-29T01:00:00Z",
    duration_s: 0,
  },
];

async function installODDDashboardRoutes(
  page: Page,
  searchRequests: unknown[] = [],
  options: {
    labelsets?: unknown;
    executions?: unknown[];
    capability?: {
      enabled: boolean;
      permitted: boolean;
      allow_full: boolean;
    };
    operationRequests?: Array<{
      path: string;
      body: unknown;
    }>;
  } = {},
) {
  await page.route("**/api/v1/flyte/executions?**", (route) =>
    fulfillJSON(route, { items: options.executions ?? oddExecutions }),
  );
  await page.route("**/api/v1/odd/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/api/v1/odd/operations") {
      return fulfillJSON(
        route,
        options.capability ?? {
          enabled: false,
          permitted: false,
          allow_full: false,
        },
      );
    }
    if (
      url.pathname === "/api/v1/odd/operations/launch" ||
      url.pathname === "/api/v1/odd/operations/retry"
    ) {
      options.operationRequests?.push({
        path: url.pathname,
        body: route.request().postDataJSON(),
      });
      return fulfillJSON(route, {
        action: url.pathname.endsWith("/retry") ? "retry" : "launch",
        execution_id: "odd-created",
        created: true,
      });
    }
    if (url.pathname === "/api/v1/odd/ontology") {
      return fulfillJSON(route, ontology);
    }
    if (url.pathname === "/api/v1/odd/statistics") {
      return fulfillJSON(route, statistics);
    }
    if (url.pathname === "/api/v1/odd/model-metrics") {
      return fulfillJSON(route, modelMetrics);
    }
    if (url.pathname === "/api/v1/odd/labelsets") {
      return fulfillJSON(route, options.labelsets ?? labelsets);
    }
    if (url.pathname === "/api/v1/odd/scenes/search") {
      if (route.request().method() === "POST") {
        searchRequests.push(route.request().postDataJSON());
      }
      return fulfillJSON(route, {
        dataset: "kitscenes",
        version: "v3.0",
        labelset_id: "oddls-test",
        manifest_sha256: "f".repeat(64),
        scenes: [
          {
            scene_uid: "scene-1",
            shard_name: "scene-1.tar",
            start_timestamp_ns: 0,
            end_timestamp_ns: 10_000_000_000,
            distance_m: 42,
            observations: [],
            matched: [
              {
                key: "odd.road.context",
                status: "valid",
                values: ["urban"],
                source: "fusion",
                confidence: 0.92,
                duration_ns: 5_000_000_000,
                first_timestamp_ns: 1_000_000_000,
              },
            ],
            matched_duration_ns: 5_000_000_000,
            match_confidence: 0.92,
            first_matched_timestamp_ns: 1_000_000_000,
            events: [
              {
                event_uid: "event-1",
                primary_event_key: "event.vehicle.interaction",
                primary_values: ["cut_in"],
                start_timestamp_ns: 2_000_000_000,
                end_timestamp_ns: 4_000_000_000,
                status: "valid",
                confidence: 0.9,
                actor_track_uids: ["vehicle-1"],
                outcome: "hazard_avoided",
              },
            ],
          },
        ],
        total: 1,
        limit: 50,
        offset: 0,
        more: false,
      });
    }
    return route.fulfill({ status: 404, body: "not mocked" });
  });
}

test("ODD Dashboard exposes weighted composition, structured search, ontology, and quality", async ({
  page,
}) => {
  const searchRequests: unknown[] = [];
  await installODDDashboardRoutes(page, searchRequests);
  await page.goto("/odd");

  await expect(page.getByRole("heading", { name: "ODD Dashboard" })).toBeVisible();
  await expect(page.getByText("Observable 100%")).toBeVisible();
  await page.getByRole("button", { name: "duration" }).click();
  await expect(page.getByRole("button", { name: "urban 50%" })).toContainText(
    "1.7 min",
  );
  await page.getByRole("button", { name: "urban 50%" }).click();
  await expect(page.getByText("1 matching scenes")).toBeVisible();
  await expect(page.getByRole("link", { name: "scene-1" })).toBeVisible();
  await expect(page.getByLabel("Event timeline")).toContainText("cut_in");

  await page.getByRole("button", { name: "Add predicate" }).click();
  await page.getByRole("button", { name: "or", exact: true }).click();
  await page.getByRole("button", { name: "Search scenes" }).click();
  await expect.poll(() => searchRequests.length).toBeGreaterThan(1);
  const latest = searchRequests.at(-1) as {
    query: { logic: string; predicates: unknown[] };
  };
  expect(latest.query.logic).toBe("or");
  expect(latest.query.predicates).toHaveLength(2);

  await page.getByRole("button", { name: "Ontology" }).click();
  await page.getByText("odd.road.context", { exact: true }).click();
  await expect(page.getByRole("button", { name: "suburban · 0" })).toBeVisible();
  await expect(page.getByText("supported_experimental").first()).toBeVisible();
  await expect(page.getByText("2 / 2 scenes observable")).toBeVisible();

  await page.getByRole("button", { name: "Model metrics" }).click();
  await expect(
    page.getByRole("combobox", { name: "Model projection" }),
  ).toHaveValue("1".repeat(64));
  await expect(page.getByText("validation only")).toBeVisible();
  await expect(page.getByText("kitscenes / v2.2")).toBeVisible();
  await expect(page.getByRole("link", { name: /Run run-42/ })).toHaveAttribute(
    "href",
    "/runs/run-42",
  );
  await expect(page.getByText("odd.road.context").first()).toBeVisible();
  await expect(page.getByText("+2.00 m")).toBeVisible();

  await page.getByRole("button", { name: "LabelSets" }).click();
  await expect(page.getByText("passed", { exact: true })).toBeVisible();
  await expect(page.getByText("pending_human_audit")).toBeVisible();
  await expect(page.getByText("experimental", { exact: true })).toBeVisible();
  await expect(page.getByText("superseded", { exact: true })).toBeVisible();
  await expect(page.getByText("training-run", { exact: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Run smoke" })).toHaveCount(0);
});

test("ODD Dashboard sends only permitted Dataset Labeler commands", async ({
  page,
}) => {
  const operationRequests: Array<{ path: string; body: unknown }> = [];
  await installODDDashboardRoutes(page, [], {
    capability: { enabled: true, permitted: true, allow_full: true },
    executions: [
      {
        execution_id: "odd-failed",
        workflow_name: "odd-dataset-labeler",
        phase: "FAILED",
        started_at: "2026-07-29T00:00:00Z",
        duration_s: 120,
      },
    ],
    operationRequests,
  });
  await page.goto("/odd?tab=labelsets");

  await page.getByRole("button", { name: "Run smoke" }).click();
  await expect.poll(() => operationRequests.length).toBe(1);
  expect(operationRequests[0]).toEqual({
    path: "/api/v1/odd/operations/launch",
    body: {
      dataset: "kitscenes",
      version: "v3.0",
      publication_scope: "smoke",
    },
  });

  await expect(page.getByRole("button", { name: "Run full" })).toBeVisible();
  await page.getByRole("button", { name: "Retry odd-failed" }).click();
  await expect.poll(() => operationRequests.length).toBe(2);
  expect(operationRequests[1]).toEqual({
    path: "/api/v1/odd/operations/retry",
    body: { execution_id: "odd-failed" },
  });
});

test("ODD Dashboard distinguishes running, failed, and not-started lifecycle states", async ({
  page,
}) => {
  const emptyCatalog = {
    dataset: "kitscenes",
    version: "v3.0",
    state: "not_started",
    labelsets: [],
  };
  await installODDDashboardRoutes(page, [], {
    labelsets: emptyCatalog,
    executions: [
      {
        execution_id: "odd-running",
        workflow_name: "odd-dataset-labeler",
        phase: "RUNNING",
        started_at: "2026-07-29T00:00:00Z",
        duration_s: 0,
      },
    ],
  });
  await page.goto("/odd");
  await expect(page.getByText("running", { exact: true }).first()).toBeVisible();

  await page.unrouteAll({ behavior: "wait" });
  await installODDDashboardRoutes(page, [], {
    executions: [
      {
        execution_id: "odd-failed",
        workflow_name: "odd-dataset-labeler",
        phase: "FAILED",
        started_at: "2026-07-29T00:00:00Z",
        duration_s: 120,
      },
    ],
  });
  await page.reload();
  await page.getByRole("button", { name: "LabelSets" }).click();
  await expect(page.getByText("failed", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("oddls-test", { exact: true })).toBeVisible();

  await page.unrouteAll({ behavior: "wait" });
  await installODDDashboardRoutes(page, [], {
    labelsets: emptyCatalog,
    executions: [],
  });
  await page.reload();
  await expect(page.getByText("not_started", { exact: true })).toBeVisible();
});

test("ODD Dashboard remains horizontally contained on mobile", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await installODDDashboardRoutes(page);
  await page.goto("/odd");
  await expect(page.getByRole("heading", { name: "ODD Dashboard" })).toBeVisible();

  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth - window.innerWidth,
    ),
  ).toBe(0);
});

test("ODD ontology deep links expand the selected scene label definition", async ({
  page,
}) => {
  await installODDDashboardRoutes(page);
  await page.goto(
    "/odd?dataset=kitscenes&version=v3.0&tab=ontology&key=odd.road.context",
  );

  const definition = page.locator("#ontology-odd\\.road\\.context");
  await expect(definition).toHaveAttribute("open", "");
  await expect(
    definition.getByRole("button", { name: "suburban · 0" }),
  ).toBeVisible();
  await expect(definition).toBeInViewport();
});

async function installODDSceneRoutes(page: Page) {
  await page.route("**/api/v1/**", (route) => {
    const url = new URL(route.request().url());
    if (
      url.pathname ===
      "/api/v1/datasets/kitscenes/shards/scene-1.tar/index"
    ) {
      return fulfillJSON(route, {
        fps: 10,
        version: "v3.0",
        shard: "scene-1.tar",
        samples: [0, 1].map((frame) => ({
          key: `sample-${frame}`,
          sample_uid: `sample-${frame}`,
          split_group_uid: "scene-1",
          split_bucket: 1,
          episode_id: "scene-1",
          frame_idx: frame,
          trip_frame: frame,
          members: { "cam_0.jpg": { offset: 512 + frame * 512, size: 200 } },
          ego_now: [5, 0, 0, 0],
          ego_history: Array.from({ length: 64 }, () => [5, 0, 0, 0]).flat(),
          ego_future: Array.from({ length: 64 }, () => [0, 0]).flat(),
          has_reasoning: false,
        })),
      });
    }
    if (url.pathname === "/api/v1/datasets/kitscenes/shards") {
      return fulfillJSON(route, {
        dataset: "kitscenes",
        shards: [
          {
            name: "scene-1.tar",
            key: "kitscenes/v3.0/shards/scene-1.tar",
            size_bytes: 2_000,
            last_modified: "2026-07-28T00:00:00Z",
          },
        ],
        page: { limit: 1000, offset: 0, total: 1, more: false },
      });
    }
    if (url.pathname === "/api/v1/odd/scenes/scene-1") {
      return fulfillJSON(route, {
        scene_uid: "scene-1",
        dataset_name: "kitscenes",
        dataset_version: "v3.0",
        start_timestamp_ns: 0,
        end_timestamp_ns: 200_000_000,
        distance_m: 42,
        provenance: {},
        evidence: [],
        events: [
          {
            event_uid: "event-1",
            scene_uid: "scene-1",
            start_timestamp_ns: 100_000_000,
            end_timestamp_ns: 200_000_000,
            primary_event_key: "event.ego.maneuver",
            actor_track_uids: [],
            observation_uids: ["oddobs-maneuver"],
            phases: [
              {
                phase: "active",
                start_timestamp_ns: 100_000_000,
                end_timestamp_ns: 200_000_000,
              },
            ],
            confidence: 0.9,
            status: "valid",
            supporting_evidence_uids: ["evidence-trajectory"],
            provenance: {},
          },
        ],
        observations: [
          {
            observation_uid: "oddobs-context",
            scene_uid: "scene-1",
            key: "odd.road.context",
            status: "valid",
            values: ["urban"],
            confidence: 0.9,
            source: "fusion",
            start_timestamp_ns: 0,
            end_timestamp_ns: 200_000_000,
            evidence_uids: ["evidence-map"],
            conflicting_evidence_uids: ["evidence-vlm"],
            measurements: {},
            provenance: {},
          },
          {
            observation_uid: "oddobs-route",
            scene_uid: "scene-1",
            key: "odd.route.action",
            status: "valid",
            values: ["turn_left"],
            confidence: 0.95,
            source: "map_route",
            start_timestamp_ns: 0,
            end_timestamp_ns: 200_000_000,
            evidence_uids: [],
            conflicting_evidence_uids: [],
            measurements: {},
            provenance: {},
          },
          {
            observation_uid: "oddobs-maneuver",
            scene_uid: "scene-1",
            key: "event.ego.maneuver",
            status: "valid",
            values: ["turn_left"],
            confidence: 0.9,
            source: "gnss_ins",
            start_timestamp_ns: 100_000_000,
            end_timestamp_ns: 200_000_000,
            evidence_uids: ["evidence-trajectory"],
            conflicting_evidence_uids: [],
            measurements: {},
            provenance: {},
            event_uid: "event-1",
          },
          {
            observation_uid: "oddobs-blur",
            scene_uid: "scene-1",
            key: "perception.image.blur",
            status: "not_observable",
            values: [],
            confidence: 0,
            source: "image_qc",
            start_timestamp_ns: 0,
            end_timestamp_ns: 200_000_000,
            evidence_uids: [],
            conflicting_evidence_uids: [],
            measurements: {},
            provenance: {},
          },
        ],
      });
    }
    if (
      url.pathname ===
      "/api/v1/scenes/scene-1/odd/evidence/oddobs-context"
    ) {
      const evidence = (uid: string, source: string, value: string) => ({
        schema_version: "odd_label_evidence_v1",
        evidence_uid: uid,
        label_key: "odd.road.context",
        cardinality: "single",
        values: [value],
        candidate_values: [],
        status: "valid",
        confidence: 0.9,
        source,
        scope: {
          dataset_name: "kitscenes",
          dataset_version: "v3.0",
          scene_uid: "scene-1",
          start_timestamp_ns: 0,
          end_timestamp_ns: 200_000_000,
          subject_type: "scene",
          camera_ids: [],
        },
        measurements: [],
        evidence_refs: [],
        provenance: {
          labeler_name: source === "map_route" ? "map_resolver" : "road_vlm",
          labeler_version: "v1",
          code_commit: "a".repeat(40),
          container_image_digest: `sha256:${"b".repeat(64)}`,
          config_sha256: "c".repeat(64),
          ontology_sha256: "d".repeat(64),
          input_artifact_sha256s: [],
          lookback_ns: 0,
          lookahead_ns: 0,
          details: {},
        },
      });
      return fulfillJSON(route, {
        dataset: "kitscenes",
        version: "v3.0",
        labelset_id: "oddls-test",
        scene_uid: "scene-1",
        observation: {},
        supporting_evidence: [evidence("evidence-map", "map_route", "urban")],
        conflicting_evidence: [evidence("evidence-vlm", "vlm", "suburban")],
        related_events: [],
        scene_provenance: {},
        manifest_sha256: "f".repeat(64),
      });
    }
    if (url.pathname.includes("/image/cam_0")) {
      return route.fulfill({
        status: 200,
        contentType: "image/png",
        body: PIXEL,
      });
    }
    if (url.pathname.endsWith("/overlay-models")) {
      return fulfillJSON(route, {
        dataset: "kitscenes",
        version: "v3.0",
        shard: "scene-1.tar",
        models: [],
      });
    }
    return route.fulfill({ status: 404, body: "not published" });
  });
}

test("Scene keeps ODD below Reasoning and supports evidence, event, and seek review", async ({
  page,
}) => {
  await installODDSceneRoutes(page);
  await page.goto("/scenes/kitscenes/scene-1.tar/0?version=v3.0");
  await expect(page.getByText("urban", { exact: true })).toBeVisible();
  await expect(page.getByText("Planned route")).toBeVisible();

  const ordered = await page
    .locator('[aria-label^="Episode player"]')
    .evaluate((player) => {
      const reasoning = Array.from(player.querySelectorAll("p")).find(
        (element) => element.textContent?.trim() === "Reasoning label",
      );
      const odd = Array.from(player.querySelectorAll("p")).find(
        (element) => element.textContent?.trim() === "ODD Labels",
      );
      return Boolean(
        reasoning &&
          odd &&
          reasoning.compareDocumentPosition(odd) &
            Node.DOCUMENT_POSITION_FOLLOWING,
      );
    });
  expect(ordered).toBe(true);

  await page.getByRole("button", { name: /odd.road.context/ }).click();
  await expect(page.getByText("Supporting evidence")).toBeVisible();
  await expect(page.getByText("Conflicting evidence")).toBeVisible();
  await expect(page.getByText(/map_route · map_resolver/)).toBeVisible();

  await page.getByRole("button", { name: "Whole scene" }).click();
  await page.getByRole("button", { name: "Events" }).click();
  await page.getByRole("button", { name: /event.ego.maneuver/ }).click();
  await expect(page.getByText(/frame 1\/1/)).toBeVisible();

  await page.getByRole("button", { name: "Perception" }).click();
  await expect(page.getByText("not_observable", { exact: true })).toBeVisible();
});
