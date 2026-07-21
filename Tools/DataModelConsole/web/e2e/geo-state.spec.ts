import { expect, test, type Page, type Route } from "@playwright/test";

function fulfillJSON(route: Route, body: unknown) {
  return route.fulfill({
    status: 200,
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

function dataset(name: string, version: string) {
  return {
    name,
    version,
    prefix: `${name}/${version}/shards/`,
  };
}

function datasetVersion(version: string) {
  return {
    version,
    total_samples: 10,
    shards: 1,
    episodes: 1,
    num_views: 1,
    has_map: false,
    has_world_model: false,
    has_gps: true,
    size_bytes: 1024,
    has_manifest: true,
  };
}

function geoStats(datasetName: string, version: string, samples: number) {
  return {
    dataset: datasetName,
    version,
    summary: {
      bbox: [10.99, 48.99, 11.02, 49.02],
      episode_count: 3,
      path_point_count: 40,
      sample_pose_count: samples,
      privacy: {
        k_anonymity: 5,
        endpoint_exclusion_frames: 10,
        heatmap_grid_degrees: 0.01,
      },
    },
    n_samples: samples,
  };
}

async function expectSelection(
  page: Page,
  datasetName: string,
  version: string,
  samples: number,
) {
  await expect(page.locator("#geo-dataset")).toHaveValue(datasetName);
  await expect(page.locator("#geo-version")).toHaveValue(version);
  await expect
    .poll(() => {
      const url = new URL(page.url());
      return [
        url.searchParams.get("dataset"),
        url.searchParams.get("version"),
      ];
    })
    .toEqual([datasetName, version]);
  await expect(
    page.getByText(samples.toLocaleString(), { exact: true }),
  ).toBeVisible();
}

test("versions API failure has an independent retry state", async ({ page }) => {
  let versionsHealthy = false;
  let datasetRequests = 0;
  let versionRequests = 0;
  let geoRequests = 0;

  await page.route("**/api/v1/**", (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/api/v1/datasets") {
      datasetRequests += 1;
      return fulfillJSON(route, {
        datasets: [dataset("alpha", "a1")],
      });
    }
    if (url.pathname === "/api/v1/datasets/alpha/versions") {
      versionRequests += 1;
      if (!versionsHealthy) {
        return route.fulfill({
          status: 502,
          contentType: "text/plain",
          body: "versions upstream unavailable",
        });
      }
      return fulfillJSON(route, {
        dataset: "alpha",
        versions: [datasetVersion("a1")],
      });
    }
    if (url.pathname === "/api/v1/datasets/alpha/geo-stats") {
      geoRequests += 1;
      return fulfillJSON(route, geoStats("alpha", "a1", 101));
    }
    return route.fulfill({ status: 404, body: "not mocked" });
  });

  await page.goto("/geo?dataset=alpha&version=a1");
  await expect(page.getByText(/An upstream service/)).toBeVisible();
  await expect(
    page.getByText("This dataset has no published GPS-enabled version."),
  ).toHaveCount(0);
  expect(geoRequests).toBe(0);

  const failedVersionRequests = versionRequests;
  const catalogRequestsBeforeRetry = datasetRequests;
  versionsHealthy = true;
  await page.getByRole("button", { name: "Retry" }).click();

  await expectSelection(page, "alpha", "a1", 101);
  expect(versionRequests).toBeGreaterThan(failedVersionRequests);
  expect(datasetRequests).toBe(catalogRequestsBeforeRetry);
  expect(geoRequests).toBeGreaterThan(0);
});

test("pending versions do not render the no-version state", async ({ page }) => {
  let releaseVersions!: () => void;
  const versionsGate = new Promise<void>((resolve) => {
    releaseVersions = resolve;
  });
  let versionRequests = 0;

  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/api/v1/datasets") {
      return fulfillJSON(route, {
        datasets: [dataset("alpha", "a1")],
      });
    }
    if (url.pathname === "/api/v1/datasets/alpha/versions") {
      versionRequests += 1;
      await versionsGate;
      return fulfillJSON(route, {
        dataset: "alpha",
        versions: [],
      });
    }
    return route.fulfill({ status: 404, body: "not mocked" });
  });

  await page.goto("/geo?dataset=alpha");
  await expect.poll(() => versionRequests).toBeGreaterThan(0);
  await expect(page.getByText("Loading dataset versions")).toBeVisible();
  await expect(
    page.getByText("This dataset has no published GPS-enabled version."),
  ).toHaveCount(0);

  releaseVersions();
  await expect(
    page.getByText("This dataset has no published GPS-enabled version."),
  ).toBeVisible();
});

test("dataset changes never request the new dataset with the old version", async ({
  page,
}) => {
  const geoRequests: string[] = [];

  await page.route("**/api/v1/**", (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/api/v1/datasets") {
      return fulfillJSON(route, {
        datasets: [dataset("alpha", "a-old"), dataset("beta", "b-new")],
      });
    }
    if (url.pathname === "/api/v1/datasets/alpha/versions") {
      return fulfillJSON(route, {
        dataset: "alpha",
        versions: [datasetVersion("a-old")],
      });
    }
    if (url.pathname === "/api/v1/datasets/beta/versions") {
      return fulfillJSON(route, {
        dataset: "beta",
        versions: [datasetVersion("b-new")],
      });
    }
    const match = url.pathname.match(
      /^\/api\/v1\/datasets\/([^/]+)\/geo-stats$/,
    );
    if (match) {
      const datasetName = decodeURIComponent(match[1]);
      const selectedVersion = url.searchParams.get("version") ?? "";
      geoRequests.push(`${datasetName}:${selectedVersion}`);
      return fulfillJSON(
        route,
        geoStats(
          datasetName,
          selectedVersion,
          datasetName === "alpha" ? 301 : 302,
        ),
      );
    }
    return route.fulfill({ status: 404, body: "not mocked" });
  });

  await page.goto("/geo?dataset=alpha&version=a-old");
  await expectSelection(page, "alpha", "a-old", 301);

  await page.locator("#geo-dataset").selectOption("beta");
  await expectSelection(page, "beta", "b-new", 302);

  expect(geoRequests).toContain("beta:b-new");
  expect(geoRequests).not.toContain("beta:a-old");
});

test("URL selection survives Back, Forward, and reload", async ({ page }) => {
  const sampleCounts = new Map([
    ["alpha:a1", 1101],
    ["alpha:a2", 1102],
    ["beta:b1", 1201],
  ]);

  await page.route("**/api/v1/**", (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/api/v1/datasets") {
      return fulfillJSON(route, {
        datasets: [dataset("alpha", "a2"), dataset("beta", "b1")],
      });
    }
    if (url.pathname === "/api/v1/datasets/alpha/versions") {
      return fulfillJSON(route, {
        dataset: "alpha",
        versions: [datasetVersion("a2"), datasetVersion("a1")],
      });
    }
    if (url.pathname === "/api/v1/datasets/beta/versions") {
      return fulfillJSON(route, {
        dataset: "beta",
        versions: [datasetVersion("b1")],
      });
    }
    const match = url.pathname.match(
      /^\/api\/v1\/datasets\/([^/]+)\/geo-stats$/,
    );
    if (match) {
      const datasetName = decodeURIComponent(match[1]);
      const selectedVersion = url.searchParams.get("version") ?? "";
      const samples = sampleCounts.get(`${datasetName}:${selectedVersion}`);
      if (samples === undefined) {
        return route.fulfill({ status: 400, body: "mixed selection" });
      }
      return fulfillJSON(
        route,
        geoStats(datasetName, selectedVersion, samples),
      );
    }
    return route.fulfill({ status: 404, body: "not mocked" });
  });

  await page.goto("/geo?dataset=alpha&version=a1");
  await expectSelection(page, "alpha", "a1", 1101);

  await page.locator("#geo-version").selectOption("a2");
  await expectSelection(page, "alpha", "a2", 1102);

  await page.locator("#geo-dataset").selectOption("beta");
  await expectSelection(page, "beta", "b1", 1201);

  await page.goBack();
  await expectSelection(page, "alpha", "a2", 1102);

  await page.goBack();
  await expectSelection(page, "alpha", "a1", 1101);

  await page.goForward();
  await expectSelection(page, "alpha", "a2", 1102);

  await page.goForward();
  await expectSelection(page, "beta", "b1", 1201);

  await page.reload();
  await expectSelection(page, "beta", "b1", 1201);
});
