import { expect, test, type Page, type Route } from "@playwright/test";

function fulfillJSON(route: Route, body: unknown, status = 200) {
  return route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

function oddResponse(where: { key: string; value: string }[]) {
  const urban = where.some((c) => c.value === "urban");
  return {
    dataset: "kitscenes",
    version: "v3.7",
    label_version: "v3.0",
    labelset_id: "oddls-0123456789abcdef0123456789abcdef",
    manifest_sha256: "a".repeat(64),
    quality: {
      structural_status: "passed",
      audit_status: "pending_human_audit",
      certification_status: "experimental",
    },
    labeled_scenes: 404,
    where,
    facets: [
      {
        key: "odd.road.context",
        values: urban
          ? [{ value: "urban", scenes: 1 }]
          : [
              { value: "urban", scenes: 1 },
              { value: "rural", scenes: 1 },
            ],
      },
    ],
    scenes: [
      {
        scene_uid: "uid-a",
        shard: "part-a-train-000000.tar",
        duration_ns: 20_000_000_000,
        distance_m: 120.4,
        matches: where.map((c) => ({ ...c, first_offset_ns: 2_000_000_000 })),
      },
      ...(urban
        ? []
        : [
            {
              scene_uid: "uid-b",
              shard: "part-b-train-000000.tar",
              duration_ns: 11_000_000_000,
              distance_m: 40,
              matches: [],
            },
          ]),
    ],
    page: { limit: 50, offset: 0, total: urban ? 1 : 2, more: false },
  };
}

async function mockAPI(page: Page, searches: URL[]) {
  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === "/api/v1/datasets") {
      return fulfillJSON(route, {
        datasets: [
          { name: "kitscenes-val", version: "v3.7", prefix: "kitscenes-val/v3.7/shards/" },
          { name: "kitscenes", version: "v3.7", prefix: "kitscenes/v3.7/shards/" },
        ],
      });
    }
    if (url.pathname === "/api/v1/odd/scenes") {
      searches.push(url);
      if (url.searchParams.get("dataset") !== "kitscenes") {
        return fulfillJSON(
          route,
          { error: "no ODD labels for this dataset version", code: "NOT_FOUND" },
          404,
        );
      }
      const where = url.searchParams.getAll("where").map((term) => {
        const [key, value] = term.split("=");
        return { key, value };
      });
      return fulfillJSON(route, oddResponse(where));
    }
    return fulfillJSON(route, { error: "unmocked", code: "NOT_FOUND" }, 404);
  });
}

test("ODD search shows the label version and drills down by facet", async ({
  page,
}) => {
  const searches: URL[] = [];
  await mockAPI(page, searches);
  await page.goto("/odd");

  await expect(page.getByText("2 of 404 scenes match.")).toBeVisible();
  await expect(page.getByText("kitscenes v3.0")).toBeVisible();
  await expect(page.getByText("experimental")).toBeVisible();
  const firstScene = page.getByRole("link", { name: "part-a-train-000000.tar" });
  await expect(firstScene).toHaveAttribute(
    "href",
    "/scenes/kitscenes/part-a-train-000000.tar/0?version=v3.7",
  );
  expect(searches[0].searchParams.get("version")).toBe("v3.7");

  await page.getByRole("button", { name: "Filter odd.road.context=urban" }).click();
  await expect(page).toHaveURL(/where=odd\.road\.context%3Durban/);
  await expect(page.getByText("1 of 404 scenes match.")).toBeVisible();
  await expect(page.getByText("odd.road.context=urban @ 2.0 s")).toBeVisible();
  expect(searches.at(-1)?.searchParams.getAll("where")).toEqual([
    "odd.road.context=urban",
  ]);

  await page
    .getByRole("button", { name: "Remove odd.road.context=urban" })
    .click();
  await expect(page.getByText("2 of 404 scenes match.")).toBeVisible();
  expect(searches.at(-1)?.searchParams.getAll("where")).toEqual([]);
});

test("ODD search says so when a dataset has no labels", async ({ page }) => {
  const searches: URL[] = [];
  await mockAPI(page, searches);
  await page.goto("/odd?dataset=kitscenes-val");

  await expect(
    page.getByText("No ODD labels for KITScenes Val v3.7."),
  ).toBeVisible();
  await page.getByLabel("Dataset").selectOption("kitscenes");
  await expect(page.getByText("2 of 404 scenes match.")).toBeVisible();
});

test("ODD search drops malformed and duplicate where terms", async ({
  page,
}) => {
  const searches: URL[] = [];
  await mockAPI(page, searches);
  await page.goto(
    "/odd?dataset=kitscenes&where=odd.road.context%3DUrban" +
      "&where=odd.road.context%3Durban&where=odd.road.context%3Durban",
  );

  await expect(page.getByText("1 of 404 scenes match.")).toBeVisible();
  expect(searches.at(-1)?.searchParams.getAll("where")).toEqual([
    "odd.road.context=urban",
  ]);
  await expect(
    page.getByRole("group", { name: "Active filters" }).getByRole("button", {
      name: "Remove odd.road.context=urban",
    }),
  ).toHaveCount(1);
});
