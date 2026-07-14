import { defineConfig } from "@playwright/test";

// E2E smoke tests assume the web server is already running on :3000 and the
// Go API on :8080 (reading real S3). We do not auto-start them here because the
// API needs AWS creds; start both manually, then `npx playwright test`.
export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  use: {
    baseURL: "http://localhost:3000",
    headless: true,
  },
});
