import { defineConfig } from '@playwright/test';

export default defineConfig({
  webServer: {
    command: 'bun run build && bun run serve',
    port: 3000,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
  testDir: './tests',
});
