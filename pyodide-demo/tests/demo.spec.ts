import { test, expect } from '@playwright/test';
import path from 'path';

const samplePath = path.resolve(new URL('./sample.txt', import.meta.url).pathname);

test('process file with pyodide', async ({ page }) => {
  test.setTimeout(180000);
  page.on('console', msg => console.log('PAGE LOG:', msg.text()));
  await page.goto('http://localhost:3000');
  await page.waitForTimeout(60000);
  const fileInput = page.locator('input[type="file"]');
  await fileInput.setInputFiles(samplePath);
  await expect(page.locator('p')).toHaveText('HELLO', { timeout: 30000 });
});
