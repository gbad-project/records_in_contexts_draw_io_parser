import { test, expect } from '@playwright/test';
import path from 'path';

const samplePath = path.resolve(new URL('./sample.txt', import.meta.url).pathname);

test('process file with pyodide - debug version', async ({ page }) => {
  test.setTimeout(180000);

  // Capture all console logs and errors
  page.on('console', msg => console.log('PAGE LOG:', msg.text()));
  page.on('pageerror', error => console.log('PAGE ERROR:', error.message));
  page.on('requestfailed', request => {
    console.log('REQUEST FAILED:', request.url(), request.failure()?.errorText);
  });

  await page.goto('http://localhost:3000');

  // Check if page loaded properly
  await expect(page.locator('h1')).toHaveText('File Upload + Pyodide Processing');
  console.log('✓ Page loaded successfully');

  // Wait for Pyodide to load and check if it's available
  console.log('Waiting for Pyodide to load…');
  await page.waitForTimeout(60000);

  // Check if Pyodide loaded successfully
  const pyodideReady = await page.evaluate(() => (window as any).pyodideReady);
  console.log('Pyodide ready status:', pyodideReady);

  if (!pyodideReady) {
    // Try to get more info about what went wrong
    const windowLoadPyodide = await page.evaluate(() => typeof (window as any).loadPyodide);
    console.log('window.loadPyodide type:', windowLoadPyodide);

    // Check for any network errors in the browser
    const networkErrors = await page.evaluate(() => {
      return (window as any).networkErrors || 'No network errors captured';
    });
    console.log('Network errors:', networkErrors);
  }

  const fileInput = page.locator('input[type="file"]');
  await fileInput.setInputFiles(samplePath);
  console.log('✓ File uploaded');

  // Wait a bit and check what’s actually rendered
  await page.waitForTimeout(5000);

  // Debug: Check what content is actually on the page
  const bodyContent = await page.locator('body').textContent();
  console.log('Full page content:', bodyContent);

  // Check if there are any error messages
  const errorElements = await page.locator('[class*="error"], .error, #error').count();
  console.log('Error elements found:', errorElements);

  // Look for the processed content container
  const processedDiv = page.locator('div:has(> h2:text("Processed Content:"))');
  const hasProcessedDiv = await processedDiv.count();
  console.log('Processed content div found:', hasProcessedDiv > 0);

  if (hasProcessedDiv > 0) {
    const pContent = await processedDiv.locator('p').first().textContent();
    console.log('Actual processed content:', pContent);
  }

  // Try the original test but with better error reporting
  try {
    await expect(processedDiv.locator('p')).toHaveText('HELLO', { timeout: 30000 });
    console.log('✓ Test passed!');
  } catch (error) {
    console.log('❌ Test failed. Let\'s see what we have instead:');

    // Count all p elements
    const pCount = await page.locator('p').count();
    console.log('Number of p elements found:', pCount);

    // Get text of all p elements
    for (let i = 0; i < pCount; i++) {
      const text = await page.locator('p').nth(i).textContent();
      console.log(`p[${i}] content:`, text);
    }

    throw error;
  }
});
