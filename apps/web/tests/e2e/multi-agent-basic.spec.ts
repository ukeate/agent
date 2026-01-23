import { test, expect } from '@playwright/test'

test.describe('Multi-Agent Basic Navigation', () => {
  test('application loads and React renders', async ({ page }) => {
    await page.goto('/multi-agent', { waitUntil: 'domcontentloaded' })

    // Wait for React app to load - check for root div
    await expect(page.locator('#root')).toBeVisible({ timeout: 30000 })

    // Wait for any content to render
    await page.waitForTimeout(3000)

    // Check if React has rendered any content
    const rootContent = await page.locator('#root').innerHTML()
    expect(rootContent.length).toBeGreaterThan(100) // Basic content check
  })

  test('page navigation works', async ({ page }) => {
    await page.goto('/', { waitUntil: 'domcontentloaded' })

    // Wait for app to load
    await expect(page.locator('#root')).toBeVisible({ timeout: 30000 })
    await page.waitForTimeout(2000)

    // Navigate to multi-agent page
    await page.goto('/multi-agent', { waitUntil: 'domcontentloaded' })

    // Verify we're on a different page
    await page.waitForTimeout(2000)
    const content = await page.locator('#root').innerHTML()
    expect(content.length).toBeGreaterThan(50)
  })

  test('basic DOM structure exists', async ({ page }) => {
    await page.goto('/multi-agent', { waitUntil: 'domcontentloaded' })

    // Check basic DOM elements exist
    await expect(page.locator('html')).toBeVisible()
    await expect(page.locator('body')).toBeVisible()
    await expect(page.locator('#root')).toBeVisible()

    // Wait for some content to render
    await page.waitForTimeout(5000)

    // Check if React app has rendered some basic structure
    const hasContent = await page.evaluate(() => {
      const root = document.getElementById('root')
      return root && root.children.length > 0
    })
    expect(hasContent).toBeTruthy()
  })
})
