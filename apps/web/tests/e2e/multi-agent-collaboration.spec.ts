import { test, expect } from '@playwright/test'
import { setupMultiAgentApiMocks } from './mocks/multi-agent-api'

test.describe('Multi-Agent Collaboration', () => {
  test('multi-agent page loads correctly', async ({ page }) => {
    await page.goto('/multi-agent')
    await page.waitForLoadState('domcontentloaded')
    
    // Wait for the multi-agent page to load - check for actual page elements
    await expect(page.locator('新建对话').or(page.locator('清空对话'))).toBeVisible({ timeout: 15000 })
    
    // Check for multi-agent specific elements
    await expect(page.locator('AI Agent').first()).toBeVisible({ timeout: 10000 })
    
    // Check if create conversation elements are present
    await expect(page.locator('text=创建').first()).toBeVisible({ timeout: 10000 })
  })

  test('basic interface elements are present', async ({ page }) => {
    await page.goto('/multi-agent')
    await page.waitForLoadState('domcontentloaded')
    
    // Wait for page to load - check for basic page structure
    await expect(page.locator('新建对话').or(page.locator('清空对话'))).toBeVisible({ timeout: 15000 })
    
    // Check if basic interface elements are present (without requiring specific content)
    const createElements = await page.locator('text=创建').all()
    expect(createElements.length).toBeGreaterThan(0)
  })

  test('page loads on mobile viewport', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 })
    await page.goto('/multi-agent')
    await page.waitForLoadState('domcontentloaded')
    
    // Check if mobile layout loads properly
    await expect(page.locator('h1:has-text("多智能体协作")')).toBeVisible({ timeout: 15000 })
    
    // Verify description is accessible on mobile
    await expect(page.locator('text=多个AI专家协作讨论')).toBeVisible({ timeout: 10000 })
  })

  test('handles API errors gracefully', async ({ page }) => {
    // Setup error mocks
    await page.route('**/api/v1/multi-agent/**', route => {
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: '服务器内部错误' })
      })
    })
    
    await page.goto('/multi-agent')
    await page.waitForLoadState('domcontentloaded')
    
    // Page should still load even if API fails
    await expect(page.locator('h1:has-text("多智能体协作")')).toBeVisible({ timeout: 15000 })
  })
})