import { test, expect } from '@playwright/test'

test.describe('离线能力与同步机制系统', () => {
  test('离线能力监控页面可访问', async ({ page }) => {
    await page.goto('/offline')
    await page.waitForLoadState('networkidle')
    await expect(page.getByText('离线能力监控')).toBeVisible()
  })

  test('向量时钟可视化页面可访问', async ({ page }) => {
    await page.goto('/vector-clock')
    await page.waitForLoadState('networkidle')
    await expect(page.getByText('向量时钟可视化')).toBeVisible()
    await expect(page.getByText('节点状态')).toBeVisible()
    await expect(page.getByText('事件历史')).toBeVisible()
  })

  test('同步引擎内部机制页面可访问', async ({ page }) => {
    await page.goto('/sync-engine')
    await page.waitForLoadState('networkidle')
    await expect(page.getByText('同步引擎内部机制展示')).toBeVisible()
    await expect(page.getByText('引擎控制面板')).toBeVisible()
  })

  test('本地模型缓存监控页面可访问', async ({ page }) => {
    await page.goto('/model-cache')
    await page.waitForLoadState('networkidle')
    await expect(page.getByText('模型缓存')).toBeVisible()
    await expect(page.locator('.ant-table')).toBeVisible()
  })

  test('网络监控详情页面可访问', async ({ page }) => {
    await page.goto('/network-monitor')
    await page.waitForLoadState('networkidle')
    await expect(page.getByText('网络监控详情')).toBeVisible()
  })

  test('离线网络异常可处理', async ({ page }) => {
    await page.goto('/vector-clock')
    await expect(page.getByText(/向量时钟/)).toBeVisible()
    await page.context().setOffline(true)
    await page.waitForTimeout(500)
    await page.context().setOffline(false)
  })
})
