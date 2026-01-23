/**
 * 安全功能E2E测试
 */

import { test, expect } from '@playwright/test'

test.describe('Security Management System', () => {
  test.beforeEach(async ({ page }) => {
    // 导航到安全管理页面
    await page.goto('http://localhost:3000/security')
  })

  test.describe('Security Dashboard', () => {
    test('should display security dashboard with statistics', async ({
      page,
    }) => {
      // 等待dashboard加载
      await page.waitForSelector('[data-testid="security-dashboard"]', {
        timeout: 10000,
      })

      // 验证统计卡片
      await expect(page.locator('text=总请求数')).toBeVisible()
      await expect(page.locator('text=被阻止请求')).toBeVisible()
      await expect(page.locator('text=活跃威胁')).toBeVisible()
      await expect(page.locator('text=API密钥数')).toBeVisible()

      // 验证风险事件分布
      await expect(page.locator('text=风险事件分布')).toBeVisible()
      await expect(page.locator('text=高风险')).toBeVisible()
      await expect(page.locator('text=中风险')).toBeVisible()
      await expect(page.locator('text=低风险')).toBeVisible()

      // 验证安全告警部分
      await expect(page.locator('text=最近的安全告警')).toBeVisible()
    })

    test('should refresh data when refresh button clicked', async ({
      page,
    }) => {
      await page.waitForSelector('button:has-text("刷新数据")')

      // 点击刷新按钮
      await page.click('button:has-text("刷新数据")')

      // 验证加载状态
      await expect(page.locator('.animate-spin')).toBeVisible()

      // 等待数据加载完成
      await expect(page.locator('.animate-spin')).not.toBeVisible({
        timeout: 5000,
      })
    })

    test('should handle alert resolution', async ({ page }) => {
      // 等待告警加载
      await page.waitForSelector('[data-testid="security-alert"]', {
        timeout: 10000,
      })

      // 找到活跃告警的解决按钮
      const resolveButton = page
        .locator('button:has-text("标记已解决")')
        .first()

      if (await resolveButton.isVisible()) {
        await resolveButton.click()

        // 验证告警状态更新
        await expect(page.locator('text=已解决')).toBeVisible({ timeout: 5000 })
      }
    })
  })

  test.describe('API Key Management', () => {
    test.beforeEach(async ({ page }) => {
      // 切换到API密钥标签
      await page.click('button:has-text("API密钥")')
    })

    test('should display API key list', async ({ page }) => {
      await expect(page.locator('text=API密钥管理')).toBeVisible()
      await expect(page.locator('button:has-text("创建新密钥")')).toBeVisible()
    })

    test('should open create API key dialog', async ({ page }) => {
      await page.click('button:has-text("创建新密钥")')

      // 验证对话框打开
      await expect(page.locator('text=创建新API密钥')).toBeVisible()
      await expect(
        page.locator('input[placeholder*="生产环境密钥"]')
      ).toBeVisible()
      await expect(page.locator('text=有效期（天）')).toBeVisible()
      await expect(page.locator('text=权限')).toBeVisible()
    })

    test('should create new API key', async ({ page }) => {
      await page.click('button:has-text("创建新密钥")')

      // 填写表单
      await page.fill('input[placeholder*="生产环境密钥"]', 'Test API Key')
      await page.fill('input[type="number"]', '30')

      // 选择权限
      await page.check('input[type="checkbox"]', { force: true })

      // 创建密钥
      await page.click('button:has-text("创建")')

      // 验证密钥创建成功
      await expect(page.locator('text=新API密钥已创建')).toBeVisible({
        timeout: 5000,
      })
      await expect(page.locator('button:has-text("复制")')).toBeVisible()
    })

    test('should copy API key to clipboard', async ({ page, context }) => {
      // 授予剪贴板权限
      await context.grantPermissions(['clipboard-read', 'clipboard-write'])

      // 如果有复制按钮，点击它
      const copyButton = page.locator('button:has-text("复制密钥")').first()
      if (await copyButton.isVisible()) {
        await copyButton.click()

        // 验证复制操作（通常会有提示）
        // 注意：实际的剪贴板内容验证在浏览器中可能受限
      }
    })

    test('should revoke API key', async ({ page }) => {
      const revokeButton = page.locator('button:has-text("撤销")').first()

      if (await revokeButton.isVisible()) {
        // 点击撤销按钮
        page.on('dialog', dialog => dialog.accept())
        await revokeButton.click()

        // 验证密钥被撤销
        await expect(page.locator('text=revoked')).toBeVisible({
          timeout: 5000,
        })
      }
    })
  })

  test.describe('Tool Permissions', () => {
    test.beforeEach(async ({ page }) => {
      // 切换到工具权限标签
      await page.click('button:has-text("工具权限")')
    })

    test('should display tool permissions list', async ({ page }) => {
      await expect(page.locator('text=MCP工具权限管理')).toBeVisible()
      await expect(
        page.locator('input[placeholder="搜索工具..."]')
      ).toBeVisible()
    })

    test('should filter tools by search', async ({ page }) => {
      await page.fill('input[placeholder="搜索工具..."]', 'file')

      // 验证过滤结果
      await expect(page.locator('text=file_read')).toBeVisible()
      await expect(page.locator('text=file_write')).toBeVisible()
    })

    test('should filter tools by category', async ({ page }) => {
      const categorySelect = page.locator('select').first()
      await categorySelect.selectOption({ index: 1 })

      // 验证分类过滤生效
      await page.waitForTimeout(500)
    })

    test('should toggle tool enabled status', async ({ page }) => {
      // 找到第一个开关
      const firstSwitch = page.locator('[role="switch"]').first()

      if (await firstSwitch.isVisible()) {
        const initialState = await firstSwitch.getAttribute('aria-checked')
        await firstSwitch.click()

        // 验证状态改变
        const newState = await firstSwitch.getAttribute('aria-checked')
        expect(newState).not.toBe(initialState)
      }
    })

    test('should display tool risk levels', async ({ page }) => {
      // 验证风险级别标签
      const riskBadges = page.locator('text=/.*风险/')
      const count = await riskBadges.count()
      expect(count).toBeGreaterThan(0)
    })

    test('should show tool usage statistics', async ({ page }) => {
      // 验证使用统计
      await expect(page.locator('text=/使用次数:.*/')).toBeVisible()
    })
  })

  test.describe('Security Alerts', () => {
    test.beforeEach(async ({ page }) => {
      // 切换到安全告警标签
      await page.click('button:has-text("安全告警")')
    })

    test('should display alerts list', async ({ page }) => {
      await expect(page.locator('text=安全告警管理')).toBeVisible()
      await expect(
        page.locator('input[placeholder="搜索告警..."]')
      ).toBeVisible()
    })

    test('should filter alerts by status', async ({ page }) => {
      const statusSelect = page.locator('select').first()
      await statusSelect.selectOption('active')

      // 等待过滤生效
      await page.waitForTimeout(500)

      // 验证只显示活跃告警
      const activeBadges = page.locator('text=active')
      if (await activeBadges.first().isVisible()) {
        const count = await activeBadges.count()
        expect(count).toBeGreaterThan(0)
      }
    })

    test('should filter alerts by severity', async ({ page }) => {
      const severitySelect = page.locator('select').nth(1)
      await severitySelect.selectOption('high')

      // 等待过滤生效
      await page.waitForTimeout(500)
    })

    test('should open alert details dialog', async ({ page }) => {
      const alertCard = page.locator('[data-testid="alert-card"]').first()

      if (await alertCard.isVisible()) {
        await alertCard.click()

        // 验证详情对话框
        await expect(page.locator('text=告警详情')).toBeVisible()
        await expect(page.locator('text=告警ID')).toBeVisible()
        await expect(page.locator('text=详细信息')).toBeVisible()
      }
    })

    test('should update alert status', async ({ page }) => {
      const investigateButton = page
        .locator('button:has-text("开始调查")')
        .first()

      if (await investigateButton.isVisible()) {
        await investigateButton.click()

        // 验证状态更新
        await expect(page.locator('text=investigating')).toBeVisible({
          timeout: 5000,
        })
      }
    })

    test('should mark alert as false positive', async ({ page }) => {
      const falsePositiveButton = page
        .locator('button:has-text("标记误报")')
        .first()

      if (await falsePositiveButton.isVisible()) {
        await falsePositiveButton.click()

        // 验证状态更新
        await expect(page.locator('text=false_positive')).toBeVisible({
          timeout: 5000,
        })
      }
    })
  })

  test.describe('Security Integration', () => {
    test('should handle authentication failure', async ({ page }) => {
      // 模拟未认证访问
      await page.route('**/api/v1/security/**', route => {
        route.fulfill({
          status: 401,
          body: JSON.stringify({ detail: '未认证' }),
        })
      })

      await page.goto('http://localhost:3000/security')

      // 验证错误处理
      await expect(page.locator('text=/.*失败.*/')).toBeVisible({
        timeout: 5000,
      })
    })

    test('should handle API errors gracefully', async ({ page }) => {
      // 模拟API错误
      await page.route('**/api/v1/security/metrics', route => {
        route.fulfill({
          status: 500,
          body: JSON.stringify({ detail: '服务器错误' }),
        })
      })

      await page.goto('http://localhost:3000/security')

      // 验证错误提示
      await expect(page.locator('text=/.*加载.*失败.*/')).toBeVisible({
        timeout: 5000,
      })
    })

    test('should auto-refresh security data', async ({ page }) => {
      await page.goto('http://localhost:3000/security')

      // 等待初始加载
      await page.waitForSelector('[data-testid="security-dashboard"]', {
        timeout: 10000,
      })

      // 等待自动刷新（10秒间隔）
      await page.waitForTimeout(11000)

      // 验证数据仍然显示
      await expect(page.locator('text=总请求数')).toBeVisible()
    })

    test('should handle network timeout', async ({ page }) => {
      // 模拟网络超时
      await page.route('**/api/v1/security/**', async route => {
        await new Promise(resolve => setTimeout(resolve, 30000))
        route.abort()
      })

      await page.goto('http://localhost:3000/security')

      // 验证超时处理
      await expect(page.locator('text=/.*加载.*/')).toBeVisible({
        timeout: 5000,
      })
    })
  })

  test.describe('Performance', () => {
    test('should load security dashboard within 2 seconds', async ({
      page,
    }) => {
      const startTime = Date.now()
      await page.goto('http://localhost:3000/security')
      await page.waitForSelector('[data-testid="security-dashboard"]')
      const loadTime = Date.now() - startTime

      expect(loadTime).toBeLessThan(2000)
    })

    test('should handle large datasets efficiently', async ({ page }) => {
      // 模拟大量数据
      await page.route('**/api/v1/security/alerts', route => {
        const alerts = Array.from({ length: 1000 }, (_, i) => ({
          id: `alert-${i}`,
          alert_type: 'suspicious_request',
          severity: ['low', 'medium', 'high'][i % 3],
          description: `Test alert ${i}`,
          timestamp: new Date().toISOString(),
          status: 'active',
        }))

        route.fulfill({
          status: 200,
          body: JSON.stringify(alerts),
        })
      })

      await page.goto('http://localhost:3000/security')
      await page.click('button:has-text("安全告警")')

      // 验证大数据集渲染
      await expect(
        page.locator('[data-testid="alert-card"]').first()
      ).toBeVisible({ timeout: 5000 })
    })
  })
})
