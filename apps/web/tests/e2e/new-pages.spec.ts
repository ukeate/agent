import { test, expect } from '@playwright/test'

test.describe('新增页面功能测试', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:3000')
  })

  test('异步智能体管理页面 - 基础功能', async ({ page }) => {
    await page.goto('http://localhost:3000/async-agents')
    await expect(page).toHaveURL(/\/async-agents$/)
    
    // 检查页面标题
    await expect(page.locator('h1')).toContainText('异步智能体管理')
    
    // 检查统计卡片
    await expect(page.locator('.ant-statistic-title').first()).toContainText('运行中的智能体')
    
    // 检查智能体表格存在（真实数据允许为空）
    await expect(page.locator('.ant-table')).toBeVisible()
    
    // 测试创建智能体模态框
    await page.getByRole('button', { name: /创建智能体/ }).click()
    const modal = page.locator('.ant-modal')
    await expect(modal.locator('.ant-modal-title')).toContainText('创建新智能体')
    
    // 填写表单并创建
    await modal.locator('input[placeholder="请输入智能体名称"]').fill('测试智能体')
    await modal.locator('.ant-select-selector').click()
    await page.locator('.ant-select-dropdown').getByText('知识检索', { exact: true }).click()
    await modal.getByRole('button', { name: /创\s*建/ }).click()
    
    // 验证成功消息
    await expect(page.locator('.ant-message')).toContainText('已创建智能体')
  })

  test('事件监控仪表板页面 - 基础功能', async ({ page }) => {
    await page.goto('http://localhost:3000/events')
    await expect(page).toHaveURL(/\/events$/)
    
    // 检查页面标题
    await expect(page.locator('h1')).toContainText('事件监控仪表板')
    
    // 检查统计卡片
    await expect(page.locator('.ant-statistic-title').first()).toContainText('总事件数')
    
    // 检查事件表格
    await expect(page.locator('.ant-table')).toBeVisible()
    
    // 测试筛选功能
    const listCard = page.locator('.ant-card:has-text("事件列表")')
    await listCard.locator('.ant-select').first().click()
    await page.locator('.ant-select-dropdown').getByText('错误', { exact: true }).click()
    
    // 检查最近事件时间线
    await expect(page.locator('.ant-timeline')).toBeVisible()
    
    // 测试实时监控切换
    await page.getByRole('button', { name: '开启实时监控' }).click()
    await expect(page.getByRole('button', { name: '关闭实时监控' })).toBeVisible()
  })

  test('安全管理页面 - 基础功能', async ({ page }) => {
    await page.goto('http://localhost:3000/security')
    await expect(page).toHaveURL(/\/security$/)
    
    // 检查页面标题
    await expect(page.locator('h1')).toContainText('安全管理中心')
    
    // 检查主要功能标签
    await expect(page.getByRole('button', { name: '安全概览' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'API密钥' })).toBeVisible()
    await expect(page.getByRole('button', { name: '工具权限' })).toBeVisible()
    await expect(page.getByRole('button', { name: '安全告警' })).toBeVisible()
  })

  test('性能监控页面 - 基础功能', async ({ page }) => {
    await page.goto('http://localhost:3000/performance')
    await expect(page).toHaveURL(/\/performance$/)
    
    // 检查页面标题
    await expect(page.locator('h1')).toContainText('性能监控')
    
    // 检查性能指标卡片
    await expect(page.locator('.ant-statistic-title').first()).toContainText('CPU使用率')
    
    // 检查进度条显示
    expect(await page.locator('.ant-progress').count()).toBeGreaterThan(0)
    
    // 切换到历史数据标签页
    await page.getByRole('tab', { name: '历史数据' }).click()
    await expect(page.locator('.ant-card-head-title:text("历史性能数据")')).toBeVisible()
    
    // 切换到性能分析标签页
    await page.getByRole('tab', { name: '性能分析' }).click()
    await expect(page.locator('.ant-card-head-title:text("性能评分")')).toBeVisible()
    
    // 测试自动刷新切换
    await page.getByRole('button', { name: '关闭自动刷新' }).click()
    await expect(page.getByRole('button', { name: '开启自动刷新' })).toBeVisible()
  })

  test('架构调试页面 - 基础功能', async ({ page }) => {
    await page.goto('http://localhost:3000/debug')
    await expect(page).toHaveURL(/\/debug$/)
    
    // 检查页面标题
    await expect(page.locator('h1')).toContainText('架构调试中心')
    
    // 检查统计卡片
    await expect(page.locator('.ant-card').first()).toContainText('健康组件')
    
    // 检查系统拓扑树
    await expect(page.locator('.ant-tree')).toBeVisible()
    
    // 测试组件选择
    await page.getByRole('treeitem', { name: /Web前端/ }).click()
    
    // 检查组件详情显示
    await expect(page.locator('.ant-descriptions')).toBeVisible()
    
    // 切换到调试会话标签页
    await page.getByRole('tab', { name: '调试会话' }).click()
    await expect(page.locator('.ant-table')).toBeVisible()
    
    // 切换到系统日志标签页
    await page.getByRole('tab', { name: '系统日志' }).click()
    await expect(page.locator('.ant-timeline')).toBeVisible()
  })

  test('导航菜单完整性测试', async ({ page }) => {
    const routes = ['/chat', '/multi-agent', '/supervisor', '/rag', '/async-agents', '/events', '/security', '/performance', '/debug']
    for (const route of routes) {
      await page.goto(`http://localhost:3000${route}`)
      await expect(page.locator('h1')).toBeVisible()
    }
  })

  test('页面响应式设计测试', async ({ page }) => {
    // 测试不同屏幕尺寸下的布局
    await page.setViewportSize({ width: 1200, height: 800 })
    await page.goto('http://localhost:3000/async-agents')
    await expect(page.locator('h1')).toBeVisible()
    
    // 模拟平板尺寸
    await page.setViewportSize({ width: 768, height: 1024 })
    await expect(page.locator('.ant-layout-content')).toBeVisible()
    
    // 恢复桌面尺寸
    await page.setViewportSize({ width: 1920, height: 1080 })
  })

  test('错误处理和加载状态测试', async ({ page }) => {
    // 访问不存在的路由
    await page.goto('http://localhost:3000/non-existent')
    
    // 应该重定向到主页或显示404
    await expect(page).toHaveURL(/\/(chat)?$/)
    
    // 测试各页面的加载状态
    const pages = ['/async-agents', '/events', '/security', '/performance', '/debug']
    
    for (const pagePath of pages) {
      await page.goto(`http://localhost:3000${pagePath}`)
      await expect(page.locator('h1')).toBeVisible()
      await page.waitForLoadState('networkidle')
    }
  })
})
