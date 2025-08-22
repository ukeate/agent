import { test, expect } from '@playwright/test'

test.describe('新增页面功能测试', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:3000')
  })

  test('异步智能体管理页面 - 基础功能', async ({ page }) => {
    // 导航到异步智能体页面
    await page.click('span:text("异步智能体")')
    await expect(page).toHaveURL('/async-agents')
    
    // 检查页面标题
    await expect(page.locator('h1')).toContainText('异步智能体管理')
    
    // 检查统计卡片
    await expect(page.locator('.ant-statistic-title').first()).toContainText('运行中的智能体')
    
    // 检查智能体表格
    const rows = page.locator('.ant-table-tbody tr')
    await expect(rows).toHaveCount(3) // 根据模拟数据调整
    
    // 测试创建智能体模态框
    await page.click('button:text("创建智能体")')
    await expect(page.locator('.ant-modal-title')).toContainText('创建新智能体')
    
    // 填写表单并创建
    await page.fill('input[placeholder="请输入智能体名称"]', '测试智能体')
    await page.selectOption('div[role="combobox"]', { label: '检索型' })
    await page.click('button:text("创建")')
    
    // 验证成功消息
    await expect(page.locator('.ant-message')).toContainText('已创建智能体')
  })

  test('事件监控仪表板页面 - 基础功能', async ({ page }) => {
    // 导航到事件监控页面
    await page.click('span:text("事件监控")')
    await expect(page).toHaveURL('/events')
    
    // 检查页面标题
    await expect(page.locator('h1')).toContainText('事件监控仪表板')
    
    // 检查统计卡片
    await expect(page.locator('.ant-statistic-title').first()).toContainText('总事件数')
    
    // 检查事件表格
    await expect(page.locator('.ant-table-tbody tr')).toHaveCount.greaterThan(0)
    
    // 测试筛选功能
    await page.click('div[title="全部类型"]')
    await page.click('div:text("错误")')
    
    // 检查最近事件时间线
    await expect(page.locator('.ant-timeline')).toBeVisible()
    
    // 测试自动刷新切换
    await page.click('button:text("关闭自动刷新")')
    await expect(page.locator('button:text("开启自动刷新")')).toBeVisible()
  })

  test('安全管理页面 - 基础功能', async ({ page }) => {
    // 导航到安全管理页面
    await page.click('span:text("安全管理")')
    await expect(page).toHaveURL('/security')
    
    // 检查页面标题
    await expect(page.locator('h1')).toContainText('企业级安全管理')
    
    // 检查安全评分
    await expect(page.locator('.ant-statistic-title').first()).toContainText('安全评分')
    
    // 测试用户管理标签页
    await expect(page.locator('.ant-tabs-tab:text("用户管理")')).toHaveClass(/ant-tabs-tab-active/)
    await expect(page.locator('.ant-table-tbody tr')).toHaveCount.greaterThan(0)
    
    // 切换到安全事件标签页
    await page.click('.ant-tabs-tab:text("安全事件")')
    await expect(page.locator('.ant-table-tbody tr')).toHaveCount.greaterThan(0)
    
    // 切换到安全策略标签页
    await page.click('.ant-tabs-tab:text("安全策略")')
    await expect(page.locator('.ant-card')).toHaveCount.greaterThan(0)
    
    // 测试策略开关
    const firstSwitch = page.locator('.ant-switch').first()
    await firstSwitch.click()
  })

  test('性能监控页面 - 基础功能', async ({ page }) => {
    // 导航到性能监控页面
    await page.click('span:text("性能监控")')
    await expect(page).toHaveURL('/performance')
    
    // 检查页面标题
    await expect(page.locator('h1')).toContainText('性能监控')
    
    // 检查性能指标卡片
    await expect(page.locator('.ant-statistic-title').first()).toContainText('CPU使用率')
    
    // 检查进度条显示
    await expect(page.locator('.ant-progress')).toHaveCount.greaterThan(0)
    
    // 切换到历史数据标签页
    await page.click('.ant-tabs-tab:text("历史数据")')
    await expect(page.locator('.ant-card-head-title:text("历史性能数据")')).toBeVisible()
    
    // 切换到性能分析标签页
    await page.click('.ant-tabs-tab:text("性能分析")')
    await expect(page.locator('.ant-card-head-title:text("性能评分")')).toBeVisible()
    
    // 测试自动刷新切换
    await page.click('button:text("关闭自动刷新")')
    await expect(page.locator('button:text("开启自动刷新")')).toBeVisible()
  })

  test('架构调试页面 - 基础功能', async ({ page }) => {
    // 导航到架构调试页面
    await page.click('span:text("架构调试")')
    await expect(page).toHaveURL('/debug')
    
    // 检查页面标题
    await expect(page.locator('h1')).toContainText('架构调试中心')
    
    // 检查统计卡片
    await expect(page.locator('.ant-card').first()).toContainText('健康组件')
    
    // 检查系统拓扑树
    await expect(page.locator('.ant-tree')).toBeVisible()
    
    // 测试组件选择
    await page.click('.ant-tree-node-content-wrapper:text("Web前端")')
    
    // 检查组件详情显示
    await expect(page.locator('.ant-descriptions')).toBeVisible()
    
    // 切换到调试会话标签页
    await page.click('.ant-tabs-tab:text("调试会话")')
    await expect(page.locator('.ant-table')).toBeVisible()
    
    // 切换到系统日志标签页
    await page.click('.ant-tabs-tab:text("系统日志")')
    await expect(page.locator('.ant-timeline')).toBeVisible()
  })

  test('导航菜单完整性测试', async ({ page }) => {
    // 检查所有新增菜单项是否存在
    const expectedMenuItems = [
      '单代理对话',
      '多代理协作', 
      '监督者模式',
      'RAG检索',
      '工作流可视化',
      '异步智能体',
      '事件监控',
      '安全管理',
      '性能监控',
      '架构调试'
    ]

    for (const item of expectedMenuItems) {
      await expect(page.locator(`span:text("${item}")`)).toBeVisible()
    }

    // 测试菜单折叠功能
    await page.click('.anticon-menu-fold')
    await expect(page.locator('.ant-layout-sider-collapsed')).toBeVisible()
    
    // 展开菜单
    await page.click('.anticon-menu-unfold')
    await expect(page.locator('.ant-layout-sider-collapsed')).not.toBeVisible()
  })

  test('页面响应式设计测试', async ({ page }) => {
    // 测试不同屏幕尺寸下的布局
    await page.setViewportSize({ width: 1200, height: 800 })
    await page.click('span:text("异步智能体")')
    await expect(page.locator('.ant-col')).toHaveCount.greaterThan(0)
    
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