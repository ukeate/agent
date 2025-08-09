import { test, expect } from '@playwright/test'
import { setupRagApiMocks, setupRagApiErrors } from './mocks/rag-api'

test.describe('基础RAG系统测试', () => {
  test.beforeEach(async ({ page }) => {
    await setupRagApiMocks(page)
  })

  test('RAG页面加载和基本界面展示', async ({ page }) => {
    await page.goto('/rag')
    
    // 验证页面标题和导航
    await expect(page.getByRole('heading', { name: /RAG.*搜索/i })).toBeVisible()
    await expect(page.locator('text=RAG 搜索')).toBeVisible()
    
    // 验证面包屑导航
    await expect(page.locator('text=RAG 搜索').first()).toBeVisible()
    
    // 验证升级提示
    await expect(page.locator('text=升级到 Agentic RAG')).toBeVisible()
    await expect(page.locator('text=🚀 智能升级')).toBeVisible()
  })

  test('RAG查询面板功能测试', async ({ page }) => {
    await page.goto('/rag')
    
    // 查找查询输入框
    const queryInput = page.locator('textarea[placeholder*="请输入"], input[placeholder*="搜索"], textarea[placeholder*="查询"]').first()
    await expect(queryInput).toBeVisible()
    
    // 测试查询输入
    await queryInput.fill('测试RAG检索功能')
    
    // 查找并点击搜索按钮
    const searchButton = page.locator('button:has-text("搜索"), button:has-text("查询"), button:has-text("检索")').first()
    await searchButton.click()
    
    // 验证搜索开始提示
    await expect(page.locator('text=开始搜索').or(page.locator('text=正在搜索'))).toBeVisible()
  })

  test('RAG搜索结果展示', async ({ page }) => {
    await page.goto('/rag')
    
    // 执行搜索
    const queryInput = page.locator('textarea[placeholder*="请输入"], input[placeholder*="搜索"], textarea[placeholder*="查询"]').first()
    await queryInput.fill('AI智能检索')
    
    const searchButton = page.locator('button:has-text("搜索"), button:has-text("查询"), button:has-text("检索")').first()
    await searchButton.click()
    
    // 等待结果加载
    await page.waitForTimeout(1000)
    
    // 验证搜索完成提示
    await expect(page.locator('text=搜索完成').or(page.locator('text=找到.*结果'))).toBeVisible({ timeout: 10000 })
    
    // 验证结果展示区域存在
    await expect(page.locator('[data-testid="rag-results"], .rag-results, .search-results').first()).toBeVisible()
    
    // 验证结果项存在
    await expect(page.locator('text=检索到的内容:AI智能检索')).toBeVisible()
    await expect(page.locator('text=相关代码片段:AI智能检索')).toBeVisible()
    
    // 验证结果来源信息
    await expect(page.locator('text=test-file.md')).toBeVisible()
    await expect(page.locator('text=test-code.py')).toBeVisible()
  })

  test('RAG索引状态监控', async ({ page }) => {
    await page.goto('/rag')
    
    // 桌面端：查看右侧状态面板
    if (await page.locator('[data-testid="index-status"], .index-status').first().isVisible()) {
      const statusPanel = page.locator('[data-testid="index-status"], .index-status').first()
      
      // 验证向量数量统计
      await expect(statusPanel.locator('text=468').or(statusPanel.locator('text=向量'))).toBeVisible()
      await expect(statusPanel.locator('text=1746').or(statusPanel.locator('text=代码'))).toBeVisible()
      
      // 验证健康状态
      await expect(statusPanel.locator('text=green').or(statusPanel.locator('text=正常'))).toBeVisible()
    }
    
    // 移动端/平板：查看状态抽屉
    const statusButton = page.locator('button:has-text("索引状态"), button:has-text("查看索引状态")').first()
    if (await statusButton.isVisible()) {
      await statusButton.click()
      
      // 等待抽屉打开
      await page.waitForTimeout(500)
      
      // 验证抽屉内容
      await expect(page.locator('.ant-drawer').locator('text=索引状态')).toBeVisible()
      
      // 关闭抽屉
      await page.locator('.ant-drawer .ant-drawer-close').click()
    }
  })

  test('RAG历史记录和清空功能', async ({ page }) => {
    await page.goto('/rag')
    
    // 执行几次搜索
    const queryInput = page.locator('textarea[placeholder*="请输入"], input[placeholder*="搜索"], textarea[placeholder*="查询"]').first()
    const searchButton = page.locator('button:has-text("搜索"), button:has-text("查询"), button:has-text("检索")').first()
    
    await queryInput.fill('第一次搜索')
    await searchButton.click()
    await page.waitForTimeout(1000)
    
    await queryInput.fill('第二次搜索') 
    await searchButton.click()
    await page.waitForTimeout(1000)
    
    // 查看历史记录（如果有历史功能）
    const historyButton = page.locator('button:has-text("历史"), text=历史记录').first()
    if (await historyButton.isVisible()) {
      await historyButton.click()
      
      // 验证历史记录页面
      await expect(page.locator('text=第一次搜索').or(page.locator('text=第二次搜索'))).toBeVisible()
    }
  })

  test('RAG响应式布局测试', async ({ page }) => {
    await page.goto('/rag')
    
    // 桌面布局
    await page.setViewportSize({ width: 1200, height: 800 })
    await page.waitForTimeout(300)
    
    // 验证三栏布局存在
    const queryPanel = page.locator('[data-testid="query-panel"], .rag-query-panel').first()
    const resultsPanel = page.locator('[data-testid="results-panel"], .rag-results').first() 
    const statusPanel = page.locator('[data-testid="status-panel"], .rag-index-status').first()
    
    await expect(queryPanel).toBeVisible()
    
    // 平板布局
    await page.setViewportSize({ width: 768, height: 1024 })
    await page.waitForTimeout(500)
    
    // 移动端布局
    await page.setViewportSize({ width: 375, height: 667 })
    await page.waitForTimeout(500)
    
    // 验证移动端菜单存在
    const mobileMenuButton = page.locator('button[data-testid="mobile-menu"], button:has(.anticon-menu)').first()
    if (await mobileMenuButton.isVisible()) {
      await mobileMenuButton.click()
      
      // 验证移动端菜单项
      await expect(page.locator('.ant-drawer').locator('text=搜索面板')).toBeVisible()
      await expect(page.locator('.ant-drawer').locator('text=索引状态')).toBeVisible()
      
      // 关闭菜单
      await page.locator('.ant-drawer .ant-drawer-close').click()
    }
  })

  test('RAG搜索参数配置', async ({ page }) => {
    await page.goto('/rag')
    
    // 查找高级搜索选项
    const advancedButton = page.locator('button:has-text("高级"), text=高级搜索, button:has-text("选项")').first()
    if (await advancedButton.isVisible()) {
      await advancedButton.click()
      
      // 配置搜索参数
      const limitInput = page.locator('input[placeholder*="结果数量"], input[type="number"]').first()
      if (await limitInput.isVisible()) {
        await limitInput.fill('10')
      }
      
      // 文件类型过滤器
      const typeFilter = page.locator('select[placeholder*="文件类型"], .ant-select').first()
      if (await typeFilter.isVisible()) {
        await typeFilter.click()
        await page.locator('text=代码').click()
      }
    }
  })

  test('RAG错误处理测试', async ({ page }) => {
    // 设置错误mock
    await setupRagApiErrors(page)
    
    await page.goto('/rag')
    
    // 执行搜索触发错误
    const queryInput = page.locator('textarea[placeholder*="请输入"], input[placeholder*="搜索"], textarea[placeholder*="查询"]').first()
    await queryInput.fill('触发错误的搜索')
    
    const searchButton = page.locator('button:has-text("搜索"), button:has-text("查询"), button:has-text("检索")').first()
    await searchButton.click()
    
    // 验证错误提示
    await expect(page.locator('text=向量数据库连接失败').or(page.locator('text=系统错误'))).toBeVisible({ timeout: 5000 })
    
    // 验证错误可以关闭
    const errorAlert = page.locator('.ant-alert-error').first()
    if (await errorAlert.isVisible()) {
      const closeButton = errorAlert.locator('.ant-alert-close-icon').first()
      if (await closeButton.isVisible()) {
        await closeButton.click()
        await expect(errorAlert).not.toBeVisible()
      }
    }
  })

  test('RAG页面导航和路由', async ({ page }) => {
    await page.goto('/')
    
    // 从首页导航到RAG页面
    const ragLink = page.locator('a[href="/rag"]').or(page.locator('text=RAG')).or(page.locator('text=检索')).first()
    if (await ragLink.isVisible()) {
      await ragLink.click()
      await expect(page).toHaveURL(/.*\/rag/)
    } else {
      // 直接导航
      await page.goto('/rag')
    }
    
    // 验证页面加载
    await expect(page.locator('text=RAG').first()).toBeVisible()
    
    // 测试升级链接
    const upgradeButton = page.locator('button:has-text("升级到 Agentic RAG"), a:has-text("智能升级")').first()
    if (await upgradeButton.isVisible()) {
      await upgradeButton.click()
      // 注意：这里可能会跳转到agentic-rag页面，但由于是button可能只是模拟跳转
      // 我们验证页面状态
    }
  })

  test('RAG键盘快捷键', async ({ page }) => {
    await page.goto('/rag')
    
    const queryInput = page.locator('textarea[placeholder*="请输入"], input[placeholder*="搜索"], textarea[placeholder*="查询"]').first()
    await queryInput.fill('快捷键测试')
    
    // 测试Enter键提交
    await queryInput.press('Enter')
    
    // 验证搜索被触发
    await expect(page.locator('text=开始搜索').or(page.locator('text=正在搜索'))).toBeVisible()
    
    // 等待搜索完成
    await page.waitForTimeout(1000)
    
    // 测试Ctrl+K聚焦搜索框（如果支持）
    await page.keyboard.press('Escape') // 清除焦点
    await page.keyboard.press('Control+K')
    
    // 验证搜索框获得焦点
    await expect(queryInput).toBeFocused()
  })
})