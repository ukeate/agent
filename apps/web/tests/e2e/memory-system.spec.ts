/**
 * 记忆管理系统E2E测试
 */
import { test, expect } from '@playwright/test'

// 测试数据
const testMemories = {
  working: {
    content: '测试工作记忆内容 - 当前用户正在学习AI记忆系统',
    importance: 0.6,
    tags: ['learning', 'ai', 'memory']
  },
  episodic: {
    content: '测试情景记忆 - 用户在2025年1月15日成功实现了记忆管理功能',
    importance: 0.8,
    tags: ['success', 'implementation', 'milestone']
  },
  semantic: {
    content: '测试语义记忆 - 记忆管理系统是AI智能体的核心能力之一，包括工作记忆、情景记忆和语义记忆三层架构',
    importance: 0.9,
    tags: ['knowledge', 'architecture', 'definition']
  }
}

test.describe('记忆管理系统E2E测试', () => {
  test.beforeEach(async ({ page }) => {
    // 启动前端应用
    await page.goto('http://localhost:3000')
    
    // 等待页面加载完成
    await page.waitForLoadState('networkidle')
  })

  test('记忆层级架构页面基本功能', async ({ page }) => {
    // 导航到记忆层级页面
    await page.click('text=记忆层级架构')
    await page.waitForURL('**/memory-hierarchy')
    
    // 验证页面标题
    await expect(page.locator('h1')).toContainText('记忆层级系统')
    
    // 验证页面描述
    await expect(page.locator('p')).toContainText('可视化展示三层记忆架构')
    
    // 验证统计概览卡片
    await expect(page.locator('text=总记忆数')).toBeVisible()
    await expect(page.locator('text=平均重要性')).toBeVisible()
    await expect(page.locator('text=记忆增长率')).toBeVisible()
    await expect(page.locator('text=存储使用')).toBeVisible()
    
    // 验证三层记忆架构
    await expect(page.locator('text=工作记忆')).toBeVisible()
    await expect(page.locator('text=情景记忆')).toBeVisible()
    await expect(page.locator('text=语义记忆')).toBeVisible()
    
    // 验证操作按钮
    await expect(page.locator('text=巩固记忆')).toBeVisible()
    await expect(page.locator('text=刷新')).toBeVisible()
    await expect(page.locator('text=清理旧记忆')).toBeVisible()
    
    // 验证记忆流转示意图
    await expect(page.locator('text=记忆通过重要性评估和访问频率，逐级提升到更持久的存储层')).toBeVisible()
  })

  test('记忆召回测试页面功能', async ({ page }) => {
    // 导航到记忆召回测试页面
    await page.click('text=记忆召回测试')
    await page.waitForURL('**/memory-recall')
    
    // 验证页面标题
    await expect(page.locator('h1')).toContainText('记忆召回测试')
    
    // 验证召回策略说明
    await expect(page.locator('text=召回策略说明')).toBeVisible()
    await expect(page.locator('text=向量搜索')).toBeVisible()
    await expect(page.locator('text=时间搜索')).toBeVisible()
    await expect(page.locator('text=实体搜索')).toBeVisible()
    await expect(page.locator('text=混合搜索')).toBeVisible()
    
    // 测试召回模式切换
    await page.click('text=向量搜索')
    await expect(page.locator('button:has-text("向量搜索")').first()).toHaveClass(/ant-btn-primary/)
    
    await page.click('text=时间搜索')
    await expect(page.locator('button:has-text("时间搜索")').first()).toHaveClass(/ant-btn-primary/)
    
    await page.click('text=混合召回')
    await expect(page.locator('button:has-text("混合召回")').first()).toHaveClass(/ant-btn-primary/)
    
    // 测试查询输入
    const queryInput = page.locator('textarea[placeholder*="输入查询内容"]')
    await queryInput.fill('Python编程学习')
    
    // 测试执行召回按钮
    await expect(page.locator('text=执行召回')).toBeVisible()
    
    // 验证结果面板
    await expect(page.locator('text=召回结果')).toBeVisible()
    await expect(page.locator('text=关联记忆链')).toBeVisible()
    
    // 测试创建测试记忆
    await page.click('text=创建测试记忆')
    // 这里应该会显示成功消息（如果API可用）
  })

  test('记忆分析仪表板功能', async ({ page }) => {
    // 导航到记忆分析仪表板
    await page.click('text=记忆分析仪表板')
    await page.waitForURL('**/memory-analytics')
    
    // 验证页面标题
    await expect(page.locator('h1')).toContainText('记忆系统分析仪表板')
    
    // 验证页面描述
    await expect(page.locator('p')).toContainText('全面展示记忆系统的运行状态、使用模式和性能指标')
    
    // 验证核心指标卡片
    await expect(page.locator('text=总记忆数')).toBeVisible()
    await expect(page.locator('text=平均重要性')).toBeVisible()
    await expect(page.locator('text=记忆增长率')).toBeVisible()
    await expect(page.locator('text=存储使用')).toBeVisible()
    
    // 验证分析标签页
    await expect(page.locator('text=记忆分布')).toBeVisible()
    await expect(page.locator('text=使用趋势')).toBeVisible()
    await expect(page.locator('text=访问模式')).toBeVisible()
    await expect(page.locator('text=网络分析')).toBeVisible()
    
    // 测试标签页切换
    await page.click('text=使用趋势')
    await expect(page.locator('text=30天记忆增长趋势')).toBeVisible()
    
    await page.click('text=访问模式')
    await expect(page.locator('text=高频访问记忆')).toBeVisible()
    await expect(page.locator('text=最近创建记忆')).toBeVisible()
    
    await page.click('text=网络分析')
    await expect(page.locator('text=节点总数')).toBeVisible()
    await expect(page.locator('text=关联边数')).toBeVisible()
    await expect(page.locator('text=平均连接度')).toBeVisible()
  })

  test('记忆系统导航和页面切换', async ({ page }) => {
    // 测试从主页导航到各个记忆页面
    
    // 1. 记忆层级架构
    await page.click('text=记忆层级架构')
    await page.waitForURL('**/memory-hierarchy')
    await expect(page.locator('h1')).toContainText('记忆层级系统')
    
    // 2. 记忆召回测试
    await page.click('text=记忆召回测试')
    await page.waitForURL('**/memory-recall')
    await expect(page.locator('h1')).toContainText('记忆召回测试')
    
    // 3. 记忆分析仪表板
    await page.click('text=记忆分析仪表板')
    await page.waitForURL('**/memory-analytics')
    await expect(page.locator('h1')).toContainText('记忆系统分析仪表板')
    
    // 4. 返回主页
    await page.click('text=单代理对话')
    await page.waitForURL(/\/chat|\//)
  })

  test('记忆层级页面交互功能', async ({ page }) => {
    await page.goto('http://localhost:3000/memory-hierarchy')
    
    // 等待页面加载
    await page.waitForTimeout(2000)
    
    // 测试刷新按钮
    const refreshButton = page.locator('button:has-text("刷新")')
    await refreshButton.click()
    
    // 测试巩固记忆按钮
    const consolidateButton = page.locator('button:has-text("巩固记忆")')
    await consolidateButton.click()
    
    // 验证记忆卡片存在（如果有数据）
    const memoryCards = page.locator('.ant-card-small')
    if (await memoryCards.count() > 0) {
      // 点击第一个记忆卡片
      await memoryCards.first().click()
    }
  })

  test('记忆召回页面交互功能', async ({ page }) => {
    await page.goto('http://localhost:3000/memory-recall')
    
    // 等待页面加载
    await page.waitForTimeout(2000)
    
    // 测试不同召回模式
    const modes = ['混合召回', '向量搜索', '时间搜索', '实体搜索']
    
    for (const mode of modes) {
      await page.click(`button:has-text("${mode}")`)
      await expect(page.locator(`button:has-text("${mode}")`).first()).toHaveClass(/ant-btn-primary/)
    }
    
    // 测试查询输入和搜索
    const queryInput = page.locator('textarea[placeholder*="输入查询内容"]')
    await queryInput.fill('React Hooks的使用方法')
    
    const searchButton = page.locator('button:has-text("执行召回")')
    await searchButton.click()
    
    // 等待搜索结果
    await page.waitForTimeout(1000)
  })

  test('记忆分析仪表板图表显示', async ({ page }) => {
    await page.goto('http://localhost:3000/memory-analytics')
    
    // 等待页面加载
    await page.waitForTimeout(3000)
    
    // 验证记忆分布标签页
    await page.click('text=记忆分布')
    
    // 检查图表容器是否存在
    const chartContainers = page.locator('.ant-card .ant-card-body')
    await expect(chartContainers.first()).toBeVisible()
    
    // 测试使用趋势标签页
    await page.click('text=使用趋势')
    await page.waitForTimeout(1000)
    
    // 验证趋势图标题
    await expect(page.locator('text=30天记忆增长趋势')).toBeVisible()
    
    // 测试访问模式标签页
    await page.click('text=访问模式')
    await page.waitForTimeout(1000)
    
    // 测试网络分析标签页
    await page.click('text=网络分析')
    await page.waitForTimeout(1000)
  })

  test('响应式设计测试', async ({ page }) => {
    // 测试桌面视图
    await page.setViewportSize({ width: 1920, height: 1080 })
    await page.goto('http://localhost:3000/memory-hierarchy')
    
    // 验证三列布局
    const columns = page.locator('.ant-col-8')
    await expect(columns).toHaveCount(3)
    
    // 测试平板视图
    await page.setViewportSize({ width: 768, height: 1024 })
    await page.reload()
    await page.waitForTimeout(1000)
    
    // 测试手机视图
    await page.setViewportSize({ width: 375, height: 667 })
    await page.reload()
    await page.waitForTimeout(1000)
    
    // 验证页面仍然可用
    await expect(page.locator('h1')).toContainText('记忆层级系统')
  })

  test('错误处理和边界情况', async ({ page }) => {
    await page.goto('http://localhost:3000/memory-recall')
    
    // 测试空查询搜索
    const searchButton = page.locator('button:has-text("执行召回")')
    await searchButton.click()
    
    // 测试长查询文本
    const queryInput = page.locator('textarea[placeholder*="输入查询内容"]')
    const longQuery = 'A'.repeat(1000)
    await queryInput.fill(longQuery)
    await searchButton.click()
    
    // 测试特殊字符查询
    await queryInput.fill('特殊字符!@#$%^&*()测试查询')
    await searchButton.click()
  })

  test('性能测试 - 页面加载时间', async ({ page }) => {
    const pages = [
      { url: '/memory-hierarchy', name: '记忆层级架构' },
      { url: '/memory-recall', name: '记忆召回测试' },
      { url: '/memory-analytics', name: '记忆分析仪表板' }
    ]
    
    for (const testPage of pages) {
      const startTime = Date.now()
      
      await page.goto(`http://localhost:3000${testPage.url}`)
      await page.waitForLoadState('networkidle')
      
      const loadTime = Date.now() - startTime
      console.log(`${testPage.name} 页面加载时间: ${loadTime}ms`)
      
      // 页面加载时间应该在合理范围内（5秒内）
      expect(loadTime).toBeLessThan(5000)
    }
  })
})