import { test, expect } from '@playwright/test'
import { setupAgenticRagApiMocks, setupRagApiMocks } from './mocks/rag-api'

test.describe('Agentic RAG系统集成测试', () => {
  test.beforeEach(async ({ page }) => {
    await setupAgenticRagApiMocks(page)
    await setupRagApiMocks(page)
  })

  test('基础RAG与Agentic RAG功能对比验证', async ({ page }) => {
    // 首先测试基础RAG功能
    await page.goto('/rag')
    
    // 基础RAG查询
    const basicQueryInput = page.locator('textarea[placeholder*="请输入"], input[placeholder*="搜索"], textarea[placeholder*="查询"]').first()
    await basicQueryInput.fill('对比测试查询')
    
    const basicSearchButton = page.locator('button:has-text("搜索"), button:has-text("查询"), button:has-text("检索")').first()
    await basicSearchButton.click()
    
    await page.waitForTimeout(2000)
    
    // 验证基础RAG结果
    await expect(page.locator('text=检索到的内容:对比测试查询')).toBeVisible()
    await expect(page.locator('text=相关代码片段:对比测试查询')).toBeVisible()
    
    // 记录基础RAG的结果特征
    const basicResultsCount = await page.locator('.result-item, [data-testid="result-item"]').count()
    
    // 切换到Agentic RAG
    await page.goto('/agentic-rag')
    await page.waitForTimeout(1000)
    
    // Agentic RAG查询
    const agenticQueryInput = page.locator('textarea[placeholder*="请输入您的查询问题"]').first()
    await agenticQueryInput.fill('对比测试查询')
    
    const agenticSearchButton = page.locator('button:has-text("智能检索")').first()
    await agenticSearchButton.click()
    
    await page.waitForTimeout(3000)
    
    // 验证Agentic RAG增强功能
    await expect(page.locator('text=智能分析结果:对比测试查询')).toBeVisible()
    await expect(page.locator('text=多代理协作发现:对比测试查询')).toBeVisible()
    
    // 验证Agentic RAG独有特性
    await expect(page.locator('text=exploratory').or(page.locator('text=意图分析'))).toBeVisible()
    await expect(page.locator('text=0.8').or(page.locator('text=80%')).first()).toBeVisible() // 置信度
    await expect(page.locator('text=semantic').or(page.locator('text=语义'))).toBeVisible()
    await expect(page.locator('text=keyword').or(page.locator('text=关键词'))).toBeVisible()
    
    // 验证处理过程展示（基础RAG没有）
    const processViewer = page.locator('[data-testid="process-viewer"], .retrieval-process-viewer').first()
    if (await processViewer.isVisible()) {
      await expect(processViewer.locator('text=查询分析')).toBeVisible()
      await expect(processViewer.locator('text=多代理检索')).toBeVisible()
      await expect(processViewer.locator('text=结果验证')).toBeVisible()
    }
  })

  test('跨页面导航和状态保持', async ({ page }) => {
    // 从基础RAG开始
    await page.goto('/rag')
    
    // 执行基础查询
    const basicQuery = '跨页面状态测试'
    const queryInput = page.locator('textarea[placeholder*="请输入"], input[placeholder*="搜索"], textarea[placeholder*="查询"]').first()
    await queryInput.fill(basicQuery)
    
    const searchButton = page.locator('button:has-text("搜索"), button:has-text("查询"), button:has-text("检索")').first()
    await searchButton.click()
    
    await page.waitForTimeout(1500)
    
    // 通过升级按钮跳转到Agentic RAG
    const upgradeButton = page.locator('button:has-text("升级到 Agentic RAG"), a:has-text("智能升级")').first()
    if (await upgradeButton.isVisible()) {
      await upgradeButton.click()
      
      // 验证跳转成功
      await expect(page).toHaveURL(/.*agentic-rag/)
      await expect(page.getByRole('heading', { name: /Agentic RAG.*智能系统/i })).toBeVisible()
    } else {
      // 手动导航
      await page.goto('/agentic-rag')
    }
    
    // 验证Agentic RAG页面状态
    await page.waitForTimeout(1000)
    
    // 检查是否保持了某些全局状态（如用户偏好）
    const sidebarStatus = page.locator('.ant-sider').first()
    if (await sidebarStatus.isVisible()) {
      // 验证基本系统状态显示
      await expect(sidebarStatus.locator('text=系统状态')).toBeVisible()
      await expect(sidebarStatus.locator('text=就绪, text=ready').first()).toBeVisible()
    }
    
    // 从Agentic RAG返回基础RAG
    await page.goto('/rag')
    await page.waitForTimeout(1000)
    
    // 验证基础RAG页面恢复
    await expect(page.locator('text=RAG 搜索')).toBeVisible()
    
    // 检查查询历史是否保持（如果有此功能）
    const historyButton = page.locator('button:has-text("历史"), text=历史记录').first()
    if (await historyButton.isVisible()) {
      await historyButton.click()
      // 验证之前的查询记录
      await expect(page.locator(`text=${basicQuery}`)).toBeVisible()
    }
  })

  test('统一的错误处理和恢复机制', async ({ page }) => {
    // 在基础RAG页面测试错误处理
    await page.goto('/rag')
    
    // 模拟网络错误
    await page.route('/api/v1/rag/query', (route) => {
      route.abort('failed')
    })
    
    const queryInput = page.locator('textarea[placeholder*="请输入"], input[placeholder*="搜索"], textarea[placeholder*="查询"]').first()
    await queryInput.fill('错误处理测试')
    
    const searchButton = page.locator('button:has-text("搜索"), button:has-text("查询"), button:has-text("检索")').first()
    await searchButton.click()
    
    // 验证基础RAG错误提示
    await expect(page.locator('text=网络连接异常').or(page.locator('text=系统错误'))).toBeVisible({ timeout: 5000 })
    
    // 切换到Agentic RAG
    await page.goto('/agentic-rag')
    
    // 设置Agentic RAG API错误
    await page.route('/api/v1/rag/agentic/query', (route) => {
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({
          success: false,
          error: 'Agentic服务暂时不可用',
          error_code: 'SERVICE_UNAVAILABLE',
          fallback_available: true
        })
      })
    })
    
    const agenticQueryInput = page.locator('textarea[placeholder*="请输入您的查询问题"]').first()
    await agenticQueryInput.fill('Agentic错误处理测试')
    
    const agenticSearchButton = page.locator('button:has-text("智能检索")').first()
    await agenticSearchButton.click()
    
    await page.waitForTimeout(2000)
    
    // 验证Agentic RAG的高级错误处理
    const fallbackHandler = page.locator('[data-testid="fallback-handler"], .fallback-handler').first()
    if (await fallbackHandler.isVisible()) {
      await expect(fallbackHandler.locator('text=Agentic服务暂时不可用')).toBeVisible()
      await expect(fallbackHandler.locator('text=SERVICE_UNAVAILABLE')).toBeVisible()
      
      // 验证fallback选项
      const fallbackButton = fallbackHandler.locator('button:has-text("备用策略"), button:has-text("降级模式")').first()
      if (await fallbackButton.isVisible()) {
        await fallbackButton.click()
        
        // 验证fallback到基础RAG
        await expect(page.locator('text=已切换到基础检索模式')).toBeVisible()
      }
    }
    
    // 恢复正常API
    await page.unroute('/api/v1/rag/query')
    await page.unroute('/api/v1/rag/agentic/query')
    await setupAgenticRagApiMocks(page)
    await setupRagApiMocks(page)
    
    // 测试恢复后的功能
    await agenticQueryInput.clear()
    await agenticQueryInput.fill('恢复测试')
    await agenticSearchButton.click()
    
    await page.waitForTimeout(2000)
    await expect(page.locator('text=智能分析结果:恢复测试')).toBeVisible()
  })

  test('性能对比和优化验证', async ({ page }) => {
    // 基础RAG性能基准测试
    await page.goto('/rag')
    
    const startTime = Date.now()
    
    const queryInput = page.locator('textarea[placeholder*="请输入"], input[placeholder*="搜索"], textarea[placeholder*="查询"]').first()
    await queryInput.fill('性能对比测试查询')
    
    const searchButton = page.locator('button:has-text("搜索"), button:has-text("查询"), button:has-text("检索")').first()
    await searchButton.click()
    
    // 等待基础RAG结果
    await expect(page.locator('text=检索到的内容:性能对比测试查询')).toBeVisible()
    const basicRagTime = Date.now() - startTime
    
    // Agentic RAG性能测试
    await page.goto('/agentic-rag')
    await page.waitForTimeout(1000)
    
    const agenticStartTime = Date.now()
    
    const agenticQueryInput = page.locator('textarea[placeholder*="请输入您的查询问题"]').first()
    await agenticQueryInput.fill('性能对比测试查询')
    
    const agenticSearchButton = page.locator('button:has-text("智能检索")').first()
    await agenticSearchButton.click()
    
    // 等待Agentic RAG结果
    await expect(page.locator('text=智能分析结果:性能对比测试查询')).toBeVisible()
    const agenticRagTime = Date.now() - agenticStartTime
    
    // 验证Agentic RAG提供了额外的性能信息
    const performanceInfo = page.locator('[data-testid="performance-info"], .performance-metrics').first()
    if (await performanceInfo.isVisible()) {
      await expect(performanceInfo.locator('text=272ms')).toBeVisible() // mock处理时间
    }
    
    // 在侧边栏验证性能指标显示
    const sidebarStats = page.locator('.ant-sider').first()
    if (await sidebarStats.isVisible()) {
      await expect(sidebarStats.locator('text=处理时间')).toBeVisible()
      await expect(sidebarStats.locator('text=272ms')).toBeVisible()
    }
    
    // 记录性能对比结果到控制台
    console.log(`基础RAG响应时间: ${basicRagTime}ms`)
    console.log(`Agentic RAG响应时间: ${agenticRagTime}ms`)
    
    // 验证Agentic RAG虽然可能更慢但提供更多价值
    expect(agenticRagTime).toBeGreaterThan(basicRagTime) // 通常智能分析会需要更多时间
    
    // 但是应该在合理范围内（小于10秒）
    expect(agenticRagTime).toBeLessThan(10000)
  })

  test('多浏览器标签页状态同步', async ({ page, context }) => {
    // 在第一个标签页打开Agentic RAG
    await page.goto('/agentic-rag')
    
    // 创建新会话
    const sessionManagerButton = page.locator('button:has-text("会话管理")').first()
    await sessionManagerButton.click()
    
    const sessionModal = page.locator('.ant-modal').first()
    const newSessionButton = sessionModal.locator('button:has-text("新建"), button:has-text("创建会话")').first()
    
    if (await newSessionButton.isVisible()) {
      await newSessionButton.click()
      const sessionNameInput = page.locator('input[placeholder*="会话名称"], input[placeholder*="名称"]').first()
      await sessionNameInput.fill('多标签页测试会话')
      
      const confirmButton = page.locator('button:has-text("确定"), button:has-text("创建")').first()
      await confirmButton.click()
      
      const closeButton = sessionModal.locator('.ant-modal-close').first()
      await closeButton.click()
    }
    
    // 在第一个标签页执行查询
    const queryInput = page.locator('textarea[placeholder*="请输入您的查询问题"]').first()
    await queryInput.fill('多标签页同步测试')
    
    const searchButton = page.locator('button:has-text("智能检索")').first()
    await searchButton.click()
    
    await page.waitForTimeout(2000)
    
    // 打开新标签页
    const newPage = await context.newPage()
    await setupAgenticRagApiMocks(newPage)
    await newPage.goto('/agentic-rag')
    
    await newPage.waitForTimeout(2000)
    
    // 验证新标签页中的状态同步
    const newPageSessionInfo = newPage.locator('text=多标签页测试会话').first()
    if (await newPageSessionInfo.isVisible()) {
      // 会话状态已同步
      console.log('会话状态已同步到新标签页')
    }
    
    // 在新标签页进行操作
    await newPageSessionInfo.click() // 如果需要切换会话
    
    // 验证查询历史是否同步（如果支持）
    const historyPanel = newPage.locator('[data-testid="query-history"], .query-history').first()
    if (await historyPanel.isVisible()) {
      await expect(historyPanel.locator('text=多标签页同步测试')).toBeVisible()
    }
    
    // 在新标签页执行新查询
    const newPageQueryInput = newPage.locator('textarea[placeholder*="请输入您的查询问题"]').first()
    await newPageQueryInput.fill('新标签页查询')
    
    const newPageSearchButton = newPage.locator('button:has-text("智能检索")').first()
    await newPageSearchButton.click()
    
    await newPage.waitForTimeout(2000)
    
    // 回到原标签页验证同步
    await page.bringToFront()
    
    // 检查原标签页是否收到新查询的更新（通过WebSocket或轮询）
    // 这取决于具体的实现方式
    
    await newPage.close()
  })

  test('完整的用户工作流程端到端测试', async ({ page }) => {
    // 模拟一个完整的用户使用场景
    
    // 1. 用户首次访问基础RAG
    await page.goto('/rag')
    await expect(page.locator('text=RAG 搜索')).toBeVisible()
    
    // 2. 进行简单搜索了解系统
    const basicQueryInput = page.locator('textarea[placeholder*="请输入"], input[placeholder*="搜索"], textarea[placeholder*="查询"]').first()
    await basicQueryInput.fill('什么是人工智能？')
    
    const basicSearchButton = page.locator('button:has-text("搜索"), button:has-text("查询"), button:has-text("检索")').first()
    await basicSearchButton.click()
    
    await page.waitForTimeout(1500)
    await expect(page.locator('text=检索到的内容:什么是人工智能？')).toBeVisible()
    
    // 3. 发现升级提示，点击体验智能版本
    const upgradeButton = page.locator('button:has-text("升级到 Agentic RAG"), a:has-text("智能升级")').first()
    if (await upgradeButton.isVisible()) {
      await upgradeButton.click()
    } else {
      await page.goto('/agentic-rag')
    }
    
    await page.waitForTimeout(1000)
    
    // 4. 首次进入显示使用指南
    const tourModal = page.locator('.ant-tour').first()
    if (await tourModal.isVisible()) {
      // 快速跳过引导或完成引导
      const skipButton = tourModal.locator('button:has-text("跳过"), .ant-tour-close').first()
      if (await skipButton.isVisible()) {
        await skipButton.click()
      }
    }
    
    // 5. 创建个人工作会话
    const sessionManagerButton = page.locator('button:has-text("会话管理")').first()
    await sessionManagerButton.click()
    
    const sessionModal = page.locator('.ant-modal').first()
    const newSessionButton = sessionModal.locator('button:has-text("新建"), button:has-text("创建会话")').first()
    
    if (await newSessionButton.isVisible()) {
      await newSessionButton.click()
      const sessionNameInput = page.locator('input[placeholder*="会话名称"], input[placeholder*="名称"]').first()
      await sessionNameInput.fill('AI学习研究会话')
      
      const confirmButton = page.locator('button:has-text("确定"), button:has-text("创建")').first()
      await confirmButton.click()
      
      const closeButton = sessionModal.locator('.ant-modal-close').first()
      await closeButton.click()
    }
    
    // 6. 进行一系列深入的研究查询
    const researchQueries = [
      '机器学习的核心算法有哪些？',
      '深度学习和机器学习的区别',
      '如何设计神经网络架构？',
      '自然语言处理的最新进展'
    ]
    
    for (const query of researchQueries) {
      const queryInput = page.locator('textarea[placeholder*="请输入您的查询问题"]').first()
      await queryInput.clear()
      await queryInput.fill(query)
      
      const searchButton = page.locator('button:has-text("智能检索")').first()
      await searchButton.click()
      
      // 等待结果并观察处理过程
      await page.waitForTimeout(2000)
      
      // 验证智能分析结果
      await expect(page.locator(`text=智能分析结果:${query}`)).toBeVisible()
      
      // 查看解释和推理过程
      const explanationViewer = page.locator('[data-testid="explanation-viewer"], .explanation-viewer').first()
      if (await explanationViewer.isVisible()) {
        // 验证有详细的解释信息
        await expect(explanationViewer.locator('text=查询分析').or(explanationViewer.locator('text=策略选择'))).toBeVisible()
      }
      
      // 提供反馈以改进系统
      const feedbackInterface = page.locator('[data-testid="feedback-interface"], .feedback-interface').first()
      if (await feedbackInterface.isVisible()) {
        const rating = feedbackInterface.locator('.ant-rate').first()
        if (await rating.isVisible()) {
          // 随机给出4-5星评分
          const starIndex = Math.floor(Math.random() * 2) + 3 // 3或4（对应4或5星）
          await rating.locator('.ant-rate-star').nth(starIndex).click()
        }
        
        const quickFeedback = feedbackInterface.locator('button:has-text("有用"), button:has-text("准确")').first()
        if (await quickFeedback.isVisible()) {
          await quickFeedback.click()
        }
      }
      
      // 短暂等待避免请求过快
      await page.waitForTimeout(500)
    }
    
    // 7. 查看研究会话的统计和历史
    await sessionManagerButton.click()
    
    if (await sessionModal.isVisible()) {
      // 验证查询历史记录
      const historySection = sessionModal.locator('[data-testid="session-history"], .session-history').first()
      if (await historySection.isVisible()) {
        await expect(historySection.locator('text=机器学习的核心算法')).toBeVisible()
        await expect(historySection.locator('text=深度学习和机器学习的区别')).toBeVisible()
      }
      
      // 验证会话统计
      const sessionStats = sessionModal.locator('[data-testid="session-stats"], .session-stats').first()
      if (await sessionStats.isVisible()) {
        await expect(sessionStats.locator('text=4').or(sessionStats.locator('text=查询次数'))).toBeVisible()
      }
      
      const closeButton = sessionModal.locator('.ant-modal-close').first()
      await closeButton.click()
    }
    
    // 8. 导出研究成果（如果支持）
    const exportButton = page.locator('button:has-text("导出"), button:has-text("分享")').first()
    if (await exportButton.isVisible()) {
      await exportButton.click()
      
      // 验证导出功能
      const exportModal = page.locator('.ant-modal').first()
      if (await exportModal.isVisible()) {
        const exportFormatSelect = exportModal.locator('select').first()
        if (await exportFormatSelect.isVisible()) {
          await exportFormatSelect.selectOption('markdown')
        }
        
        const confirmExportButton = exportModal.locator('button:has-text("确定"), button:has-text("导出")').first()
        if (await confirmExportButton.isVisible()) {
          await confirmExportButton.click()
          await expect(page.locator('text=导出成功')).toBeVisible()
        }
      }
    }
    
    // 9. 查看系统使用统计
    const performanceButton = page.locator('button:has-text("性能统计"), button:has-text("统计")').first()
    if (await performanceButton.isVisible()) {
      await performanceButton.click()
      
      const statsModal = page.locator('.ant-modal').first()
      await expect(statsModal.locator('text=总查询次数')).toBeVisible()
      await expect(statsModal.locator('text=平均置信度')).toBeVisible()
      
      const closeButton = statsModal.locator('.ant-modal-close').first()
      await closeButton.click()
    }
    
    // 10. 完成研究，用户满意地离开系统
    // 验证最终状态
    await expect(page.locator('text=AI学习研究会话')).toBeVisible()
    
    // 用户体验评分（在实际场景中可能通过其他方式收集）
    console.log('完整用户工作流程测试完成')
    console.log('用户成功完成了从基础RAG到智能RAG的完整体验流程')
  })
})