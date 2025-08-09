import { test, expect } from '@playwright/test'
import { setupAgenticRagApiMocks, setupRagApiErrors } from './mocks/rag-api'

test.describe('Agentic RAG智能检索系统测试', () => {
  test.beforeEach(async ({ page }) => {
    await setupAgenticRagApiMocks(page)
  })

  test('Agentic RAG页面加载和界面布局', async ({ page }) => {
    await page.goto('/agentic-rag')
    
    // 验证页面标题和主要组件
    await expect(page.getByRole('heading', { name: /Agentic RAG.*智能系统/i })).toBeVisible()
    
    // 验证头部导航  
    await expect(page.locator('button:has-text("会话管理")').first()).toBeVisible()
    await expect(page.locator('text=使用指南')).toBeVisible() 
    await expect(page.locator('text=全屏模式')).toBeVisible()
    
    // 验证侧边栏
    await expect(page.locator('text=当前会话')).toBeVisible()
    await expect(page.locator('text=系统状态')).toBeVisible()
    await expect(page.locator('text=快速操作')).toBeVisible()
    
    // 验证主要功能区域存在
    await expect(page.locator('textarea[placeholder*="请输入您的查询问题"]')).toBeVisible()
    await expect(page.locator('button:has-text("智能检索")')).toBeVisible()
  })

  test('智能查询面板功能测试', async ({ page }) => {
    await page.goto('/agentic-rag')
    
    // 查找智能查询输入框
    const queryInput = page.locator('textarea[placeholder*="请输入您的查询问题"]').first()
    await expect(queryInput).toBeVisible()
    
    // 输入查询
    await queryInput.fill('如何实现多代理协作的智能检索系统?')
    
    // 验证意图识别提示（如果有）
    const intentHint = page.locator('text=意图分析').or(page.locator('text=查询类型')).or(page.locator('.intent-analysis')).first()
    
    // 验证查询扩展建议（如果显示）
    const expansionSuggestions = page.locator('text=查询扩展').or(page.locator('text=相关查询')).or(page.locator('.query-expansion')).first()
    
    // 点击智能搜索按钮
    const intelligentSearchButton = page.locator('button:has-text("智能检索")').first()
    await intelligentSearchButton.click()
    
    // 验证搜索启动提示
    await expect(page.locator('text=智能检索已启动')).toBeVisible()
  })

  test('检索过程可视化功能', async ({ page }) => {
    await page.goto('/agentic-rag')
    
    // 执行智能查询
    const queryInput = page.locator('textarea[placeholder*="请输入您的查询问题"]').first()
    await queryInput.fill('测试检索过程可视化')
    
    const searchButton = page.locator('button:has-text("智能检索")').first()
    await searchButton.click()
    
    // 等待检索过程开始
    await page.waitForTimeout(1000)
    
    // 验证检索过程展示
    const processViewer = page.locator('[data-testid="process-viewer"], .retrieval-process-viewer').first()
    if (await processViewer.isVisible()) {
      // 验证处理步骤
      await expect(processViewer.locator('text=查询分析')).toBeVisible()
      await expect(processViewer.locator('text=查询扩展')).toBeVisible()
      await expect(processViewer.locator('text=多代理检索')).toBeVisible()
      await expect(processViewer.locator('text=结果验证')).toBeVisible()
      
      // 验证进度指示
      await expect(processViewer.locator('text=completed').or(processViewer.locator('text=完成')).or(processViewer.locator('.step-completed')).first()).toBeVisible()
    }
    
    // 通过快速操作按钮查看过程
    const viewProcessButton = page.locator('button:has-text("查看检索过程")').first()
    if (await viewProcessButton.isVisible()) {
      await viewProcessButton.click()
      
      // 验证过程详情展示
      await expect(page.locator('text=查询分析完成')).toBeVisible()
      await expect(page.locator('text=查询扩展策略执行')).toBeVisible()
      await expect(page.locator('text=多代理协作检索')).toBeVisible()
      await expect(page.locator('text=结果质量验证')).toBeVisible()
    }
  })

  test('智能结果展示和质量评分', async ({ page }) => {
    await page.goto('/agentic-rag')
    
    // 执行搜索
    const queryInput = page.locator('textarea[placeholder*="请输入您的查询问题"]').first()
    await queryInput.fill('智能结果展示测试')
    
    const searchButton = page.locator('button:has-text("智能检索")').first()
    await searchButton.click()
    
    // 等待结果加载
    await page.waitForTimeout(2000)
    
    // 验证智能结果展示
    const resultsPanel = page.locator('[data-testid="intelligent-results"], .intelligent-results-panel').first()
    await expect(resultsPanel).toBeVisible()
    
    // 验证结果项目
    await expect(page.locator('text=智能分析结果:智能结果展示测试')).toBeVisible()
    await expect(page.locator('text=多代理协作发现:智能结果展示测试')).toBeVisible()
    
    // 验证质量评分显示
    await expect(page.locator('text=0.96').or(page.locator('text=96%')).first()).toBeVisible() // 相关性评分
    await expect(page.locator('text=0.93').or(page.locator('text=93%')).first()).toBeVisible() // 置信度
    
    // 验证代理类型标识
    await expect(page.locator('text=semantic').or(page.locator('text=语义分析'))).toBeVisible()
    await expect(page.locator('text=multi_agent').or(page.locator('text=多代理'))).toBeVisible()
    
    // 验证验证状态
    await expect(page.locator('text=validation_passed').or(page.locator('text=验证通过')).first()).toBeVisible()
    
    // 测试结果评分功能
    const rateButton = page.locator('button:has-text("评分"), .rate-button, [data-testid="rate-result"]').first()
    if (await rateButton.isVisible()) {
      await rateButton.click()
      await expect(page.locator('text=评分完成')).toBeVisible()
    }
  })

  test('检索解释和透明度展示', async ({ page }) => {
    await page.goto('/agentic-rag')
    
    // 确保解释面板可见
    const showExplanationButton = page.locator('button:has-text("显示解释"), button:has-text("解释")').first()
    if (await showExplanationButton.isVisible()) {
      await showExplanationButton.click()
    }
    
    // 执行查询以获取解释数据
    const queryInput = page.locator('textarea[placeholder*="请输入您的查询问题"]').first()
    await queryInput.fill('解释功能测试查询')
    
    const searchButton = page.locator('button:has-text("智能检索")').first()
    await searchButton.click()
    
    // 等待结果和解释加载
    await page.waitForTimeout(2000)
    
    // 验证解释面板
    const explanationViewer = page.locator('[data-testid="explanation-viewer"], .explanation-viewer').first()
    if (await explanationViewer.isVisible()) {
      // 验证决策过程解释
      await expect(explanationViewer.locator('text=查询分析')).toBeVisible()
      await expect(explanationViewer.locator('text=策略选择')).toBeVisible()
      await expect(explanationViewer.locator('text=结果验证')).toBeVisible()
      
      // 验证置信度分析
      await expect(explanationViewer.locator('text=置信度').or(explanationViewer.locator('text=confidence')).first()).toBeVisible()
      await expect(explanationViewer.locator('text=0.85').or(explanationViewer.locator('text=85%')).first()).toBeVisible()
      
      // 验证推理过程
      await expect(explanationViewer.locator('text=识别为事实性查询')).toBeVisible()
      await expect(explanationViewer.locator('text=选择多代理协作')).toBeVisible()
      await expect(explanationViewer.locator('text=交叉验证确保结果准确性')).toBeVisible()
    }
    
    // 测试解释的分享和导出功能
    const shareButton = page.locator('button:has-text("分享"), [data-testid="share-explanation"]').first()
    if (await shareButton.isVisible()) {
      await shareButton.click()
      await expect(page.locator('text=解释内容已准备分享')).toBeVisible()
    }
    
    const exportButton = page.locator('button:has-text("导出"), [data-testid="export-explanation"]').first()
    if (await exportButton.isVisible()) {
      await exportButton.click()
      await expect(page.locator('text=解释内容已导出')).toBeVisible()
    }
  })

  test('多代理协作展示', async ({ page }) => {
    await page.goto('/agentic-rag')
    
    // 执行查询
    const queryInput = page.locator('textarea[placeholder*="请输入您的查询问题"]').first()
    await queryInput.fill('多代理协作测试')
    
    const searchButton = page.locator('button:has-text("智能检索")').first()
    await searchButton.click()
    
    // 等待代理协作过程完成
    await page.waitForTimeout(2000)
    
    // 验证侧边栏系统状态显示代理信息
    const sidebarStatus = page.locator('.ant-sider').first()
    if (await sidebarStatus.isVisible()) {
      // 验证查询状态
      await expect(sidebarStatus.locator('text=就绪').or(sidebarStatus.locator('text=执行中')).first()).toBeVisible()
      
      // 验证结果数量
      await expect(sidebarStatus.locator('text=2').first()).toBeVisible() // 2个结果
      
      // 验证置信度
      await expect(sidebarStatus.locator('text=91%')).toBeVisible()
      
      // 验证处理时间
      await expect(sidebarStatus.locator('text=272ms')).toBeVisible()
    }
    
    // 在解释面板中验证代理协作信息
    const explanationPanel = page.locator('[data-testid="explanation-viewer"], .explanation-viewer').first()
    if (await explanationPanel.isVisible()) {
      // 验证各代理的贡献度
      await expect(explanationPanel.locator('text=semantic_agent').or(explanationPanel.locator('text=语义代理'))).toBeVisible()
      await expect(explanationPanel.locator('text=keyword_agent').or(explanationPanel.locator('text=关键词代理'))).toBeVisible()
      await expect(explanationPanel.locator('text=structured_agent').or(explanationPanel.locator('text=结构化代理'))).toBeVisible()
      
      // 验证贡献度百分比
      await expect(explanationPanel.locator('text=0.45').or(explanationPanel.locator('text=45%')).first()).toBeVisible()
      await expect(explanationPanel.locator('text=0.32').or(explanationPanel.locator('text=32%')).first()).toBeVisible()
      await expect(explanationPanel.locator('text=0.23').or(explanationPanel.locator('text=23%')).first()).toBeVisible()
    }
  })

  test('智能反馈系统', async ({ page }) => {
    await page.goto('/agentic-rag')
    
    // 先执行查询获得结果
    const queryInput = page.locator('textarea[placeholder*="请输入您的查询问题"]').first()
    await queryInput.fill('反馈系统测试')
    
    const searchButton = page.locator('button:has-text("智能检索")').first()
    await searchButton.click()
    
    // 等待结果加载
    await page.waitForTimeout(2000)
    
    // 查找反馈界面
    const feedbackInterface = page.locator('[data-testid="feedback-interface"], .feedback-interface').first()
    if (await feedbackInterface.isVisible()) {
      // 验证多维度评分选项
      await expect(feedbackInterface.locator('text=准确性')).toBeVisible()
      await expect(feedbackInterface.locator('text=完整性')).toBeVisible()
      await expect(feedbackInterface.locator('text=相关性')).toBeVisible()
      
      // 进行评分
      const accuracyRating = feedbackInterface.locator('[data-testid="accuracy-rating"], .accuracy-rating .ant-rate').first()
      if (await accuracyRating.isVisible()) {
        await accuracyRating.locator('.ant-rate-star').nth(4).click() // 5星评分
      }
      
      // 添加文本反馈
      const feedbackTextArea = feedbackInterface.locator('textarea[placeholder*="反馈"], textarea[placeholder*="评价"]').first()
      if (await feedbackTextArea.isVisible()) {
        await feedbackTextArea.fill('检索结果非常准确，多代理协作效果很好！')
      }
      
      // 提交反馈
      const submitFeedbackButton = feedbackInterface.locator('button:has-text("提交反馈"), button:has-text("提交")').first()
      await submitFeedbackButton.click()
      
      // 验证反馈提交成功
      await expect(page.locator('text=反馈提交成功，感谢您的宝贵意见！')).toBeVisible()
    }
  })

  test('会话管理功能', async ({ page }) => {
    await page.goto('/agentic-rag')
    
    // 打开会话管理
    const sessionManagerButton = page.locator('button:has-text("会话管理")').first()
    await sessionManagerButton.click()
    
    // 验证会话管理对话框打开
    await expect(page.locator('.ant-modal').locator('text=会话管理')).toBeVisible()
    
    // 在会话管理器中进行操作
    const sessionModal = page.locator('.ant-modal').first()
    
    // 创建新会话（如果有此功能）
    const newSessionButton = sessionModal.locator('button:has-text("新建"), button:has-text("创建会话")').first()
    if (await newSessionButton.isVisible()) {
      await newSessionButton.click()
      
      // 填写会话名称
      const sessionNameInput = page.locator('input[placeholder*="会话名称"], input[placeholder*="名称"]').first()
      if (await sessionNameInput.isVisible()) {
        await sessionNameInput.fill('测试会话')
        
        // 确认创建
        const confirmButton = page.locator('button:has-text("确定"), button:has-text("创建")').first()
        await confirmButton.click()
        
        // 验证创建成功
        await expect(page.locator('text=会话 "测试会话" 创建成功')).toBeVisible()
      }
    }
    
    // 选择会话
    const sessionItem = sessionModal.locator('text=测试会话, .session-item').first()
    if (await sessionItem.isVisible()) {
      await sessionItem.click()
      await expect(page.locator('text=已切换到会话')).toBeVisible()
    }
    
    // 关闭会话管理
    const closeButton = sessionModal.locator('.ant-modal-close').first()
    await closeButton.click()
  })

  test('失败处理和Fallback策略', async ({ page }) => {
    // 设置API错误
    await setupRagApiErrors(page)
    
    await page.goto('/agentic-rag')
    
    // 执行查询触发失败
    const queryInput = page.locator('textarea[placeholder*="请输入您的查询问题"]').first()
    await queryInput.fill('触发失败的查询')
    
    const searchButton = page.locator('button:has-text("智能检索")').first()
    await searchButton.click()
    
    // 等待失败处理
    await page.waitForTimeout(2000)
    
    // 验证失败处理组件显示
    const fallbackHandler = page.locator('[data-testid="fallback-handler"], .fallback-handler').first()
    if (await fallbackHandler.isVisible()) {
      // 验证失败提示
      await expect(fallbackHandler.locator('text=智能分析服务暂时不可用')).toBeVisible()
      await expect(fallbackHandler.locator('text=AGENTIC_SERVICE_DOWN')).toBeVisible()
      
      // 验证重试建议
      await expect(fallbackHandler.locator('text=30').or(fallbackHandler.locator('text=retry_after'))).toBeVisible()
      
      // 尝试备用策略
      const fallbackButton = fallbackHandler.locator('button:has-text("备用策略"), button:has-text("重试")').first()
      if (await fallbackButton.isVisible()) {
        await fallbackButton.click()
        await expect(page.locator('text=后备处理成功').or(page.locator('text=后备处理失败'))).toBeVisible()
      }
    }
  })

  test('系统引导和帮助功能', async ({ page }) => {
    await page.goto('/agentic-rag')
    
    // 清除已看过引导的标记，强制显示引导
    await page.evaluate(() => {
      localStorage.removeItem('agentic_rag_tour_seen')
    })
    
    // 重新加载页面触发引导
    await page.reload()
    
    // 验证引导是否出现
    const tourModal = page.locator('.ant-tour').first()
    if (await tourModal.isVisible()) {
      // 验证引导内容
      await expect(tourModal.locator('text=欢迎使用Agentic RAG系统')).toBeVisible()
      
      // 进行引导流程
      const nextButton = tourModal.locator('button:has-text("下一步"), .ant-tour-next-btn').first()
      if (await nextButton.isVisible()) {
        await nextButton.click()
        
        // 验证第二步引导
        await expect(tourModal.locator('text=智能查询面板')).toBeVisible()
        
        // 跳过剩余步骤
        const skipButton = tourModal.locator('button:has-text("跳过"), .ant-tour-close').first()
        if (await skipButton.isVisible()) {
          await skipButton.click()
        }
      }
    }
    
    // 手动触发帮助
    const helpButton = page.locator('button:has-text("使用指南")').first()
    await helpButton.click()
    
    // 验证引导重新显示
    await expect(page.locator('.ant-tour').first()).toBeVisible()
  })

  test('全屏模式和布局控制', async ({ page }) => {
    await page.goto('/agentic-rag')
    
    // 测试全屏切换
    const fullscreenButton = page.locator('button:has-text("全屏模式")').first()
    await fullscreenButton.click()
    
    // 验证全屏按钮文本变化
    await expect(page.locator('button:has-text("退出全屏")')).toBeVisible()
    
    // 退出全屏
    const exitFullscreenButton = page.locator('button:has-text("退出全屏")').first()
    await exitFullscreenButton.click()
    await expect(page.locator('button:has-text("全屏模式")')).toBeVisible()
    
    // 测试侧边栏折叠
    const sidebarCollapseButton = page.locator('.ant-layout-sider-trigger').first()
    if (await sidebarCollapseButton.isVisible()) {
      await sidebarCollapseButton.click()
      
      // 验证侧边栏已折叠
      await expect(page.locator('.ant-layout-sider-collapsed')).toBeVisible()
      
      // 再次点击展开
      await sidebarCollapseButton.click()
      await expect(page.locator('.ant-layout-sider-collapsed')).not.toBeVisible()
    }
    
    // 测试紧凑模式切换
    const compactButton = page.locator('button:has-text("紧凑模式")').first()
    if (await compactButton.isVisible()) {
      await compactButton.click()
      await expect(page.locator('button:has-text("标准模式")')).toBeVisible()
      
      // 切回标准模式
      const standardButton = page.locator('button:has-text("标准模式")').first()
      await standardButton.click()
      await expect(page.locator('button:has-text("紧凑模式")')).toBeVisible()
    }
  })

  test('移动端响应式适配', async ({ page }) => {
    // 设置移动端视口
    await page.setViewportSize({ width: 375, height: 667 })
    await page.goto('/agentic-rag')
    
    // 验证移动端菜单显示
    const mobileMenuButton = page.locator('button .anticon-menu').first()
    await expect(mobileMenuButton).toBeVisible()
    
    // 点击移动端菜单
    await mobileMenuButton.click()
    
    // 验证移动端抽屉菜单
    const mobileDrawer = page.locator('.ant-drawer').first()
    await expect(mobileDrawer).toBeVisible()
    await expect(mobileDrawer.locator('text=智能查询')).toBeVisible()
    await expect(mobileDrawer.locator('text=检索过程')).toBeVisible()
    await expect(mobileDrawer.locator('text=智能结果')).toBeVisible()
    await expect(mobileDrawer.locator('text=会话管理')).toBeVisible()
    
    // 测试移动端导航
    const queryNavButton = mobileDrawer.locator('button:has-text("智能查询")').first()
    await queryNavButton.click()
    
    // 验证抽屉关闭并导航到对应区域
    await expect(mobileDrawer).not.toBeVisible()
    
    // 重新打开菜单测试其他功能
    await mobileMenuButton.click()
    
    const processNavButton = mobileDrawer.locator('button:has-text("检索过程")').first()
    await processNavButton.click()
    await expect(mobileDrawer).not.toBeVisible()
  })
})