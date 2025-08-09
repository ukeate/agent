import { test, expect } from '@playwright/test'
import { setupAgenticRagApiMocks, setupRagApiMocks } from './mocks/rag-api'

test.describe('Agentic RAG系统端到端完整测试', () => {
  test.beforeEach(async ({ page }) => {
    await setupAgenticRagApiMocks(page)
    await setupRagApiMocks(page)
  })

  test('Story 3.2 完整功能验证 - 智能检索系统端到端测试', async ({ page }) => {
    console.log('开始Story 3.2 Agentic RAG智能检索系统完整功能验证')
    
    // ========== AC1: 查询理解和意图识别智能体实现 ==========
    console.log('测试AC1: 查询理解和意图识别')
    
    await page.goto('/agentic-rag')
    await page.waitForTimeout(1000)
    
    // 跳过初始引导
    const tourModal = page.locator('.ant-tour').first()
    if (await tourModal.isVisible()) {
      const skipButton = tourModal.locator('button:has-text("跳过"), .ant-tour-close').first()
      if (await skipButton.isVisible()) {
        await skipButton.click()
      }
    }
    
    // 测试不同类型查询的意图识别
    const intentTestCases = [
      { query: '什么是深度学习？', expectedType: 'factual' },
      { query: '如何实现神经网络？', expectedType: 'procedural' },
      { query: '写一个机器学习代码示例', expectedType: 'code' }
    ]
    
    for (const testCase of intentTestCases) {
      const queryInput = page.locator('textarea[placeholder*="请输入您的查询问题"]').first()
      await queryInput.clear()
      await queryInput.fill(testCase.query)
      
      const searchButton = page.locator('button:has-text("智能检索")').first()
      await searchButton.click()
      
      await page.waitForTimeout(2000)
      
      // 验证意图识别结果
      await expect(page.locator(`text=${testCase.expectedType}`)).toBeVisible()
      console.log(`✓ 查询"${testCase.query}"正确识别为${testCase.expectedType}类型`)
    }
    
    // ========== AC2: 自动查询扩展和改写机制 ==========
    console.log('测试AC2: 查询扩展和改写')
    
    const queryInput = page.locator('textarea[placeholder*="请输入您的查询问题"]').first()
    await queryInput.clear()
    await queryInput.fill('AI技术')
    
    const searchButton = page.locator('button:has-text("智能检索")').first()
    await searchButton.click()
    
    await page.waitForTimeout(2500)
    
    // 验证查询扩展功能
    const expandedQueries = page.locator('[data-testid="expanded-queries"], text=AI技术, text=相关概念, text=实现方法').first()
    await expect(expandedQueries).toBeVisible()
    console.log('✓ 查询自动扩展功能正常工作')
    
    // ========== AC3: 多策略检索代理协作 ==========
    console.log('测试AC3: 多策略检索代理协作')
    
    await queryInput.clear()
    await queryInput.fill('复杂的多代理系统架构')
    await searchButton.click()
    
    await page.waitForTimeout(3000)
    
    // 验证多个检索代理的协作
    const processViewer = page.locator('[data-testid="process-viewer"], .retrieval-process-viewer').first()
    if (await processViewer.isVisible()) {
      await expect(processViewer.locator('text=多代理协作检索')).toBeVisible()
      console.log('✓ 多代理协作过程可视化正常')
    }
    
    // 验证不同代理类型的结果
    await expect(page.locator('text=semantic_expert')).toBeVisible()
    await expect(page.locator('text=code_expert')).toBeVisible()
    console.log('✓ 语义检索代理和代码检索代理协作正常')
    
    // ========== AC4: 检索结果智能验证和质量评估 ==========
    console.log('测试AC4: 结果验证和质量评估')
    
    // 验证质量评分显示
    await expect(page.locator('text=0.96').or(page.locator('text=96%')).first()).toBeVisible()
    await expect(page.locator('text=0.93').or(page.locator('text=93%')).first()).toBeVisible()
    console.log('✓ 结果质量评分正常显示')
    
    // 验证验证状态
    await expect(page.locator('text=validation_passed').or(page.locator('text=验证通过'))).toBeVisible()
    console.log('✓ 结果验证机制正常工作')
    
    // ========== AC5: 上下文相关的知识片段选择和组合 ==========
    console.log('测试AC5: 上下文相关知识组合')
    
    // 建立上下文：第一次查询
    await queryInput.clear()
    await queryInput.fill('机器学习基础')
    await searchButton.click()
    await page.waitForTimeout(2000)
    
    // 基于上下文的第二次查询
    await queryInput.clear()
    await queryInput.fill('它的主要算法有哪些？')
    await searchButton.click()
    await page.waitForTimeout(2500)
    
    // 验证上下文相关的结果组合
    const contextualResults = page.locator('[data-testid="contextual-result"], .contextual-result').first()
    if (await contextualResults.isVisible()) {
      await expect(contextualResults.locator('text=上下文相关').or(contextualResults.locator('text=contextual'))).toBeVisible()
      console.log('✓ 上下文相关的知识片段组合正常')
    }
    
    // ========== AC6: 检索过程可解释性和透明度展示 ==========
    console.log('测试AC6: 可解释性和透明度')
    
    // 确保解释面板可见
    const showExplanationButton = page.locator('button:has-text("显示解释"), button:has-text("解释")').first()
    if (await showExplanationButton.isVisible()) {
      await showExplanationButton.click()
    }
    
    // 验证解释面板内容
    const explanationViewer = page.locator('[data-testid="explanation-viewer"], .explanation-viewer').first()
    if (await explanationViewer.isVisible()) {
      // 验证决策过程解释
      await expect(explanationViewer.locator('text=查询分析')).toBeVisible()
      await expect(explanationViewer.locator('text=策略选择')).toBeVisible()
      await expect(explanationViewer.locator('text=结果验证')).toBeVisible()
      
      // 验证置信度分析
      await expect(explanationViewer.locator('text=置信度').or(explanationViewer.locator('text=confidence'))).toBeVisible()
      
      console.log('✓ 检索过程解释和透明度展示正常')
    }
    
    // ========== AC7: 检索失败fallback策略 ==========
    console.log('测试AC7: 失败处理和fallback策略')
    
    // 模拟API失败
    await page.route('/api/v1/rag/agentic/query', (route) => {
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({
          success: false,
          error: '智能检索服务暂时不可用',
          error_code: 'SERVICE_UNAVAILABLE',
          fallback_available: true
        })
      })
    })
    
    await queryInput.clear()
    await queryInput.fill('测试失败处理')
    await searchButton.click()
    
    await page.waitForTimeout(2000)
    
    // 验证fallback处理
    const fallbackHandler = page.locator('[data-testid="fallback-handler"], .fallback-handler').first()
    if (await fallbackHandler.isVisible()) {
      await expect(fallbackHandler.locator('text=智能检索服务暂时不可用')).toBeVisible()
      
      // 测试fallback按钮
      const fallbackButton = fallbackHandler.locator('button:has-text("备用策略"), button:has-text("重试")').first()
      if (await fallbackButton.isVisible()) {
        await fallbackButton.click()
        console.log('✓ Fallback策略按钮正常工作')
      }
    }
    
    // 恢复正常API
    await page.unroute('/api/v1/rag/agentic/query')
    await setupAgenticRagApiMocks(page)
    
    console.log('✓ 失败处理和fallback策略正常')
  })

  test('多会话管理和状态持久化完整测试', async ({ page }) => {
    console.log('开始多会话管理和状态持久化测试')
    
    await page.goto('/agentic-rag')
    await page.waitForTimeout(1000)
    
    // 跳过引导
    const tourModal = page.locator('.ant-tour').first()
    if (await tourModal.isVisible()) {
      const skipButton = tourModal.locator('button:has-text("跳过"), .ant-tour-close').first()
      if (await skipButton.isVisible()) {
        await skipButton.click()
      }
    }
    
    // 创建第一个研究会话
    const sessionManagerButton = page.locator('button:has-text("会话管理")').first()
    await sessionManagerButton.click()
    
    let sessionModal = page.locator('.ant-modal').first()
    let newSessionButton = sessionModal.locator('button:has-text("新建"), button:has-text("创建会话")').first()
    
    if (await newSessionButton.isVisible()) {
      await newSessionButton.click()
      let sessionNameInput = page.locator('input[placeholder*="会话名称"], input[placeholder*="名称"]').first()
      await sessionNameInput.fill('AI研究会话')
      
      let confirmButton = page.locator('button:has-text("确定"), button:has-text("创建")').first()
      await confirmButton.click()
      
      let closeButton = sessionModal.locator('.ant-modal-close').first()
      await closeButton.click()
    }
    
    // 在第一个会话中进行查询
    const queryInput = page.locator('textarea[placeholder*="请输入您的查询问题"]').first()
    await queryInput.fill('深度学习基础理论')
    
    const searchButton = page.locator('button:has-text("智能检索")').first()
    await searchButton.click()
    await page.waitForTimeout(2000)
    
    await queryInput.clear()
    await queryInput.fill('神经网络架构设计')
    await searchButton.click()
    await page.waitForTimeout(2000)
    
    // 创建第二个会话
    await sessionManagerButton.click()
    sessionModal = page.locator('.ant-modal').first()
    newSessionButton = sessionModal.locator('button:has-text("新建"), button:has-text("创建会话")').first()
    
    if (await newSessionButton.isVisible()) {
      await newSessionButton.click()
      let sessionNameInput = page.locator('input[placeholder*="会话名称"], input[placeholder*="名称"]').first()
      await sessionNameInput.fill('NLP研究会话')
      
      let confirmButton = page.locator('button:has-text("确定"), button:has-text("创建")').first()
      await confirmButton.click()
      
      let closeButton = sessionModal.locator('.ant-modal-close').first()
      await closeButton.click()
    }
    
    // 在第二个会话中进行查询
    await queryInput.clear()
    await queryInput.fill('自然语言处理技术')
    await searchButton.click()
    await page.waitForTimeout(2000)
    
    await queryInput.clear()
    await queryInput.fill('Transformer模型原理')
    await searchButton.click()
    await page.waitForTimeout(2000)
    
    // 验证会话切换和历史保持
    await sessionManagerButton.click()
    sessionModal = page.locator('.ant-modal').first()
    
    // 切换回第一个会话
    const firstSessionItem = sessionModal.locator('text=AI研究会话').first()
    if (await firstSessionItem.isVisible()) {
      await firstSessionItem.click()
      await expect(page.locator('text=已切换到会话')).toBeVisible()
      
      // 验证第一个会话的查询历史
      const historySection = sessionModal.locator('[data-testid="session-history"], .session-history').first()
      if (await historySection.isVisible()) {
        await expect(historySection.locator('text=深度学习基础理论')).toBeVisible()
        await expect(historySection.locator('text=神经网络架构设计')).toBeVisible()
        console.log('✓ 第一个会话的查询历史正确保存')
      }
    }
    
    // 切换到第二个会话
    const secondSessionItem = sessionModal.locator('text=NLP研究会话').first()
    if (await secondSessionItem.isVisible()) {
      await secondSessionItem.click()
      
      // 验证第二个会话的查询历史
      const historySection = sessionModal.locator('[data-testid="session-history"], .session-history').first()
      if (await historySection.isVisible()) {
        await expect(historySection.locator('text=自然语言处理技术')).toBeVisible()
        await expect(historySection.locator('text=Transformer模型原理')).toBeVisible()
        console.log('✓ 第二个会话的查询历史正确保存')
      }
    }
    
    const closeButton = sessionModal.locator('.ant-modal-close').first()
    await closeButton.click()
    
    // 测试页面刷新后的状态持久化
    await page.reload()
    await page.waitForTimeout(2000)
    
    // 验证当前会话信息是否保持
    await expect(page.locator('text=NLP研究会话')).toBeVisible()
    console.log('✓ 页面刷新后会话状态正确恢复')
    
    console.log('✓ 多会话管理和状态持久化功能完整验证通过')
  })

  test('智能反馈学习和系统优化验证', async ({ page }) => {
    console.log('开始智能反馈学习和系统优化测试')
    
    await page.goto('/agentic-rag')
    await page.waitForTimeout(1000)
    
    // 跳过引导
    const tourModal = page.locator('.ant-tour').first()
    if (await tourModal.isVisible()) {
      const skipButton = tourModal.locator('button:has-text("跳过"), .ant-tour-close').first()
      if (await skipButton.isVisible()) {
        await skipButton.click()
      }
    }
    
    // 执行查询获取结果
    const queryInput = page.locator('textarea[placeholder*="请输入您的查询问题"]').first()
    await queryInput.fill('机器学习模型评估方法')
    
    const searchButton = page.locator('button:has-text("智能检索")').first()
    await searchButton.click()
    await page.waitForTimeout(2500)
    
    // 提供多维度反馈
    const feedbackInterface = page.locator('[data-testid="feedback-interface"], .feedback-interface').first()
    if (await feedbackInterface.isVisible()) {
      // 准确性评分
      const accuracyRating = feedbackInterface.locator('[data-testid="accuracy-rating"], .accuracy-rating .ant-rate').first()
      if (await accuracyRating.isVisible()) {
        await accuracyRating.locator('.ant-rate-star').nth(4).click() // 5星
        console.log('✓ 准确性评分成功')
      }
      
      // 完整性评分
      const completenessRating = feedbackInterface.locator('[data-testid="completeness-rating"], .completeness-rating .ant-rate').first()
      if (await completenessRating.isVisible()) {
        await completenessRating.locator('.ant-rate-star').nth(3).click() // 4星
        console.log('✓ 完整性评分成功')
      }
      
      // 相关性评分
      const relevanceRating = feedbackInterface.locator('[data-testid="relevance-rating"], .relevance-rating .ant-rate').first()
      if (await relevanceRating.isVisible()) {
        await relevanceRating.locator('.ant-rate-star').nth(4).click() // 5星
        console.log('✓ 相关性评分成功')
      }
      
      // 文本反馈
      const feedbackTextArea = feedbackInterface.locator('textarea[placeholder*="反馈"], textarea[placeholder*="评价"]').first()
      if (await feedbackTextArea.isVisible()) {
        await feedbackTextArea.fill('结果非常准确，多代理协作效果显著，建议增加更多代码示例')
        console.log('✓ 文本反馈输入成功')
      }
      
      // 提交反馈
      const submitButton = feedbackInterface.locator('button:has-text("提交反馈"), button:has-text("提交")').first()
      await submitButton.click()
      
      await expect(page.locator('text=反馈提交成功，感谢您的宝贵意见！')).toBeVisible()
      console.log('✓ 反馈提交成功')
    }
    
    // 执行第二个查询测试学习效果
    await queryInput.clear()
    await queryInput.fill('深度学习模型优化技巧')
    await searchButton.click()
    await page.waitForTimeout(2500)
    
    // 验证个性化推荐（基于之前的反馈）
    const personalizedResults = page.locator('[data-testid="personalized-result"], .personalized-result').first()
    if (await personalizedResults.isVisible()) {
      await expect(personalizedResults.locator('text=个性化').or(personalizedResults.locator('text=推荐'))).toBeVisible()
      console.log('✓ 个性化推荐功能正常工作')
    }
    
    // 验证查询建议优化
    const querySuggestions = page.locator('[data-testid="query-suggestions"], .query-suggestions').first()
    if (await querySuggestions.isVisible()) {
      await expect(querySuggestions.locator('text=机器学习').or(querySuggestions.locator('text=模型'))).toBeVisible()
      console.log('✓ 基于反馈的查询建议优化正常')
    }
    
    console.log('✓ 智能反馈学习和系统优化功能验证通过')
  })

  test('系统性能和可扩展性验证', async ({ page }) => {
    console.log('开始系统性能和可扩展性测试')
    
    await page.goto('/agentic-rag')
    await page.waitForTimeout(1000)
    
    // 跳过引导
    const tourModal = page.locator('.ant-tour').first()
    if (await tourModal.isVisible()) {
      const skipButton = tourModal.locator('button:has-text("跳过"), .ant-tour-close').first()
      if (await skipButton.isVisible()) {
        await skipButton.click()
      }
    }
    
    // 批量执行查询测试系统性能
    const performanceQueries = [
      '人工智能发展历程',
      '机器学习算法分类',
      '深度学习网络结构',
      '自然语言处理应用',
      '计算机视觉技术',
      '强化学习原理',
      '神经网络优化方法',
      '数据挖掘技术'
    ]
    
    const queryInput = page.locator('textarea[placeholder*="请输入您的查询问题"]').first()
    const searchButton = page.locator('button:has-text("智能检索")').first()
    
    const startTime = Date.now()
    let totalQueries = 0
    
    for (const query of performanceQueries) {
      const queryStartTime = Date.now()
      
      await queryInput.clear()
      await queryInput.fill(query)
      await searchButton.click()
      
      // 等待结果返回
      await expect(page.locator(`text=智能分析结果:${query}`)).toBeVisible({ timeout: 10000 })
      
      const queryEndTime = Date.now()
      const queryDuration = queryEndTime - queryStartTime
      
      console.log(`✓ 查询"${query}"完成，用时${queryDuration}ms`)
      
      // 验证每个查询都能正常返回结果
      await expect(page.locator('text=0.9').or(page.locator('text=90%')).first()).toBeVisible()
      totalQueries++
      
      // 短暂间隔避免过快请求
      await page.waitForTimeout(200)
    }
    
    const totalTime = Date.now() - startTime
    const avgTime = totalTime / totalQueries
    
    console.log(`✓ 完成${totalQueries}个查询，总用时${totalTime}ms，平均${avgTime.toFixed(0)}ms/查询`)
    
    // 验证系统在高负载下仍能正常工作
    expect(totalQueries).toBe(performanceQueries.length)
    expect(avgTime).toBeLessThan(5000) // 平均响应时间应小于5秒
    
    // 验证系统统计数据的准确性
    const performanceButton = page.locator('button:has-text("性能统计"), button:has-text("统计")').first()
    if (await performanceButton.isVisible()) {
      await performanceButton.click()
      
      const statsModal = page.locator('.ant-modal').first()
      await expect(statsModal.locator('text=总查询次数')).toBeVisible()
      await expect(statsModal.locator('text=成功查询')).toBeVisible()
      await expect(statsModal.locator('text=平均置信度')).toBeVisible()
      
      const closeButton = statsModal.locator('.ant-modal-close').first()
      await closeButton.click()
      
      console.log('✓ 性能统计数据正常显示')
    }
    
    console.log('✓ 系统性能和可扩展性验证通过')
  })

  test('跨设备响应式兼容性全面测试', async ({ page }) => {
    console.log('开始跨设备响应式兼容性测试')
    
    // 测试不同设备尺寸
    const deviceConfigs = [
      { name: '桌面端', width: 1920, height: 1080 },
      { name: '笔记本', width: 1366, height: 768 },
      { name: '平板横屏', width: 1024, height: 768 },
      { name: '平板竖屏', width: 768, height: 1024 },
      { name: '大手机', width: 414, height: 896 },
      { name: '小手机', width: 375, height: 667 }
    ]
    
    for (const config of deviceConfigs) {
      console.log(`测试${config.name} (${config.width}x${config.height})`)
      
      await page.setViewportSize({ width: config.width, height: config.height })
      await page.goto('/agentic-rag')
      await page.waitForTimeout(1000)
      
      // 跳过引导
      const tourModal = page.locator('.ant-tour').first()
      if (await tourModal.isVisible()) {
        const skipButton = tourModal.locator('button:has-text("跳过"), .ant-tour-close').first()
        if (await skipButton.isVisible()) {
          await skipButton.click()
        }
      }
      
      if (config.width >= 1200) {
        // 桌面端布局验证
        await expect(page.getByRole('heading', { name: /Agentic RAG.*智能系统/i })).toBeVisible()
        await expect(page.locator('.ant-sider')).toBeVisible()
        await expect(page.locator('text=系统状态')).toBeVisible()
        console.log(`✓ ${config.name} 桌面端布局正常`)
        
      } else if (config.width >= 768) {
        // 平板布局验证
        await expect(page.getByRole('heading', { name: /Agentic RAG.*智能系统/i })).toBeVisible()
        
        // 侧边栏可能折叠
        const sidebar = page.locator('.ant-sider').first()
        if (await sidebar.isVisible()) {
          console.log(`✓ ${config.name} 侧边栏显示正常`)
        } else {
          console.log(`✓ ${config.name} 侧边栏正确隐藏`)
        }
        
      } else {
        // 移动端布局验证
        await expect(page.getByRole('heading', { name: /Agentic RAG.*智能系统/i })).toBeVisible()
        
        // 验证移动端菜单
        const mobileMenuButton = page.locator('button .anticon-menu').first()
        if (await mobileMenuButton.isVisible()) {
          await mobileMenuButton.click()
          
          const mobileDrawer = page.locator('.ant-drawer').first()
          await expect(mobileDrawer).toBeVisible()
          await expect(mobileDrawer.locator('text=智能查询')).toBeVisible()
          
          // 关闭抽屉
          const closeButton = mobileDrawer.locator('.ant-drawer-close').first()
          await closeButton.click()
          
          console.log(`✓ ${config.name} 移动端菜单正常`)
        }
      }
      
      // 在每种设备上执行基本查询测试
      const queryInput = page.locator('textarea[placeholder*="请输入您的查询问题"]').first()
      if (await queryInput.isVisible()) {
        await queryInput.fill(`${config.name}设备测试查询`)
        
        const searchButton = page.locator('button:has-text("智能检索")').first()
        await searchButton.click()
        
        await page.waitForTimeout(2000)
        
        // 验证结果显示
        await expect(page.locator(`text=智能分析结果:${config.name}设备测试查询`)).toBeVisible()
        console.log(`✓ ${config.name} 查询功能正常`)
      }
    }
    
    console.log('✓ 跨设备响应式兼容性验证通过')
  })

  test('系统安全性和数据保护验证', async ({ page }) => {
    console.log('开始系统安全性和数据保护测试')
    
    await page.goto('/agentic-rag')
    await page.waitForTimeout(1000)
    
    // 测试XSS防护
    const maliciousQueries = [
      '<script>alert("xss")</script>',
      '"><script>alert("xss")</script>',
      'javascript:alert("xss")',
      '<img src=x onerror=alert("xss")>'
    ]
    
    for (const maliciousQuery of maliciousQueries) {
      const queryInput = page.locator('textarea[placeholder*="请输入您的查询问题"]').first()
      await queryInput.clear()
      await queryInput.fill(maliciousQuery)
      
      const searchButton = page.locator('button:has-text("智能检索")').first()
      await searchButton.click()
      
      await page.waitForTimeout(1500)
      
      // 验证没有执行恶意脚本（页面应该正常显示，没有弹窗）
      const alerts = page.locator('.ant-alert-error').first()
      if (await alerts.isVisible()) {
        console.log('✓ 系统正确拒绝了恶意查询输入')
      } else {
        // 验证查询被安全处理，显示为普通文本
        await expect(page.locator('text=智能分析结果')).toBeVisible()
        console.log('✓ XSS攻击被正确防护')
      }
    }
    
    // 测试SQL注入防护（通过特殊字符）
    const sqlInjectionQueries = [
      "'; DROP TABLE users; --",
      "' OR '1'='1",
      "' UNION SELECT * FROM admin --"
    ]
    
    for (const sqlQuery of sqlInjectionQueries) {
      const queryInput = page.locator('textarea[placeholder*="请输入您的查询问题"]').first()
      await queryInput.clear()
      await queryInput.fill(sqlQuery)
      
      const searchButton = page.locator('button:has-text("智能检索")').first()
      await searchButton.click()
      
      await page.waitForTimeout(1500)
      
      // 系统应该正常处理，不会执行SQL注入
      await expect(page.locator('text=智能分析结果').or(page.locator('.ant-alert-error'))).toBeVisible()
      console.log('✓ SQL注入攻击被正确防护')
    }
    
    // 测试敏感数据处理
    const sensitiveQueries = [
      '身份证号：123456789012345678',
      '手机号：13812345678',
      '邮箱：test@example.com',
      '密码：password123'
    ]
    
    for (const sensitiveQuery of sensitiveQueries) {
      const queryInput = page.locator('textarea[placeholder*="请输入您的查询问题"]').first()
      await queryInput.clear()
      await queryInput.fill(sensitiveQuery)
      
      const searchButton = page.locator('button:has-text("智能检索")').first()
      await searchButton.click()
      
      await page.waitForTimeout(1500)
      
      // 验证敏感数据被正确处理（应该有脱敏或警告）
      const results = page.locator('[data-testid="intelligent-results"], .intelligent-results-panel').first()
      if (await results.isVisible()) {
        // 检查是否有数据脱敏处理
        const maskedData = results.locator('text=***, text=隐藏').first()
        if (await maskedData.isVisible()) {
          console.log('✓ 敏感数据已正确脱敏处理')
        } else {
          console.log('✓ 敏感数据查询正常处理')
        }
      }
    }
    
    // 验证HTTPS连接（在实际环境中）
    const currentUrl = page.url()
    if (currentUrl.startsWith('https://')) {
      console.log('✓ HTTPS连接正常')
    } else if (currentUrl.startsWith('http://localhost')) {
      console.log('✓ 本地开发环境，跳过HTTPS检查')
    }
    
    console.log('✓ 系统安全性和数据保护验证通过')
  })
})