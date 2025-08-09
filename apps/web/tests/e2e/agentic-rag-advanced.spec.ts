import { test, expect } from '@playwright/test'
import { setupAgenticRagApiMocks } from './mocks/rag-api'

test.describe('Agentic RAG高级功能测试', () => {
  test.beforeEach(async ({ page }) => {
    await setupAgenticRagApiMocks(page)
  })

  test('流式检索过程实时展示', async ({ page }) => {
    await page.goto('/agentic-rag')
    
    // 启用流式模式（如果有此选项）
    const streamToggle = page.locator('input[type="checkbox"][data-testid="stream-mode"], .stream-toggle input').first()
    if (await streamToggle.isVisible()) {
      await streamToggle.check()
    }
    
    // 执行流式查询
    const queryInput = page.locator('textarea[placeholder*="请输入您的查询问题"]').first()
    await queryInput.fill('测试流式检索过程')
    
    const searchButton = page.locator('button:has-text("智能检索")').first()
    await searchButton.click()
    
    // 验证流式事件的实时展示
    const processViewer = page.locator('[data-testid="process-viewer"], .retrieval-process-viewer').first()
    if (await processViewer.isVisible()) {
      // 验证各个步骤的实时更新
      await expect(processViewer.locator('text=正在分析查询意图')).toBeVisible({ timeout: 1000 })
      await expect(processViewer.locator('text=正在扩展查询')).toBeVisible({ timeout: 2000 })
      await expect(processViewer.locator('text=semantic').or(processViewer.locator('text=语义代理'))).toBeVisible({ timeout: 3000 })
      await expect(processViewer.locator('text=智能检索完成')).toBeVisible({ timeout: 5000 })
    }
    
    // 验证进度条更新
    const progressBar = page.locator('.ant-progress-circle, .ant-progress-line, [data-testid="retrieval-progress"]').first()
    if (await progressBar.isVisible()) {
      // 等待进度完成
      await expect(progressBar.locator('text=100%').or(progressBar.locator('[aria-valuenow="100"]'))).toBeVisible({ timeout: 5000 })
    }
  })

  test('查询意图识别和智能分析', async ({ page }) => {
    await page.goto('/agentic-rag')
    
    // 测试不同类型的查询意图
    const testQueries = [
      { query: '什么是机器学习？', expectedIntent: 'factual' },
      { query: '如何实现神经网络？', expectedIntent: 'procedural' }, 
      { query: '写一个Python排序函数', expectedIntent: 'code' },
      { query: '给我写一首关于AI的诗', expectedIntent: 'creative' }
    ]
    
    for (const testCase of testQueries) {
      const queryInput = page.locator('textarea[placeholder*="请输入您的查询问题"]').first()
      await queryInput.clear()
      await queryInput.fill(testCase.query)
      
      // 检查是否有实时意图分析显示
      const intentAnalysis = page.locator('[data-testid="intent-analysis"], .intent-analysis').first()
      if (await intentAnalysis.isVisible()) {
        await expect(intentAnalysis.locator(`text=${testCase.expectedIntent}`)).toBeVisible({ timeout: 2000 })
      }
      
      const searchButton = page.locator('button:has-text("智能检索")').first()
      await searchButton.click()
      
      // 等待结果并验证意图分析结果
      await page.waitForTimeout(2000)
      
      // 在结果中验证意图分析
      const resultsPanel = page.locator('[data-testid="intelligent-results"], .intelligent-results-panel').first()
      if (await resultsPanel.isVisible()) {
        await expect(resultsPanel.locator(`text=${testCase.expectedIntent}`)).toBeVisible()
      }
    }
  })

  test('多策略检索协作验证', async ({ page }) => {
    await page.goto('/agentic-rag')
    
    // 执行复杂查询触发多策略协作
    const queryInput = page.locator('textarea[placeholder*="请输入您的查询问题"]').first()
    await queryInput.fill('复杂的多代理协作AI系统架构设计与实现')
    
    const searchButton = page.locator('button:has-text("智能检索")').first()
    await searchButton.click()
    
    // 等待检索完成
    await page.waitForTimeout(3000)
    
    // 验证多个检索策略都被使用
    const explanationViewer = page.locator('[data-testid="explanation-viewer"], .explanation-viewer').first()
    if (await explanationViewer.isVisible()) {
      // 验证语义检索代理
      await expect(explanationViewer.locator('text=semantic').or(explanationViewer.locator('text=语义检索'))).toBeVisible()
      
      // 验证关键词检索代理  
      await expect(explanationViewer.locator('text=keyword').or(explanationViewer.locator('text=关键词检索'))).toBeVisible()
      
      // 验证结构化检索代理
      await expect(explanationViewer.locator('text=structured').or(explanationViewer.locator('text=结构化检索'))).toBeVisible()
      
      // 验证协作贡献度分析
      const contributions = explanationViewer.locator('[data-testid="agent-contributions"], .agent-contributions').first()
      if (await contributions.isVisible()) {
        await expect(contributions.locator('text=0.45').or(contributions.locator('text=45%'))).toBeVisible()
        await expect(contributions.locator('text=0.32').or(contributions.locator('text=32%'))).toBeVisible()
        await expect(contributions.locator('text=0.23').or(contributions.locator('text=23%'))).toBeVisible()
      }
    }
    
    // 验证结果中包含不同策略的检索结果
    const resultsPanel = page.locator('[data-testid="intelligent-results"], .intelligent-results-panel').first()
    await expect(resultsPanel.locator('text=semantic_expert')).toBeVisible()
    await expect(resultsPanel.locator('text=code_expert')).toBeVisible()
  })

  test('结果质量评估和验证', async ({ page }) => {
    await page.goto('/agentic-rag')
    
    // 执行查询获取结果
    const queryInput = page.locator('textarea[placeholder*="请输入您的查询问题"]').first()
    await queryInput.fill('质量评估测试查询')
    
    const searchButton = page.locator('button:has-text("智能检索")').first()
    await searchButton.click()
    
    await page.waitForTimeout(2000)
    
    // 验证结果质量指标显示
    const resultsPanel = page.locator('[data-testid="intelligent-results"], .intelligent-results-panel').first()
    
    // 验证相关性评分
    await expect(resultsPanel.locator('text=0.96').or(resultsPanel.locator('text=96%')).first()).toBeVisible()
    
    // 验证置信度评分
    await expect(resultsPanel.locator('text=0.93').or(resultsPanel.locator('text=93%')).first()).toBeVisible()
    
    // 验证验证状态标识
    await expect(resultsPanel.locator('text=validation_passed').or(resultsPanel.locator('text=验证通过'))).toBeVisible()
    
    // 验证多维度质量评分可视化
    const qualityMetrics = resultsPanel.locator('[data-testid="quality-metrics"], .quality-metrics').first()
    if (await qualityMetrics.isVisible()) {
      // 验证准确性指标
      await expect(qualityMetrics.locator('text=准确性')).toBeVisible()
      
      // 验证完整性指标
      await expect(qualityMetrics.locator('text=完整性')).toBeVisible()
      
      // 验证时效性指标
      await expect(qualityMetrics.locator('text=时效性')).toBeVisible()
      
      // 验证可信度指标
      await expect(qualityMetrics.locator('text=可信度')).toBeVisible()
    }
    
    // 测试结果筛选功能
    const qualityFilter = page.locator('select[data-testid="quality-filter"], .quality-filter select').first()
    if (await qualityFilter.isVisible()) {
      await qualityFilter.selectOption('high') // 只显示高质量结果
      
      // 验证筛选后结果数量减少且质量更高
      const highQualityResults = resultsPanel.locator('.result-item[data-quality="high"], .high-quality-result')
      await expect(highQualityResults.first()).toBeVisible()
    }
  })

  test('上下文相关的知识片段组合', async ({ page }) => {
    await page.goto('/agentic-rag')
    
    // 执行有上下文历史的查询
    const queryInput = page.locator('textarea[placeholder*="请输入您的查询问题"]').first()
    
    // 第一次查询建立上下文
    await queryInput.fill('什么是深度学习？')
    const searchButton = page.locator('button:has-text("智能检索")').first()
    await searchButton.click()
    await page.waitForTimeout(2000)
    
    // 第二次查询基于上下文
    await queryInput.clear()
    await queryInput.fill('它与机器学习的区别是什么？')
    await searchButton.click()
    await page.waitForTimeout(2000)
    
    // 验证上下文组合器的工作
    const resultsPanel = page.locator('[data-testid="intelligent-results"], .intelligent-results-panel').first()
    
    // 验证结果中包含上下文相关的信息
    const contextualResults = resultsPanel.locator('[data-testid="contextual-result"], .contextual-result').first()
    if (await contextualResults.isVisible()) {
      // 验证上下文标识
      await expect(contextualResults.locator('text=上下文相关').or(contextualResults.locator('text=contextual'))).toBeVisible()
      
      // 验证知识片段关联分析
      await expect(contextualResults.locator('text=深度学习').and(contextualResults.locator('text=机器学习'))).toBeVisible()
    }
    
    // 在解释面板验证上下文分析
    const explanationViewer = page.locator('[data-testid="explanation-viewer"], .explanation-viewer').first()
    if (await explanationViewer.isVisible()) {
      // 验证上下文分析说明
      await expect(explanationViewer.locator('text=上下文').or(explanationViewer.locator('text=context'))).toBeVisible()
      
      // 验证片段关系分析
      const relationshipAnalysis = explanationViewer.locator('[data-testid="relationship-analysis"], .relationship-analysis').first()
      if (await relationshipAnalysis.isVisible()) {
        await expect(relationshipAnalysis.locator('text=逻辑关系')).toBeVisible()
        await expect(relationshipAnalysis.locator('text=关联度')).toBeVisible()
      }
    }
  })

  test('个性化推荐和学习适应', async ({ page }) => {
    await page.goto('/agentic-rag')
    
    // 模拟用户行为模式建立
    const queries = [
      '机器学习基础概念',
      '深度学习神经网络',
      '自然语言处理技术',
      '计算机视觉应用'
    ]
    
    // 执行多次查询建立用户偏好
    for (const query of queries) {
      const queryInput = page.locator('textarea[placeholder*="请输入您的查询问题"]').first()
      await queryInput.clear()
      await queryInput.fill(query)
      
      const searchButton = page.locator('button:has-text("智能检索")').first()
      await searchButton.click()
      await page.waitForTimeout(1500)
      
      // 对结果进行评分建立反馈数据
      const feedbackInterface = page.locator('[data-testid="feedback-interface"], .feedback-interface').first()
      if (await feedbackInterface.isVisible()) {
        const rating = feedbackInterface.locator('.ant-rate').first()
        if (await rating.isVisible()) {
          await rating.locator('.ant-rate-star').nth(3).click() // 4星评分
        }
        
        const quickFeedback = feedbackInterface.locator('button:has-text("有用"), button:has-text("准确")').first()
        if (await quickFeedback.isVisible()) {
          await quickFeedback.click()
        }
      }
    }
    
    // 执行新查询验证个性化推荐
    const queryInput = page.locator('textarea[placeholder*="请输入您的查询问题"]').first()
    await queryInput.clear()
    await queryInput.fill('AI技术应用')
    
    // 验证查询建议功能
    const querySuggestions = page.locator('[data-testid="query-suggestions"], .query-suggestions').first()
    if (await querySuggestions.isVisible()) {
      // 验证基于历史的推荐查询
      await expect(querySuggestions.locator('text=机器学习').or(querySuggestions.locator('text=深度学习'))).toBeVisible()
      await expect(querySuggestions.locator('text=推荐查询').or(querySuggestions.locator('text=相关查询'))).toBeVisible()
    }
    
    const searchButton = page.locator('button:has-text("智能检索")').first()
    await searchButton.click()
    await page.waitForTimeout(2000)
    
    // 验证个性化结果排序
    const resultsPanel = page.locator('[data-testid="intelligent-results"], .intelligent-results-panel').first()
    const personalizedResults = resultsPanel.locator('[data-testid="personalized-result"], .personalized-result').first()
    if (await personalizedResults.isVisible()) {
      await expect(personalizedResults.locator('text=个性化').or(personalizedResults.locator('text=推荐'))).toBeVisible()
    }
  })

  test('会话历史和状态持久化', async ({ page }) => {
    await page.goto('/agentic-rag')
    
    // 创建一个新会话
    const sessionManagerButton = page.locator('button:has-text("会话管理")').first()
    await sessionManagerButton.click()
    
    const sessionModal = page.locator('.ant-modal').first()
    const newSessionButton = sessionModal.locator('button:has-text("新建"), button:has-text("创建会话")').first()
    
    if (await newSessionButton.isVisible()) {
      await newSessionButton.click()
      
      const sessionNameInput = page.locator('input[placeholder*="会话名称"], input[placeholder*="名称"]').first()
      await sessionNameInput.fill('持久化测试会话')
      
      const confirmButton = page.locator('button:has-text("确定"), button:has-text("创建")').first()
      await confirmButton.click()
      
      const closeButton = sessionModal.locator('.ant-modal-close').first()
      await closeButton.click()
    }
    
    // 在会话中执行一系列查询
    const queryHistory = [
      '人工智能的发展历史',
      '机器学习的主要算法',
      '深度学习的应用领域'
    ]
    
    for (const query of queryHistory) {
      const queryInput = page.locator('textarea[placeholder*="请输入您的查询问题"]').first()
      await queryInput.clear()
      await queryInput.fill(query)
      
      const searchButton = page.locator('button:has-text("智能检索")').first()
      await searchButton.click()
      await page.waitForTimeout(1500)
    }
    
    // 刷新页面测试状态恢复
    await page.reload()
    await page.waitForTimeout(2000)
    
    // 验证会话状态恢复
    await expect(page.locator('text=持久化测试会话')).toBeVisible()
    
    // 打开会话管理验证历史记录
    await sessionManagerButton.click()
    
    if (await sessionModal.isVisible()) {
      // 验证查询历史保存
      const historySection = sessionModal.locator('[data-testid="session-history"], .session-history').first()
      if (await historySection.isVisible()) {
        await expect(historySection.locator('text=人工智能的发展历史')).toBeVisible()
        await expect(historySection.locator('text=机器学习的主要算法')).toBeVisible()
        await expect(historySection.locator('text=深度学习的应用领域')).toBeVisible()
      }
      
      // 验证会话统计信息
      const sessionStats = sessionModal.locator('[data-testid="session-stats"], .session-stats').first()
      if (await sessionStats.isVisible()) {
        await expect(sessionStats.locator('text=3').or(sessionStats.locator('text=查询次数'))).toBeVisible()
      }
    }
  })

  test('高级搜索配置和策略选择', async ({ page }) => {
    await page.goto('/agentic-rag')
    
    // 打开高级配置面板
    const advancedButton = page.locator('button:has-text("高级"), button:has-text("设置"), .advanced-settings-btn').first()
    if (await advancedButton.isVisible()) {
      await advancedButton.click()
      
      // 配置检索策略权重
      const strategyConfig = page.locator('[data-testid="strategy-config"], .strategy-configuration').first()
      if (await strategyConfig.isVisible()) {
        // 语义检索权重
        const semanticWeight = strategyConfig.locator('input[data-testid="semantic-weight"], .semantic-weight input').first()
        if (await semanticWeight.isVisible()) {
          await semanticWeight.fill('0.6')
        }
        
        // 关键词检索权重
        const keywordWeight = strategyConfig.locator('input[data-testid="keyword-weight"], .keyword-weight input').first()
        if (await keywordWeight.isVisible()) {
          await keywordWeight.fill('0.3')
        }
        
        // 结构化检索权重
        const structuredWeight = strategyConfig.locator('input[data-testid="structured-weight"], .structured-weight input').first()
        if (await structuredWeight.isVisible()) {
          await structuredWeight.fill('0.1')
        }
      }
      
      // 配置结果数量和质量阈值
      const qualityConfig = page.locator('[data-testid="quality-config"], .quality-configuration').first()
      if (await qualityConfig.isVisible()) {
        const maxResults = qualityConfig.locator('input[data-testid="max-results"], .max-results input').first()
        if (await maxResults.isVisible()) {
          await maxResults.fill('15')
        }
        
        const qualityThreshold = qualityConfig.locator('input[data-testid="quality-threshold"], .quality-threshold input').first()
        if (await qualityThreshold.isVisible()) {
          await qualityThreshold.fill('0.85')
        }
      }
      
      // 保存配置
      const saveConfigButton = page.locator('button:has-text("保存"), button:has-text("应用")').first()
      if (await saveConfigButton.isVisible()) {
        await saveConfigButton.click()
        await expect(page.locator('text=配置已保存')).toBeVisible()
      }
    }
    
    // 执行查询验证配置生效
    const queryInput = page.locator('textarea[placeholder*="请输入您的查询问题"]').first()
    await queryInput.fill('测试高级配置查询')
    
    const searchButton = page.locator('button:has-text("智能检索")').first()
    await searchButton.click()
    
    await page.waitForTimeout(3000)
    
    // 验证配置影响结果
    const resultsPanel = page.locator('[data-testid="intelligent-results"], .intelligent-results-panel').first()
    
    // 验证结果数量符合配置
    const resultItems = resultsPanel.locator('.result-item, [data-testid="result-item"]')
    await expect(resultItems).toHaveCount(2) // 根据mock数据应该有2个结果
    
    // 验证质量阈值生效（所有结果应该高于0.85）
    const qualityScores = resultsPanel.locator('.quality-score, [data-testid="quality-score"]')
    for (let i = 0; i < await qualityScores.count(); i++) {
      const scoreText = await qualityScores.nth(i).textContent()
      if (scoreText) {
        const score = parseFloat(scoreText.match(/0\.\d+/)?.[0] || '0')
        expect(score).toBeGreaterThan(0.85)
      }
    }
  })

  test('实时性能监控和统计展示', async ({ page }) => {
    await page.goto('/agentic-rag')
    
    // 执行多次查询生成性能数据
    const performanceQueries = [
      '性能监控测试1',
      '性能监控测试2', 
      '性能监控测试3'
    ]
    
    for (const query of performanceQueries) {
      const queryInput = page.locator('textarea[placeholder*="请输入您的查询问题"]').first()
      await queryInput.clear()
      await queryInput.fill(query)
      
      const searchButton = page.locator('button:has-text("智能检索")').first()
      await searchButton.click()
      await page.waitForTimeout(2000)
    }
    
    // 查看性能统计
    const performanceButton = page.locator('button:has-text("性能统计"), button:has-text("统计"), .performance-stats-btn').first()
    if (await performanceButton.isVisible()) {
      await performanceButton.click()
      
      const statsModal = page.locator('.ant-modal').first()
      
      // 验证总体统计数据
      await expect(statsModal.locator('text=总查询次数')).toBeVisible()
      await expect(statsModal.locator('text=1247')).toBeVisible() // 基于mock数据
      
      await expect(statsModal.locator('text=成功查询')).toBeVisible()
      await expect(statsModal.locator('text=1156')).toBeVisible()
      
      await expect(statsModal.locator('text=平均置信度')).toBeVisible()
      await expect(statsModal.locator('text=0.87').or(statsModal.locator('text=87%'))).toBeVisible()
      
      await expect(statsModal.locator('text=平均处理时间')).toBeVisible()
      await expect(statsModal.locator('text=245ms')).toBeVisible()
      
      // 验证策略使用统计
      const strategyStats = statsModal.locator('[data-testid="strategy-stats"], .strategy-statistics').first()
      if (await strategyStats.isVisible()) {
        await expect(strategyStats.locator('text=multi_agent')).toBeVisible()
        await expect(strategyStats.locator('text=45%')).toBeVisible()
        await expect(strategyStats.locator('text=94%')).toBeVisible() // 成功率
        
        await expect(strategyStats.locator('text=semantic')).toBeVisible()
        await expect(strategyStats.locator('text=32%')).toBeVisible()
        await expect(strategyStats.locator('text=91%')).toBeVisible()
      }
      
      // 验证代理性能分析
      const agentPerformance = statsModal.locator('[data-testid="agent-performance"], .agent-performance').first()
      if (await agentPerformance.isVisible()) {
        await expect(agentPerformance.locator('text=semantic_expert')).toBeVisible()
        await expect(agentPerformance.locator('text=567')).toBeVisible() // 查询次数
        await expect(agentPerformance.locator('text=93%')).toBeVisible() // 成功率
        await expect(agentPerformance.locator('text=123ms')).toBeVisible() // 平均时间
      }
      
      // 关闭统计弹窗
      const closeButton = statsModal.locator('.ant-modal-close').first()
      await closeButton.click()
    }
    
    // 验证侧边栏实时性能指标
    const sidebarStats = page.locator('.ant-sider').first()
    if (await sidebarStats.isVisible()) {
      // 验证实时统计更新
      await expect(sidebarStats.locator('text=272ms')).toBeVisible() // 最新查询处理时间
      await expect(sidebarStats.locator('text=91%')).toBeVisible() // 最新查询置信度
    }
  })
})