/**
 * 可解释AI系统E2E测试
 */

import { test, expect } from '@playwright/test'

test.describe('可解释AI系统E2E测试', () => {
  test.beforeEach(async ({ page }) => {
    // 设置API mock拦截
    await page.route('**/api/v1/explainable-ai/**', route => {
      const url = route.request().url()

      if (url.includes('/health')) {
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            status: 'healthy',
            service: 'explainable-ai',
            timestamp: new Date().toISOString(),
            version: '1.0.0',
            components: {
              explanation_generator: 'active',
              cot_reasoner: 'active',
              workflow_explainer: 'active',
              formatter: 'active',
            },
          }),
        })
      } else if (url.includes('/explanation-types')) {
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            explanation_types: [
              {
                value: 'decision',
                label: '决策解释',
                description: '解释决策过程和结果',
              },
              {
                value: 'reasoning',
                label: '推理解释',
                description: '详细的逐步推理过程',
              },
              {
                value: 'workflow',
                label: '工作流解释',
                description: '工作流执行过程解释',
              },
            ],
            explanation_levels: [
              {
                value: 'summary',
                label: '概要',
                description: '简洁的解释摘要',
              },
              {
                value: 'detailed',
                label: '详细',
                description: '详细的解释内容',
              },
              {
                value: 'technical',
                label: '技术',
                description: '技术性深度解释',
              },
            ],
            reasoning_modes: [
              {
                value: 'analytical',
                label: '分析性推理',
                description: '逐步分解分析',
              },
              {
                value: 'deductive',
                label: '演绎推理',
                description: '从一般到具体',
              },
              {
                value: 'inductive',
                label: '归纳推理',
                description: '从具体到一般',
              },
              {
                value: 'abductive',
                label: '溯因推理',
                description: '寻找最佳解释',
              },
            ],
            output_formats: [
              { value: 'html', label: 'HTML', description: '网页格式' },
              {
                value: 'markdown',
                label: 'Markdown',
                description: 'Markdown文档',
              },
              { value: 'json', label: 'JSON', description: '结构化数据' },
              { value: 'text', label: '纯文本', description: '纯文本格式' },
              { value: 'xml', label: 'XML', description: 'XML文档' },
            ],
          }),
        })
      } else if (url.includes('/demo-scenarios')) {
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            scenarios: [
              {
                type: 'loan_approval',
                name: '贷款审批',
                description: '银行贷款审批决策场景',
                complexity_levels: ['simple', 'medium', 'complex'],
              },
              {
                type: 'medical_diagnosis',
                name: '医疗诊断',
                description: '医疗诊断辅助决策场景',
                complexity_levels: ['simple', 'medium', 'complex'],
              },
              {
                type: 'investment_recommendation',
                name: '投资建议',
                description: '投资决策建议场景',
                complexity_levels: ['simple', 'medium', 'complex'],
              },
            ],
          }),
        })
      } else if (url.includes('/generate-explanation')) {
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            id: 'explanation-001',
            decision_id: 'test_decision_001',
            explanation_type: 'decision',
            explanation_level: 'detailed',
            decision_description: '贷款审批决策',
            decision_outcome: 'approved',
            summary_explanation:
              '基于申请人的信用评分750分、年收入8万元等因素，系统建议批准此贷款申请。',
            detailed_explanation:
              '经过综合分析，申请人具备良好的信用记录和稳定的收入来源，风险评估结果为低风险，符合贷款批准标准。',
            components: [
              {
                id: 'comp-001',
                factor_name: 'credit_score',
                factor_value: 750,
                weight: 0.8,
                impact_score: 0.7,
                evidence_type: 'external_source',
                evidence_source: 'credit_bureau',
                evidence_content: '信用评分750分，表现良好',
                causal_relationship: '信用评分直接影响贷款批准概率',
              },
              {
                id: 'comp-002',
                factor_name: 'annual_income',
                factor_value: 80000,
                weight: 0.7,
                impact_score: 0.6,
                evidence_type: 'external_source',
                evidence_source: 'financial_statement',
                evidence_content: '年收入8万元，收入稳定',
                causal_relationship: '年收入水平影响还款能力评估',
              },
            ],
            confidence_metrics: {
              overall_confidence: 0.85,
              uncertainty_score: 0.15,
              confidence_sources: ['model_probability'],
            },
            counterfactuals: [
              {
                id: 'cf-001',
                scenario_name: '信用评分降低场景',
                changed_factors: { credit_score: 650 },
                predicted_outcome: '可能被拒绝',
                probability: 0.7,
                impact_difference: -0.3,
                explanation: '如果信用评分降至650分，批准概率将显著降低',
              },
            ],
            visualization_data: {
              factor_importance: {
                chart_type: 'bar',
                data: [
                  {
                    name: 'credit_score',
                    value: 0.56,
                    weight: 0.8,
                    impact: 0.7,
                  },
                  {
                    name: 'annual_income',
                    value: 0.42,
                    weight: 0.7,
                    impact: 0.6,
                  },
                ],
              },
              confidence_breakdown: {
                chart_type: 'pie',
                data: [
                  { label: '模型置信度', value: 0.5, color: '#4CAF50' },
                  { label: '证据强度', value: 0.35, color: '#2196F3' },
                  { label: '不确定性', value: 0.15, color: '#FF9800' },
                ],
              },
            },
            metadata: {
              generation_style: 'user_friendly',
              generation_time_ms: 1250,
              model_version: 'gpt-4o-mini',
            },
          }),
        })
      } else if (url.includes('/demo-scenario')) {
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            id: 'demo-explanation-001',
            decision_id: 'loan_demo_001',
            explanation_type: 'decision',
            explanation_level: 'detailed',
            decision_description: '贷款审批决策',
            decision_outcome: 'approved',
            summary_explanation:
              '基于综合评估，建议批准此贷款申请。主要依据包括良好的信用记录、稳定的收入来源和合理的债务比例。',
            detailed_explanation:
              '申请人信用评分为750分，年收入8万元，就业期间5年，债务收入比为25%。根据风险评估模型，该申请属于低风险类别，符合银行贷款审批标准。',
            components: [
              {
                id: 'demo-comp-001',
                factor_name: 'credit_score',
                factor_value: 750,
                weight: 0.35,
                impact_score: 0.8,
                evidence_type: 'external_source',
                evidence_source: 'credit_bureau',
                evidence_content: '信用评分750分，信用历史良好',
                causal_relationship: '信用评分是贷款批准的关键因素',
              },
              {
                id: 'demo-comp-002',
                factor_name: 'annual_income',
                factor_value: 80000,
                weight: 0.25,
                impact_score: 0.7,
                evidence_type: 'external_source',
                evidence_source: 'payroll',
                evidence_content: '年收入8万元，收入来源稳定',
                causal_relationship: '收入水平决定还款能力',
              },
              {
                id: 'demo-comp-003',
                factor_name: 'employment_duration',
                factor_value: 5,
                weight: 0.2,
                impact_score: 0.6,
                evidence_type: 'external_source',
                evidence_source: 'hr_system',
                evidence_content: '就业5年，工作稳定性良好',
                causal_relationship: '就业稳定性影响收入持续性',
              },
              {
                id: 'demo-comp-004',
                factor_name: 'debt_ratio',
                factor_value: 0.25,
                weight: 0.2,
                impact_score: 0.5,
                evidence_type: 'external_source',
                evidence_source: 'financial_calc',
                evidence_content: '债务收入比25%，负债水平合理',
                causal_relationship: '债务比例影响整体财务风险',
              },
            ],
            confidence_metrics: {
              overall_confidence: 0.82,
              uncertainty_score: 0.18,
              confidence_sources: ['model_probability', 'evidence_strength'],
            },
            counterfactuals: [
              {
                id: 'demo-cf-001',
                scenario_name: '信用评分变化场景',
                changed_factors: { credit_score: 600 },
                predicted_outcome: '可能会显著改变决策结果',
                probability: 0.8,
                impact_difference: -0.168,
                explanation: '如果信用评分降至600分，预计决策置信度会下降约17%',
              },
            ],
            visualization_data: {
              factor_importance: {
                chart_type: 'bar',
                data: [
                  {
                    name: 'credit_score',
                    value: 0.28,
                    weight: 0.35,
                    impact: 0.8,
                  },
                  {
                    name: 'annual_income',
                    value: 0.175,
                    weight: 0.25,
                    impact: 0.7,
                  },
                  {
                    name: 'employment_duration',
                    value: 0.12,
                    weight: 0.2,
                    impact: 0.6,
                  },
                  { name: 'debt_ratio', value: 0.1, weight: 0.2, impact: 0.5 },
                ],
              },
              confidence_breakdown: {
                chart_type: 'pie',
                data: [
                  { label: '模型置信度', value: 0.5, color: '#4CAF50' },
                  { label: '证据强度', value: 0.32, color: '#2196F3' },
                  { label: '不确定性', value: 0.18, color: '#FF9800' },
                ],
              },
            },
            metadata: {
              generation_style: 'user_friendly',
              generation_time_ms: 1800,
              model_version: 'gpt-4o-mini',
              demo_scenario: true,
              scenario_type: 'loan_approval',
              complexity: 'medium',
            },
          }),
        })
      } else {
        // 默认处理其他请求
        route.continue()
      }
    })

    await page.goto('/explainable-ai')
  })

  test('可解释AI页面基本功能测试', async ({ page }) => {
    // 验证页面标题
    await expect(page.locator('h1')).toContainText('可解释AI决策系统')

    // 验证主要功能区域存在
    await expect(
      page.locator('[data-testid="explanation-controls"]')
    ).toBeVisible()
    await expect(
      page.locator('[data-testid="explanation-result"]')
    ).toBeVisible()

    // 验证解释类型选择器
    const explanationTypeSelect = page.locator(
      '[data-testid="explanation-type-select"]'
    )
    await expect(explanationTypeSelect).toBeVisible()

    // 验证解释级别选择器
    const explanationLevelSelect = page.locator(
      '[data-testid="explanation-level-select"]'
    )
    await expect(explanationLevelSelect).toBeVisible()

    // 验证生成解释按钮
    const generateButton = page.locator(
      '[data-testid="generate-explanation-btn"]'
    )
    await expect(generateButton).toBeVisible()
    await expect(generateButton).toContainText('生成解释')
  })

  test('演示场景功能测试', async ({ page }) => {
    // 点击演示场景标签
    await page.click('[data-testid="demo-scenarios-tab"]')

    // 验证演示场景选择器
    const scenarioSelect = page.locator('[data-testid="scenario-select"]')
    await expect(scenarioSelect).toBeVisible()

    // 选择贷款审批场景
    await scenarioSelect.selectOption('loan_approval')

    // 选择复杂度
    const complexitySelect = page.locator('[data-testid="complexity-select"]')
    await complexitySelect.selectOption('medium')

    // 启用CoT推理
    const cotCheckbox = page.locator('[data-testid="cot-reasoning-checkbox"]')
    await cotCheckbox.check()

    // 生成演示解释
    const generateDemoButton = page.locator('[data-testid="generate-demo-btn"]')
    await generateDemoButton.click()

    // 等待解释结果显示
    await expect(
      page.locator('[data-testid="explanation-result"]')
    ).toBeVisible()

    // 验证基本信息显示
    await expect(page.locator('[data-testid="decision-id"]')).toContainText(
      'loan_demo_001'
    )
    await expect(
      page.locator('[data-testid="decision-outcome"]')
    ).toContainText('approved')

    // 验证概要解释
    const summaryText = page.locator('[data-testid="summary-explanation"]')
    await expect(summaryText).toBeVisible()
    await expect(summaryText).toContainText('基于综合评估，建议批准此贷款申请')

    // 验证置信度显示
    const confidenceMetrics = page.locator('[data-testid="confidence-metrics"]')
    await expect(confidenceMetrics).toBeVisible()
    await expect(confidenceMetrics).toContainText('82%')
  })

  test('解释组件功能测试', async ({ page }) => {
    // 生成演示解释
    await page.click('[data-testid="demo-scenarios-tab"]')
    await page.selectOption('[data-testid="scenario-select"]', 'loan_approval')
    await page.click('[data-testid="generate-demo-btn"]')

    // 等待解释结果
    await page.waitForSelector('[data-testid="explanation-result"]')

    // 验证解释组件标签
    const componentsTab = page.locator('[data-testid="components-tab"]')
    await componentsTab.click()

    // 验证关键因素表格
    const factorsTable = page.locator('[data-testid="factors-table"]')
    await expect(factorsTable).toBeVisible()

    // 验证表格内容
    const tableRows = factorsTable.locator('tbody tr')
    await expect(tableRows).toHaveCount(4) // credit_score, annual_income, employment_duration, debt_ratio

    // 验证第一行数据（信用评分）
    const firstRow = tableRows.first()
    await expect(firstRow.locator('td').nth(0)).toContainText('credit_score')
    await expect(firstRow.locator('td').nth(1)).toContainText('750')
    await expect(firstRow.locator('td').nth(2)).toContainText('35%') // 权重
    await expect(firstRow.locator('td').nth(3)).toContainText('80%') // 影响分数
  })

  test('反事实分析功能测试', async ({ page }) => {
    // 生成演示解释
    await page.click('[data-testid="demo-scenarios-tab"]')
    await page.selectOption('[data-testid="scenario-select"]', 'loan_approval')
    await page.click('[data-testid="generate-demo-btn"]')

    // 等待解释结果
    await page.waitForSelector('[data-testid="explanation-result"]')

    // 切换到反事实分析标签
    const counterfactualsTab = page.locator(
      '[data-testid="counterfactuals-tab"]'
    )
    await counterfactualsTab.click()

    // 验证反事实场景显示
    const counterfactualCard = page
      .locator('[data-testid="counterfactual-card"]')
      .first()
    await expect(counterfactualCard).toBeVisible()

    // 验证场景内容
    await expect(counterfactualCard).toContainText('信用评分变化场景')
    await expect(counterfactualCard).toContainText('600') // 变化后的值
    await expect(counterfactualCard).toContainText('可能会显著改变决策结果')
    await expect(counterfactualCard).toContainText('80%') // 概率
  })

  test('可视化图表功能测试', async ({ page }) => {
    // 生成演示解释
    await page.click('[data-testid="demo-scenarios-tab"]')
    await page.selectOption('[data-testid="scenario-select"]', 'loan_approval')
    await page.click('[data-testid="generate-demo-btn"]')

    // 等待解释结果
    await page.waitForSelector('[data-testid="explanation-result"]')

    // 切换到可视化标签
    const visualizationTab = page.locator('[data-testid="visualization-tab"]')
    await visualizationTab.click()

    // 验证因子重要性图表
    const factorChart = page.locator('[data-testid="factor-importance-chart"]')
    await expect(factorChart).toBeVisible()

    // 验证置信度分解图表
    const confidenceChart = page.locator(
      '[data-testid="confidence-breakdown-chart"]'
    )
    await expect(confidenceChart).toBeVisible()

    // 验证图表控制按钮
    const chartToggleButtons = page.locator(
      '[data-testid="chart-toggle-buttons"] button'
    )
    await expect(chartToggleButtons).toHaveCount(2) // 柱状图和饼图切换
  })

  test('自定义解释生成测试', async ({ page }) => {
    // 切换到自定义解释标签
    const customTab = page.locator('[data-testid="custom-explanation-tab"]')
    await customTab.click()

    // 填写决策ID
    const decisionIdInput = page.locator('[data-testid="decision-id-input"]')
    await decisionIdInput.fill('custom_test_001')

    // 填写决策上下文
    const contextInput = page.locator('[data-testid="decision-context-input"]')
    await contextInput.fill('自定义决策测试')

    // 选择解释类型
    const typeSelect = page.locator('[data-testid="explanation-type-select"]')
    await typeSelect.selectOption('decision')

    // 选择解释级别
    const levelSelect = page.locator('[data-testid="explanation-level-select"]')
    await levelSelect.selectOption('detailed')

    // 添加置信度因子
    const addFactorButton = page.locator('[data-testid="add-factor-btn"]')
    await addFactorButton.click()

    // 填写因子信息
    const factorNameInput = page
      .locator('[data-testid="factor-name-input"]')
      .first()
    await factorNameInput.fill('test_factor')

    const factorValueInput = page
      .locator('[data-testid="factor-value-input"]')
      .first()
    await factorValueInput.fill('test_value')

    const factorWeightInput = page
      .locator('[data-testid="factor-weight-input"]')
      .first()
    await factorWeightInput.fill('0.8')

    const factorImpactInput = page
      .locator('[data-testid="factor-impact-input"]')
      .first()
    await factorImpactInput.fill('0.7')

    // 生成自定义解释
    const generateButton = page.locator('[data-testid="generate-custom-btn"]')
    await generateButton.click()

    // 验证解释结果
    await expect(
      page.locator('[data-testid="explanation-result"]')
    ).toBeVisible()
    await expect(page.locator('[data-testid="decision-id"]')).toContainText(
      'test_decision_001'
    )
  })

  test('解释导出功能测试', async ({ page }) => {
    // 生成演示解释
    await page.click('[data-testid="demo-scenarios-tab"]')
    await page.selectOption('[data-testid="scenario-select"]', 'loan_approval')
    await page.click('[data-testid="generate-demo-btn"]')

    // 等待解释结果
    await page.waitForSelector('[data-testid="explanation-result"]')

    // 验证导出按钮组
    const exportButtons = page.locator('[data-testid="export-buttons"]')
    await expect(exportButtons).toBeVisible()

    // 验证不同格式的导出按钮
    const exportFormats = ['HTML', 'Markdown', 'JSON', 'XML', '纯文本']

    for (const format of exportFormats) {
      const exportButton = page.locator(
        `[data-testid="export-${format.toLowerCase()}-btn"]`
      )
      await expect(exportButton).toBeVisible()
      await expect(exportButton).toContainText(format)
    }

    // 测试HTML导出（模拟点击）
    const htmlExportButton = page.locator('[data-testid="export-html-btn"]')

    // 监听下载事件
    const downloadPromise = page.waitForEvent('download')
    await htmlExportButton.click()

    // 由于是mock环境，下载可能不会真正发生，所以这里主要测试UI响应
    await expect(page.locator('[data-testid="export-status"]')).toContainText(
      '正在导出',
      { timeout: 3000 }
    )
  })

  test('错误处理测试', async ({ page }) => {
    // 模拟API错误
    await page.route('**/api/v1/explainable-ai/generate-explanation', route => {
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({
          detail: '解释生成失败: 内部服务器错误',
        }),
      })
    })

    // 尝试生成自定义解释
    await page.click('[data-testid="custom-explanation-tab"]')
    await page.fill('[data-testid="decision-id-input"]', 'error_test')
    await page.fill('[data-testid="decision-context-input"]', '错误测试')
    await page.click('[data-testid="generate-custom-btn"]')

    // 验证错误消息显示
    const errorMessage = page.locator('[data-testid="error-message"]')
    await expect(errorMessage).toBeVisible()
    await expect(errorMessage).toContainText('解释生成失败')
  })

  test('响应式设计测试', async ({ page }) => {
    // 测试移动端视口
    await page.setViewportSize({ width: 375, height: 667 })

    // 验证页面在小屏幕下的布局
    await expect(page.locator('h1')).toBeVisible()
    await expect(
      page.locator('[data-testid="explanation-controls"]')
    ).toBeVisible()

    // 验证标签页在移动端的响应
    const tabContainer = page.locator('[data-testid="tab-container"]')
    await expect(tabContainer).toBeVisible()

    // 测试平板视口
    await page.setViewportSize({ width: 768, height: 1024 })

    // 验证在平板尺寸下的布局
    await expect(
      page.locator('[data-testid="explanation-controls"]')
    ).toBeVisible()
    await expect(
      page.locator('[data-testid="explanation-result"]')
    ).toBeVisible()

    // 恢复桌面视口
    await page.setViewportSize({ width: 1280, height: 720 })
  })

  test('键盘导航测试', async ({ page }) => {
    // 测试Tab键导航
    await page.keyboard.press('Tab')

    // 验证焦点在第一个可聚焦元素上
    const focusedElement = page.locator(':focus')
    await expect(focusedElement).toBeVisible()

    // 继续Tab导航几次
    for (let i = 0; i < 5; i++) {
      await page.keyboard.press('Tab')
      await page.waitForTimeout(100)
    }

    // 测试Enter键激活按钮
    const generateButton = page.locator('[data-testid="generate-demo-btn"]')
    await generateButton.focus()
    await page.keyboard.press('Enter')

    // 验证操作被触发（这里可能需要根据实际实现调整）
    await page.waitForTimeout(500)
  })

  test('加载状态测试', async ({ page }) => {
    // 模拟慢速API响应
    await page.route('**/api/v1/explainable-ai/demo-scenario', route => {
      setTimeout(() => {
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            id: 'loading-test',
            decision_id: 'loading_test_001',
            explanation_type: 'decision',
            explanation_level: 'detailed',
            decision_description: '加载测试',
            decision_outcome: 'completed',
            summary_explanation: '加载测试完成',
            components: [],
            confidence_metrics: {
              overall_confidence: 0.8,
              uncertainty_score: 0.2,
              confidence_sources: [],
            },
            counterfactuals: [],
            metadata: {},
          }),
        })
      }, 2000) // 2秒延迟
    })

    // 触发加载状态
    await page.click('[data-testid="demo-scenarios-tab"]')
    await page.selectOption('[data-testid="scenario-select"]', 'loan_approval')
    await page.click('[data-testid="generate-demo-btn"]')

    // 验证加载指示器显示
    const loadingIndicator = page.locator('[data-testid="loading-indicator"]')
    await expect(loadingIndicator).toBeVisible()

    // 验证按钮在加载时被禁用
    const generateButton = page.locator('[data-testid="generate-demo-btn"]')
    await expect(generateButton).toBeDisabled()

    // 等待加载完成
    await expect(loadingIndicator).toBeHidden({ timeout: 5000 })
    await expect(generateButton).toBeEnabled()
  })

  test('数据持久化测试', async ({ page }) => {
    // 生成解释
    await page.click('[data-testid="demo-scenarios-tab"]')
    await page.selectOption('[data-testid="scenario-select"]', 'loan_approval')
    await page.click('[data-testid="generate-demo-btn"]')

    // 等待结果
    await page.waitForSelector('[data-testid="explanation-result"]')

    // 记录当前解释ID
    const decisionId = await page
      .locator('[data-testid="decision-id"]')
      .textContent()

    // 刷新页面
    await page.reload()

    // 验证解释历史或缓存是否保持（根据实际实现）
    // 这个测试可能需要根据具体的持久化策略进行调整
    await page.waitForSelector('[data-testid="explanation-controls"]')
  })
})

test.describe('可解释AI系统性能测试', () => {
  test('页面加载性能测试', async ({ page }) => {
    const startTime = Date.now()

    await page.goto('/explainable-ai')

    // 等待主要内容加载完成
    await page.waitForSelector('[data-testid="explanation-controls"]')
    await page.waitForSelector('[data-testid="explanation-result"]')

    const loadTime = Date.now() - startTime

    // 验证页面在3秒内加载完成
    expect(loadTime).toBeLessThan(3000)
  })

  test('大量数据渲染性能测试', async ({ page }) => {
    // 模拟返回大量组件的API响应
    await page.route('**/api/v1/explainable-ai/demo-scenario', route => {
      const largeComponentsList = Array.from({ length: 50 }, (_, i) => ({
        id: `comp-${i}`,
        factor_name: `factor_${i}`,
        factor_value: Math.random() * 1000,
        weight: Math.random(),
        impact_score: Math.random(),
        evidence_type: 'external_source',
        evidence_source: `source_${i}`,
        evidence_content: `Factor ${i} description`,
        causal_relationship: `Factor ${i} affects decision`,
      }))

      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          id: 'performance-test',
          decision_id: 'perf_test_001',
          explanation_type: 'decision',
          explanation_level: 'detailed',
          decision_description: '性能测试',
          decision_outcome: 'completed',
          summary_explanation: '性能测试解释',
          components: largeComponentsList,
          confidence_metrics: {
            overall_confidence: 0.8,
            uncertainty_score: 0.2,
            confidence_sources: [],
          },
          counterfactuals: [],
          metadata: {},
        }),
      })
    })

    const startTime = Date.now()

    // 生成包含大量组件的解释
    await page.goto('/explainable-ai')
    await page.click('[data-testid="demo-scenarios-tab"]')
    await page.selectOption('[data-testid="scenario-select"]', 'loan_approval')
    await page.click('[data-testid="generate-demo-btn"]')

    // 等待渲染完成
    await page.waitForSelector('[data-testid="explanation-result"]')
    await page.click('[data-testid="components-tab"]')
    await page.waitForSelector('[data-testid="factors-table"]')

    const renderTime = Date.now() - startTime

    // 验证即使有大量数据，渲染时间也在合理范围内
    expect(renderTime).toBeLessThan(5000)

    // 验证所有组件都被正确渲染
    const tableRows = page.locator('[data-testid="factors-table"] tbody tr')
    await expect(tableRows).toHaveCount(50)
  })
})
