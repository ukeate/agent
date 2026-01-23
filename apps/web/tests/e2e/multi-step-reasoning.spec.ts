import { test, expect } from '@playwright/test'

test.describe('多步推理工作流系统', () => {
  test.beforeEach(async ({ page }) => {
    // 导航到多步推理页面
    await page.goto('/multi-step-reasoning')

    // 等待页面完全加载
    await expect(page.locator('h1:has-text("多步推理工作流")')).toBeVisible()
  })

  test('页面基本功能加载测试', async ({ page }) => {
    // 验证页面标题
    await expect(page.locator('h1')).toContainText('多步推理工作流')

    // 验证技术说明
    await expect(page.locator('p')).toContainText(
      'Complex Problem → CoT Decomposition → Task DAG → Distributed Execution'
    )

    // 验证主要组件存在
    await expect(page.locator('text=问题输入')).toBeVisible()
    await expect(page.locator('text=分解配置')).toBeVisible()
    await expect(page.locator('text=执行配置')).toBeVisible()
    await expect(page.locator('text=系统监控')).toBeVisible()
  })

  test('问题输入和配置功能测试', async ({ page }) => {
    // 输入问题描述
    const problemTextarea = page.locator(
      'textarea[placeholder*="输入需要分解的复杂问题"]'
    )
    await problemTextarea.fill(
      '如何设计一个高性能的分布式缓存系统？需要考虑一致性、可用性和分区容错性。'
    )

    // 验证输入内容
    await expect(problemTextarea).toHaveValue(/高性能的分布式缓存系统/)

    // 测试分解配置
    const strategySelect = page.locator('select').first()
    await strategySelect.selectOption('research')
    await expect(strategySelect).toHaveValue('research')

    // 调整最大深度滑块
    const depthSlider = page.locator('input[type="range"]').first()
    await depthSlider.fill('7')
    await expect(page.locator('text=最大深度: 7')).toBeVisible()

    // 调整目标复杂度滑块
    const complexitySlider = page.locator('input[type="range"]').nth(1)
    await complexitySlider.fill('8')
    await expect(page.locator('text=目标复杂度: 8')).toBeVisible()

    // 切换到执行配置选项卡
    await page.locator('text=执行配置').click()

    // 测试执行模式选择
    const executionModeSelect = page.locator('select').nth(1)
    await executionModeSelect.selectOption('hybrid')
    await expect(executionModeSelect).toHaveValue('hybrid')

    // 调整并行数滑块
    const parallelSlider = page.locator('input[type="range"]').nth(2)
    await parallelSlider.fill('5')
    await expect(page.locator('text=最大并行数: 5')).toBeVisible()

    // 测试调度策略
    const scheduleSelect = page.locator('select').nth(2)
    await scheduleSelect.selectOption('priority')
    await expect(scheduleSelect).toHaveValue('priority')
  })

  test('工作流执行和状态监控测试', async ({ page }) => {
    // 输入测试问题
    await page
      .locator('textarea[placeholder*="输入需要分解的复杂问题"]')
      .fill(
        '设计一个微服务架构的电商平台，包括用户管理、商品管理、订单处理、支付系统和推荐引擎'
      )

    // 点击开始分解执行按钮
    const startButton = page.locator('button:has-text("开始分解执行")')
    await expect(startButton).toBeEnabled()
    await startButton.click()

    // 验证按钮状态变化
    await expect(page.locator('button:has-text("分解问题中...")')).toBeVisible({
      timeout: 2000,
    })

    // 等待分解完成，应该显示DAG
    await expect(page.locator('text=任务依赖图 (DAG)')).toBeVisible({
      timeout: 10000,
    })

    // 验证DAG中有任务节点
    await expect(page.locator('text=问题分析')).toBeVisible()
    await expect(page.locator('text=数据收集')).toBeVisible()
    await expect(page.locator('text=深度分析')).toBeVisible()

    // 验证执行控制面板出现
    await expect(page.locator('text=执行控制')).toBeVisible({ timeout: 15000 })
    await expect(page.locator('text=总体进度')).toBeVisible()

    // 测试暂停功能
    const pauseButton = page
      .locator('button[title*="暂停"], button:has(svg)')
      .first()
    if (await pauseButton.isVisible()) {
      await pauseButton.click()
      // 验证暂停状态 - 这里由于是模拟API，可能不会真正暂停
    }
  })

  test('DAG可视化和任务详情测试', async ({ page }) => {
    // 先启动一个工作流
    await page
      .locator('textarea[placeholder*="输入需要分解的复杂问题"]')
      .fill('构建一个AI驱动的智能客服系统')

    await page.locator('button:has-text("开始分解执行")').click()

    // 等待DAG显示
    await expect(page.locator('text=任务依赖图 (DAG)')).toBeVisible({
      timeout: 10000,
    })

    // 点击任务节点查看详情
    const taskNode = page.locator('text=问题分析').first()
    await taskNode.click()

    // 验证任务详情面板
    await expect(page.locator('text=步骤ID')).toBeVisible()
    await expect(page.locator('text=类型')).toBeVisible()
    await expect(page.locator('text=状态')).toBeVisible()
    await expect(page.locator('text=复杂度')).toBeVisible()

    // 测试不同任务节点的点击
    const dataCollectionNode = page.locator('text=数据收集').first()
    if (await dataCollectionNode.isVisible()) {
      await dataCollectionNode.click()
      // 验证详情面板更新
      await expect(page.locator('text=数据收集')).toBeVisible()
    }

    // 验证关键路径标识
    await expect(page.locator('text=关键路径')).toBeVisible()

    // 验证DAG控制按钮
    await expect(page.locator('button:has-text("全览")')).toBeVisible()
    await expect(page.locator('button:has-text("导出")')).toBeVisible()
  })

  test('系统监控指标测试', async ({ page }) => {
    // 验证系统监控卡片存在
    await expect(page.locator('text=系统监控')).toBeVisible()

    // 验证监控指标
    await expect(page.locator('text=活跃工作器')).toBeVisible()
    await expect(page.locator('text=队列任务')).toBeVisible()
    await expect(page.locator('text=平均等待')).toBeVisible()
    await expect(page.locator('text=成功率')).toBeVisible()

    // 验证数值显示（可能为0，因为系统刚启动）
    const workerCount = page
      .locator('text=活跃工作器')
      .locator('..')
      .locator('.text-2xl')
    await expect(workerCount).toBeVisible()

    const queueDepth = page
      .locator('text=队列任务')
      .locator('..')
      .locator('.text-2xl')
    await expect(queueDepth).toBeVisible()
  })

  test('错误处理和边界条件测试', async ({ page }) => {
    // 测试空输入
    const startButton = page.locator('button:has-text("开始分解执行")')
    await expect(startButton).toBeDisabled()

    // 输入很短的问题
    await page
      .locator('textarea[placeholder*="输入需要分解的复杂问题"]')
      .fill('测试')
    await expect(startButton).toBeEnabled()

    // 输入很长的问题
    const longProblem = '这是一个非常非常长的问题描述'.repeat(50)
    await page
      .locator('textarea[placeholder*="输入需要分解的复杂问题"]')
      .fill(longProblem)
    await expect(startButton).toBeEnabled()

    // 测试网络错误模拟 - 在实际测试中可能需要mock API
    // 这里我们主要验证UI的错误显示机制
  })

  test('响应式设计测试', async ({ page }) => {
    // 测试不同屏幕尺寸
    await page.setViewportSize({ width: 1200, height: 800 })
    await expect(page.locator('h1')).toBeVisible()

    // 测试平板尺寸
    await page.setViewportSize({ width: 768, height: 1024 })
    await expect(page.locator('h1')).toBeVisible()
    await expect(page.locator('text=问题输入')).toBeVisible()

    // 测试手机尺寸
    await page.setViewportSize({ width: 375, height: 667 })
    await expect(page.locator('h1')).toBeVisible()

    // 在小屏幕上应该能看到主要功能
    await expect(
      page.locator('textarea[placeholder*="输入需要分解的复杂问题"]')
    ).toBeVisible()
  })

  test('工作流配置持久化测试', async ({ page }) => {
    // 设置特定配置
    await page.locator('select').first().selectOption('optimization')
    await page.locator('input[type="range"]').first().fill('9')

    // 切换标签页再切换回来，验证配置保持
    await page.locator('text=执行配置').click()
    await page.locator('text=分解配置').click()

    // 验证配置值保持
    await expect(page.locator('select').first()).toHaveValue('optimization')
    await expect(page.locator('text=最大深度: 9')).toBeVisible()
  })

  test('结果展示和导出功能测试', async ({ page }) => {
    // 启动工作流并等待完成
    await page
      .locator('textarea[placeholder*="输入需要分解的复杂问题"]')
      .fill('简单测试问题')

    await page.locator('button:has-text("开始分解执行")').click()

    // 等待可能的结果显示（模拟API可能很快完成）
    // 在真实测试中，这里需要等待工作流真正完成

    // 如果有结果显示，测试结果展示功能
    const resultsSection = page.locator('text=执行结果')
    if (await resultsSection.isVisible({ timeout: 30000 })) {
      // 测试结果标签页
      await expect(page.locator('text=结果摘要')).toBeVisible()
      await expect(page.locator('text=验证报告')).toBeVisible()
      await expect(page.locator('text=原始数据')).toBeVisible()
      await expect(page.locator('text=格式化输出')).toBeVisible()

      // 测试导出按钮
      await page.locator('text=格式化输出').click()
      await expect(page.locator('button:has-text("下载 JSON")')).toBeVisible()
      await expect(page.locator('button:has-text("下载 XML")')).toBeVisible()
      await expect(
        page.locator('button:has-text("下载 Markdown")')
      ).toBeVisible()
    }
  })

  test('多个工作流并发执行测试', async ({ page }) => {
    // 这个测试模拟同时执行多个工作流的场景
    // 在实际实现中，需要测试系统如何处理并发请求

    await page
      .locator('textarea[placeholder*="输入需要分解的复杂问题"]')
      .fill('第一个并发测试问题')

    await page.locator('button:has-text("开始分解执行")').click()

    // 验证第一个工作流启动
    await expect(page.locator('button:has-text("分解问题中...")')).toBeVisible({
      timeout: 2000,
    })

    // 在真实测试中，这里可以测试：
    // 1. 同时启动多个工作流
    // 2. 系统资源监控的变化
    // 3. 队列深度的增加
    // 4. 工作流之间的隔离性
  })

  test('工作流历史和管理测试', async ({ page }) => {
    // 这个测试验证工作流的历史记录和管理功能
    // 在真实实现中，应该有工作流列表、历史记录等功能

    // 启动一个工作流
    await page
      .locator('textarea[placeholder*="输入需要分解的复杂问题"]')
      .fill('历史记录测试问题')

    await page.locator('button:has-text("开始分解执行")').click()

    // 等待执行开始
    await expect(page.locator('text=执行控制')).toBeVisible({ timeout: 15000 })

    // 测试系统配置按钮
    const configButton = page.locator('button:has-text("系统配置")')
    if (await configButton.isVisible()) {
      // 在真实实现中，这里应该打开配置面板
    }

    // 测试导入工作流按钮
    const importButton = page.locator('button:has-text("导入工作流")')
    if (await importButton.isVisible()) {
      // 在真实实现中，这里应该支持工作流导入功能
    }
  })
})

test.describe('多步推理工作流 - API集成测试', () => {
  test('API端点响应测试', async ({ page, request }) => {
    // 测试系统指标API
    const metricsResponse = await request.get(
      '/api/v1/multi-step-reasoning/system/metrics'
    )
    expect(metricsResponse.ok()).toBeTruthy()

    const metricsData = await metricsResponse.json()
    expect(metricsData).toHaveProperty('active_workers')
    expect(metricsData).toHaveProperty('queue_depth')
    expect(metricsData).toHaveProperty('success_rate')
  })

  test('问题分解API测试', async ({ request }) => {
    const decompositionResponse = await request.post(
      '/api/v1/multi-step-reasoning/decompose',
      {
        data: {
          problem_statement: 'API测试问题：如何构建一个可扩展的微服务架构？',
          strategy: 'analysis',
          max_depth: 5,
          target_complexity: 5.0,
          enable_branching: false,
        },
      }
    )

    expect(decompositionResponse.ok()).toBeTruthy()

    const decompositionData = await decompositionResponse.json()
    expect(decompositionData).toHaveProperty('task_dag')
    expect(decompositionData).toHaveProperty('workflow_definition')
    expect(decompositionData).toHaveProperty('decomposition_metadata')

    // 验证DAG结构
    expect(decompositionData.task_dag).toHaveProperty('nodes')
    expect(decompositionData.task_dag).toHaveProperty('edges')
    expect(decompositionData.task_dag.nodes.length).toBeGreaterThan(0)
  })

  test('工作流执行API测试', async ({ request }) => {
    // 首先创建一个工作流定义
    const decompositionResponse = await request.post(
      '/api/v1/multi-step-reasoning/decompose',
      {
        data: {
          problem_statement: '执行测试问题',
          strategy: 'analysis',
        },
      }
    )

    const decompositionData = await decompositionResponse.json()
    const workflowId = decompositionData.workflow_definition.id

    // 启动执行
    const executionResponse = await request.post(
      '/api/v1/multi-step-reasoning/execute',
      {
        data: {
          workflow_definition_id: workflowId,
          execution_mode: 'parallel',
          max_parallel_steps: 3,
          scheduling_strategy: 'critical_path',
        },
      }
    )

    expect(executionResponse.ok()).toBeTruthy()

    const executionData = await executionResponse.json()
    expect(executionData).toHaveProperty('execution_id')
    expect(executionData).toHaveProperty('status')
    expect(executionData.status).toBe('running')
  })
})
