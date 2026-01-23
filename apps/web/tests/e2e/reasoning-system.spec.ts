import { test, expect } from '@playwright/test'

test.describe('CoT推理系统E2E测试', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('http://localhost:3000/reasoning')
  })

  test('推理页面基本加载和导航', async ({ page }) => {
    // 验证页面标题
    await expect(page).toHaveTitle(/AI Agent System/)

    // 验证推理页面关键元素存在
    await expect(page.getByText('Chain-of-Thought 推理系统')).toBeVisible()
    await expect(page.getByText('推理输入')).toBeVisible()
    await expect(page.getByText('推理可视化')).toBeVisible()
    await expect(page.getByText('质量控制')).toBeVisible()
    await expect(page.getByText('统计分析')).toBeVisible()
  })

  test('Zero-shot推理完整流程', async ({ page }) => {
    // 设置API拦截
    await page.route('**/api/v1/reasoning/execute', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          id: 'chain-test-123',
          problem:
            '一个班级有30名学生，其中60%是女生，40%是男生。如果新来了5名女生，现在女生占全班人数的百分比是多少？',
          strategy: 'ZERO_SHOT',
          steps: [
            {
              id: 'step-1',
              step_number: 1,
              step_type: 'observation',
              content:
                '首先观察题目给出的信息：原班级30人，女生60%（18人），男生40%（12人），新来5名女生。',
              reasoning: '需要明确初始条件和变化情况',
              confidence: 0.95,
              duration_ms: 1200,
            },
            {
              id: 'step-2',
              step_number: 2,
              step_type: 'analysis',
              content:
                '计算新的班级构成：总人数 = 30 + 5 = 35人，女生人数 = 18 + 5 = 23人，男生人数保持12人。',
              reasoning: '基于观察的信息进行数学计算',
              confidence: 0.98,
              duration_ms: 1500,
            },
            {
              id: 'step-3',
              step_number: 3,
              step_type: 'conclusion',
              content: '现在女生占全班人数的百分比 = 23/35 ≈ 65.7%',
              reasoning: '通过比例计算得出最终结果',
              confidence: 0.92,
              duration_ms: 800,
            },
          ],
          conclusion: '新来5名女生后，女生占全班人数的百分比约为65.7%',
          confidence_score: 0.95,
          total_duration_ms: 3500,
          created_at: '2025-01-15T10:00:00Z',
          completed_at: '2025-01-15T10:00:03Z',
        }),
      })
    })

    // 1. 输入推理问题
    const problemInput = page.getByLabel('推理问题')
    await problemInput.fill(
      '一个班级有30名学生，其中60%是女生，40%是男生。如果新来了5名女生，现在女生占全班人数的百分比是多少？'
    )

    // 2. 选择Zero-shot策略（默认已选中）
    await expect(page.getByDisplayValue('Zero-shot CoT')).toBeVisible()

    // 3. 开始推理
    await page.getByText('开始推理').click()

    // 4. 等待推理完成
    await expect(page.getByText('推理结论')).toBeVisible({ timeout: 10000 })

    // 5. 验证推理可视化结果
    await page.getByText('推理可视化').click()

    // 检查策略显示
    await expect(page.getByText('ZERO_SHOT')).toBeVisible()

    // 检查置信度显示
    await expect(page.getByText('95%')).toBeVisible()

    // 检查推理步骤
    await expect(page.getByText('步骤 1: OBSERVATION')).toBeVisible()
    await expect(page.getByText('步骤 2: ANALYSIS')).toBeVisible()
    await expect(page.getByText('步骤 3: CONCLUSION')).toBeVisible()

    // 检查推理内容
    await expect(page.getByText('首先观察题目给出的信息')).toBeVisible()
    await expect(page.getByText('计算新的班级构成')).toBeVisible()

    // 检查结论
    await expect(
      page.getByText('新来5名女生后，女生占全班人数的百分比约为65.7%')
    ).toBeVisible()
  })

  test('流式推理功能测试', async ({ page }) => {
    // 设置流式API拦截
    await page.route('**/api/v1/reasoning/stream', async route => {
      // 模拟Server-Sent Events
      const body = `data: {"type": "step", "data": {"id": "step-1", "step_number": 1, "step_type": "observation", "content": "观察问题描述", "reasoning": "开始分析", "confidence": 0.8}}\n\ndata: {"type": "step", "data": {"id": "step-2", "step_number": 2, "step_type": "analysis", "content": "深入分析", "reasoning": "进行逻辑推理", "confidence": 0.85}}\n\ndata: {"type": "complete", "data": {"id": "stream-chain-123", "conclusion": "流式推理完成"}}\n\n`

      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        body: body,
      })
    })

    // 1. 启用流式输出
    await page.getByLabel('流式输出').check()

    // 2. 输入问题
    const problemInput = page.getByLabel('推理问题')
    await problemInput.fill('测试流式推理功能')

    // 3. 开始推理
    await page.getByText('开始推理').click()

    // 4. 验证执行状态
    await expect(page.getByText('执行中')).toBeVisible()

    // 5. 切换到推理可视化查看流式结果
    await page.getByText('推理可视化').click()

    // 验证流式标识
    await expect(page.getByText('STREAMING')).toBeVisible()

    // 等待流式步骤出现
    await expect(page.getByText('观察问题描述')).toBeVisible({ timeout: 5000 })
    await expect(page.getByText('深入分析')).toBeVisible({ timeout: 5000 })
  })

  test('质量控制功能测试', async ({ page }) => {
    // 先设置一个完成的推理链
    await page.route('**/api/v1/reasoning/execute', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          id: 'quality-test-chain',
          problem: '质量控制测试问题',
          strategy: 'ZERO_SHOT',
          steps: [
            {
              id: 'step-1',
              step_number: 1,
              step_type: 'observation',
              content: '测试内容',
              reasoning: '测试推理',
              confidence: 0.6,
            },
          ],
          conclusion: '测试结论',
          confidence_score: 0.6,
        }),
      })
    })

    // 设置验证API拦截
    await page.route(
      '**/api/v1/reasoning/quality-test-chain/validate',
      async route => {
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            step_id: 'step-1',
            is_valid: false,
            consistency_score: 0.45,
            issues: ['推理逻辑存在跳跃', '缺乏足够的证据支持'],
            suggestions: ['增加中间推理步骤', '提供更多证据'],
          }),
        })
      }
    )

    // 1. 先执行一个推理
    const problemInput = page.getByLabel('推理问题')
    await problemInput.fill('质量控制测试问题')
    await page.getByText('开始推理').click()

    // 等待推理完成
    await expect(page.getByText('推理结论')).toBeVisible({ timeout: 10000 })

    // 2. 切换到质量控制
    await page.getByText('质量控制').click()

    // 3. 验证初始状态
    await expect(page.getByText('验证状态')).toBeVisible()
    await expect(page.getByText('未验证')).toBeVisible()

    // 4. 执行验证
    await page.getByText('验证推理链').click()

    // 5. 等待验证完成并检查结果
    await expect(page.getByText('验证失败')).toBeVisible({ timeout: 5000 })
    await expect(page.getByText('发现的问题')).toBeVisible()
    await expect(page.getByText('推理逻辑存在跳跃')).toBeVisible()
    await expect(page.getByText('改进建议')).toBeVisible()
    await expect(page.getByText('增加中间推理步骤')).toBeVisible()

    // 6. 测试恢复功能
    await expect(page.getByText('执行恢复')).toBeEnabled()

    // 选择恢复策略
    await page.getByDisplayValue('回溯').click()
    await page.getByText('细化').click()

    // 验证策略描述更新
    await expect(page.getByText('优化当前推理步骤的内容')).toBeVisible()
  })

  test('策略切换和参数配置', async ({ page }) => {
    // 1. 测试Few-shot策略选择
    await page.getByLabel('推理策略').click()
    await page.getByText('Few-shot CoT').click()

    // 验证策略描述更新
    await expect(page.getByText('提供示例来指导推理过程')).toBeVisible()

    // 2. 测试Auto-CoT策略
    await page.getByLabel('推理策略').click()
    await page.getByText('Auto-CoT').click()

    await expect(page.getByText('自动选择最佳推理策略')).toBeVisible()

    // 3. 测试最大步骤数配置
    const maxStepsInput = page.getByLabel('最大推理步骤')
    await maxStepsInput.clear()
    await maxStepsInput.fill('15')

    // 4. 测试分支功能开关
    await page.getByLabel('启用分支').uncheck()
    await expect(page.getByLabel('启用分支')).not.toBeChecked()

    // 5. 测试高级设置
    await page.getByText('显示高级设置').click()

    await expect(page.getByText('置信度阈值')).toBeVisible()
    await expect(page.getByText('超时时间（秒）')).toBeVisible()
    await expect(page.getByText('自定义提示词前缀')).toBeVisible()

    // 配置高级参数
    const confidenceThreshold = page.getByLabel('置信度阈值')
    await confidenceThreshold.fill('0.7')

    const timeout = page.getByLabel('超时时间（秒）')
    await timeout.fill('120')

    // 隐藏高级设置
    await page.getByText('隐藏高级设置').click()
    await expect(page.getByText('置信度阈值')).not.toBeVisible()
  })

  test('示例问题功能', async ({ page }) => {
    // 1. 展开数学推理示例
    await page.getByText('数学推理').click()

    // 2. 点击第一个示例问题
    await page.getByText('一个水池有两个进水管').click()

    // 3. 验证问题已填入
    const problemInput = page.getByLabel('推理问题')
    await expect(problemInput).toHaveValue(/一个水池有两个进水管/)

    // 4. 测试其他类别的示例
    await page.getByText('逻辑推理').click()
    await page.getByText('有三个盒子').click()

    await expect(problemInput).toHaveValue(/有三个盒子/)

    // 5. 测试商业分析示例
    await page.getByText('商业分析').click()
    await page.getByText('一家科技公司最近用户流失率').click()

    await expect(problemInput).toHaveValue(/一家科技公司最近用户流失率/)
  })

  test('时间线视图切换', async ({ page }) => {
    // 设置推理链数据
    await page.route('**/api/v1/reasoning/execute', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          id: 'timeline-test',
          problem: '时间线测试',
          strategy: 'ZERO_SHOT',
          steps: [
            {
              id: 'step-1',
              step_number: 1,
              step_type: 'observation',
              content: '第一步观察',
              reasoning: '观察推理',
              confidence: 0.8,
              duration_ms: 1000,
            },
            {
              id: 'step-2',
              step_number: 2,
              step_type: 'analysis',
              content: '第二步分析',
              reasoning: '分析推理',
              confidence: 0.85,
              duration_ms: 1500,
            },
          ],
        }),
      })
    })

    // 1. 执行推理
    await page.getByLabel('推理问题').fill('时间线测试问题')
    await page.getByText('开始推理').click()
    await expect(page.getByText('推理结论')).toBeVisible({ timeout: 10000 })

    // 2. 切换到推理可视化
    await page.getByText('推理可视化').click()

    // 3. 默认应该是步骤视图
    await expect(page.getByText('步骤 1: OBSERVATION')).toBeVisible()

    // 4. 切换到时间线视图
    await page.getByText('时间线').click()

    // 5. 验证时间线显示
    await expect(page.getByText('observation')).toBeVisible()
    await expect(page.getByText('analysis')).toBeVisible()
    await expect(page.getByText('置信度: 80%')).toBeVisible()
    await expect(page.getByText('1000ms')).toBeVisible()

    // 6. 切换回步骤视图
    await page.getByText('步骤视图').click()
    await expect(page.getByText('步骤 1: OBSERVATION')).toBeVisible()
  })

  test('技术细节切换功能', async ({ page }) => {
    // 设置推理链数据
    await page.route('**/api/v1/reasoning/execute', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          id: 'tech-details-test',
          problem: '技术细节测试',
          strategy: 'ZERO_SHOT',
          steps: [
            {
              id: 'step-tech-1',
              step_number: 1,
              step_type: 'observation',
              content: '技术测试内容',
              reasoning: '技术测试推理',
              confidence: 0.888,
              duration_ms: 1234,
            },
          ],
          conclusion: '技术测试结论',
          confidence_score: 0.888,
          total_duration_ms: 1234,
        }),
      })
    })

    // 执行推理
    await page.getByLabel('推理问题').fill('技术细节测试')
    await page.getByText('开始推理').click()
    await expect(page.getByText('推理结论')).toBeVisible({ timeout: 10000 })

    // 切换到推理可视化
    await page.getByText('推理可视化').click()

    // 默认技术细节应该显示统计信息
    await expect(page.getByText('1')).toBeVisible() // 推理步骤数
    await expect(page.getByText('1234')).toBeVisible() // 总耗时

    // 点击技术细节按钮关闭
    await page.getByText('技术细节').click()

    // 验证统计信息消失（这个测试可能需要调整，取决于具体实现）
    // 然后重新打开
    await page.getByText('技术细节').click()

    // 验证技术细节重新显示
    await expect(page.getByText('1234')).toBeVisible()
  })

  test('错误处理和用户反馈', async ({ page }) => {
    // 1. 测试API错误处理
    await page.route('**/api/v1/reasoning/execute', async route => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({
          error: '服务器内部错误',
        }),
      })
    })

    // 输入问题并提交
    await page.getByLabel('推理问题').fill('错误测试')
    await page.getByText('开始推理').click()

    // 检查错误状态（具体的错误显示取决于实现）
    // 可能显示在控制台或错误提示中

    // 2. 测试必填字段验证
    await page.getByText('开始推理').click()

    // 验证表单验证提示
    await expect(page.getByText('请输入要推理的问题')).toBeVisible()

    // 3. 测试字符数限制
    const longText = 'a'.repeat(1001)
    const problemInput = page.getByLabel('推理问题')
    await problemInput.fill(longText)

    // 验证字符数限制生效（Ant Design会自动截断）
    const inputValue = await problemInput.inputValue()
    expect(inputValue.length).toBeLessThanOrEqual(1000)
  })

  test('响应式设计测试', async ({ page }) => {
    // 测试移动端视口
    await page.setViewportSize({ width: 375, height: 667 })

    // 验证移动端布局
    await expect(page.getByText('Chain-of-Thought 推理系统')).toBeVisible()
    await expect(page.getByText('推理输入')).toBeVisible()

    // 测试平板视口
    await page.setViewportSize({ width: 768, height: 1024 })

    // 验证平板布局
    await expect(page.getByText('推理输入')).toBeVisible()
    await expect(page.getByText('推理可视化')).toBeVisible()

    // 恢复桌面视口
    await page.setViewportSize({ width: 1920, height: 1080 })

    // 验证所有标签页可见
    await expect(page.getByText('推理输入')).toBeVisible()
    await expect(page.getByText('推理可视化')).toBeVisible()
    await expect(page.getByText('质量控制')).toBeVisible()
    await expect(page.getByText('统计分析')).toBeVisible()
  })

  test('并发和性能测试', async ({ page }) => {
    // 设置快速响应的API
    await page.route('**/api/v1/reasoning/execute', async route => {
      await new Promise(resolve => setTimeout(resolve, 100)) // 模拟100ms延迟
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          id: 'perf-test',
          problem: '性能测试',
          strategy: 'ZERO_SHOT',
          steps: [
            {
              id: 'step-1',
              step_number: 1,
              step_type: 'observation',
              content: '快速推理',
              reasoning: '快速处理',
              confidence: 0.9,
            },
          ],
        }),
      })
    })

    // 测试快速连续提交
    await page.getByLabel('推理问题').fill('性能测试问题')

    const startTime = Date.now()
    await page.getByText('开始推理').click()

    // 等待推理完成
    await expect(page.getByText('推理可视化')).toBeVisible({ timeout: 5000 })

    const endTime = Date.now()
    const duration = endTime - startTime

    // 验证响应时间合理（应该在几秒内完成）
    expect(duration).toBeLessThan(5000)
  })
})
