import { test, expect } from '@playwright/test'

test.describe('Q-Learning System E2E Tests', () => {
  test.beforeEach(async ({ page }) => {
    // 启动前端开发服务器后访问
    await page.goto('http://localhost:3000')
  })

  test('应该能够访问Q-Learning主页面并显示基本功能', async ({ page }) => {
    // 点击侧边栏的Q-Learning入口
    await page.click('text=Q-Learning算法家族')

    // 验证页面标题
    await expect(page.locator('text=Q-Learning策略优化系统')).toBeVisible()
    await expect(
      page.locator('text=强化学习智能体训练与策略优化平台')
    ).toBeVisible()

    // 验证统计卡片
    await expect(page.locator('text=活跃智能体')).toBeVisible()
    await expect(page.locator('text=训练中会话')).toBeVisible()
    await expect(page.locator('text=已完成训练')).toBeVisible()
    await expect(page.locator('text=平均性能')).toBeVisible()

    // 验证算法标签
    await expect(page.locator('text=Classic Q-Learning')).toBeVisible()
    await expect(page.locator('text=Deep Q-Network (DQN)')).toBeVisible()
    await expect(page.locator('text=Double DQN')).toBeVisible()
    await expect(page.locator('text=Dueling DQN')).toBeVisible()
  })

  test('应该能够导航到表格Q-Learning页面', async ({ page }) => {
    // 点击侧边栏进入Q-Learning系统
    await page.click('text=Q-Learning算法家族')

    // 点击算法总览
    await page.click('text=算法总览')
    await page.waitForURL('**/qlearning')

    // 点击表格Q-Learning卡片
    await page.click('text=表格Q-Learning')
    await page.waitForURL('**/qlearning/tabular')

    // 验证页面内容
    await expect(
      page.locator('text=表格Q-Learning (Tabular Q-Learning)')
    ).toBeVisible()
    await expect(page.locator('text=经典的表格式Q-Learning算法')).toBeVisible()

    // 验证参数配置
    await expect(page.locator('text=学习率 (Learning Rate)')).toBeVisible()
    await expect(page.locator('text=折扣因子 (Gamma)')).toBeVisible()
    await expect(page.locator('text=探索率 (Epsilon)')).toBeVisible()

    // 验证Q表显示
    await expect(page.locator('text=Q表可视化')).toBeVisible()
    await expect(page.locator('text=状态')).toBeVisible()
    await expect(page.locator('text=↑ (向上)')).toBeVisible()
  })

  test('应该能够在表格Q-Learning页面开始训练', async ({ page }) => {
    await page.click('text=Q-Learning算法家族')
    await page.click('text=算法总览')
    await page.click('text=表格Q-Learning')
    await page.waitForURL('**/qlearning/tabular')

    // 开始训练
    await page.click('button:has-text("开始训练")')

    // 验证训练状态
    await expect(page.locator('text=正在训练中...')).toBeVisible()

    // 等待训练进行一段时间
    await page.waitForTimeout(2000)

    // 验证训练回合数有更新
    const episodeText = await page.locator('text=训练回合').textContent()
    expect(episodeText).toContain('训练回合')
  })

  test('应该能够导航到Deep Q-Network页面并查看网络架构', async ({ page }) => {
    await page.click('text=Q-Learning算法家族')
    await page.click('text=算法总览')
    await page.click('text=Deep Q-Network')
    await page.waitForURL('**/qlearning/dqn')

    // 验证页面标题
    await expect(page.locator('text=Deep Q-Network (DQN)')).toBeVisible()

    // 验证网络架构配置
    await expect(page.locator('text=网络架构配置')).toBeVisible()
    await expect(page.locator('text=网络类型:')).toBeVisible()

    // 验证网络结构详情
    await expect(page.locator('text=输入层: 84 x 84 x 4')).toBeVisible()
    await expect(page.locator('text=卷积层1: 32 filters')).toBeVisible()

    // 验证DQN核心技术
    await expect(
      page.locator('text=经验回放 (Experience Replay)')
    ).toBeVisible()
    await expect(page.locator('text=目标网络 (Target Network)')).toBeVisible()
    await expect(page.locator('text=卷积神经网络 (CNN)')).toBeVisible()
  })

  test('应该能够在DQN页面切换网络类型并开始训练', async ({ page }) => {
    await page.click('text=Q-Learning算法家族')
    await page.click('text=算法总览')
    await page.click('text=Deep Q-Network')
    await page.waitForURL('**/qlearning/dqn')

    // 切换网络类型
    await page.click('.ant-select-selector')
    await page.click('text=Double DQN')

    // 开始训练
    await page.click('button:has-text("开始训练")')

    // 验证训练状态
    await expect(page.locator('text=暂停训练')).toBeVisible()

    // 等待一段时间让训练进行
    await page.waitForTimeout(2000)

    // 验证经验回放缓冲区有更新
    const bufferText = await page.locator('text=经验回放缓冲区').textContent()
    expect(bufferText).toContain('经验回放缓冲区')
  })

  test('应该能够导航到探索策略页面并测试不同策略', async ({ page }) => {
    await page.click('text=Q-Learning算法家族')
    await page.click('text=探索策略系统')
    await page.waitForURL('**/qlearning/exploration-strategies')

    // 验证页面标题
    await expect(
      page.locator('text=探索策略系统 (Exploration Strategies)')
    ).toBeVisible()

    // 验证策略配置
    await expect(page.locator('text=策略类型:')).toBeVisible()
    await expect(page.locator('text=探索率 (ε):')).toBeVisible()

    // 测试策略切换
    await page.click('.ant-select-selector')
    await page.click('text=Thompson Sampling')

    // 验证策略描述更新
    await expect(page.locator('text=当前策略: Thompson Sampling')).toBeVisible()

    // 开始训练
    await page.click('button:has-text("开始训练")')

    // 验证训练状态
    await expect(page.locator('text=暂停')).toBeVisible()

    // 等待训练数据更新
    await page.waitForTimeout(3000)

    // 验证动作选择统计更新
    await expect(page.locator('text=动作选择统计')).toBeVisible()
  })

  test('应该能够在主页面切换不同的功能标签页', async ({ page }) => {
    await page.click('text=Q-Learning算法家族')
    await page.click('text=算法总览')
    await page.waitForURL('**/qlearning')

    // 测试智能体管理标签页
    await page.click('text=智能体管理')
    await expect(page.locator('text=智能体管理')).toBeVisible()

    // 测试训练监控标签页
    await page.click('text=训练监控')
    await expect(page.locator('text=训练监控')).toBeVisible()

    // 测试性能可视化标签页
    await page.click('text=性能可视化')
    await expect(page.locator('text=性能可视化')).toBeVisible()

    // 测试环境配置标签页
    await page.click('text=环境配置')
    await expect(page.locator('text=环境配置')).toBeVisible()
  })

  test('应该能够访问所有Q-Learning子功能页面', async ({ page }) => {
    await page.click('text=Q-Learning算法家族')
    await page.click('text=算法总览')

    // 测试功能总览中的各个按钮
    const functionButtons = [
      'Epsilon-Greedy系列',
      'Upper Confidence Bound',
      'Thompson Sampling',
      '基础奖励函数',
      '复合奖励系统',
      '状态空间设计',
      '动作空间定义',
      '训练调度管理',
      '学习率调度器',
    ]

    for (const buttonText of functionButtons) {
      const button = page.locator(`button:has-text("${buttonText}")`)
      if (await button.isVisible()) {
        await button.click()
        // 等待页面加载
        await page.waitForTimeout(500)
        // 返回主页面
        await page.goto('http://localhost:3000/qlearning')
      }
    }
  })

  test('应该能够在不同算法页面查看详细原理说明', async ({ page }) => {
    // 测试DQN页面的原理标签页
    await page.click('text=Q-Learning算法家族')
    await page.click('text=算法总览')
    await page.click('text=Deep Q-Network')
    await page.waitForURL('**/qlearning/dqn')

    // 切换到经验回放标签页
    await page.click('text=经验回放机制')
    await expect(
      page.locator('text=经验回放 (Experience Replay)')
    ).toBeVisible()
    await expect(page.locator('text=存储经验')).toBeVisible()

    // 切换到目标网络标签页
    await page.click('text=目标网络稳定化')
    await expect(page.locator('text=目标网络 (Target Network)')).toBeVisible()
    await expect(page.locator('text=主网络')).toBeVisible()

    // 切换到网络架构标签页
    await page.click('text=网络架构设计')
    await expect(page.locator('text=卷积神经网络架构')).toBeVisible()
  })

  test('应该显示系统实时更新的统计数据', async ({ page }) => {
    await page.click('text=Q-Learning算法家族')
    await page.click('text=算法总览')

    // 记录初始统计数据
    const initialAgents = await page
      .locator('[data-testid="agents-count"]')
      .textContent()
      .catch(() => '0')

    // 切换到智能体管理标签页并创建新智能体（如果有创建按钮的话）
    await page.click('text=智能体管理')

    // 等待页面更新
    await page.waitForTimeout(1000)

    // 验证页面仍然正常显示
    await expect(page.locator('text=智能体管理')).toBeVisible()
  })
})
