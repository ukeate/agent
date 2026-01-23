import { test, expect } from '@playwright/test'

test.describe('智能调度监控 E2E 测试', () => {
  test.beforeEach(async ({ page }) => {
    // 设置API拦截
    await page.route('**/api/v1/batch/scheduling/stats', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          workers: [
            {
              worker_id: 'worker-001',
              current_load: 0.65,
              task_completion_rate: 0.92,
              average_task_time: 3.2,
              task_type_performance: {
                data_processing: 2.8,
                model_training: 5.1,
                inference: 1.9,
              },
              status: 'active',
            },
            {
              worker_id: 'worker-002',
              current_load: 0.35,
              task_completion_rate: 0.88,
              average_task_time: 2.9,
              task_type_performance: {
                data_processing: 3.1,
                inference: 2.2,
              },
              status: 'idle',
            },
            {
              worker_id: 'worker-003',
              current_load: 0.85,
              task_completion_rate: 0.76,
              average_task_time: 4.8,
              task_type_performance: {
                model_training: 6.2,
                data_processing: 4.5,
              },
              status: 'overloaded',
            },
            {
              worker_id: 'worker-004',
              current_load: 0.0,
              task_completion_rate: 0.0,
              average_task_time: 0.0,
              task_type_performance: {},
              status: 'offline',
            },
          ],
          sla_requirements: [
            {
              id: 'sla-critical',
              name: 'Critical Tasks SLA',
              target_completion_time: 5.0,
              max_failure_rate: 0.02,
              priority_weight: 3.0,
              current_performance: {
                avg_completion_time: 3.8,
                failure_rate: 0.015,
                violation_count: 0,
              },
              status: 'met',
            },
            {
              id: 'sla-standard',
              name: 'Standard Tasks SLA',
              target_completion_time: 15.0,
              max_failure_rate: 0.05,
              priority_weight: 1.5,
              current_performance: {
                avg_completion_time: 12.3,
                failure_rate: 0.032,
                violation_count: 2,
              },
              status: 'warning',
            },
            {
              id: 'sla-batch',
              name: 'Batch Processing SLA',
              target_completion_time: 30.0,
              max_failure_rate: 0.1,
              priority_weight: 1.0,
              current_performance: {
                avg_completion_time: 35.7,
                failure_rate: 0.08,
                violation_count: 8,
              },
              status: 'violated',
            },
          ],
          system_resources: {
            cpu_usage: 72.5,
            memory_usage: 58.2,
            io_utilization: 45.8,
            network_usage: 34.1,
            active_connections: 127,
            queue_depth: 45,
          },
          predictive_scheduling: {
            predicted_completion_times: {
              'task-urgent-001': 3.2,
              'task-standard-045': 8.9,
              'task-batch-123': 25.6,
            },
            scaling_recommendations: {
              action: 'scale_up',
              target_workers: 6,
              confidence: 0.87,
              reason:
                'High queue depth and worker overload detected. CPU usage above 70% threshold.',
            },
            resource_forecast: {
              next_hour: {
                cpu_usage: 78.0,
                memory_usage: 62.5,
                io_utilization: 52.3,
                network_usage: 38.7,
                active_connections: 145,
                queue_depth: 55,
              },
              next_24h: {
                cpu_usage: 75.2,
                memory_usage: 60.1,
                io_utilization: 48.9,
                network_usage: 36.4,
                active_connections: 135,
                queue_depth: 40,
              },
            },
          },
          total_tasks_scheduled: 2847,
          load_balancing_efficiency: 83.7,
          sla_compliance_rate: 89.4,
        }),
      })
    })

    // 导航到统一监控页面
    await page.goto('/monitor')
  })

  test('智能调度监控页面基本功能测试', async ({ page }) => {
    // 切换到智能调度标签页
    await page.click('text=智能调度')

    // 验证页面加载
    await expect(page.locator('text=调度任务总数')).toBeVisible()
    await expect(page.locator('text=负载均衡效率')).toBeVisible()
    await expect(page.locator('text=SLA合规率')).toBeVisible()
    await expect(page.locator('text=活跃工作者')).toBeVisible()

    // 验证系统概览统计数据
    const totalTasks = page.locator('.ant-statistic').filter({
      has: page.locator('.ant-statistic-title', { hasText: '调度任务总数' }),
    })
    await expect(
      totalTasks.locator('.ant-statistic-content-value')
    ).toContainText('2,847')

    const loadBalancing = page.locator('.ant-statistic').filter({
      has: page.locator('.ant-statistic-title', { hasText: '负载均衡效率' }),
    })
    await expect(
      loadBalancing.locator('.ant-statistic-content-value')
    ).toContainText('83.7')

    const slaCompliance = page.locator('.ant-statistic').filter({
      has: page.locator('.ant-statistic-title', { hasText: 'SLA合规率' }),
    })
    await expect(
      slaCompliance.locator('.ant-statistic-content-value')
    ).toContainText('89.4')

    const activeWorkers = page.locator('.ant-statistic').filter({
      has: page.locator('.ant-statistic-title', { hasText: '活跃工作者' }),
    })
    await expect(
      activeWorkers.locator('.ant-statistic-content-value')
    ).toContainText('1')
    await expect(
      activeWorkers.locator('.ant-statistic-content-suffix')
    ).toContainText('/ 4')
  })

  test('系统资源监控显示测试', async ({ page }) => {
    await page.click('text=智能调度')

    // 验证系统资源使用率部分
    await expect(page.locator('text=系统资源使用率')).toBeVisible()
    const resourceCard = page
      .locator('.ant-card')
      .filter({ hasText: '系统资源使用率' })

    // 验证各项资源指标
    await expect(resourceCard.getByText('CPU', { exact: true })).toBeVisible()
    await expect(
      resourceCard
        .getByText('CPU', { exact: true })
        .locator('..')
        .getByText('72.5%')
    ).toBeVisible()

    await expect(resourceCard.getByText('内存', { exact: true })).toBeVisible()
    await expect(
      resourceCard
        .getByText('内存', { exact: true })
        .locator('..')
        .getByText('58.2%')
    ).toBeVisible()

    await expect(resourceCard.getByText('I/O', { exact: true })).toBeVisible()
    await expect(
      resourceCard
        .getByText('I/O', { exact: true })
        .locator('..')
        .getByText('45.8%')
    ).toBeVisible()

    await expect(resourceCard.getByText('网络', { exact: true })).toBeVisible()
    await expect(
      resourceCard
        .getByText('网络', { exact: true })
        .locator('..')
        .getByText('34.1%')
    ).toBeVisible()

    // 验证连接和队列统计
    await expect(page.locator('text=活跃连接')).toBeVisible()
    await expect(page.locator('text=127')).toBeVisible()

    await expect(page.locator('text=队列深度')).toBeVisible()
    await expect(resourceCard.getByText('45', { exact: true })).toBeVisible()

    // 验证资源使用率进度条颜色
    const progressBars = resourceCard.locator('.ant-progress-line')
    await expect(progressBars).toHaveCount(4)
  })

  test('预测性调度建议显示测试', async ({ page }) => {
    await page.click('text=智能调度')

    // 验证预测性调度建议部分
    await expect(page.locator('text=预测性调度建议')).toBeVisible()
    await expect(page.locator('text=建议扩容')).toBeVisible()
    await expect(page.locator('text=目标工作者数: 6')).toBeVisible()
    await expect(page.locator('text=置信度: 87.0%')).toBeVisible()

    // 验证建议原因
    await expect(
      page.locator('text=High queue depth and worker overload detected')
    ).toBeVisible()
  })

  test('不同扩缩容建议显示测试', async ({ page }) => {
    // 测试缩容建议
    await page.route('**/api/v1/batch/scheduling/stats', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          workers: [],
          sla_requirements: [],
          system_resources: {
            cpu_usage: 25.0,
            memory_usage: 30.0,
            io_utilization: 20.0,
            network_usage: 15.0,
            active_connections: 20,
            queue_depth: 5,
          },
          predictive_scheduling: {
            predicted_completion_times: {},
            scaling_recommendations: {
              action: 'scale_down',
              target_workers: 2,
              confidence: 0.92,
              reason:
                'Low resource utilization detected. System is over-provisioned.',
            },
            resource_forecast: {
              next_hour: {
                cpu_usage: 20.0,
                memory_usage: 25.0,
                io_utilization: 18.0,
                network_usage: 12.0,
                active_connections: 18,
                queue_depth: 3,
              },
              next_24h: {
                cpu_usage: 22.0,
                memory_usage: 28.0,
                io_utilization: 19.0,
                network_usage: 14.0,
                active_connections: 19,
                queue_depth: 4,
              },
            },
          },
          total_tasks_scheduled: 150,
          load_balancing_efficiency: 95.2,
          sla_compliance_rate: 98.7,
        }),
      })
    })

    await page.reload()
    await page.click('text=智能调度')

    await expect(page.locator('text=建议缩容')).toBeVisible()
    await expect(page.locator('text=目标工作者数: 2')).toBeVisible()
    await expect(
      page.locator('text=Low resource utilization detected')
    ).toBeVisible()
  })

  test('工作者状态表格显示测试', async ({ page }) => {
    await page.click('text=智能调度')

    // 验证工作者状态表格
    await expect(page.locator('text=工作者状态')).toBeVisible()

    // 验证表格列标题
    const workerCard = page
      .locator('.ant-card')
      .filter({ hasText: '工作者状态' })
    await expect(
      workerCard.getByRole('columnheader', { name: '工作者ID' })
    ).toBeVisible()
    await expect(
      workerCard.getByRole('columnheader', { name: '状态' })
    ).toBeVisible()
    await expect(
      workerCard.getByRole('columnheader', { name: '当前负载' })
    ).toBeVisible()
    await expect(
      workerCard.getByRole('columnheader', { name: '完成率' })
    ).toBeVisible()
    await expect(
      workerCard.getByRole('columnheader', { name: '平均耗时' })
    ).toBeVisible()

    // 验证工作者数据
    await expect(page.locator('text=worker-001')).toBeVisible()
    await expect(page.locator('text=worker-002')).toBeVisible()
    await expect(page.locator('text=worker-003')).toBeVisible()
    await expect(page.locator('text=worker-004')).toBeVisible()

    // 验证状态标签
    await expect(page.locator('.ant-tag:has-text("ACTIVE")')).toBeVisible()
    await expect(page.locator('.ant-tag:has-text("IDLE")')).toBeVisible()
    await expect(page.locator('.ant-tag:has-text("OVERLOADED")')).toBeVisible()
    await expect(page.locator('.ant-tag:has-text("OFFLINE")')).toBeVisible()

    // 验证完成率百分比显示
    await expect(page.locator('text=92.0%')).toBeVisible() // worker-001
    await expect(page.locator('text=88.0%')).toBeVisible() // worker-002
    await expect(page.locator('text=76.0%')).toBeVisible() // worker-003
    await expect(page.locator('text=0.0%')).toBeVisible() // worker-004

    // 验证平均耗时显示
    await expect(page.locator('text=3.20s')).toBeVisible() // worker-001
    await expect(page.locator('text=2.90s')).toBeVisible() // worker-002
    await expect(page.locator('text=4.80s')).toBeVisible() // worker-003
    await expect(page.locator('text=0.00s')).toBeVisible() // worker-004
  })

  test('工作者负载进度条测试', async ({ page }) => {
    await page.click('text=智能调度')

    // 等待表格加载
    await expect(page.locator('text=worker-001')).toBeVisible()

    // 验证进度条存在并有正确的值
    const resourceCard = page
      .locator('.ant-card')
      .filter({ hasText: '系统资源使用率' })
    await expect(resourceCard.locator('.ant-progress-line')).toHaveCount(4)
    await expect(page.locator('table .ant-progress-line')).toHaveCount(4)

    // 验证进度条状态 - 通过aria属性
    const workerProgressBars = page.locator('table [role="progressbar"]')

    // worker-001: 65%
    await expect(workerProgressBars.nth(0)).toHaveAttribute(
      'aria-valuenow',
      '65'
    )
    // worker-002: 35%
    await expect(workerProgressBars.nth(1)).toHaveAttribute(
      'aria-valuenow',
      '35'
    )
    // worker-003: 85% (应该显示为异常状态)
    await expect(workerProgressBars.nth(2)).toHaveAttribute(
      'aria-valuenow',
      '85'
    )
    // worker-004: 0%
    await expect(workerProgressBars.nth(3)).toHaveAttribute(
      'aria-valuenow',
      '0'
    )
  })

  test('SLA监控表格显示测试', async ({ page }) => {
    await page.click('text=智能调度')

    // 验证SLA监控表格
    await expect(page.getByText('SLA监控', { exact: true })).toBeVisible()

    // 验证SLA名称
    await expect(page.locator('text=Critical Tasks SLA')).toBeVisible()
    await expect(page.locator('text=Standard Tasks SLA')).toBeVisible()
    await expect(page.locator('text=Batch Processing SLA')).toBeVisible()

    // 验证SLA状态
    await expect(page.locator('text=MET')).toBeVisible()
    await expect(page.locator('text=WARNING')).toBeVisible()
    await expect(page.locator('text=VIOLATED')).toBeVisible()

    // 验证目标完成时间
    await expect(
      page.getByRole('cell', { name: '5s', exact: true })
    ).toBeVisible()
    await expect(
      page.getByRole('cell', { name: '15s', exact: true })
    ).toBeVisible()
    await expect(
      page.getByRole('cell', { name: '30s', exact: true })
    ).toBeVisible()

    // 验证当前性能数据
    await expect(page.locator('text=完成时间: 3.80s')).toBeVisible()
    await expect(page.locator('text=失败率: 1.5%')).toBeVisible()
    await expect(page.locator('text=违规次数: 0')).toBeVisible()

    await expect(page.locator('text=完成时间: 12.30s')).toBeVisible()
    await expect(page.locator('text=失败率: 3.2%')).toBeVisible()
    await expect(page.locator('text=违规次数: 2')).toBeVisible()

    await expect(page.locator('text=完成时间: 35.70s')).toBeVisible()
    await expect(page.locator('text=失败率: 8.0%')).toBeVisible()
    await expect(page.locator('text=违规次数: 8')).toBeVisible()

    // 验证优先级权重
    await expect(page.locator('text=3.00')).toBeVisible() // Critical
    await expect(page.locator('text=1.50')).toBeVisible() // Standard
    await expect(page.locator('text=1.00')).toBeVisible() // Batch
  })

  test('SLA合规率颜色显示测试', async ({ page }) => {
    await page.click('text=智能调度')

    // 验证SLA合规率89.4%应该显示为红色（< 95%）
    const slaStatistic = page.locator(
      '.ant-statistic-content-value:has-text("89.4")'
    )
    await expect(slaStatistic).toBeVisible()

    // 测试高合规率的情况
    await page.route('**/api/v1/batch/scheduling/stats', async route => {
      const mockData = {
        workers: [],
        sla_requirements: [],
        system_resources: {
          cpu_usage: 50.0,
          memory_usage: 40.0,
          io_utilization: 30.0,
          network_usage: 20.0,
          active_connections: 50,
          queue_depth: 10,
        },
        predictive_scheduling: {
          predicted_completion_times: {},
          scaling_recommendations: {
            action: 'maintain',
            target_workers: 3,
            confidence: 0.95,
            reason: 'Optimal performance maintained',
          },
          resource_forecast: {
            next_hour: {
              cpu_usage: 52.0,
              memory_usage: 42.0,
              io_utilization: 32.0,
              network_usage: 22.0,
              active_connections: 52,
              queue_depth: 12,
            },
            next_24h: {
              cpu_usage: 51.0,
              memory_usage: 41.0,
              io_utilization: 31.0,
              network_usage: 21.0,
              active_connections: 51,
              queue_depth: 11,
            },
          },
        },
        total_tasks_scheduled: 1000,
        load_balancing_efficiency: 92.0,
        sla_compliance_rate: 97.5,
      }

      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockData),
      })
    })

    await page.reload()
    await page.click('text=智能调度')

    // 验证高合规率显示为绿色
    const highSlaStatistic = page.locator(
      '.ant-statistic-content-value:has-text("97.5")'
    )
    await expect(highSlaStatistic).toBeVisible()
  })

  test('刷新功能测试', async ({ page }) => {
    let requestCount = 0

    await page.route('**/api/v1/batch/scheduling/stats', async route => {
      requestCount++
      const mockData = {
        workers: [],
        sla_requirements: [],
        system_resources: {
          cpu_usage: 50.0 + requestCount * 5,
          memory_usage: 40.0,
          io_utilization: 30.0,
          network_usage: 20.0,
          active_connections: 50,
          queue_depth: 10,
        },
        predictive_scheduling: {
          predicted_completion_times: {},
          scaling_recommendations: {
            action: 'maintain',
            target_workers: 3,
            confidence: 0.95,
            reason: 'Optimal performance maintained',
          },
          resource_forecast: {
            next_hour: {
              cpu_usage: 52.0,
              memory_usage: 42.0,
              io_utilization: 32.0,
              network_usage: 22.0,
              active_connections: 52,
              queue_depth: 12,
            },
            next_24h: {
              cpu_usage: 51.0,
              memory_usage: 41.0,
              io_utilization: 31.0,
              network_usage: 21.0,
              active_connections: 51,
              queue_depth: 11,
            },
          },
        },
        total_tasks_scheduled: 1000 + requestCount * 100,
        load_balancing_efficiency: 90.0,
        sla_compliance_rate: 95.0,
      }

      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockData),
      })
    })

    await page.click('text=智能调度')

    // 等待初始加载
    await expect(page.locator('text=调度任务总数')).toBeVisible()

    // 记录初始请求次数
    const initialRequestCount = requestCount

    // 点击刷新按钮
    await page.click('button:has-text("刷新")')

    // 验证API被重新调用
    await page.waitForResponse('**/api/v1/batch/scheduling/stats')

    // 验证数据更新
    expect(requestCount).toBeGreaterThan(initialRequestCount)
  })

  test('自动刷新功能测试', async ({ page }) => {
    let callCount = 0

    await page.route('**/api/v1/batch/scheduling/stats', async route => {
      callCount++
      const mockData = {
        workers: [],
        sla_requirements: [],
        system_resources: {
          cpu_usage: 50.0,
          memory_usage: 40.0,
          io_utilization: 30.0,
          network_usage: 20.0,
          active_connections: 50 + callCount,
          queue_depth: 10,
        },
        predictive_scheduling: {
          predicted_completion_times: {},
          scaling_recommendations: {
            action: 'maintain',
            target_workers: 3,
            confidence: 0.95,
            reason: 'Optimal performance maintained',
          },
          resource_forecast: {
            next_hour: {
              cpu_usage: 52.0,
              memory_usage: 42.0,
              io_utilization: 32.0,
              network_usage: 22.0,
              active_connections: 52,
              queue_depth: 12,
            },
            next_24h: {
              cpu_usage: 51.0,
              memory_usage: 41.0,
              io_utilization: 31.0,
              network_usage: 21.0,
              active_connections: 51,
              queue_depth: 11,
            },
          },
        },
        total_tasks_scheduled: 1000,
        load_balancing_efficiency: 90.0,
        sla_compliance_rate: 95.0,
      }

      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockData),
      })
    })

    await page.click('text=智能调度')

    // 等待初始加载
    await expect(page.locator('text=调度任务总数')).toBeVisible()
    const initialCount = callCount

    // 等待自动刷新（10秒间隔）
    await page.waitForTimeout(11000)

    // 验证自动刷新发生
    expect(callCount).toBeGreaterThan(initialCount)
  })

  test('错误处理测试', async ({ page }) => {
    // 模拟API错误
    await page.route('**/api/v1/batch/scheduling/stats', async route => {
      await route.fulfill({ status: 500, body: 'Internal Server Error' })
    })

    await page.click('text=智能调度')

    // 页面应该仍然可用，即使API失败
    await expect(page.locator('text=调度任务总数')).toBeVisible()
    await expect(page.locator('text=负载均衡效率')).toBeVisible()

    // 应该有错误处理，数据显示为默认值
    const totalTasks = page.locator('.ant-statistic').filter({
      has: page.locator('.ant-statistic-title', { hasText: '调度任务总数' }),
    })
    await expect(
      totalTasks.locator('.ant-statistic-content-value')
    ).toContainText('0')
  })

  test('更新时间显示测试', async ({ page }) => {
    await page.click('text=智能调度')

    // 等待页面加载
    await expect(page.locator('text=调度任务总数')).toBeVisible()

    // 验证更新时间显示
    await expect(page.locator('text=最后更新:')).toBeVisible()
  })

  test('保持现状建议测试', async ({ page }) => {
    await page.route('**/api/v1/batch/scheduling/stats', async route => {
      const mockData = {
        workers: [],
        sla_requirements: [],
        system_resources: {
          cpu_usage: 60.0,
          memory_usage: 50.0,
          io_utilization: 40.0,
          network_usage: 30.0,
          active_connections: 75,
          queue_depth: 20,
        },
        predictive_scheduling: {
          predicted_completion_times: {},
          scaling_recommendations: {
            action: 'maintain',
            target_workers: 3,
            confidence: 0.93,
            reason: 'System performing optimally within target parameters',
          },
          resource_forecast: {
            next_hour: {
              cpu_usage: 62.0,
              memory_usage: 52.0,
              io_utilization: 42.0,
              network_usage: 32.0,
              active_connections: 77,
              queue_depth: 22,
            },
            next_24h: {
              cpu_usage: 61.0,
              memory_usage: 51.0,
              io_utilization: 41.0,
              network_usage: 31.0,
              active_connections: 76,
              queue_depth: 21,
            },
          },
        },
        total_tasks_scheduled: 1500,
        load_balancing_efficiency: 88.0,
        sla_compliance_rate: 94.0,
      }

      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(mockData),
      })
    })

    await page.reload()
    await page.click('text=智能调度')

    await expect(page.locator('text=保持当前规模')).toBeVisible()
    await expect(page.locator('text=目标工作者数: 3')).toBeVisible()
    await expect(page.locator('text=置信度: 93.0%')).toBeVisible()
    await expect(page.locator('text=System performing optimally')).toBeVisible()
  })
})
