/**
 * 超参数优化系统前端集成测试
 */
import React from 'react'
import {
  render,
  screen,
  fireEvent,
  waitFor,
  within,
} from '@testing-library/react'
import '@testing-library/jest-dom'
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest'
import { BrowserRouter } from 'react-router-dom'
import { ConfigProvider } from 'antd'
import App from '../../src/App'

// 模拟API客户端
vi.mock('../../src/services/apiClient', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
  },
}))

import apiClient from '../../src/services/apiClient'

// 测试数据
const mockExperiment = {
  id: 1,
  name: 'Test Experiment',
  description: '集成测试实验',
  state: 'created',
  algorithm: 'tpe',
  parameter_ranges: {
    learning_rate: { type: 'float', low: 0.001, high: 0.1, log: true },
    batch_size: { type: 'int', low: 16, high: 128, step: 16 },
    optimizer: { type: 'categorical', choices: ['adam', 'sgd', 'rmsprop'] },
  },
  optimization_config: {
    n_trials: 100,
    timeout: 3600,
    direction: 'minimize',
  },
  created_at: '2024-01-01T10:00:00Z',
}

const mockTrials = [
  {
    id: 1,
    experiment_id: 1,
    parameters: { learning_rate: 0.01, batch_size: 32, optimizer: 'adam' },
    value: 0.95,
    state: 'complete',
    metrics: { accuracy: 0.95, loss: 0.05 },
    created_at: '2024-01-01T10:30:00Z',
  },
  {
    id: 2,
    experiment_id: 1,
    parameters: { learning_rate: 0.001, batch_size: 64, optimizer: 'sgd' },
    value: 0.89,
    state: 'complete',
    metrics: { accuracy: 0.89, loss: 0.11 },
    created_at: '2024-01-01T11:00:00Z',
  },
  {
    id: 3,
    experiment_id: 1,
    parameters: { learning_rate: 0.05, batch_size: 128, optimizer: 'rmsprop' },
    value: 0.92,
    state: 'running',
    created_at: '2024-01-01T11:30:00Z',
  },
]

describe('超参数优化系统集成测试', () => {
  beforeEach(() => {
    vi.clearAllMocks()

    // 设置默认的API响应
    ;(apiClient.get as any).mockImplementation((url: string) => {
      if (url.includes('/experiments')) {
        if (url.includes('/1')) {
          return Promise.resolve({ data: mockExperiment })
        }
        return Promise.resolve({ data: [mockExperiment] })
      }
      if (url.includes('/trials')) {
        return Promise.resolve({ data: mockTrials })
      }
      if (url.includes('/statistics')) {
        return Promise.resolve({
          data: {
            total_trials: 3,
            completed_trials: 2,
            running_trials: 1,
            failed_trials: 0,
            best_value: 0.89,
            average_value: 0.92,
          },
        })
      }
      return Promise.resolve({ data: [] })
    })

    ;(apiClient.post as any).mockResolvedValue({ data: { success: true } })
    ;(apiClient.put as any).mockResolvedValue({ data: { success: true } })
    ;(apiClient.delete as any).mockResolvedValue({ data: { success: true } })
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  const renderApp = () => {
    return render(
      <BrowserRouter>
        <ConfigProvider>
          <App />
        </ConfigProvider>
      </BrowserRouter>
    )
  }

  it('完整的实验创建和运行流程', async () => {
    renderApp()

    // 1. 导航到超参数优化页面
    const navItem = screen.getByText('超参数优化系统')
    fireEvent.click(navItem)

    await waitFor(() => {
      expect(screen.getByText('实验管理中心')).toBeInTheDocument()
    })

    // 2. 创建新实验
    const createButton = screen.getByRole('button', { name: /创建实验/ })
    fireEvent.click(createButton)

    // 填写实验表单
    const nameInput = screen.getByLabelText(/实验名称/)
    fireEvent.change(nameInput, { target: { value: 'New Test Experiment' } })

    const descInput = screen.getByLabelText(/描述/)
    fireEvent.change(descInput, { target: { value: '新的集成测试实验' } })

    // 选择算法
    const algorithmSelect = screen.getByLabelText(/优化算法/)
    fireEvent.click(algorithmSelect)
    const tpeOption = screen.getByText('TPE')
    fireEvent.click(tpeOption)

    // 配置参数
    const addParamButton = screen.getByRole('button', { name: /添加参数/ })
    fireEvent.click(addParamButton)

    // 提交表单
    const submitButton = screen.getByRole('button', { name: /确认创建/ })
    fireEvent.click(submitButton)

    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalledWith(
        '/api/v1/hyperparameter-optimization/experiments',
        expect.objectContaining({
          name: 'New Test Experiment',
        })
      )
    })

    // 3. 启动实验
    const startButton = screen.getByRole('button', { name: /启动实验/ })
    fireEvent.click(startButton)

    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalledWith(
        expect.stringContaining('/start')
      )
    })

    // 4. 查看试验进度
    const trialsTab = screen.getByText('试验列表')
    fireEvent.click(trialsTab)

    await waitFor(() => {
      expect(screen.getByText('0.95')).toBeInTheDocument()
      expect(screen.getByText('0.89')).toBeInTheDocument()
    })

    // 5. 查看可视化
    const vizTab = screen.getByText('可视化分析')
    fireEvent.click(vizTab)

    await waitFor(() => {
      expect(screen.getByText(/优化历史/)).toBeInTheDocument()
    })
  })

  it('实验列表页面交互', async () => {
    renderApp()

    // 导航到实验列表
    const navItem = screen.getByText('超参数优化系统')
    fireEvent.click(navItem)

    const experimentsItem = screen.getByText('实验列表')
    fireEvent.click(experimentsItem)

    await waitFor(() => {
      expect(screen.getByText('Test Experiment')).toBeInTheDocument()
    })

    // 搜索功能
    const searchInput = screen.getByPlaceholderText(/搜索实验/)
    fireEvent.change(searchInput, { target: { value: 'Test' } })

    await waitFor(() => {
      expect(apiClient.get).toHaveBeenCalledWith(
        expect.stringContaining('/search?q=Test')
      )
    })

    // 状态筛选
    const stateFilter = screen.getByRole('combobox', { name: /状态筛选/ })
    if (stateFilter) {
      fireEvent.click(stateFilter)
      const createdOption = screen.getByText('已创建')
      fireEvent.click(createdOption)
    }

    // 批量操作
    const selectAll = screen.getByRole('checkbox', { name: /全选/ })
    if (selectAll) {
      fireEvent.click(selectAll)

      const batchDelete = screen.getByRole('button', { name: /批量删除/ })
      if (batchDelete) {
        fireEvent.click(batchDelete)

        // 确认删除
        const confirmButton = screen.getByRole('button', { name: /确认/ })
        fireEvent.click(confirmButton)

        await waitFor(() => {
          expect(apiClient.delete).toHaveBeenCalled()
        })
      }
    }
  })

  it('算法配置页面功能', async () => {
    renderApp()

    // 导航到算法配置
    const navItem = screen.getByText('超参数优化系统')
    fireEvent.click(navItem)

    const algorithmsItem = screen.getByText('算法配置')
    fireEvent.click(algorithmsItem)

    await waitFor(() => {
      expect(screen.getByText(/算法选择/)).toBeInTheDocument()
    })

    // 切换算法
    const algorithmTabs = screen.getAllByRole('tab')
    const cmaesTab = algorithmTabs.find(tab =>
      tab.textContent?.includes('CMA-ES')
    )
    if (cmaesTab) {
      fireEvent.click(cmaesTab)

      await waitFor(() => {
        expect(screen.getByText(/协方差矩阵/)).toBeInTheDocument()
      })
    }

    // 修改算法参数
    const paramInput = screen.getByLabelText(/种群大小/)
    if (paramInput) {
      fireEvent.change(paramInput, { target: { value: '50' } })
    }

    // 保存配置
    const saveButton = screen.getByRole('button', { name: /保存配置/ })
    if (saveButton) {
      fireEvent.click(saveButton)

      await waitFor(() => {
        expect(apiClient.put).toHaveBeenCalled()
      })
    }
  })

  it('可视化分析页面功能', async () => {
    renderApp()

    // 导航到可视化分析
    const navItem = screen.getByText('超参数优化系统')
    fireEvent.click(navItem)

    const vizItem = screen.getByText('可视化分析')
    fireEvent.click(vizItem)

    await waitFor(() => {
      expect(screen.getByText(/优化历史图表/)).toBeInTheDocument()
    })

    // 切换图表类型
    const chartTypeSelect = screen.getByRole('combobox', { name: /图表类型/ })
    if (chartTypeSelect) {
      fireEvent.click(chartTypeSelect)
      const scatterOption = screen.getByText('散点图')
      fireEvent.click(scatterOption)
    }

    // 选择参数
    const xAxisSelect = screen.getByLabelText(/X轴参数/)
    if (xAxisSelect) {
      fireEvent.click(xAxisSelect)
      const learningRateOption = screen.getByText('learning_rate')
      fireEvent.click(learningRateOption)
    }

    // 导出图表
    const exportButton = screen.getByRole('button', { name: /导出/ })
    if (exportButton) {
      fireEvent.click(exportButton)

      await waitFor(() => {
        expect(screen.getByText(/导出成功/)).toBeInTheDocument()
      })
    }
  })

  it('性能监控页面实时更新', async () => {
    renderApp()

    // 导航到性能监控
    const navItem = screen.getByText('超参数优化系统')
    fireEvent.click(navItem)

    const monitorItem = screen.getByText('性能监控')
    fireEvent.click(monitorItem)

    await waitFor(() => {
      expect(screen.getByText(/系统性能监控/)).toBeInTheDocument()
    })

    // 验证自动刷新
    await waitFor(() => {
      expect(screen.getByText(/CPU使用率/)).toBeInTheDocument()
    })

    // 模拟数据更新
    ;(apiClient.get as any).mockResolvedValueOnce({
      data: {
        cpu_usage: 75,
        memory_usage: 60,
        gpu_usage: 80,
        active_trials: 5,
      },
    })

    // 等待自动刷新
    await waitFor(
      () => {
        expect(apiClient.get).toHaveBeenCalledWith(
          expect.stringContaining('/monitoring/metrics')
        )
      },
      { timeout: 6000 }
    )

    // 手动刷新
    const refreshButton = screen.getByRole('button', { name: /刷新/ })
    if (refreshButton) {
      fireEvent.click(refreshButton)

      await waitFor(() => {
        expect(apiClient.get).toHaveBeenCalled()
      })
    }
  })

  it('资源管理页面功能', async () => {
    renderApp()

    // 导航到资源管理
    const navItem = screen.getByText('超参数优化系统')
    fireEvent.click(navItem)

    const resourceItem = screen.getByText('资源管理')
    fireEvent.click(resourceItem)

    await waitFor(() => {
      expect(screen.getByText(/资源池管理/)).toBeInTheDocument()
    })

    // 创建资源池
    const createPoolButton = screen.getByRole('button', { name: /创建资源池/ })
    fireEvent.click(createPoolButton)

    const poolNameInput = screen.getByLabelText(/资源池名称/)
    fireEvent.change(poolNameInput, { target: { value: 'GPU Pool' } })

    const gpuCountInput = screen.getByLabelText(/GPU数量/)
    fireEvent.change(gpuCountInput, { target: { value: '4' } })

    const submitButton = screen.getByRole('button', { name: /确认/ })
    fireEvent.click(submitButton)

    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalledWith(
        expect.stringContaining('/resource-pools'),
        expect.objectContaining({
          name: 'GPU Pool',
          gpu_count: 4,
        })
      )
    })

    // 分配资源
    const allocateTab = screen.getByText('资源分配')
    fireEvent.click(allocateTab)

    await waitFor(() => {
      expect(screen.getByText(/分配资源/)).toBeInTheDocument()
    })
  })

  it('试验调度器页面功能', async () => {
    renderApp()

    // 导航到试验调度器
    const navItem = screen.getByText('超参数优化系统')
    fireEvent.click(navItem)

    const schedulerItem = screen.getByText('试验调度器')
    fireEvent.click(schedulerItem)

    await waitFor(() => {
      expect(screen.getByText(/调度队列/)).toBeInTheDocument()
    })

    // 修改调度策略
    const strategySelect = screen.getByRole('combobox', { name: /调度策略/ })
    if (strategySelect) {
      fireEvent.click(strategySelect)
      const priorityOption = screen.getByText('优先级调度')
      fireEvent.click(priorityOption)
    }

    // 暂停/恢复调度
    const pauseButton = screen.getByRole('button', { name: /暂停调度/ })
    if (pauseButton) {
      fireEvent.click(pauseButton)

      await waitFor(() => {
        expect(apiClient.post).toHaveBeenCalledWith(
          expect.stringContaining('/scheduler/pause')
        )
      })
    }

    // 查看工作节点
    const workersTab = screen.getByText('工作节点')
    fireEvent.click(workersTab)

    await waitFor(() => {
      expect(screen.getByText(/节点状态/)).toBeInTheDocument()
    })
  })

  it('分析报告页面生成', async () => {
    renderApp()

    // 导航到分析报告
    const navItem = screen.getByText('超参数优化系统')
    fireEvent.click(navItem)

    const reportsItem = screen.getByText('分析报告')
    fireEvent.click(reportsItem)

    await waitFor(() => {
      expect(screen.getByText(/生成报告/)).toBeInTheDocument()
    })

    // 选择报告类型
    const reportTypeSelect = screen.getByRole('combobox', { name: /报告类型/ })
    fireEvent.click(reportTypeSelect)
    const performanceOption = screen.getByText('性能分析报告')
    fireEvent.click(performanceOption)

    // 选择实验
    const experimentSelect = screen.getByRole('combobox', { name: /选择实验/ })
    fireEvent.click(experimentSelect)
    const testExpOption = screen.getByText('Test Experiment')
    fireEvent.click(testExpOption)

    // 生成报告
    const generateButton = screen.getByRole('button', { name: /生成报告/ })
    fireEvent.click(generateButton)

    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalledWith(
        expect.stringContaining('/reports/generate'),
        expect.objectContaining({
          type: 'performance',
          experiment_id: 1,
        })
      )
    })

    // 导出报告
    const exportButton = screen.getByRole('button', { name: /导出PDF/ })
    if (exportButton) {
      fireEvent.click(exportButton)

      await waitFor(() => {
        expect(apiClient.get).toHaveBeenCalledWith(
          expect.stringContaining('/reports/export')
        )
      })
    }
  })

  it('端到端工作流：从创建到完成', async () => {
    renderApp()

    // 1. 创建实验
    const navItem = screen.getByText('超参数优化系统')
    fireEvent.click(navItem)

    await waitFor(() => {
      expect(screen.getByText('实验管理中心')).toBeInTheDocument()
    })

    // 2. 配置参数
    const configButton = screen.getByRole('button', { name: /配置参数/ })
    fireEvent.click(configButton)

    // 添加参数
    const addParamButton = screen.getByRole('button', { name: /添加参数/ })
    fireEvent.click(addParamButton)
    fireEvent.click(addParamButton) // 添加两个参数

    // 3. 启动实验
    const startButton = screen.getByRole('button', { name: /启动实验/ })
    fireEvent.click(startButton)

    // 4. 监控进度
    await waitFor(() => {
      expect(screen.getByText(/运行中/)).toBeInTheDocument()
    })

    // 5. 模拟实验完成
    ;(apiClient.get as any).mockResolvedValueOnce({
      data: {
        ...mockExperiment,
        state: 'completed',
      },
    })

    // 6. 查看结果
    const resultsTab = screen.getByText('实验结果')
    if (resultsTab) {
      fireEvent.click(resultsTab)

      await waitFor(() => {
        expect(screen.getByText(/最佳参数/)).toBeInTheDocument()
      })
    }

    // 7. 生成报告
    const reportButton = screen.getByRole('button', { name: /生成分析报告/ })
    if (reportButton) {
      fireEvent.click(reportButton)

      await waitFor(() => {
        expect(screen.getByText(/报告已生成/)).toBeInTheDocument()
      })
    }
  })

  it('错误处理和恢复', async () => {
    renderApp()

    // 模拟API错误
    ;(apiClient.get as any).mockRejectedValueOnce(new Error('Network error'))

    const navItem = screen.getByText('超参数优化系统')
    fireEvent.click(navItem)

    await waitFor(() => {
      expect(screen.getByText(/加载失败/)).toBeInTheDocument()
    })

    // 重试功能
    const retryButton = screen.getByRole('button', { name: /重试/ })
    fireEvent.click(retryButton)

    // 恢复正常响应
    ;(apiClient.get as any).mockResolvedValueOnce({
      data: [mockExperiment],
    })

    await waitFor(() => {
      expect(screen.getByText('Test Experiment')).toBeInTheDocument()
    })
  })

  it('并发操作处理', async () => {
    renderApp()

    const navItem = screen.getByText('超参数优化系统')
    fireEvent.click(navItem)

    await waitFor(() => {
      expect(screen.getByText('实验管理中心')).toBeInTheDocument()
    })

    // 同时启动多个操作
    const promises = []

    // 操作1：启动实验
    const startButton = screen.getByRole('button', { name: /启动实验/ })
    promises.push(fireEvent.click(startButton))

    // 操作2：刷新列表
    const refreshButton = screen.getByRole('button', { name: /刷新/ })
    promises.push(fireEvent.click(refreshButton))

    // 操作3：获取统计
    const statsButton = screen.getByRole('button', { name: /查看统计/ })
    if (statsButton) {
      promises.push(fireEvent.click(statsButton))
    }

    await Promise.all(promises)

    // 验证所有API调用都被触发
    await waitFor(() => {
      expect(apiClient.post).toHaveBeenCalled()
      expect(apiClient.get).toHaveBeenCalledTimes(3)
    })
  })
})
