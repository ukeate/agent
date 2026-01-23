/**
 * BatchProcessingDashboard 组件单元测试
 */

// React import removed - not used in test
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import { describe, it, expect, beforeEach, vi } from 'vitest'
import { BatchProcessingDashboard } from '../BatchProcessingDashboard'
import { batchService, BatchStatus } from '../../../services/batchService'

// Mock batchService
vi.mock('../../../services/batchService', () => ({
  batchService: {
    getJobs: vi.fn(),
    getMetrics: vi.fn(),
    getJobDetails: vi.fn(),
    cancelJob: vi.fn(),
    retryFailedTasks: vi.fn(),
  },
  BatchStatus: {
    PENDING: 'pending',
    RUNNING: 'running',
    COMPLETED: 'completed',
    FAILED: 'failed',
    CANCELLED: 'cancelled',
  },
}))

describe('BatchProcessingDashboard', () => {
  const mockMetrics = {
    active_jobs: 3,
    pending_jobs: 2,
    completed_jobs: 10,
    failed_jobs: 1,
    total_tasks: 1000,
    completed_tasks: 850,
    failed_tasks: 50,
    pending_tasks: 100,
    tasks_per_second: 25.5,
    avg_task_duration: 2.3,
    success_rate: 85,
    queue_depth: 150,
    active_workers: 8,
    max_workers: 10,
  }

  const mockJobs = {
    jobs: [
      {
        id: 'job-1',
        tasks: [],
        status: 'running' as BatchStatus,
        total_tasks: 100,
        completed_tasks: 60,
        failed_tasks: 5,
        created_at: '2025-08-15T10:00:00Z',
        started_at: '2025-08-15T10:01:00Z',
      },
      {
        id: 'job-2',
        tasks: [],
        status: 'completed' as BatchStatus,
        total_tasks: 50,
        completed_tasks: 50,
        failed_tasks: 0,
        created_at: '2025-08-15T09:00:00Z',
        started_at: '2025-08-15T09:01:00Z',
        completed_at: '2025-08-15T09:10:00Z',
      },
    ],
  }

  const mockJobDetails = {
    id: 'job-1',
    tasks: [
      {
        id: 'task-1',
        type: 'process',
        data: { input: 'test' },
        priority: 5,
        retry_count: 0,
        max_retries: 3,
        status: 'completed' as BatchStatus,
        created_at: '2025-08-15T10:00:00Z',
        started_at: '2025-08-15T10:01:00Z',
        completed_at: '2025-08-15T10:02:00Z',
        result: { output: 'success' },
      },
      {
        id: 'task-2',
        type: 'process',
        data: { input: 'test2' },
        priority: 5,
        retry_count: 1,
        max_retries: 3,
        status: 'failed' as BatchStatus,
        created_at: '2025-08-15T10:00:00Z',
        started_at: '2025-08-15T10:01:00Z',
        error: 'Processing failed',
      },
    ],
    status: 'running' as BatchStatus,
    total_tasks: 100,
    completed_tasks: 60,
    failed_tasks: 5,
    created_at: '2025-08-15T10:00:00Z',
    started_at: '2025-08-15T10:01:00Z',
  }

  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(batchService.getJobs).mockResolvedValue(mockJobs)
    vi.mocked(batchService.getMetrics).mockResolvedValue(mockMetrics)
    vi.mocked(batchService.getJobDetails).mockResolvedValue(mockJobDetails)
    vi.mocked(batchService.cancelJob).mockResolvedValue({ success: true })
    vi.mocked(batchService.retryFailedTasks).mockResolvedValue({
      retried_count: 5,
    })
  })

  it('渲染批处理面板并显示加载状态', () => {
    render(<BatchProcessingDashboard />)
    expect(screen.getByText('加载中...')).toBeInTheDocument()
  })

  it('成功加载并显示批处理指标', async () => {
    render(<BatchProcessingDashboard />)

    await waitFor(() => {
      expect(screen.getByText('批处理系统指标')).toBeInTheDocument()
      expect(screen.getByText('3')).toBeInTheDocument() // active_jobs
      expect(screen.getByText('25.5 任务/秒')).toBeInTheDocument()
      expect(screen.getByText('8 / 10')).toBeInTheDocument() // workers
      expect(screen.getByText('150')).toBeInTheDocument() // queue_depth
    })
  })

  it('显示作业列表', async () => {
    render(<BatchProcessingDashboard />)

    await waitFor(() => {
      expect(screen.getByText('批处理作业')).toBeInTheDocument()
      expect(screen.getByText(/作业 job-1/)).toBeInTheDocument()
      expect(screen.getByText(/作业 job-2/)).toBeInTheDocument()
      expect(screen.getByText('running')).toBeInTheDocument()
      expect(screen.getByText('completed')).toBeInTheDocument()
    })
  })

  it('显示作业进度', async () => {
    render(<BatchProcessingDashboard />)

    await waitFor(() => {
      // job-1: (60 + 5) / 100 = 65%
      expect(screen.getByText('65.0%')).toBeInTheDocument()
      // job-2: (50 + 0) / 50 = 100%
      expect(screen.getByText('100.0%')).toBeInTheDocument()
    })
  })

  it('切换自动刷新', async () => {
    render(<BatchProcessingDashboard />)

    await waitFor(() => {
      const checkbox = screen.getByRole('checkbox')
      expect(checkbox).toBeChecked()

      fireEvent.click(checkbox)
      expect(checkbox).not.toBeChecked()
    })
  })

  it('取消运行中的作业', async () => {
    render(<BatchProcessingDashboard />)

    await waitFor(() => {
      const cancelButton = screen.getByText('取消作业')
      fireEvent.click(cancelButton)
    })

    await waitFor(() => {
      expect(batchService.cancelJob).toHaveBeenCalledWith('job-1')
      expect(batchService.getJobs).toHaveBeenCalled()
    })
  })

  it('重试失败的任务', async () => {
    render(<BatchProcessingDashboard />)

    await waitFor(() => {
      const retryButton = screen.getByText('重试失败任务')
      fireEvent.click(retryButton)
    })

    await waitFor(() => {
      expect(batchService.retryFailedTasks).toHaveBeenCalledWith('job-1')
      expect(batchService.getJobs).toHaveBeenCalled()
    })
  })

  it('查看作业详情', async () => {
    render(<BatchProcessingDashboard />)

    await waitFor(() => {
      const detailsButton = screen.getAllByText('查看详情')[0]
      fireEvent.click(detailsButton)
    })

    await waitFor(() => {
      expect(batchService.getJobDetails).toHaveBeenCalledWith('job-1')
      expect(screen.getByText(/作业详情: job-1/)).toBeInTheDocument()
      expect(screen.getByText(/任务 task-1/)).toBeInTheDocument()
      expect(screen.getByText(/任务 task-2/)).toBeInTheDocument()
    })
  })

  it('关闭作业详情模态框', async () => {
    render(<BatchProcessingDashboard />)

    await waitFor(() => {
      const detailsButton = screen.getAllByText('查看详情')[0]
      fireEvent.click(detailsButton)
    })

    await waitFor(() => {
      expect(screen.getByText(/作业详情: job-1/)).toBeInTheDocument()
      const closeButton = screen.getByText('✕')
      fireEvent.click(closeButton)
    })

    await waitFor(() => {
      expect(screen.queryByText(/作业详情: job-1/)).not.toBeInTheDocument()
    })
  })

  it('处理空作业列表', async () => {
    vi.mocked(batchService.getJobs).mockResolvedValue({ jobs: [] })

    render(<BatchProcessingDashboard />)

    await waitFor(() => {
      expect(screen.getByText('暂无批处理作业')).toBeInTheDocument()
    })
  })

  it('处理API错误', async () => {
    const errorMessage = '获取作业列表失败'
    vi.mocked(batchService.getJobs).mockRejectedValue(new Error(errorMessage))

    render(<BatchProcessingDashboard />)

    await waitFor(() => {
      expect(screen.getByText(errorMessage)).toBeInTheDocument()
    })
  })

  it('根据作业状态显示不同颜色', async () => {
    const testCases = [
      { status: 'pending', expectedClass: 'text-yellow-600' },
      { status: 'running', expectedClass: 'text-blue-600' },
      { status: 'completed', expectedClass: 'text-green-600' },
      { status: 'failed', expectedClass: 'text-red-600' },
      { status: 'cancelled', expectedClass: 'text-gray-600' },
    ]

    for (const testCase of testCases) {
      vi.mocked(batchService.getJobs).mockResolvedValue({
        jobs: [
          {
            ...mockJobs.jobs[0],
            status: testCase.status as any,
          },
        ],
      })

      const { container } = render(<BatchProcessingDashboard />)

      await waitFor(() => {
        const statusElement = container.querySelector(
          `.${testCase.expectedClass}`
        )
        expect(statusElement).toBeInTheDocument()
      })
    }
  })

  it('定期刷新数据', async () => {
    vi.useFakeTimers()
    render(<BatchProcessingDashboard />)

    await waitFor(() => {
      expect(batchService.getJobs).toHaveBeenCalledTimes(1)
    })

    vi.advanceTimersByTime(3000)

    await waitFor(() => {
      expect(batchService.getJobs).toHaveBeenCalledTimes(2)
    })

    vi.useRealTimers()
  })
})
