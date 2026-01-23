import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import '@testing-library/jest-dom'
import FaultToleranceMonitor from '../FaultToleranceMonitor'

// Mock fetch
global.fetch = jest.fn()

const mockFaultToleranceStats = {
  total_active_connections: 5,
  healthy_connections: 4,
  failed_connections: 1,
  average_uptime: 3600,
  total_reconnections: 3,
  connections: [
    {
      session_id: 'session-1',
      state: 'connected',
      retry_count: 0,
      uptime_seconds: 1800,
      total_reconnections: 0,
      heartbeat_alive: true,
      buffered_messages: 0,
      metrics: {
        total_connections: 1,
        successful_connections: 1,
        failed_connections: 0,
        last_failure_reason: undefined,
      },
    },
    {
      session_id: 'session-2',
      state: 'reconnecting',
      retry_count: 2,
      uptime_seconds: 900,
      total_reconnections: 2,
      heartbeat_alive: false,
      buffered_messages: 3,
      metrics: {
        total_connections: 3,
        successful_connections: 2,
        failed_connections: 1,
        last_failure_reason: 'Connection timeout',
      },
    },
  ],
}

describe('FaultToleranceMonitor', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    ;(fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => mockFaultToleranceStats,
    })
  })

  it('渲染基本组件结构', async () => {
    render(<FaultToleranceMonitor />)

    // 检查标题和描述
    expect(screen.getByText('容错连接监控')).toBeInTheDocument()
    expect(screen.getByText(/实时监控流式处理连接状态/)).toBeInTheDocument()

    // 等待数据加载
    await waitFor(() => {
      expect(screen.getByText('5')).toBeInTheDocument() // 活跃连接数
    })
  })

  it('正确显示统计数据', async () => {
    render(<FaultToleranceMonitor />)

    await waitFor(() => {
      // 检查统计卡片
      expect(screen.getByText('活跃连接')).toBeInTheDocument()
      expect(screen.getByText('健康连接')).toBeInTheDocument()
      expect(screen.getByText('失败连接')).toBeInTheDocument()
      expect(screen.getByText('总重连次数')).toBeInTheDocument()

      // 检查数值
      expect(screen.getByText('5')).toBeInTheDocument() // 总连接数
      expect(screen.getByText('4')).toBeInTheDocument() // 健康连接数
      expect(screen.getByText('1')).toBeInTheDocument() // 失败连接数
      expect(screen.getByText('3')).toBeInTheDocument() // 重连次数
    })
  })

  it('正确计算连接健康度', async () => {
    render(<FaultToleranceMonitor />)

    await waitFor(() => {
      // 健康度应该是 4/5 = 80%
      expect(screen.getByText('80%')).toBeInTheDocument()
      expect(screen.getByText('4/5 连接健康')).toBeInTheDocument()
    })
  })

  it('显示连接详情表格', async () => {
    render(<FaultToleranceMonitor />)

    await waitFor(() => {
      // 检查表格列标题
      expect(screen.getByText('会话ID')).toBeInTheDocument()
      expect(screen.getByText('连接状态')).toBeInTheDocument()
      expect(screen.getByText('运行时间')).toBeInTheDocument()
      expect(screen.getByText('重连次数')).toBeInTheDocument()
      expect(screen.getByText('心跳状态')).toBeInTheDocument()

      // 检查连接数据
      expect(screen.getByText('session-1')).toBeInTheDocument()
      expect(screen.getByText('session-2')).toBeInTheDocument()
      expect(screen.getByText('CONNECTED')).toBeInTheDocument()
      expect(screen.getByText('RECONNECTING')).toBeInTheDocument()
    })
  })

  it('正确显示状态标签和图标', async () => {
    render(<FaultToleranceMonitor />)

    await waitFor(() => {
      // 检查状态标签
      const connectedTag = screen.getByText('CONNECTED')
      const reconnectingTag = screen.getByText('RECONNECTING')

      expect(connectedTag).toBeInTheDocument()
      expect(reconnectingTag).toBeInTheDocument()

      // 检查心跳状态
      expect(screen.getByText('正常')).toBeInTheDocument()
      expect(screen.getByText('异常')).toBeInTheDocument()
    })
  })

  it('处理重连操作', async () => {
    ;(fetch as jest.Mock)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockFaultToleranceStats,
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockFaultToleranceStats,
      })

    render(<FaultToleranceMonitor />)

    await waitFor(() => {
      expect(screen.getByText('session-2')).toBeInTheDocument()
    })

    // 点击重连按钮
    const reconnectButtons = screen.getAllByText('重连')
    fireEvent.click(reconnectButtons[0])

    await waitFor(() => {
      // 验证重连API调用
      expect(fetch).toHaveBeenCalledWith(
        '/api/v1/streaming/fault-tolerance/reconnect/session-2',
        { method: 'POST' }
      )
    })
  })

  it('禁用已连接会话的重连按钮', async () => {
    render(<FaultToleranceMonitor />)

    await waitFor(() => {
      const reconnectButtons = screen.getAllByText('重连')

      // 第一个按钮（connected状态）应该被禁用
      // 第二个按钮（reconnecting状态）应该可用
      expect(reconnectButtons[0]).toBeDisabled()
      expect(reconnectButtons[1]).not.toBeDisabled()
    })
  })

  it('显示最近错误时间线', async () => {
    render(<FaultToleranceMonitor />)

    await waitFor(() => {
      // 检查错误部分
      expect(screen.getByText('最近错误')).toBeInTheDocument()
      expect(screen.getByText(/session-2/)).toBeInTheDocument()
      expect(screen.getByText('Connection timeout')).toBeInTheDocument()
    })
  })

  it('处理API错误', async () => {
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation()
    ;(fetch as jest.Mock).mockRejectedValue(new Error('API Error'))

    render(<FaultToleranceMonitor />)

    await waitFor(() => {
      expect(consoleSpy).toHaveBeenCalledWith(
        '获取容错统计失败:',
        expect.any(Error)
      )
    })

    consoleSpy.mockRestore()
  })

  it('处理刷新操作', async () => {
    render(<FaultToleranceMonitor />)

    await waitFor(() => {
      expect(screen.getByText('活跃连接')).toBeInTheDocument()
    })

    // 点击刷新按钮
    const refreshButton = screen.getByText('刷新')
    fireEvent.click(refreshButton)

    // 验证再次调用API
    expect(fetch).toHaveBeenCalledWith(
      '/api/v1/streaming/fault-tolerance/stats'
    )
  })

  it('正确格式化运行时间', async () => {
    render(<FaultToleranceMonitor />)

    await waitFor(() => {
      // 1800秒 = 0h 30m
      expect(screen.getByText('0h 30m')).toBeInTheDocument()
      // 900秒 = 0h 15m
      expect(screen.getByText('0h 15m')).toBeInTheDocument()
    })
  })

  it('定期自动刷新数据', async () => {
    jest.useFakeTimers()

    render(<FaultToleranceMonitor />)

    // 初始调用
    expect(fetch).toHaveBeenCalledTimes(1)

    // 快进5秒
    jest.advanceTimersByTime(5000)

    await waitFor(() => {
      // 应该再次调用API
      expect(fetch).toHaveBeenCalledTimes(2)
    })

    jest.useRealTimers()
  })

  it('显示空状态', async () => {
    ;(fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({
        total_active_connections: 0,
        healthy_connections: 0,
        failed_connections: 0,
        average_uptime: 0,
        total_reconnections: 0,
        connections: [],
      }),
    })

    render(<FaultToleranceMonitor />)

    await waitFor(() => {
      expect(screen.getByText('100%')).toBeInTheDocument() // 健康度默认100%
      expect(screen.getByText('0/0 连接健康')).toBeInTheDocument()
    })
  })
})
