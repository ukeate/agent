import React from 'react'
import { render, screen, waitFor, fireEvent, act } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest'
import AgentClusterManagementPage from '../../../src/pages/AgentClusterManagementPage'

// Mock WebSocket
class MockWebSocket {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSING = 2
  static CLOSED = 3

  readyState = MockWebSocket.CONNECTING
  onopen: ((event: Event) => void) | null = null
  onclose: ((event: CloseEvent) => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null
  onerror: ((event: Event) => void) | null = null

  constructor(public url: string) {
    setTimeout(() => {
      this.readyState = MockWebSocket.OPEN
      this.onopen?.(new Event('open'))
    }, 10)
  }

  close() {
    this.readyState = MockWebSocket.CLOSED
    this.onclose?.(new CloseEvent('close'))
  }

  send(data: string) {
    // Mock send implementation
  }

  simulateMessage(data: any) {
    if (this.onmessage) {
      this.onmessage(
        new MessageEvent('message', { data: JSON.stringify(data) })
      )
    }
  }
}

// Mock fetch
const mockFetch = vi.fn()
global.fetch = mockFetch

// Mock WebSocket
global.WebSocket = MockWebSocket as any

// Mock data
const mockClusterStats = {
  cluster_id: 'test-cluster',
  total_agents: 3,
  running_agents: 2,
  healthy_agents: 2,
  health_score: 0.85,
  resource_usage: {
    cpu_usage: 45.5,
    memory_usage: 60.2,
    active_tasks: 8,
    total_requests: 1250,
    error_rate: 0.02,
    avg_response_time: 185.5,
  },
  groups_count: 1,
  updated_at: Date.now() / 1000,
}

const mockAgents = [
  {
    agent_id: 'agent-1',
    name: 'Agent 1',
    endpoint: 'http://localhost:8081',
    status: 'running',
    capabilities: ['compute', 'reasoning'],
    is_healthy: true,
    uptime: 3661, // 1 hour 1 minute 1 second
    resource_usage: {
      cpu_usage: 35.5,
      memory_usage: 55.0,
      active_tasks: 3,
      error_rate: 0.01,
    },
    labels: { environment: 'production' },
    created_at: Date.now() / 1000 - 7200,
    updated_at: Date.now() / 1000,
  },
  {
    agent_id: 'agent-2',
    name: 'Agent 2',
    endpoint: 'http://localhost:8082',
    status: 'running',
    capabilities: ['multimodal'],
    is_healthy: true,
    uptime: 1800, // 30 minutes
    resource_usage: {
      cpu_usage: 55.5,
      memory_usage: 65.2,
      active_tasks: 5,
      error_rate: 0.03,
    },
    labels: { environment: 'production' },
    created_at: Date.now() / 1000 - 3600,
    updated_at: Date.now() / 1000,
  },
  {
    agent_id: 'agent-3',
    name: 'Agent 3',
    endpoint: 'http://localhost:8083',
    status: 'stopped',
    capabilities: ['storage'],
    is_healthy: false,
    uptime: 0,
    resource_usage: {
      cpu_usage: 0,
      memory_usage: 0,
      active_tasks: 0,
      error_rate: 0,
    },
    labels: { environment: 'staging' },
    created_at: Date.now() / 1000 - 1800,
    updated_at: Date.now() / 1000,
  },
]

const mockScalingRecommendations = {
  'group-1': {
    action: 'scale_up',
    reason: 'high_cpu',
    current_instances: 2,
    target_instances: 3,
    confidence: 0.8,
    metrics: {
      cpu_usage_percent: 85.0,
      memory_usage_percent: 70.0,
    },
  },
}

const mockMetricsData = {
  cpu_usage_percent: Array.from({ length: 20 }, (_, i) => ({
    value: 40 + Math.random() * 20,
    timestamp: Date.now() / 1000 - (19 - i) * 60,
  })),
  memory_usage_percent: Array.from({ length: 20 }, (_, i) => ({
    value: 50 + Math.random() * 15,
    timestamp: Date.now() / 1000 - (19 - i) * 60,
  })),
  error_rate: Array.from({ length: 20 }, (_, i) => ({
    value: Math.random() * 0.05,
    timestamp: Date.now() / 1000 - (19 - i) * 60,
  })),
}

describe('AgentClusterManagementPage', () => {
  let mockWebSocket: MockWebSocket

  beforeEach(() => {
    vi.clearAllMocks()

    // Setup default fetch responses
    mockFetch.mockImplementation((url: string, options?: any) => {
      const path = url.replace('/api/v1/cluster', '')

      switch (path) {
        case '/status':
          return Promise.resolve({
            ok: true,
            json: () =>
              Promise.resolve({ success: true, data: mockClusterStats }),
          })
        case '/agents':
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ success: true, data: mockAgents }),
          })
        case '/scaling/recommendations':
          return Promise.resolve({
            ok: true,
            json: () =>
              Promise.resolve({
                success: true,
                data: mockScalingRecommendations,
              }),
          })
        case '/metrics/query':
          return Promise.resolve({
            ok: true,
            json: () =>
              Promise.resolve({ success: true, data: mockMetricsData }),
          })
        default:
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ success: true, data: {} }),
          })
      }
    })
  })

  afterEach(() => {
    if (mockWebSocket) {
      mockWebSocket.close()
    }
  })

  it('renders loading state initially', () => {
    render(<AgentClusterManagementPage />)
    expect(screen.getByRole('progressbar')).toBeInTheDocument()
  })

  it('renders cluster overview after loading', async () => {
    render(<AgentClusterManagementPage />)

    await waitFor(() => {
      expect(screen.getByText('智能体集群管理平台')).toBeInTheDocument()
    })

    // Check cluster stats cards
    expect(screen.getByText('集群状态')).toBeInTheDocument()
    expect(screen.getByText('3')).toBeInTheDocument() // total agents
    expect(screen.getByText('运行状态')).toBeInTheDocument()
    expect(screen.getByText('2')).toBeInTheDocument() // running agents

    // Check health score
    expect(screen.getByText('健康度: 85.0%')).toBeInTheDocument()
  })

  it('displays agents in the management tab', async () => {
    render(<AgentClusterManagementPage />)

    await waitFor(() => {
      expect(screen.getByText('智能体集群管理平台')).toBeInTheDocument()
    })

    // Switch to agent management tab
    fireEvent.click(screen.getByText('智能体管理'))

    await waitFor(() => {
      expect(screen.getByText('Agent 1')).toBeInTheDocument()
      expect(screen.getByText('Agent 2')).toBeInTheDocument()
      expect(screen.getByText('Agent 3')).toBeInTheDocument()
    })

    // Check agent statuses
    expect(screen.getAllByText('running')).toHaveLength(2)
    expect(screen.getByText('stopped')).toBeInTheDocument()

    // Check resource usage display
    expect(screen.getByText('35.5%')).toBeInTheDocument() // CPU usage for agent-1
    expect(screen.getByText('55.0%')).toBeInTheDocument() // Memory usage for agent-1
  })

  it('displays uptime correctly', async () => {
    render(<AgentClusterManagementPage />)

    await waitFor(() => {
      expect(screen.getByText('智能体集群管理平台')).toBeInTheDocument()
    })

    fireEvent.click(screen.getByText('智能体管理'))

    await waitFor(() => {
      // Agent 1: 3661 seconds = 1h 1m
      expect(screen.getByText('1h 1m')).toBeInTheDocument()
      // Agent 2: 1800 seconds = 0h 30m
      expect(screen.getByText('0h 30m')).toBeInTheDocument()
    })
  })

  it('handles agent operations', async () => {
    const user = userEvent.setup()
    render(<AgentClusterManagementPage />)

    await waitFor(() => {
      expect(screen.getByText('智能体集群管理平台')).toBeInTheDocument()
    })

    fireEvent.click(screen.getByText('智能体管理'))

    await waitFor(() => {
      expect(screen.getByText('Agent 3')).toBeInTheDocument()
    })

    // Mock successful start operation
    mockFetch.mockImplementationOnce(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ success: true, data: { success: true } }),
      })
    )

    // Find and click start button for stopped agent (Agent 3)
    const startButtons = screen.getAllByTitle('启动')
    await user.click(startButtons[0])

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        '/api/v1/cluster/agents/agent-3/start',
        expect.objectContaining({ method: 'POST' })
      )
    })
  })

  it('opens agent details dialog', async () => {
    const user = userEvent.setup()
    render(<AgentClusterManagementPage />)

    await waitFor(() => {
      expect(screen.getByText('智能体集群管理平台')).toBeInTheDocument()
    })

    fireEvent.click(screen.getByText('智能体管理'))

    await waitFor(() => {
      expect(screen.getByText('Agent 1')).toBeInTheDocument()
    })

    // Click view details button
    const viewButtons = screen.getAllByTitle('查看详情')
    await user.click(viewButtons[0])

    await waitFor(() => {
      expect(screen.getByText('智能体详情 - Agent 1')).toBeInTheDocument()
      expect(screen.getByText('ID: agent-1')).toBeInTheDocument()
      expect(
        screen.getByText('端点: http://localhost:8081')
      ).toBeInTheDocument()
    })

    // Close dialog
    await user.click(screen.getByText('关闭'))

    await waitFor(() => {
      expect(screen.queryByText('智能体详情 - Agent 1')).not.toBeInTheDocument()
    })
  })

  it('displays scaling recommendations', async () => {
    render(<AgentClusterManagementPage />)

    await waitFor(() => {
      expect(screen.getByText('智能体集群管理平台')).toBeInTheDocument()
    })

    // Switch to auto scaling tab
    fireEvent.click(screen.getByText('自动扩缩容'))

    await waitFor(() => {
      expect(screen.getByText('分组: group-1')).toBeInTheDocument()
      expect(screen.getByText('扩容')).toBeInTheDocument()
      expect(screen.getByText('2 → 3')).toBeInTheDocument()
      expect(screen.getByText('置信度: 80%')).toBeInTheDocument()
    })
  })

  it('handles WebSocket connection status', async () => {
    render(<AgentClusterManagementPage />)

    await waitFor(() => {
      expect(screen.getByText('智能体集群管理平台')).toBeInTheDocument()
    })

    // Initially should show connected status
    await waitFor(() => {
      expect(screen.getByText('连接状态: connected')).toBeInTheDocument()
    })
  })

  it('handles WebSocket messages', async () => {
    render(<AgentClusterManagementPage />)

    await waitFor(() => {
      expect(screen.getByText('智能体集群管理平台')).toBeInTheDocument()
    })

    // Find the WebSocket instance
    await waitFor(() => {
      mockWebSocket = (global.WebSocket as any).mock.instances[0]
    })

    // Simulate real-time update message
    act(() => {
      mockWebSocket.simulateMessage({
        type: 'realtime_update',
        data: {
          cluster_stats: {
            ...mockClusterStats,
            total_agents: 4,
            running_agents: 3,
          },
        },
      })
    })

    // The component should update with new data
    // Note: This test would need the component to properly handle the WebSocket message
    // and update the displayed data accordingly
  })

  it('refreshes data when refresh button is clicked', async () => {
    const user = userEvent.setup()
    render(<AgentClusterManagementPage />)

    await waitFor(() => {
      expect(screen.getByText('智能体集群管理平台')).toBeInTheDocument()
    })

    // Clear previous fetch calls
    mockFetch.mockClear()

    // Click refresh button
    await user.click(screen.getByText('刷新数据'))

    // Should make new API calls
    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        '/api/v1/cluster/status',
        expect.any(Object)
      )
      expect(mockFetch).toHaveBeenCalledWith(
        '/api/v1/cluster/agents',
        expect.any(Object)
      )
      expect(mockFetch).toHaveBeenCalledWith(
        '/api/v1/cluster/scaling/recommendations',
        expect.any(Object)
      )
    })
  })

  it('displays error message on API failure', async () => {
    // Mock API failure
    mockFetch.mockImplementationOnce(() =>
      Promise.reject(new Error('Network error'))
    )

    render(<AgentClusterManagementPage />)

    await waitFor(() => {
      expect(
        screen.getByText('Failed to load initial data')
      ).toBeInTheDocument()
    })
  })

  it('displays monitoring alerts tab', async () => {
    render(<AgentClusterManagementPage />)

    await waitFor(() => {
      expect(screen.getByText('智能体集群管理平台')).toBeInTheDocument()
    })

    // Switch to monitoring alerts tab
    fireEvent.click(screen.getByText('监控告警'))

    await waitFor(() => {
      expect(screen.getByText('监控告警')).toBeInTheDocument()
      expect(screen.getByText('告警规则')).toBeInTheDocument()
      expect(screen.getByText('活跃告警')).toBeInTheDocument()
      expect(screen.getByText('当前无活跃告警')).toBeInTheDocument()
    })
  })

  it('shows connection status correctly', async () => {
    render(<AgentClusterManagementPage />)

    await waitFor(() => {
      expect(screen.getByText('智能体集群管理平台')).toBeInTheDocument()
    })

    // Should initially show connected status
    await waitFor(() => {
      const connectionChip = screen.getByText('连接状态: connected')
      expect(connectionChip).toBeInTheDocument()
    })
  })

  it('handles tab switching correctly', async () => {
    const user = userEvent.setup()
    render(<AgentClusterManagementPage />)

    await waitFor(() => {
      expect(screen.getByText('智能体集群管理平台')).toBeInTheDocument()
    })

    // Initially should be on overview tab
    expect(screen.getByText('集群状态')).toBeInTheDocument()

    // Switch to agent management
    await user.click(screen.getByText('智能体管理'))
    await waitFor(() => {
      expect(screen.getByText('添加智能体')).toBeInTheDocument()
    })

    // Switch to auto scaling
    await user.click(screen.getByText('自动扩缩容'))
    await waitFor(() => {
      expect(screen.getByText('自动扩缩容')).toBeInTheDocument()
    })

    // Switch to monitoring
    await user.click(screen.getByText('监控告警'))
    await waitFor(() => {
      expect(screen.getByText('告警规则')).toBeInTheDocument()
    })
  })

  it('displays charts correctly', async () => {
    render(<AgentClusterManagementPage />)

    await waitFor(() => {
      expect(screen.getByText('智能体集群管理平台')).toBeInTheDocument()
    })

    // Check if chart title is present
    expect(screen.getByText('实时性能指标')).toBeInTheDocument()

    // The chart itself would be rendered by Recharts,
    // which might be difficult to test without additional setup
  })

  it('handles add agent dialog', async () => {
    const user = userEvent.setup()
    render(<AgentClusterManagementPage />)

    await waitFor(() => {
      expect(screen.getByText('智能体集群管理平台')).toBeInTheDocument()
    })

    fireEvent.click(screen.getByText('智能体管理'))

    await waitFor(() => {
      expect(screen.getByText('添加智能体')).toBeInTheDocument()
    })

    // Open add agent dialog
    await user.click(screen.getByText('添加智能体'))

    await waitFor(() => {
      expect(screen.getByText('智能体创建功能开发中')).toBeInTheDocument()
    })

    // Close dialog
    await user.click(screen.getByText('关闭'))

    await waitFor(() => {
      expect(screen.queryByText('智能体创建功能开发中')).not.toBeInTheDocument()
    })
  })
})
