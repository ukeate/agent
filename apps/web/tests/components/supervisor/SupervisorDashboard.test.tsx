/**
 * SupervisorDashboard组件测试
 */

import { render, screen } from '@testing-library/react'
import { SupervisorDashboard } from '../../../src/components/supervisor'
import { describe, it, expect, vi, beforeEach } from 'vitest'

// Mock Zustand store
const mockStore = {
  currentSupervisorId: null,
  status: null,
  tasks: [],
  decisions: [],
  stats: null,
  config: null,
  agentMetrics: [],
  loading: {
    status: false,
    tasks: false,
    decisions: false,
    stats: false,
    config: false,
    metrics: false,
    submitting: false,
  },
  error: null,
  pagination: {
    tasks: {
      page: 1,
      pageSize: 20,
      total: 0,
      totalPages: 0,
    },
    decisions: {
      page: 1,
      pageSize: 20,
      total: 0,
      totalPages: 0,
    },
  },
  autoRefresh: false,
  refreshInterval: 10000,
  setSupervisorId: mockFn()fn(),
  loadStatus: mockFn()fn(),
  loadTasks: mockFn()fn(),
  loadDecisions: mockFn()fn(),
  loadStats: mockFn()fn(),
  loadConfig: mockFn()fn(),
  loadMetrics: mockFn()fn(),
  refreshAll: mockFn()fn(),
  setAutoRefresh: mockFn()fn(),
  setRefreshInterval: mockFn()fn(),
  clearError: mockFn()fn(),
  submitTask: mockFn()fn(),
  updateConfig: mockFn()fn(),
  reset: mockFn()fn(),
}

// Mock the store
mockFn()mock('../../../src/stores/supervisorStore', () => ({
  useSupervisorStore: mockFn()fn(() => mockStore)
}))

// Mock子组件
mockFn()mock('../../../src/components/supervisor/SupervisorStatus', () => ({
  SupervisorStatus: () => <div>SupervisorStatus Mock</div>
}))
mockFn()mock('../../../src/components/supervisor/TaskList', () => ({
  TaskList: () => <div>TaskList Mock</div>
}))
mockFn()mock('../../../src/components/supervisor/DecisionHistory', () => ({
  DecisionHistory: () => <div>DecisionHistory Mock</div>
}))
mockFn()mock('../../../src/components/supervisor/AgentMetrics', () => ({
  AgentMetrics: () => <div>AgentMetrics Mock</div>
}))
mockFn()mock('../../../src/components/supervisor/SupervisorConfig', () => ({
  SupervisorConfig: () => <div>SupervisorConfig Mock</div>
}))
mockFn()mock('../../../src/components/supervisor/TaskSubmissionForm', () => ({
  TaskSubmissionForm: () => <div>TaskSubmissionForm Mock</div>
}))

describe('SupervisorDashboard', () => {
  beforeEach(() => {
    mockFn()clearAllMocks()
  })

  it('应该渲染Supervisor监控面板标题', () => {
    render(<SupervisorDashboard supervisorId="test_supervisor" />)
    
    expect(screen.getByText('Supervisor 监控面板')).toBeInTheDocument()
  })

  it('应该显示Supervisor ID', () => {
    render(<SupervisorDashboard supervisorId="test_supervisor" />)
    
    // supervisor ID应该显示
    expect(screen.getByText('Supervisor ID:')).toBeInTheDocument()
  })

  it('应该包含标签页导航', () => {
    render(<SupervisorDashboard supervisorId="test_supervisor" />)
    
    expect(screen.getByText('概览')).toBeInTheDocument()
    expect(screen.getByText('任务')).toBeInTheDocument()
    expect(screen.getByText('决策')).toBeInTheDocument()
    expect(screen.getByText('指标')).toBeInTheDocument()
    expect(screen.getByText('配置')).toBeInTheDocument()
  })

  it('应该包含操作按钮', () => {
    render(<SupervisorDashboard supervisorId="test_supervisor" />)
    
    expect(screen.getByText('手动刷新')).toBeInTheDocument()
    expect(screen.getByText('提交任务')).toBeInTheDocument()
  })

  it('应该显示自动刷新控制', () => {
    render(<SupervisorDashboard supervisorId="test_supervisor" />)
    
    expect(screen.getByText('自动刷新')).toBeInTheDocument()
    // 检查 select 中是否有10秒选项
    expect(screen.getByText('10秒')).toBeInTheDocument()
  })
})