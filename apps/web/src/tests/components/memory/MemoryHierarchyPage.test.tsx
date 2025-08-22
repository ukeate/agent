/**
 * 记忆层级页面组件测试
 */
import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import '@testing-library/jest-dom'
import MemoryHierarchyPage from '@/pages/MemoryHierarchyPage'
import { memoryService } from '@/services/memoryService'
import { MemoryType, MemoryStatus } from '@/types/memory'

// Mock memory service
jest.mock('@/services/memoryService', () => ({
  memoryService: {
    getSessionMemories: jest.fn(),
    getMemoryAnalytics: jest.fn(),
    updateMemory: jest.fn(),
    consolidateMemories: jest.fn(),
  }
}))

// Mock localStorage
const mockLocalStorage = {
  getItem: jest.fn(),
  setItem: jest.fn(),
}
Object.defineProperty(window, 'localStorage', {
  value: mockLocalStorage,
})

const mockMemories = {
  working: [
    {
      id: 'working_1',
      type: MemoryType.WORKING,
      content: '当前正在处理的任务信息',
      importance: 0.6,
      access_count: 5,
      created_at: '2025-01-15T10:00:00Z',
      last_accessed: '2025-01-15T12:00:00Z',
      status: MemoryStatus.ACTIVE,
      tags: ['task', 'current'],
      session_id: 'test_session'
    },
    {
      id: 'working_2', 
      type: MemoryType.WORKING,
      content: '用户询问的技术问题',
      importance: 0.7,
      access_count: 3,
      created_at: '2025-01-15T11:00:00Z',
      last_accessed: '2025-01-15T11:30:00Z',
      status: MemoryStatus.ACTIVE,
      tags: ['question', 'tech'],
      session_id: 'test_session'
    }
  ],
  episodic: [
    {
      id: 'episodic_1',
      type: MemoryType.EPISODIC,
      content: '用户成功解决了React hooks的问题',
      importance: 0.8,
      access_count: 8,
      created_at: '2025-01-14T15:00:00Z',
      last_accessed: '2025-01-15T09:00:00Z',
      status: MemoryStatus.ACTIVE,
      tags: ['success', 'react', 'hooks'],
      session_id: 'test_session'
    }
  ],
  semantic: [
    {
      id: 'semantic_1',
      type: MemoryType.SEMANTIC,
      content: 'React Hooks是用于在函数组件中使用状态和其他React特性的函数',
      importance: 0.9,
      access_count: 15,
      created_at: '2025-01-10T10:00:00Z',
      last_accessed: '2025-01-15T10:00:00Z',
      status: MemoryStatus.ACTIVE,
      tags: ['react', 'hooks', 'knowledge'],
      session_id: 'test_session'
    }
  ]
}

const mockAnalytics = {
  total_memories: 4,
  memories_by_type: {
    working: 2,
    episodic: 1,
    semantic: 1
  },
  memories_by_status: {
    active: 4
  },
  avg_importance: 0.75,
  total_access_count: 31,
  avg_access_count: 7.75,
  most_accessed_memories: [],
  recent_memories: [],
  memory_growth_rate: 0.5,
  storage_usage_mb: 2.5
}

const renderWithRouter = (component: React.ReactElement) => {
  return render(
    <BrowserRouter>
      {component}
    </BrowserRouter>
  )
}

describe('MemoryHierarchyPage', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    mockLocalStorage.getItem.mockReturnValue('test_session')
  })

  it('应该正确渲染页面标题和描述', async () => {
    // Mock API responses
    ;(memoryService.getSessionMemories as jest.Mock)
      .mockResolvedValueOnce(mockMemories.working)
      .mockResolvedValueOnce(mockMemories.episodic)
      .mockResolvedValueOnce(mockMemories.semantic)
    ;(memoryService.getMemoryAnalytics as jest.Mock).mockResolvedValue(mockAnalytics)

    renderWithRouter(<MemoryHierarchyPage />)

    expect(screen.getByText('记忆层级系统')).toBeInTheDocument()
    expect(screen.getByText(/可视化展示三层记忆架构/)).toBeInTheDocument()
  })

  it('应该显示统计概览卡片', async () => {
    ;(memoryService.getSessionMemories as jest.Mock)
      .mockResolvedValueOnce(mockMemories.working)
      .mockResolvedValueOnce(mockMemories.episodic)
      .mockResolvedValueOnce(mockMemories.semantic)
    ;(memoryService.getMemoryAnalytics as jest.Mock).mockResolvedValue(mockAnalytics)

    renderWithRouter(<MemoryHierarchyPage />)

    await waitFor(() => {
      expect(screen.getByText('总记忆数')).toBeInTheDocument()
      expect(screen.getByText('平均重要性')).toBeInTheDocument()
      expect(screen.getByText('记忆增长率')).toBeInTheDocument()
      expect(screen.getByText('存储使用')).toBeInTheDocument()
    })
  })

  it('应该显示三层记忆架构', async () => {
    ;(memoryService.getSessionMemories as jest.Mock)
      .mockResolvedValueOnce(mockMemories.working)
      .mockResolvedValueOnce(mockMemories.episodic)
      .mockResolvedValueOnce(mockMemories.semantic)
    ;(memoryService.getMemoryAnalytics as jest.Mock).mockResolvedValue(mockAnalytics)

    renderWithRouter(<MemoryHierarchyPage />)

    await waitFor(() => {
      expect(screen.getByText('工作记忆')).toBeInTheDocument()
      expect(screen.getByText('情景记忆')).toBeInTheDocument()
      expect(screen.getByText('语义记忆')).toBeInTheDocument()
    })
  })

  it('应该正确显示记忆内容', async () => {
    ;(memoryService.getSessionMemories as jest.Mock)
      .mockResolvedValueOnce(mockMemories.working)
      .mockResolvedValueOnce(mockMemories.episodic)
      .mockResolvedValueOnce(mockMemories.semantic)
    ;(memoryService.getMemoryAnalytics as jest.Mock).mockResolvedValue(mockAnalytics)

    renderWithRouter(<MemoryHierarchyPage />)

    await waitFor(() => {
      expect(screen.getByText(/当前正在处理的任务信息/)).toBeInTheDocument()
      expect(screen.getByText(/用户成功解决了React hooks的问题/)).toBeInTheDocument()
      expect(screen.getByText(/React Hooks是用于在函数组件中使用状态/)).toBeInTheDocument()
    })
  })

  it('应该显示记忆标签和重要性', async () => {
    ;(memoryService.getSessionMemories as jest.Mock)
      .mockResolvedValueOnce(mockMemories.working)
      .mockResolvedValueOnce(mockMemories.episodic)
      .mockResolvedValueOnce(mockMemories.semantic)
    ;(memoryService.getMemoryAnalytics as jest.Mock).mockResolvedValue(mockAnalytics)

    renderWithRouter(<MemoryHierarchyPage />)

    await waitFor(() => {
      expect(screen.getByText('工作')).toBeInTheDocument()
      expect(screen.getByText('情景')).toBeInTheDocument()
      expect(screen.getByText('语义')).toBeInTheDocument()
      expect(screen.getByText('task')).toBeInTheDocument()
      expect(screen.getByText('react')).toBeInTheDocument()
    })
  })

  it('应该支持记忆巩固操作', async () => {
    ;(memoryService.getSessionMemories as jest.Mock)
      .mockResolvedValueOnce(mockMemories.working)
      .mockResolvedValueOnce(mockMemories.episodic)
      .mockResolvedValueOnce(mockMemories.semantic)
    ;(memoryService.getMemoryAnalytics as jest.Mock).mockResolvedValue(mockAnalytics)
    ;(memoryService.consolidateMemories as jest.Mock).mockResolvedValue(undefined)

    renderWithRouter(<MemoryHierarchyPage />)

    await waitFor(() => {
      const consolidateButton = screen.getByText('巩固记忆')
      expect(consolidateButton).toBeInTheDocument()
    })

    const consolidateButton = screen.getByText('巩固记忆')
    fireEvent.click(consolidateButton)

    await waitFor(() => {
      expect(memoryService.consolidateMemories).toHaveBeenCalledWith('test_session')
    })
  })

  it('应该支持记忆提升操作', async () => {
    ;(memoryService.getSessionMemories as jest.Mock)
      .mockResolvedValueOnce(mockMemories.working)
      .mockResolvedValueOnce(mockMemories.episodic)
      .mockResolvedValueOnce(mockMemories.semantic)
    ;(memoryService.getMemoryAnalytics as jest.Mock).mockResolvedValue(mockAnalytics)
    ;(memoryService.updateMemory as jest.Mock).mockResolvedValue({})

    renderWithRouter(<MemoryHierarchyPage />)

    await waitFor(() => {
      const promoteButtons = screen.getAllByTitle('提升记忆层级')
      expect(promoteButtons.length).toBeGreaterThan(0)
    })

    // 点击第一个提升按钮
    const promoteButtons = screen.getAllByTitle('提升记忆层级')
    fireEvent.click(promoteButtons[0])

    await waitFor(() => {
      expect(memoryService.updateMemory).toHaveBeenCalled()
    })
  })

  it('应该正确显示容量信息', async () => {
    ;(memoryService.getSessionMemories as jest.Mock)
      .mockResolvedValueOnce(mockMemories.working)
      .mockResolvedValueOnce(mockMemories.episodic)
      .mockResolvedValueOnce(mockMemories.semantic)
    ;(memoryService.getMemoryAnalytics as jest.Mock).mockResolvedValue(mockAnalytics)

    renderWithRouter(<MemoryHierarchyPage />)

    await waitFor(() => {
      expect(screen.getByText(/容量: 2 \/ 100/)).toBeInTheDocument()  // 工作记忆
      expect(screen.getByText(/容量: 1 \/ 10000/)).toBeInTheDocument()  // 情景记忆
      expect(screen.getByText(/容量: 1 \/ 5000/)).toBeInTheDocument()   // 语义记忆
    })
  })

  it('应该显示记忆流转示意图', async () => {
    ;(memoryService.getSessionMemories as jest.Mock)
      .mockResolvedValueOnce(mockMemories.working)
      .mockResolvedValueOnce(mockMemories.episodic)
      .mockResolvedValueOnce(mockMemories.semantic)
    ;(memoryService.getMemoryAnalytics as jest.Mock).mockResolvedValue(mockAnalytics)

    renderWithRouter(<MemoryHierarchyPage />)

    await waitFor(() => {
      expect(screen.getByText('记忆通过重要性评估和访问频率，逐级提升到更持久的存储层')).toBeInTheDocument()
    })
  })

  it('应该处理API错误', async () => {
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {})
    
    ;(memoryService.getSessionMemories as jest.Mock).mockRejectedValue(new Error('API Error'))
    ;(memoryService.getMemoryAnalytics as jest.Mock).mockRejectedValue(new Error('API Error'))

    renderWithRouter(<MemoryHierarchyPage />)

    await waitFor(() => {
      expect(consoleSpy).toHaveBeenCalledWith('加载记忆失败:', expect.any(Error))
    })

    consoleSpy.mockRestore()
  })

  it('应该支持刷新操作', async () => {
    ;(memoryService.getSessionMemories as jest.Mock)
      .mockResolvedValueOnce(mockMemories.working)
      .mockResolvedValueOnce(mockMemories.episodic)
      .mockResolvedValueOnce(mockMemories.semantic)
    ;(memoryService.getMemoryAnalytics as jest.Mock).mockResolvedValue(mockAnalytics)

    renderWithRouter(<MemoryHierarchyPage />)

    await waitFor(() => {
      const refreshButton = screen.getByText('刷新')
      expect(refreshButton).toBeInTheDocument()
    })

    // 清除之前的调用
    jest.clearAllMocks()

    const refreshButton = screen.getByText('刷新')
    fireEvent.click(refreshButton)

    await waitFor(() => {
      expect(memoryService.getSessionMemories).toHaveBeenCalledTimes(3) // 三层记忆
    })
  })
})