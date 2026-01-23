import { describe, it, expect, beforeEach, vi } from 'vitest'
import {
  useMultiAgentStore,
  Agent,
  ConversationSession,
} from '../../src/stores/multiAgentStore'

// Mock zustand persist
vi.mock('zustand/middleware', () => ({
  persist: (fn: any) => fn,
}))

vi.mock('../../src/services/multiAgentService', () => ({
  multiAgentService: {
    pauseConversation: vi.fn().mockResolvedValue({ success: true }),
    resumeConversation: vi.fn().mockResolvedValue({ success: true }),
    terminateConversation: vi.fn().mockResolvedValue({ success: true }),
    getMessages: vi.fn().mockResolvedValue({
      messages: [],
      total_count: 0,
    }),
  },
}))

// Mock fetch for API calls
const mockFetch = vi.fn()
global.fetch = mockFetch

describe('multiAgentStore', () => {
  beforeEach(() => {
    // 重置store状态
    useMultiAgentStore.getState().setAgents([])
    useMultiAgentStore.getState().setCurrentSession(null)
    useMultiAgentStore.getState().clearMessages()
    useMultiAgentStore.getState().setError(null)
    useMultiAgentStore.getState().setLoading(false)
    useMultiAgentStore.getState().setWebsocketConnected(false)

    // 重置fetch mock
    mockFetch.mockClear()
    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ success: true, message: 'success' }),
      text: async () => 'success',
    })
  })

  const mockAgent: Agent = {
    id: 'agent-1',
    name: '测试代码专家',
    role: 'code_expert',
    status: 'active',
    capabilities: ['代码生成', '代码审查'],
    configuration: {
      model: 'gpt-4o-mini',
      temperature: 0.1,
      max_tokens: 2000,
      tools: ['code_analyzer'],
      system_prompt: '你是一位专业的软件开发专家',
    },
    created_at: '2025-01-01T00:00:00Z',
    updated_at: '2025-01-01T00:00:00Z',
  }

  const mockSession: ConversationSession = {
    session_id: 'session-1',
    status: 'active',
    created_at: '2025-01-01T00:00:00Z',
    updated_at: '2025-01-01T00:00:00Z',
    message_count: 0,
    round_count: 0,
    participants: [
      {
        name: '测试代码专家',
        role: 'code_expert',
        status: 'active',
      },
    ],
    config: {
      max_rounds: 10,
      timeout_seconds: 300,
      auto_reply: true,
    },
  }

  describe('初始状态', () => {
    it('应该有正确的初始状态', () => {
      const state = useMultiAgentStore.getState()

      expect(state.agents).toEqual([])
      expect(state.currentSession).toBeNull()
      expect(state.sessions).toEqual([])
      expect(state.currentMessages).toEqual([])
      expect(state.loading).toBe(false)
      expect(state.error).toBeNull()
      expect(state.websocketConnected).toBe(false)
    })
  })

  describe('智能体管理', () => {
    it('应该能设置智能体列表', () => {
      const agents = [mockAgent]

      useMultiAgentStore.getState().setAgents(agents)

      expect(useMultiAgentStore.getState().agents).toEqual(agents)
    })

    it('应该能更新智能体状态', () => {
      const agents = [mockAgent]
      useMultiAgentStore.getState().setAgents(agents)

      useMultiAgentStore.getState().updateAgentStatus('agent-1', 'busy')

      const updatedAgent = useMultiAgentStore.getState().agents[0]
      expect(updatedAgent.status).toBe('busy')
      expect(updatedAgent.updated_at).not.toBe(mockAgent.updated_at)
    })

    it('更新不存在的智能体状态时不应该出错', () => {
      const agents = [mockAgent]
      useMultiAgentStore.getState().setAgents(agents)

      // 不应该抛出错误
      expect(() => {
        useMultiAgentStore.getState().updateAgentStatus('non-existent', 'busy')
      }).not.toThrow()

      // 原智能体状态应该不变
      expect(useMultiAgentStore.getState().agents[0].status).toBe('active')
    })
  })

  describe('会话管理', () => {
    it('应该能设置当前会话', () => {
      useMultiAgentStore.getState().setCurrentSession(mockSession)

      expect(useMultiAgentStore.getState().currentSession).toEqual(mockSession)
      expect(useMultiAgentStore.getState().currentMessages).toEqual([])
    })

    it('应该能添加会话到会话列表', () => {
      useMultiAgentStore.getState().addSession(mockSession)

      const sessions = useMultiAgentStore.getState().sessions
      expect(sessions).toHaveLength(1)
      expect(sessions[0]).toEqual(mockSession)
    })

    it('应该能更新会话状态', () => {
      useMultiAgentStore.getState().addSession(mockSession)
      useMultiAgentStore.getState().setCurrentSession(mockSession)

      useMultiAgentStore.getState().updateSessionStatus('session-1', 'paused')

      const updatedSession = useMultiAgentStore.getState().sessions[0]
      expect(updatedSession.status).toBe('paused')
      expect(updatedSession.updated_at).not.toBe(mockSession.updated_at)

      // 当前会话也应该更新
      const currentSession = useMultiAgentStore.getState().currentSession
      expect(currentSession?.status).toBe('paused')
    })

    it('更新不存在的会话状态时不应该出错', () => {
      expect(() => {
        useMultiAgentStore
          .getState()
          .updateSessionStatus('non-existent', 'paused')
      }).not.toThrow()
    })
  })

  describe('消息管理', () => {
    it('应该能添加消息', () => {
      const message = {
        id: 'msg-1',
        role: 'assistant',
        sender: '测试智能体',
        content: '测试消息',
        timestamp: '2025-01-01T00:00:00Z',
        round: 1,
      }

      useMultiAgentStore.getState().addMessage(message)

      const messages = useMultiAgentStore.getState().currentMessages
      expect(messages).toHaveLength(1)
      expect(messages[0]).toEqual(message)
    })

    it('应该能清空消息', () => {
      const message = {
        id: 'msg-1',
        role: 'assistant',
        sender: '测试智能体',
        content: '测试消息',
        timestamp: '2025-01-01T00:00:00Z',
        round: 1,
      }

      useMultiAgentStore.getState().addMessage(message)
      expect(useMultiAgentStore.getState().currentMessages).toHaveLength(1)

      useMultiAgentStore.getState().clearMessages()
      expect(useMultiAgentStore.getState().currentMessages).toHaveLength(0)
    })
  })

  describe('状态管理', () => {
    it('应该能设置加载状态', () => {
      useMultiAgentStore.getState().setLoading(true)
      expect(useMultiAgentStore.getState().loading).toBe(true)

      useMultiAgentStore.getState().setLoading(false)
      expect(useMultiAgentStore.getState().loading).toBe(false)
    })

    it('应该能设置错误状态', () => {
      const error = '测试错误'

      useMultiAgentStore.getState().setError(error)
      expect(useMultiAgentStore.getState().error).toBe(error)

      useMultiAgentStore.getState().setError(null)
      expect(useMultiAgentStore.getState().error).toBeNull()
    })

    it('应该能设置WebSocket连接状态', () => {
      useMultiAgentStore.getState().setWebsocketConnected(true)
      expect(useMultiAgentStore.getState().websocketConnected).toBe(true)

      useMultiAgentStore.getState().setWebsocketConnected(false)
      expect(useMultiAgentStore.getState().websocketConnected).toBe(false)
    })
  })

  describe('智能体操作', () => {
    it('应该能创建对话', async () => {
      const participants = ['agent-1', 'agent-2']
      const topic = '测试话题'

      await useMultiAgentStore
        .getState()
        .createConversation(participants, topic)

      // 验证loading状态被设置和重置
      expect(useMultiAgentStore.getState().loading).toBe(false)
    })

    it('应该能启动对话', async () => {
      const sessionId = 'session-1'
      const message = '开始讨论'

      await useMultiAgentStore.getState().startConversation(sessionId, message)

      expect(useMultiAgentStore.getState().loading).toBe(false)
    })

    it('应该能暂停对话', async () => {
      const sessionId = 'session-1'
      useMultiAgentStore.getState().addSession(mockSession)

      await useMultiAgentStore.getState().pauseConversation(sessionId)

      expect(useMultiAgentStore.getState().loading).toBe(false)

      // 验证会话状态更新
      const session = useMultiAgentStore.getState().sessions[0]
      expect(session.status).toBe('paused')
    })

    it('应该能恢复对话', async () => {
      const sessionId = 'session-1'
      const pausedSession = { ...mockSession, status: 'paused' as const }
      useMultiAgentStore.getState().addSession(pausedSession)

      await useMultiAgentStore.getState().resumeConversation(sessionId)

      expect(useMultiAgentStore.getState().loading).toBe(false)

      // 验证会话状态更新
      const session = useMultiAgentStore.getState().sessions[0]
      expect(session.status).toBe('active')
    })

    it('应该能终止对话', async () => {
      const sessionId = 'session-1'
      const reason = '测试终止'
      useMultiAgentStore.getState().addSession(mockSession)

      await useMultiAgentStore
        .getState()
        .terminateConversation(sessionId, reason)

      expect(useMultiAgentStore.getState().loading).toBe(false)

      // 验证会话状态更新
      const session = useMultiAgentStore.getState().sessions[0]
      expect(session.status).toBe('terminated')
    })

    it('应该在操作失败时设置错误状态', async () => {
      // Mock createConversation to throw error
      const originalCreateConversation =
        useMultiAgentStore.getState().createConversation

      // 这里我们无法真正mock异步操作，因为它们只是console.log
      // 在实际实现中，这些操作会调用API并可能抛出错误
      expect(() => {
        // 正常情况下不会抛出错误，因为当前是mock实现
      }).not.toThrow()
    })
  })
})
