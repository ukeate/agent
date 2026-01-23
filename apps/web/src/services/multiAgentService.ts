import { apiFetch, buildApiUrl, buildWsUrl } from '../utils/apiBase'
import apiClient from './apiClient'
import type {
  Agent,
  AgentRole,
  ConversationConfig,
  ConversationParticipant,
  ConversationStatus,
  MessageRole,
} from '@/types/multiAgent'
import { logger } from '../utils/logger'
// ==================== 类型定义 ====================

const unwrap = <T>(payload: any): T => {
  if (payload && typeof payload === 'object' && 'success' in payload) {
    if (!payload.success) throw new Error(payload.error || '请求失败')
    return payload.data as T
  }
  return payload as T
}

export interface CreateConversationRequest {
  message: string
  agent_roles?: AgentRole[]
  user_context?: string
  max_rounds?: number
  timeout_seconds?: number
  auto_reply?: boolean
}

export interface ConversationResponse {
  conversation_id: string
  status: ConversationStatus
  participants: ConversationParticipant[]
  created_at: string
  updated_at?: string
  message_count?: number
  round_count?: number
  config: ConversationConfig
  initial_status: Record<string, any>
}

export interface ConversationStatusResponse {
  conversation_id: string
  status: ConversationStatus
  created_at: string
  updated_at: string
  message_count: number
  round_count: number
  participants: ConversationParticipant[]
  config: ConversationConfig
  real_time_stats?: {
    active_agents: number
    pending_messages: number
    avg_response_time: number
  }
}

export interface Message {
  id: string
  conversation_id: string
  sender: string
  role: MessageRole
  content: string
  timestamp: string
  round?: number
  metadata?: {
    agent_id?: string
    agent_role?: AgentRole
    round?: number
    tool_calls?: any[]
    thinking?: string
  }
}

export interface MessagesResponse {
  conversation_id: string
  messages: Message[]
  total_count: number
  returned_count: number
  offset: number
}

export interface StartConversationRequest {
  initial_message?: string
  context?: string
}

export interface TerminateConversationRequest {
  reason?: string
}

export interface WebSocketMessage {
  type: 'message' | 'status' | 'error' | 'agent_thinking' | 'round_complete'
  data: any
  timestamp: string
}

export interface AgentStats {
  agent_id: string
  messages_sent: number
  avg_response_time: number
  error_rate: number
  last_active: string
}

export interface ConversationSummary {
  conversation_id: string
  key_points: string[]
  decisions_made: string[]
  action_items: string[]
  participants_summary: Record<string, string>
}

// ==================== Service Class ====================

class MultiAgentService {
  private baseUrl = '/multi-agent'

  // ==================== 对话管理 ====================

  async createConversation(
    request: CreateConversationRequest
  ): Promise<ConversationResponse> {
    const response = await apiClient.post(
      `${this.baseUrl}/conversation`,
      request
    )
    return response.data
  }

  async getConversationStatus(
    conversationId: string
  ): Promise<ConversationStatusResponse> {
    const response = await apiClient.get(
      `${this.baseUrl}/conversation/${conversationId}/status`
    )
    return response.data
  }

  async listConversations(
    limit?: number,
    offset?: number
  ): Promise<ConversationResponse[]> {
    const params = { limit: limit || 20, offset: offset || 0 }
    const response = await apiClient.get(`${this.baseUrl}/conversations`, {
      params,
    })
    return response.data
  }

  async startConversation(
    conversationId: string,
    request?: StartConversationRequest
  ): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post(
      `${this.baseUrl}/conversation/${conversationId}/start`,
      request || {}
    )
    return response.data
  }

  async pauseConversation(
    conversationId: string
  ): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post(
      `${this.baseUrl}/conversation/${conversationId}/pause`
    )
    return response.data
  }

  async resumeConversation(
    conversationId: string
  ): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post(
      `${this.baseUrl}/conversation/${conversationId}/resume`
    )
    return response.data
  }

  async terminateConversation(
    conversationId: string,
    request?: TerminateConversationRequest
  ): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post(
      `${this.baseUrl}/conversation/${conversationId}/terminate`,
      request || {}
    )
    return response.data
  }

  // ==================== 消息管理 ====================

  async getMessages(
    conversationId: string,
    limit?: number,
    offset?: number
  ): Promise<MessagesResponse> {
    const params = { limit: limit || 100, offset: offset || 0 }
    const response = await apiClient.get(
      `${this.baseUrl}/conversation/${conversationId}/messages`,
      { params }
    )
    return response.data
  }

  async sendMessage(
    conversationId: string,
    message: string,
    metadata?: Record<string, any>
  ): Promise<Message> {
    const response = await apiClient.post(
      `${this.baseUrl}/conversation/${conversationId}/message`,
      { message, metadata }
    )
    return response.data
  }

  async getMessageStream(conversationId: string): Promise<ReadableStream> {
    const response = await apiFetch(
      buildApiUrl(`${this.baseUrl}/conversation/${conversationId}/stream`),
      {
        method: 'GET',
        headers: {
          Accept: 'text/event-stream',
        },
      }
    )

    if (!response.body) {
      throw new Error('Response body is null')
    }

    return response.body
  }

  // ==================== 智能体管理 ====================

  async updateAgentConfig(
    agentId: string,
    config: Record<string, any>
  ): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.put(
      `${this.baseUrl}/agents/${agentId}/config`,
      config
    )
    return response.data
  }

  async getAgentStats(conversationId: string): Promise<AgentStats[]> {
    const response = await apiClient.get(
      `${this.baseUrl}/conversation/${conversationId}/agent-stats`
    )
    return response.data.stats || []
  }

  // ==================== Agent管理 ====================

  async listAgents(): Promise<Agent[]> {
    const response = await apiClient.get(`${this.baseUrl}/agents`)
    return unwrap<{ agents: Agent[] }>(response.data).agents || []
  }

  // ==================== 分析和总结 ====================

  async getConversationSummary(
    conversationId: string
  ): Promise<ConversationSummary> {
    const response = await apiClient.get(
      `${this.baseUrl}/conversation/${conversationId}/summary`
    )
    return response.data
  }

  async exportConversation(
    conversationId: string,
    format: 'json' | 'markdown' | 'pdf' = 'json'
  ): Promise<Blob> {
    const response = await apiClient.get(
      `${this.baseUrl}/conversation/${conversationId}/export`,
      {
        params: { format },
        responseType: 'blob',
      }
    )
    return response.data
  }

  async analyzeConversation(conversationId: string): Promise<{
    sentiment_analysis: Record<string, number>
    topic_distribution: Record<string, number>
    interaction_patterns: any[]
    recommendations: string[]
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/conversation/${conversationId}/analyze`
    )
    return response.data
  }

  // ==================== WebSocket连接 ====================

  connectWebSocket(conversationId: string): WebSocket {
    const wsUrl = buildWsUrl(`/multi-agent/ws/${conversationId}`)
    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      logger.log('对话WebSocket已连接:', conversationId)
    }

    ws.onerror = error => {
      logger.error('WebSocket错误:', error)
      if (
        ws.readyState !== WebSocket.CLOSING &&
        ws.readyState !== WebSocket.CLOSED
      ) {
        ws.close()
      }
    }

    ws.onclose = () => {
      logger.log('WebSocket连接已断开')
    }

    return ws
  }

  // ==================== 批量操作 ====================

  async batchTerminateConversations(
    conversationIds: string[],
    reason?: string
  ): Promise<{ success: number; failed: number; errors: string[] }> {
    const response = await apiClient.post(
      `${this.baseUrl}/conversations/batch/terminate`,
      {
        conversation_ids: conversationIds,
        reason,
      }
    )
    return response.data
  }

  async cleanupOldConversations(
    olderThanDays: number
  ): Promise<{ deleted: number; message: string }> {
    const response = await apiClient.post(
      `${this.baseUrl}/conversations/cleanup`,
      {
        older_than_days: olderThanDays,
      }
    )
    return response.data
  }

  // ==================== 统计和监控 ====================

  /**
   * 获取多智能体系统状态（匹配后端API）
   */
  async getStatus(): Promise<{
    active_agents: number
    total_agents: number
    system_load: number
    message: string
    agents: Array<{
      id: string
      type: string
      status: string
      tasks: number
    }>
  }> {
    const response = await apiClient.get(`${this.baseUrl}/status`)
    return response.data
  }

  async getSystemStats(): Promise<{
    total_conversations: number
    active_conversations: number
    total_messages: number
    avg_conversation_duration: number
    agent_utilization: Record<string, number>
  }> {
    const response = await apiClient.get(`${this.baseUrl}/stats`)
    return response.data
  }

  async getHealthStatus(): Promise<{
    status: 'healthy' | 'degraded' | 'unhealthy'
    agents_online: number
    agents_total: number
    issues?: string[]
  }> {
    const response = await apiClient.get(`${this.baseUrl}/health`)
    return response.data
  }
}

// ==================== 导出 ====================

export const multiAgentService = new MultiAgentService()
export default multiAgentService
