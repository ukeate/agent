import apiClient from './apiClient';
import { apiFetch, buildApiUrl } from '../utils/apiBase';
import { consumeSseJson } from '../utils/sse';

import { logger } from '../utils/logger'
// ==================== 类型定义 ====================

export interface ChatRequest {
  message: string;
  conversation_id?: string;
  context?: Record<string, any>;
  stream?: boolean;
  max_tokens?: number;
  temperature?: number;
  tools?: ToolDefinition[];
}

export interface ChatResponse {
  message_id: string;
  conversation_id: string;
  content: string;
  role: 'assistant';
  tool_calls?: ToolCall[];
  metadata?: {
    model?: string;
    tokens_used?: number;
    processing_time?: number;
  };
  timestamp: string;
}

export interface ToolDefinition {
  name: string;
  description: string;
  parameters: Record<string, any>;
}

export interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, any>;
  result?: any;
}

export interface TaskRequest {
  task_type: string;
  description: string;
  parameters?: Record<string, any>;
  priority?: 'low' | 'medium' | 'high' | 'urgent';
  timeout_seconds?: number;
  callback_url?: string;
}

export interface TaskResponse {
  task_id: string;
  status: TaskStatus;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  result?: TaskResult;
  error?: string;
  progress?: number;
}

export type TaskStatus = 
  | 'pending'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'timeout';

export interface TaskResult {
  output: any;
  metrics?: Record<string, any>;
  artifacts?: string[];
}

export interface AgentStatus {
  agent_id: string;
  status: 'online' | 'offline' | 'busy' | 'error';
  health: AgentHealth;
  info: AgentInfo;
  resources: SystemResource;
  performance: PerformanceMetrics;
}

export interface AgentHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  checks: {
    name: string;
    status: 'pass' | 'fail';
    message?: string;
  }[];
  last_check: string;
}

export interface AgentInfo {
  version: string;
  capabilities: string[];
  models: string[];
  tools: string[];
  uptime_seconds: number;
}

export interface SystemResource {
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  gpu_usage?: number;
}

export interface PerformanceMetrics {
  requests_per_second: number;
  average_latency_ms: number;
  error_rate: number;
  success_rate: number;
}

export interface ConversationHistory {
  conversation_id: string;
  messages: Array<{
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp: string;
    metadata?: Record<string, any>;
  }>;
  created_at: string;
  updated_at: string;
}

export interface AgentSession {
  session_id: string;
  conversation_id: string;
  agent_type: string;
  created_at: string;
  expires_at: string;
  metadata?: Record<string, any>;
}

// ==================== Service Class ====================

class AgentInterfaceService {
  private baseUrl = '/agent';

  // ==================== 对话管理 ====================

  async chat(request: ChatRequest): Promise<ChatResponse> {
    const response = await apiClient.post(`${this.baseUrl}/chat`, request);
    return response.data.data;
  }

  async streamChat(
    request: ChatRequest,
    onMessage: (chunk: string) => void,
    onError?: (error: any) => void,
    onComplete?: () => void
  ): Promise<void> {
    try {
      const authHeader = apiClient.defaults.headers.common['Authorization'];
      const headers: HeadersInit = {
        'Content-Type': 'application/json',
        'Accept': 'text/event-stream',
      };
      if (authHeader) headers['Authorization'] = String(authHeader);

      const response = await apiFetch(buildApiUrl(`${this.baseUrl}/chat/stream`), {
        method: 'POST',
        headers,
        body: JSON.stringify({ ...request, stream: true })
      });

      await consumeSseJson(
        response,
        (payload: any) => {
          if (payload?.error) {
            onError?.(payload.error);
            return;
          }
          const delta = payload?.choices?.[0]?.delta?.content;
          if (delta) {
            onMessage(delta);
          }
        },
        {
          onDone: onComplete,
          onParseError: (error) => {
            logger.error('解析SSE数据失败:', error);
          },
        }
      );
    } catch (error) {
      onError?.(error);
      throw error;
    }
  }

  // ==================== 任务管理 ====================

  async submitTask(request: TaskRequest): Promise<TaskResponse> {
    const response = await apiClient.post(`${this.baseUrl}/task`, request);
    return response.data.data;
  }

  async getTaskStatus(taskId: string): Promise<TaskResponse> {
    const response = await apiClient.get(`${this.baseUrl}/task/${taskId}`);
    return response.data.data;
  }

  async cancelTask(taskId: string): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.post(`${this.baseUrl}/task/${taskId}/cancel`);
    return response.data;
  }

  async listTasks(params?: {
    status?: TaskStatus;
    limit?: number;
    offset?: number;
  }): Promise<{
    tasks: TaskResponse[];
    total: number;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/tasks`, { params });
    return response.data;
  }

  // ==================== 状态监控 ====================

  async getStatus(): Promise<AgentStatus> {
    const response = await apiClient.get(`${this.baseUrl}/status`);
    return (response.data?.data ?? response.data) as AgentStatus;
  }

  async getHealth(): Promise<AgentHealth> {
    const response = await apiClient.get(`${this.baseUrl}/health`);
    return response.data;
  }

  async getMetrics(): Promise<PerformanceMetrics> {
    const response = await apiClient.get(`${this.baseUrl}/metrics`);
    return (response.data?.data ?? response.data) as PerformanceMetrics;
  }

  // ==================== 会话管理 ====================

  async createSession(params?: {
    agent_type?: string;
    config?: Record<string, any>;
  }): Promise<AgentSession> {
    const response = await apiClient.post(`${this.baseUrl}/session`, params || {});
    return response.data.data;
  }

  async getSession(sessionId: string): Promise<AgentSession> {
    const response = await apiClient.get(`${this.baseUrl}/session/${sessionId}`);
    return response.data.data;
  }

  async deleteSession(sessionId: string): Promise<{ success: boolean }> {
    const response = await apiClient.delete(`${this.baseUrl}/session/${sessionId}`);
    return response.data;
  }

  async getConversationHistory(conversationId: string): Promise<ConversationHistory> {
    const response = await apiClient.get(`${this.baseUrl}/conversation/${conversationId}`);
    return response.data.data;
  }

  // ==================== 工具管理 ====================

  async listAvailableTools(): Promise<ToolDefinition[]> {
    const response = await apiClient.get(`${this.baseUrl}/tools`);
    return response.data.data;
  }

  async executeTool(toolName: string, args: Record<string, any>): Promise<any> {
    const response = await apiClient.post(`${this.baseUrl}/tools/${toolName}/execute`, {
      arguments: args
    });
    return response.data.data;
  }

  // ==================== 配置管理 ====================

  async getConfiguration(): Promise<{
    models: string[];
    default_model: string;
    max_tokens: number;
    temperature: number;
    tools_enabled: boolean;
    streaming_enabled: boolean;
  }> {
    const response = await apiClient.get(`${this.baseUrl}/config`);
    return response.data;
  }

  async updateConfiguration(config: Partial<{
    default_model?: string;
    max_tokens?: number;
    temperature?: number;
    tools_enabled?: boolean;
    streaming_enabled?: boolean;
  }>): Promise<{ success: boolean; message: string }> {
    const response = await apiClient.put(`${this.baseUrl}/config`, config);
    return response.data;
  }

  // ==================== 批处理 ====================

  async batchChat(requests: ChatRequest[]): Promise<ChatResponse[]> {
    const response = await apiClient.post(`${this.baseUrl}/batch/chat`, {
      requests
    });
    return response.data.data;
  }

  async batchTasks(requests: TaskRequest[]): Promise<TaskResponse[]> {
    const response = await apiClient.post(`${this.baseUrl}/batch/tasks`, {
      requests
    });
    return response.data.data;
  }

  // ==================== WebSocket连接 ====================

  connectAgentStream(
    onMessage: (data: any) => void,
    onError?: (error: any) => void,
    onClose?: () => void
  ): () => void {
    const wsUrl = buildWsUrl(`${this.baseUrl}/stream`);
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      logger.log('智能体流连接已建立');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (error) {
        logger.error('解析智能体消息失败:', error);
        onError?.(error);
      }
    };

    ws.onerror = (error) => {
      logger.error('智能体流连接错误:', error);
      onError?.(error);
      if (ws.readyState !== WebSocket.CLOSING && ws.readyState !== WebSocket.CLOSED) {
        ws.close();
      }
    };

    ws.onclose = () => {
      logger.log('智能体流连接已断开');
      onClose?.();
    };

    // 返回断开连接函数
    return () => {
      ws.close();
    };
  }
}

// ==================== 导出 ====================

export const agentInterfaceService = new AgentInterfaceService();
export default agentInterfaceService;
