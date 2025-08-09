/**
 * 智能体API客户端
 */

import { ApiClient, type ApiClientConfig } from './base';
import type { Agent, AgentConfig, Session, Message, Task } from '../types/agent';
import type { ApiResponse, PaginatedResponse } from '../types/api';
import { API_ENDPOINTS } from '../constants/api';

export class AgentApiClient extends ApiClient {
  constructor(config: ApiClientConfig) {
    super(config);
  }

  // 智能体管理
  async getAgents(): Promise<ApiResponse<Agent[]>> {
    return this.get<Agent[]>(API_ENDPOINTS.AGENTS.LIST);
  }

  async getAgent(id: string): Promise<ApiResponse<Agent>> {
    const endpoint = API_ENDPOINTS.AGENTS.DETAIL.replace(':id', id);
    return this.get<Agent>(endpoint);
  }

  async createAgent(config: Omit<AgentConfig, 'id' | 'created_at' | 'updated_at'>): Promise<ApiResponse<Agent>> {
    return this.post<Agent>(API_ENDPOINTS.AGENTS.CREATE, config);
  }

  async updateAgent(id: string, config: Partial<AgentConfig>): Promise<ApiResponse<Agent>> {
    const endpoint = API_ENDPOINTS.AGENTS.UPDATE.replace(':id', id);
    return this.put<Agent>(endpoint, config);
  }

  async deleteAgent(id: string): Promise<ApiResponse<void>> {
    const endpoint = API_ENDPOINTS.AGENTS.DELETE.replace(':id', id);
    return this.delete<void>(endpoint);
  }

  async getAgentStatus(id: string): Promise<ApiResponse<{ status: string; last_activity: string }>> {
    const endpoint = API_ENDPOINTS.AGENTS.STATUS.replace(':id', id);
    return this.get(endpoint);
  }

  // 多智能体聊天
  async sendMessage(sessionId: string, message: string): Promise<ApiResponse<Message>> {
    return this.post<Message>(API_ENDPOINTS.MULTI_AGENT.CHAT, {
      session_id: sessionId,
      message,
      stream: false
    });
  }

  async createSession(agents: string[]): Promise<ApiResponse<Session>> {
    return this.post<Session>(API_ENDPOINTS.MULTI_AGENT.SESSIONS, { agents });
  }

  async getSessions(): Promise<ApiResponse<Session[]>> {
    return this.get<Session[]>(API_ENDPOINTS.MULTI_AGENT.SESSIONS);
  }

  async getSession(id: string): Promise<ApiResponse<Session>> {
    return this.get<Session>(`${API_ENDPOINTS.MULTI_AGENT.SESSIONS}/${id}`);
  }

  // 任务管理
  async getTasks(sessionId?: string): Promise<ApiResponse<Task[]>> {
    const params = sessionId ? `?session_id=${sessionId}` : '';
    return this.get<Task[]>(`${API_ENDPOINTS.TASKS.LIST}${params}`);
  }

  async getTask(id: string): Promise<ApiResponse<Task>> {
    const endpoint = API_ENDPOINTS.TASKS.DETAIL.replace(':id', id);
    return this.get<Task>(endpoint);
  }

  async createTask(task: Omit<Task, 'id' | 'created_at' | 'updated_at'>): Promise<ApiResponse<Task>> {
    return this.post<Task>(API_ENDPOINTS.TASKS.CREATE, task);
  }

  async updateTask(id: string, updates: Partial<Task>): Promise<ApiResponse<Task>> {
    const endpoint = API_ENDPOINTS.TASKS.UPDATE.replace(':id', id);
    return this.put<Task>(endpoint, updates);
  }

  async deleteTask(id: string): Promise<ApiResponse<void>> {
    const endpoint = API_ENDPOINTS.TASKS.DELETE.replace(':id', id);
    return this.delete<void>(endpoint);
  }

  // 系统健康检查
  async getMultiAgentHealth(): Promise<ApiResponse<{ status: string; timestamp: string }>> {
    return this.get(API_ENDPOINTS.MULTI_AGENT.HEALTH);
  }

  async getSystemHealth(): Promise<ApiResponse<{ status: string; services: Record<string, any> }>> {
    return this.get(API_ENDPOINTS.SYSTEM.HEALTH);
  }

  async getSystemMetrics(): Promise<ApiResponse<Record<string, any>>> {
    return this.get(API_ENDPOINTS.SYSTEM.METRICS);
  }

  // WebSocket连接辅助方法
  createWebSocketUrl(sessionId: string): string {
    const wsProtocol = this.config.baseURL.startsWith('https') ? 'wss' : 'ws';
    const baseUrl = this.config.baseURL.replace(/^https?:/, wsProtocol);
    const endpoint = API_ENDPOINTS.MULTI_AGENT.WEBSOCKET.replace(':sessionId', sessionId);
    return `${baseUrl}${endpoint}`;
  }
}

// 创建默认的智能体API客户端实例
export const createAgentApiClient = (baseURL: string, config?: Partial<ApiClientConfig>): AgentApiClient => {
  return new AgentApiClient({
    baseURL,
    timeout: 30000,
    retries: 3,
    headers: {
      'Content-Type': 'application/json'
    },
    ...config
  });
};