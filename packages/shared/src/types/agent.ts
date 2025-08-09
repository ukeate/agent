/**
 * 智能体相关类型定义
 */

// 智能体类型
export enum AgentType {
  SINGLE = 'single',
  MULTI = 'multi',
  SUPERVISOR = 'supervisor',
  WORKER = 'worker'
}

// 智能体状态
export enum AgentStatus {
  IDLE = 'idle',
  RUNNING = 'running',
  WAITING = 'waiting',
  COMPLETED = 'completed',
  ERROR = 'error',
  CANCELLED = 'cancelled'
}

// 消息类型
export enum MessageType {
  USER = 'user',
  ASSISTANT = 'assistant',
  SYSTEM = 'system',
  FUNCTION = 'function',
  TOOL = 'tool'
}

// 消息角色
export type MessageRole = 'user' | 'assistant' | 'system' | 'function';

// 消息接口
export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  type: MessageType;
  timestamp: string;
  metadata?: Record<string, any>;
  agent_id?: string;
  session_id?: string;
}

// 智能体配置
export interface AgentConfig {
  id: string;
  name: string;
  type: AgentType;
  description: string;
  model: string;
  temperature?: number;
  max_tokens?: number;
  system_prompt?: string;
  tools?: string[];
  capabilities?: string[];
  created_at: string;
  updated_at: string;
}

// 智能体实例
export interface Agent {
  id: string;
  config: AgentConfig;
  status: AgentStatus;
  session_id?: string;
  current_task?: string;
  error_message?: string;
  last_activity: string;
  metrics: AgentMetrics;
}

// 智能体指标
export interface AgentMetrics {
  messages_processed: number;
  tasks_completed: number;
  errors_count: number;
  average_response_time: number;
  uptime: number;
}

// 会话接口
export interface Session {
  id: string;
  user_id?: string;
  agents: string[];
  messages: Message[];
  status: 'active' | 'completed' | 'error';
  created_at: string;
  updated_at: string;
  metadata?: Record<string, any>;
}

// 任务接口
export interface Task {
  id: string;
  title: string;
  description: string;
  agent_id: string;
  session_id: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  priority: 'low' | 'medium' | 'high' | 'urgent';
  created_at: string;
  updated_at: string;
  result?: any;
  error_message?: string;
}