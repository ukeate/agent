export type AgentRole =
  | 'code_expert'
  | 'architect'
  | 'doc_expert'
  | 'supervisor'
  | 'knowledge_retrieval'
  | 'assistant'
  | 'critic'
  | 'coder'
  | 'planner'
  | 'executor'

export type AgentStatus = 'active' | 'idle' | 'busy' | 'offline'

export type ConversationStatus =
  | 'created'
  | 'active'
  | 'paused'
  | 'completed'
  | 'terminated'
  | 'error'

export type MessageRole = 'user' | 'assistant' | 'system' | 'agent'

export interface AgentConfiguration {
  model: string
  temperature: number
  max_tokens: number
  tools: string[]
  system_prompt: string
}

export interface Agent {
  id: string
  name: string
  role: AgentRole
  status: AgentStatus
  capabilities: string[]
  configuration: AgentConfiguration
  created_at: string
  updated_at: string
}

export interface ConversationParticipant {
  name: string
  role: AgentRole
  status?: string
  message_count?: number
  capabilities?: string[]
}

export interface ConversationConfig {
  max_rounds?: number
  timeout_seconds?: number
  auto_reply?: boolean
  temperature?: number
  max_tokens?: number
}
