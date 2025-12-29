export interface Message {
  id: string
  content: string
  role: 'user' | 'agent'
  timestamp: string
  toolCalls?: ToolCall[]
  reasoningSteps?: ReasoningStep[]
}

export interface ToolCall {
  id: string
  name: string
  args: Record<string, any>
  result?: any
  status: 'pending' | 'success' | 'error'
  timestamp: string
}

export interface ReasoningStep {
  id: string
  type: 'thought' | 'action' | 'observation'
  content: string
  timestamp: string
}

export interface Conversation {
  id: string
  title: string
  messages: Message[]
  createdAt: string
  updatedAt: string
  messageCount?: number
  userMessageCount?: number
}

export interface AgentStatus {
  id: string
  name: string
  status: 'idle' | 'thinking' | 'acting' | 'error'
  currentTask?: string
}

export interface ApiResponse<T = any> {
  data: T
  success: boolean
  error?: string
  message?: string
}

export interface ChatRequest {
  message: string
  conversationId?: string
  stream?: boolean
}

export interface ChatResponse {
  message: string
  conversation_id?: string
  message_id: string
  reasoning_steps: string[]
  tool_calls: any[]
  metadata: Record<string, any>
  response_time: number
  token_usage?: Record<string, number> | null
}
