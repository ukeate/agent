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
  status?: string
  lastMessage?: Pick<Message, 'id' | 'content' | 'role' | 'timestamp'>
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
