/**
 * 记忆系统类型定义
 */

export enum MemoryType {
  WORKING = 'working',    // 工作记忆
  EPISODIC = 'episodic',  // 情景记忆
  SEMANTIC = 'semantic'   // 语义记忆
}

export enum MemoryStatus {
  ACTIVE = 'active',
  ARCHIVED = 'archived',
  COMPRESSED = 'compressed',
  DELETED = 'deleted'
}

export interface Memory {
  id: string
  type: MemoryType
  content: string
  metadata: Record<string, any>
  importance: number
  access_count: number
  created_at: string
  last_accessed: string
  status: MemoryStatus
  tags: string[]
  relevance_score?: number
  related_memories?: string[]
  session_id?: string
  user_id?: string
}

export interface MemoryCreateRequest {
  type: MemoryType
  content: string
  metadata?: Record<string, any>
  importance?: number
  tags?: string[]
  source?: string
}

export interface MemoryUpdateRequest {
  content?: string
  metadata?: Record<string, any>
  importance?: number
  tags?: string[]
  status?: MemoryStatus
}

export interface MemoryFilters {
  memory_types?: MemoryType[]
  status?: MemoryStatus[]
  min_importance?: number
  max_importance?: number
  created_after?: string
  created_before?: string
  tags?: string[]
  session_id?: string
  user_id?: string
}

export interface MemoryAnalytics {
  total_memories: number
  memories_by_type: Record<string, number>
  memories_by_status: Record<string, number>
  avg_importance: number
  total_access_count: number
  avg_access_count: number
  most_accessed_memories: Memory[]
  recent_memories: Memory[]
  memory_growth_rate: number
  storage_usage_mb: number
}

export interface MemoryGraphNode {
  id: string
  type: MemoryType
  importance: number
  label: string
  x?: number
  y?: number
}

export interface MemoryGraphEdge {
  source: string
  target: string
  weight: number
  type: string
}

export interface MemoryPattern {
  frequently_accessed: string[]
  recently_accessed: string[]
  co_accessed: string[]
  central_memories: string[]
}

export interface MemoryTrend {
  daily_counts: Record<string, number>
  type_trends: Record<string, Record<string, number>>
  total_memories: number
  growth_rate_percentage: number
  avg_daily_memories: number
}