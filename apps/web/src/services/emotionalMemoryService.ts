/**
 * 情感记忆管理系统API服务（真实接口版）
 * 对齐后端：apps/api/src/api/v1/emotional_memory.py
 */

import { apiClient } from './apiClient'

const API_BASE = '/emotional-memory'

export enum StorageLayer {
  HOT = 'hot',
  WARM = 'warm',
  COLD = 'cold',
}

export enum EmotionType {
  JOY = 'joy',
  SADNESS = 'sadness',
  ANGER = 'anger',
  FEAR = 'fear',
  SURPRISE = 'surprise',
  DISGUST = 'disgust',
  CALM = 'calm',
  ANXIETY = 'anxiety',
  EXCITEMENT = 'excitement',
}

export type EmotionalMemory = {
  id: string
  user_id: string
  timestamp: string
  emotion_type: string
  intensity: number
  content: string
  storage_layer: string
  importance_score?: number
  tags?: string[]
  privacy_level?: string
  access_count?: number
  is_encrypted?: boolean
  valence?: number
  arousal?: number
}

export type EmotionalEvent = {
  id: string
  memory_id: string
  event_type: string
  trigger_source: string | null
  timestamp: string
  impact_score: number | null
  affected_emotions: string[] | null
  causal_strength: number | null
}

export type UserPreference = {
  user_id: string
  dominant_emotions: string[]
  emotion_weights: Record<string, number>
  preferred_responses: string[]
  avoided_triggers: string[]
  learning_rate: number
  model_accuracy: number
  confidence_score: number
  training_samples: number
  last_training: string | null
  interaction_style: string | null
}

export type TriggerPattern = {
  id: string
  pattern_name: string
  pattern_type: string
  frequency: number
  avg_intensity: number
  confidence: number
  reliability: number
  triggered_emotions: string[]
  is_active: boolean
  last_triggered: string | null
}

export type MemoryStatistics = {
  total_memories: number
  storage_distribution: Record<string, number>
  emotion_distribution: Record<string, number>
  avg_intensity: number
}

export type CreateMemoryInput = {
  emotion_type: EmotionType
  intensity: number
  context: string
  importance_score?: number
  tags?: string[]
  metadata?: Record<string, any>
  storage_layer?: StorageLayer
  trigger_factors?: string[]
  related_memories?: string[]
}

export type MemorySearchQuery = {
  query: string
  emotion_filter?: EmotionType[]
  importance_min?: number
  time_range?: string[]
  storage_layer?: StorageLayer
  limit?: number
}

export class EmotionalMemoryService {
  async createMemory(userId: string, input: CreateMemoryInput) {
    const response = await apiClient.post(`${API_BASE}/memories`, {
      user_id: userId,
      emotion_type: input.emotion_type,
      intensity: input.intensity,
      context: input.context,
      importance_score: input.importance_score ?? 0.5,
      storage_layer: input.storage_layer ?? StorageLayer.HOT,
      trigger_factors: input.trigger_factors ?? [],
      related_memories: input.related_memories ?? [],
      tags: input.tags ?? [],
      metadata: input.metadata ?? {},
    })
    return response.data as Record<string, any>
  }

  async listMemories(params?: {
    storage_layer?: StorageLayer
    emotion_type?: EmotionType
    importance_min?: number
    limit?: number
  }) {
    const response = await apiClient.get(`${API_BASE}/memories`, { params })
    return response.data as EmotionalMemory[]
  }

  async searchMemories(query: MemorySearchQuery) {
    const response = await apiClient.post(`${API_BASE}/memories/search`, query)
    return response.data as any[]
  }

  async detectEvents(timeWindowHours: number = 24) {
    const response = await apiClient.post(`${API_BASE}/events/detect`, null, {
      params: { time_window: timeWindowHours },
    })
    return response.data as any
  }

  async getEvents(
    userId: string,
    params?: { limit?: number; offset?: number }
  ) {
    const response = await apiClient.get(`${API_BASE}/events/${userId}`, {
      params,
    })
    return response.data as EmotionalEvent[]
  }

  async learnPreferences(feedback_data?: Record<string, any>) {
    const response = await apiClient.post(
      `${API_BASE}/preferences/learn`,
      feedback_data ?? null
    )
    return response.data as any
  }

  async getPreferences(userId: string) {
    const response = await apiClient.get(`${API_BASE}/preferences/${userId}`)
    return response.data as UserPreference
  }

  async getTriggerPatterns(
    userId: string,
    params?: { confidence_min?: number }
  ) {
    const response = await apiClient.get(`${API_BASE}/patterns/${userId}`, {
      params,
    })
    return response.data as TriggerPattern[]
  }

  async listTriggerPatterns(params?: { min_frequency?: number }) {
    const response = await apiClient.get(`${API_BASE}/patterns/triggers`, {
      params,
    })
    return response.data as any[]
  }

  async identifyPatterns(userId: string, minFrequency: number = 3) {
    const response = await apiClient.post(
      `${API_BASE}/patterns/identify`,
      null,
      {
        params: { user_id: userId, min_frequency: minFrequency },
      }
    )
    return response.data as any[]
  }

  async predictRisk(userId: string, current_context: Record<string, any>) {
    const response = await apiClient.post(
      `${API_BASE}/patterns/predict`,
      current_context,
      {
        params: { user_id: userId },
      }
    )
    return response.data as Record<string, any>
  }

  async optimizeStorage() {
    const response = await apiClient.post(`${API_BASE}/storage/optimize`)
    return response.data as Record<string, any>
  }

  async deleteMemory(userId: string, memoryId: string) {
    const response = await apiClient.delete(
      `${API_BASE}/memories/${userId}/${memoryId}`
    )
    return response.data as Record<string, any>
  }

  async exportMemories(userId: string, format: 'json' | 'csv' = 'json') {
    const response = await apiClient.post(
      `${API_BASE}/memories/export/${userId}`,
      null,
      {
        params: { format },
      }
    )
    return response.data as { data: any; format: string; count: number }
  }

  async getStatistics(userId: string) {
    const response = await apiClient.get(`${API_BASE}/statistics/${userId}`)
    return response.data as MemoryStatistics
  }
}

export const emotionalMemoryService = new EmotionalMemoryService()
