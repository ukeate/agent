import apiClient from './apiClient'

import { logger } from '../utils/logger'
// ==================== 类型定义 ====================

export type HealthStatus = 'healthy' | 'degraded' | 'unhealthy'

export interface HealthCheckResult {
  status: HealthStatus
  timestamp: string
  check_duration_ms?: number
  components?: Record<string, HealthComponent>
  failed_components?: string[]
  degraded_components?: string[]
  metrics?: Record<string, unknown>
  error?: string
}

export interface HealthComponent {
  status: HealthStatus
  [key: string]: unknown
}

export interface SystemMetrics {
  timestamp?: string
  system?: {
    cpu_percent?: number
    memory_percent?: number
    disk_percent?: number
    network_connections?: number | null
    network_connections_error?: string | null
  }
  performance?: {
    uptime_seconds?: number
    total_requests?: number
    active_requests?: number
    requests_per_minute?: number
    average_response_time_ms?: number
    recent_error_count?: number
    error_rate?: number
    error_counts?: Record<string, number>
  }
  custom?: Record<string, unknown>
  error?: string
}

export interface LivenessCheck {
  status: 'alive' | 'dead'
  error?: string
}

export interface ReadinessCheck {
  status: 'ready' | 'not_ready'
  services_ready?: boolean
  database_ready?: boolean
  cache_ready?: boolean
  message?: string
}

export interface HealthAlert {
  name: string
  message: string
  severity: 'info' | 'warning' | 'error' | 'critical'
  timestamp: string
  metrics?: Record<string, unknown>
}

export interface HealthAlertsResponse {
  total_alerts: number
  alerts: HealthAlert[]
  error?: string
}

// ==================== Service Class ====================

class HealthService {
  private baseUrl = '/health'
  private pollingInterval: number | null = null
  private pollingInFlight = false
  private healthCallbacks = new Set<(health: HealthCheckResult) => void>()

  private notifyHealth = (health: HealthCheckResult) => {
    this.healthCallbacks.forEach(callback => callback(health))
  }

  private pollHealth = async () => {
    if (this.pollingInFlight) return
    this.pollingInFlight = true
    try {
      const health = await this.getHealth()
      this.notifyHealth(health)
    } catch (error) {
      logger.error('健康监控错误:', error)
      this.notifyHealth({
        status: 'unhealthy',
        timestamp: new Date().toISOString(),
        error: '获取健康状态失败',
      })
    } finally {
      this.pollingInFlight = false
    }
  }

  // ==================== 基础健康检查 ====================

  async getHealth(detailed: boolean = false): Promise<HealthCheckResult> {
    const response = await apiClient.get(this.baseUrl, {
      params: { detailed },
    })
    return response.data
  }

  async checkLiveness(): Promise<LivenessCheck> {
    const response = await apiClient.get(`${this.baseUrl}/live`)
    return response.data
  }

  async checkReadiness(): Promise<ReadinessCheck> {
    const response = await apiClient.get(`${this.baseUrl}/ready`)
    return response.data
  }

  // ==================== 详细健康信息 ====================

  async getDetailedHealth(): Promise<HealthCheckResult> {
    return this.getHealth(true)
  }

  async getSystemMetrics(): Promise<SystemMetrics> {
    const response = await apiClient.get(`${this.baseUrl}/metrics`)
    return response.data
  }

  // ==================== 健康监控 ====================

  startHealthMonitoring(
    intervalMs: number = 30000,
    callback?: (health: HealthCheckResult) => void
  ): () => void {
    if (callback) {
      this.healthCallbacks.add(callback)
    }

    if (!this.pollingInterval) {
      this.pollingInterval = window.setInterval(this.pollHealth, intervalMs)
      this.pollHealth()
    }

    if (!callback) return () => undefined

    return () => {
      this.healthCallbacks.delete(callback)
      if (this.healthCallbacks.size === 0) {
        this.stopHealthMonitoring()
      }
    }
  }

  stopHealthMonitoring(): void {
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval)
      this.pollingInterval = null
    }
    this.healthCallbacks.clear()
  }

  // ==================== 健康告警 ====================

  async getHealthAlerts(params?: {
    severity?: string
    component?: string
    resolved?: boolean
    limit?: number
  }): Promise<HealthAlertsResponse> {
    const response = await apiClient.get(`${this.baseUrl}/alerts`, { params })
    return response.data
  }
}

// ==================== 导出 ====================

export const healthService = new HealthService()
export default healthService
