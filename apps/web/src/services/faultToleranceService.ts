import apiClient from './apiClient'

export interface ServiceHealth {
  service_id: string
  service_name: string
  status: 'healthy' | 'degraded' | 'unhealthy' | 'unknown'
  health_score: number
  last_check: string
  response_time_ms: number
  error_rate: number
  success_rate: number
  circuit_breaker_status: 'closed' | 'open' | 'half_open'
  retries_count: number
  failures_count: number
}

export interface FaultEvent {
  id: string
  service_id: string
  event_type: 'failure' | 'recovery' | 'timeout' | 'circuit_break' | 'retry'
  severity: 'low' | 'medium' | 'high' | 'critical'
  message: string
  timestamp: string
  duration_ms?: number
  error_details?: string
  recovery_action?: string
  impact_score: number
}

export interface ResiliencePolicy {
  id: string
  name: string
  service_pattern: string
  enabled: boolean
  retry_policy?: RetryPolicy
  circuit_breaker_policy?: CircuitBreakerPolicy
  timeout_policy?: TimeoutPolicy
  bulkhead_policy?: BulkheadPolicy
  fallback_policy?: FallbackPolicy
  created_at: string
  updated_at?: string
}

export interface RetryPolicy {
  enabled: boolean
  max_attempts: number
  initial_delay_ms: number
  max_delay_ms: number
  backoff_multiplier: number
  retry_on_errors: string[]
  jitter: boolean
}

export interface CircuitBreakerPolicy {
  enabled: boolean
  failure_threshold: number
  success_threshold: number
  timeout_ms: number
  half_open_max_calls: number
  sliding_window_size: number
  sliding_window_type: 'count' | 'time'
}

export interface TimeoutPolicy {
  enabled: boolean
  timeout_ms: number
  cancel_on_timeout: boolean
}

export interface BulkheadPolicy {
  enabled: boolean
  max_concurrent_calls: number
  max_wait_duration_ms: number
  queue_size: number
}

export interface FallbackPolicy {
  enabled: boolean
  fallback_action: 'default_value' | 'cache' | 'alternate_service' | 'custom'
  fallback_value?: any
  fallback_service?: string
  cache_duration_ms?: number
}

export interface FailureInjection {
  id: string
  name: string
  target_service: string
  injection_type: 'latency' | 'error' | 'timeout' | 'resource_exhaustion'
  enabled: boolean
  probability: number
  latency_ms?: number
  error_code?: number
  error_message?: string
  duration_ms?: number
  schedule?: string
}

export interface RecoveryStrategy {
  id: string
  name: string
  trigger_condition: string
  actions: RecoveryAction[]
  priority: number
  enabled: boolean
  auto_execute: boolean
  cooldown_ms: number
}

export interface RecoveryAction {
  type: 'restart' | 'scale' | 'failover' | 'rollback' | 'notify' | 'custom'
  target: string
  parameters: Record<string, any>
  timeout_ms: number
  on_failure: 'continue' | 'abort' | 'retry'
}

export interface ResilienceMetrics {
  availability: number
  mttr: number // Mean Time To Recovery
  mtbf: number // Mean Time Between Failures
  error_budget_remaining: number
  slo_compliance: number
  recovery_success_rate: number
  fault_detection_time_ms: number
  service_dependencies: ServiceDependency[]
}

export interface ServiceDependency {
  service: string
  dependency: string
  criticality: 'low' | 'medium' | 'high' | 'critical'
  failure_impact: number
  redundancy_available: boolean
}

export interface ChaosExperiment {
  id: string
  name: string
  description: string
  target_services: string[]
  scenarios: ChaosScenario[]
  status: 'planned' | 'running' | 'completed' | 'failed'
  started_at?: string
  completed_at?: string
  results?: ChaosResults
}

export interface ChaosScenario {
  name: string
  type:
    | 'network_delay'
    | 'service_failure'
    | 'resource_stress'
    | 'data_corruption'
  duration_ms: number
  intensity: number
  target_percentage: number
}

export interface ChaosResults {
  impact_assessment: string
  services_affected: string[]
  recovery_time_ms: number
  data_loss: boolean
  recommendations: string[]
  resilience_score: number
}

class FaultToleranceService {
  private baseUrl = '/fault-tolerance'

  async getSystemStatus(): Promise<any> {
    const response = await apiClient.get(`${this.baseUrl}/status`)
    return response.data
  }

  async getMetrics(): Promise<any> {
    const response = await apiClient.get(`${this.baseUrl}/metrics`)
    return response.data
  }

  async getHealth(): Promise<any> {
    const response = await apiClient.get(`${this.baseUrl}/health`)
    return response.data
  }

  // 服务健康管理
  async getServiceHealth(serviceId?: string): Promise<ServiceHealth[]> {
    if (serviceId) {
      const response = await apiClient.get(
        `${this.baseUrl}/health/${serviceId}`
      )
      return [response.data]
    }
    const response = await apiClient.get(`${this.baseUrl}/health`)
    return response.data?.services || []
  }

  async checkServiceHealth(serviceId: string): Promise<ServiceHealth> {
    const response = await apiClient.get(`${this.baseUrl}/health/${serviceId}`)
    return response.data
  }

  // 故障事件
  async listFaultEvents(
    serviceId?: string,
    severity?: string,
    resolved?: boolean
  ): Promise<FaultEvent[]> {
    const params: any = {}
    if (severity) params.severity = severity
    if (resolved !== undefined) params.resolved = resolved
    const response = await apiClient.get(`${this.baseUrl}/faults`, { params })
    const items: FaultEvent[] = response.data || []
    if (!serviceId) {
      return items
    }
    return items.filter((item: any) =>
      Array.isArray(item.affected_components)
        ? item.affected_components.includes(serviceId)
        : false
    )
  }

  async getFaultEvent(eventId: string): Promise<FaultEvent> {
    const response = await apiClient.get(`${this.baseUrl}/faults`, {
      params: { limit: 200 },
    })
    const items: FaultEvent[] = response.data || []
    const match = items.find(
      (item: any) => item.fault_id === eventId || item.id === eventId
    )
    if (!match) {
      throw new Error('故障事件不存在')
    }
    return match
  }

  // 恢复统计
  async getRecoveryStatistics(): Promise<{
    total_recoveries: number
    success_rate: number
    avg_recovery_time: number
    strategy_success_rates: Record<string, number>
    recent_recoveries?: RecoveryRecord[]
  }> {
    const response = await apiClient.get(`${this.baseUrl}/recovery/statistics`)
    return response.data
  }

  // 弹性策略管理
  async listResiliencePolicies(
    servicePattern?: string
  ): Promise<ResiliencePolicy[]> {
    const params = servicePattern ? { service_pattern: servicePattern } : {}
    const response = await apiClient.get(`${this.baseUrl}/policies`, { params })
    return response.data
  }

  async getResiliencePolicy(policyId: string): Promise<ResiliencePolicy> {
    const response = await apiClient.get(`${this.baseUrl}/policies/${policyId}`)
    return response.data
  }

  async createResiliencePolicy(
    policy: Omit<ResiliencePolicy, 'id' | 'created_at'>
  ): Promise<ResiliencePolicy> {
    const response = await apiClient.post(`${this.baseUrl}/policies`, policy)
    return response.data
  }

  async updateResiliencePolicy(
    policyId: string,
    updates: Partial<ResiliencePolicy>
  ): Promise<ResiliencePolicy> {
    const response = await apiClient.put(
      `${this.baseUrl}/policies/${policyId}`,
      updates
    )
    return response.data
  }

  async deleteResiliencePolicy(policyId: string): Promise<void> {
    await apiClient.delete(`${this.baseUrl}/policies/${policyId}`)
  }

  // 故障注入
  async listFailureInjections(): Promise<FailureInjection[]> {
    const response = await apiClient.get(`${this.baseUrl}/injections`)
    return response.data
  }

  async createFailureInjection(
    injection: Omit<FailureInjection, 'id'>
  ): Promise<FailureInjection> {
    const response = await apiClient.post(
      `${this.baseUrl}/injections`,
      injection
    )
    return response.data
  }

  async toggleFailureInjection(
    injectionId: string,
    enabled: boolean
  ): Promise<void> {
    await apiClient.patch(`${this.baseUrl}/injections/${injectionId}`, {
      enabled,
    })
  }

  async deleteFailureInjection(injectionId: string): Promise<void> {
    await apiClient.delete(`${this.baseUrl}/injections/${injectionId}`)
  }

  // 恢复策略
  async listRecoveryStrategies(): Promise<RecoveryStrategy[]> {
    const response = await apiClient.get(`${this.baseUrl}/recovery-strategies`)
    return response.data
  }

  async createRecoveryStrategy(
    strategy: Omit<RecoveryStrategy, 'id'>
  ): Promise<RecoveryStrategy> {
    const response = await apiClient.post(
      `${this.baseUrl}/recovery-strategies`,
      strategy
    )
    return response.data
  }

  async executeRecoveryStrategy(strategyId: string): Promise<{
    execution_id: string
    status: string
    actions_executed: number
    actions_failed: number
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/recovery-strategies/${strategyId}/execute`
    )
    return response.data
  }

  // 弹性指标
  async getResilienceMetrics(serviceId?: string): Promise<ResilienceMetrics> {
    const params = serviceId ? { service_id: serviceId } : {}
    const response = await apiClient.get(`${this.baseUrl}/metrics`, { params })
    return response.data
  }

  async calculateMTTR(serviceId: string, timeRange: string): Promise<number> {
    const response = await apiClient.get(`${this.baseUrl}/metrics/mttr`, {
      params: { service_id: serviceId, time_range: timeRange },
    })
    return response.data.mttr
  }

  async calculateAvailability(
    serviceId: string,
    timeRange: string
  ): Promise<number> {
    const response = await apiClient.get(
      `${this.baseUrl}/metrics/availability`,
      {
        params: { service_id: serviceId, time_range: timeRange },
      }
    )
    return response.data.availability
  }

  // 混沌工程
  async listChaosExperiments(): Promise<ChaosExperiment[]> {
    const response = await apiClient.get(`${this.baseUrl}/chaos/experiments`)
    return response.data
  }

  async createChaosExperiment(
    experiment: Omit<ChaosExperiment, 'id' | 'status'>
  ): Promise<ChaosExperiment> {
    const response = await apiClient.post(
      `${this.baseUrl}/chaos/experiments`,
      experiment
    )
    return response.data
  }

  async runChaosExperiment(experimentId: string): Promise<{
    execution_id: string
    status: string
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/chaos/experiments/${experimentId}/run`
    )
    return response.data
  }

  async getChaosExperimentResults(experimentId: string): Promise<ChaosResults> {
    const response = await apiClient.get(
      `${this.baseUrl}/chaos/experiments/${experimentId}/results`
    )
    return response.data
  }

  // 服务依赖分析
  async analyzeServiceDependencies(
    serviceId: string
  ): Promise<ServiceDependency[]> {
    const response = await apiClient.get(
      `${this.baseUrl}/dependencies/${serviceId}`
    )
    return response.data
  }

  async getFailurePropagationAnalysis(serviceId: string): Promise<{
    direct_impact: string[]
    cascading_failures: string[]
    risk_score: number
    mitigation_recommendations: string[]
  }> {
    const response = await apiClient.get(
      `${this.baseUrl}/dependencies/${serviceId}/impact-analysis`
    )
    return response.data
  }

  // 报告和导出
  async generateResilienceReport(timeRange: string): Promise<{
    summary: string
    key_metrics: ResilienceMetrics
    incidents: FaultEvent[]
    recommendations: string[]
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/reports/resilience`,
      {
        time_range: timeRange,
      }
    )
    return response.data
  }

  async exportFaultEvents(
    format: 'json' | 'csv' | 'pdf' = 'json'
  ): Promise<Blob> {
    const response = await apiClient.get(`${this.baseUrl}/export/events`, {
      params: { format },
      responseType: 'blob',
    })
    return response.data
  }

  // 新增：未使用的API方法 - 根据api-ui.md
  async getFaultToleranceReport(): Promise<{
    summary: string
    active_faults: FaultEvent[]
    recovery_statistics: any
    system_health: any
  }> {
    const response = await apiClient.get(`${this.baseUrl}/report`)
    return response.data
  }

  async listActiveFaults(): Promise<FaultEvent[]> {
    const response = await apiClient.get(`${this.baseUrl}/faults`)
    return response.data
  }

  async getRecoveryStatistics(): Promise<{
    total_recoveries: number
    success_rate: number
    average_recovery_time: number
    recent_recoveries: any[]
  }> {
    const response = await apiClient.get(`${this.baseUrl}/recovery/statistics`)
    return response.data
  }

  // 手动备份操作
  async triggerManualBackup(
    componentIds: string[],
    backupType?: string
  ): Promise<{
    backup_results: Record<string, boolean>
    success_count: number
    total_count: number
    timestamp: string
  }> {
    const response = await apiClient.post(`${this.baseUrl}/backup/manual`, {
      component_ids: componentIds,
      backup_type: backupType || 'full_backup',
    })
    return response.data
  }

  async getBackupStatistics(): Promise<{
    total_backups: number
    successful_backups: number
    failed_backups: number
    last_backup: string
    backup_size_mb: number
  }> {
    const response = await apiClient.get(`${this.baseUrl}/backup/statistics`)
    return response.data
  }

  async restoreFromBackup(backupId: string): Promise<{
    restore_id: string
    status: string
    started_at: string
  }> {
    const response = await apiClient.post(`${this.baseUrl}/backup/restore`, {
      backup_id: backupId,
    })
    return response.data
  }

  async validateBackup(backupId: string): Promise<{
    valid: boolean
    integrity_score: number
    validation_details: string[]
  }> {
    const response = await apiClient.post(`${this.baseUrl}/backup/validate`, {
      backup_id: backupId,
    })
    return response.data
  }

  // 数据一致性检查
  async checkDataConsistency(dataKeys: string[] = []): Promise<{
    check_id: string
    checked_at: string
    components: string[]
    consistent: boolean
    inconsistencies_count: number
    inconsistencies: any[]
    repair_actions: any[]
  }> {
    const response = await apiClient.post(`${this.baseUrl}/consistency/check`, {
      data_keys: dataKeys,
    })
    return response.data
  }

  async getConsistencyStatistics(): Promise<{
    total_checks: number
    passed_checks: number
    failed_checks: number
    last_check: string
    consistency_score: number
  }> {
    const response = await apiClient.get(
      `${this.baseUrl}/consistency/statistics`
    )
    return response.data
  }

  async repairConsistencyIssue(checkId: string): Promise<{
    repair_id: string
    status: string
    started_at: string
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/consistency/${checkId}/repair`
    )
    return response.data
  }

  async forceRepairConsistency(
    dataKey: string,
    authoritativeComponentId: string
  ): Promise<{
    status: string
    data_key: string
    authoritative_component: string
    timestamp: string
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/consistency/force-repair`,
      {
        data_key: dataKey,
        authoritative_component_id: authoritativeComponentId,
      }
    )
    return response.data
  }

  // 故障注入测试
  async injectFault(
    faultType: string,
    targetService: string,
    duration?: number
  ): Promise<{
    status: string
    fault_id: string
    component_id: string
    fault_type: string
    duration_seconds: number
    timestamp: string
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/testing/inject-fault`,
      {
        component_id: targetService,
        fault_type: faultType,
        duration_seconds: duration
          ? Math.max(1, Math.round(duration / 1000))
          : 60,
      }
    )
    return response.data
  }

  // 枚举API
  async getFaultTypes(): Promise<string[]> {
    const response = await apiClient.get(`${this.baseUrl}/enums/fault-types`)
    const items = response.data?.fault_types || []
    return items.map((item: any) => item.value)
  }

  async getSeverityLevels(): Promise<string[]> {
    const response = await apiClient.get(`${this.baseUrl}/enums/severities`)
    const items = response.data?.severities || []
    return items.map((item: any) => item.value)
  }

  async getBackupTypes(): Promise<string[]> {
    const response = await apiClient.get(`${this.baseUrl}/enums/backup-types`)
    const items = response.data?.backup_types || []
    return items.map((item: any) => item.value)
  }

  async getRecoveryStrategies(): Promise<string[]> {
    const response = await apiClient.get(
      `${this.baseUrl}/enums/recovery-strategies`
    )
    const items = response.data?.recovery_strategies || []
    return items.map((item: any) => item.value)
  }

  // 系统控制
  async startSystem(): Promise<{
    status: string
    message: string
    started_at: string
  }> {
    const response = await apiClient.post(`${this.baseUrl}/system/start`)
    return response.data
  }

  async stopSystem(): Promise<{
    status: string
    message: string
    stopped_at: string
  }> {
    const response = await apiClient.post(`${this.baseUrl}/system/stop`)
    return response.data
  }
}

export const faultToleranceService = new FaultToleranceService()
