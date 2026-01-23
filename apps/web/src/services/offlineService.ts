import { apiClient } from './apiClient'

export interface OfflineStatusResponse {
  mode: string
  network_status: string
  connection_quality: number
  pending_operations: number
  has_conflicts: boolean
  sync_in_progress: boolean
  last_sync_at?: string
}

export interface OfflineConfigResponse {
  max_concurrent_tasks: number
  batch_size: number
  sync_interval_seconds: number
  retry_max_count: number
  retry_backoff_factor: number
  connection_timeout_seconds: number
  vector_clock_enabled: boolean
  conflict_resolution: string
}

export interface SyncRequest {
  force: boolean
  batch_size: number
}

export interface ConflictResolutionRequest {
  conflict_id: string
  resolution_strategy: string
  resolved_data?: any
}

export interface OfflineOperation {
  id: string
  operation_type: string
  table_name: string
  object_id: string
  timestamp: string
  is_synced: boolean
  retry_count: number
}

export interface OfflineConflict {
  id: string
  table_name: string
  object_id: string
  conflict_type: string
  local_data: any
  remote_data: any
  created_at: string
}

export class OfflineService {
  private baseUrl = '/offline'

  async getConfig(): Promise<OfflineConfigResponse> {
    const response = await apiClient.get(`${this.baseUrl}/config`)
    return response.data
  }

  async getOfflineStatus(): Promise<OfflineStatusResponse> {
    const response = await apiClient.get(`${this.baseUrl}/status`)
    return response.data
  }

  async manualSync(request: SyncRequest): Promise<any> {
    const response = await apiClient.post(`${this.baseUrl}/sync`, request)
    return response.data
  }

  async getConflicts(): Promise<OfflineConflict[]> {
    const response = await apiClient.get(`${this.baseUrl}/conflicts`)
    return response.data
  }

  async resolveConflict(request: ConflictResolutionRequest): Promise<any> {
    const response = await apiClient.post(`${this.baseUrl}/resolve`, request)
    return response.data
  }

  async getOperations(
    limit: number = 100,
    offset: number = 0
  ): Promise<OfflineOperation[]> {
    const response = await apiClient.get(`${this.baseUrl}/operations`, {
      params: { limit, offset },
    })
    return response.data
  }

  async getStatistics(): Promise<any> {
    const response = await apiClient.get(`${this.baseUrl}/statistics`)
    return response.data
  }

  async setOfflineMode(mode: string): Promise<any> {
    const response = await apiClient.post(`${this.baseUrl}/mode/${mode}`)
    return response.data
  }

  async getNetworkStatus(): Promise<any> {
    const response = await apiClient.get(`${this.baseUrl}/network`)
    return response.data
  }

  async cleanupOldData(days: number = 30): Promise<any> {
    const response = await apiClient.post(`${this.baseUrl}/cleanup`, null, {
      params: { days },
    })
    return response.data
  }

  async healthCheck(): Promise<any> {
    const response = await apiClient.get(`${this.baseUrl}/health`)
    return response.data
  }

  // 便捷方法
  async forceSyncNow(): Promise<any> {
    return this.manualSync({ force: true, batch_size: 100 })
  }

  async startBackgroundSync(): Promise<any> {
    return this.manualSync({ force: false, batch_size: 50 })
  }

  async resolveConflictUseLocal(
    conflictId: string,
    localData: any
  ): Promise<any> {
    return this.resolveConflict({
      conflict_id: conflictId,
      resolution_strategy: 'client_wins',
      resolved_data: localData,
    })
  }

  async resolveConflictUseRemote(
    conflictId: string,
    remoteData: any
  ): Promise<any> {
    return this.resolveConflict({
      conflict_id: conflictId,
      resolution_strategy: 'server_wins',
      resolved_data: remoteData,
    })
  }

  async resolveConflictMerge(
    conflictId: string,
    mergedData: any
  ): Promise<any> {
    return this.resolveConflict({
      conflict_id: conflictId,
      resolution_strategy: 'merge',
      resolved_data: mergedData,
    })
  }

  async getPendingOperationsCount(): Promise<number> {
    const status = await this.getOfflineStatus()
    return status.pending_operations
  }

  async hasUnresolvedConflicts(): Promise<boolean> {
    const status = await this.getOfflineStatus()
    return status.has_conflicts
  }

  async isOnlineMode(): Promise<boolean> {
    const status = await this.getOfflineStatus()
    return status.mode === 'online'
  }

  async isOfflineMode(): Promise<boolean> {
    const status = await this.getOfflineStatus()
    return status.mode === 'offline'
  }

  async goOnline(): Promise<any> {
    return this.setOfflineMode('online')
  }

  async goOffline(): Promise<any> {
    return this.setOfflineMode('offline')
  }

  async setAutoMode(): Promise<any> {
    return this.setOfflineMode('auto')
  }
}

export const offlineService = new OfflineService()
