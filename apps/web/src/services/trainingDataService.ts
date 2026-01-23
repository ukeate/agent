import apiClient from './apiClient'

// ==================== 类型定义 ====================

export interface DataSourceCreate {
  source_id: string
  source_type: 'api' | 'file' | 'web' | 'database'
  name: string
  description: string
  config: Record<string, any>
}

export interface DataSourceResponse {
  id: string
  source_id: string
  source_type: string
  name: string
  description: string
  config: Record<string, any>
  is_active: boolean
  created_at: string
  updated_at: string
}

export interface DataCollectionRequest {
  source_id: string
  preprocessing_config?: Record<string, any>
}

export interface AnnotationTaskCreate {
  name: string
  description: string
  task_type: string
  data_records: string[]
  annotation_schema: Record<string, any>
  guidelines: string
  assignees: string[]
  created_by: string
  deadline?: string
}

export interface AnnotationSubmit {
  task_id: string
  record_id: string
  annotation_data: Record<string, any>
  confidence?: number
  time_spent?: number
}

export interface DataVersionCreate {
  dataset_name: string
  version_number: string
  description: string
  data_record_ids?: string[]
  parent_version?: string
  metadata?: Record<string, any>
}

export interface DataRecord {
  id: string
  record_id?: string
  source_id: string
  raw_data: any
  processed_data?: any
  metadata?: any
  status: string
  quality_score?: number
  created_at: string
  processed_at?: string
  updated_at: string
}

export interface AnnotationTask {
  id: string
  task_id: string
  name: string
  description: string
  task_type: string
  record_ids: string[]
  schema: Record<string, any>
  guidelines: string
  annotators: string[]
  status: string
  created_by: string
  created_at: string
  deadline?: string
}

export interface DataVersion {
  version_id: string
  dataset_name: string
  version_number: string
  description: string
  created_by: string
  created_at: string
  parent_version?: string
  metadata?: Record<string, any>
  record_count: number
}

export interface CollectionStatistics {
  total_records: number
  status_distribution: Record<string, number>
  quality_stats: {
    average: number
    minimum: number
    maximum: number
    records_with_quality_score: number
  }
  source_distribution: Record<
    string,
    {
      record_count: number
      average_quality: number
    }
  >
  time_range: {
    first_record: string | null
    latest_record: string | null
  }
}

export interface AnnotationProgress {
  task_id: string
  total_records: number
  annotated_records: number
  completion_percentage: number
  annotators_progress: Array<{
    annotator_id: string
    assigned_records: number
    completed_records: number
    completion_rate: number
  }>
}

export interface VersionStatistics {
  total_versions: number
  total_datasets: number
  total_records: number
  total_size_bytes: number
  average_records_per_version: number
  average_size_per_version: number
  by_dataset: Record<
    string,
    {
      version_count: number
      total_records: number
      total_size: number
      latest_version: {
        version_id: string
        version_number: string
        created_at: string
      } | null
    }
  >
  time_statistics: {
    earliest_version?: string
    latest_version?: string
    versions_last_30_days?: number
  }
}

//==================== Service Class ====================

class TrainingDataService {
  private baseUrl = '/training-data'

  // ==================== 数据源管理 ====================

  async createDataSource(sourceData: DataSourceCreate): Promise<{
    id: string
    source_id: string
    message: string
  }> {
    const response = await apiClient.post(`${this.baseUrl}/sources`, sourceData)
    return response.data
  }

  async listDataSources(
    activeOnly: boolean = true
  ): Promise<DataSourceResponse[]> {
    const response = await apiClient.get(`${this.baseUrl}/sources`, {
      params: { active_only: activeOnly },
    })
    return response.data
  }

  async updateDataSource(
    sourceId: string,
    updates: Record<string, any>
  ): Promise<{
    message: string
  }> {
    const response = await apiClient.put(
      `${this.baseUrl}/sources/${sourceId}`,
      updates
    )
    return response.data
  }

  async deleteDataSource(sourceId: string): Promise<{
    message: string
  }> {
    const response = await apiClient.delete(
      `${this.baseUrl}/sources/${sourceId}`
    )
    return response.data
  }

  // ==================== 数据收集 ====================

  async collectData(request: DataCollectionRequest): Promise<{
    message: string
    source_id: string
  }> {
    const response = await apiClient.post(`${this.baseUrl}/collect`, request)
    return response.data
  }

  async getDataRecords(
    params: {
      source_id?: string
      status?: string
      min_quality_score?: number
      limit?: number
      offset?: number
    } = {}
  ): Promise<{
    records: DataRecord[]
    count: number
    offset: number
    limit: number
  }> {
    const response = await apiClient.get(`${this.baseUrl}/records`, { params })
    return response.data
  }

  async getCollectionStatistics(
    sourceId?: string
  ): Promise<CollectionStatistics> {
    const params = sourceId ? { source_id: sourceId } : {}
    const response = await apiClient.get(`${this.baseUrl}/statistics`, {
      params,
    })
    return response.data
  }

  async reprocessRecords(
    params: {
      record_ids?: string[]
      source_id?: string
      status_filter?: string
      preprocessing_config?: Record<string, any>
    } = {}
  ): Promise<any> {
    const response = await apiClient.post(`${this.baseUrl}/reprocess`, params)
    return response.data
  }

  // ==================== 标注管理 ====================

  async createAnnotationTask(taskData: AnnotationTaskCreate): Promise<{
    id: string
    task_id: string
    message: string
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/annotation-tasks`,
      taskData
    )
    return response.data
  }

  async listAnnotationTasks(
    params: {
      assignee_id?: string
      status?: string
      created_by?: string
      limit?: number
      offset?: number
    } = {}
  ): Promise<{
    tasks: AnnotationTask[]
    count: number
    offset: number
    limit: number
  }> {
    const response = await apiClient.get(`${this.baseUrl}/annotation-tasks`, {
      params,
    })
    return response.data
  }

  async getAnnotationTaskDetails(taskId: string): Promise<AnnotationTask> {
    const response = await apiClient.get(
      `${this.baseUrl}/annotation-tasks/${taskId}`
    )
    return response.data
  }

  async getAnnotationProgress(taskId: string): Promise<AnnotationProgress> {
    const response = await apiClient.get(
      `${this.baseUrl}/annotation-tasks/${taskId}/progress`
    )
    return response.data
  }

  async assignAnnotationTask(
    taskId: string,
    userIds: string[]
  ): Promise<{
    message: string
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/annotation-tasks/${taskId}/assign`,
      userIds
    )
    return response.data
  }

  async submitAnnotation(annotationData: AnnotationSubmit): Promise<{
    id: string
    annotation_id: string
    message: string
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/annotations`,
      annotationData
    )
    return response.data
  }

  async getInterAnnotatorAgreement(taskId: string): Promise<any> {
    const response = await apiClient.get(
      `${this.baseUrl}/annotation-tasks/${taskId}/agreement`
    )
    return response.data
  }

  async getQualityControlReport(taskId: string): Promise<any> {
    const response = await apiClient.get(
      `${this.baseUrl}/annotation-tasks/${taskId}/quality-report`
    )
    return response.data
  }

  async getUserAnnotations(params: {
    task_id?: string
    status?: string
    limit?: number
    offset?: number
  }): Promise<{
    annotations: any[]
    count: number
    offset: number
    limit: number
  }> {
    const response = await apiClient.get(`${this.baseUrl}/annotations`, {
      params,
    })
    return response.data
  }

  // ==================== 版本管理 ====================

  async createDataVersion(versionData: DataVersionCreate): Promise<{
    version_id: string
    dataset_name: string
    version_number: string
    record_count: number
    message: string
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/versions`,
      versionData
    )
    return response.data
  }

  async listDatasets(): Promise<{
    datasets: string[]
  }> {
    const response = await apiClient.get(`${this.baseUrl}/datasets`)
    return response.data
  }

  async listDatasetVersions(datasetName: string): Promise<{
    dataset_name: string
    versions: DataVersion[]
    count: number
  }> {
    const response = await apiClient.get(
      `${this.baseUrl}/datasets/${datasetName}/versions`
    )
    return response.data
  }

  async getVersionData(versionId: string): Promise<{
    version_id: string
    data: any[]
    record_count: number
  }> {
    const response = await apiClient.get(
      `${this.baseUrl}/versions/${versionId}`
    )
    return response.data
  }

  async getVersionHistory(versionId: string): Promise<{
    version_id: string
    history: DataVersion[]
    count: number
  }> {
    const response = await apiClient.get(
      `${this.baseUrl}/versions/${versionId}/history`
    )
    return response.data
  }

  async compareVersions(versionId1: string, versionId2: string): Promise<any> {
    const response = await apiClient.post(
      `${this.baseUrl}/versions/${versionId1}/compare/${versionId2}`
    )
    return response.data
  }

  async exportVersion(
    versionId: string,
    format: 'jsonl' | 'json' | 'csv' = 'jsonl'
  ): Promise<{
    version_id: string
    export_path: string
    format: string
    message: string
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/versions/${versionId}/export`,
      null,
      {
        params: { format },
      }
    )
    return response.data
  }

  async rollbackDataset(
    datasetName: string,
    targetVersionId: string
  ): Promise<{
    dataset_name: string
    target_version_id: string
    new_version_id: string
    message: string
  }> {
    const response = await apiClient.post(
      `${this.baseUrl}/datasets/${datasetName}/rollback`,
      null,
      {
        params: { target_version_id: targetVersionId },
      }
    )
    return response.data
  }

  async getVersionStatistics(datasetName?: string): Promise<VersionStatistics> {
    const params = datasetName ? { dataset_name: datasetName } : {}
    const response = await apiClient.get(`${this.baseUrl}/version-statistics`, {
      params,
    })
    return response.data
  }

  async deleteVersion(
    versionId: string,
    removeFiles: boolean = true
  ): Promise<{
    message: string
  }> {
    const response = await apiClient.delete(
      `${this.baseUrl}/versions/${versionId}`,
      {
        params: { remove_files: removeFiles },
      }
    )
    return response.data
  }

  // ==================== 系统状态 ====================

  async getHealthCheck(): Promise<{
    status: string
    timestamp: string
    services: Record<string, string>
  }> {
    const response = await apiClient.get(`${this.baseUrl}/health`)
    return response.data
  }

  async getQueueStatus(): Promise<any> {
    const response = await apiClient.get(`${this.baseUrl}/queue-status`)
    return response.data
  }
}

// ==================== 导出 ====================

export const trainingDataService = new TrainingDataService()
export default trainingDataService
