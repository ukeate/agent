import { apiClient } from './apiClient'

export enum LayerType {
  MUTUALLY_EXCLUSIVE = 'mutually_exclusive',
  ORTHOGONAL = 'orthogonal',
  HOLDBACK = 'holdback',
}

export enum ConflictResolution {
  PRIORITY_BASED = 'priority_based',
  FIRST_COME_FIRST_SERVE = 'first_come_first_serve',
  ROUND_ROBIN = 'round_robin',
  RANDOM = 'random',
}

export interface ExperimentLayer {
  layer_id: string
  name: string
  description: string
  layer_type: LayerType
  traffic_percentage: number
  priority: number
  is_active: boolean
  max_experiments?: number
  conflict_resolution: ConflictResolution
  metadata?: any
  created_at: string
}

export class LayeredExperimentsService {
  private baseUrl = '/experiments/layers'

  async createLayer(layerData: Partial<ExperimentLayer>): Promise<any> {
    const response = await apiClient.post(`${this.baseUrl}/`, layerData)
    return response.data
  }

  async listLayers(activeOnly: boolean = true): Promise<any> {
    const response = await apiClient.get(`${this.baseUrl}/`, {
      params: { active_only: activeOnly },
    })
    return response.data
  }

  async getLayer(layerId: string): Promise<ExperimentLayer> {
    const response = await apiClient.get(`${this.baseUrl}/${layerId}`)
    return response.data
  }

  async updateLayer(
    layerId: string,
    layerData: Partial<ExperimentLayer>
  ): Promise<any> {
    const response = await apiClient.put(
      `${this.baseUrl}/${layerId}`,
      layerData
    )
    return response.data
  }

  async deleteLayer(layerId: string): Promise<any> {
    const response = await apiClient.delete(`${this.baseUrl}/${layerId}`)
    return response.data
  }

  async assignExperimentToLayer(
    layerId: string,
    experimentId: string
  ): Promise<any> {
    const response = await apiClient.post(
      `${this.baseUrl}/${layerId}/experiments`,
      {
        experiment_id: experimentId,
      }
    )
    return response.data
  }

  async removeExperimentFromLayer(
    layerId: string,
    experimentId: string
  ): Promise<any> {
    const response = await apiClient.delete(
      `${this.baseUrl}/${layerId}/experiments/${experimentId}`
    )
    return response.data
  }

  async getLayerExperiments(layerId: string): Promise<any> {
    const response = await apiClient.get(
      `${this.baseUrl}/${layerId}/experiments`
    )
    return response.data
  }

  async getConflicts(layerId?: string): Promise<any> {
    const params = layerId ? { layer_id: layerId } : {}
    const response = await apiClient.get(`${this.baseUrl}/conflicts`, {
      params,
    })
    return response.data
  }

  async resolveConflict(conflictId: string, resolution: string): Promise<any> {
    const response = await apiClient.post(
      `${this.baseUrl}/conflicts/${conflictId}/resolve`,
      {
        resolution,
      }
    )
    return response.data
  }

  async validateLayerConfiguration(
    layerData: Partial<ExperimentLayer>
  ): Promise<any> {
    const response = await apiClient.post(`${this.baseUrl}/validate`, layerData)
    return response.data
  }

  async getSystemMetrics(): Promise<any> {
    const response = await apiClient.get(`${this.baseUrl}/metrics`)
    return response.data
  }
}

export const layeredExperimentsService = new LayeredExperimentsService()
