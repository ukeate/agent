import apiClient from './apiClient'

export interface BatchJobSummary {
  batch_id: string
  status: string
  total_documents: number
  processed_documents: number
  successful_documents: number
  failed_documents: number
  progress: number
  created_at: string
  updated_at: string
}

class KnowledgeExtractionService {
  private baseUrl = '/knowledge'

  async listBatchJobs(limit: number = 100): Promise<BatchJobSummary[]> {
    const response = await apiClient.get(`${this.baseUrl}/batch`, {
      params: { limit },
    })
    return response.data.batches || []
  }
}

export const knowledgeExtractionService = new KnowledgeExtractionService()
