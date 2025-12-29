import { apiFetch, buildApiUrl } from '../utils/apiBase'
import { consumeSseJson } from '../utils/sse'
import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios'
import { ApiResponse, ChatRequest, ChatResponse } from '../types'
import { FRONTEND_TIMEOUT_CONSTANTS } from '../constants/timeout'

import { logger } from '../utils/logger'
class ApiClient {
  private client: AxiosInstance

  constructor() {
    const baseUrl = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')
    this.client = axios.create({
      baseURL: `${baseUrl}/api/v1`,
      timeout: FRONTEND_TIMEOUT_CONSTANTS.API_CLIENT_TIMEOUT_MS,
      headers: {
        'Content-Type': 'application/json',
      },
    })

    this.setupInterceptors()
  }

  get defaults() {
    return this.client.defaults
  }

  private normalizeUrl(url?: string) {
    if (!url) return ''
    if (url.startsWith('/api/v1/')) return url.slice('/api/v1'.length)
    if (url.startsWith('api/v1/')) return url.slice('api/v1'.length)
    return url
  }

  private setupInterceptors() {
    // 请求拦截器
    this.client.interceptors.request.use(
      (config) => {
        return config
      },
      (error) => {
        return Promise.reject(error)
      }
    )

    // 响应拦截器
    this.client.interceptors.response.use(
      (response: AxiosResponse<ApiResponse>) => {
        return response
      },
      (error) => {
        // 统一错误处理
        const errorMessage = error.response?.data?.detail ||
                            error.response?.data?.error || 
                            error.response?.data?.message || 
                            error.message || 
                            '网络请求失败'
        
        return Promise.reject(new Error(errorMessage))
      }
    )
  }

  // 通用请求方法
  private async request<T>(config: AxiosRequestConfig): Promise<T> {
    try {
      const normalizedUrl = this.normalizeUrl(config.url as string | undefined)
      const response = await this.client.request<ApiResponse<T>>({
        ...config,
        url: normalizedUrl,
      })
      
      // 检查是否是标准的ApiResponse格式
      if (response.data && typeof response.data === 'object' && 'success' in response.data) {
        if (!response.data.success) {
          throw new Error(response.data.error || '请求失败')
        }
        return response.data.data
      } else {
        // 直接返回数据（用于工作流等API）
        return response.data as T
      }
    } catch (error) {
      throw error
    }
  }

  // 通用GET方法
  async get<T>(url: string, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.client.get<T>(this.normalizeUrl(url), config)
  }

  // 通用POST方法
  async post<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.client.post<T>(this.normalizeUrl(url), data, config)
  }

  // 通用PUT方法
  async put<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.client.put<T>(this.normalizeUrl(url), data, config)
  }

  // 通用DELETE方法
  async delete<T>(url: string, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.client.delete<T>(this.normalizeUrl(url), config)
  }

  // 聊天接口
  async sendMessage(request: ChatRequest): Promise<ChatResponse> {
    return this.request<ChatResponse>({
      method: 'POST',
      url: '/agent/chat',
      data: request,
    })
  }

  // 任务执行接口
  async executeTask(task: string): Promise<any> {
    return this.request({
      method: 'POST',
      url: '/agent/task',
      data: { task },
    })
  }

  // 获取智能体状态
  async getAgentStatus(): Promise<any> {
    return this.request({
      method: 'GET',
      url: '/agent/status',
    })
  }

  // 工作流相关接口
  async createWorkflow(workflowData: any): Promise<any> {
    return this.request({
      method: 'POST',
      url: '/workflows',
      data: workflowData,
    })
  }

  async startWorkflow(workflowId: string, inputData?: any): Promise<any> {
    return this.request({
      method: 'POST',
      url: `/workflows/${workflowId}/start`,
      data: inputData ? { input_data: inputData } : {},
    })
  }

  async getWorkflowStatus(workflowId: string): Promise<any> {
    return this.request({
      method: 'GET',
      url: `/workflows/${workflowId}/status`,
    })
  }

  async listWorkflows(): Promise<any> {
    return this.request({
      method: 'GET',
      url: '/workflows',
    })
  }

  async controlWorkflow(workflowId: string, action: string): Promise<any> {
    return this.request({
      method: 'PUT',
      url: `/workflows/${workflowId}/control`,
      data: { action },
    })
  }

  // 健康监控相关接口
  async getHealthStatus(detailed: boolean = false): Promise<any> {
    return this.request({
      method: 'GET',
      url: `/health${detailed ? '?detailed=true' : ''}`,
    })
  }

  async getLivenessCheck(): Promise<any> {
    return this.request({
      method: 'GET',
      url: '/health/live',
    })
  }

  async getReadinessCheck(): Promise<any> {
    return this.request({
      method: 'GET',
      url: '/health/ready',
    })
  }

  async getSystemMetrics(): Promise<any> {
    return this.request({
      method: 'GET',
      url: '/health/metrics',
    })
  }

  async getActiveAlerts(): Promise<any> {
    return this.request({
      method: 'GET',
      url: '/health/alerts',
    })
  }

  // 流式聊天接口 (Server-Sent Events)
  async sendMessageStream(
    request: ChatRequest,
    onMessage: (data: any) => void,
    onError: (error: Error) => void,
    onComplete: () => void
  ): Promise<void> {
    try {
      const response = await apiFetch(buildApiUrl('/agent/chat'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        body: JSON.stringify({ ...request, stream: true }),
      })
      await consumeSseJson(
        response,
        (data) => {
          onMessage(data)
        },
        {
          onDone: onComplete,
          onParseError: () => {
            logger.warn('[API] 无法解析SSE数据')
          },
        }
      )
    } catch (error) {
      logger.error('[API] 流式请求失败:', error)
      onError(error as Error)
    }
  }
}

// 创建单例实例
export const apiClient = new ApiClient()
export default apiClient
