import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios'
import { ApiResponse, ChatRequest, ChatResponse } from '../types'
import { FRONTEND_TIMEOUT_CONSTANTS } from '../constants/timeout'

class ApiClient {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: '/api/v1',
      timeout: FRONTEND_TIMEOUT_CONSTANTS.API_CLIENT_TIMEOUT_MS,
      headers: {
        'Content-Type': 'application/json',
      },
    })

    this.setupInterceptors()
  }

  private setupInterceptors() {
    // 请求拦截器
    this.client.interceptors.request.use(
      (config) => {
        console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`, config.data)
        return config
      },
      (error) => {
        console.error('[API] Request error:', error)
        return Promise.reject(error)
      }
    )

    // 响应拦截器
    this.client.interceptors.response.use(
      (response: AxiosResponse<ApiResponse>) => {
        console.log(`[API] Response:`, response.data)
        return response
      },
      (error) => {
        console.error('[API] Response error:', error)
        
        // 统一错误处理
        const errorMessage = error.response?.data?.error || 
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
      const response = await this.client.request<ApiResponse<T>>(config)
      
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
      console.error('[API] Request failed:', error)
      throw error
    }
  }

  // 通用GET方法
  async get<T>(url: string, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.client.get<T>(url, config)
  }

  // 通用POST方法
  async post<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.client.post<T>(url, data, config)
  }

  // 通用PUT方法
  async put<T>(url: string, data?: any, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.client.put<T>(url, data, config)
  }

  // 通用DELETE方法
  async delete<T>(url: string, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.client.delete<T>(url, config)
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

  // 流式聊天接口 (Server-Sent Events)
  async sendMessageStream(
    request: ChatRequest,
    onMessage: (data: any) => void,
    onError: (error: Error) => void,
    onComplete: () => void
  ): Promise<void> {
    try {
      const response = await fetch('/api/v1/agent/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'text/event-stream',
        },
        body: JSON.stringify({ ...request, stream: true }),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const reader = response.body?.getReader()
      if (!reader) {
        throw new Error('无法创建流读取器')
      }

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        
        if (done) {
          onComplete()
          break
        }

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.trim() === '') continue
          
          if (line.startsWith('data: ')) {
            const dataContent = line.slice(6).trim()
            
            // 检查是否是流结束标记
            if (dataContent === '[DONE]') {
              onComplete()
              break
            }
            
            try {
              const data = JSON.parse(dataContent)
              onMessage(data)
            } catch (error) {
              console.warn('[API] 无法解析SSE数据:', line)
            }
          }
        }
      }
    } catch (error) {
      console.error('[API] Stream error:', error)
      onError(error as Error)
    }
  }
}

// 创建单例实例
export const apiClient = new ApiClient()
export default apiClient