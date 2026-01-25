import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios'
import { ApiResponse } from '../types'
import { FRONTEND_TIMEOUT_CONSTANTS } from '../constants/timeout'
import {
  HttpError,
  normalizeHttpErrorMessage,
  extractApiErrorMessage,
  API_PREFIX,
  normalizeApiBaseUrl,
} from '../utils/apiBase'
import {
  clearStoredTokens,
  getStoredAuthToken,
  getStoredRefreshToken,
  setStoredTokens,
} from '../utils/authStorage'

type RefreshableRequestConfig = AxiosRequestConfig & {
  _retry?: boolean
  skipAuth?: boolean
  skipAuthRefresh?: boolean
  preserveEnvelope?: boolean
}

class ApiClient {
  private client: AxiosInstance
  private refreshTokenPromise?: Promise<string | null>

  constructor() {
    const baseUrl = normalizeApiBaseUrl(import.meta.env.VITE_API_BASE_URL || '')
    this.client = axios.create({
      baseURL: `${baseUrl}${API_PREFIX}`,
      timeout: FRONTEND_TIMEOUT_CONSTANTS.API_CLIENT_TIMEOUT_MS,
      withCredentials: true,
      headers: {
        'Content-Type': 'application/json',
      },
    })

    this.setupInterceptors()
  }

  get defaults() {
    return this.client.defaults
  }

  private getAccessToken() {
    return getStoredAuthToken()
  }

  private getRefreshToken() {
    return getStoredRefreshToken()
  }

  private storeTokens(accessToken: string, refreshToken?: string) {
    setStoredTokens(accessToken, refreshToken)
    this.client.defaults.headers.common['Authorization'] =
      `Bearer ${accessToken}`
  }

  private clearTokens() {
    clearStoredTokens()
    delete this.client.defaults.headers.common['Authorization']
  }

  private stripAuthHeader(config: RefreshableRequestConfig) {
    const defaultHeaders = this.client.defaults.headers.common as {
      delete?: (key: string) => void
      [key: string]: unknown
    }
    if (typeof defaultHeaders.delete === 'function') {
      defaultHeaders.delete('Authorization')
      defaultHeaders.delete('authorization')
    } else {
      delete defaultHeaders.Authorization
      delete defaultHeaders.authorization
    }

    const headers = config.headers
    if (!headers) return
    const headerBag = headers as {
      delete?: (key: string) => void
    }
    if (typeof headerBag.delete === 'function') {
      headerBag.delete('Authorization')
      headerBag.delete('authorization')
      return
    }
    const mutableHeaders = headers as Record<string, unknown>
    delete mutableHeaders.Authorization
    delete mutableHeaders.authorization
    config.headers = mutableHeaders
  }

  private applyAuthHeader(config: RefreshableRequestConfig) {
    if (config.skipAuth) {
      this.stripAuthHeader(config)
      return
    }
    const token = this.getAccessToken()
    if (!token) {
      this.stripAuthHeader(config)
      return
    }

    const headers = config.headers ?? {}
    const headerBag = headers as {
      get?: (key: string) => string | null | undefined
      set?: (key: string, value: string) => void
    }

    if (typeof headerBag.set === 'function') {
      if (!headerBag.get?.('Authorization')) {
        headerBag.set('Authorization', `Bearer ${token}`)
      }
      return
    }

    const mutableHeaders = headers as Record<string, unknown>
    if (
      !('Authorization' in mutableHeaders) &&
      !('authorization' in mutableHeaders)
    ) {
      mutableHeaders.Authorization = `Bearer ${token}`
    }
    config.headers = mutableHeaders
  }

  private async refreshAccessToken(): Promise<string | null> {
    const refreshToken = this.getRefreshToken()
    if (!refreshToken) return null
    if (!this.refreshTokenPromise) {
      const refreshConfig: RefreshableRequestConfig = {
        skipAuth: true,
        skipAuthRefresh: true,
      }
      this.refreshTokenPromise = this.client
        .post(
          this.normalizeUrl('/auth/refresh'),
          { refresh_token: refreshToken },
          refreshConfig
        )
        .then(response => {
          const payload = response.data as {
            access_token?: string
            refresh_token?: string
          }
          if (!payload?.access_token) return null
          this.storeTokens(payload.access_token, payload.refresh_token)
          return payload.access_token
        })
        .catch(error => {
          const status = error?.response?.status
          if (status === 401 || status === 403) {
            this.clearTokens()
          }
          throw error
        })
        .finally(() => {
          this.refreshTokenPromise = undefined
        })
    }
    return this.refreshTokenPromise
  }

  private normalizeUrl(url?: string) {
    if (!url) return ''
    const trimmed = url.trim()
    if (!trimmed) return ''
    if (/^https?:\/\//i.test(trimmed) || trimmed.startsWith('//')) {
      return trimmed
    }
    const withoutPrefix = trimmed.replace(/^\/?api\/v1\/?/i, '')
    if (withoutPrefix.startsWith('/')) {
      return withoutPrefix.slice(1)
    }
    return withoutPrefix
  }

  private unwrapApiResponse<T>(payload: unknown): T | unknown {
    if (!payload || typeof payload !== 'object') return payload
    if (!('success' in payload)) return payload
    const apiPayload = payload as ApiResponse<T>
    if (!apiPayload.success) return payload
    if (!('data' in apiPayload) || apiPayload.data === undefined) return payload
    return apiPayload.data
  }

  private setupInterceptors() {
    // 请求拦截器
    this.client.interceptors.request.use(
      config => {
        if (typeof FormData !== 'undefined' && config.data instanceof FormData) {
          // 表单上传交给浏览器设置边界，避免错误的 Content-Type
          const headers = config.headers
          if (headers) {
            const headerMethods = headers as {
              delete?: (key: string) => void
            }
            if (typeof headerMethods.delete === 'function') {
              headerMethods.delete('Content-Type')
              headerMethods.delete('content-type')
            } else {
              const mutableHeaders = headers as Record<string, unknown>
              delete mutableHeaders['Content-Type']
              delete mutableHeaders['content-type']
            }
          }
        }
        this.applyAuthHeader(config as RefreshableRequestConfig)
        return config
      },
      error => {
        return Promise.reject(error)
      }
    )

    // 响应拦截器
    this.client.interceptors.response.use(
      (response: AxiosResponse<ApiResponse>) => {
        const data = response.data as ApiResponse | undefined
        if (
          data &&
          typeof data === 'object' &&
          'success' in data &&
          data.success === false
        ) {
          const errorMessage = extractApiErrorMessage(data, '请求失败')
          return Promise.reject(
            new HttpError(
              normalizeHttpErrorMessage(
                response.status,
                errorMessage,
                response.statusText
              ),
              response.status,
              response.statusText
            )
          )
        }
        const preserveEnvelope =
          (response.config as RefreshableRequestConfig | undefined)
            ?.preserveEnvelope ?? false
        if (!preserveEnvelope) {
          const unwrapped = this.unwrapApiResponse(response.data)
          if (unwrapped !== response.data) {
            response.data = unwrapped as ApiResponse
          }
        }
        return response
      },
      async error => {
        const status = error.response?.status
        const originalConfig = error.config as RefreshableRequestConfig | undefined
        if (
          status === 401 &&
          originalConfig &&
          !originalConfig._retry &&
          !originalConfig.skipAuthRefresh
        ) {
          originalConfig._retry = true
          const refreshed = await this.refreshAccessToken().catch(() => null)
          if (refreshed) {
            this.applyAuthHeader(originalConfig)
            return this.client.request(originalConfig)
          }
        }

        // 统一错误处理
        const errorMessage = extractApiErrorMessage(
          error.response?.data,
          error.message || '网络请求失败'
        )
        if (typeof status === 'number') {
          return Promise.reject(
            new HttpError(
              normalizeHttpErrorMessage(
                status,
                errorMessage,
                error.response?.statusText
              ),
              status,
              error.response?.statusText
            )
          )
        }
        return Promise.reject(new Error(errorMessage))
      }
    )
  }

  // 通用GET方法
  async get<T>(
    url: string,
    config?: RefreshableRequestConfig
  ): Promise<AxiosResponse<T>> {
    return this.client.get<T>(this.normalizeUrl(url), config)
  }

  // 通用POST方法
  async post<T>(
    url: string,
    data?: any,
    config?: RefreshableRequestConfig
  ): Promise<AxiosResponse<T>> {
    return this.client.post<T>(this.normalizeUrl(url), data, config)
  }

  // 通用PUT方法
  async put<T>(
    url: string,
    data?: any,
    config?: RefreshableRequestConfig
  ): Promise<AxiosResponse<T>> {
    return this.client.put<T>(this.normalizeUrl(url), data, config)
  }

  // 通用DELETE方法
  async delete<T>(
    url: string,
    config?: RefreshableRequestConfig
  ): Promise<AxiosResponse<T>> {
    return this.client.delete<T>(this.normalizeUrl(url), config)
  }
}

// 创建单例实例
export const apiClient = new ApiClient()
export default apiClient
