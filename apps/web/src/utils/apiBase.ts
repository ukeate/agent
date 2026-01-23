import { FRONTEND_TIMEOUT_CONSTANTS } from '../constants/timeout'
import {
  clearStoredTokens,
  getStoredAuthToken,
  getStoredRefreshToken,
  setStoredTokens,
} from './authStorage'

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

const API_PREFIX = '/api/v1'

const ABSOLUTE_URL = /^https?:\/\//i

export class HttpError extends Error {
  status: number
  statusText?: string

  constructor(message: string, status: number, statusText?: string) {
    super(message)
    this.name = 'HttpError'
    this.status = status
    this.statusText = statusText
  }
}

const isObjectRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === 'object' && value !== null
}

export const extractApiErrorMessage = (
  payload: unknown,
  fallback: string = '请求失败'
): string => {
  if (typeof payload === 'string') {
    const trimmed = payload.trim()
    return trimmed ? trimmed : fallback
  }
  if (!isObjectRecord(payload)) return fallback
  const message = payload.message
  if (typeof message === 'string' && message.trim()) return message.trim()
  const error = payload.error
  if (typeof error === 'string' && error.trim()) return error.trim()
  const detail = payload.detail
  if (typeof detail === 'string' && detail.trim()) return detail.trim()
  return fallback
}

const ensureApiSuccess = (payload: unknown, status?: number): void => {
  if (!isObjectRecord(payload) || !('success' in payload)) return
  if ((payload as { success?: boolean }).success === false) {
    const message = extractApiErrorMessage(payload, '请求失败')
    throw new HttpError(
      normalizeHttpErrorMessage(status ?? 200, message),
      status ?? 200
    )
  }
}

const STATUS_MESSAGE_MAP: Record<number, string> = {
  400: '请求参数错误',
  401: '身份验证失败，请重新登录',
  403: '权限不足，无法执行此操作',
  404: '请求资源不存在',
  408: '请求超时，请稍后重试',
  409: '请求冲突，请稍后重试',
  429: '请求过于频繁，请稍后重试',
}

const isGenericHttpMessage = (
  message: string,
  status?: number,
  statusText?: string
) => {
  const normalized = message.trim().toLowerCase()
  if (!normalized) return true
  const normalizedStatusText = (statusText || '').trim().toLowerCase()
  if (normalizedStatusText && normalized === normalizedStatusText) return true
  if (status && normalized === String(status)) return true
  if (
    status &&
    normalizedStatusText &&
    normalized === `${status} ${normalizedStatusText}`
  ) {
    return true
  }
  if (
    normalized.includes('request failed') ||
    normalized.includes('status code') ||
    normalized.includes('http error')
  ) {
    return true
  }
  if (
    normalized === 'unauthorized' ||
    normalized === 'forbidden' ||
    normalized === 'not found' ||
    normalized === 'internal server error'
  ) {
    return true
  }
  return false
}

export const normalizeHttpErrorMessage = (
  status?: number,
  message?: string,
  statusText?: string
): string => {
  const trimmed = (message || '').trim()
  if (!status) return trimmed || '网络请求失败'
  const mapped =
    STATUS_MESSAGE_MAP[status] ||
    (status >= 500 ? '服务器错误，请稍后重试' : '')
  if (!mapped) return trimmed || '网络请求失败'
  if (!trimmed) return mapped
  if (isGenericHttpMessage(trimmed, status, statusText)) return mapped
  return trimmed
}

const resolveApiUrl = (endpoint: string): string => {
  if (!endpoint) return `${API_BASE_URL}${API_PREFIX}`
  if (ABSOLUTE_URL.test(endpoint)) return endpoint
  const normalized = endpoint.startsWith('/') ? endpoint : `/${endpoint}`
  if (normalized.startsWith(API_PREFIX)) return `${API_BASE_URL}${normalized}`
  return `${API_BASE_URL}${API_PREFIX}${normalized}`
}

const resolveAuthHeaders = (headers?: HeadersInit): Headers => {
  const resolved = new Headers(headers)
  if (!resolved.has('Authorization')) {
    const token = getStoredAuthToken()
    if (token) resolved.set('Authorization', `Bearer ${token}`)
  }
  return resolved
}

export const buildApiUrl = (endpoint: string): string => {
  return resolveApiUrl(endpoint)
}

const extractErrorMessage = async (response: Response): Promise<string> => {
  const contentType = response.headers.get('content-type') || ''
  if (contentType.includes('application/json')) {
    try {
      const data = await response.json()
      return (
        data?.detail || data?.message || data?.error || JSON.stringify(data)
      )
    } catch {
      return `${response.status} ${response.statusText}`.trim()
    }
  }
  try {
    const text = await response.text()
    if (text) return text
  } catch {
    return `${response.status} ${response.statusText}`.trim()
  }
  return `${response.status} ${response.statusText}`.trim()
}

type ApiFetchInit = RequestInit & {
  timeoutMs?: number
  skipAuthRefresh?: boolean
}

let refreshPromise: Promise<string | null> | null = null

const refreshAccessToken = async (): Promise<string | null> => {
  const refreshToken = getStoredRefreshToken()
  if (!refreshToken) return null
  if (refreshPromise) return refreshPromise

  refreshPromise = (async () => {
    const response = await fetch(resolveApiUrl('/auth/refresh'), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      credentials: 'include',
      body: JSON.stringify({ refresh_token: refreshToken }),
    })
    if (!response.ok) {
      if (response.status === 401 || response.status === 403) {
        clearStoredTokens()
      }
      return null
    }
    const data = (await response.json()) as {
      access_token?: string
      refresh_token?: string
    }
    if (!data?.access_token) return null
    setStoredTokens(data.access_token, data.refresh_token)
    return data.access_token
  })().finally(() => {
    refreshPromise = null
  })

  return refreshPromise
}

export const apiFetch = async (
  endpoint: string,
  init?: ApiFetchInit
): Promise<Response> => {
  const { timeoutMs, skipAuthRefresh, ...fetchInit } = init ?? {}
  const resolvedTimeoutMs =
    timeoutMs ?? FRONTEND_TIMEOUT_CONSTANTS.API_CLIENT_TIMEOUT_MS
  const resolvedUrl = resolveApiUrl(endpoint)

  const attemptFetch = async (allowRetry: boolean): Promise<Response> => {
    const shouldTimeout = !fetchInit.signal && resolvedTimeoutMs > 0
    const controller = shouldTimeout ? new AbortController() : null
    const timer = controller
      ? setTimeout(() => controller.abort(), resolvedTimeoutMs)
      : null

    try {
      const response = await fetch(resolvedUrl, {
        ...fetchInit,
        headers: resolveAuthHeaders(fetchInit.headers),
        credentials: fetchInit.credentials ?? 'include',
        signal: controller?.signal ?? fetchInit.signal,
      })
      if (response.status === 401 && allowRetry && !skipAuthRefresh) {
        const refreshed = await refreshAccessToken().catch(() => null)
        if (refreshed) {
          return await attemptFetch(false)
        }
      }
      if (!response.ok) {
        const message = await extractErrorMessage(response)
        throw new HttpError(
          normalizeHttpErrorMessage(
            response.status,
            message,
            response.statusText
          ),
          response.status,
          response.statusText
        )
      }
      return response
    } catch (error) {
      if (error instanceof Error) {
        if (error instanceof HttpError) throw error
        if (error.name === 'AbortError') {
          if (controller) throw new Error('请求超时')
          throw error
        }
        const lowerMessage = error.message.toLowerCase()
        if (
          lowerMessage.includes('network') ||
          lowerMessage.includes('failed to fetch') ||
          lowerMessage.includes('fetch')
        ) {
          throw new Error('网络连接失败')
        }
        throw error
      }
      throw new Error('网络请求失败')
    } finally {
      if (timer) clearTimeout(timer)
    }
  }

  return attemptFetch(true)
}

export const apiFetchJson = async <T>(
  endpoint: string,
  init?: ApiFetchInit
): Promise<T> => {
  const response = await apiFetch(endpoint, init)
  if (response.status === 204) return null as T
  const payload = await response.json()
  ensureApiSuccess(payload, response.status)
  return payload as T
}

export const buildWsUrl = (endpoint: string): string => {
  const httpUrl = buildApiUrl(endpoint)
  if (httpUrl.startsWith('https://')) {
    return `wss://${httpUrl.slice('https://'.length)}`
  }
  if (httpUrl.startsWith('http://')) {
    return `ws://${httpUrl.slice('http://'.length)}`
  }
  if (httpUrl.startsWith('/')) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    return `${protocol}//${window.location.host}${httpUrl}`
  }
  return httpUrl
}
