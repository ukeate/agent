import { useCallback, useEffect, useRef, useState } from 'react'
import {
  healthService,
  type HealthCheckResult,
  type HealthStatus,
} from '../services/healthService'

type DisplayStatus = HealthStatus | 'unknown'

type HealthMonitorOptions = {
  intervalMs?: number
  detailed?: boolean
  pauseWhenHidden?: boolean
}

const isBrowserOnline = () => {
  if (typeof navigator === 'undefined') return true
  return navigator.onLine
}

const isDocumentVisible = () => {
  if (typeof document === 'undefined') return true
  return document.visibilityState === 'visible'
}

export const useHealthMonitor = (options: HealthMonitorOptions = {}) => {
  const {
    intervalMs = 30000,
    detailed = false,
    pauseWhenHidden = true,
  } = options
  const [data, setData] = useState<HealthCheckResult | null>(null)
  const [status, setStatus] = useState<DisplayStatus>('unknown')
  const [timestamp, setTimestamp] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const unsubscribeRef = useRef<(() => void) | null>(null)

  const applyHealth = useCallback((next?: HealthCheckResult | null) => {
    const fallbackTimestamp = new Date().toISOString()
    if (!next) {
      setStatus('unhealthy')
      setTimestamp(fallbackTimestamp)
      setError('获取健康状态失败')
      setLoading(false)
      setData(null)
      return
    }
    setData(next)
    setStatus(next.status ?? 'unhealthy')
    setTimestamp(next.timestamp || fallbackTimestamp)
    setError(next.error ?? null)
    setLoading(false)
  }, [])

  const markPaused = useCallback((message: string) => {
    setStatus('unknown')
    setError(message)
    setLoading(false)
  }, [])

  const stopMonitoring = useCallback(() => {
    if (unsubscribeRef.current) {
      unsubscribeRef.current()
      unsubscribeRef.current = null
    }
  }, [])

  const startMonitoring = useCallback(() => {
    if (unsubscribeRef.current) return
    if (!isBrowserOnline()) return
    if (pauseWhenHidden && !isDocumentVisible()) return
    setLoading(true)
    unsubscribeRef.current = healthService.startHealthMonitoring(
      intervalMs,
      health => {
        applyHealth(health)
      }
    )
  }, [applyHealth, intervalMs, pauseWhenHidden])

  const refresh = useCallback(async () => {
    if (!isBrowserOnline()) {
      stopMonitoring()
      markPaused('网络离线')
      return
    }
    if (pauseWhenHidden && !isDocumentVisible()) {
      stopMonitoring()
      markPaused('页面隐藏，已暂停监控')
      return
    }
    setLoading(true)
    try {
      const next = await healthService.getHealth(detailed)
      applyHealth(next)
    } catch {
      applyHealth(null)
    }
  }, [applyHealth, detailed, markPaused, pauseWhenHidden, stopMonitoring])

  useEffect(() => {
    if (typeof window === 'undefined') return
    if (!isBrowserOnline()) {
      markPaused('网络离线')
    } else if (pauseWhenHidden && !isDocumentVisible()) {
      markPaused('页面隐藏，已暂停监控')
    } else {
      startMonitoring()
    }

    const handleOnline = () => {
      setError(null)
      startMonitoring()
      refresh()
    }

    const handleOffline = () => {
      stopMonitoring()
      markPaused('网络离线')
    }

    const handleVisibilityChange = () => {
      if (!pauseWhenHidden) return
      if (isDocumentVisible()) {
        setError(null)
        startMonitoring()
        refresh()
      } else {
        stopMonitoring()
        markPaused('页面隐藏，已暂停监控')
      }
    }

    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)
    document.addEventListener('visibilitychange', handleVisibilityChange)

    return () => {
      stopMonitoring()
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
      document.removeEventListener('visibilitychange', handleVisibilityChange)
    }
  }, [markPaused, pauseWhenHidden, refresh, startMonitoring, stopMonitoring])

  return {
    data,
    status,
    timestamp,
    loading,
    error,
    refresh,
    setSnapshot: applyHealth,
  }
}

export default useHealthMonitor
