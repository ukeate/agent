import type { HealthStatus } from '../services/healthService'

export type HealthStatusInfo = {
  color: 'success' | 'warning' | 'error' | 'default'
  label: string
}

export const getHealthStatusInfo = (
  status: HealthStatus | 'unknown',
  error?: string | null
): HealthStatusInfo => {
  if (error) {
    if (error.includes('网络离线')) {
      return { color: 'warning', label: '网络离线' }
    }
    if (error.includes('页面隐藏')) {
      return { color: 'default', label: '已暂停' }
    }
  }
  if (status === 'healthy') return { color: 'success', label: '服务正常' }
  if (status === 'degraded') return { color: 'warning', label: '服务降级' }
  if (status === 'unhealthy') return { color: 'error', label: '服务异常' }
  return { color: 'default', label: '状态未知' }
}
