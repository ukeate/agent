// 安全监控面板组件

import React, { useState, useEffect } from 'react'
import { Card } from '../ui/card'
import { Button } from '../ui/button'
import { Badge } from '../ui/badge'
import { Alert } from '../ui/alert'
import { securityApi } from '../../services/securityApi'

import { logger } from '../../utils/logger'
interface SecurityStats {
  total_requests: number
  blocked_requests: number
  active_threats: number
  api_keys_count: number
  high_risk_events: number
  medium_risk_events: number
  low_risk_events: number
}

interface SecurityAlert {
  id: string
  alert_type: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  description: string
  timestamp: string
  status: 'active' | 'investigating' | 'resolved' | 'false_positive'
}

export const SecurityDashboard: React.FC = () => {
  const [stats, setStats] = useState<SecurityStats | null>(null)
  const [alerts, setAlerts] = useState<SecurityAlert[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadSecurityData()
    const interval = setInterval(loadSecurityData, 10000) // 每10秒刷新
    return () => clearInterval(interval)
  }, [])

  const loadSecurityData = async () => {
    try {
      const [statsData, alertsData] = await Promise.all([
        securityApi.getSecurityStats(),
        securityApi.getSecurityAlerts(),
      ])
      setStats(statsData)
      setAlerts(alertsData)
      setError(null)
    } catch (err) {
      setError('加载安全数据失败')
      logger.error('加载安全数据失败:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleResolveAlert = async (alertId: string) => {
    try {
      await securityApi.resolveAlert(alertId)
      await loadSecurityData()
    } catch (err) {
      logger.error('处理告警失败:', err)
    }
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-500'
      case 'high':
        return 'bg-orange-500'
      case 'medium':
        return 'bg-yellow-500'
      case 'low':
        return 'bg-blue-500'
      default:
        return 'bg-gray-500'
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    )
  }

  if (error) {
    return (
      <Alert variant="destructive">
        <p>{error}</p>
      </Alert>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">安全监控面板</h1>
        <Button onClick={loadSecurityData} variant="outline">
          刷新数据
        </Button>
      </div>

      {/* 统计卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="text-sm text-gray-500">总请求数</div>
          <div className="text-2xl font-bold">{stats?.total_requests || 0}</div>
        </Card>

        <Card className="p-4">
          <div className="text-sm text-gray-500">被阻止请求</div>
          <div className="text-2xl font-bold text-red-500">
            {stats?.blocked_requests || 0}
          </div>
        </Card>

        <Card className="p-4">
          <div className="text-sm text-gray-500">活跃威胁</div>
          <div className="text-2xl font-bold text-orange-500">
            {stats?.active_threats || 0}
          </div>
        </Card>

        <Card className="p-4">
          <div className="text-sm text-gray-500">API密钥数</div>
          <div className="text-2xl font-bold">{stats?.api_keys_count || 0}</div>
        </Card>
      </div>

      {/* 风险事件分布 */}
      <Card className="p-6">
        <h2 className="text-lg font-semibold mb-4">风险事件分布</h2>
        <div className="flex space-x-6">
          <div className="flex items-center">
            <div className="w-4 h-4 bg-red-500 rounded mr-2"></div>
            <span className="text-sm">
              高风险: {stats?.high_risk_events || 0}
            </span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 bg-yellow-500 rounded mr-2"></div>
            <span className="text-sm">
              中风险: {stats?.medium_risk_events || 0}
            </span>
          </div>
          <div className="flex items-center">
            <div className="w-4 h-4 bg-blue-500 rounded mr-2"></div>
            <span className="text-sm">
              低风险: {stats?.low_risk_events || 0}
            </span>
          </div>
        </div>
      </Card>

      {/* 安全告警 */}
      <Card className="p-6">
        <h2 className="text-lg font-semibold mb-4">最近的安全告警</h2>
        <div className="space-y-3">
          {alerts.length === 0 ? (
            <p className="text-gray-500">暂无安全告警</p>
          ) : (
            alerts.slice(0, 10).map(alert => (
              <div
                key={alert.id}
                className="border rounded-lg p-4 flex items-center justify-between"
              >
                <div className="flex items-center space-x-3">
                  <div
                    className={`w-3 h-3 rounded-full ${getSeverityColor(alert.severity)}`}
                  />
                  <div>
                    <div className="font-medium">{alert.description}</div>
                    <div className="text-sm text-gray-500">
                      类型: {alert.alert_type} | 时间:{' '}
                      {new Date(alert.timestamp).toLocaleString()}
                    </div>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <Badge
                    variant={
                      alert.status === 'active' ? 'destructive' : 'secondary'
                    }
                  >
                    {alert.status}
                  </Badge>
                  {alert.status === 'active' && (
                    <Button
                      size="sm"
                      onClick={() => handleResolveAlert(alert.id)}
                    >
                      标记已解决
                    </Button>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      </Card>
    </div>
  )
}
