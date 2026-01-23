import React, { useState, useEffect } from 'react'
import { Card } from '../../components/ui/card'
import { Button } from '../../components/ui/button'
import { Badge } from '../../components/ui/badge'
import { Alert } from '../../components/ui/alert'
import { Progress } from '../../components/ui/progress'
import { behaviorAnalyticsService } from '../../services/behaviorAnalyticsService'

import { logger } from '../../utils/logger'
interface RealtimeMetrics {
  active_users: number
  events_per_second: number
  avg_response_time: number
  error_rate: number
  session_count: number
  queue_depth: number
  memory_usage: number
  cpu_usage: number
}

interface RealtimeEvent {
  event_id: string
  user_id: string
  session_id: string
  event_type: string
  timestamp: string
  processing_time: number
  status: 'processed' | 'failed' | 'pending'
}

interface SystemAlert {
  id: string
  type: 'warning' | 'error' | 'info'
  title: string
  message: string
  timestamp: string
  acknowledged: boolean
}

interface WebSocketStats {
  total_connections: number
  active_connections: number
  messages_sent: number
  messages_failed: number
  subscription_stats: Record<string, number>
}

export const RealTimeMonitorPage: React.FC = () => {
  const [metrics, setMetrics] = useState<RealtimeMetrics>({
    active_users: 0,
    events_per_second: 0,
    avg_response_time: 0,
    error_rate: 0,
    session_count: 0,
    queue_depth: 0,
    memory_usage: 0,
    cpu_usage: 0,
  })

  const [recentEvents, setRecentEvents] = useState<RealtimeEvent[]>([])
  const [systemAlerts, setSystemAlerts] = useState<SystemAlert[]>([])
  const [wsStats, setWsStats] = useState<WebSocketStats>({
    total_connections: 0,
    active_connections: 0,
    messages_sent: 0,
    messages_failed: 0,
    subscription_stats: {},
  })

  const [connected, setConnected] = useState(false)
  const [selectedMetricsPeriod, setSelectedMetricsPeriod] = useState<
    '1m' | '5m' | '15m' | '1h'
  >('5m')
  const [autoRefresh, setAutoRefresh] = useState(true)

  // WebSocket连接管理
  useEffect(() => {
    let ws: WebSocket

    if (autoRefresh) {
      const connectWebSocket = () => {
        try {
          ws = behaviorAnalyticsService.connectWebSocket(
            ['system_metrics', 'recent_events', 'system_alerts'],
            {
              onMessage: data => {
                if (data.type === 'system_metrics') {
                  setMetrics(data.payload)
                } else if (data.type === 'recent_events') {
                  setRecentEvents(prev => [data.payload, ...prev.slice(0, 49)])
                } else if (data.type === 'system_alert') {
                  setSystemAlerts(prev => [data.payload, ...prev.slice(0, 19)])
                } else if (data.type === 'websocket_stats') {
                  setWsStats(data.payload)
                }
              },
              onConnect: () => setConnected(true),
              onDisconnect: () => setConnected(false),
              onError: error => logger.error('WebSocket错误:', error),
            }
          )
        } catch (error) {
          logger.error('WebSocket连接失败:', error)
          setConnected(false)
        }
      }

      connectWebSocket()
    }

    return () => {
      if (ws) {
        ws.close()
      }
    }
  }, [autoRefresh])

  // 手动刷新数据
  const handleRefresh = async () => {
    try {
      const [metricsData, eventsData, alertsData, statsData] =
        await Promise.all([
          behaviorAnalyticsService.getRealtimeMetrics(),
          behaviorAnalyticsService.getRecentEvents({ limit: 50 }),
          behaviorAnalyticsService.getSystemAlerts(),
          behaviorAnalyticsService.getWebSocketStats(),
        ])

      setMetrics(metricsData)
      setRecentEvents(eventsData.events || [])
      setSystemAlerts(alertsData.alerts || [])
      setWsStats(statsData)
    } catch (error) {
      logger.error('刷新数据失败:', error)
    }
  }

  // 确认告警
  const acknowledgeAlert = (alertId: string) => {
    setSystemAlerts(prev =>
      prev.map(alert =>
        alert.id === alertId ? { ...alert, acknowledged: true } : alert
      )
    )
  }

  // 清除所有告警
  const clearAllAlerts = () => {
    setSystemAlerts(prev =>
      prev.map(alert => ({ ...alert, acknowledged: true }))
    )
  }

  // 格式化时间
  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString('zh-CN')
  }

  // 获取状态颜色
  const getStatusColor = (status: string) => {
    const colors = {
      processed: 'text-green-600',
      failed: 'text-red-600',
      pending: 'text-yellow-600',
    }
    return colors[status as keyof typeof colors] || 'text-gray-600'
  }

  // 获取告警类型样式
  const getAlertStyle = (type: string) => {
    const styles = {
      error: 'border-red-200 bg-red-50',
      warning: 'border-yellow-200 bg-yellow-50',
      info: 'border-blue-200 bg-blue-50',
    }
    return styles[type as keyof typeof styles] || styles.info
  }

  // 获取指标状态
  const getMetricStatus = (
    value: number,
    thresholds: { warning: number; critical: number }
  ) => {
    if (value >= thresholds.critical)
      return { status: 'critical', color: 'text-red-600' }
    if (value >= thresholds.warning)
      return { status: 'warning', color: 'text-yellow-600' }
    return { status: 'normal', color: 'text-green-600' }
  }

  return (
    <div className="p-6 space-y-6">
      {/* 页面标题 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">实时监控面板</h1>
          <p className="text-sm text-gray-600 mt-1">
            监控系统实时状态、事件流和性能指标
          </p>
        </div>
        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2">
            <Badge variant={connected ? 'success' : 'danger'}>
              {connected ? '🟢 已连接' : '🔴 已断开'}
            </Badge>
            <label className="flex items-center space-x-1">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={e => setAutoRefresh(e.target.checked)}
              />
              <span className="text-sm">自动刷新</span>
            </label>
          </div>
          <Button variant="outline" onClick={handleRefresh}>
            🔄 手动刷新
          </Button>
        </div>
      </div>

      {/* 连接状态警告 */}
      {!connected && (
        <Alert variant="warning">
          <p>
            ⚠️ 实时连接已断开，数据可能不是最新的。请检查网络连接或刷新页面。
          </p>
        </Alert>
      )}

      {/* 核心指标 */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">活跃用户</p>
              <p className="text-2xl font-bold text-blue-600">
                {metrics.active_users}
              </p>
            </div>
            <div className="p-2 bg-blue-50 rounded-full">
              <span className="text-2xl">👥</span>
            </div>
          </div>
          <div className="mt-2 flex items-center">
            <div
              className={`w-2 h-2 rounded-full mr-2 ${connected ? 'bg-green-500 animate-pulse' : 'bg-gray-400'}`}
            />
            <span className="text-xs text-gray-500">实时</span>
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">事件/秒</p>
              <p className="text-2xl font-bold text-green-600">
                {metrics.events_per_second.toFixed(1)}
              </p>
            </div>
            <div className="p-2 bg-green-50 rounded-full">
              <span className="text-2xl">⚡</span>
            </div>
          </div>
          <div className="mt-2">
            <Progress
              value={Math.min(100, (metrics.events_per_second / 100) * 100)}
              max={100}
              className="h-1"
            />
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">响应时间</p>
              <p
                className={`text-2xl font-bold ${getMetricStatus(metrics.avg_response_time, { warning: 200, critical: 500 }).color}`}
              >
                {metrics.avg_response_time.toFixed(0)}ms
              </p>
            </div>
            <div className="p-2 bg-purple-50 rounded-full">
              <span className="text-2xl">⏱️</span>
            </div>
          </div>
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">错误率</p>
              <p
                className={`text-2xl font-bold ${getMetricStatus(metrics.error_rate, { warning: 5, critical: 10 }).color}`}
              >
                {metrics.error_rate.toFixed(2)}%
              </p>
            </div>
            <div className="p-2 bg-red-50 rounded-full">
              <span className="text-2xl">🚨</span>
            </div>
          </div>
        </Card>
      </div>

      {/* 系统资源监控 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">系统资源</h3>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>CPU使用率</span>
                <span
                  className={
                    getMetricStatus(metrics.cpu_usage, {
                      warning: 70,
                      critical: 90,
                    }).color
                  }
                >
                  {metrics.cpu_usage.toFixed(1)}%
                </span>
              </div>
              <Progress value={metrics.cpu_usage} max={100} className="h-2" />
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>内存使用率</span>
                <span
                  className={
                    getMetricStatus(metrics.memory_usage, {
                      warning: 80,
                      critical: 95,
                    }).color
                  }
                >
                  {metrics.memory_usage.toFixed(1)}%
                </span>
              </div>
              <Progress
                value={metrics.memory_usage}
                max={100}
                className="h-2"
              />
            </div>
            <div>
              <div className="flex justify-between text-sm mb-1">
                <span>事件队列深度</span>
                <span>{metrics.queue_depth} 个</span>
              </div>
              <Progress
                value={Math.min(100, (metrics.queue_depth / 1000) * 100)}
                max={100}
                className="h-2"
              />
            </div>
          </div>
        </Card>

        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">WebSocket统计</h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="text-center p-3 bg-blue-50 rounded-md">
              <p className="text-xl font-bold text-blue-600">
                {wsStats.active_connections}
              </p>
              <p className="text-xs text-blue-800">活跃连接</p>
            </div>
            <div className="text-center p-3 bg-green-50 rounded-md">
              <p className="text-xl font-bold text-green-600">
                {wsStats.messages_sent}
              </p>
              <p className="text-xs text-green-800">已发送消息</p>
            </div>
            <div className="text-center p-3 bg-red-50 rounded-md">
              <p className="text-xl font-bold text-red-600">
                {wsStats.messages_failed}
              </p>
              <p className="text-xs text-red-800">失败消息</p>
            </div>
            <div className="text-center p-3 bg-purple-50 rounded-md">
              <p className="text-xl font-bold text-purple-600">
                {(
                  ((wsStats.messages_sent - wsStats.messages_failed) /
                    Math.max(1, wsStats.messages_sent)) *
                  100
                ).toFixed(1)}
                %
              </p>
              <p className="text-xs text-purple-800">成功率</p>
            </div>
          </div>

          {Object.keys(wsStats.subscription_stats || {}).length > 0 && (
            <div className="mt-4">
              <h4 className="text-sm font-medium text-gray-700 mb-2">
                订阅分布
              </h4>
              <div className="space-y-2">
                {Object.entries(wsStats.subscription_stats).map(
                  ([type, count]) => (
                    <div
                      key={type}
                      className="flex items-center justify-between text-sm"
                    >
                      <span>{type}</span>
                      <span className="font-medium">{count}</span>
                    </div>
                  )
                )}
              </div>
            </div>
          )}
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 实时事件流 */}
        <Card className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">实时事件流</h3>
            <Badge variant={connected ? 'success' : 'default'}>
              {recentEvents.length} 个事件
            </Badge>
          </div>

          <div className="space-y-2 max-h-96 overflow-y-auto">
            {recentEvents.map(event => (
              <div
                key={event.event_id}
                className="flex items-center justify-between p-2 bg-gray-50 rounded text-sm"
              >
                <div className="flex items-center space-x-2">
                  <div
                    className={`w-2 h-2 rounded-full ${
                      event.status === 'processed'
                        ? 'bg-green-500'
                        : event.status === 'failed'
                          ? 'bg-red-500'
                          : 'bg-yellow-500'
                    }`}
                  />
                  <span className="font-medium">{event.event_type}</span>
                  <span className="text-gray-500">
                    用户:{event.user_id.substring(0, 8)}
                  </span>
                </div>
                <div className="text-right">
                  <div className={`text-xs ${getStatusColor(event.status)}`}>
                    {event.processing_time}ms
                  </div>
                  <div className="text-xs text-gray-500">
                    {formatTime(event.timestamp)}
                  </div>
                </div>
              </div>
            ))}

            {recentEvents.length === 0 && (
              <div className="text-center py-8 text-gray-500">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500 mx-auto mb-2" />
                <p className="text-sm">等待事件数据...</p>
              </div>
            )}
          </div>
        </Card>

        {/* 系统告警 */}
        <Card className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold">系统告警</h3>
            <div className="flex space-x-2">
              <Badge variant="danger">
                {systemAlerts.filter(a => !a.acknowledged).length} 未确认
              </Badge>
              {systemAlerts.filter(a => !a.acknowledged).length > 0 && (
                <Button size="sm" variant="outline" onClick={clearAllAlerts}>
                  全部确认
                </Button>
              )}
            </div>
          </div>

          <div className="space-y-2 max-h-96 overflow-y-auto">
            {systemAlerts
              .filter(alert => !alert.acknowledged)
              .map(alert => (
                <div
                  key={alert.id}
                  className={`p-3 border rounded-md ${getAlertStyle(alert.type)}`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-1">
                        <span className="text-sm font-medium">
                          {alert.title}
                        </span>
                        <Badge
                          variant={
                            alert.type === 'error'
                              ? 'danger'
                              : alert.type === 'warning'
                                ? 'warning'
                                : 'default'
                          }
                        >
                          {alert.type}
                        </Badge>
                      </div>
                      <p className="text-xs text-gray-600">{alert.message}</p>
                      <p className="text-xs text-gray-500 mt-1">
                        {formatTime(alert.timestamp)}
                      </p>
                    </div>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => acknowledgeAlert(alert.id)}
                    >
                      确认
                    </Button>
                  </div>
                </div>
              ))}

            {systemAlerts.filter(a => !a.acknowledged).length === 0 && (
              <div className="text-center py-8 text-gray-500">
                <span className="text-4xl mb-2 block">✅</span>
                <p className="text-sm">暂无未确认告警</p>
              </div>
            )}
          </div>
        </Card>
      </div>
    </div>
  )
}

export default RealTimeMonitorPage
