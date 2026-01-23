import React, { useState, useEffect } from 'react'
import { Card } from '../../components/ui/card'
import { Button } from '../../components/ui/button'
import { Input } from '../../components/ui/input'
import { Badge } from '../../components/ui/badge'
import { Progress } from '../../components/ui/progress'
import { behaviorAnalyticsService } from '../../services/behaviorAnalyticsService'

import { logger } from '../../utils/logger'
interface UserSession {
  session_id: string
  user_id: string
  start_time: string
  end_time?: string
  duration_seconds?: number
  event_count: number
  unique_event_types: number
  last_activity: string
  status: 'active' | 'inactive' | 'expired'
  metadata?: Record<string, any>
}

interface SessionFilter {
  user_id?: string
  status?: 'active' | 'inactive' | 'expired'
  start_time?: string
  end_time?: string
  min_duration?: number
  max_duration?: number
  min_events?: number
  limit?: number
  offset?: number
}

export const SessionManagePage: React.FC = () => {
  const [sessions, setSessions] = useState<UserSession[]>([])
  const [loading, setLoading] = useState(false)
  const [selectedSession, setSelectedSession] = useState<UserSession | null>(
    null
  )
  const [filter, setFilter] = useState<SessionFilter>({
    limit: 20,
    offset: 0,
  })
  const [totalSessions, setTotalSessions] = useState(0)
  const [sessionStats, setSessionStats] = useState({
    active_sessions: 0,
    avg_duration: 0,
    total_events: 0,
    unique_users: 0,
  })

  // 获取会话数据
  const fetchSessions = async () => {
    setLoading(true)
    try {
      const response = await behaviorAnalyticsService.getSessions(filter)
      setSessions(response.sessions || [])
      setTotalSessions(response.total || 0)

      // 获取统计数据
      const stats = await behaviorAnalyticsService.getSessionStats()
      setSessionStats(stats)
    } catch (error) {
      logger.error('获取会话数据失败:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchSessions()
  }, [filter])

  // 格式化持续时间
  const formatDuration = (seconds?: number) => {
    if (!seconds) return '未知'
    if (seconds < 60) return `${seconds}秒`
    if (seconds < 3600) return `${Math.floor(seconds / 60)}分钟`
    return `${(seconds / 3600).toFixed(1)}小时`
  }

  // 格式化时间
  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleString('zh-CN')
  }

  // 获取状态颜色
  const getStatusColor = (status: string) => {
    const colors = {
      active: 'bg-green-100 text-green-800',
      inactive: 'bg-yellow-100 text-yellow-800',
      expired: 'bg-gray-100 text-gray-800',
    }
    return colors[status as keyof typeof colors] || colors.expired
  }

  // 获取状态图标
  const getStatusIcon = (status: string) => {
    const icons = {
      active: '🟢',
      inactive: '🟡',
      expired: '⚫',
    }
    return icons[status as keyof typeof icons] || '⚫'
  }

  return (
    <div className="p-6 space-y-6">
      {/* 页面标题 */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">会话管理中心</h1>
          <p className="text-sm text-gray-600 mt-1">
            管理和监控用户会话，支持会话重放、状态管理和生命周期分析
          </p>
        </div>
        <div className="flex space-x-3">
          <Button variant="default" onClick={fetchSessions}>
            🔄 刷新数据
          </Button>
        </div>
      </div>

      {/* 统计卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-green-600">
              {sessionStats.active_sessions}
            </p>
            <p className="text-sm text-gray-600">活跃会话</p>
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-blue-600">
              {sessionStats.unique_users}
            </p>
            <p className="text-sm text-gray-600">独立用户</p>
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-purple-600">
              {formatDuration(sessionStats.avg_duration)}
            </p>
            <p className="text-sm text-gray-600">平均时长</p>
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-orange-600">
              {sessionStats.total_events}
            </p>
            <p className="text-sm text-gray-600">总事件数</p>
          </div>
        </Card>
      </div>

      {/* 筛选器 */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">会话筛选</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              用户ID
            </label>
            <Input
              placeholder="输入用户ID"
              value={filter.user_id || ''}
              onChange={e =>
                setFilter(prev => ({
                  ...prev,
                  user_id: e.target.value || undefined,
                }))
              }
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              会话状态
            </label>
            <select
              className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
              value={filter.status || ''}
              onChange={e =>
                setFilter(prev => ({
                  ...prev,
                  status: (e.target.value as any) || undefined,
                }))
              }
            >
              <option value="">全部状态</option>
              <option value="active">活跃</option>
              <option value="inactive">非活跃</option>
              <option value="expired">已过期</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              最小事件数
            </label>
            <Input
              type="number"
              placeholder="0"
              value={filter.min_events || ''}
              onChange={e =>
                setFilter(prev => ({
                  ...prev,
                  min_events: Number(e.target.value) || undefined,
                }))
              }
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              最小时长(秒)
            </label>
            <Input
              type="number"
              placeholder="0"
              value={filter.min_duration || ''}
              onChange={e =>
                setFilter(prev => ({
                  ...prev,
                  min_duration: Number(e.target.value) || undefined,
                }))
              }
            />
          </div>
        </div>
        <div className="mt-4 flex space-x-2">
          <Button onClick={fetchSessions}>🔍 应用筛选</Button>
          <Button
            variant="outline"
            onClick={() => setFilter({ limit: 20, offset: 0 })}
          >
            🔄 重置
          </Button>
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 会话列表 */}
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">会话列表</h3>

          {loading ? (
            <div className="text-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto"></div>
              <p className="mt-2 text-gray-600">加载中...</p>
            </div>
          ) : (
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {sessions.map(session => (
                <div
                  key={session.session_id}
                  className={`p-4 border rounded-md cursor-pointer transition-colors ${
                    selectedSession?.session_id === session.session_id
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                  onClick={() => setSelectedSession(session)}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <span className="text-lg">
                        {getStatusIcon(session.status)}
                      </span>
                      <Badge className={getStatusColor(session.status)}>
                        {session.status}
                      </Badge>
                    </div>
                    <div className="text-right text-sm text-gray-500">
                      {formatTime(session.last_activity)}
                    </div>
                  </div>

                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600">会话ID:</span>
                      <span className="font-mono">
                        {session.session_id.substring(0, 12)}...
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">用户:</span>
                      <span className="font-medium">{session.user_id}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">持续时长:</span>
                      <span>{formatDuration(session.duration_seconds)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">事件数:</span>
                      <span>{session.event_count} 个</span>
                    </div>
                  </div>

                  {session.duration_seconds && (
                    <div className="mt-2">
                      <div className="flex justify-between text-xs text-gray-500 mb-1">
                        <span>活跃度</span>
                        <span>
                          {Math.min(
                            100,
                            (session.event_count /
                              (session.duration_seconds / 60)) *
                              10
                          ).toFixed(0)}
                          %
                        </span>
                      </div>
                      <Progress
                        value={Math.min(
                          100,
                          (session.event_count /
                            (session.duration_seconds / 60)) *
                            10
                        )}
                        max={100}
                        className="h-2"
                      />
                    </div>
                  )}
                </div>
              ))}

              {sessions.length === 0 && !loading && (
                <div className="text-center py-8 text-gray-500">
                  <p>暂无会话数据</p>
                  <p className="text-sm mt-1">尝试调整筛选条件</p>
                </div>
              )}
            </div>
          )}
        </Card>

        {/* 会话详情 */}
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">会话详情</h3>

          {selectedSession ? (
            <div className="space-y-6">
              {/* 基本信息 */}
              <div>
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-2">
                    <span className="text-xl">
                      {getStatusIcon(selectedSession.status)}
                    </span>
                    <h4 className="font-medium">会话基本信息</h4>
                  </div>
                  <Badge className={getStatusColor(selectedSession.status)}>
                    {selectedSession.status}
                  </Badge>
                </div>

                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-500">会话ID:</span>
                    <p className="font-mono text-xs mt-1">
                      {selectedSession.session_id}
                    </p>
                  </div>
                  <div>
                    <span className="text-gray-500">用户ID:</span>
                    <p className="font-medium mt-1">
                      {selectedSession.user_id}
                    </p>
                  </div>
                  <div>
                    <span className="text-gray-500">开始时间:</span>
                    <p className="mt-1">
                      {formatTime(selectedSession.start_time)}
                    </p>
                  </div>
                  <div>
                    <span className="text-gray-500">最后活动:</span>
                    <p className="mt-1">
                      {formatTime(selectedSession.last_activity)}
                    </p>
                  </div>
                </div>
              </div>

              {/* 统计信息 */}
              <div>
                <h5 className="font-medium mb-3">会话统计</h5>
                <div className="grid grid-cols-3 gap-4">
                  <div className="text-center p-3 bg-blue-50 rounded-md">
                    <p className="text-xl font-bold text-blue-600">
                      {selectedSession.event_count}
                    </p>
                    <p className="text-xs text-blue-800">总事件数</p>
                  </div>
                  <div className="text-center p-3 bg-green-50 rounded-md">
                    <p className="text-xl font-bold text-green-600">
                      {selectedSession.unique_event_types}
                    </p>
                    <p className="text-xs text-green-800">事件类型</p>
                  </div>
                  <div className="text-center p-3 bg-purple-50 rounded-md">
                    <p className="text-xl font-bold text-purple-600">
                      {formatDuration(selectedSession.duration_seconds)}
                    </p>
                    <p className="text-xs text-purple-800">会话时长</p>
                  </div>
                </div>
              </div>

              {/* 元数据 */}
              {selectedSession.metadata &&
                Object.keys(selectedSession.metadata).length > 0 && (
                  <div>
                    <h5 className="font-medium mb-3">元数据</h5>
                    <div className="bg-gray-50 p-3 rounded-md">
                      <pre className="text-xs overflow-x-auto">
                        {JSON.stringify(selectedSession.metadata, null, 2)}
                      </pre>
                    </div>
                  </div>
                )}

              {/* 操作按钮 */}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              <span className="text-4xl mb-4 block">👆</span>
              <p>选择左侧的会话查看详细信息</p>
            </div>
          )}
        </Card>
      </div>
    </div>
  )
}

export default SessionManagePage
