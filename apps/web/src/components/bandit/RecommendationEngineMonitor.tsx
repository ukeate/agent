import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card'
import { Button } from '../ui/button'
import { Badge } from '../ui/badge'
import { Alert } from '../ui/alert'
import { banditRecommendationService } from '../../services/banditRecommendationService'

interface EngineStatus {
  status: string
  is_initialized: boolean
  timestamp: string
  engine_stats?: {
    total_requests: number
    cache_hits: number
    active_users: number
    cache_size: number
    algorithms: string[]
  }
}

interface EngineMonitorProps {
  refreshInterval?: number
  onStatusChange?: (status: EngineStatus) => void
}

const RecommendationEngineMonitor: React.FC<EngineMonitorProps> = ({
  refreshInterval = 5000,
  onStatusChange,
}) => {
  const [status, setStatus] = useState<EngineStatus | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [autoRefresh, setAutoRefresh] = useState(true)

  const fetchStatus = async () => {
    try {
      setLoading(true)
      setError(null)

      const healthResponse = await banditRecommendationService.getHealth()
      setStatus(healthResponse)

      if (onStatusChange) {
        onStatusChange(healthResponse)
      }
    } catch (err: any) {
      setError(`获取状态失败: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchStatus()
  }, [])

  useEffect(() => {
    if (!autoRefresh) return

    const interval = setInterval(fetchStatus, refreshInterval)
    return () => clearInterval(interval)
  }, [autoRefresh, refreshInterval])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'text-green-600 bg-green-50 border-green-200'
      case 'degraded':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200'
      case 'not_initialized':
        return 'text-gray-600 bg-gray-50 border-gray-200'
      default:
        return 'text-red-600 bg-red-50 border-red-200'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return '🟢'
      case 'degraded':
        return '🟡'
      case 'not_initialized':
        return '⚪'
      default:
        return '🔴'
    }
  }

  return (
    <div className="space-y-4">
      {/* 状态卡片 */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span>引擎状态监控</span>
            <div className="flex items-center space-x-2">
              <Button
                size="sm"
                variant="outline"
                onClick={() => setAutoRefresh(!autoRefresh)}
              >
                {autoRefresh ? '⏸️ 暂停' : '▶️ 启动'}自动刷新
              </Button>
              <Button size="sm" onClick={fetchStatus} disabled={loading}>
                {loading ? '刷新中...' : '🔄 手动刷新'}
              </Button>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {error && (
            <Alert className="border-red-200 bg-red-50 mb-4">
              <div className="text-red-800">{error}</div>
            </Alert>
          )}

          {status ? (
            <div className="space-y-4">
              {/* 基础状态信息 */}
              <div
                className={`p-4 rounded-lg border ${getStatusColor(status.status)}`}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <span className="text-2xl">
                      {getStatusIcon(status.status)}
                    </span>
                    <div>
                      <div className="font-semibold">
                        引擎状态: {status.status}
                      </div>
                      <div className="text-sm opacity-75">
                        初始化状态:{' '}
                        {status.is_initialized ? '已初始化' : '未初始化'}
                      </div>
                    </div>
                  </div>
                  <div className="text-sm opacity-75">
                    更新时间: {new Date(status.timestamp).toLocaleTimeString()}
                  </div>
                </div>

                {/* 引擎统计信息 */}
                {status.engine_stats && (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
                    <div className="bg-white bg-opacity-50 p-3 rounded">
                      <div className="text-lg font-semibold">
                        {status.engine_stats.total_requests}
                      </div>
                      <div className="text-sm">总请求数</div>
                    </div>

                    <div className="bg-white bg-opacity-50 p-3 rounded">
                      <div className="text-lg font-semibold">
                        {status.engine_stats.cache_hits}
                      </div>
                      <div className="text-sm">缓存命中</div>
                    </div>

                    <div className="bg-white bg-opacity-50 p-3 rounded">
                      <div className="text-lg font-semibold">
                        {status.engine_stats.active_users}
                      </div>
                      <div className="text-sm">活跃用户</div>
                    </div>

                    <div className="bg-white bg-opacity-50 p-3 rounded">
                      <div className="text-lg font-semibold">
                        {status.engine_stats.cache_size}
                      </div>
                      <div className="text-sm">缓存大小</div>
                    </div>
                  </div>
                )}
              </div>

              {/* 算法状态 */}
              {status.engine_stats?.algorithms && (
                <div>
                  <h4 className="font-medium mb-2">可用算法</h4>
                  <div className="flex flex-wrap gap-2">
                    {status.engine_stats.algorithms.map(algorithm => (
                      <Badge
                        key={algorithm}
                        variant="outline"
                        className="capitalize"
                      >
                        {algorithm.replace('_', ' ')}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* 性能指标 */}
              {status.engine_stats && (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="bg-blue-50 p-3 rounded-lg">
                    <div className="text-blue-800 font-medium">缓存效率</div>
                    <div className="text-2xl text-blue-600">
                      {status.engine_stats.total_requests > 0
                        ? (
                            (status.engine_stats.cache_hits /
                              status.engine_stats.total_requests) *
                            100
                          ).toFixed(1)
                        : '0.0'}
                      %
                    </div>
                    <div className="text-sm text-blue-700">
                      命中率: {status.engine_stats.cache_hits}/
                      {status.engine_stats.total_requests}
                    </div>
                  </div>

                  <div className="bg-green-50 p-3 rounded-lg">
                    <div className="text-green-800 font-medium">用户覆盖</div>
                    <div className="text-2xl text-green-600">
                      {status.engine_stats.active_users}
                    </div>
                    <div className="text-sm text-green-700">活跃用户数量</div>
                  </div>

                  <div className="bg-purple-50 p-3 rounded-lg">
                    <div className="text-purple-800 font-medium">系统负载</div>
                    <div className="text-2xl text-purple-600">
                      {status.engine_stats.cache_size < 1000
                        ? '低'
                        : status.engine_stats.cache_size < 5000
                          ? '中'
                          : '高'}
                    </div>
                    <div className="text-sm text-purple-700">
                      基于缓存大小评估
                    </div>
                  </div>
                </div>
              )}

              {/* 健康度评分 */}
              <div className="bg-gray-50 p-4 rounded-lg">
                <h4 className="font-medium mb-2">系统健康度评分</h4>
                <div className="flex items-center space-x-4">
                  {/* 健康度计算逻辑 */}
                  {(() => {
                    if (!status.is_initialized)
                      return (
                        <div className="flex items-center space-x-2">
                          <div className="w-20 h-4 bg-gray-200 rounded">
                            <div className="w-0 h-4 bg-gray-400 rounded"></div>
                          </div>
                          <span className="text-gray-600">
                            未初始化 (0/100)
                          </span>
                        </div>
                      )

                    let score = 0
                    if (status.status === 'healthy') score += 40
                    else if (status.status === 'degraded') score += 20

                    if (status.engine_stats) {
                      if (status.engine_stats.total_requests > 0) score += 30
                      if (status.engine_stats.algorithms.length > 0) score += 20
                      if (status.engine_stats.cache_hits > 0) score += 10
                    }

                    const scoreColor =
                      score >= 80
                        ? 'bg-green-500'
                        : score >= 60
                          ? 'bg-yellow-500'
                          : 'bg-red-500'

                    return (
                      <div className="flex items-center space-x-2">
                        <div className="w-20 h-4 bg-gray-200 rounded">
                          <div
                            className={`h-4 ${scoreColor} rounded`}
                            style={{ width: `${score}%` }}
                          ></div>
                        </div>
                        <span
                          className={`font-medium ${
                            score >= 80
                              ? 'text-green-600'
                              : score >= 60
                                ? 'text-yellow-600'
                                : 'text-red-600'
                          }`}
                        >
                          {score >= 80
                            ? '优秀'
                            : score >= 60
                              ? '良好'
                              : '需要关注'}{' '}
                          ({score}/100)
                        </span>
                      </div>
                    )
                  })()}
                </div>
              </div>
            </div>
          ) : (
            <div className="text-center text-gray-500 py-8">
              {loading ? '正在获取状态信息...' : '无法获取状态信息'}
            </div>
          )}
        </CardContent>
      </Card>

      {/* 自动刷新指示器 */}
      {autoRefresh && (
        <div className="text-center">
          <div className="inline-flex items-center space-x-2 text-sm text-gray-500">
            <div className="animate-pulse w-2 h-2 bg-blue-500 rounded-full"></div>
            <span>自动刷新中 (每{refreshInterval / 1000}秒)</span>
          </div>
        </div>
      )}
    </div>
  )
}

export default RecommendationEngineMonitor
