import React, { useState, useMemo } from 'react'
import { Card } from '../ui/card'
import { Badge } from '../ui/badge'
import { Button } from '../ui/button'
import { Alert } from '../ui/alert'

interface Anomaly {
  anomaly_id: string
  user_id: string
  event_type: string
  timestamp: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  confidence: number
  description: string
  anomaly_type: string
  detected_by: string[]
  context: Record<string, any>
  resolved: boolean
}

interface AnomalyDetectionProps {
  anomalies: Anomaly[]
}

export const AnomalyDetection: React.FC<AnomalyDetectionProps> = ({
  anomalies,
}) => {
  const [selectedSeverity, setSelectedSeverity] = useState<string>('all')
  const [selectedType, setSelectedType] = useState<string>('all')
  const [showResolved, setShowResolved] = useState<boolean>(false)
  const [selectedAnomaly, setSelectedAnomaly] = useState<Anomaly | null>(null)

  // 过滤和统计
  const filteredAnomalies = useMemo(() => {
    return anomalies.filter(anomaly => {
      if (selectedSeverity !== 'all' && anomaly.severity !== selectedSeverity)
        return false
      if (selectedType !== 'all' && anomaly.anomaly_type !== selectedType)
        return false
      if (!showResolved && anomaly.resolved) return false
      return true
    })
  }, [anomalies, selectedSeverity, selectedType, showResolved])

  // 统计数据
  const stats = useMemo(() => {
    const totalAnomalies = anomalies.length
    const unresolvedAnomalies = anomalies.filter(a => !a.resolved).length
    const criticalAnomalies = anomalies.filter(
      a => a.severity === 'critical'
    ).length
    const severityDistribution = anomalies.reduce(
      (acc, anomaly) => {
        acc[anomaly.severity] = (acc[anomaly.severity] || 0) + 1
        return acc
      },
      {} as Record<string, number>
    )

    return {
      totalAnomalies,
      unresolvedAnomalies,
      criticalAnomalies,
      severityDistribution,
    }
  }, [anomalies])

  // 获取严重程度颜色和图标
  const getSeverityStyle = (severity: string) => {
    const styles = {
      critical: { color: 'bg-red-100 text-red-800 border-red-200', icon: '🚨' },
      high: {
        color: 'bg-orange-100 text-orange-800 border-orange-200',
        icon: '⚠️',
      },
      medium: {
        color: 'bg-yellow-100 text-yellow-800 border-yellow-200',
        icon: '⚡',
      },
      low: { color: 'bg-blue-100 text-blue-800 border-blue-200', icon: 'ℹ️' },
    }
    return styles[severity as keyof typeof styles] || styles.low
  }

  // 获取置信度颜色
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return 'text-green-600'
    if (confidence >= 0.7) return 'text-yellow-600'
    return 'text-red-600'
  }

  // 格式化时间
  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleString('zh-CN')
  }

  // 获取检测方法标签颜色
  const getMethodColor = (method: string) => {
    const colors = {
      statistical: 'bg-blue-100 text-blue-800',
      isolation_forest: 'bg-green-100 text-green-800',
      local_outlier_factor: 'bg-purple-100 text-purple-800',
      one_class_svm: 'bg-orange-100 text-orange-800',
    }
    return colors[method as keyof typeof colors] || 'bg-gray-100 text-gray-800'
  }

  // 获取异常类型的唯一值
  const anomalyTypes = useMemo(() => {
    return Array.from(new Set(anomalies.map(a => a.anomaly_type)))
  }, [anomalies])

  return (
    <div className="space-y-6">
      {/* 控制面板 */}
      <Card className="p-6">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">异常检测</h3>
            <p className="text-sm text-gray-600">
              实时监控和识别用户行为中的异常模式
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-4">
            {/* 严重程度过滤 */}
            <div className="flex items-center space-x-2">
              <span className="text-sm text-gray-600">严重程度：</span>
              <select
                value={selectedSeverity}
                onChange={e => setSelectedSeverity(e.target.value)}
                className="text-sm border border-gray-300 rounded px-2 py-1"
              >
                <option value="all">全部</option>
                <option value="critical">严重</option>
                <option value="high">高</option>
                <option value="medium">中等</option>
                <option value="low">低</option>
              </select>
            </div>

            {/* 类型过滤 */}
            <div className="flex items-center space-x-2">
              <span className="text-sm text-gray-600">类型：</span>
              <select
                value={selectedType}
                onChange={e => setSelectedType(e.target.value)}
                className="text-sm border border-gray-300 rounded px-2 py-1"
              >
                <option value="all">全部</option>
                {anomalyTypes.map(type => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
            </div>

            {/* 显示已解决 */}
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={showResolved}
                onChange={e => setShowResolved(e.target.checked)}
                className="rounded border-gray-300"
              />
              <span className="text-sm text-gray-600">显示已解决</span>
            </label>
          </div>
        </div>
      </Card>

      {/* 统计概览 */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-gray-900">
              {stats.totalAnomalies}
            </p>
            <p className="text-sm text-gray-600">总异常数</p>
          </div>
        </Card>

        <Card className="p-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-red-600">
              {stats.unresolvedAnomalies}
            </p>
            <p className="text-sm text-gray-600">未解决异常</p>
          </div>
        </Card>

        <Card className="p-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-orange-600">
              {stats.criticalAnomalies}
            </p>
            <p className="text-sm text-gray-600">严重异常</p>
          </div>
        </Card>

        <Card className="p-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-blue-600">
              {filteredAnomalies.length}
            </p>
            <p className="text-sm text-gray-600">当前筛选</p>
          </div>
        </Card>
      </div>

      {/* 严重程度分布 */}
      <Card className="p-6">
        <h4 className="font-semibold text-gray-900 mb-4">严重程度分布</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {Object.entries(stats.severityDistribution).map(
            ([severity, count]) => {
              const style = getSeverityStyle(severity)
              return (
                <div
                  key={severity}
                  className={`p-3 rounded-md border ${style.color}`}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-lg font-bold">{count}</p>
                      <p className="text-sm capitalize">{severity}</p>
                    </div>
                    <span className="text-xl">{style.icon}</span>
                  </div>
                </div>
              )
            }
          )}
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* 异常列表 */}
        <Card className="p-6">
          <h4 className="font-semibold text-gray-900 mb-4">
            异常列表 ({filteredAnomalies.length})
          </h4>

          <div className="space-y-4 max-h-96 overflow-y-auto">
            {filteredAnomalies.map(anomaly => {
              const severityStyle = getSeverityStyle(anomaly.severity)

              return (
                <div
                  key={anomaly.anomaly_id}
                  className={`p-4 border rounded-md cursor-pointer transition-colors ${
                    selectedAnomaly?.anomaly_id === anomaly.anomaly_id
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                  onClick={() => setSelectedAnomaly(anomaly)}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-1">
                        <span className="text-lg">{severityStyle.icon}</span>
                        <Badge className={severityStyle.color}>
                          {anomaly.severity}
                        </Badge>
                        {anomaly.resolved && (
                          <Badge variant="secondary">已解决</Badge>
                        )}
                      </div>
                      <p className="text-sm font-medium text-gray-900">
                        {anomaly.description}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">
                        用户: {anomaly.user_id} | 事件: {anomaly.event_type}
                      </p>
                    </div>
                    <div className="text-right ml-4">
                      <p
                        className={`text-sm font-medium ${getConfidenceColor(anomaly.confidence)}`}
                      >
                        置信度: {(anomaly.confidence * 100).toFixed(1)}%
                      </p>
                      <p className="text-xs text-gray-500">
                        {formatTime(anomaly.timestamp)}
                      </p>
                    </div>
                  </div>

                  {/* 检测方法标签 */}
                  <div className="flex flex-wrap gap-1 mt-2">
                    {anomaly.detected_by.map(method => (
                      <Badge
                        key={method}
                        className={`text-xs ${getMethodColor(method)}`}
                      >
                        {method}
                      </Badge>
                    ))}
                  </div>
                </div>
              )
            })}

            {filteredAnomalies.length === 0 && (
              <div className="text-center py-8 text-gray-500">
                <p>没有找到符合条件的异常</p>
                <p className="text-sm mt-1">尝试调整筛选条件</p>
              </div>
            )}
          </div>
        </Card>

        {/* 异常详情 */}
        <Card className="p-6">
          <h4 className="font-semibold text-gray-900 mb-4">异常详情</h4>

          {selectedAnomaly ? (
            <div className="space-y-6">
              {/* 基本信息 */}
              <div>
                <div className="flex items-center space-x-2 mb-3">
                  <span className="text-2xl">
                    {getSeverityStyle(selectedAnomaly.severity).icon}
                  </span>
                  <Badge
                    className={getSeverityStyle(selectedAnomaly.severity).color}
                  >
                    {selectedAnomaly.severity}
                  </Badge>
                  {selectedAnomaly.resolved && (
                    <Badge variant="secondary">已解决</Badge>
                  )}
                </div>

                <h5 className="font-medium text-gray-900 mb-2">
                  {selectedAnomaly.description}
                </h5>

                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <span className="text-gray-500">用户ID:</span>
                    <p className="font-medium">{selectedAnomaly.user_id}</p>
                  </div>
                  <div>
                    <span className="text-gray-500">事件类型:</span>
                    <p className="font-medium">{selectedAnomaly.event_type}</p>
                  </div>
                  <div>
                    <span className="text-gray-500">异常类型:</span>
                    <p className="font-medium">
                      {selectedAnomaly.anomaly_type}
                    </p>
                  </div>
                  <div>
                    <span className="text-gray-500">检测时间:</span>
                    <p className="font-medium">
                      {formatTime(selectedAnomaly.timestamp)}
                    </p>
                  </div>
                </div>
              </div>

              {/* 置信度和检测方法 */}
              <div>
                <h5 className="text-sm font-medium text-gray-700 mb-3">
                  检测信息
                </h5>
                <div className="bg-gray-50 p-4 rounded-md">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-sm text-gray-600">置信度</span>
                    <span
                      className={`font-bold ${getConfidenceColor(selectedAnomaly.confidence)}`}
                    >
                      {(selectedAnomaly.confidence * 100).toFixed(1)}%
                    </span>
                  </div>

                  <div className="mb-3">
                    <span className="text-sm text-gray-600 block mb-2">
                      检测方法
                    </span>
                    <div className="flex flex-wrap gap-2">
                      {selectedAnomaly.detected_by.map(method => (
                        <Badge
                          key={method}
                          className={`text-xs ${getMethodColor(method)}`}
                        >
                          {method}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* 上下文信息 */}
              {Object.keys(selectedAnomaly.context).length > 0 && (
                <div>
                  <h5 className="text-sm font-medium text-gray-700 mb-3">
                    上下文信息
                  </h5>
                  <div className="bg-gray-50 p-4 rounded-md">
                    <div className="space-y-2">
                      {Object.entries(selectedAnomaly.context).map(
                        ([key, value]) => (
                          <div
                            key={key}
                            className="flex justify-between text-sm"
                          >
                            <span className="text-gray-600 capitalize">
                              {key.replace(/_/g, ' ')}:
                            </span>
                            <span className="font-medium text-gray-900">
                              {typeof value === 'object'
                                ? JSON.stringify(value)
                                : String(value)}
                            </span>
                          </div>
                        )
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* 建议操作 */}
              <div>
                <h5 className="text-sm font-medium text-gray-700 mb-3">
                  建议操作
                </h5>
                <div className="space-y-2">
                  {selectedAnomaly.severity === 'critical' && (
                    <Alert variant="destructive" className="text-sm">
                      严重异常需要立即处理，建议深入调查用户行为和系统状态。
                    </Alert>
                  )}

                  {selectedAnomaly.confidence >= 0.9 && (
                    <Alert variant="warning" className="text-sm">
                      高置信度异常，建议记录并监控类似模式。
                    </Alert>
                  )}

                  {selectedAnomaly.detected_by.length > 2 && (
                    <Alert variant="default" className="text-sm">
                      多种算法都检测到此异常，说明异常模式明显。
                    </Alert>
                  )}
                </div>
              </div>

              {/* 操作按钮 */}
              <div className="flex space-x-2 pt-4 border-t">
                <Button size="sm" variant="default">
                  标记为已解决
                </Button>
                <Button size="sm" variant="outline">
                  忽略此异常
                </Button>
                <Button size="sm" variant="outline">
                  查看相关事件
                </Button>
              </div>
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              <span className="text-4xl mb-4 block">🔍</span>
              <p>选择左侧的异常查看详细信息</p>
            </div>
          )}
        </Card>
      </div>
    </div>
  )
}
