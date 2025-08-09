/**
 * 智能体指标组件
 * 显示智能体的负载、性能和可用性指标
 */

import React, { useEffect } from 'react'
import { useSupervisorStore } from '../../stores/supervisorStore'
import { AgentLoadMetrics } from '../../types/supervisor'

export const AgentMetrics: React.FC = () => {
  const {
    agentMetrics,
    status,
    loading,
    loadMetrics,
    loadStatus
  } = useSupervisorStore()

  useEffect(() => {
    if ((!agentMetrics || agentMetrics.length === 0) && !loading.metrics) {
      loadMetrics()
    }
    if (!status && !loading.status) {
      loadStatus()
    }
  }, [agentMetrics, status, loading.metrics, loading.status, loadMetrics, loadStatus])

  const formatPercentage = (value: number) => `${(value * 100).toFixed(1)}%`

  const getLoadColor = (load: number) => {
    if (load >= 0.8) return 'text-red-600 bg-red-50'
    if (load >= 0.5) return 'text-yellow-600 bg-yellow-50'
    return 'text-green-600 bg-green-50'
  }

  const getAvailabilityColor = (availability: number) => {
    if (availability >= 0.95) return 'text-green-600 bg-green-50'
    if (availability >= 0.8) return 'text-yellow-600 bg-yellow-50'
    return 'text-red-600 bg-red-50'
  }

  const formatDuration = (seconds: number) => {
    if (seconds < 60) return `${seconds.toFixed(1)}秒`
    if (seconds < 3600) return `${(seconds / 60).toFixed(1)}分钟`
    return `${(seconds / 3600).toFixed(1)}小时`
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString('zh-CN')
  }

  if (loading.metrics || loading.status) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-500">加载指标数据中...</div>
      </div>
    )
  }

  if (!agentMetrics || agentMetrics.length === 0) {
    return (
      <div className="text-center py-12 text-gray-500">
        暂无智能体指标数据
      </div>
    )
  }

  // 合并当前状态和历史指标数据
  const agentNames = status?.available_agents || []
  const metricsMap = new Map<string, AgentLoadMetrics>();
  (agentMetrics || []).forEach(metric => {
    metricsMap.set(metric.agent_name, metric)
  })

  return (
    <div className="agent-metrics space-y-6">
      {/* 概览卡片 */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">平均负载</p>
              <p className="text-2xl font-bold text-gray-900">
                {agentMetrics && agentMetrics.length > 0 
                  ? formatPercentage((agentMetrics || []).reduce((sum, m) => sum + m.current_load, 0) / agentMetrics.length)
                  : '0.0%'
                }
              </p>
            </div>
            <div className="text-blue-500 text-2xl">📊</div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">平均成功率</p>
              <p className="text-2xl font-bold text-gray-900">
                {agentMetrics && agentMetrics.length > 0 
                  ? formatPercentage((agentMetrics || []).reduce((sum, m) => sum + (m.success_rate || 0), 0) / agentMetrics.length)
                  : '0.0%'
                }
              </p>
            </div>
            <div className="text-green-500 text-2xl">✅</div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">平均可用性</p>
              <p className="text-2xl font-bold text-gray-900">
                {agentMetrics && agentMetrics.length > 0 
                  ? formatPercentage((agentMetrics || []).reduce((sum, m) => sum + m.availability_score, 0) / agentMetrics.length)
                  : '0.0%'
                }
              </p>
            </div>
            <div className="text-purple-500 text-2xl">🔄</div>
          </div>
        </div>
      </div>

      {/* 智能体详细指标 */}
      <div className="space-y-4">
        {agentNames.map((agentName) => {
          const metric = metricsMap.get(agentName)
          const currentLoad = status?.agent_loads[agentName] || 0

          return (
            <div key={agentName} className="bg-white p-6 rounded-lg shadow-sm border">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900">{agentName}</h3>
                <div className="flex space-x-3">
                  <span className={`px-3 py-1 text-sm font-medium rounded-full ${getLoadColor(currentLoad)}`}>
                    当前负载: {formatPercentage(currentLoad)}
                  </span>
                  {metric && (
                    <span className={`px-3 py-1 text-sm font-medium rounded-full ${getAvailabilityColor(metric.availability_score)}`}>
                      可用性: {formatPercentage(metric.availability_score)}
                    </span>
                  )}
                </div>
              </div>

              {metric ? (
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                  <div className="bg-gray-50 p-3 rounded-lg">
                    <div className="text-xs text-gray-500 mb-1">任务数量</div>
                    <div className="text-lg font-semibold text-gray-900">{metric.task_count}</div>
                  </div>
                  
                  <div className="bg-gray-50 p-3 rounded-lg">
                    <div className="text-xs text-gray-500 mb-1">平均任务时间</div>
                    <div className="text-lg font-semibold text-gray-900">
                      {metric.average_task_time ? formatDuration(metric.average_task_time) : '-'}
                    </div>
                  </div>
                  
                  <div className="bg-gray-50 p-3 rounded-lg">
                    <div className="text-xs text-gray-500 mb-1">成功率</div>
                    <div className="text-lg font-semibold text-green-600">
                      {metric.success_rate ? formatPercentage(metric.success_rate) : '-'}
                    </div>
                  </div>
                  
                  <div className="bg-gray-50 p-3 rounded-lg">
                    <div className="text-xs text-gray-500 mb-1">平均响应时间</div>
                    <div className="text-lg font-semibold text-blue-600">
                      {metric.response_time_avg ? formatDuration(metric.response_time_avg) : '-'}
                    </div>
                  </div>
                  
                  <div className="bg-gray-50 p-3 rounded-lg">
                    <div className="text-xs text-gray-500 mb-1">错误率</div>
                    <div className="text-lg font-semibold text-red-600">
                      {formatPercentage(metric.error_rate)}
                    </div>
                  </div>
                  
                  <div className="bg-gray-50 p-3 rounded-lg">
                    <div className="text-xs text-gray-500 mb-1">可用性评分</div>
                    <div className="text-lg font-semibold text-purple-600">
                      {formatPercentage(metric.availability_score)}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-gray-500 text-center py-8">
                  暂无该智能体的详细指标数据
                </div>
              )}

              {metric && (
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <div className="flex justify-between items-center text-sm text-gray-600">
                    <div>
                      统计窗口: {formatDate(metric.window_start)} - {formatDate(metric.window_end)}
                    </div>
                    <div>
                      最后更新: {formatDate(metric.updated_at)}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>

      {/* 负载分布图表 */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">智能体负载分布</h3>
        <div className="space-y-3">
          {agentNames.map((agentName) => {
            const currentLoad = status?.agent_loads[agentName] || 0
            return (
              <div key={agentName}>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-sm font-medium text-gray-700">{agentName}</span>
                  <span className="text-sm text-gray-600">{formatPercentage(currentLoad)}</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div
                    className={`h-3 rounded-full transition-all duration-300 ${
                      currentLoad >= 0.8
                        ? 'bg-red-500'
                        : currentLoad >= 0.5
                        ? 'bg-yellow-500'
                        : 'bg-green-500'
                    }`}
                    style={{ width: `${currentLoad * 100}%` }}
                  />
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* 性能趋势 */}
      {agentMetrics && agentMetrics.length > 0 && agentMetrics.some(m => m.success_rate !== undefined) && (
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">智能体性能对比</h3>
          <div className="space-y-4">
            {(agentMetrics || []).map((metric) => (
              <div key={metric.id} className="border rounded-lg p-4">
                <div className="flex justify-between items-center mb-2">
                  <h4 className="font-medium text-gray-900">{metric.agent_name}</h4>
                  <div className="text-sm text-gray-500">
                    {formatDate(metric.window_start)} - {formatDate(metric.window_end)}
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="text-gray-500">成功率:</span>
                    <div className={`font-medium ${
                      (metric.success_rate || 0) >= 0.9 ? 'text-green-600' : 
                      (metric.success_rate || 0) >= 0.7 ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                      {metric.success_rate ? formatPercentage(metric.success_rate) : '-'}
                    </div>
                  </div>
                  <div>
                    <span className="text-gray-500">错误率:</span>
                    <div className={`font-medium ${
                      metric.error_rate <= 0.1 ? 'text-green-600' : 
                      metric.error_rate <= 0.2 ? 'text-yellow-600' : 'text-red-600'
                    }`}>
                      {formatPercentage(metric.error_rate)}
                    </div>
                  </div>
                  <div>
                    <span className="text-gray-500">平均响应:</span>
                    <div className="font-medium text-blue-600">
                      {metric.response_time_avg ? formatDuration(metric.response_time_avg) : '-'}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

export default AgentMetrics