/**
 * Supervisor状态概览组件
 * 显示Supervisor的当前状态、智能体负载和关键统计指标
 */

import React, { useEffect } from 'react'
import { useSupervisorStore } from '../../stores/supervisorStore'

export const SupervisorStatus: React.FC = () => {
  const {
    status,
    stats,
    agentMetrics,
    loading,
    loadStatus,
    loadStats,
    loadMetrics,
  } = useSupervisorStore()

  useEffect(() => {
    // 确保数据已加载
    if (!status) loadStatus()
    if (!stats) loadStats()
    if (agentMetrics.length === 0) loadMetrics()
  }, [status, stats, agentMetrics.length, loadStatus, loadStats, loadMetrics])

  if (loading.status || loading.stats || loading.metrics) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-500">加载中...</div>
      </div>
    )
  }

  if (!status || !stats) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-500">暂无状态数据</div>
      </div>
    )
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'text-green-600 bg-green-50'
      case 'busy':
        return 'text-yellow-600 bg-yellow-50'
      case 'idle':
        return 'text-blue-600 bg-blue-50'
      case 'offline':
        return 'text-red-600 bg-red-50'
      default:
        return 'text-gray-600 bg-gray-50'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active':
        return '🟢'
      case 'busy':
        return '🟡'
      case 'idle':
        return '🔵'
      case 'offline':
        return '🔴'
      default:
        return '⚫'
    }
  }

  const formatPercentage = (value: number) => `${(value * 100).toFixed(1)}%`

  return (
    <div className="supervisor-status space-y-6">
      {/* 基本状态信息 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">
                Supervisor状态
              </p>
              <p className="text-2xl font-bold text-gray-900">
                {status.supervisor_name}
              </p>
            </div>
            <div
              className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(status.status)}`}
            >
              {getStatusIcon(status.status)} {status.status}
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">可用智能体</p>
              <p className="text-2xl font-bold text-gray-900">
                {status.available_agents.length}
              </p>
            </div>
            <div className="text-blue-500">👥</div>
          </div>
          <div className="mt-2">
            <p className="text-xs text-gray-500">
              {status.available_agents.join(', ')}
            </p>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">任务队列</p>
              <p className="text-2xl font-bold text-gray-900">
                {status.task_queue_length}
              </p>
            </div>
            <div className="text-purple-500">📋</div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">决策历史</p>
              <p className="text-2xl font-bold text-gray-900">
                {status.decision_history_count}
              </p>
            </div>
            <div className="text-green-500">🧠</div>
          </div>
        </div>
      </div>

      {/* 任务统计 */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">任务统计</h3>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">
              {stats.total_tasks}
            </div>
            <div className="text-sm text-gray-600">总任务数</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">
              {stats.completed_tasks}
            </div>
            <div className="text-sm text-gray-600">已完成</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-red-600">
              {stats.failed_tasks}
            </div>
            <div className="text-sm text-gray-600">失败</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-yellow-600">
              {stats.running_tasks}
            </div>
            <div className="text-sm text-gray-600">进行中</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-600">
              {stats.pending_tasks}
            </div>
            <div className="text-sm text-gray-600">待处理</div>
          </div>
        </div>
      </div>

      {/* 性能指标 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">性能指标</h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-gray-600">成功率</span>
              <span className="text-green-600 font-semibold">
                {formatPercentage(stats.success_rate)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">平均完成时间</span>
              <span className="text-blue-600 font-semibold">
                {Math.round(stats.average_completion_time)}秒
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-600">决策准确率</span>
              <span className="text-purple-600 font-semibold">
                {formatPercentage(stats.decision_accuracy)}
              </span>
            </div>
          </div>
        </div>

        {/* 智能体负载情况 */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            智能体负载
          </h3>
          <div className="space-y-3">
            {Object.entries(status.agent_loads).map(([agentName, load]) => (
              <div key={agentName}>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-gray-600 text-sm">{agentName}</span>
                  <span className="text-gray-900 font-medium">
                    {formatPercentage(load)}
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full transition-all duration-300 ${
                      load > 0.8
                        ? 'bg-red-500'
                        : load > 0.5
                          ? 'bg-yellow-500'
                          : 'bg-green-500'
                    }`}
                    style={{ width: `${load * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* 智能体利用率图表 */}
      {stats.agent_utilization &&
        Object.keys(stats.agent_utilization).length > 0 && (
          <div className="bg-white p-6 rounded-lg shadow-sm border">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
              智能体利用率
            </h3>
            <div className="space-y-3">
              {Object.entries(stats.agent_utilization).map(
                ([agentName, utilization]) => (
                  <div key={agentName}>
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-gray-600">{agentName}</span>
                      <span className="text-gray-900 font-medium">
                        {formatPercentage(utilization)}
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-3">
                      <div
                        className="bg-blue-500 h-3 rounded-full transition-all duration-300"
                        style={{ width: `${utilization * 100}%` }}
                      />
                    </div>
                  </div>
                )
              )}
            </div>
          </div>
        )}
    </div>
  )
}

export default SupervisorStatus
