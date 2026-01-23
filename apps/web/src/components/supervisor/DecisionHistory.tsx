/**
 * 决策历史组件
 * 显示Supervisor的决策过程和可解释性信息
 */

import React, { useEffect, useState } from 'react'
import { useSupervisorStore } from '../../stores/supervisorStore'
import { RoutingStrategy } from '../../types/supervisor'

export const DecisionHistory: React.FC = () => {
  const { decisions, pagination, loading, loadDecisions } = useSupervisorStore()

  const [strategyFilter, setStrategyFilter] = useState<RoutingStrategy | 'all'>(
    'all'
  )
  const [confidenceFilter, setConfidenceFilter] = useState<
    'all' | 'high' | 'medium' | 'low'
  >('all')
  const [successFilter, setSuccessFilter] = useState<
    'all' | 'success' | 'failed'
  >('all')

  useEffect(() => {
    if (decisions && decisions.length === 0 && !loading.decisions) {
      loadDecisions()
    }
  }, [decisions, loading.decisions, loadDecisions])

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'bg-green-100 text-green-800'
    if (confidence >= 0.6) return 'bg-yellow-100 text-yellow-800'
    return 'bg-red-100 text-red-800'
  }

  const getConfidenceLevel = (confidence: number) => {
    if (confidence >= 0.8) return '高'
    if (confidence >= 0.6) return '中'
    return '低'
  }

  const getSuccessIcon = (success?: boolean) => {
    if (success === undefined) return '⏳'
    return success ? '✅' : '❌'
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString('zh-CN')
  }

  const formatDuration = (seconds?: number) => {
    if (!seconds) return '-'
    if (seconds < 60) return `${seconds}秒`
    if (seconds < 3600) return `${Math.floor(seconds / 60)}分${seconds % 60}秒`
    return `${Math.floor(seconds / 3600)}时${Math.floor((seconds % 3600) / 60)}分`
  }

  const filteredDecisions = (decisions || []).filter(decision => {
    if (
      strategyFilter !== 'all' &&
      decision.routing_strategy !== strategyFilter
    )
      return false
    if (confidenceFilter !== 'all') {
      const level = getConfidenceLevel(decision.confidence_level)
      if (confidenceFilter === 'high' && level !== '高') return false
      if (confidenceFilter === 'medium' && level !== '中') return false
      if (confidenceFilter === 'low' && level !== '低') return false
    }
    if (successFilter !== 'all') {
      if (successFilter === 'success' && !decision.task_success) return false
      if (successFilter === 'failed' && decision.task_success !== false)
        return false
    }
    return true
  })

  const handlePageChange = (newPage: number) => {
    loadDecisions(newPage, pagination.decisions.pageSize)
  }

  if (loading.decisions && (!decisions || decisions.length === 0)) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-500">加载决策历史中...</div>
      </div>
    )
  }

  return (
    <div className="decision-history space-y-4">
      {/* 过滤器 */}
      <div className="bg-white p-4 rounded-lg shadow-sm border">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              路由策略
            </label>
            <select
              value={strategyFilter}
              onChange={e =>
                setStrategyFilter(e.target.value as RoutingStrategy | 'all')
              }
              className="w-full border border-gray-300 rounded-md px-3 py-2"
            >
              <option value="all">全部策略</option>
              <option value="round_robin">轮询</option>
              <option value="capability_based">基于能力</option>
              <option value="load_balanced">负载均衡</option>
              <option value="hybrid">混合策略</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              信心水平
            </label>
            <select
              value={confidenceFilter}
              onChange={e =>
                setConfidenceFilter(
                  e.target.value as 'all' | 'high' | 'medium' | 'low'
                )
              }
              className="w-full border border-gray-300 rounded-md px-3 py-2"
            >
              <option value="all">全部水平</option>
              <option value="high">高 (≥80%)</option>
              <option value="medium">中 (60-80%)</option>
              <option value="low">低 (&lt;60%)</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              执行结果
            </label>
            <select
              value={successFilter}
              onChange={e =>
                setSuccessFilter(e.target.value as 'all' | 'success' | 'failed')
              }
              className="w-full border border-gray-300 rounded-md px-3 py-2"
            >
              <option value="all">全部结果</option>
              <option value="success">成功</option>
              <option value="failed">失败</option>
            </select>
          </div>
        </div>
      </div>

      {/* 决策统计 */}
      <div className="bg-white p-4 rounded-lg shadow-sm border">
        <div className="flex justify-between items-center">
          <div>
            <span className="text-sm text-gray-600">
              显示 {filteredDecisions.length} 个决策，共{' '}
              {pagination.decisions.total} 个
            </span>
          </div>
          <div className="flex space-x-4 text-sm">
            {loading.decisions && (
              <span className="text-blue-600">🔄 刷新中...</span>
            )}
          </div>
        </div>
      </div>

      {/* 决策列表 */}
      <div className="space-y-4">
        {filteredDecisions.map(decision => (
          <div
            key={decision.id}
            className="bg-white p-6 rounded-lg shadow-sm border hover:shadow-md transition-shadow"
          >
            <div className="flex items-start justify-between mb-4">
              <div className="flex-1">
                <div className="flex items-center space-x-3 mb-2">
                  <h3 className="text-lg font-semibold text-gray-900">
                    决策 {decision.decision_id}
                  </h3>
                  <span
                    className={`px-2 py-1 text-xs font-medium rounded-full ${getConfidenceColor(decision.confidence_level)}`}
                  >
                    信心: {(decision.confidence_level * 100).toFixed(0)}%
                  </span>
                  <span className="px-2 py-1 text-xs font-medium rounded-full bg-blue-100 text-blue-800">
                    {decision.routing_strategy || '默认'}
                  </span>
                  <span className="text-lg">
                    {getSuccessIcon(decision.task_success)}
                  </span>
                </div>

                <p className="text-gray-600 mb-3">
                  {decision.task_description}
                </p>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm mb-4">
                  <div>
                    <span className="text-gray-500">分配智能体:</span>
                    <div className="font-medium text-blue-600">
                      {decision.assigned_agent}
                    </div>
                  </div>
                  <div>
                    <span className="text-gray-500">匹配评分:</span>
                    <div className="font-medium">
                      {(decision.match_score * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <span className="text-gray-500">完成时间:</span>
                    <div className="font-medium">
                      {formatDuration(decision.task_completion_time)}
                    </div>
                  </div>
                  <div>
                    <span className="text-gray-500">质量评分:</span>
                    <div className="font-medium">
                      {decision.quality_score?.toFixed(1) || '-'}
                    </div>
                  </div>
                </div>
              </div>

              <div className="ml-4 text-right text-sm text-gray-500">
                <div>{formatDate(decision.timestamp)}</div>
              </div>
            </div>

            {/* 分配理由 */}
            <div className="mb-4">
              <h4 className="text-sm font-semibold text-gray-800 mb-2">
                分配理由
              </h4>
              <p className="text-sm text-gray-600 bg-gray-50 p-3 rounded">
                {decision.assignment_reason}
              </p>
            </div>

            {/* 替代方案 */}
            {decision.alternatives_considered &&
              decision.alternatives_considered.length > 0 && (
                <div className="mb-4">
                  <h4 className="text-sm font-semibold text-gray-800 mb-2">
                    考虑的替代方案
                  </h4>
                  <div className="space-y-2">
                    {decision.alternatives_considered.map(
                      (alternative, index) => (
                        <div
                          key={index}
                          className="flex justify-between items-center text-sm bg-gray-50 p-2 rounded"
                        >
                          <span className="text-gray-700">
                            {alternative.agent}
                          </span>
                          <div className="flex space-x-4">
                            <span className="text-gray-600">
                              评分: {(alternative.score * 100).toFixed(1)}%
                            </span>
                            <span className="text-gray-500">
                              {alternative.reason}
                            </span>
                          </div>
                        </div>
                      )
                    )}
                  </div>
                </div>
              )}

            {/* 决策元数据 */}
            {decision.decision_metadata &&
              Object.keys(decision.decision_metadata).length > 0 && (
                <div className="mb-4">
                  <details>
                    <summary className="cursor-pointer text-sm font-medium text-gray-700">
                      决策元数据
                    </summary>
                    <pre className="mt-2 text-xs text-gray-600 bg-gray-50 p-3 rounded overflow-x-auto">
                      {JSON.stringify(decision.decision_metadata, null, 2)}
                    </pre>
                  </details>
                </div>
              )}

            {/* 路由元数据 */}
            {decision.routing_metadata &&
              Object.keys(decision.routing_metadata).length > 0 && (
                <div>
                  <details>
                    <summary className="cursor-pointer text-sm font-medium text-gray-700">
                      路由元数据
                    </summary>
                    <pre className="mt-2 text-xs text-gray-600 bg-gray-50 p-3 rounded overflow-x-auto">
                      {JSON.stringify(decision.routing_metadata, null, 2)}
                    </pre>
                  </details>
                </div>
              )}
          </div>
        ))}
      </div>

      {/* 分页 */}
      {pagination.decisions.totalPages > 1 && (
        <div className="bg-white p-4 rounded-lg shadow-sm border">
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-600">
              第 {pagination.decisions.page} 页，共{' '}
              {pagination.decisions.totalPages} 页
            </div>
            <div className="flex space-x-2">
              <button
                onClick={() => handlePageChange(pagination.decisions.page - 1)}
                disabled={pagination.decisions.page <= 1}
                className="px-3 py-1 text-sm border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                上一页
              </button>
              <button
                onClick={() => handlePageChange(pagination.decisions.page + 1)}
                disabled={
                  pagination.decisions.page >= pagination.decisions.totalPages
                }
                className="px-3 py-1 text-sm border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                下一页
              </button>
            </div>
          </div>
        </div>
      )}

      {filteredDecisions.length === 0 && !loading.decisions && (
        <div className="text-center py-12 text-gray-500">暂无决策历史数据</div>
      )}
    </div>
  )
}

export default DecisionHistory
