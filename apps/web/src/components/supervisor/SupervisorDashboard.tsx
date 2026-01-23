/**
 * Supervisor监控仪表板主组件
 * 提供完整的Supervisor系统监控和管理界面
 */

import React, { useEffect, useState } from 'react'
import { useSupervisorStore } from '../../stores/supervisorStore'
import { SupervisorStatus } from './SupervisorStatus'
import { TaskList } from './TaskList'
import { DecisionHistory } from './DecisionHistory'
import { AgentMetrics } from './AgentMetrics'
import { SupervisorConfig } from './SupervisorConfig'
import { TaskSubmissionForm } from './TaskSubmissionForm'

interface SupervisorDashboardProps {
  supervisorId?: string
  className?: string
}

type TabType = 'overview' | 'tasks' | 'decisions' | 'metrics' | 'config'

export const SupervisorDashboard: React.FC<SupervisorDashboardProps> = ({
  supervisorId = 'main_supervisor',
  className = '',
}) => {
  const [activeTab, setActiveTab] = useState<TabType>('overview')
  const [showTaskForm, setShowTaskForm] = useState(false)

  const {
    currentSupervisorId,
    setSupervisorId,
    refreshAll,
    autoRefresh,
    setAutoRefresh,
    refreshInterval,
    setRefreshInterval,
    error,
    clearError,
  } = useSupervisorStore()

  // 初始化Supervisor ID
  useEffect(() => {
    if (supervisorId && supervisorId !== currentSupervisorId) {
      setSupervisorId(supervisorId)
    }
  }, [supervisorId, currentSupervisorId, setSupervisorId])

  // 初始加载数据
  useEffect(() => {
    if (currentSupervisorId) {
      refreshAll()
    }
  }, [currentSupervisorId, refreshAll])

  // 智能自动刷新逻辑 - 只在页面可见且用户激活时启用
  useEffect(() => {
    let intervalId: ReturnType<typeof setTimeout> | null = null

    // 只有在页面可见、用户明确启用自动刷新、且有supervisor ID时才启用轮询
    if (autoRefresh && currentSupervisorId && !document.hidden) {
      intervalId = setInterval(() => {
        // 检查页面是否仍然可见
        if (!document.hidden) {
          refreshAll()
        }
      }, refreshInterval)
    }

    // 页面可见性变化处理
    const handleVisibilityChange = () => {
      if (document.hidden && intervalId) {
        clearInterval(intervalId)
        intervalId = null
      } else if (!document.hidden && autoRefresh && currentSupervisorId) {
        intervalId = setInterval(() => {
          if (!document.hidden) {
            refreshAll()
          }
        }, refreshInterval)
      }
    }

    document.addEventListener('visibilitychange', handleVisibilityChange)

    return () => {
      if (intervalId) {
        clearInterval(intervalId)
      }
      document.removeEventListener('visibilitychange', handleVisibilityChange)
    }
  }, [autoRefresh, currentSupervisorId, refreshAll, refreshInterval])

  const tabs = [
    { id: 'overview' as TabType, name: '概览', icon: '📊' },
    { id: 'tasks' as TabType, name: '任务', icon: '📋' },
    { id: 'decisions' as TabType, name: '决策', icon: '🧠' },
    { id: 'metrics' as TabType, name: '指标', icon: '📈' },
    { id: 'config' as TabType, name: '配置', icon: '⚙️' },
  ]

  const handleRefreshIntervalChange = (
    event: React.ChangeEvent<HTMLSelectElement>
  ) => {
    const newInterval = parseInt(event.target.value)
    setRefreshInterval(newInterval)
  }

  const renderTabContent = () => {
    switch (activeTab) {
      case 'overview':
        return <SupervisorStatus />
      case 'tasks':
        return <TaskList />
      case 'decisions':
        return <DecisionHistory />
      case 'metrics':
        return <AgentMetrics />
      case 'config':
        return <SupervisorConfig />
      default:
        return <SupervisorStatus />
    }
  }

  return (
    <div className={`supervisor-dashboard ${className}`}>
      {/* 头部工具栏 */}
      <div className="dashboard-header bg-white shadow-sm border-b p-4 mb-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <h1 className="text-2xl font-bold text-gray-900">
              Supervisor 监控面板
            </h1>
            <div className="flex items-center space-x-2 text-sm text-gray-600">
              <span>Supervisor ID:</span>
              <code className="bg-gray-100 px-2 py-1 rounded">
                {currentSupervisorId}
              </code>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            {/* 刷新控制 */}
            <div className="flex items-center space-x-2">
              <label className="flex items-center space-x-1">
                <input
                  type="checkbox"
                  name="autoRefresh"
                  checked={autoRefresh}
                  onChange={e => setAutoRefresh(e.target.checked)}
                  className="rounded"
                />
                <span className="text-sm text-gray-600">自动刷新</span>
              </label>

              <select
                value={refreshInterval}
                onChange={handleRefreshIntervalChange}
                disabled={!autoRefresh}
                name="refreshInterval"
                className="text-sm border border-gray-300 rounded px-2 py-1"
              >
                <option value={5000}>5秒</option>
                <option value={10000}>10秒</option>
                <option value={30000}>30秒</option>
                <option value={60000}>1分钟</option>
              </select>
            </div>

            {/* 操作按钮 */}
            <button
              onClick={() => refreshAll()}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
            >
              手动刷新
            </button>

            <button
              onClick={() => setShowTaskForm(true)}
              className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors"
            >
              提交任务
            </button>
          </div>
        </div>

        {/* 错误提示 */}
        {error && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-md">
            <div className="flex justify-between items-center">
              <span className="text-red-700">{error}</span>
              <button
                onClick={clearError}
                className="text-red-500 hover:text-red-700"
              >
                ✕
              </button>
            </div>
          </div>
        )}
      </div>

      {/* 标签页导航 */}
      <div className="tab-navigation mb-6">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <span className="mr-2">{tab.icon}</span>
                {tab.name}
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* 标签页内容 */}
      <div className="tab-content">{renderTabContent()}</div>

      {/* 任务提交表单模态框 */}
      {showTaskForm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full max-h-screen overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold">提交新任务</h2>
              <button
                onClick={() => setShowTaskForm(false)}
                className="text-gray-500 hover:text-gray-700"
              >
                ✕
              </button>
            </div>
            <TaskSubmissionForm
              onSubmit={() => setShowTaskForm(false)}
              onCancel={() => setShowTaskForm(false)}
            />
          </div>
        </div>
      )}
    </div>
  )
}

export default SupervisorDashboard
