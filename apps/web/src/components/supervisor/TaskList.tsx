/**
 * Supervisor任务列表组件
 * 显示和管理Supervisor分配的任务
 */

import React, { useEffect, useState } from 'react'
import { useSupervisorStore } from '../../stores/supervisorStore'
import { TaskStatus, TaskType, TaskPriority } from '../../types/supervisor'

export const TaskList: React.FC = () => {
  const {
    tasks,
    selectedTask,
    pagination,
    loading,
    loadTasks,
    loadTaskDetails
  } = useSupervisorStore()

  const [statusFilter, setStatusFilter] = useState<TaskStatus | 'all'>('all')
  const [typeFilter, setTypeFilter] = useState<TaskType | 'all'>('all')
  const [priorityFilter, setPriorityFilter] = useState<TaskPriority | 'all'>('all')
  const [expandedTask, setExpandedTask] = useState<string | null>(null)

  useEffect(() => {
    if (tasks.length === 0 && !loading.tasks) {
      loadTasks()
    }
  }, [tasks.length, loading.tasks, loadTasks])

  const getStatusColor = (status: TaskStatus) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800'
      case 'running':
        return 'bg-blue-100 text-blue-800'
      case 'failed':
        return 'bg-red-100 text-red-800'
      case 'cancelled':
        return 'bg-gray-100 text-gray-800'
      case 'pending':
      default:
        return 'bg-yellow-100 text-yellow-800'
    }
  }

  const getStatusIcon = (status: TaskStatus) => {
    switch (status) {
      case 'completed':
        return '✅'
      case 'running':
        return '🔄'
      case 'failed':
        return '❌'
      case 'cancelled':
        return '⏹️'
      case 'pending':
      default:
        return '⏳'
    }
  }

  const getPriorityColor = (priority: TaskPriority) => {
    switch (priority) {
      case 'urgent':
        return 'bg-red-100 text-red-800'
      case 'high':
        return 'bg-orange-100 text-orange-800'
      case 'medium':
        return 'bg-blue-100 text-blue-800'
      case 'low':
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  const getTypeIcon = (type: TaskType) => {
    switch (type) {
      case 'code_generation':
        return '💻'
      case 'code_review':
        return '🔍'
      case 'documentation':
        return '📝'
      case 'analysis':
        return '📊'
      case 'planning':
        return '📋'
      default:
        return '📄'
    }
  }

  const formatDuration = (seconds?: number) => {
    if (!seconds) return '-'
    if (seconds < 60) return `${seconds}秒`
    if (seconds < 3600) return `${Math.floor(seconds / 60)}分${seconds % 60}秒`
    return `${Math.floor(seconds / 3600)}时${Math.floor((seconds % 3600) / 60)}分`
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString('zh-CN')
  }

  const filteredTasks = tasks.filter(task => {
    if (statusFilter !== 'all' && task.status !== statusFilter) return false
    if (typeFilter !== 'all' && task.task_type !== typeFilter) return false
    if (priorityFilter !== 'all' && task.priority !== priorityFilter) return false
    return true
  })

  const handlePageChange = (newPage: number) => {
    loadTasks(newPage, pagination.tasks.pageSize)
  }

  const handleToggleTaskDetails = async (taskId: string) => {
    if (expandedTask === taskId) {
      // 如果是关闭当前展开的任务，清理状态
      setExpandedTask(null)
    } else {
      // 展开新任务，加载详情
      setExpandedTask(taskId)
      await loadTaskDetails(taskId)
    }
  }

  const renderTaskOutput = (task: any) => {
    if (!selectedTask || selectedTask.id !== task.id) return null
    
    const outputData = selectedTask.output_data
    
    // 根据任务状态显示不同的信息
    if (!outputData) {
      // 对于pending和running状态的任务，显示状态信息而不是"暂无输出数据"
      if (selectedTask.status === 'pending' || selectedTask.status === 'running') {
        return (
          <div className="space-y-3">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div>
                <span className="text-gray-500">执行状态:</span>
                <div className="font-medium">
                  {selectedTask.status === 'pending' ? (
                    <span className="text-yellow-600">⏳ 等待执行</span>
                  ) : selectedTask.status === 'running' ? (
                    <span className="text-blue-600">🔄 执行中</span>
                  ) : (
                    <span className="text-gray-600">➖ 未知</span>
                  )}
                </div>
              </div>
              <div>
                <span className="text-gray-500">分配智能体:</span>
                <div className="font-medium">{selectedTask.assigned_agent_name || '未分配'}</div>
              </div>
              <div>
                <span className="text-gray-500">任务类型:</span>
                <div className="font-medium">{selectedTask.task_type || '-'}</div>
              </div>
            </div>
            <div className="text-gray-500 italic">任务尚未开始执行，暂无输出数据</div>
          </div>
        )
      }
      return <div className="text-gray-500 italic">暂无输出数据</div>
    }
    
    return (
      <div className="space-y-3">
        {/* 基本结果信息 */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div>
            <span className="text-gray-500">执行状态:</span>
            <div className="font-medium">
              {outputData.success ? (
                <span className="text-green-600">✅ 成功</span>
              ) : outputData.success === false ? (
                <span className="text-red-600">❌ 失败</span>
              ) : (
                <span className="text-gray-600">➖ 执行中</span>
              )}
            </div>
          </div>
          <div>
            <span className="text-gray-500">执行智能体:</span>
            <div className="font-medium">{outputData.agent_name || '-'}</div>
          </div>
          <div>
            <span className="text-gray-500">任务类型:</span>
            <div className="font-medium">{outputData.task_type || '-'}</div>
          </div>
        </div>

        {/* 输出内容 */}
        {outputData.result && (
          <div>
            <h4 className="font-medium text-gray-700 mb-2">任务输出内容:</h4>
            <div className="bg-gray-50 border rounded-md p-3">
              {typeof outputData.result === 'string' ? (
                <div className="whitespace-pre-wrap text-sm">{outputData.result}</div>
              ) : outputData.result.design ? (
                <div className="space-y-2">
                  <div className="text-sm">
                    <span className="font-medium">设计类型:</span> {outputData.result.design_type}
                  </div>
                  <div className="text-sm">
                    <span className="font-medium">需求长度:</span> {outputData.result.requirements_length}
                  </div>
                  <div className="mt-3">
                    <span className="font-medium text-gray-700">设计内容:</span>
                    <div className="mt-1 whitespace-pre-wrap text-sm text-gray-600 max-h-96 overflow-y-auto">
                      {outputData.result.design}
                    </div>
                  </div>
                </div>
              ) : (
                <pre className="text-sm text-gray-600 overflow-x-auto whitespace-pre-wrap">
                  {JSON.stringify(outputData.result, null, 2)}
                </pre>
              )}
            </div>
          </div>
        )}

        {/* 错误信息 */}
        {outputData.error && (
          <div>
            <h4 className="font-medium text-red-700 mb-2">错误信息:</h4>
            <div className="bg-red-50 border border-red-200 rounded-md p-3 text-red-800 text-sm">
              {outputData.error}
            </div>
          </div>
        )}
      </div>
    )
  }

  if (loading.tasks && tasks.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-500">加载任务中...</div>
      </div>
    )
  }

  return (
    <div className="task-list space-y-4">
      {/* 过滤器 */}
      <div className="bg-white p-4 rounded-lg shadow-sm border">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">状态过滤</label>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value as TaskStatus | 'all')}
              className="w-full border border-gray-300 rounded-md px-3 py-2"
            >
              <option value="all">全部状态</option>
              <option value="pending">待处理</option>
              <option value="running">进行中</option>
              <option value="completed">已完成</option>
              <option value="failed">失败</option>
              <option value="cancelled">已取消</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">类型过滤</label>
            <select
              value={typeFilter}
              onChange={(e) => setTypeFilter(e.target.value as TaskType | 'all')}
              className="w-full border border-gray-300 rounded-md px-3 py-2"
            >
              <option value="all">全部类型</option>
              <option value="code_generation">代码生成</option>
              <option value="code_review">代码审查</option>
              <option value="documentation">文档编写</option>
              <option value="analysis">分析</option>
              <option value="planning">规划</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">优先级过滤</label>
            <select
              value={priorityFilter}
              onChange={(e) => setPriorityFilter(e.target.value as TaskPriority | 'all')}
              className="w-full border border-gray-300 rounded-md px-3 py-2"
            >
              <option value="all">全部优先级</option>
              <option value="urgent">紧急</option>
              <option value="high">高</option>
              <option value="medium">中</option>
              <option value="low">低</option>
            </select>
          </div>
        </div>
      </div>

      {/* 任务统计 */}
      <div className="bg-white p-4 rounded-lg shadow-sm border">
        <div className="flex justify-between items-center">
          <div>
            <span className="text-sm text-gray-600">
              显示 {filteredTasks.length} 个任务，共 {pagination.tasks.total} 个
            </span>
          </div>
          <div className="flex space-x-4 text-sm">
            {loading.tasks && (
              <span className="text-blue-600">🔄 刷新中...</span>
            )}
          </div>
        </div>
      </div>

      {/* 任务列表 */}
      <div className="space-y-3">
        {filteredTasks.map((task) => (
          <div key={task.id} className="bg-white p-6 rounded-lg shadow-sm border hover:shadow-md transition-shadow">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center space-x-3 mb-2">
                  <span className="text-lg">{getTypeIcon(task.task_type)}</span>
                  <h3 className="text-lg font-semibold text-gray-900">{task.name}</h3>
                  <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(task.status)}`}>
                    {getStatusIcon(task.status)} {task.status}
                  </span>
                  <span className={`px-2 py-1 text-xs font-medium rounded-full ${getPriorityColor(task.priority)}`}>
                    {task.priority}
                  </span>
                </div>
                
                <p className="text-gray-600 mb-3">{task.description}</p>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-gray-500">分配智能体:</span>
                    <div className="font-medium">{task.assigned_agent_name || '未分配'}</div>
                  </div>
                  <div>
                    <span className="text-gray-500">复杂度评分:</span>
                    <div className="font-medium">{task.complexity_score?.toFixed(2) || '-'}</div>
                  </div>
                  <div>
                    <span className="text-gray-500">预估时间:</span>
                    <div className="font-medium">{formatDuration(task.estimated_time_seconds)}</div>
                  </div>
                  <div>
                    <span className="text-gray-500">实际时间:</span>
                    <div className="font-medium">{formatDuration(task.actual_time_seconds)}</div>
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm mt-3">
                  <div>
                    <span className="text-gray-500">创建时间:</span>
                    <div className="font-medium">{formatDate(task.created_at)}</div>
                  </div>
                  {task.started_at && (
                    <div>
                      <span className="text-gray-500">开始时间:</span>
                      <div className="font-medium">{formatDate(task.started_at)}</div>
                    </div>
                  )}
                  {task.completed_at && (
                    <div>
                      <span className="text-gray-500">完成时间:</span>
                      <div className="font-medium">{formatDate(task.completed_at)}</div>
                    </div>
                  )}
                </div>
              </div>
              
              <div className="ml-4 flex flex-col space-y-2">
                <span className="text-xs text-gray-500">ID: {task.id}</span>
                <button
                  onClick={() => handleToggleTaskDetails(task.id)}
                  disabled={loading.taskDetails}
                  className="px-3 py-1 text-xs bg-blue-50 text-blue-600 border border-blue-200 rounded-md hover:bg-blue-100 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading.taskDetails && expandedTask === task.id ? (
                    '加载中...'
                  ) : expandedTask === task.id ? (
                    '收起详情'
                  ) : (
                    '查看详情'
                  )}
                </button>
              </div>
            </div>
            
            {/* 执行元数据 */}
            {task.execution_metadata && Object.keys(task.execution_metadata).length > 0 && (
              <div className="mt-4 p-3 bg-gray-50 rounded-md">
                <details>
                  <summary className="cursor-pointer text-sm font-medium text-gray-700">
                    执行元数据
                  </summary>
                  <pre className="mt-2 text-xs text-gray-600 overflow-x-auto">
                    {JSON.stringify(task.execution_metadata, null, 2)}
                  </pre>
                </details>
              </div>
            )}

            {/* 任务输出详情 */}
            {expandedTask === task.id && (
              <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-md">
                <div className="flex items-center mb-3">
                  <h4 className="font-medium text-blue-800">任务执行结果</h4>
                  {loading.taskDetails && (
                    <span className="ml-2 text-xs text-blue-600">加载详情中...</span>
                  )}
                </div>
                {renderTaskOutput(task)}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* 分页 */}
      {pagination.tasks.totalPages > 1 && (
        <div className="bg-white p-4 rounded-lg shadow-sm border">
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-600">
              第 {pagination.tasks.page} 页，共 {pagination.tasks.totalPages} 页
            </div>
            <div className="flex space-x-2">
              <button
                onClick={() => handlePageChange(pagination.tasks.page - 1)}
                disabled={pagination.tasks.page <= 1}
                className="px-3 py-1 text-sm border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                上一页
              </button>
              <button
                onClick={() => handlePageChange(pagination.tasks.page + 1)}
                disabled={pagination.tasks.page >= pagination.tasks.totalPages}
                className="px-3 py-1 text-sm border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                下一页
              </button>
            </div>
          </div>
        </div>
      )}

      {filteredTasks.length === 0 && !loading.tasks && (
        <div className="text-center py-12 text-gray-500">
          暂无任务数据
        </div>
      )}
    </div>
  )
}

export default TaskList