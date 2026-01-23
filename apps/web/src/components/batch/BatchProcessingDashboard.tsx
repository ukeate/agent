/**
 * 批处理监控面板
 *
 * 提供批处理任务的实时监控、进度跟踪和管理界面
 */

import React, { useState, useEffect } from 'react'
import {
  batchService,
  BatchJob,
  BatchTask,
  BatchMetrics,
} from '../../services/batchService'

import { logger } from '../../utils/logger'
interface BatchDashboardState {
  jobs: BatchJob[]
  metrics: BatchMetrics | null
  selectedJob: BatchJob | null
  loading: boolean
  error: string | null
}

export const BatchProcessingDashboard: React.FC = () => {
  const [state, setState] = useState<BatchDashboardState>({
    jobs: [],
    metrics: null,
    selectedJob: null,
    loading: true,
    error: null,
  })

  const [autoRefresh, setAutoRefresh] = useState(true)
  const [refreshInterval] = useState(3000)

  // 获取批处理作业列表
  const fetchJobs = async () => {
    try {
      const jobs = await batchService.getJobs()
      setState(prev => ({ ...prev, jobs }))
    } catch (error) {
      logger.error('获取批处理作业失败:', error)
      setState(prev => ({
        ...prev,
        error: error instanceof Error ? error.message : '获取作业列表失败',
      }))
    }
  }

  // 获取批处理指标
  const fetchMetrics = async () => {
    try {
      const metrics = await batchService.getMetrics()
      setState(prev => ({ ...prev, metrics }))
    } catch (error) {
      logger.error('获取批处理指标失败:', error)
    }
  }

  // 获取作业详情
  const fetchJobDetails = async (jobId: string) => {
    try {
      const job = await batchService.getJobDetails(jobId)
      setState(prev => ({ ...prev, selectedJob: job }))
    } catch (error) {
      logger.error('获取作业详情失败:', error)
    }
  }

  useEffect(() => {
    const fetchData = async () => {
      setState(prev => ({ ...prev, loading: true }))
      await Promise.all([fetchJobs(), fetchMetrics()])
      setState(prev => ({ ...prev, loading: false }))
    }

    fetchData()

    if (autoRefresh) {
      const interval = setInterval(fetchData, refreshInterval)
      return () => clearInterval(interval)
    }
  }, [autoRefresh, refreshInterval])

  // 取消作业
  const handleCancelJob = async (jobId: string) => {
    try {
      await batchService.cancelJob(jobId)
      await fetchJobs()
    } catch (error) {
      logger.error('取消作业失败:', error)
    }
  }

  // 重试失败任务
  const handleRetryTasks = async (jobId: string) => {
    try {
      await batchService.retryFailedTasks(jobId)
      await fetchJobs()
    } catch (error) {
      logger.error('重试任务失败:', error)
    }
  }

  const getJobStatusColor = (status: string) => {
    switch (status) {
      case 'pending':
        return 'text-yellow-600 bg-yellow-100'
      case 'running':
        return 'text-blue-600 bg-blue-100'
      case 'completed':
        return 'text-green-600 bg-green-100'
      case 'failed':
        return 'text-red-600 bg-red-100'
      case 'cancelled':
        return 'text-gray-600 bg-gray-100'
      default:
        return 'text-gray-600 bg-gray-100'
    }
  }

  const calculateProgress = (job: BatchJob) => {
    if (job.total_tasks === 0) return 0
    return ((job.completed_tasks + job.failed_tasks) / job.total_tasks) * 100
  }

  return (
    <div className="space-y-6 mt-6">
      {/* 批处理指标概览 */}
      {state.metrics && (
        <div className="bg-white p-6 rounded-lg shadow">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-gray-900">
              批处理系统指标
            </h3>
            <div className="flex items-center space-x-2">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={autoRefresh}
                  onChange={e => setAutoRefresh(e.target.checked)}
                  className="mr-2"
                />
                <span className="text-sm text-gray-600">自动刷新</span>
              </label>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-gray-50 p-4 rounded">
              <div className="text-sm text-gray-600">活跃作业</div>
              <div className="text-2xl font-semibold text-gray-900">
                {state.metrics.active_jobs}
              </div>
            </div>
            <div className="bg-gray-50 p-4 rounded">
              <div className="text-sm text-gray-600">处理速率</div>
              <div className="text-2xl font-semibold text-gray-900">
                {state.metrics.tasks_per_second.toFixed(1)} 任务/秒
              </div>
            </div>
            <div className="bg-gray-50 p-4 rounded">
              <div className="text-sm text-gray-600">工作线程</div>
              <div className="text-2xl font-semibold text-gray-900">
                {state.metrics.active_workers} / {state.metrics.max_workers}
              </div>
            </div>
            <div className="bg-gray-50 p-4 rounded">
              <div className="text-sm text-gray-600">队列深度</div>
              <div className="text-2xl font-semibold text-gray-900">
                {state.metrics.queue_depth}
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
            <div className="bg-gray-50 p-4 rounded">
              <div className="text-sm text-gray-600">总任务数</div>
              <div className="text-xl font-semibold text-gray-900">
                {state.metrics.total_tasks}
              </div>
            </div>
            <div className="bg-gray-50 p-4 rounded">
              <div className="text-sm text-gray-600">完成任务</div>
              <div className="text-xl font-semibold text-green-600">
                {state.metrics.completed_tasks}
              </div>
            </div>
            <div className="bg-gray-50 p-4 rounded">
              <div className="text-sm text-gray-600">失败任务</div>
              <div className="text-xl font-semibold text-red-600">
                {state.metrics.failed_tasks}
              </div>
            </div>
            <div className="bg-gray-50 p-4 rounded">
              <div className="text-sm text-gray-600">成功率</div>
              <div className="text-xl font-semibold text-gray-900">
                {state.metrics.success_rate.toFixed(1)}%
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 批处理作业列表 */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">批处理作业</h3>

        {state.loading && (
          <div className="text-center py-8">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
            <p className="mt-2 text-gray-600">加载中...</p>
          </div>
        )}

        {!state.loading && state.jobs.length === 0 && (
          <p className="text-gray-500 text-center py-8">暂无批处理作业</p>
        )}

        {!state.loading && state.jobs.length > 0 && (
          <div className="space-y-4">
            {state.jobs.map(job => (
              <div key={job.id} className="border rounded-lg p-4">
                <div className="flex justify-between items-start mb-3">
                  <div>
                    <h4 className="font-medium text-gray-900">
                      作业 {job.id.substring(0, 8)}...
                    </h4>
                    <p className="text-sm text-gray-600 mt-1">
                      创建时间: {new Date(job.created_at).toLocaleString()}
                    </p>
                  </div>
                  <span
                    className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getJobStatusColor(job.status)}`}
                  >
                    {job.status}
                  </span>
                </div>

                {/* 进度条 */}
                <div className="mb-3">
                  <div className="flex justify-between text-sm text-gray-600 mb-1">
                    <span>进度</span>
                    <span>{calculateProgress(job).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${calculateProgress(job)}%` }}
                    />
                  </div>
                </div>

                {/* 任务统计 */}
                <div className="grid grid-cols-3 gap-4 text-sm mb-3">
                  <div>
                    <span className="text-gray-600">总任务:</span>
                    <span className="ml-1 font-medium">{job.total_tasks}</span>
                  </div>
                  <div>
                    <span className="text-gray-600">完成:</span>
                    <span className="ml-1 font-medium text-green-600">
                      {job.completed_tasks}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-600">失败:</span>
                    <span className="ml-1 font-medium text-red-600">
                      {job.failed_tasks}
                    </span>
                  </div>
                </div>

                {/* 操作按钮 */}
                <div className="flex space-x-2">
                  <button
                    onClick={() => fetchJobDetails(job.id)}
                    className="bg-blue-500 text-white px-3 py-1 rounded text-sm hover:bg-blue-600"
                  >
                    查看详情
                  </button>
                  {job.status === 'running' && (
                    <button
                      onClick={() => handleCancelJob(job.id)}
                      className="bg-yellow-500 text-white px-3 py-1 rounded text-sm hover:bg-yellow-600"
                    >
                      取消作业
                    </button>
                  )}
                  {job.failed_tasks > 0 && (
                    <button
                      onClick={() => handleRetryTasks(job.id)}
                      className="bg-orange-500 text-white px-3 py-1 rounded text-sm hover:bg-orange-600"
                    >
                      重试失败任务
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 作业详情模态框 */}
      {state.selectedJob && (
        <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-4xl w-full max-h-[80vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold">
                作业详情: {state.selectedJob.id}
              </h3>
              <button
                onClick={() =>
                  setState(prev => ({ ...prev, selectedJob: null }))
                }
                className="text-gray-400 hover:text-gray-600"
              >
                ✕
              </button>
            </div>

            {/* 任务列表 */}
            <div className="space-y-2">
              <h4 className="font-medium text-gray-900 mb-2">任务列表</h4>
              <div className="max-h-96 overflow-y-auto">
                {state.selectedJob.tasks.map((task: BatchTask) => (
                  <div key={task.id} className="border rounded p-3 mb-2">
                    <div className="flex justify-between items-center">
                      <div>
                        <span className="font-medium">
                          任务 {task.id.substring(0, 8)}...
                        </span>
                        <span className="ml-2 text-sm text-gray-600">
                          类型: {task.type}
                        </span>
                      </div>
                      <span
                        className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getJobStatusColor(task.status)}`}
                      >
                        {task.status}
                      </span>
                    </div>
                    {task.error && (
                      <div className="mt-2 text-sm text-red-600">
                        错误: {task.error}
                      </div>
                    )}
                    {task.result && (
                      <div className="mt-2 text-sm text-gray-600">
                        结果: {JSON.stringify(task.result).substring(0, 100)}...
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
