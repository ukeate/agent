import React, { useState, useEffect } from 'react'
import { Card } from '../ui/card'
import { Badge } from '../ui/badge'
import { Button } from '../ui/button'
import { Progress } from '../ui/progress'
import { Tabs } from '../ui/tabs'

interface Trial {
  id: string
  trial_number: number
  parameters: Record<string, any>
  value?: number
  state: string
  start_time?: string
  end_time?: string
  duration?: number
  error_message?: string
}

interface ExperimentDetailProps {
  experiment: {
    id: string
    name: string
    description?: string
    status: string
    algorithm: string
    objective: string
    config: any
    parameters: any[]
    best_value?: number
    best_params?: Record<string, any>
    total_trials: number
    successful_trials: number
    pruned_trials: number
    failed_trials: number
    created_at: string
    started_at?: string
    completed_at?: string
  }
  trials: Trial[]
  onRefresh?: () => void
  onStart?: () => void
  onStop?: () => void
  onDelete?: () => void
}

const statusColors = {
  created: 'bg-gray-500',
  running: 'bg-blue-500',
  completed: 'bg-green-500',
  failed: 'bg-red-500',
  stopped: 'bg-yellow-500',
}

const trialStateColors = {
  COMPLETE: 'text-green-600',
  PRUNED: 'text-yellow-600',
  FAIL: 'text-red-600',
  RUNNING: 'text-blue-600',
}

export const ExperimentDetail: React.FC<ExperimentDetailProps> = ({
  experiment,
  trials,
  onRefresh,
  onStart,
  onStop,
  onDelete,
}) => {
  const [activeTab, setActiveTab] = useState('overview')
  const [refreshInterval, setRefreshInterval] = useState<ReturnType<
    typeof setTimeout
  > | null>(null)

  // 自动刷新逻辑
  useEffect(() => {
    if (experiment.status === 'running') {
      const interval = setInterval(() => {
        onRefresh?.()
      }, 5000) // 每5秒刷新一次
      setRefreshInterval(interval)

      return () => {
        clearInterval(interval)
      }
    } else {
      if (refreshInterval) {
        clearInterval(refreshInterval)
        setRefreshInterval(null)
      }
    }
  }, [experiment.status, onRefresh])

  const getStatusColor = (status: string) =>
    statusColors[status as keyof typeof statusColors] || 'bg-gray-500'

  const successRate =
    experiment.total_trials > 0
      ? (experiment.successful_trials / experiment.total_trials) * 100
      : 0

  const renderOverview = () => (
    <div className="space-y-6">
      {/* 状态和操作 */}
      <Card className="p-6">
        <div className="flex justify-between items-start">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 mb-2">
              {experiment.name}
            </h2>
            {experiment.description && (
              <p className="text-gray-600 mb-4">{experiment.description}</p>
            )}
            <div className="flex items-center space-x-3">
              <Badge
                className={`${getStatusColor(experiment.status)} text-white`}
              >
                {experiment.status}
              </Badge>
              <Badge className="bg-blue-100 text-blue-800">
                {experiment.algorithm.toUpperCase()}
              </Badge>
              <Badge className="bg-purple-100 text-purple-800">
                {experiment.objective === 'maximize' ? '最大化' : '最小化'}
              </Badge>
            </div>
          </div>
          <div className="flex space-x-2">
            <Button variant="outline" onClick={onRefresh}>
              刷新
            </Button>
            {experiment.status === 'created' && (
              <Button onClick={onStart}>启动</Button>
            )}
            {experiment.status === 'running' && (
              <Button variant="destructive" onClick={onStop}>
                停止
              </Button>
            )}
            {['completed', 'failed', 'stopped'].includes(experiment.status) && (
              <Button variant="destructive" onClick={onDelete}>
                删除
              </Button>
            )}
          </div>
        </div>
      </Card>

      {/* 统计信息 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="p-4">
          <div className="text-sm font-medium text-gray-500 mb-1">总试验数</div>
          <div className="text-2xl font-bold text-gray-900">
            {experiment.total_trials}
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-sm font-medium text-gray-500 mb-1">成功试验</div>
          <div className="text-2xl font-bold text-green-600">
            {experiment.successful_trials}
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-sm font-medium text-gray-500 mb-1">剪枝试验</div>
          <div className="text-2xl font-bold text-yellow-600">
            {experiment.pruned_trials}
          </div>
        </Card>
        <Card className="p-4">
          <div className="text-sm font-medium text-gray-500 mb-1">失败试验</div>
          <div className="text-2xl font-bold text-red-600">
            {experiment.failed_trials}
          </div>
        </Card>
      </div>

      {/* 进度和最佳结果 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="p-6">
          <h3 className="text-lg font-semibold mb-4">优化进度</h3>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm text-gray-600 mb-2">
                <span>成功率</span>
                <span>{successRate.toFixed(1)}%</span>
              </div>
              <Progress value={successRate} className="h-3" />
            </div>
            {experiment.config.n_trials && (
              <div>
                <div className="flex justify-between text-sm text-gray-600 mb-2">
                  <span>完成进度</span>
                  <span>
                    {experiment.total_trials}/{experiment.config.n_trials}
                  </span>
                </div>
                <Progress
                  value={
                    (experiment.total_trials / experiment.config.n_trials) * 100
                  }
                  className="h-3"
                />
              </div>
            )}
          </div>
        </Card>

        {experiment.best_value !== undefined && (
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4">最佳结果</h3>
            <div className="space-y-3">
              <div>
                <span className="text-sm font-medium text-gray-500">
                  最佳值
                </span>
                <div className="text-xl font-bold text-blue-600 font-mono">
                  {experiment.best_value.toFixed(6)}
                </div>
              </div>
              {experiment.best_params && (
                <div>
                  <span className="text-sm font-medium text-gray-500">
                    最佳参数
                  </span>
                  <div className="mt-2 space-y-1">
                    {Object.entries(experiment.best_params).map(
                      ([key, value]) => (
                        <div key={key} className="flex justify-between text-sm">
                          <span className="font-medium">{key}:</span>
                          <span className="font-mono text-gray-600">
                            {typeof value === 'number'
                              ? value.toFixed(4)
                              : String(value)}
                          </span>
                        </div>
                      )
                    )}
                  </div>
                </div>
              )}
            </div>
          </Card>
        )}
      </div>

      {/* 时间信息 */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">时间信息</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div>
            <span className="font-medium text-gray-500">创建时间:</span>
            <div className="text-gray-900">
              {new Date(experiment.created_at).toLocaleString('zh-CN')}
            </div>
          </div>
          {experiment.started_at && (
            <div>
              <span className="font-medium text-gray-500">启动时间:</span>
              <div className="text-gray-900">
                {new Date(experiment.started_at).toLocaleString('zh-CN')}
              </div>
            </div>
          )}
          {experiment.completed_at && (
            <div>
              <span className="font-medium text-gray-500">完成时间:</span>
              <div className="text-gray-900">
                {new Date(experiment.completed_at).toLocaleString('zh-CN')}
              </div>
            </div>
          )}
        </div>
      </Card>
    </div>
  )

  const renderParameters = () => (
    <Card className="p-6">
      <h3 className="text-lg font-semibold mb-4">搜索空间配置</h3>
      <div className="space-y-4">
        {experiment.parameters.map((param: any, index: number) => (
          <div key={index} className="border rounded-lg p-4">
            <div className="flex items-center space-x-3 mb-2">
              <span className="font-semibold text-gray-900">{param.name}</span>
              <Badge className="bg-gray-100 text-gray-800">{param.type}</Badge>
            </div>
            <div className="text-sm text-gray-600">
              {param.type === 'float' || param.type === 'int' ? (
                <span>
                  范围: [{param.low}, {param.high}]{param.log && ' (对数尺度)'}
                  {param.step && ` (步长: ${param.step})`}
                </span>
              ) : param.type === 'categorical' ? (
                <span>可选值: {param.choices.join(', ')}</span>
              ) : param.type === 'boolean' ? (
                <span>布尔值: True/False</span>
              ) : null}
            </div>
          </div>
        ))}
      </div>
    </Card>
  )

  const renderTrials = () => (
    <Card className="p-6">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-semibold">试验历史</h3>
        <div className="text-sm text-gray-500">共 {trials.length} 个试验</div>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                试验 #
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                状态
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                值
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                参数
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                持续时间
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {trials
              .slice()
              .reverse()
              .map(trial => (
                <tr key={trial.id} className="hover:bg-gray-50">
                  <td className="px-4 py-3 whitespace-nowrap text-sm font-medium text-gray-900">
                    {trial.trial_number}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm">
                    <span
                      className={`font-medium ${trialStateColors[trial.state as keyof typeof trialStateColors] || 'text-gray-600'}`}
                    >
                      {trial.state}
                    </span>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm font-mono text-gray-900">
                    {trial.value !== undefined ? trial.value.toFixed(6) : '-'}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-500">
                    <div className="max-w-xs truncate">
                      {Object.entries(trial.parameters).map(([key, value]) => (
                        <span key={key} className="inline-block mr-2">
                          {key}:
                          {typeof value === 'number'
                            ? value.toFixed(3)
                            : String(value)}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                    {trial.duration ? `${trial.duration.toFixed(1)}s` : '-'}
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </Card>
  )

  const tabsData = [
    { id: 'overview', label: '概览', content: renderOverview() },
    { id: 'parameters', label: '参数配置', content: renderParameters() },
    {
      id: 'trials',
      label: `试验历史 (${trials.length})`,
      content: renderTrials(),
    },
  ]

  return (
    <div className="space-y-6">
      <Tabs tabs={tabsData} activeTab={activeTab} onTabChange={setActiveTab} />
    </div>
  )
}
