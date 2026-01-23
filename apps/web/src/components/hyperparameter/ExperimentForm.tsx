import React, { useState } from 'react'
import { Card } from '../ui/card'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import { Textarea } from '../ui/textarea'
import { Badge } from '../ui/badge'
import { Dialog } from '../ui/dialog'

import { logger } from '../../utils/logger'
interface Parameter {
  name: string
  type: 'float' | 'int' | 'categorical' | 'boolean'
  low?: number
  high?: number
  choices?: string[]
  log?: boolean
  step?: number
}

interface ExperimentFormProps {
  isOpen: boolean
  onClose: () => void
  onSubmit: (experiment: {
    name: string
    description: string
    algorithm: string
    objective: string
    n_trials: number
    early_stopping: boolean
    patience: number
    parameters: Parameter[]
  }) => void
  presetTasks?: string[]
  onLoadPreset?: (taskName: string) => Promise<any>
}

const algorithmOptions = [
  { value: 'tpe', label: 'TPE (推荐)' },
  { value: 'cmaes', label: 'CMA-ES' },
  { value: 'random', label: '随机搜索' },
  { value: 'grid', label: '网格搜索' },
]

const parameterTypeOptions = [
  { value: 'float', label: '浮点数' },
  { value: 'int', label: '整数' },
  { value: 'categorical', label: '分类' },
  { value: 'boolean', label: '布尔值' },
]

export const ExperimentForm: React.FC<ExperimentFormProps> = ({
  isOpen,
  onClose,
  onSubmit,
  presetTasks = [],
  onLoadPreset,
}) => {
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    algorithm: 'tpe',
    objective: 'maximize',
    n_trials: 100,
    early_stopping: true,
    patience: 20,
  })

  const [parameters, setParameters] = useState<Parameter[]>([])
  const [newParameter, setNewParameter] = useState<Parameter>({
    name: '',
    type: 'float',
    low: 0,
    high: 1,
    log: false,
  })

  const [selectedPreset, setSelectedPreset] = useState('')

  const handleAddParameter = () => {
    if (!newParameter.name) return

    setParameters([...parameters, { ...newParameter }])
    setNewParameter({
      name: '',
      type: 'float',
      low: 0,
      high: 1,
      log: false,
    })
  }

  const handleRemoveParameter = (index: number) => {
    setParameters(parameters.filter((_, i) => i !== index))
  }

  const handleLoadPreset = async () => {
    if (!selectedPreset || !onLoadPreset) return

    try {
      const presetData = await onLoadPreset(selectedPreset)

      setFormData({
        ...formData,
        name: `${selectedPreset}_experiment`,
        algorithm: presetData.algorithm || 'tpe',
      })

      if (presetData.parameters) {
        setParameters(presetData.parameters)
      }
    } catch (error) {
      logger.error('加载预设失败:', error)
    }
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()

    if (parameters.length === 0) {
      alert('请至少添加一个参数')
      return
    }

    onSubmit({
      ...formData,
      parameters,
    })

    // 重置表单
    setFormData({
      name: '',
      description: '',
      algorithm: 'tpe',
      objective: 'maximize',
      n_trials: 100,
      early_stopping: true,
      patience: 20,
    })
    setParameters([])
    onClose()
  }

  const renderParameterForm = () => {
    return (
      <div className="space-y-4 p-4 bg-gray-50 rounded-lg">
        <h4 className="font-medium text-gray-900">添加参数</h4>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              参数名称
            </label>
            <Input
              value={newParameter.name}
              onChange={e =>
                setNewParameter({ ...newParameter, name: e.target.value })
              }
              placeholder="如: learning_rate"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              参数类型
            </label>
            <select
              className="w-full p-2 border border-gray-300 rounded-md"
              value={newParameter.type}
              onChange={e =>
                setNewParameter({
                  ...newParameter,
                  type: e.target.value as Parameter['type'],
                })
              }
            >
              {parameterTypeOptions.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>
        </div>

        {(newParameter.type === 'float' || newParameter.type === 'int') && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                最小值
              </label>
              <Input
                type="number"
                value={newParameter.low || ''}
                onChange={e =>
                  setNewParameter({
                    ...newParameter,
                    low: parseFloat(e.target.value),
                  })
                }
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                最大值
              </label>
              <Input
                type="number"
                value={newParameter.high || ''}
                onChange={e =>
                  setNewParameter({
                    ...newParameter,
                    high: parseFloat(e.target.value),
                  })
                }
              />
            </div>
          </div>
        )}

        {newParameter.type === 'categorical' && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              可选值 (用逗号分隔)
            </label>
            <Input
              placeholder="如: adam,sgd,rmsprop"
              onChange={e =>
                setNewParameter({
                  ...newParameter,
                  choices: e.target.value.split(',').map(s => s.trim()),
                })
              }
            />
          </div>
        )}

        {newParameter.type === 'float' && (
          <div className="flex items-center">
            <input
              type="checkbox"
              id="log-scale"
              checked={newParameter.log || false}
              onChange={e =>
                setNewParameter({
                  ...newParameter,
                  log: e.target.checked,
                })
              }
              className="mr-2"
            />
            <label htmlFor="log-scale" className="text-sm text-gray-700">
              对数尺度
            </label>
          </div>
        )}

        <Button
          type="button"
          onClick={handleAddParameter}
          disabled={!newParameter.name}
        >
          添加参数
        </Button>
      </div>
    )
  }

  return (
    <Dialog isOpen={isOpen} onClose={onClose} maxWidth="4xl">
      <div className="p-6">
        <h2 className="text-xl font-semibold mb-6">创建超参数优化实验</h2>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* 预设任务选择 */}
          {presetTasks.length > 0 && (
            <Card className="p-4">
              <h3 className="font-medium text-gray-900 mb-3">使用预设配置</h3>
              <div className="flex space-x-2">
                <select
                  className="flex-1 p-2 border border-gray-300 rounded-md"
                  value={selectedPreset}
                  onChange={e => setSelectedPreset(e.target.value)}
                >
                  <option value="">选择预设任务</option>
                  {presetTasks.map(task => (
                    <option key={task} value={task}>
                      {task.replace(/_/g, ' ')}
                    </option>
                  ))}
                </select>
                <Button
                  type="button"
                  onClick={handleLoadPreset}
                  disabled={!selectedPreset}
                  variant="outline"
                >
                  加载预设
                </Button>
              </div>
            </Card>
          )}

          {/* 基本信息 */}
          <Card className="p-4">
            <h3 className="font-medium text-gray-900 mb-3">基本信息</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  实验名称 *
                </label>
                <Input
                  value={formData.name}
                  onChange={e =>
                    setFormData({ ...formData, name: e.target.value })
                  }
                  placeholder="输入实验名称"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  实验描述
                </label>
                <textarea
                  className="w-full p-2 border border-gray-300 rounded-md"
                  value={formData.description}
                  onChange={e =>
                    setFormData({ ...formData, description: e.target.value })
                  }
                  placeholder="输入实验描述（可选）"
                  rows={3}
                />
              </div>
            </div>
          </Card>

          {/* 优化配置 */}
          <Card className="p-4">
            <h3 className="font-medium text-gray-900 mb-3">优化配置</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  算法
                </label>
                <select
                  className="w-full p-2 border border-gray-300 rounded-md"
                  value={formData.algorithm}
                  onChange={e =>
                    setFormData({ ...formData, algorithm: e.target.value })
                  }
                >
                  {algorithmOptions.map(option => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  优化目标
                </label>
                <select
                  className="w-full p-2 border border-gray-300 rounded-md"
                  value={formData.objective}
                  onChange={e =>
                    setFormData({ ...formData, objective: e.target.value })
                  }
                >
                  <option value="maximize">最大化</option>
                  <option value="minimize">最小化</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  试验次数
                </label>
                <Input
                  type="number"
                  min="1"
                  max="10000"
                  value={formData.n_trials}
                  onChange={e =>
                    setFormData({
                      ...formData,
                      n_trials: parseInt(e.target.value) || 100,
                    })
                  }
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  早停耐心值
                </label>
                <Input
                  type="number"
                  min="1"
                  max="100"
                  value={formData.patience}
                  onChange={e =>
                    setFormData({
                      ...formData,
                      patience: parseInt(e.target.value) || 20,
                    })
                  }
                />
              </div>
            </div>

            <div className="mt-4">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={formData.early_stopping}
                  onChange={e =>
                    setFormData({
                      ...formData,
                      early_stopping: e.target.checked,
                    })
                  }
                  className="mr-2"
                />
                <span className="text-sm text-gray-700">启用早停机制</span>
              </label>
            </div>
          </Card>

          {/* 参数配置 */}
          <Card className="p-4">
            <h3 className="font-medium text-gray-900 mb-3">参数配置</h3>

            {/* 已添加的参数列表 */}
            {parameters.length > 0 && (
              <div className="mb-4">
                <h4 className="text-sm font-medium text-gray-700 mb-2">
                  已添加的参数:
                </h4>
                <div className="space-y-2">
                  {parameters.map((param, index) => (
                    <div
                      key={index}
                      className="flex items-center justify-between bg-white p-3 rounded border"
                    >
                      <div className="flex items-center space-x-2">
                        <Badge className="bg-blue-100 text-blue-800">
                          {param.name}
                        </Badge>
                        <Badge className="bg-gray-100 text-gray-800">
                          {param.type}
                        </Badge>
                        {param.type === 'float' || param.type === 'int' ? (
                          <span className="text-sm text-gray-600">
                            [{param.low}, {param.high}]{param.log && ' (log)'}
                          </span>
                        ) : param.type === 'categorical' ? (
                          <span className="text-sm text-gray-600">
                            {param.choices?.join(', ')}
                          </span>
                        ) : null}
                      </div>
                      <Button
                        type="button"
                        variant="destructive"
                        size="sm"
                        onClick={() => handleRemoveParameter(index)}
                      >
                        移除
                      </Button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {renderParameterForm()}
          </Card>

          {/* 提交按钮 */}
          <div className="flex justify-end space-x-4">
            <Button type="button" variant="outline" onClick={onClose}>
              取消
            </Button>
            <Button
              type="submit"
              disabled={!formData.name || parameters.length === 0}
            >
              创建实验
            </Button>
          </div>
        </form>
      </div>
    </Dialog>
  )
}
