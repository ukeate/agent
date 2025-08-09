/**
 * 任务提交表单组件
 * 允许用户向Supervisor提交新任务
 */

import React, { useState } from 'react'
import { useSupervisorStore } from '../../stores/supervisorStore'
import { TaskType, TaskPriority, TaskSubmissionRequest } from '../../types/supervisor'

interface TaskSubmissionFormProps {
  onSubmit?: () => void
  onCancel?: () => void
}

export const TaskSubmissionForm: React.FC<TaskSubmissionFormProps> = ({
  onSubmit,
  onCancel
}) => {
  const { submitTask, loading } = useSupervisorStore()

  const [formData, setFormData] = useState<TaskSubmissionRequest>({
    name: '',
    description: '',
    task_type: 'code_generation',
    priority: 'medium',
    input_data: {},
    constraints: {},
    timeout_minutes: 30
  })

  const [constraintKey, setConstraintKey] = useState('')
  const [constraintValue, setConstraintValue] = useState('')
  const [inputDataText, setInputDataText] = useState('')
  const [submitError, setSubmitError] = useState<string | null>(null)
  const [submitSuccess, setSubmitSuccess] = useState(false)

  const taskTypeOptions = [
    { value: 'code_generation', label: '代码生成', icon: '💻' },
    { value: 'code_review', label: '代码审查', icon: '🔍' },
    { value: 'documentation', label: '文档编写', icon: '📝' },
    { value: 'analysis', label: '分析', icon: '📊' },
    { value: 'planning', label: '规划', icon: '📋' },
  ]

  const priorityOptions = [
    { value: 'low', label: '低优先级', color: 'text-gray-600' },
    { value: 'medium', label: '中等优先级', color: 'text-blue-600' },
    { value: 'high', label: '高优先级', color: 'text-orange-600' },
    { value: 'urgent', label: '紧急', color: 'text-red-600' },
  ]

  const handleInputChange = (field: keyof TaskSubmissionRequest, value: any) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }))
  }

  const addConstraint = () => {
    if (constraintKey.trim() && constraintValue.trim()) {
      setFormData(prev => ({
        ...prev,
        constraints: {
          ...(prev.constraints || {}),
          [constraintKey.trim()]: constraintValue.trim()
        }
      }))
      setConstraintKey('')
      setConstraintValue('')
    }
  }

  const removeConstraint = (key: string) => {
    setFormData(prev => {
      const newConstraints = { ...(prev.constraints || {}) }
      delete newConstraints[key]
      return {
        ...prev,
        constraints: newConstraints
      }
    })
  }

  const handleInputDataChange = (text: string) => {
    setInputDataText(text)
    try {
      if (text.trim()) {
        const parsed = JSON.parse(text)
        setFormData(prev => ({ ...prev, input_data: parsed }))
      } else {
        setFormData(prev => ({ ...prev, input_data: {} }))
      }
    } catch (error) {
      // 保持输入文本，但不更新 formData
    }
  }

  const validateForm = () => {
    if (!formData.name.trim()) {
      return '任务名称不能为空'
    }
    if (!formData.description.trim()) {
      return '任务描述不能为空'
    }
    if (inputDataText.trim()) {
      try {
        JSON.parse(inputDataText)
      } catch (error) {
        return '输入数据必须是有效的JSON格式'
      }
    }
    return null
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    const validationError = validateForm()
    if (validationError) {
      setSubmitError(validationError)
      return
    }

    setSubmitError(null)
    setSubmitSuccess(false)

    try {
      await submitTask(formData)
      setSubmitSuccess(true)
      setTimeout(() => {
        onSubmit?.()
      }, 2000)
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : '提交失败')
    }
  }

  const handleReset = () => {
    setFormData({
      name: '',
      description: '',
      task_type: 'code_generation',
      priority: 'medium',
      input_data: {},
      constraints: {},
      timeout_minutes: 30
    })
    setConstraintKey('')
    setConstraintValue('')
    setInputDataText('')
    setSubmitError(null)
    setSubmitSuccess(false)
  }

  if (submitSuccess) {
    return (
      <div className="text-center py-8">
        <div className="text-6xl mb-4">✅</div>
        <h3 className="text-xl font-semibold text-green-600 mb-2">任务提交成功！</h3>
        <p className="text-gray-600 mb-4">任务已分配给最合适的智能体处理</p>
        <button
          onClick={onSubmit}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
        >
          关闭
        </button>
      </div>
    )
  }

  return (
    <form onSubmit={handleSubmit} className="task-submission-form space-y-6">
      {/* 基本信息 */}
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            任务名称 *
          </label>
          <input
            type="text"
            value={formData.name}
            onChange={(e) => handleInputChange('name', e.target.value)}
            className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            placeholder="请输入任务名称"
            required
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            任务描述 *
          </label>
          <textarea
            value={formData.description}
            onChange={(e) => handleInputChange('description', e.target.value)}
            rows={4}
            className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            placeholder="详细描述任务需求和目标"
            required
          />
        </div>
      </div>

      {/* 任务类型和优先级 */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            任务类型
          </label>
          <select
            value={formData.task_type}
            onChange={(e) => handleInputChange('task_type', e.target.value as TaskType)}
            className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            {taskTypeOptions.map(option => (
              <option key={option.value} value={option.value}>
                {option.icon} {option.label}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            优先级
          </label>
          <select
            value={formData.priority}
            onChange={(e) => handleInputChange('priority', e.target.value as TaskPriority)}
            className="w-full border border-gray-300 rounded-md px-3 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            {priorityOptions.map(option => (
              <option key={option.value} value={option.value} className={option.color}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* 约束条件 */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          约束条件
        </label>
        <div className="flex space-x-2 mb-2">
          <input
            type="text"
            value={constraintKey}
            onChange={(e) => setConstraintKey(e.target.value)}
            className="flex-1 border border-gray-300 rounded-md px-3 py-2"
            placeholder="约束名称"
          />
          <input
            type="text"
            value={constraintValue}
            onChange={(e) => setConstraintValue(e.target.value)}
            className="flex-1 border border-gray-300 rounded-md px-3 py-2"
            placeholder="约束值"
            onKeyPress={(e) => e.key === 'Enter' && (e.preventDefault(), addConstraint())}
          />
          <button
            type="button"
            onClick={addConstraint}
            className="px-3 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700"
          >
            添加
          </button>
        </div>
        <div className="space-y-1">
          {Object.entries(formData.constraints || {}).map(([key, value]) => (
            <div key={key} className="flex items-center justify-between bg-gray-100 px-3 py-2 rounded">
              <span className="text-sm"><strong>{key}:</strong> {value}</span>
              <button
                type="button"
                onClick={() => removeConstraint(key)}
                className="text-red-500 hover:text-red-700"
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      </div>


      {/* 输入数据 */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          输入数据（JSON格式）
        </label>
        <textarea
          value={inputDataText}
          onChange={(e) => handleInputDataChange(e.target.value)}
          rows={4}
          className="w-full border border-gray-300 rounded-md px-3 py-2 font-mono text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          placeholder='{"key": "value"}'
        />
        {inputDataText.trim() && (
          <div className="mt-2 text-xs text-gray-500">
            {(() => {
              try {
                JSON.parse(inputDataText)
                return '✅ 有效的JSON格式'
              } catch {
                return '❌ JSON格式错误'
              }
            })()}
          </div>
        )}
      </div>

      {/* 错误提示 */}
      {submitError && (
        <div className="bg-red-50 border border-red-200 rounded-md p-4">
          <div className="text-red-700">{submitError}</div>
        </div>
      )}

      {/* 操作按钮 */}
      <div className="flex justify-between space-x-4">
        <div className="flex space-x-2">
          <button
            type="button"
            onClick={handleReset}
            className="px-4 py-2 text-gray-600 border border-gray-300 rounded-md hover:bg-gray-50"
          >
            重置
          </button>
        </div>
        
        <div className="flex space-x-2">
          {onCancel && (
            <button
              type="button"
              onClick={onCancel}
              className="px-4 py-2 text-gray-600 border border-gray-300 rounded-md hover:bg-gray-50"
            >
              取消
            </button>
          )}
          <button
            type="submit"
            disabled={loading.submitting}
            className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading.submitting ? '提交中...' : '提交任务'}
          </button>
        </div>
      </div>
    </form>
  )
}

export default TaskSubmissionForm