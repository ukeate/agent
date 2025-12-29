/**
 * Supervisor配置组件
 * 管理Supervisor的路由策略、权重和其他配置选项
 */

import React, { useEffect, useState } from 'react'
import { useSupervisorStore } from '../../stores/supervisorStore'
import { SupervisorConfig as ConfigType, RoutingStrategy } from '../../types/supervisor'

import { logger } from '../../utils/logger'
export const SupervisorConfig: React.FC = () => {
  const {
    config,
    loading,
    loadConfig,
    updateConfig,
    error
  } = useSupervisorStore()

  const [isEditing, setIsEditing] = useState(false)
  const [editingConfig, setEditingConfig] = useState<Partial<ConfigType> | null>(null)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (!config && !loading.config) {
      loadConfig()
    }
  }, [config, loading.config, loadConfig])

  const handleEdit = () => {
    if (config) {
      setEditingConfig({ ...config })
      setIsEditing(true)
    }
  }

  const handleCancel = () => {
    setEditingConfig(null)
    setIsEditing(false)
  }

  const handleSave = async () => {
    if (!editingConfig) return

    setSaving(true)
    try {
      await updateConfig(editingConfig)
      setIsEditing(false)
      setEditingConfig(null)
    } catch (err) {
      logger.error('更新配置失败:', err)
    } finally {
      setSaving(false)
    }
  }

  const handleInputChange = (field: keyof ConfigType, value: any) => {
    if (!editingConfig) return
    setEditingConfig({
      ...editingConfig,
      [field]: value
    })
  }

  if (loading.config && !config) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-500">加载配置中...</div>
      </div>
    )
  }

  if (!config) {
    return (
      <div className="text-center py-12 text-gray-500">
        暂无配置数据
      </div>
    )
  }

  const displayConfig = isEditing ? editingConfig : config

  return (
    <div className="supervisor-config space-y-6">
      {/* 头部操作 */}
      <div className="bg-white p-4 rounded-lg shadow-sm border">
        <div className="flex justify-between items-center">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">Supervisor配置管理</h2>
            <p className="text-sm text-gray-600 mt-1">
              配置版本: {config.config_version} | 最后更新: {new Date(config.updated_at).toLocaleString('zh-CN')}
            </p>
          </div>
          <div className="flex space-x-2">
            {isEditing ? (
              <>
                <button
                  onClick={handleCancel}
                  disabled={saving}
                  className="px-4 py-2 text-gray-600 border border-gray-300 rounded-md hover:bg-gray-50 disabled:opacity-50"
                >
                  取消
                </button>
                <button
                  onClick={handleSave}
                  disabled={saving}
                  className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
                >
                  {saving ? '保存中...' : '保存'}
                </button>
              </>
            ) : (
              <button
                onClick={handleEdit}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
              >
                编辑配置
              </button>
            )}
          </div>
        </div>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-md p-4">
          <div className="text-red-700">{error}</div>
        </div>
      )}

      {/* 基本配置 */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">基本配置</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">配置名称</label>
            {isEditing ? (
              <input
                type="text"
                value={displayConfig?.config_name || ''}
                onChange={(e) => handleInputChange('config_name', e.target.value)}
                className="w-full border border-gray-300 rounded-md px-3 py-2"
              />
            ) : (
              <div className="text-gray-900">{displayConfig?.config_name}</div>
            )}
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">是否激活</label>
            {isEditing ? (
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={displayConfig?.is_active || false}
                  onChange={(e) => handleInputChange('is_active', e.target.checked)}
                  className="rounded"
                />
                <span className="text-sm">激活此配置</span>
              </label>
            ) : (
              <div className={`inline-flex px-2 py-1 text-xs font-medium rounded-full ${
                displayConfig?.is_active 
                  ? 'bg-green-100 text-green-800' 
                  : 'bg-red-100 text-red-800'
              }`}>
                {displayConfig?.is_active ? '已激活' : '未激活'}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 路由配置 */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">路由配置</h3>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">路由策略</label>
            {isEditing ? (
              <select
                value={displayConfig?.routing_strategy || 'hybrid'}
                onChange={(e) => handleInputChange('routing_strategy', e.target.value as RoutingStrategy)}
                className="w-full border border-gray-300 rounded-md px-3 py-2"
              >
                <option value="round_robin">轮询策略</option>
                <option value="capability_based">基于能力</option>
                <option value="load_balanced">负载均衡</option>
                <option value="hybrid">混合策略</option>
              </select>
            ) : (
              <div className="text-gray-900 capitalize">{displayConfig?.routing_strategy}</div>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              负载阈值 ({displayConfig?.load_threshold ? (displayConfig.load_threshold * 100).toFixed(0) + '%' : '80%'})
            </label>
            {isEditing ? (
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={displayConfig?.load_threshold || 0.8}
                onChange={(e) => handleInputChange('load_threshold', parseFloat(e.target.value))}
                className="w-full"
              />
            ) : (
              <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                <div
                  className="bg-blue-500 h-2 rounded-full"
                  style={{ width: `${(displayConfig?.load_threshold || 0.8) * 100}%` }}
                />
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 权重配置 */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">权重配置</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              能力权重 ({displayConfig?.capability_weight ? (displayConfig.capability_weight * 100).toFixed(0) + '%' : '50%'})
            </label>
            {isEditing ? (
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={displayConfig?.capability_weight || 0.5}
                onChange={(e) => handleInputChange('capability_weight', parseFloat(e.target.value))}
                className="w-full"
              />
            ) : (
              <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                <div
                  className="bg-green-500 h-2 rounded-full"
                  style={{ width: `${(displayConfig?.capability_weight || 0.5) * 100}%` }}
                />
              </div>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              负载权重 ({displayConfig?.load_weight ? (displayConfig.load_weight * 100).toFixed(0) + '%' : '30%'})
            </label>
            {isEditing ? (
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={displayConfig?.load_weight || 0.3}
                onChange={(e) => handleInputChange('load_weight', parseFloat(e.target.value))}
                className="w-full"
              />
            ) : (
              <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                <div
                  className="bg-yellow-500 h-2 rounded-full"
                  style={{ width: `${(displayConfig?.load_weight || 0.3) * 100}%` }}
                />
              </div>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              可用性权重 ({displayConfig?.availability_weight ? (displayConfig.availability_weight * 100).toFixed(0) + '%' : '20%'})
            </label>
            {isEditing ? (
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={displayConfig?.availability_weight || 0.2}
                onChange={(e) => handleInputChange('availability_weight', parseFloat(e.target.value))}
                className="w-full"
              />
            ) : (
              <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                <div
                  className="bg-purple-500 h-2 rounded-full"
                  style={{ width: `${(displayConfig?.availability_weight || 0.2) * 100}%` }}
                />
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 质量评估配置 */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">质量评估配置</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="flex items-center space-x-2">
              {isEditing ? (
                <input
                  type="checkbox"
                  checked={displayConfig?.enable_quality_assessment || false}
                  onChange={(e) => handleInputChange('enable_quality_assessment', e.target.checked)}
                  className="rounded"
                />
              ) : (
                <div className={`w-4 h-4 rounded ${displayConfig?.enable_quality_assessment ? 'bg-green-500' : 'bg-gray-300'}`} />
              )}
              <span className="text-sm font-medium text-gray-700">启用质量评估</span>
            </label>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              最小信心阈值 ({displayConfig?.min_confidence_threshold ? (displayConfig.min_confidence_threshold * 100).toFixed(0) + '%' : '50%'})
            </label>
            {isEditing ? (
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={displayConfig?.min_confidence_threshold || 0.5}
                onChange={(e) => handleInputChange('min_confidence_threshold', parseFloat(e.target.value))}
                className="w-full"
                disabled={!displayConfig?.enable_quality_assessment}
              />
            ) : (
              <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                <div
                  className="bg-red-500 h-2 rounded-full"
                  style={{ width: `${(displayConfig?.min_confidence_threshold || 0.5) * 100}%` }}
                />
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 学习和优化配置 */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">学习和优化配置</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="flex items-center space-x-2">
              {isEditing ? (
                <input
                  type="checkbox"
                  checked={displayConfig?.enable_learning || false}
                  onChange={(e) => handleInputChange('enable_learning', e.target.checked)}
                  className="rounded"
                />
              ) : (
                <div className={`w-4 h-4 rounded ${displayConfig?.enable_learning ? 'bg-green-500' : 'bg-gray-300'}`} />
              )}
              <span className="text-sm font-medium text-gray-700">启用学习功能</span>
            </label>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              学习率 ({displayConfig?.learning_rate ? (displayConfig.learning_rate * 100).toFixed(0) + '%' : '10%'})
            </label>
            {isEditing ? (
              <input
                type="range"
                min="0.01"
                max="0.5"
                step="0.01"
                value={displayConfig?.learning_rate || 0.1}
                onChange={(e) => handleInputChange('learning_rate', parseFloat(e.target.value))}
                className="w-full"
                disabled={!displayConfig?.enable_learning}
              />
            ) : (
              <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                <div
                  className="bg-indigo-500 h-2 rounded-full"
                  style={{ width: `${((displayConfig?.learning_rate || 0.1) / 0.5) * 100}%` }}
                />
              </div>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">优化间隔（小时）</label>
            {isEditing ? (
              <input
                type="number"
                min="1"
                max="168"
                value={displayConfig?.optimization_interval_hours || 24}
                onChange={(e) => handleInputChange('optimization_interval_hours', parseInt(e.target.value))}
                className="w-full border border-gray-300 rounded-md px-3 py-2"
                disabled={!displayConfig?.enable_learning}
              />
            ) : (
              <div className="text-gray-900">{displayConfig?.optimization_interval_hours || 24} 小时</div>
            )}
          </div>
        </div>
      </div>

      {/* 系统限制配置 */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">系统限制配置</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">最大并发任务数</label>
            {isEditing ? (
              <input
                type="number"
                min="1"
                max="100"
                value={displayConfig?.max_concurrent_tasks || 10}
                onChange={(e) => handleInputChange('max_concurrent_tasks', parseInt(e.target.value))}
                className="w-full border border-gray-300 rounded-md px-3 py-2"
              />
            ) : (
              <div className="text-gray-900">{displayConfig?.max_concurrent_tasks || 10}</div>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">任务超时时间（分钟）</label>
            {isEditing ? (
              <input
                type="number"
                min="1"
                max="1440"
                value={displayConfig?.task_timeout_minutes || 30}
                onChange={(e) => handleInputChange('task_timeout_minutes', parseInt(e.target.value))}
                className="w-full border border-gray-300 rounded-md px-3 py-2"
              />
            ) : (
              <div className="text-gray-900">{displayConfig?.task_timeout_minutes || 30} 分钟</div>
            )}
          </div>

          <div>
            <label className="flex items-center space-x-2">
              {isEditing ? (
                <input
                  type="checkbox"
                  checked={displayConfig?.enable_fallback || false}
                  onChange={(e) => handleInputChange('enable_fallback', e.target.checked)}
                  className="rounded"
                />
              ) : (
                <div className={`w-4 h-4 rounded ${displayConfig?.enable_fallback ? 'bg-green-500' : 'bg-gray-300'}`} />
              )}
              <span className="text-sm font-medium text-gray-700">启用回退机制</span>
            </label>
          </div>
        </div>
      </div>

      {/* 配置元数据 */}
      {displayConfig?.config_metadata && Object.keys(displayConfig.config_metadata).length > 0 && (
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">配置元数据</h3>
          <pre className="text-sm text-gray-600 bg-gray-50 p-4 rounded overflow-x-auto">
            {JSON.stringify(displayConfig.config_metadata, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}

export default SupervisorConfig
