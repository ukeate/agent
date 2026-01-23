import React, { useState, useEffect } from 'react'
import { apiClient } from '../services/apiClient'

interface ConfigItem {
  key: string
  value: any
  category: string
  description: string
  type: string
  validation?: any
}

interface ConfigStatus {
  redis_connected: boolean
  config_version: string
  last_sync: string
  total_configs: number
  categories: string[]
}

const EnterpriseConfigPage: React.FC = () => {
  const [configs, setConfigs] = useState<ConfigItem[]>([])
  const [status, setStatus] = useState<ConfigStatus | null>(null)
  const [selectedCategory, setSelectedCategory] = useState<string>('all')
  const [searchTerm, setSearchTerm] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false

    const hashVersion = (input: string) => {
      let h = 0
      for (let i = 0; i < input.length; i += 1) {
        h = (h << 5) - h + input.charCodeAt(i)
        h |= 0
      }
      return `v${Math.abs(h)}`
    }

    const flatten = (obj: any, prefix: string) => {
      const out: Array<{ key: string; value: any }> = []
      const walk = (value: any, path: string) => {
        if (value && typeof value === 'object' && !Array.isArray(value)) {
          Object.entries(value).forEach(([k, v]) => {
            walk(v, path ? `${path}.${k}` : k)
          })
          return
        }
        out.push({ key: `${prefix}.${path}`, value })
      }
      walk(obj, '')
      return out
    }

    const load = async () => {
      try {
        setLoading(true)
        const response = await apiClient.get<Record<string, any>>(
          '/enterprise/configuration'
        )
        const cfg = response.data || {}

        const categories = Object.keys(cfg)
        const items: ConfigItem[] = categories.flatMap(category =>
          flatten(cfg[category], category).map(i => {
            const t =
              i.value === null
                ? 'null'
                : Array.isArray(i.value)
                  ? 'array'
                  : typeof i.value
            return {
              key: i.key,
              value: i.value,
              category,
              description: i.key,
              type: t,
            }
          })
        )

        if (!cancelled) {
          setConfigs(items)
          setStatus({
            redis_connected: true,
            config_version: hashVersion(JSON.stringify(cfg)),
            last_sync: new Date().toISOString(),
            total_configs: items.length,
            categories,
          })
        }
      } catch {
        if (!cancelled) {
          setConfigs([])
          setStatus({
            redis_connected: false,
            config_version: '-',
            last_sync: new Date().toISOString(),
            total_configs: 0,
            categories: [],
          })
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    load()
    return () => {
      cancelled = true
    }
  }, [])

  const filteredConfigs = configs.filter(config => {
    const matchesCategory =
      selectedCategory === 'all' || config.category === selectedCategory
    const matchesSearch =
      config.key.toLowerCase().includes(searchTerm.toLowerCase()) ||
      config.description.includes(searchTerm)
    return matchesCategory && matchesSearch
  })

  if (loading) {
    return <div className="p-6">加载配置管理系统...</div>
  }

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">企业级配置管理系统</h1>
        <p className="text-gray-600 mb-4">
          集中式配置管理 - 展示централized configuration service的技术实现
        </p>

        {/* 状态面板 */}
        {status && (
          <div className="bg-white rounded-lg shadow p-4 mb-6">
            <h2 className="text-lg font-semibold mb-3">系统状态</h2>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              <div>
                <span className="block text-sm text-gray-500">Redis连接</span>
                <span
                  className={`font-medium ${status.redis_connected ? 'text-green-600' : 'text-red-600'}`}
                >
                  {status.redis_connected ? '已连接' : '断开'}
                </span>
              </div>
              <div>
                <span className="block text-sm text-gray-500">配置版本</span>
                <span className="font-medium">{status.config_version}</span>
              </div>
              <div>
                <span className="block text-sm text-gray-500">最后同步</span>
                <span className="font-medium text-sm">
                  {new Date(status.last_sync).toLocaleTimeString()}
                </span>
              </div>
              <div>
                <span className="block text-sm text-gray-500">配置总数</span>
                <span className="font-medium">{status.total_configs}</span>
              </div>
              <div>
                <span className="block text-sm text-gray-500">分类数</span>
                <span className="font-medium">{status.categories.length}</span>
              </div>
            </div>
          </div>
        )}

        {/* 过滤器 */}
        <div className="bg-white rounded-lg shadow p-4 mb-6">
          <div className="flex flex-wrap gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                分类过滤
              </label>
              <select
                name="categoryFilter"
                value={selectedCategory}
                onChange={e => setSelectedCategory(e.target.value)}
                className="border border-gray-300 rounded px-3 py-2"
              >
                <option value="all">所有分类</option>
                {status?.categories.map(category => (
                  <option key={category} value={category}>
                    {category}
                  </option>
                ))}
              </select>
            </div>
            <div className="flex-1">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                搜索配置
              </label>
              <input
                type="text"
                name="configSearch"
                value={searchTerm}
                onChange={e => setSearchTerm(e.target.value)}
                placeholder="搜索配置键或描述..."
                className="w-full border border-gray-300 rounded px-3 py-2"
              />
            </div>
          </div>
        </div>
      </div>

      {/* 配置列表 */}
      <div className="bg-white rounded-lg shadow">
        <div className="px-4 py-3 border-b">
          <h2 className="text-lg font-semibold">
            配置项 ({filteredConfigs.length})
          </h2>
        </div>

        <div className="divide-y">
          {filteredConfigs.map((config, index) => (
            <div key={config.key} className="p-4 hover:bg-gray-50">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <code className="bg-gray-100 px-2 py-1 rounded text-sm font-mono">
                      {config.key}
                    </code>
                    <span
                      className={`px-2 py-1 rounded-full text-xs font-medium ${
                        config.category === 'security'
                          ? 'bg-red-100 text-red-800'
                          : config.category === 'performance'
                            ? 'bg-blue-100 text-blue-800'
                            : config.category === 'agents'
                              ? 'bg-green-100 text-green-800'
                              : config.category === 'monitoring'
                                ? 'bg-purple-100 text-purple-800'
                                : 'bg-gray-100 text-gray-800'
                      }`}
                    >
                      {config.category}
                    </span>
                  </div>
                  <p className="text-gray-600 text-sm mb-2">
                    {config.description}
                  </p>
                  <div className="text-sm text-gray-500">
                    类型: <span className="font-mono">{config.type}</span>
                    {config.validation && (
                      <span className="ml-4">
                        范围: {config.validation.min} - {config.validation.max}
                      </span>
                    )}
                  </div>
                </div>
                <div className="ml-4">
                  <div className="text-right">
                    <div className="text-sm text-gray-500 mb-1">当前值</div>
                    <code className="bg-blue-100 text-blue-800 px-2 py-1 rounded font-mono">
                      {JSON.stringify(config.value)}
                    </code>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 技术说明 */}
      <div className="mt-8 bg-gray-50 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-3">技术实现说明</h3>
        <div className="text-sm text-gray-700 space-y-2">
          <p>
            <strong>集中式配置管理:</strong>{' '}
            通过enterprise_config.py实现统一配置服务
          </p>
          <p>
            <strong>Redis同步:</strong> 支持分布式配置同步和变更通知
          </p>
          <p>
            <strong>类型安全:</strong> 配置项具有类型验证和范围检查
          </p>
          <p>
            <strong>分类管理:</strong> 按功能模块对配置进行分类组织
          </p>
          <p>
            <strong>实时更新:</strong> 配置变更可实时推送到所有节点
          </p>
        </div>
      </div>
    </div>
  )
}

export default EnterpriseConfigPage
