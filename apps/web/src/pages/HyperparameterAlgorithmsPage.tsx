import React, { useState, useEffect } from 'react'
import { Card } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Badge } from '../components/ui/badge'
import { Alert } from '../components/ui/alert'
import { Input } from '../components/ui/input'
import { Tabs } from '../components/ui/tabs'
import { hyperparameterServiceEnhanced } from '../services/hyperparameterServiceEnhanced'

import { logger } from '../utils/logger'
interface AlgorithmConfig {
  name: string
  displayName: string
  description: string
  parameters: Record<string, any>
}

const HyperparameterAlgorithmsPage: React.FC = () => {
  const [algorithms, setAlgorithms] = useState<AlgorithmConfig[]>([])
  const [selectedAlgorithm, setSelectedAlgorithm] =
    useState<AlgorithmConfig | null>(null)
  const [algorithmConfigs, setAlgorithmConfigs] = useState<Record<string, any>>(
    {}
  )
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState('overview')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    loadAlgorithms()
  }, [])

  const formatAlgorithmName = (name: string) => {
    const map: Record<string, string> = {
      tpe: 'TPE',
      cmaes: 'CMA-ES',
      random: '随机搜索',
      grid: '网格搜索',
      nsga2: 'NSGA-II',
    }
    return map[name] || name.toUpperCase()
  }

  const loadAlgorithms = async () => {
    try {
      setLoading(true)
      setError(null)
      const info = await hyperparameterServiceEnhanced.getAvailableAlgorithms()
      const list = (info.algorithms || []).map(name => ({
        name,
        displayName: formatAlgorithmName(name),
        description: info.descriptions?.[name] || '暂无描述',
        parameters: info.parameters?.[name] || {},
      }))
      const initialConfigs: Record<string, any> = {}
      list.forEach(alg => {
        initialConfigs[alg.name] = {}
        Object.entries(alg.parameters || {}).forEach(([param, config]) => {
          initialConfigs[alg.name][param] = (config as any).default
        })
      })
      setAlgorithms(list)
      setSelectedAlgorithm(list[0] || null)
      setAlgorithmConfigs(initialConfigs)
    } catch (err) {
      logger.error('加载算法信息失败:', err)
      setError('加载算法信息失败')
      setAlgorithms([])
      setSelectedAlgorithm(null)
      setAlgorithmConfigs({})
    } finally {
      setLoading(false)
    }
  }

  // 更新参数配置
  const updateParameter = (
    algorithmName: string,
    paramName: string,
    value: any
  ) => {
    setAlgorithmConfigs(prev => ({
      ...prev,
      [algorithmName]: {
        ...prev[algorithmName],
        [paramName]: value,
      },
    }))
  }

  // 重置配置
  const resetConfig = (algorithmName: string) => {
    const algorithm = algorithms.find(alg => alg.name === algorithmName)
    if (algorithm) {
      const resetConfig: Record<string, any> = {}
      Object.entries(algorithm.parameters).forEach(([param, config]) => {
        resetConfig[param] = config.default
      })
      setAlgorithmConfigs(prev => ({
        ...prev,
        [algorithmName]: resetConfig,
      }))
    }
  }

  // 渲染参数配置表单
  const renderParameterForm = (algorithm: AlgorithmConfig) => {
    if (!algorithm || Object.keys(algorithm.parameters || {}).length === 0) {
      return <div className="text-sm text-gray-500">该算法暂无可配置参数</div>
    }
    const config = algorithmConfigs[algorithm.name] || {}

    return (
      <div className="space-y-4">
        {Object.entries(algorithm.parameters).map(
          ([paramName, paramConfig]) => (
            <div
              key={paramName}
              className="grid grid-cols-1 md:grid-cols-2 gap-4 items-center"
            >
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  {paramName}
                </label>
                <p className="text-xs text-gray-500">
                  {paramConfig.description}
                </p>
              </div>

              <div>
                {paramConfig.type === 'int' ? (
                  <Input
                    type="number"
                    value={config[paramName] ?? paramConfig.default}
                    min={paramConfig.min}
                    max={paramConfig.max}
                    onChange={e =>
                      updateParameter(
                        algorithm.name,
                        paramName,
                        parseInt(e.target.value) || paramConfig.default
                      )
                    }
                  />
                ) : paramConfig.type === 'float' ? (
                  <Input
                    type="number"
                    step="0.01"
                    value={config[paramName] ?? paramConfig.default}
                    min={paramConfig.min}
                    max={paramConfig.max}
                    onChange={e =>
                      updateParameter(
                        algorithm.name,
                        paramName,
                        parseFloat(e.target.value) || paramConfig.default
                      )
                    }
                  />
                ) : paramConfig.type === 'select' ? (
                  <select
                    className="w-full p-2 border border-gray-300 rounded-md"
                    value={config[paramName] ?? paramConfig.default}
                    onChange={e =>
                      updateParameter(algorithm.name, paramName, e.target.value)
                    }
                  >
                    {paramConfig.options.map((option: string) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                ) : paramConfig.type === 'boolean' ? (
                  <select
                    className="w-full p-2 border border-gray-300 rounded-md"
                    value={
                      (config[paramName] ?? paramConfig.default)
                        ? 'true'
                        : 'false'
                    }
                    onChange={e =>
                      updateParameter(
                        algorithm.name,
                        paramName,
                        e.target.value === 'true'
                      )
                    }
                  >
                    <option value="true">true</option>
                    <option value="false">false</option>
                  </select>
                ) : (
                  <Input
                    value={config[paramName] || paramConfig.default}
                    onChange={e =>
                      updateParameter(algorithm.name, paramName, e.target.value)
                    }
                  />
                )}
              </div>
            </div>
          )
        )}
      </div>
    )
  }

  // 渲染算法概览
  const renderAlgorithmOverview = () => (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-semibold mb-2">算法描述</h3>
        <p className="text-gray-600">
          {selectedAlgorithm?.description || '暂无描述'}
        </p>
      </div>
    </div>
  )

  // 渲染参数配置
  const renderParameterConfig = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold">参数配置</h3>
        <Button
          variant="outline"
          onClick={() =>
            selectedAlgorithm && resetConfig(selectedAlgorithm.name)
          }
          disabled={!selectedAlgorithm}
        >
          重置默认值
        </Button>
      </div>

      <Card className="p-6">
        {selectedAlgorithm ? (
          renderParameterForm(selectedAlgorithm)
        ) : (
          <div className="text-sm text-gray-500">暂无算法数据</div>
        )}
      </Card>

      <Card className="p-4 bg-blue-50">
        <h4 className="font-semibold text-blue-800 mb-2">当前配置</h4>
        <pre className="text-sm text-blue-700 bg-blue-100 p-3 rounded overflow-x-auto">
          {JSON.stringify(
            selectedAlgorithm
              ? algorithmConfigs[selectedAlgorithm.name] || {}
              : {},
            null,
            2
          )}
        </pre>
      </Card>
    </div>
  )

  const tabsData = [
    { id: 'overview', label: '算法概览', content: renderAlgorithmOverview() },
    { id: 'parameters', label: '参数配置', content: renderParameterConfig() },
  ]

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="space-y-6">
        {/* 页面标题 */}
        <div>
          <h1 className="text-3xl font-bold text-gray-900">算法配置管理</h1>
          <p className="mt-2 text-gray-600">配置和比较不同的超参数优化算法</p>
        </div>

        {error && <Alert variant="destructive">{error}</Alert>}

        {/* 算法选择 */}
        <Card className="p-6">
          <h2 className="text-lg font-semibold mb-4">选择算法</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {algorithms.map(algorithm => (
              <div
                key={algorithm.name}
                className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                  selectedAlgorithm?.name === algorithm.name
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => setSelectedAlgorithm(algorithm)}
              >
                <div className="flex items-center justify-between mb-2">
                  <Badge className="bg-blue-100 text-blue-800">
                    {algorithm.name.toUpperCase()}
                  </Badge>
                  {selectedAlgorithm?.name === algorithm.name && (
                    <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
                  )}
                </div>
                <h3 className="font-medium text-gray-900 mb-1">
                  {algorithm.displayName}
                </h3>
                <p className="text-sm text-gray-600 line-clamp-2">
                  {algorithm.description}
                </p>
              </div>
            ))}
          </div>
          {!loading && algorithms.length === 0 && (
            <div className="text-sm text-gray-500">暂无可用算法</div>
          )}
        </Card>

        {/* 算法详情 */}
        <Card className="p-6">
          {selectedAlgorithm ? (
            <>
              <div className="flex items-center space-x-4 mb-6">
                <Badge className="bg-blue-500 text-white px-3 py-1">
                  {selectedAlgorithm.name.toUpperCase()}
                </Badge>
                <h2 className="text-xl font-semibold text-gray-900">
                  {selectedAlgorithm.displayName}
                </h2>
              </div>

              <Tabs
                tabs={tabsData}
                activeTab={activeTab}
                onTabChange={setActiveTab}
              />
            </>
          ) : (
            <div className="text-sm text-gray-500">暂无算法数据</div>
          )}
        </Card>
      </div>
    </div>
  )
}

export default HyperparameterAlgorithmsPage
