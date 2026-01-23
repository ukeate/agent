import React, { useState, useEffect } from 'react'
import { Card } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Badge } from '../components/ui/badge'
import { Alert } from '../components/ui/alert'
import { buildApiUrl, apiFetch } from '../utils/apiBase'

interface Experiment {
  id: string
  name: string
  status: string
  algorithm: string
  total_trials: number
  best_value?: number
}

interface VisualizationData {
  optimization_history: string
  param_importance: string
  parallel_coordinate: string
  contour: string
  [key: string]: string
}

interface ChartData {
  trials: Array<{
    trial_number: number
    value: number
    best_value_so_far: number
  }>
  param_importance: Array<{
    param_name: string
    importance: number
  }>
  parameter_relations: Array<{
    param1: string
    param2: string
    values: Array<{
      x: number
      y: number
      objective: number
    }>
  }>
}

const HyperparameterVisualizationsPage: React.FC = () => {
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [selectedExperiment, setSelectedExperiment] = useState<string | null>(
    null
  )
  const [visualizations, setVisualizations] =
    useState<VisualizationData | null>(null)
  const [chartData, setChartData] = useState<ChartData | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const API_BASE = buildApiUrl('/api/v1/hyperparameter-optimization')

  // 加载实验列表
  const loadExperiments = async () => {
    try {
      const response = await apiFetch(`${API_BASE}/experiments`)

      const data = await response.json()
      setExperiments(data.filter((exp: Experiment) => exp.total_trials > 0))
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : '未知错误')
    }
  }

  // 加载可视化数据
  const loadVisualization = async (experimentId: string) => {
    try {
      setLoading(true)

      const [vizResponse, trialsResponse] = await Promise.all([
        apiFetch(`${API_BASE}/experiments/${experimentId}/visualizations`),
        apiFetch(`${API_BASE}/experiments/${experimentId}/trials`),
      ])

      const vizData = await vizResponse.json()
      const trialsData = await trialsResponse.json()

      setVisualizations(vizData)

      // 处理图表数据
      const processedData = processTrialsData(trialsData)
      setChartData(processedData)

      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : '未知错误')
    } finally {
      setLoading(false)
    }
  }

  // 处理试验数据为图表格式
  const processTrialsData = (trials: any[]): ChartData => {
    // 优化历史数据
    let bestSoFar = null
    const optimizationHistory = trials
      .filter(trial => trial.value !== undefined)
      .sort((a, b) => a.trial_number - b.trial_number)
      .map(trial => {
        if (bestSoFar === null || trial.value > bestSoFar) {
          bestSoFar = trial.value
        }
        return {
          trial_number: trial.trial_number,
          value: trial.value,
          best_value_so_far: bestSoFar,
        }
      })

    // 参数重要性（基于相关性粗略估计）
    const allParams = new Set<string>()
    trials.forEach(trial => {
      if (trial.parameters) {
        Object.keys(trial.parameters).forEach(param => allParams.add(param))
      }
    })

    const computeCorrelation = (xs: number[], ys: number[]) => {
      if (xs.length < 3) return 0
      const n = xs.length
      const meanX = xs.reduce((s, v) => s + v, 0) / n
      const meanY = ys.reduce((s, v) => s + v, 0) / n
      let num = 0
      let denX = 0
      let denY = 0
      for (let i = 0; i < n; i++) {
        const dx = xs[i] - meanX
        const dy = ys[i] - meanY
        num += dx * dy
        denX += dx * dx
        denY += dy * dy
      }
      if (denX === 0 || denY === 0) return 0
      return num / Math.sqrt(denX * denY)
    }

    const paramImportance = Array.from(allParams)
      .map(param => {
        const values: number[] = []
        const objectives: number[] = []
        trials.forEach(trial => {
          if (
            trial.parameters &&
            trial.value !== undefined &&
            typeof trial.parameters[param] === 'number'
          ) {
            values.push(trial.parameters[param])
            objectives.push(trial.value)
          }
        })
        const corr = computeCorrelation(values, objectives)
        return {
          param_name: param,
          importance: Math.abs(corr),
        }
      })
      .sort((a, b) => b.importance - a.importance)

    // 参数关系（取前两个重要参数）
    const parameterRelations =
      paramImportance.length >= 2
        ? [
            {
              param1: paramImportance[0].param_name,
              param2: paramImportance[1].param_name,
              values: trials
                .filter(
                  trial =>
                    trial.parameters &&
                    trial.value !== undefined &&
                    trial.parameters[paramImportance[0].param_name] !==
                      undefined &&
                    trial.parameters[paramImportance[1].param_name] !==
                      undefined
                )
                .map(trial => ({
                  x: trial.parameters[paramImportance[0].param_name],
                  y: trial.parameters[paramImportance[1].param_name],
                  objective: trial.value,
                })),
            },
          ]
        : []

    return {
      trials: optimizationHistory,
      param_importance: paramImportance,
      parameter_relations: parameterRelations,
    }
  }

  useEffect(() => {
    loadExperiments()
  }, [])

  useEffect(() => {
    if (selectedExperiment) {
      loadVisualization(selectedExperiment)
    }
  }, [selectedExperiment])

  // 渲染优化历史图表
  const renderOptimizationHistory = () => {
    if (!chartData?.trials.length) return null

    const maxValue = Math.max(...chartData.trials.map(t => t.value))
    const minValue = Math.min(...chartData.trials.map(t => t.value))
    const valueRange = maxValue - minValue

    return (
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">优化历史</h3>
        <div className="h-64 relative">
          <svg className="w-full h-full" viewBox="0 0 800 200">
            {/* 坐标轴 */}
            <line
              x1="50"
              y1="170"
              x2="750"
              y2="170"
              stroke="#e5e5e5"
              strokeWidth="2"
            />
            <line
              x1="50"
              y1="20"
              x2="50"
              y2="170"
              stroke="#e5e5e5"
              strokeWidth="2"
            />

            {/* 数据点和线条 */}
            {chartData.trials.map((trial, index) => {
              const x = 50 + (index / (chartData.trials.length - 1)) * 700
              const y = 170 - ((trial.value - minValue) / valueRange) * 150
              const bestY =
                170 - ((trial.best_value_so_far - minValue) / valueRange) * 150

              return (
                <g key={index}>
                  {/* 最佳值线 */}
                  {index > 0 && (
                    <line
                      x1={
                        50 + ((index - 1) / (chartData.trials.length - 1)) * 700
                      }
                      y1={
                        170 -
                        ((chartData.trials[index - 1].best_value_so_far -
                          minValue) /
                          valueRange) *
                          150
                      }
                      x2={x}
                      y2={bestY}
                      stroke="#3b82f6"
                      strokeWidth="2"
                    />
                  )}

                  {/* 当前试验点 */}
                  <circle cx={x} cy={y} r="3" fill="#ef4444" opacity="0.6" />

                  {/* 最佳值点 */}
                  <circle cx={x} cy={bestY} r="2" fill="#3b82f6" />
                </g>
              )
            })}

            {/* 图例 */}
            <g transform="translate(600, 30)">
              <circle cx="0" cy="0" r="3" fill="#ef4444" opacity="0.6" />
              <text x="10" y="3" fontSize="12" fill="#666">
                试验值
              </text>
              <circle cx="0" cy="20" r="2" fill="#3b82f6" />
              <text x="10" y="23" fontSize="12" fill="#666">
                最佳值
              </text>
            </g>
          </svg>
        </div>
        <div className="flex justify-between text-sm text-gray-600 mt-2">
          <span>试验 1</span>
          <span>试验 {chartData.trials.length}</span>
        </div>
      </Card>
    )
  }

  // 渲染参数重要性图表
  const renderParameterImportance = () => {
    if (!chartData?.param_importance.length) return null

    return (
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">参数重要性</h3>
        <div className="space-y-3">
          {chartData.param_importance.slice(0, 8).map((param, index) => (
            <div key={param.param_name} className="flex items-center space-x-3">
              <div className="w-24 text-sm font-medium text-gray-700">
                {param.param_name}
              </div>
              <div className="flex-1 bg-gray-200 rounded-full h-4">
                <div
                  className="bg-blue-500 h-4 rounded-full transition-all duration-300"
                  style={{ width: `${param.importance * 100}%` }}
                />
              </div>
              <div className="w-12 text-sm text-gray-600">
                {(param.importance * 100).toFixed(1)}%
              </div>
            </div>
          ))}
        </div>
      </Card>
    )
  }

  // 渲染参数关系散点图
  const renderParameterRelations = () => {
    if (!chartData?.parameter_relations.length) return null

    const relation = chartData.parameter_relations[0]
    if (!relation.values.length) return null

    const xMin = Math.min(...relation.values.map(v => v.x))
    const xMax = Math.max(...relation.values.map(v => v.x))
    const yMin = Math.min(...relation.values.map(v => v.y))
    const yMax = Math.max(...relation.values.map(v => v.y))
    const objMin = Math.min(...relation.values.map(v => v.objective))
    const objMax = Math.max(...relation.values.map(v => v.objective))

    return (
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">
          参数关系: {relation.param1} vs {relation.param2}
        </h3>
        <div className="h-64 relative">
          <svg className="w-full h-full" viewBox="0 0 300 200">
            {/* 坐标轴 */}
            <line
              x1="40"
              y1="170"
              x2="270"
              y2="170"
              stroke="#e5e5e5"
              strokeWidth="2"
            />
            <line
              x1="40"
              y1="20"
              x2="40"
              y2="170"
              stroke="#e5e5e5"
              strokeWidth="2"
            />

            {/* 数据点 */}
            {relation.values.map((point, index) => {
              const x = 40 + ((point.x - xMin) / (xMax - xMin)) * 230
              const y = 170 - ((point.y - yMin) / (yMax - yMin)) * 150
              const colorIntensity =
                (point.objective - objMin) / (objMax - objMin)

              return (
                <circle
                  key={index}
                  cx={x}
                  cy={y}
                  r="4"
                  fill={`hsl(${colorIntensity * 120}, 70%, 50%)`}
                  opacity="0.7"
                />
              )
            })}

            {/* 坐标轴标签 */}
            <text x="150" y="195" textAnchor="middle" fontSize="12" fill="#666">
              {relation.param1}
            </text>
            <text
              x="15"
              y="100"
              textAnchor="middle"
              fontSize="12"
              fill="#666"
              transform="rotate(-90, 15, 100)"
            >
              {relation.param2}
            </text>
          </svg>
        </div>
        <div className="flex justify-center mt-4">
          <div className="flex items-center space-x-2 text-sm">
            <div
              className="w-4 h-4 rounded"
              style={{
                background:
                  'linear-gradient(to right, hsl(0, 70%, 50%), hsl(120, 70%, 50%))',
              }}
            ></div>
            <span className="text-gray-600">低目标值 → 高目标值</span>
          </div>
        </div>
      </Card>
    )
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="space-y-6">
        {/* 页面标题 */}
        <div>
          <h1 className="text-3xl font-bold text-gray-900">可视化分析</h1>
          <p className="mt-2 text-gray-600">
            通过图表分析超参数优化的进展和模式
          </p>
        </div>

        {error && <Alert variant="destructive">{error}</Alert>}

        {/* 实验选择 */}
        <Card className="p-6">
          <h2 className="text-lg font-semibold mb-4">选择实验</h2>
          {experiments.length === 0 ? (
            <div className="text-gray-500">暂无可用的实验数据</div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {experiments.map(experiment => (
                <div
                  key={experiment.id}
                  className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                    selectedExperiment === experiment.id
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                  onClick={() => setSelectedExperiment(experiment.id)}
                >
                  <div className="flex items-center justify-between mb-2">
                    <Badge
                      className={`${
                        experiment.status === 'completed'
                          ? 'bg-green-100 text-green-800'
                          : experiment.status === 'running'
                            ? 'bg-blue-100 text-blue-800'
                            : 'bg-gray-100 text-gray-800'
                      }`}
                    >
                      {experiment.status}
                    </Badge>
                    <Badge className="bg-purple-100 text-purple-800">
                      {experiment.algorithm.toUpperCase()}
                    </Badge>
                  </div>
                  <h3 className="font-medium text-gray-900 mb-1">
                    {experiment.name}
                  </h3>
                  <div className="text-sm text-gray-600">
                    {experiment.total_trials} 个试验
                    {experiment.best_value !== undefined && (
                      <span> • 最佳值: {experiment.best_value.toFixed(4)}</span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </Card>

        {/* 加载状态 */}
        {loading && (
          <div className="flex justify-center items-center h-32">
            <div className="text-gray-500">加载可视化数据中...</div>
          </div>
        )}

        {/* 可视化图表 */}
        {selectedExperiment && chartData && !loading && (
          <div className="space-y-6">
            {/* 优化历史 */}
            {renderOptimizationHistory()}

            {/* 参数重要性和关系 */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {renderParameterImportance()}
              {renderParameterRelations()}
            </div>

            {/* 外部可视化链接 */}
            {visualizations && (
              <Card className="p-6">
                <h2 className="text-lg font-semibold mb-4">高级可视化</h2>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <Button
                    variant="outline"
                    onClick={() =>
                      window.open(visualizations.optimization_history, '_blank')
                    }
                    className="h-20 flex-col"
                  >
                    <div className="text-sm font-medium">优化历史</div>
                    <div className="text-xs text-gray-500">详细视图</div>
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() =>
                      window.open(visualizations.param_importance, '_blank')
                    }
                    className="h-20 flex-col"
                  >
                    <div className="text-sm font-medium">参数重要性</div>
                    <div className="text-xs text-gray-500">详细视图</div>
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() =>
                      window.open(visualizations.parallel_coordinate, '_blank')
                    }
                    className="h-20 flex-col"
                  >
                    <div className="text-sm font-medium">平行坐标</div>
                    <div className="text-xs text-gray-500">多维分析</div>
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() =>
                      window.open(visualizations.contour, '_blank')
                    }
                    className="h-20 flex-col"
                  >
                    <div className="text-sm font-medium">等高线图</div>
                    <div className="text-xs text-gray-500">参数空间</div>
                  </Button>
                </div>
              </Card>
            )}
          </div>
        )}

        {/* 提示信息 */}
        {!selectedExperiment && experiments.length > 0 && (
          <Card className="p-8 text-center">
            <div className="text-gray-500 mb-2">
              请选择一个实验来查看可视化分析
            </div>
            <p className="text-sm text-gray-400">
              选择上方的实验卡片即可开始分析
            </p>
          </Card>
        )}
      </div>
    </div>
  )
}

export default HyperparameterVisualizationsPage
