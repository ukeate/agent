import React, { useState, useEffect } from 'react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '../components/ui/card'
import { Alert } from '../components/ui/alert'
import {
  modelEvaluationService,
  EvaluationRequest,
  BenchmarkRequest,
  EvaluationResult,
  BenchmarkResult,
  ModelComparison,
  EvaluationHistory,
  PerformanceMetrics,
} from '../services/modelEvaluationService'

interface EvaluationFormData {
  model_name: string
  model_path: string
  task_type: string
  device: string
  batch_size: number
  max_length: number
  precision: string
  enable_optimizations: boolean
}

interface BenchmarkFormData {
  name: string
  tasks: string[]
  num_fewshot: number
  limit?: number
  batch_size: number
  device: string
}

const ModelEvaluationManagementPage: React.FC = () => {
  // 状态管理
  const [activeTab, setActiveTab] = useState<
    'overview' | 'evaluate' | 'benchmark' | 'compare' | 'history' | 'reports'
  >('overview')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  // 概览数据
  const [evaluationHistory, setEvaluationHistory] =
    useState<EvaluationHistory | null>(null)
  const [availableBenchmarks, setAvailableBenchmarks] = useState<any[]>([])

  // 表单数据
  const [evaluationForm, setEvaluationForm] = useState<EvaluationFormData>({
    model_name: '',
    model_path: '',
    task_type: 'text_generation',
    device: 'auto',
    batch_size: 8,
    max_length: 512,
    precision: 'fp16',
    enable_optimizations: true,
  })

  const [benchmarkForm, setBenchmarkForm] = useState<BenchmarkFormData>({
    name: '',
    tasks: [],
    num_fewshot: 0,
    limit: 100,
    batch_size: 8,
    device: 'auto',
  })

  // 比较数据
  const [selectedModels, setSelectedModels] = useState<string[]>([])
  const [compareResults, setCompareResults] = useState<ModelComparison | null>(
    null
  )

  // 性能指标
  const [performanceMetrics, setPerformanceMetrics] =
    useState<PerformanceMetrics | null>(null)
  const [monitoredModel, setMonitoredModel] = useState<string>('')

  // 初始化数据
  useEffect(() => {
    loadInitialData()
  }, [])

  const loadInitialData = async () => {
    setLoading(true)
    try {
      const [history, benchmarks] = await Promise.all([
        modelEvaluationService.getEvaluationHistory(),
        modelEvaluationService.listAvailableBenchmarks(),
      ])

      setEvaluationHistory(history)
      setAvailableBenchmarks(benchmarks)
    } catch (err) {
      setError('加载初始数据失败: ' + (err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  // 处理评估提交
  const handleEvaluationSubmit = async () => {
    setLoading(true)
    setError(null)
    setSuccess(null)

    try {
      const result =
        await modelEvaluationService.startEvaluation(evaluationForm)
      setSuccess(`评估任务已创建: ${result.evaluation_id}`)

      // 重新加载历史记录
      const updatedHistory = await modelEvaluationService.getEvaluationHistory()
      setEvaluationHistory(updatedHistory)
    } catch (err) {
      setError('提交评估任务失败: ' + (err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  // 处理基准测试提交
  const handleBenchmarkSubmit = async () => {
    setLoading(true)
    setError(null)
    setSuccess(null)

    try {
      const result = await modelEvaluationService.runBenchmark(benchmarkForm)
      setSuccess(`基准测试任务已创建: ${result.benchmark_id}`)
    } catch (err) {
      setError('提交基准测试失败: ' + (err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  // 处理模型比较
  const handleCompareModels = async () => {
    if (selectedModels.length < 2) {
      setError('请至少选择两个模型进行比较')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const result = await modelEvaluationService.compareModels({
        models: selectedModels,
        benchmarks: availableBenchmarks.slice(0, 3).map(b => b.name),
      })

      setCompareResults(result)
      setSuccess('模型比较完成')
    } catch (err) {
      setError('模型比较失败: ' + (err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  // 获取性能指标
  const handleGetPerformance = async () => {
    if (!monitoredModel) {
      setError('请输入要监控的模型名称')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const metrics =
        await modelEvaluationService.getPerformanceMetrics(monitoredModel)
      setPerformanceMetrics(metrics)
      setSuccess('性能指标获取成功')
    } catch (err) {
      setError('获取性能指标失败: ' + (err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  // 概览视图
  const renderOverview = () => (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <Card>
        <CardHeader>
          <CardTitle>评估历史</CardTitle>
          <CardDescription>最近的模型评估记录</CardDescription>
        </CardHeader>
        <CardContent>
          {evaluationHistory ? (
            <div className="space-y-2">
              <p>总评估次数: {evaluationHistory.total}</p>
              <div className="space-y-1">
                {evaluationHistory.evaluations
                  .slice(0, 5)
                  .map((evaluation, index) => (
                    <div
                      key={evaluation.id}
                      className="flex justify-between text-sm"
                    >
                      <span>{evaluation.model_name}</span>
                      <span
                        className={`px-2 py-1 rounded ${
                          evaluation.status === 'completed'
                            ? 'bg-green-100 text-green-800'
                            : evaluation.status === 'failed'
                              ? 'bg-red-100 text-red-800'
                              : 'bg-yellow-100 text-yellow-800'
                        }`}
                      >
                        {evaluation.status}
                      </span>
                    </div>
                  ))}
              </div>
            </div>
          ) : (
            <p>暂无评估历史</p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>可用基准测试</CardTitle>
          <CardDescription>系统支持的基准测试</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {availableBenchmarks.slice(0, 5).map((benchmark, index) => (
              <div key={index} className="border p-2 rounded">
                <h4 className="font-medium">{benchmark.name}</h4>
                <p className="text-sm text-gray-600">{benchmark.description}</p>
                <p className="text-xs text-gray-500">
                  难度: {benchmark.difficulty} | 预计时间:{' '}
                  {benchmark.estimated_duration}分钟
                </p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )

  // 评估表单视图
  const renderEvaluationForm = () => (
    <Card>
      <CardHeader>
        <CardTitle>模型评估</CardTitle>
        <CardDescription>创建新的模型评估任务</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-1">模型名称</label>
            <input
              type="text"
              className="w-full p-2 border rounded"
              value={evaluationForm.model_name}
              onChange={e =>
                setEvaluationForm(prev => ({
                  ...prev,
                  model_name: e.target.value,
                }))
              }
              placeholder="输入模型名称"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">模型路径</label>
            <input
              type="text"
              className="w-full p-2 border rounded"
              value={evaluationForm.model_path}
              onChange={e =>
                setEvaluationForm(prev => ({
                  ...prev,
                  model_path: e.target.value,
                }))
              }
              placeholder="输入模型路径或HuggingFace模型ID"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">任务类型</label>
            <select
              className="w-full p-2 border rounded"
              value={evaluationForm.task_type}
              onChange={e =>
                setEvaluationForm(prev => ({
                  ...prev,
                  task_type: e.target.value,
                }))
              }
            >
              <option value="text_generation">文本生成</option>
              <option value="text_classification">文本分类</option>
              <option value="question_answering">问答</option>
              <option value="summarization">摘要</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">设备</label>
            <select
              className="w-full p-2 border rounded"
              value={evaluationForm.device}
              onChange={e =>
                setEvaluationForm(prev => ({ ...prev, device: e.target.value }))
              }
            >
              <option value="auto">自动</option>
              <option value="cpu">CPU</option>
              <option value="cuda">GPU (CUDA)</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">批次大小</label>
            <input
              type="number"
              className="w-full p-2 border rounded"
              value={evaluationForm.batch_size}
              onChange={e =>
                setEvaluationForm(prev => ({
                  ...prev,
                  batch_size: parseInt(e.target.value),
                }))
              }
              min="1"
              max="32"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">最大长度</label>
            <input
              type="number"
              className="w-full p-2 border rounded"
              value={evaluationForm.max_length}
              onChange={e =>
                setEvaluationForm(prev => ({
                  ...prev,
                  max_length: parseInt(e.target.value),
                }))
              }
              min="128"
              max="2048"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">精度</label>
            <select
              className="w-full p-2 border rounded"
              value={evaluationForm.precision}
              onChange={e =>
                setEvaluationForm(prev => ({
                  ...prev,
                  precision: e.target.value,
                }))
              }
            >
              <option value="fp16">FP16</option>
              <option value="fp32">FP32</option>
              <option value="int8">INT8</option>
            </select>
          </div>
          <div>
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={evaluationForm.enable_optimizations}
                onChange={e =>
                  setEvaluationForm(prev => ({
                    ...prev,
                    enable_optimizations: e.target.checked,
                  }))
                }
              />
              <span className="text-sm font-medium">启用优化</span>
            </label>
          </div>
        </div>

        <button
          onClick={handleEvaluationSubmit}
          disabled={
            loading || !evaluationForm.model_name || !evaluationForm.model_path
          }
          className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? '提交中...' : '开始评估'}
        </button>
      </CardContent>
    </Card>
  )

  // 基准测试表单视图
  const renderBenchmarkForm = () => (
    <Card>
      <CardHeader>
        <CardTitle>基准测试</CardTitle>
        <CardDescription>运行标准基准测试</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-1">
              基准测试名称
            </label>
            <select
              className="w-full p-2 border rounded"
              value={benchmarkForm.name}
              onChange={e =>
                setBenchmarkForm(prev => ({ ...prev, name: e.target.value }))
              }
            >
              <option value="">选择基准测试</option>
              {availableBenchmarks.map((benchmark, index) => (
                <option key={index} value={benchmark.name}>
                  {benchmark.name}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">
              Few-shot 样本数
            </label>
            <input
              type="number"
              className="w-full p-2 border rounded"
              value={benchmarkForm.num_fewshot}
              onChange={e =>
                setBenchmarkForm(prev => ({
                  ...prev,
                  num_fewshot: parseInt(e.target.value),
                }))
              }
              min="0"
              max="10"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">样本限制</label>
            <input
              type="number"
              className="w-full p-2 border rounded"
              value={benchmarkForm.limit || ''}
              onChange={e =>
                setBenchmarkForm(prev => ({
                  ...prev,
                  limit: parseInt(e.target.value) || undefined,
                }))
              }
              placeholder="不限制"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">批次大小</label>
            <input
              type="number"
              className="w-full p-2 border rounded"
              value={benchmarkForm.batch_size}
              onChange={e =>
                setBenchmarkForm(prev => ({
                  ...prev,
                  batch_size: parseInt(e.target.value),
                }))
              }
              min="1"
              max="32"
            />
          </div>
        </div>

        <button
          onClick={handleBenchmarkSubmit}
          disabled={loading || !benchmarkForm.name}
          className="mt-4 px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
        >
          {loading ? '运行中...' : '开始基准测试'}
        </button>
      </CardContent>
    </Card>
  )

  // 模型比较视图
  const renderModelComparison = () => (
    <Card>
      <CardHeader>
        <CardTitle>模型比较</CardTitle>
        <CardDescription>对比多个模型的性能</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">
              选择要比较的模型
            </label>
            <div className="space-y-2">
              {evaluationHistory?.evaluations.slice(0, 5).map(evaluation => (
                <label
                  key={evaluation.id}
                  className="flex items-center space-x-2"
                >
                  <input
                    type="checkbox"
                    checked={selectedModels.includes(evaluation.model_name)}
                    onChange={e => {
                      if (e.target.checked) {
                        setSelectedModels(prev => [
                          ...prev,
                          evaluation.model_name,
                        ])
                      } else {
                        setSelectedModels(prev =>
                          prev.filter(name => name !== evaluation.model_name)
                        )
                      }
                    }}
                  />
                  <span>{evaluation.model_name}</span>
                  <span
                    className={`px-2 py-1 text-xs rounded ${
                      evaluation.status === 'completed'
                        ? 'bg-green-100 text-green-800'
                        : 'bg-gray-100 text-gray-800'
                    }`}
                  >
                    {evaluation.status}
                  </span>
                </label>
              ))}
            </div>
          </div>

          <button
            onClick={handleCompareModels}
            disabled={loading || selectedModels.length < 2}
            className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:opacity-50"
          >
            {loading ? '比较中...' : '开始比较'}
          </button>

          {compareResults && (
            <div className="mt-4 p-4 border rounded">
              <h3 className="font-medium mb-2">比较结果</h3>
              <p className="mb-2">
                获胜者: <strong>{compareResults.winner}</strong>
              </p>
              <div className="space-y-2">
                <h4 className="font-medium">各任务最佳模型:</h4>
                {Object.entries(compareResults.summary.best_per_task).map(
                  ([task, model]) => (
                    <div key={task} className="flex justify-between">
                      <span>{task}:</span>
                      <span className="font-medium">{model}</span>
                    </div>
                  )
                )}
              </div>
              {compareResults.summary.recommendations.length > 0 && (
                <div className="mt-2">
                  <h4 className="font-medium">建议:</h4>
                  <ul className="list-disc list-inside">
                    {compareResults.summary.recommendations.map(
                      (rec, index) => (
                        <li key={index} className="text-sm">
                          {rec}
                        </li>
                      )
                    )}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )

  // 性能监控视图
  const renderPerformanceMonitoring = () => (
    <Card>
      <CardHeader>
        <CardTitle>性能监控</CardTitle>
        <CardDescription>实时监控模型性能指标</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">模型名称</label>
            <input
              type="text"
              className="w-full p-2 border rounded"
              value={monitoredModel}
              onChange={e => setMonitoredModel(e.target.value)}
              placeholder="输入要监控的模型名称"
            />
          </div>

          <button
            onClick={handleGetPerformance}
            disabled={loading || !monitoredModel}
            className="px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700 disabled:opacity-50"
          >
            {loading ? '获取中...' : '获取性能指标'}
          </button>

          {performanceMetrics && (
            <div className="grid grid-cols-2 gap-4 mt-4">
              <div className="p-4 border rounded">
                <h3 className="font-medium mb-2">延迟指标</h3>
                <div className="space-y-1 text-sm">
                  <div>P50: {performanceMetrics.latency.p50}ms</div>
                  <div>P95: {performanceMetrics.latency.p95}ms</div>
                  <div>P99: {performanceMetrics.latency.p99}ms</div>
                  <div>平均: {performanceMetrics.latency.mean}ms</div>
                </div>
              </div>

              <div className="p-4 border rounded">
                <h3 className="font-medium mb-2">吞吐量指标</h3>
                <div className="space-y-1 text-sm">
                  <div>
                    请求/秒: {performanceMetrics.throughput.requests_per_second}
                  </div>
                  <div>
                    令牌/秒: {performanceMetrics.throughput.tokens_per_second}
                  </div>
                  <div>错误率: {performanceMetrics.error_rate}%</div>
                </div>
              </div>

              <div className="p-4 border rounded">
                <h3 className="font-medium mb-2">资源使用</h3>
                <div className="space-y-1 text-sm">
                  <div>
                    CPU: {performanceMetrics.resource_usage.cpu_percent}%
                  </div>
                  <div>
                    内存: {performanceMetrics.resource_usage.memory_mb}MB
                  </div>
                  {performanceMetrics.resource_usage.gpu_percent && (
                    <>
                      <div>
                        GPU: {performanceMetrics.resource_usage.gpu_percent}%
                      </div>
                      <div>
                        GPU内存:{' '}
                        {performanceMetrics.resource_usage.gpu_memory_mb}MB
                      </div>
                    </>
                  )}
                </div>
              </div>

              <div className="p-4 border rounded">
                <h3 className="font-medium mb-2">监控信息</h3>
                <div className="space-y-1 text-sm">
                  <div>模型: {performanceMetrics.model_name}</div>
                  <div>
                    时间:{' '}
                    {new Date(performanceMetrics.timestamp).toLocaleString()}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">模型评估管理</h1>
        <p className="text-gray-600">管理和监控AI模型的评估任务</p>
      </div>

      {/* 错误和成功提示 */}
      {error && (
        <Alert className="mb-4 border-red-200 bg-red-50">
          <div className="text-red-800">{error}</div>
        </Alert>
      )}

      {success && (
        <Alert className="mb-4 border-green-200 bg-green-50">
          <div className="text-green-800">{success}</div>
        </Alert>
      )}

      {/* 标签导航 */}
      <div className="mb-6 border-b">
        <nav className="flex space-x-8">
          {[
            { key: 'overview', label: '概览' },
            { key: 'evaluate', label: '模型评估' },
            { key: 'benchmark', label: '基准测试' },
            { key: 'compare', label: '模型比较' },
            { key: 'history', label: '性能监控' },
          ].map(tab => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key as any)}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === tab.key
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {/* 内容区域 */}
      <div className="space-y-6">
        {activeTab === 'overview' && renderOverview()}
        {activeTab === 'evaluate' && renderEvaluationForm()}
        {activeTab === 'benchmark' && renderBenchmarkForm()}
        {activeTab === 'compare' && renderModelComparison()}
        {activeTab === 'history' && renderPerformanceMonitoring()}
      </div>

      {loading && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white p-4 rounded-lg">
            <div className="flex items-center space-x-2">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
              <span>处理中...</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default ModelEvaluationManagementPage
