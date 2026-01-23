import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'

import { logger } from '../utils/logger'
const FlowControlPage: React.FC = () => {
  const [metrics, setMetrics] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const loadData = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await apiFetch(
        buildApiUrl('/api/v1/streaming/flow-control/metrics')
      )
      const data = await res.json()
      setMetrics(data.flow_control_metrics || data)
    } catch (e: any) {
      logger.error('获取流控数据失败', e)
      setError(e?.message || '获取流控数据失败')
      setMetrics(null)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
    const timer = setInterval(loadData, 5000)
    return () => clearInterval(timer)
  }, [])

  if (loading) return <div className="p-6">加载流控监控系统...</div>
  if (error) return <div className="p-6 text-red-600">失败: {error}</div>
  if (!metrics) return <div className="p-6 text-red-600">无数据</div>

  const backpressure = metrics.backpressure_status || {}
  const circuit = metrics.circuit_breaker_state || {}
  const queues = metrics.queue_metrics || {}
  const flowStats = metrics.flow_stats || {}

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">流控与背压监控</h1>
          <p className="text-gray-600">
            展示后端实时流控/背压状态，来自 streaming flow-control 真实接口
          </p>
        </div>
        <button
          className="px-3 py-1 rounded bg-blue-600 text-white"
          onClick={loadData}
        >
          刷新
        </button>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard title="当前会话" value={metrics.current_sessions ?? 0} />
        <StatCard
          title="最大并发"
          value={metrics.max_concurrent_sessions ?? 0}
        />
        <StatCard
          title="背压启用"
          value={metrics.backpressure_enabled ? '是' : '否'}
        />
        <StatCard
          title="当前吞吐"
          value={flowStats.current_throughput ?? 0}
          suffix="/s"
        />
      </div>

      <div className="bg-white rounded shadow p-4">
        <h2 className="text-lg font-semibold mb-2">背压状态</h2>
        <pre className="bg-gray-50 rounded p-3 text-sm overflow-auto">
          {JSON.stringify(backpressure, null, 2)}
        </pre>
      </div>

      <div className="bg-white rounded shadow p-4">
        <h2 className="text-lg font-semibold mb-2">熔断器状态</h2>
        <pre className="bg-gray-50 rounded p-3 text-sm overflow-auto">
          {JSON.stringify(circuit, null, 2)}
        </pre>
      </div>

      <div className="bg-white rounded shadow p-4">
        <h2 className="text-lg font-semibold mb-2">队列监控</h2>
        <pre className="bg-gray-50 rounded p-3 text-sm overflow-auto">
          {JSON.stringify(queues, null, 2)}
        </pre>
      </div>
    </div>
  )
}

const StatCard = ({
  title,
  value,
  suffix,
}: {
  title: string
  value: any
  suffix?: string
}) => (
  <div className="bg-white rounded shadow p-3">
    <div className="text-gray-500 text-sm">{title}</div>
    <div className="text-xl font-bold">
      {value}
      {suffix}
    </div>
  </div>
)

export default FlowControlPage
