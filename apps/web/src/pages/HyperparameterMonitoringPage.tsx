import React, { useEffect, useState } from 'react'
import { Card } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Alert } from '../components/ui/alert'
import { Table } from 'antd'
import { buildApiUrl, apiFetch } from '../utils/apiBase'

type ResourceStats = {
  current_trials: number
  max_concurrent: number
  resource_usage?: Record<string, number>
  active_trials?: string[]
}
type ActiveExperiment = {
  id: string
  status?: string
  algorithm?: string
  objective?: string
}

const API = buildApiUrl('/api/v1/hyperparameter-optimization')

const HyperparameterMonitoringPage: React.FC = () => {
  const [resource, setResource] = useState<ResourceStats | null>(null)
  const [experiments, setExperiments] = useState<ActiveExperiment[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const [rRes, eRes] = await Promise.all([
        apiFetch(`${API}/resource-status`),
        apiFetch(`${API}/active-experiments`),
      ])
      setResource(await rRes.json())
      setExperiments(await eRes.json())
    } catch (e: any) {
      setError(e?.message || '加载失败')
      setResource(null)
      setExperiments([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h2 className="text-lg font-semibold">超参数监控</h2>
        <Button onClick={load} disabled={loading}>
          刷新
        </Button>
      </div>
      {error && <Alert variant="destructive">{error}</Alert>}

      <Card className="p-4">
        <h3 className="font-medium mb-2">资源状态</h3>
        {resource ? (
          <div className="text-sm text-gray-700 space-y-1">
            <div>
              当前试验: {resource.current_trials} / {resource.max_concurrent}
            </div>
            <div>活跃试验ID: {resource.active_trials?.join(', ') || '无'}</div>
            <div>资源占用:</div>
            <ul className="list-disc ml-5">
              {Object.entries(resource.resource_usage || {}).map(([k, v]) => (
                <li key={k}>
                  {k}: {v}%
                </li>
              ))}
            </ul>
          </div>
        ) : (
          <div className="text-sm text-gray-500">暂无数据</div>
        )}
      </Card>

      <Card className="p-4">
        <h3 className="font-medium mb-2">活跃实验</h3>
        <Table
          rowKey="id"
          loading={loading}
          dataSource={experiments}
          locale={{ emptyText: '暂无活跃实验' }}
          columns={[
            { title: 'ID', dataIndex: 'id' },
            { title: '状态', dataIndex: 'status' },
            { title: '算法', dataIndex: 'algorithm' },
            { title: '目标', dataIndex: 'objective' },
          ]}
        />
      </Card>
    </div>
  )
}

export default HyperparameterMonitoringPage
