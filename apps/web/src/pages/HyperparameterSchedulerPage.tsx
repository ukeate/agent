import React, { useEffect, useState } from 'react'
import { Card } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Badge } from '../components/ui/badge'
import { Alert } from '../components/ui/alert'
import { Input } from '../components/ui/input'
import { buildApiUrl, apiFetch } from '../utils/apiBase'

type Experiment = {
  id: string
  name: string
  status: string
  algorithm: string
  objective: string
}

type Trial = {
  id: string
  trial_number: number
  parameters: Record<string, any>
  value?: number
  state: string
}

type ResourceStats = {
  current_trials: number
  max_concurrent: number
  resource_usage: Record<string, number>
  active_trials: string[]
}

const API = buildApiUrl('/api/v1/hyperparameter-optimization')

const HyperparameterSchedulerPage: React.FC = () => {
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [selected, setSelected] = useState<string>('')
  const [trials, setTrials] = useState<Trial[]>([])
  const [resource, setResource] = useState<ResourceStats | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchJson = async <T,>(
    url: string,
    options?: RequestInit
  ): Promise<T> => {
    const res = await apiFetch(url, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    })
    return res.json()
  }

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const [expList, resStats] = await Promise.all([
        fetchJson<Experiment[]>(`${API}/experiments`),
        fetchJson<ResourceStats>(`${API}/resource-status`),
      ])
      setExperiments(expList)
      setResource(resStats)
      if (selected) {
        const ts = await fetchJson<Trial[]>(
          `${API}/experiments/${selected}/trials`
        )
        setTrials(ts)
      }
    } catch (e: any) {
      setError(e?.message || '加载失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  const loadTrials = async (experimentId: string) => {
    setSelected(experimentId)
    setLoading(true)
    setError(null)
    try {
      const ts = await fetchJson<Trial[]>(
        `${API}/experiments/${experimentId}/trials`
      )
      setTrials(ts)
    } catch (e: any) {
      setError(e?.message || '加载失败')
    } finally {
      setLoading(false)
    }
  }

  const startExperiment = async (id: string) => {
    setLoading(true)
    setError(null)
    try {
      await fetchJson(`${API}/experiments/${id}/start`, { method: 'POST' })
      await load()
    } catch (e: any) {
      setError(e?.message || '启动失败')
      setLoading(false)
    }
  }

  const stopExperiment = async (id: string) => {
    setLoading(true)
    setError(null)
    try {
      await fetchJson(`${API}/experiments/${id}/stop`, { method: 'POST' })
      await load()
    } catch (e: any) {
      setError(e?.message || '停止失败')
      setLoading(false)
    }
  }

  const formatState = (state: string) => state || 'unknown'

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">超参数调度</h2>
        <Button onClick={load} disabled={loading}>
          刷新
        </Button>
      </div>

      {error && <Alert variant="destructive">{error}</Alert>}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <Card className="p-4">
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-medium">实验列表</h3>
            <span className="text-xs text-gray-500">
              {experiments.length} 个
            </span>
          </div>
          <div className="space-y-2 max-h-[420px] overflow-auto">
            {experiments.map(exp => (
              <div
                key={exp.id}
                className={`p-3 border rounded-md ${selected === exp.id ? 'border-blue-500' : 'border-gray-200'}`}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-semibold">{exp.name}</div>
                    <div className="text-xs text-gray-500">
                      {exp.algorithm} · {exp.objective}
                    </div>
                  </div>
                  <Badge className="bg-blue-100 text-blue-800">
                    {formatState(exp.status)}
                  </Badge>
                </div>
                <div className="mt-2 flex gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => loadTrials(exp.id)}
                  >
                    查看
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => startExperiment(exp.id)}
                  >
                    启动
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => stopExperiment(exp.id)}
                  >
                    停止
                  </Button>
                </div>
              </div>
            ))}
            {experiments.length === 0 && (
              <div className="text-sm text-gray-500">
                暂无实验，请先通过 API 创建实验。
              </div>
            )}
          </div>
        </Card>

        <Card className="p-4 lg:col-span-2">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-medium">试验明细</h3>
            <Input
              placeholder="输入实验ID并回车"
              value={selected}
              onChange={e => setSelected(e.target.value)}
              onPressEnter={e =>
                loadTrials((e.target as HTMLInputElement).value)
              }
            />
          </div>
          {selected === '' && (
            <div className="text-sm text-gray-500">
              选择或输入实验ID以查看试验。
            </div>
          )}
          {selected !== '' && (
            <div className="space-y-2 max-h-[520px] overflow-auto">
              {trials.map(t => (
                <div key={t.id} className="p-3 border rounded-md">
                  <div className="flex items-center justify-between">
                    <div className="font-semibold">试验 #{t.trial_number}</div>
                    <Badge className="bg-green-100 text-green-800">
                      {formatState(t.state)}
                    </Badge>
                  </div>
                  <div className="text-xs text-gray-600 mt-1">
                    结果: {t.value ?? '暂无'}
                  </div>
                  <div className="text-xs text-gray-600 mt-1">
                    参数:{' '}
                    {Object.entries(t.parameters || {})
                      .map(([k, v]) => `${k}=${v}`)
                      .join(', ') || '无'}
                  </div>
                </div>
              ))}
              {trials.length === 0 && (
                <div className="text-sm text-gray-500">未查询到试验记录。</div>
              )}
            </div>
          )}
        </Card>

        <Card className="p-4">
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-medium">资源状态</h3>
            <Button
              size="sm"
              variant="outline"
              onClick={load}
              disabled={loading}
            >
              刷新
            </Button>
          </div>
          {resource ? (
            <div className="space-y-2 text-sm text-gray-700">
              <div>
                当前试验: {resource.current_trials} / {resource.max_concurrent}
              </div>
              <div>
                活跃试验ID: {resource.active_trials?.join(', ') || '无'}
              </div>
              <div className="space-y-1">
                {Object.entries(resource.resource_usage || {}).map(([k, v]) => (
                  <div key={k}>
                    {k}: {v}%
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="text-sm text-gray-500">暂无资源数据。</div>
          )}
        </Card>
      </div>
    </div>
  )
}

export default HyperparameterSchedulerPage
