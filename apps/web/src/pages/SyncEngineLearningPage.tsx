import React, { useEffect, useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card'
import { Badge } from '../components/ui/badge'
import { Button } from '../components/ui/button'
import { Progress } from '../components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs'
import {
  AlertTriangle,
  CheckCircle,
  RefreshCw,
  Activity,
  Clock,
} from 'lucide-react'
import { buildApiUrl, apiFetch } from '../utils/apiBase'

interface OfflineStatus {
  mode: string
  network_status: string
  connection_quality: number
  pending_operations: number
  has_conflicts: boolean
  sync_in_progress: boolean
  last_sync_at?: string
}

interface Operation {
  id: string
  operation_type: string
  table_name: string
  object_id: string
  timestamp: string
  is_synced: boolean
  retry_count: number
}

const SyncEngineLearningPage: React.FC = () => {
  const [status, setStatus] = useState<OfflineStatus | null>(null)
  const [operations, setOperations] = useState<Operation[]>([])
  const [stats, setStats] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState('ops')

  const loadData = async () => {
    setLoading(true)
    setError(null)
    try {
      const [s, ops, st] = await Promise.all([
        fetchJson('/api/v1/offline/status'),
        fetchJson('/api/v1/offline/operations'),
        fetchJson('/api/v1/offline/statistics'),
      ])
      setStatus(s)
      setOperations(ops || [])
      setStats(st)
    } catch (e: any) {
      setError(e?.message || '加载失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
    const timer = setInterval(loadData, 5000)
    return () => clearInterval(timer)
  }, [])

  if (loading) return <div className="p-6">同步引擎加载中...</div>
  if (error) return <div className="p-6 text-red-600">错误: {error}</div>

  return (
    <div className="p-6 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">增量同步引擎监控</h1>
          <p className="text-gray-600">
            所有数据来自 /api/v1/offline 真实接口，无本地模拟
          </p>
        </div>
        <Button onClick={loadData} variant="default" className="gap-2">
          <RefreshCw size={16} />
          刷新
        </Button>
      </div>

      {status && (
        <Card>
          <CardHeader>
            <CardTitle>状态</CardTitle>
          </CardHeader>
          <CardContent className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
            <Stat label="模式" value={status.mode} />
            <Stat label="网络" value={status.network_status} />
            <Stat
              label="连接质量"
              value={`${Math.round((status.connection_quality || 0) * 100)}%`}
            />
            <Stat label="待同步" value={status.pending_operations} />
            <Stat label="冲突" value={status.has_conflicts ? '是' : '否'} />
            <Stat
              label="同步中"
              value={status.sync_in_progress ? '进行中' : '空闲'}
            />
            <Stat
              label="上次同步"
              value={
                status.last_sync_at
                  ? new Date(status.last_sync_at).toLocaleString()
                  : '—'
              }
            />
          </CardContent>
        </Card>
      )}

      {stats && (
        <Card>
          <CardHeader>
            <CardTitle>统计</CardTitle>
          </CardHeader>
          <CardContent className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <Stat label="同步总量" value={stats.total_synced_operations || 0} />
            <Stat label="失败次数" value={stats.total_failed_operations || 0} />
            <Stat
              label="冲突解决"
              value={stats.total_conflicts_resolved || 0}
            />
            <Stat
              label="平均吞吐"
              value={`${stats.average_throughput || 0}/s`}
            />
            <Stat
              label="网络使用"
              value={`${stats.network_usage_mb || 0} MB`}
            />
            <Stat
              label="同步效率"
              value={`${Math.round((stats.sync_efficiency || 0) * 100)}%`}
            />
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>最近操作</CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList>
              <TabsTrigger value="ops">操作</TabsTrigger>
            </TabsList>
            <TabsContent value="ops">
              <div className="space-y-2">
                {operations.slice(0, 20).map(op => (
                  <div
                    key={op.id}
                    className="border rounded p-3 flex justify-between items-center text-sm"
                  >
                    <div className="space-y-1">
                      <div className="font-semibold">
                        {op.table_name} / {op.object_id}
                      </div>
                      <div className="text-gray-500">
                        {new Date(op.timestamp).toLocaleString()}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge>{op.operation_type}</Badge>
                      <Badge variant={op.is_synced ? 'default' : 'destructive'}>
                        {op.is_synced ? '已同步' : '待同步'}
                      </Badge>
                      <Badge variant="secondary">重试 {op.retry_count}</Badge>
                    </div>
                  </div>
                ))}
                {operations.length === 0 && (
                  <div className="text-gray-500 text-sm">暂无操作记录</div>
                )}
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  )
}

const Stat = ({ label, value }: { label: string; value: any }) => (
  <div className="bg-gray-50 rounded p-3">
    <div className="text-gray-500 text-xs">{label}</div>
    <div className="text-base font-semibold flex items-center gap-1">
      <CheckCircle size={14} className="text-green-500" />
      {value}
    </div>
  </div>
)

async function fetchJson(url: string) {
  const res = await apiFetch(buildApiUrl(url))
  return res.json()
}

export default SyncEngineLearningPage
