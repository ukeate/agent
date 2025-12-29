import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Tabs, Typography, Row, Col, Button, Alert, Progress, Statistic, Space } from 'antd'
import { ReloadOutlined, DatabaseOutlined, BranchesOutlined } from '@ant-design/icons'

const { Title } = Typography
const { TabPane } = Tabs

type SyncTask = {
  task_id: string
  session_id: string
  direction: string
  priority: number
  status: string
  progress?: number
  total_operations?: number
  completed_operations?: number
  failed_operations?: number
  created_at?: string
  started_at?: string
  completed_at?: string
}

type SyncStatistics = {
  active_tasks?: number
  queued_tasks?: number
  total_tasks?: number
  status_distribution?: Record<string, number>
  priority_distribution?: Record<string, number>
  total_synced_operations?: number
  total_failed_operations?: number
  total_conflicts_resolved?: number
  last_sync_time?: string | null
  sync_efficiency?: number
}

type VectorClockStats = {
  total_syncs?: number
  conflicts_detected?: number
  conflict_rate?: number
  active_nodes?: number
  recent_sync_time?: string | null
}

type DeltaStats = {
  total_deltas?: number
  total_original_size?: number
  total_compressed_size?: number
  average_compression_ratio?: number
  compression_algorithms_used?: string[]
}

const SyncManagementPage: React.FC = () => {
  const [tasks, setTasks] = useState<SyncTask[]>([])
  const [stats, setStats] = useState<SyncStatistics | null>(null)
  const [clockStats, setClockStats] = useState<VectorClockStats | null>(null)
  const [deltaStats, setDeltaStats] = useState<DeltaStats | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const [tRes, sRes, cRes, dRes] = await Promise.all([
        apiFetch(buildApiUrl('/api/v1/offline/sync/tasks'),
        apiFetch(buildApiUrl('/api/v1/offline/sync/statistics'),
        apiFetch(buildApiUrl('/api/v1/offline/vector-clocks'),
        apiFetch(buildApiUrl('/api/v1/offline/deltas/statistics'))
      ])
      const tData = await tRes.json()
      const sData = await sRes.json()
      const cData = await cRes.json()
      const dData = await dRes.json()
      setTasks(Array.isArray(tData?.tasks) ? tData.tasks : [])
      setStats(sData || null)
      setClockStats(cData?.stats || null)
      setDeltaStats(dData || null)
    } catch (e: any) {
      setError(e?.message || '加载失败')
      setTasks([])
      setStats(null)
      setClockStats(null)
      setDeltaStats(null)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Space align="center" style={{ justifyContent: 'space-between', width: '100%' }}>
          <Title level={3} style={{ margin: 0 }}>
            同步管理
          </Title>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        </Space>

        {error && <Alert type="error" message={error} />}

        <Row gutter={16}>
          <Col span={6}>
            <Card>
              <Statistic title="任务总数" value={stats?.total_tasks ?? 0} prefix={<DatabaseOutlined />} />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic title="活跃任务" value={stats?.active_tasks ?? 0} />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic title="已解决冲突" value={stats?.total_conflicts_resolved ?? 0} />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic title="最近同步" value={stats?.last_sync_time || '-'} />
            </Card>
          </Col>
        </Row>

        <Card title="任务列表">
          <Tabs>
            <TabPane tab="全部任务" key="all">
              {tasks.length === 0 ? (
                <Alert type="info" message="暂无同步任务" />
              ) : (
                tasks.map((task) => (
                  <Card key={task.task_id} style={{ marginBottom: 12 }}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Space>
                        <BranchesOutlined />
                        <span>{task.task_id}</span>
                        <Tag color="blue">{task.direction}</Tag>
                        <Tag color="orange">{task.status}</Tag>
                      </Space>
                      <Progress percent={Math.round((task.progress || 0) * 100)} />
                      <div>会话: {task.session_id}</div>
                      <div>创建: {task.created_at}</div>
                    </Space>
                  </Card>
                ))
              )}
            </TabPane>
          </Tabs>
        </Card>

        <Card title="向量时钟">
          {clockStats ? (
            <Space size="large">
              <Statistic title="总同步次数" value={clockStats.total_syncs ?? 0} />
              <Statistic title="冲突次数" value={clockStats.conflicts_detected ?? 0} />
              <Statistic title="冲突率" value={(clockStats.conflict_rate ?? 0) * 100} suffix="%" />
              <Statistic title="活跃节点" value={clockStats.active_nodes ?? 0} />
            </Space>
          ) : (
            <Alert type="info" message="暂无向量时钟统计" />
          )}
        </Card>

        <Card title="增量统计">
          {deltaStats ? (
            <Space size="large">
              <Statistic title="增量总数" value={deltaStats.total_deltas ?? 0} />
              <Statistic title="平均压缩率" value={(deltaStats.average_compression_ratio ?? 0) * 100} suffix="%" />
            </Space>
          ) : (
            <Alert type="info" message="暂无增量统计" />
          )}
        </Card>
      </Space>
    </div>
  )
}

export default SyncManagementPage
