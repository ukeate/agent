import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Space, Typography, Button, Alert, Table, Spin } from 'antd'
import { ReloadOutlined, ThunderboltOutlined } from '@ant-design/icons'

type Job = { job_id: string; status: string; started_at?: string; finished_at?: string; updated_nodes?: number }
type Conflict = { id: string; status: string; description?: string; resolved_at?: string }
type Metrics = { last_update?: string; total_nodes?: number; total_edges?: number }

const KnowledgeGraphIncrementalUpdate: React.FC = () => {
  const [jobs, setJobs] = useState<Job[]>([])
  const [conflicts, setConflicts] = useState<Conflict[]>([])
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const [jobRes, conflictRes, metricRes] = await Promise.all([
        apiFetch(buildApiUrl('/api/v1/knowledge-graph/update/jobs'),
        apiFetch(buildApiUrl('/api/v1/knowledge-graph/update/conflicts'),
        apiFetch(buildApiUrl('/api/v1/knowledge-graph/update/metrics'))
      ])
      const jobData = await jobRes.json()
      const conflictData = await conflictRes.json()
      const metricData = await metricRes.json()
      setJobs(Array.isArray(jobData?.jobs) ? jobData.jobs : [])
      setConflicts(Array.isArray(conflictData?.conflicts) ? conflictData.conflicts : [])
      setMetrics(metricData || null)
    } catch (e: any) {
      setError(e?.message || '加载失败')
      setJobs([])
      setConflicts([])
      setMetrics(null)
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
          <Space>
            <ThunderboltOutlined />
            <Typography.Title level={3} style={{ margin: 0 }}>
              知识图增量更新
            </Typography.Title>
          </Space>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        </Space>

        {error && <Alert type="error" message="加载失败" description={error} />}

        <Card title="更新作业">
          {loading ? (
            <Spin />
          ) : (
            <Table
              rowKey="job_id"
              dataSource={jobs}
              columns={[
                { title: '作业ID', dataIndex: 'job_id' },
                { title: '状态', dataIndex: 'status' },
                { title: '开始时间', dataIndex: 'started_at' },
                { title: '结束时间', dataIndex: 'finished_at' },
                { title: '更新节点数', dataIndex: 'updated_nodes' }
              ]}
              locale={{ emptyText: '暂无作业，先触发 /api/v1/knowledge-graph/update/start。' }}
            />
          )}
        </Card>

        <Card title="冲突列表">
          {loading ? (
            <Spin />
          ) : (
            <Table
              rowKey="id"
              dataSource={conflicts}
              columns={[
                { title: 'ID', dataIndex: 'id' },
                { title: '状态', dataIndex: 'status' },
                { title: '描述', dataIndex: 'description' },
                { title: '解决时间', dataIndex: 'resolved_at' }
              ]}
              locale={{ emptyText: '暂无冲突。' }}
            />
          )}
        </Card>

        <Card title="指标">
          {loading ? (
            <Spin />
          ) : metrics ? (
            <Space>
              <Typography.Text>最近更新时间: {metrics.last_update || '-'}</Typography.Text>
              <Typography.Text>节点数: {metrics.total_nodes ?? '-'}</Typography.Text>
              <Typography.Text>边数: {metrics.total_edges ?? '-'}</Typography.Text>
            </Space>
          ) : (
            <Alert type="info" message="暂无指标数据。" />
          )}
        </Card>
      </Space>
    </div>
  )
}

export default KnowledgeGraphIncrementalUpdate
