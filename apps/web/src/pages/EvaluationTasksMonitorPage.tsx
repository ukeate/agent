import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Table, Button, Space, Typography, Drawer, Tag, message, Select } from 'antd'
import { ReloadOutlined } from '@ant-design/icons'

type Job = {
  job_id: string
  status: string
  created_at: string
  started_at?: string | null
  completed_at?: string | null
  progress?: number
  current_task?: string | null
  results?: any[]
  error?: string | null
  models_count?: number
  benchmarks_count?: number
}

const EvaluationTasksMonitorPage: React.FC = () => {
  const [jobs, setJobs] = useState<Job[]>([])
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState<string>('all')
  const [selected, setSelected] = useState<Job | null>(null)
  const [open, setOpen] = useState(false)

  const load = async () => {
    setLoading(true)
    try {
      const params = new URLSearchParams({ limit: '50', offset: '0' })
      if (status !== 'all') params.set('status', status)
      const res = await apiFetch(buildApiUrl(`/api/v1/model-evaluation/jobs?${params.toString()}`))
      const data = await res.json().catch(() => null)
      setJobs(Array.isArray(data?.jobs) ? data.jobs : [])
    } catch (e: any) {
      message.error(e?.message || '加载失败')
      setJobs([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
    const timer = setInterval(load, 5000)
    return () => clearInterval(timer)
  }, [status])

  const cancelJob = async (jobId: string) => {
    setLoading(true)
    try {
      const res = await apiFetch(buildApiUrl(`/api/v1/model-evaluation/jobs/${jobId}`), { method: 'DELETE' })
      const data = await res.json().catch(() => null)
      message.success('任务已取消')
      await load()
    } catch (e: any) {
      message.error(e?.message || '取消失败')
    } finally {
      setLoading(false)
    }
  }

  const statusTag = (v: string) => {
    const map: Record<string, { color: any; text: string }> = {
      pending: { color: 'default', text: 'pending' },
      running: { color: 'processing', text: 'running' },
      completed: { color: 'success', text: 'completed' },
      failed: { color: 'error', text: 'failed' },
      cancelled: { color: 'default', text: 'cancelled' },
    }
    const c = map[v] || { color: 'default', text: v }
    return <Tag color={c.color}>{c.text}</Tag>
  }

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Space align="center" style={{ justifyContent: 'space-between', width: '100%' }}>
          <Typography.Title level={3} style={{ margin: 0 }}>
            评估任务监控
          </Typography.Title>
          <Space>
            <Select
              value={status}
              onChange={setStatus}
              options={[
                { value: 'all', label: '全部' },
                { value: 'pending', label: 'pending' },
                { value: 'running', label: 'running' },
                { value: 'completed', label: 'completed' },
                { value: 'failed', label: 'failed' },
                { value: 'cancelled', label: 'cancelled' },
              ]}
              style={{ width: 160 }}
            />
            <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
              刷新
            </Button>
          </Space>
        </Space>

        <Card>
          <Table
            rowKey="job_id"
            dataSource={jobs}
            loading={loading}
            pagination={{ pageSize: 20 }}
            columns={[
              { title: 'Job ID', dataIndex: 'job_id' },
              { title: '状态', dataIndex: 'status', render: (v: string) => statusTag(v) },
              {
                title: '进度',
                dataIndex: 'progress',
                render: (v: number) => `${Math.round((v || 0) * 100)}%`,
              },
              { title: '当前任务', dataIndex: 'current_task', render: (v: string) => v || '-' },
              { title: '创建时间', dataIndex: 'created_at' },
              { title: '开始时间', dataIndex: 'started_at', render: (v: string) => v || '-' },
              { title: '完成时间', dataIndex: 'completed_at', render: (v: string) => v || '-' },
              { title: '错误', dataIndex: 'error', render: (v: string) => v || '-' },
              {
                title: '操作',
                render: (_, r: Job) => (
                  <Space size="small">
                    <Button
                      size="small"
                      onClick={() => {
                        setSelected(r)
                        setOpen(true)
                      }}
                    >
                      详情
                    </Button>
                    <Button
                      size="small"
                      danger
                      disabled={!['pending', 'running'].includes(r.status)}
                      onClick={() => cancelJob(r.job_id)}
                    >
                      取消
                    </Button>
                  </Space>
                ),
              },
            ]}
          />
        </Card>

        <Drawer
          title="任务详情"
          open={open}
          onClose={() => setOpen(false)}
          width={720}
          destroyOnClose
        >
          <pre style={{ whiteSpace: 'pre-wrap' }}>{selected ? JSON.stringify(selected, null, 2) : ''}</pre>
        </Drawer>
      </Space>
    </div>
  )
}

export default EvaluationTasksMonitorPage
