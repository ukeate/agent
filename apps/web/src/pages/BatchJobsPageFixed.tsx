import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Table, Button, Space, Typography, Alert, Spin, Tag } from 'antd'
import {
  ThunderboltOutlined,
  ReloadOutlined,
  PlayCircleOutlined,
  StopOutlined,
} from '@ant-design/icons'

interface BatchJob {
  job_id: string
  name?: string
  status?: string
  job_type?: string
  created_at?: string
  progress?: number
}

const BatchJobsPageFixed: React.FC = () => {
  const [jobs, setJobs] = useState<BatchJob[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/batch/jobs'))
      const data = await res.json()
      setJobs(Array.isArray(data?.jobs) ? data.jobs : [])
    } catch (e: any) {
      setError(e?.message || '加载失败')
      setJobs([])
    } finally {
      setLoading(false)
    }
  }

  const startJob = async (jobId: string) => {
    await apiFetch(buildApiUrl(`/api/v1/batch/jobs/${jobId}/start`), {
      method: 'POST',
    })
    load()
  }

  const stopJob = async (jobId: string) => {
    await apiFetch(buildApiUrl(`/api/v1/batch/jobs/${jobId}/stop`), {
      method: 'POST',
    })
    load()
  }

  useEffect(() => {
    load()
  }, [])

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Space
          align="center"
          style={{ justifyContent: 'space-between', width: '100%' }}
        >
          <Space>
            <ThunderboltOutlined />
            <Typography.Title level={3} style={{ margin: 0 }}>
              批处理作业
            </Typography.Title>
          </Space>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        </Space>

        {error && <Alert type="error" message="加载失败" description={error} />}

        <Card>
          {loading ? (
            <Spin />
          ) : (
            <Table
              rowKey="job_id"
              dataSource={jobs}
              columns={[
                { title: 'ID', dataIndex: 'job_id' },
                { title: '名称', dataIndex: 'name' },
                { title: '类型', dataIndex: 'job_type' },
                {
                  title: '状态',
                  dataIndex: 'status',
                  render: v => <Tag>{v || '未知'}</Tag>,
                },
                { title: '进度', dataIndex: 'progress' },
                { title: '创建时间', dataIndex: 'created_at' },
                {
                  title: '操作',
                  render: (_, r) => (
                    <Space>
                      <Button
                        icon={<PlayCircleOutlined />}
                        size="small"
                        onClick={() => startJob(r.job_id)}
                      >
                        启动
                      </Button>
                      <Button
                        icon={<StopOutlined />}
                        size="small"
                        danger
                        onClick={() => stopJob(r.job_id)}
                      >
                        停止
                      </Button>
                    </Space>
                  ),
                },
              ]}
              locale={{
                emptyText: '暂无作业，先调用 /api/v1/batch/jobs/create 创建。',
              }}
            />
          )}
        </Card>
      </Space>
    </div>
  )
}

export default BatchJobsPageFixed
