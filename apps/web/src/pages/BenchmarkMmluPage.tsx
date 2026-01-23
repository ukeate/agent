import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Table, Space, Typography, Button, Alert, Spin } from 'antd'
import { FileTextOutlined, ReloadOutlined } from '@ant-design/icons'

type BenchmarkRow = {
  name: string
  tasks?: string[]
  description?: string
  type?: string
  difficulty?: string
}

const BenchmarkMmluPage: React.FC = () => {
  const [rows, setRows] = useState<BenchmarkRow[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await apiFetch(
        buildApiUrl('/api/v1/model-evaluation/benchmarks')
      )
      const data = await res.json()
      setRows(Array.isArray(data) ? data : data?.benchmarks || [])
    } catch (e: any) {
      setError(e?.message || '加载失败')
      setRows([])
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
        <Space
          align="center"
          style={{ justifyContent: 'space-between', width: '100%' }}
        >
          <Space>
            <FileTextOutlined />
            <Typography.Title level={3} style={{ margin: 0 }}>
              MMLU 基准管理
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
              rowKey={r => r.name}
              dataSource={rows}
              pagination={{ pageSize: 10 }}
              columns={[
                { title: '名称', dataIndex: 'name' },
                { title: '类型', dataIndex: 'type' },
                { title: '难度', dataIndex: 'difficulty' },
                {
                  title: '任务',
                  render: (_, r) => (r.tasks || []).join(', ') || '-',
                },
                { title: '描述', dataIndex: 'description' },
              ]}
              locale={{
                emptyText: '暂无基准数据，请先在后端 BenchmarkManager 注册。',
              }}
            />
          )}
        </Card>
      </Space>
    </div>
  )
}

export default BenchmarkMmluPage
