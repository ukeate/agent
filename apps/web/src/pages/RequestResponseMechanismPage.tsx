import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Row, Col, Table, Button, Typography, Tag, Space, message } from 'antd'
import { ReloadOutlined, ApiOutlined } from '@ant-design/icons'

interface RequestRecord {
  id: string
  correlation_id: string
  requester: string
  responder?: string
  subject: string
  method: string
  status: string
  request_time: string
  response_time?: string
  latency_ms?: number
  retry_count: number
}

const RequestResponseMechanismPage: React.FC = () => {
  const [records, setRecords] = useState<RequestRecord[]>([])
  const [loading, setLoading] = useState(false)

  const loadData = async () => {
    setLoading(true)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/events/request-response'))
      const data = await res.json()
      setRecords(data.records || [])
    } catch (e: any) {
      message.error(e?.message || '加载请求响应数据失败')
      setRecords([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
    const timer = setInterval(loadData, 5000)
    return () => clearInterval(timer)
  }, [])

  const columns = [
    { title: 'ID', dataIndex: 'id', key: 'id' },
    { title: '关联ID', dataIndex: 'correlation_id', key: 'correlation_id' },
    { title: '请求方', dataIndex: 'requester', key: 'requester' },
    { title: '响应方', dataIndex: 'responder', key: 'responder', render: (v: string) => v || '—' },
    { title: '主题', dataIndex: 'subject', key: 'subject' },
    { title: '方法', dataIndex: 'method', key: 'method' },
    { title: '状态', dataIndex: 'status', key: 'status', render: (s: string) => <Tag color={s === 'completed' ? 'green' : s === 'pending' ? 'blue' : 'red'}>{s}</Tag> },
    { title: '请求时间', dataIndex: 'request_time', key: 'request_time', render: (t: string) => new Date(t).toLocaleString() },
    { title: '响应时间', dataIndex: 'response_time', key: 'response_time', render: (t?: string) => t ? new Date(t).toLocaleString() : '—' },
    { title: '延迟(ms)', dataIndex: 'latency_ms', key: 'latency_ms', render: (v?: number) => v ?? '—' },
    { title: '重试', dataIndex: 'retry_count', key: 'retry_count' },
  ]

  return (
    <div style={{ padding: 24 }}>
      <Row justify="space-between" align="middle" style={{ marginBottom: 16 }}>
        <Col>
          <Space>
            <ApiOutlined />
            <Typography.Title level={3} style={{ margin: 0 }}>请求-响应机制</Typography.Title>
          </Space>
          <Typography.Text type="secondary">数据来自 /api/v1/events/request-response，无本地静态数据</Typography.Text>
        </Col>
        <Col>
          <Button icon={<ReloadOutlined />} onClick={loadData} loading={loading}>刷新</Button>
        </Col>
      </Row>

      <Card>
        <Table
          rowKey="id"
          loading={loading}
          dataSource={records}
          columns={columns}
          pagination={{ pageSize: 20 }}
        />
      </Card>
    </div>
  )
}

export default RequestResponseMechanismPage
