import React, { useEffect, useState } from 'react'
import { Card, Table, Button, Space, Typography, Tag, Alert, Statistic, Row, Col } from 'antd'
import { ReloadOutlined, DatabaseOutlined } from '@ant-design/icons'
import { buildApiUrl, apiFetch } from '../utils/apiBase'

type QualityMetrics = {
  overall_score?: number
  completeness?: number
  accuracy?: number
  consistency?: number
  timeliness?: number
  uniqueness?: number
  validity?: number
}

type QualityIssue = {
  id: string
  type: string
  severity: string
  entity_id?: string
  entity_name?: string
  description?: string
  status?: string
}

const KnowledgeGraphQualityAssessment: React.FC = () => {
  const [metrics, setMetrics] = useState<QualityMetrics | null>(null)
  const [issues, setIssues] = useState<QualityIssue[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const [mRes, iRes] = await Promise.all([
        apiFetch(buildApiUrl('/knowledge-graph/quality/metrics')),
        apiFetch(buildApiUrl('/knowledge-graph/quality/issues'))
      ])
      setMetrics(await mRes.json())
      const issueData = await iRes.json()
      setIssues(Array.isArray(issueData?.issues) ? issueData.issues : [])
    } catch (e: any) {
      setError(e?.message || '加载失败')
      setMetrics(null)
      setIssues([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  const columns = [
    { title: 'ID', dataIndex: 'id' },
    { title: '类型', dataIndex: 'type' },
    { title: '严重性', dataIndex: 'severity', render: (s: string) => <Tag color="red">{s}</Tag> },
    { title: '实体', dataIndex: 'entity_name' },
    { title: '状态', dataIndex: 'status' }
  ]

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Space align="center" style={{ justifyContent: 'space-between', width: '100%' }}>
          <Space>
            <DatabaseOutlined />
            <Typography.Title level={3} style={{ margin: 0 }}>
              知识图谱质量评估
            </Typography.Title>
          </Space>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        </Space>

        {error && <Alert type="error" message={error} />}

        <Row gutter={16}>
          <Col span={6}><Card><Statistic title="总体质量" value={metrics?.overall_score ?? 0} /></Card></Col>
          <Col span={6}><Card><Statistic title="完整性" value={metrics?.completeness ?? 0} /></Card></Col>
          <Col span={6}><Card><Statistic title="准确性" value={metrics?.accuracy ?? 0} /></Card></Col>
          <Col span={6}><Card><Statistic title="一致性" value={metrics?.consistency ?? 0} /></Card></Col>
        </Row>

        <Card title="质量问题">
          <Table rowKey="id" dataSource={issues} columns={columns} loading={loading} />
        </Card>
      </Space>
    </div>
  )
}

export default KnowledgeGraphQualityAssessment
