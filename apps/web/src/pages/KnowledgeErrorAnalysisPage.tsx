import React, { useEffect, useMemo, useState } from 'react'
import { Alert, Card, Table, Tag, Typography, Timeline } from 'antd'
import { ExclamationCircleOutlined } from '@ant-design/icons'
import { knowledgeGraphService } from '../services/knowledgeGraphService'

const { Title, Paragraph } = Typography

const KnowledgeErrorAnalysisPage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [issues, setIssues] = useState<Array<{
    issue_id: string
    issue_type: string
    description: string
    severity: 'low' | 'medium' | 'high' | 'critical'
    affected_entities: string[]
    recommended_actions: string[]
    confidence: number
    detected_at: string
  }>>([])

  const columns = [
    { title: '错误类型', dataIndex: 'type', key: 'type' },
    { title: '描述', dataIndex: 'description', key: 'description' },
    { 
      title: '严重性', 
      dataIndex: 'severity', 
      key: 'severity',
      render: (severity: string) => {
        const labelMap: Record<string, string> = {
          low: '低',
          medium: '中等',
          high: '高',
          critical: '严重'
        }
        const label = labelMap[severity] || severity
        const color = severity === 'high' || severity === 'critical' ? 'red' : severity === 'medium' ? 'orange' : 'green'
        return (
          <Tag color={color}>
            {label}
          </Tag>
        )
      }
    },
    { title: '影响实体', dataIndex: 'count', key: 'count' },
    { title: '置信度', dataIndex: 'confidence', key: 'confidence', render: (value: number) => value.toFixed(2) },
    { title: '检测时间', dataIndex: 'detected_at', key: 'detected_at', render: (value: string) => new Date(value).toLocaleString() }
  ]

  const dataSource = useMemo(() => (
    issues.map(issue => ({
      key: issue.issue_id,
      type: issue.issue_type,
      description: issue.description,
      severity: issue.severity,
      count: issue.affected_entities?.length || 0,
      confidence: issue.confidence,
      detected_at: issue.detected_at
    }))
  ), [issues])

  const timelineItems = useMemo(() => (
    [...issues]
      .sort((a, b) => new Date(b.detected_at).getTime() - new Date(a.detected_at).getTime())
      .map(issue => ({
        key: issue.issue_id,
        color: issue.severity === 'high' || issue.severity === 'critical' ? 'red' : issue.severity === 'medium' ? 'orange' : 'green',
        text: `${new Date(issue.detected_at).toLocaleString()} - ${issue.issue_type}：${issue.description}`
      }))
  ), [issues])

  useEffect(() => {
    const loadIssues = async () => {
      setLoading(true)
      setError(null)
      try {
        const data = await knowledgeGraphService.getQualityIssues()
        setIssues(Array.isArray(data) ? data : [])
      } catch (err) {
        setError((err as Error).message || '加载错误分析数据失败')
      } finally {
        setLoading(false)
      }
    }
    loadIssues()
  }, [])

  return (
    <div style={{ padding: 24 }}>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>
          <ExclamationCircleOutlined style={{ marginRight: 8 }} />
          错误分析
        </Title>
        <Paragraph type="secondary">
          分析和识别知识抽取过程中的错误和问题
        </Paragraph>
      </div>

      <Card title="错误统计" style={{ marginBottom: 16 }}>
        {error && <Alert type="error" message={error} showIcon style={{ marginBottom: 12 }} />}
        <Table columns={columns} dataSource={dataSource} loading={loading} />
      </Card>

      <Card title="错误日志">
        <Timeline>
          {timelineItems.length === 0 && (
            <Timeline.Item color="gray">暂无错误记录</Timeline.Item>
          )}
          {timelineItems.map(item => (
            <Timeline.Item key={item.key} color={item.color}>
              {item.text}
            </Timeline.Item>
          ))}
        </Timeline>
      </Card>
    </div>
  )
}

export default KnowledgeErrorAnalysisPage
