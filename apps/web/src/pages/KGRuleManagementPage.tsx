import React, { useEffect, useState } from 'react'
import { Card, Row, Col, Typography, Button, Table, Tag, Space, Alert } from 'antd'
import { RocketOutlined, SettingOutlined, DatabaseOutlined } from '@ant-design/icons'
import { kgReasoningService } from '../services/kgReasoningService'

const { Title, Paragraph } = Typography

const KGRuleManagementPage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [rules, setRules] = useState<Array<{
    id: string
    name: string
    rule_text: string
    status: string
    confidence: number
  }>>([])

  const columns = [
    { title: 'Rule ID', dataIndex: 'id', key: 'id' },
    { title: 'Name', dataIndex: 'name', key: 'name' },
    { title: 'Expression', dataIndex: 'expression', key: 'expression' },
    { title: 'Status', dataIndex: 'status', key: 'status', render: (status: string) => <Tag color={status === 'active' ? 'green' : 'default'}>{status}</Tag> },
    { title: 'Confidence', dataIndex: 'confidence', key: 'confidence', render: (value: number) => value.toFixed(2) },
    { 
      title: 'Actions', 
      key: 'actions',
      render: () => (
        <Space>
          <Button size="small" icon={<SettingOutlined />}>Edit</Button>
          <Button size="small" icon={<DatabaseOutlined />}>Test</Button>
        </Space>
      )
    }
  ]

  useEffect(() => {
    const loadRules = async () => {
      setLoading(true)
      setError(null)
      try {
        const data = await kgReasoningService.listRules()
        setRules(Array.isArray(data) ? data : [])
      } catch (err) {
        setError((err as Error).message || '加载规则失败')
      } finally {
        setLoading(false)
      }
    }
    loadRules()
  }, [])

  const dataSource = rules.map(rule => ({
    id: rule.id,
    name: rule.name,
    expression: rule.rule_text,
    status: rule.status,
    confidence: rule.confidence
  }))

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <RocketOutlined /> KG规则管理
        </Title>
        <Paragraph>
          管理知识图推理规则的创建、编辑、验证和部署
        </Paragraph>
      </div>

      <Alert
        message="规则管理系统"
        description="此页面用于管理知识图推理引擎的规则库，包括SWRL规则的创建和管理。"
        type="info"
        showIcon
        style={{ marginBottom: '24px' }}
      />

      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Card title="推理规则列表" extra={<Button type="primary">添加规则</Button>}>
            {error && <Alert type="error" message={error} showIcon style={{ marginBottom: 12 }} />}
            <Table 
              dataSource={dataSource}
              columns={columns}
              rowKey="id"
              loading={loading}
            />
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default KGRuleManagementPage
