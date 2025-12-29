import React, { useEffect, useState } from 'react'
import { Alert, Card, Table, Typography } from 'antd'
import { CheckCircleOutlined } from '@ant-design/icons'
import { knowledgeManagementService } from '../services/knowledgeManagementService'

const { Title, Paragraph } = Typography

const KnowledgeValidationPage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<{
    valid: boolean
    violations: Array<{
      rule_id: string
      rule_type: string
      message: string
      count: number
      details?: any[]
    }>
    checked_rules: number
    execution_time_ms: number
  } | null>(null)

  const columns = [
    { title: '规则ID', dataIndex: 'rule_id', key: 'rule_id' },
    { title: '规则类型', dataIndex: 'rule_type', key: 'rule_type' },
    { title: '问题描述', dataIndex: 'message', key: 'message' },
    { title: '影响数量', dataIndex: 'count', key: 'count' },
    { title: '详情数', dataIndex: 'details_count', key: 'details_count' }
  ]

  useEffect(() => {
    const loadValidation = async () => {
      setLoading(true)
      setError(null)
      try {
        const data = await knowledgeManagementService.validateGraph()
        setResult(data)
      } catch (err) {
        setError((err as Error).message || '加载验证数据失败')
      } finally {
        setLoading(false)
      }
    }
    loadValidation()
  }, [])

  const dataSource = (result?.violations || []).map((item, index) => ({
    key: `${item.rule_id}-${index}`,
    rule_id: item.rule_id,
    rule_type: item.rule_type,
    message: item.message,
    count: item.count,
    details_count: item.details ? item.details.length : 0
  }))

  return (
    <div style={{ padding: 24 }}>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>
          <CheckCircleOutlined style={{ marginRight: 8 }} />
          知识验证
        </Title>
        <Paragraph type="secondary">
          验证和评估知识图谱中的实体和关系的准确性
        </Paragraph>
      </div>

      <Card title="知识验证列表">
        {error && <Alert type="error" message={error} showIcon style={{ marginBottom: 12 }} />}
        {!error && result && (
          <Alert
            type={result.valid ? 'success' : 'warning'}
            showIcon
            message={result.valid ? '当前验证通过' : '存在验证问题'}
            description={`检查规则数: ${result.checked_rules}，耗时: ${result.execution_time_ms.toFixed(2)}ms`}
            style={{ marginBottom: 12 }}
          />
        )}
        <Table columns={columns} dataSource={dataSource} loading={loading} />
      </Card>
    </div>
  )
}

export default KnowledgeValidationPage
