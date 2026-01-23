import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Table, Alert, Button, Space } from 'antd'

type Risk = {
  assessment_id: string
  user_id?: string
  risk_level?: string
  risk_score?: number
  timestamp?: string
}

const EmotionalRiskAssessmentDashboardPage: React.FC = () => {
  const [items, setItems] = useState<Risk[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await apiFetch(
        buildApiUrl('/api/v1/emotional-intelligence/risk-assessment')
      )
      const data = await res.json()
      setItems(data || [])
    } catch (e: any) {
      setError(e?.message || '加载失败')
      setItems([])
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
        <Button onClick={load} loading={loading}>
          刷新
        </Button>
        {error && <Alert type="error" message={error} />}
        <Card title="情感风险评估">
          <Table
            rowKey="assessment_id"
            loading={loading}
            dataSource={items}
            locale={{ emptyText: '暂无风险记录。' }}
            columns={[
              { title: 'ID', dataIndex: 'assessment_id' },
              { title: '用户', dataIndex: 'user_id' },
              { title: '风险级别', dataIndex: 'risk_level' },
              { title: '得分', dataIndex: 'risk_score' },
              { title: '时间', dataIndex: 'timestamp' },
            ]}
          />
        </Card>
      </Space>
    </div>
  )
}

export default EmotionalRiskAssessmentDashboardPage
