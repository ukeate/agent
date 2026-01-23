import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useState } from 'react'
import { Card, Space, Typography, Button, Input, Alert, message } from 'antd'
import { PlayCircleOutlined } from '@ant-design/icons'

const KGReasoningAnalysisPage: React.FC = () => {
  const [query, setQuery] = useState('test query')
  const [result, setResult] = useState<string>('')
  const [explanation, setExplanation] = useState<string>('')
  const [loading, setLoading] = useState(false)

  const run = async () => {
    setLoading(true)
    try {
      const queryRes = await apiFetch(
        buildApiUrl('/api/v1/kg-reasoning/query'),
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query }),
        }
      )
      const queryData = await queryRes.json().catch(() => null)
      setResult(JSON.stringify(queryData, null, 2))

      const explainRes = await apiFetch(
        buildApiUrl('/api/v1/kg-reasoning/explain'),
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(queryData),
        }
      )
      const explainData = await explainRes.json().catch(() => null)
      setExplanation(explainData?.explanation || '')
    } catch (e: any) {
      message.error(e?.message || '执行失败')
      setResult('')
      setExplanation('')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Typography.Title level={3}>推理解释</Typography.Title>

        <Card title="查询">
          <Input
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="输入推理查询"
          />
          <Button
            icon={<PlayCircleOutlined />}
            type="primary"
            style={{ marginTop: 12 }}
            onClick={run}
            loading={loading}
          >
            执行查询并生成解释
          </Button>
        </Card>

        <Card title="推理结果">
          {result ? (
            <pre style={{ whiteSpace: 'pre-wrap' }}>{result}</pre>
          ) : (
            <Alert type="info" message="尚无结果，执行后查看。" />
          )}
        </Card>

        <Card title="解释">
          {explanation ? (
            <pre style={{ whiteSpace: 'pre-wrap' }}>{explanation}</pre>
          ) : (
            <Alert type="info" message="尚无解释，执行后查看。" />
          )}
        </Card>
      </Space>
    </div>
  )
}

export default KGReasoningAnalysisPage
