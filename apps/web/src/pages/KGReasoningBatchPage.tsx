import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useState } from 'react'
import { Card, Space, Typography, Button, Alert, Input, message } from 'antd'
import { PlayCircleOutlined } from '@ant-design/icons'

const { TextArea } = Input

const KGReasoningBatchPage: React.FC = () => {
  const [payload, setPayload] = useState<string>(
    '{\n  "queries": [\n    { "query": "find shortest path", "entities": [], "relations": [] }\n  ]\n}'
  )
  const [result, setResult] = useState<string>('')
  const [loading, setLoading] = useState(false)

  const run = async () => {
    setLoading(true)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/kg-reasoning/batch'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: payload,
      })
      const data = await res.json()
      setResult(JSON.stringify(data, null, 2))
    } catch (e) {
      message.error('调用失败')
      setResult('')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Typography.Title level={3}>KG 批量推理</Typography.Title>

        <Card title="请求体">
          <TextArea
            rows={8}
            value={payload}
            onChange={e => setPayload(e.target.value)}
          />
          <Button
            icon={<PlayCircleOutlined />}
            type="primary"
            style={{ marginTop: 12 }}
            onClick={run}
            loading={loading}
          >
            提交到 /api/v1/kg-reasoning/batch
          </Button>
        </Card>

        <Card title="响应">
          {result ? (
            <pre style={{ whiteSpace: 'pre-wrap' }}>{result}</pre>
          ) : (
            <Alert type="info" message="尚无结果，提交请求后查看。" />
          )}
        </Card>
      </Space>
    </div>
  )
}

export default KGReasoningBatchPage
