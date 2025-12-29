import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useState } from 'react'
import { Card, Space, Typography, Button, Input, Alert, message } from 'antd'
import { SaveOutlined } from '@ant-design/icons'

const { TextArea } = Input

const KGReasoningConfigPage: React.FC = () => {
  const [payload, setPayload] = useState<string>(
    '{\n  "confidence_weights": {\n    "rule": 0.3,\n    "embedding": 0.25,\n    "path": 0.25,\n    "uncertainty": 0.2\n  },\n  "adaptive_thresholds": {\n    "high_confidence": 0.8,\n    "medium_confidence": 0.6,\n    "low_confidence": 0.4\n  }\n}',
  )
  const [result, setResult] = useState<string>('')
  const [loading, setLoading] = useState(false)

  const submit = async () => {
    setLoading(true)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/kg-reasoning/config'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: payload,
      })
      const text = await res.text()
      setResult(text)
      message.success('配置已更新')
    } catch (e: any) {
      message.error(e?.message || '更新失败')
      setResult('')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Typography.Title level={3}>推理策略配置</Typography.Title>

        <Card title="请求体">
          <TextArea rows={12} value={payload} onChange={(e) => setPayload(e.target.value)} />
          <Button icon={<SaveOutlined />} type="primary" style={{ marginTop: 12 }} onClick={submit} loading={loading}>
            提交到 /api/v1/kg-reasoning/config
          </Button>
        </Card>

        <Card title="响应">
          {result ? <pre style={{ whiteSpace: 'pre-wrap' }}>{result}</pre> : <Alert type="info" message="尚无结果，提交后查看。" />}
        </Card>
      </Space>
    </div>
  )
}

export default KGReasoningConfigPage
