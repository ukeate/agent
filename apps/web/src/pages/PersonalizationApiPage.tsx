import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useState } from 'react'
import { Card, Space, Typography, Button, Alert, Input, message } from 'antd'
import { SendOutlined } from '@ant-design/icons'

const { TextArea } = Input

const PersonalizationApiPage: React.FC = () => {
  const [requestBody, setRequestBody] = useState<string>('{\n  "user_id": "demo-user",\n  "context": {},\n  "candidate_items": []\n}')
  const [responseBody, setResponseBody] = useState<string>('')
  const [loading, setLoading] = useState(false)

  const send = async () => {
    setLoading(true)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/personalization/recommend'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: requestBody
      })
      const data = await res.json()
      setResponseBody(JSON.stringify(data, null, 2))
    } catch (e) {
      message.error('请求失败')
      setResponseBody('')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Typography.Title level={3}>个性化 API</Typography.Title>

        <Card title="请求体">
          <TextArea rows={8} value={requestBody} onChange={(e) => setRequestBody(e.target.value)} />
          <Button type="primary" icon={<SendOutlined />} style={{ marginTop: 12 }} onClick={send} loading={loading}>
            调用 /api/v1/personalization/recommend
          </Button>
        </Card>

        <Card title="响应">
          {responseBody ? (
            <pre style={{ whiteSpace: 'pre-wrap' }}>{responseBody}</pre>
          ) : (
            <Alert type="info" message="提交后显示真实响应。" />
          )}
        </Card>
      </Space>
    </div>
  )
}

export default PersonalizationApiPage
