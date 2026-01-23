import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useState } from 'react'
import { Card, Space, Typography, Button, Alert, Input, message } from 'antd'
import { SearchOutlined, ReloadOutlined } from '@ant-design/icons'

const { TextArea } = Input

const KnowledgeGraphVisualizationPage: React.FC = () => {
  const [query, setQuery] = useState<string>(
    '{\n  "query": "match (n)-[r]->(m) return n,r,m limit 20"\n}'
  )
  const [result, setResult] = useState<string>('')
  const [loading, setLoading] = useState(false)

  const runQuery = async () => {
    setLoading(true)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/knowledge-graph/query'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: query,
      })
      const data = await res.json()
      setResult(JSON.stringify(data, null, 2))
    } catch (e) {
      message.error('查询失败')
      setResult('')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Space
          align="center"
          style={{ justifyContent: 'space-between', width: '100%' }}
        >
          <Space>
            <SearchOutlined />
            <Typography.Title level={3} style={{ margin: 0 }}>
              知识图谱可视化
            </Typography.Title>
          </Space>
          <Button
            icon={<ReloadOutlined />}
            onClick={runQuery}
            loading={loading}
          >
            运行查询
          </Button>
        </Space>

        <Card title="查询请求">
          <TextArea
            rows={6}
            value={query}
            onChange={e => setQuery(e.target.value)}
          />
        </Card>

        <Card title="查询结果">
          {result ? (
            <pre style={{ whiteSpace: 'pre-wrap' }}>{result}</pre>
          ) : (
            <Alert type="info" message="提交查询后显示返回的真实数据。" />
          )}
        </Card>
      </Space>
    </div>
  )
}

export default KnowledgeGraphVisualizationPage
