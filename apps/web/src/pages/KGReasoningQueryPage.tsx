import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import {
  Card,
  Table,
  Button,
  Space,
  Typography,
  Form,
  Input,
  message,
} from 'antd'
import { ReloadOutlined, SearchOutlined } from '@ant-design/icons'

type QueryResult = {
  query: string
  data: any
  execution_time_ms?: number
}

const KGReasoningQueryPage: React.FC = () => {
  const [history, setHistory] = useState<QueryResult[]>([])
  const [loading, setLoading] = useState(false)
  const [form] = Form.useForm()

  const loadHistory = async () => {
    setLoading(true)
    try {
      const res = await apiFetch(
        buildApiUrl('/api/v1/kg-reasoning/strategies/performance')
      )
      const data = await res.json()
      setHistory([{ query: 'summary', data, execution_time_ms: 0 }])
    } catch (e) {
      setHistory([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadHistory()
  }, [])

  const onFinish = async (values: any) => {
    setLoading(true)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/kg-reasoning/query'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: values.query }),
      })
      const data = await res.json()
      setHistory(
        [
          {
            query: values.query,
            data,
            execution_time_ms: data?.execution_time_ms,
          },
          ...history,
        ].slice(0, 20)
      )
    } catch (e: any) {
      message.error(e?.message || '查询失败')
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
          <Typography.Title level={3} style={{ margin: 0 }}>
            知识图推理查询
          </Typography.Title>
          <Button
            icon={<ReloadOutlined />}
            onClick={loadHistory}
            loading={loading}
          >
            刷新
          </Button>
        </Space>

        <Card>
          <Form layout="inline" form={form} onFinish={onFinish}>
            <Form.Item
              name="query"
              rules={[{ required: true, message: '请输入查询' }]}
              style={{ flex: 1 }}
            >
              <Input placeholder="MATCH (n) RETURN n LIMIT 5" />
            </Form.Item>
            <Form.Item>
              <Button
                type="primary"
                htmlType="submit"
                icon={<SearchOutlined />}
                loading={loading}
              >
                执行
              </Button>
            </Form.Item>
          </Form>
        </Card>

        <Card title="历史">
          <Table
            rowKey={r => r.query + (r.execution_time_ms || 0)}
            dataSource={history}
            loading={loading}
            columns={[
              { title: '查询', dataIndex: 'query' },
              { title: '耗时(ms)', dataIndex: 'execution_time_ms' },
              {
                title: '结果',
                dataIndex: 'data',
                render: d => (
                  <pre style={{ whiteSpace: 'pre-wrap' }}>
                    {JSON.stringify(d, null, 2)}
                  </pre>
                ),
              },
            ]}
          />
        </Card>
      </Space>
    </div>
  )
}

export default KGReasoningQueryPage
