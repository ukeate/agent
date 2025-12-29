import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Table, Button, Space, Typography, Tag, Alert, Form, Input, Select, message } from 'antd'
import { ReloadOutlined, TeamOutlined } from '@ant-design/icons'

type AdaptationStrategy = {
  id?: string
  context_type?: string
  strategy?: string
  effectiveness?: number
}

type AdaptationResult = {
  id?: string
  context?: string
  result?: string
  timestamp?: string
}

const { Option } = Select

const SocialContextAdaptationPage: React.FC = () => {
  const [strategies, setStrategies] = useState<AdaptationStrategy[]>([])
  const [history, setHistory] = useState<AdaptationResult[]>([])
  const [loading, setLoading] = useState(false)
  const [form] = Form.useForm()

  const load = async () => {
    setLoading(true)
    try {
      const [sRes, hRes] = await Promise.all([
        apiFetch(buildApiUrl('/api/v1/social-emotional/context-adaptation/strategies'),
        apiFetch(buildApiUrl('/api/v1/social-emotional/context-adaptation/history'))
      ])
      setStrategies((await sRes.json())?.strategies || [])
      setHistory((await hRes.json())?.history || [])
    } catch (e: any) {
      message.error(e?.message || '加载失败')
      setStrategies([])
      setHistory([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  const createStrategy = async (values: any) => {
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/social-emotional/context-adaptation/strategies'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(values)
      })
      message.success('策略已创建')
      form.resetFields()
      load()
    } catch (e: any) {
      message.error(e?.message || '创建失败')
    }
  }

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Space align="center" style={{ justifyContent: 'space-between', width: '100%' }}>
          <Space>
            <TeamOutlined />
            <Typography.Title level={3} style={{ margin: 0 }}>
              社交情境适配
            </Typography.Title>
          </Space>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        </Space>

        <Card title="新增策略">
          <Form layout="inline" form={form} onFinish={createStrategy}>
            <Form.Item name="context_type" rules={[{ required: true }]} label="情境">
              <Select style={{ width: 200 }}>
                <Option value="business">商务</Option>
                <Option value="casual">日常</Option>
                <Option value="conflict">冲突</Option>
              </Select>
            </Form.Item>
            <Form.Item name="strategy" rules={[{ required: true }]} label="策略">
              <Input style={{ width: 260 }} />
            </Form.Item>
            <Form.Item>
              <Button type="primary" htmlType="submit" loading={loading}>
                保存
              </Button>
            </Form.Item>
          </Form>
        </Card>

        <Card title="策略列表">
          <Table
            rowKey={(r) => r.id || r.strategy}
            dataSource={strategies}
            loading={loading}
            columns={[
              { title: '情境', dataIndex: 'context_type' },
              { title: '策略', dataIndex: 'strategy' },
              { title: '有效性', dataIndex: 'effectiveness' }
            ]}
            locale={{ emptyText: '暂无策略' }}
          />
        </Card>

        <Card title="适配历史">
          <Table
            rowKey={(r) => r.id || r.timestamp}
            dataSource={history}
            loading={loading}
            columns={[
              { title: '情境', dataIndex: 'context' },
              { title: '结果', dataIndex: 'result' },
              { title: '时间', dataIndex: 'timestamp' }
            ]}
            locale={{ emptyText: '暂无记录' }}
          />
        </Card>
      </Space>
    </div>
  )
}

export default SocialContextAdaptationPage
