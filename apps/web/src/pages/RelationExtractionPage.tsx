import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Table, Button, Space, Typography, Tag, Form, Input, Select, message } from 'antd'
import { ReloadOutlined, LinkOutlined, SearchOutlined } from '@ant-design/icons'

type Relation = {
  source: string
  target: string
  relation_type: string
  confidence?: number
  evidence?: string[]
}

const { TextArea } = Input
const { Option } = Select

const RelationExtractionPage: React.FC = () => {
  const [relations, setRelations] = useState<Relation[]>([])
  const [loading, setLoading] = useState(false)
  const [text, setText] = useState('')
  const [model, setModel] = useState('default')

  const load = async () => {
    setLoading(true)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/knowledge-extraction/relation-models'))
      await res.json().catch(() => null)
      // 这里仅更新模型列表的验证，用不上的数据忽略
    } catch (e) {
      // ignore
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  const extract = async () => {
    if (!text.trim()) {
      message.warning('请输入文本')
      return
    }
    setLoading(true)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/knowledge-extraction/relations'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, model_name: model, link_entities: true })
      })
      const data = await res.json()
      setRelations(Array.isArray(data?.relations) ? data.relations : [])
      message.success('关系抽取完成')
    } catch (e: any) {
      message.error(e?.message || '关系抽取失败')
      setRelations([])
    } finally {
      setLoading(false)
    }
  }

  const columns = [
    { title: '源', dataIndex: 'source', key: 'source' },
    { title: '目标', dataIndex: 'target', key: 'target' },
    { title: '关系', dataIndex: 'relation_type', key: 'relation_type', render: (v: string) => <Tag>{v}</Tag> },
    { title: '置信度', dataIndex: 'confidence', key: 'confidence' }
  ]

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Space align="center" style={{ justifyContent: 'space-between', width: '100%' }}>
          <Space>
            <LinkOutlined />
            <Typography.Title level={3} style={{ margin: 0 }}>
              关系抽取
            </Typography.Title>
          </Space>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新模型
          </Button>
        </Space>

        <Card title="文本关系抽取">
          <Form layout="vertical" onFinish={extract}>
            <Form.Item label="模型">
              <Select value={model} onChange={setModel} style={{ width: 240 }}>
                <Option value="default">default</Option>
                <Option value="bert-re">bert-re</Option>
              </Select>
            </Form.Item>
            <Form.Item label="文本">
              <TextArea rows={4} value={text} onChange={(e) => setText(e.target.value)} />
            </Form.Item>
            <Button type="primary" htmlType="submit" icon={<SearchOutlined />} loading={loading}>
              抽取关系
            </Button>
          </Form>
        </Card>

        <Card title="关系列表">
          <Table rowKey={(r) => `${r.source}-${r.target}-${r.relation_type}`} dataSource={relations} columns={columns} loading={loading} />
        </Card>
      </Space>
    </div>
  )
}

export default RelationExtractionPage
