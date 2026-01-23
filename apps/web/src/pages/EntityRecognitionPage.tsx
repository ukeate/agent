import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Table, Button, Space, Typography, Tag, Form, Input, Select, Upload, message } from 'antd'
import { ReloadOutlined, SearchOutlined, UploadOutlined, PlusOutlined } from '@ant-design/icons'
import type { UploadProps } from 'antd'

type Entity = {
  id?: string
  text?: string
  label?: string
  confidence?: number
  context?: string
  source?: string
  linked_entity?: string
}

const { TextArea } = Input
const { Option } = Select

const EntityRecognitionPage: React.FC = () => {
  const [entities, setEntities] = useState<Entity[]>([])
  const [loading, setLoading] = useState(false)
  const [form] = Form.useForm()
  const [text, setText] = useState('')
  const [model, setModel] = useState('default')

  const load = async () => {
    setLoading(true)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/knowledge-extraction/entities/search'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: '', limit: 50 })
      })
      const data = await res.json()
      setEntities(Array.isArray(data?.entities) ? data.entities : [])
    } catch (e: any) {
      message.error(e?.message || '加载实体失败')
      setEntities([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  const runExtract = async () => {
    if (!text.trim()) {
      message.warning('请输入文本')
      return
    }
    setLoading(true)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/knowledge-extraction/ner'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, model_name: model, return_context: true, link_entities: true })
      })
      const data = await res.json()
      const list = Array.isArray(data?.entities) ? data.entities : []
      setEntities(list)
      message.success('实体抽取完成')
    } catch (e: any) {
      message.error(e?.message || '抽取失败')
    } finally {
      setLoading(false)
    }
  }

  const uploadProps: UploadProps = {
    accept: '.txt,.json',
    beforeUpload: () => false,
    maxCount: 1
  }

  const columns = [
    { title: '文本', dataIndex: 'text', key: 'text' },
    { title: '标签', dataIndex: 'label', key: 'label', render: (v: string) => <Tag>{v}</Tag> },
    { title: '置信度', dataIndex: 'confidence', key: 'confidence' },
    { title: '来源', dataIndex: 'source', key: 'source' },
    { title: '上下文', dataIndex: 'context', key: 'context' },
    { title: '链接实体', dataIndex: 'linked_entity', key: 'linked_entity' }
  ]

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Space align="center" style={{ justifyContent: 'space-between', width: '100%' }}>
          <Typography.Title level={3} style={{ margin: 0 }}>
            实体识别
          </Typography.Title>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        </Space>

        <Card title="文本抽取">
          <Form layout="vertical" form={form} onFinish={runExtract}>
            <Form.Item label="模型">
              <Select value={model} onChange={setModel} style={{ width: 240 }}>
                <Option value="default">default</Option>
                <Option value="bert-large">bert-large</Option>
                <Option value="spacy-zh">spacy-zh</Option>
              </Select>
            </Form.Item>
            <Form.Item label="文本">
              <TextArea rows={4} value={text} onChange={(e) => setText(e.target.value)} />
            </Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" icon={<SearchOutlined />} loading={loading}>
                抽取实体
              </Button>
              <Upload {...uploadProps}>
                <Button icon={<UploadOutlined />}>上传文本文件</Button>
              </Upload>
            </Space>
          </Form>
        </Card>

        <Card title="实体列表">
          <Table rowKey={(r) => r.id || `${r.text}-${r.start}`} dataSource={entities} columns={columns} loading={loading} />
        </Card>
      </Space>
    </div>
  )
}

export default EntityRecognitionPage
