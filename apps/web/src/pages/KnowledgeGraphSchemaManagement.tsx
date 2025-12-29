import React, { useEffect, useState } from 'react'
import { Card, Table, Button, Space, Typography, Tag, Alert, Form, Input, Select, message } from 'antd'
import { ReloadOutlined, SchemaOutlined, PlusOutlined } from '@ant-design/icons'
import { buildApiUrl, apiFetch } from '../utils/apiBase'

type SchemaNode = {
  label: string
  properties: any[]
  constraints: string[]
  indexes: string[]
}

type SchemaRelationship = {
  type: string
  from_node: string
  to_node: string
  properties: any[]
}

type SchemaSummary = {
  nodes?: number
  relationships?: number
  properties?: number
}

const { TextArea } = Input
const { Option } = Select

const KnowledgeGraphSchemaManagement: React.FC = () => {
  const [nodes, setNodes] = useState<SchemaNode[]>([])
  const [rels, setRels] = useState<SchemaRelationship[]>([])
  const [summary, setSummary] = useState<SchemaSummary | null>(null)
  const [loading, setLoading] = useState(false)
  const [form] = Form.useForm()

  const load = async () => {
    setLoading(true)
    try {
      const [nRes, rRes, sRes] = await Promise.all([
        apiFetch(buildApiUrl('/knowledge-graph/schema/nodes')),
        apiFetch(buildApiUrl('/knowledge-graph/schema/relationships')),
        apiFetch(buildApiUrl('/knowledge-graph/schema/statistics'))
      ])
      const nodesData = await nRes.json()
      const relsData = await rRes.json()
      const summaryData = await sRes.json()
      setNodes(nodesData?.nodes || [])
      setRels(relsData?.relationships || [])
      setSummary(summaryData || null)
    } catch (e: any) {
      message.error(e?.message || '加载失败')
      setNodes([])
      setRels([])
      setSummary(null)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  const createNode = async (values: any) => {
    try {
      await apiFetch(buildApiUrl('/knowledge-graph/schema/nodes'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(values)
      })
      message.success('节点类型已创建')
      form.resetFields()
      load()
    } catch (e: any) {
      message.error(e?.message || '创建失败')
    }
  }

  const nodeColumns = [
    { title: '标签', dataIndex: 'label', key: 'label' },
    { title: '属性数', dataIndex: 'properties', key: 'properties', render: (p: any[]) => p?.length || 0 },
    { title: '约束', dataIndex: 'constraints', key: 'constraints', render: (c: string[]) => (c || []).length },
    { title: '索引', dataIndex: 'indexes', key: 'indexes', render: (i: string[]) => (i || []).length }
  ]

  const relColumns = [
    { title: '类型', dataIndex: 'type', key: 'type', render: (v: string) => <Tag>{v}</Tag> },
    { title: '起点', dataIndex: 'from_node', key: 'from_node' },
    { title: '终点', dataIndex: 'to_node', key: 'to_node' },
    { title: '属性数', dataIndex: 'properties', key: 'properties', render: (p: any[]) => p?.length || 0 }
  ]

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Space align="center" style={{ justifyContent: 'space-between', width: '100%' }}>
          <Space>
            <SchemaOutlined />
            <Typography.Title level={3} style={{ margin: 0 }}>
              知识图谱模式管理
            </Typography.Title>
          </Space>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        </Space>

        <Card>
          <Space size="large">
            <Typography.Text>节点类型: {nodes.length}</Typography.Text>
            <Typography.Text>关系类型: {rels.length}</Typography.Text>
            <Typography.Text>统计: {JSON.stringify(summary || {})}</Typography.Text>
          </Space>
        </Card>

        <Card title="新增节点类型">
          <Form layout="vertical" form={form} onFinish={createNode}>
            <Form.Item name="label" label="标签" rules={[{ required: true }]}>
              <Input />
            </Form.Item>
            <Form.Item name="properties" label="属性(JSON)">
              <TextArea rows={4} placeholder='[{"name":"id","type":"string"}]' />
            </Form.Item>
            <Button type="primary" htmlType="submit" icon={<PlusOutlined />} loading={loading}>
              创建
            </Button>
          </Form>
        </Card>

        <Card title="节点类型">
          <Table rowKey="label" dataSource={nodes} columns={nodeColumns} loading={loading} />
        </Card>

        <Card title="关系类型">
          <Table rowKey="type" dataSource={rels} columns={relColumns} loading={loading} />
        </Card>
      </Space>
    </div>
  )
}

export default KnowledgeGraphSchemaManagement
