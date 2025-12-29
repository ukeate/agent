import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Row, Col, Button, Input, Select, Table, Space, Typography, Alert, Spin, Tag } from 'antd'
import { LinkOutlined, ReloadOutlined, SearchOutlined } from '@ant-design/icons'

const { Title } = Typography
const { Option } = Select

type LinkedEntity = {
  id: string
  mention: string
  linked_entity: string
  confidence?: number
  knowledge_base?: string
  source?: string
  created_at?: string
}

type KnowledgeBase = { id: string; name: string; type?: string; language?: string; status?: string }

const EntityLinkingPage: React.FC = () => {
  const [linked, setLinked] = useState<LinkedEntity[]>([])
  const [kbs, setKbs] = useState<KnowledgeBase[]>([])
  const [text, setText] = useState('')
  const [kb, setKb] = useState<string | undefined>()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const [lRes, kbRes] = await Promise.all([
        apiFetch(buildApiUrl('/api/v1/knowledge-graph/entity-linking/history'),
        apiFetch(buildApiUrl('/api/v1/knowledge-graph/knowledge-bases'))
      ])
      const lData = await lRes.json()
      const kbData = await kbRes.json()
      setLinked(Array.isArray(lData?.links) ? lData.links : [])
      setKbs(Array.isArray(kbData?.knowledge_bases) ? kbData.knowledge_bases : [])
    } catch (e: any) {
      setError(e?.message || '加载失败')
      setLinked([])
      setKbs([])
    } finally {
      setLoading(false)
    }
  }

  const submit = async () => {
    if (!text) return
    setLoading(true)
    setError(null)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/knowledge-graph/entity-linking'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, knowledge_base: kb })
      })
      const data = await res.json()
      setLinked(data?.links || [])
    } catch (e: any) {
      setError(e?.message || '提交失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Space align="center" style={{ justifyContent: 'space-between', width: '100%' }}>
          <Space>
            <LinkOutlined />
            <Title level={3} style={{ margin: 0 }}>
              实体链接
            </Title>
          </Space>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        </Space>

        {error && <Alert type="error" message="加载失败" description={error} />}

        <Card title="链接请求">
          <Space>
            <Input
              style={{ width: 400 }}
              placeholder="输入待链接的文本片段"
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
            <Select
              allowClear
              style={{ width: 200 }}
              placeholder="选择知识库"
              value={kb}
              onChange={setKb}
            >
              {kbs.map((item) => (
                <Option key={item.id} value={item.id}>
                  {item.name}
                </Option>
              ))}
            </Select>
            <Button icon={<SearchOutlined />} type="primary" onClick={submit} loading={loading}>
              链接
            </Button>
          </Space>
        </Card>

        <Card title="链接结果">
          {loading ? (
            <Spin />
          ) : (
            <Table
              rowKey="id"
              dataSource={linked}
              columns={[
                { title: '提及', dataIndex: 'mention' },
                { title: '实体', dataIndex: 'linked_entity' },
                { title: '置信度', dataIndex: 'confidence' },
                { title: '知识库', dataIndex: 'knowledge_base' },
                { title: '来源', dataIndex: 'source' },
                { title: '时间', dataIndex: 'created_at' },
              ]}
              locale={{ emptyText: '暂无数据，提交请求后查看。' }}
            />
          )}
        </Card>
      </Space>
    </div>
  )
}

export default EntityLinkingPage
