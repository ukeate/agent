import React, { useEffect, useState } from 'react'
import { Card, Row, Col, Input, Button, Typography, List, Tag, Space, message } from 'antd'
import { SearchOutlined, ReloadOutlined } from '@ant-design/icons'
import { graphRAGService, type GraphRAGResponse } from '../services/graphRAGService'

const { Title, Text } = Typography

const GraphRAGPageEnhanced: React.FC = () => {
  const [question, setQuestion] = useState('什么是GraphRAG的核心优势？')
  const [result, setResult] = useState<GraphRAGResponse | null>(null)
  const [loading, setLoading] = useState(false)

  const runQuery = async () => {
    setLoading(true)
    try {
      const data = await graphRAGService.query({
        query: question,
        retrieval_mode: 'hybrid',
        max_docs: 10,
        include_reasoning: true,
        expansion_depth: 2,
        confidence_threshold: 0.6,
      })
      setResult(data)
    } catch (e: any) {
      message.error(e?.message || '查询失败')
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    runQuery()
  }, [])

  return (
    <div style={{ padding: 24 }}>
      <Row justify="space-between" align="middle" style={{ marginBottom: 16 }}>
        <Col>
          <Title level={3}>GraphRAG 增强查询</Title>
          <Text type="secondary">直接调用后端 GraphRAG 接口，无任何本地静态数据</Text>
        </Col>
        <Col>
          <Button icon={<ReloadOutlined />} onClick={runQuery} loading={loading}>重新查询</Button>
        </Col>
      </Row>

      <Card style={{ marginBottom: 16 }}>
        <Input.Search
          value={question}
          onChange={e => setQuestion(e.target.value)}
          enterButton={<><SearchOutlined /> 搜索</>}
          onSearch={runQuery}
          loading={loading}
          name="graphrag-question"
          placeholder="输入问题并执行 GraphRAG 查询"
        />
      </Card>

      <Card title="回答">
        {result ? (
          <div>
            <Space>
              <Tag color="blue">文档 {result.documents.length}</Tag>
              <Tag color="green">实体 {(result.graph_context?.entities || []).length}</Tag>
              <Tag color="gold">关系 {(result.graph_context?.relations || []).length}</Tag>
              <Tag color="purple">用时 {(((result.performance_metrics.total_time as number) || 0) * 1000).toFixed(0)}ms</Tag>
            </Space>
            <div className="mt-3">
              <Text strong>查询:</Text> <Text>{result.query}</Text>
            </div>
          </div>
        ) : (
          <Text type="secondary">暂无结果</Text>
        )}
      </Card>

      <Row gutter={16} style={{ marginTop: 16 }}>
        <Col span={12}>
          <Card title="文档来源">
            <List
              dataSource={result?.documents || []}
              renderItem={(d: any, index) => (
                <List.Item>
                  <Space direction="vertical">
                    <Text strong>Top {index + 1}</Text>
                    <Text type="secondary">{d.content || '-'}</Text>
                    <Tag color="blue">分数 {(((d.final_score ?? d.score) || 0) * 100).toFixed(1)}%</Tag>
                  </Space>
                </List.Item>
              )}
            />
          </Card>
        </Col>
        <Col span={12}>
          <Card title="图谱上下文">
            <List
              dataSource={[
                { key: 'entities', title: '实体数', value: (result?.graph_context?.entities || []).length },
                { key: 'relations', title: '关系数', value: (result?.graph_context?.relations || []).length },
              ]}
              renderItem={(g: any) => (
                <List.Item>
                  <Space direction="vertical">
                    <Text strong>{g.title}</Text>
                    <Text>{g.value}</Text>
                  </Space>
                </List.Item>
              )}
            />
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default GraphRAGPageEnhanced
