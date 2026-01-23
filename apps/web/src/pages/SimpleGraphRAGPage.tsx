import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useState, useEffect } from 'react'
import { logger } from '../utils/logger'
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Button,
  Tag,
  List,
  Avatar,
  Divider,
  Input,
  message,
  Spin,
  Alert,
  Progress,
  Statistic,
} from 'antd'
import {
  NodeIndexOutlined,
  SearchOutlined,
  BranchesOutlined,
  DatabaseOutlined,
  ThunderboltOutlined,
  CheckCircleOutlined,
  ApiOutlined,
} from '@ant-design/icons'

const { Title, Text } = Typography
const { Search } = Input

interface GraphRAGStats {
  total_entities: number
  total_relationships: number
  total_documents: number
  index_health: string
  query_performance: number
}

interface QueryResult {
  query: string
  answer: string
  sources: string[]
  entities: string[]
  confidence: number
  response_time: number
  results: any[]
}

const SimpleGraphRAGPage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [queryLoading, setQueryLoading] = useState(false)
  const [stats, setStats] = useState<GraphRAGStats>({
    total_entities: 0,
    total_relationships: 0,
    total_documents: 0,
    index_health: 'unknown',
    query_performance: 0,
  })
  const [queryResults, setQueryResults] = useState<QueryResult[]>([])
  const [searchQuery, setSearchQuery] = useState('')

  useEffect(() => {
    loadStats()
  }, [])

  const loadStats = async () => {
    setLoading(true)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/graphrag/stats'))
      const data = await res.json()
      if (data.success) {
        const s = data.stats
        setStats({
          total_entities: s.total_entities,
          total_relationships: s.total_relationships,
          total_documents: s.total_documents,
          index_health: s.index_health,
          query_performance: s.query_performance * 1000,
        })
        message.success('GraphRAG系统状态加载成功')
      } else {
        message.error('加载GraphRAG统计失败')
      }
    } catch (error) {
      logger.error('API调用失败:', error)
      message.error('连接服务器失败')
    } finally {
      setLoading(false)
    }
  }

  const performQuery = async (query: string) => {
    setQueryLoading(true)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/graphrag/query'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      })
      const data = await res.json()
      if (data.success) {
        const newResult: QueryResult = {
          query: data.query,
          answer: data.answer,
          sources: data.sources || [],
          entities: data.entities || [],
          confidence: data.confidence || 0,
          response_time: (data.response_time || 0) * 1000,
          results: data.results || [],
        }
        setQueryResults(prev => [newResult, ...prev.slice(0, 4)])
        message.success(
          `查询完成，耗时 ${newResult.response_time.toFixed(0)}ms`
        )
      } else {
        message.error('GraphRAG查询失败')
      }
    } catch (error) {
      logger.error('查询失败:', error)
      message.error('查询服务连接失败')
    } finally {
      setQueryLoading(false)
    }
  }

  const handleSearch = (value: string) => {
    setSearchQuery(value)
    if (value.trim()) {
      performQuery(value)
    }
  }

  const getHealthColor = (health: string) => {
    const colors: { [key: string]: string } = {
      healthy: 'green',
      degraded: 'orange',
      error: 'red',
    }
    return colors[health] || 'default'
  }

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <NodeIndexOutlined style={{ color: '#722ed1' }} /> GraphRAG知识图谱系统
      </Title>
      <Text type="secondary">基于图结构的检索增强生成系统</Text>

      <Divider />

      {/* 系统状态统计 */}
      <Card style={{ marginBottom: '24px' }} loading={loading}>
        <Row gutter={16}>
          <Col span={6}>
            <Statistic
              title="知识实体"
              value={stats.total_entities}
              prefix={<DatabaseOutlined style={{ color: '#1890ff' }} />}
              suffix="个"
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="关系连接"
              value={stats.total_relationships}
              prefix={<BranchesOutlined style={{ color: '#52c41a' }} />}
              suffix="条"
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="文档数量"
              value={stats.total_documents}
              prefix={<ApiOutlined style={{ color: '#f5222d' }} />}
              suffix="篇"
            />
          </Col>
          <Col span={6}>
            <Card size="small">
              <div style={{ textAlign: 'center' }}>
                <Tag
                  color={getHealthColor(stats.index_health)}
                  icon={<CheckCircleOutlined />}
                >
                  {stats.index_health}
                </Tag>
                <div>
                  <Text strong>{stats.query_performance.toFixed(0)}ms</Text>
                  <div>
                    <Text type="secondary">平均查询时间</Text>
                  </div>
                </div>
              </div>
            </Card>
          </Col>
        </Row>
      </Card>

      {/* 查询界面 */}
      <Card style={{ marginBottom: '24px' }}>
        <Space direction="vertical" style={{ width: '100%' }}>
          <Search
            placeholder="输入问题进行GraphRAG查询..."
            allowClear
            enterButton={
              queryLoading ? <Spin size="small" /> : <SearchOutlined />
            }
            size="large"
            onSearch={handleSearch}
            loading={queryLoading}
            style={{ marginBottom: '16px' }}
          />

          <Space>
            <Button
              onClick={() => handleSearch('什么是机器学习？')}
              disabled={queryLoading}
            >
              示例查询1
            </Button>
            <Button
              onClick={() => handleSearch('深度学习的应用场景有哪些？')}
              disabled={queryLoading}
            >
              示例查询2
            </Button>
            <Button
              icon={<ThunderboltOutlined />}
              onClick={loadStats}
              loading={loading}
            >
              刷新系统状态
            </Button>
          </Space>
        </Space>
      </Card>

      {/* 查询结果 */}
      <Card title="GraphRAG查询结果" loading={queryLoading}>
        {queryResults.length === 0 && !queryLoading ? (
          <Alert
            message="暂无查询结果"
            description="输入问题进行GraphRAG查询，系统将基于知识图谱返回答案"
            type="info"
            showIcon
          />
        ) : (
          <List
            itemLayout="vertical"
            dataSource={queryResults}
            renderItem={(result, index) => (
              <List.Item
                key={index}
                extra={
                  <Space direction="vertical" align="end">
                    <Progress
                      type="circle"
                      size={60}
                      percent={Math.round(result.confidence * 100)}
                      format={percent => `${percent}%`}
                    />
                    <Text type="secondary">
                      {result.response_time.toFixed(0)}ms
                    </Text>
                  </Space>
                }
              >
                <List.Item.Meta
                  avatar={
                    <Avatar
                      icon={<SearchOutlined />}
                      style={{ backgroundColor: '#722ed1' }}
                    />
                  }
                  title={
                    <Space direction="vertical" size="small">
                      <Text strong style={{ color: '#722ed1' }}>
                        问题: {result.query}
                      </Text>
                    </Space>
                  }
                  description={
                    <Space
                      direction="vertical"
                      size="small"
                      style={{ width: '100%' }}
                    >
                      <div>
                        <Text strong>答案: </Text>
                        <Text>{result.answer}</Text>
                      </div>

                      {result.entities.length > 0 && (
                        <div>
                          <Text strong>相关实体: </Text>
                          {result.entities.map((entity, idx) => (
                            <Tag
                              key={idx}
                              color="blue"
                              style={{ margin: '2px' }}
                            >
                              {entity}
                            </Tag>
                          ))}
                        </div>
                      )}

                      {result.sources.length > 0 && (
                        <div>
                          <Text strong>数据源: </Text>
                          {result.sources.map((source, idx) => (
                            <Tag
                              key={idx}
                              color="green"
                              style={{ margin: '2px' }}
                            >
                              {source}
                            </Tag>
                          ))}
                        </div>
                      )}
                    </Space>
                  }
                />
              </List.Item>
            )}
          />
        )}
      </Card>

      {/* 系统能力展示 */}
      <Card title="GraphRAG系统特性" style={{ marginTop: '24px' }}>
        <Row gutter={[16, 16]}>
          <Col span={8}>
            <Card size="small" style={{ textAlign: 'center' }}>
              <NodeIndexOutlined
                style={{
                  fontSize: '24px',
                  color: '#1890ff',
                  marginBottom: '8px',
                }}
              />
              <div>
                <Text strong>知识图谱构建</Text>
                <div>
                  <Text type="secondary">自动实体关系提取</Text>
                </div>
              </div>
            </Card>
          </Col>
          <Col span={8}>
            <Card size="small" style={{ textAlign: 'center' }}>
              <BranchesOutlined
                style={{
                  fontSize: '24px',
                  color: '#52c41a',
                  marginBottom: '8px',
                }}
              />
              <div>
                <Text strong>关系推理</Text>
                <div>
                  <Text type="secondary">多跳路径查询</Text>
                </div>
              </div>
            </Card>
          </Col>
          <Col span={8}>
            <Card size="small" style={{ textAlign: 'center' }}>
              <ThunderboltOutlined
                style={{
                  fontSize: '24px',
                  color: '#f5222d',
                  marginBottom: '8px',
                }}
              />
              <div>
                <Text strong>快速检索</Text>
                <div>
                  <Text type="secondary">向量+图双重索引</Text>
                </div>
              </div>
            </Card>
          </Col>
        </Row>
      </Card>
    </div>
  )
}

export default SimpleGraphRAGPage
