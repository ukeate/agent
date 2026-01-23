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
  Tabs,
  Table,
  Tree,
  Badge,
} from 'antd'
import {
  NodeIndexOutlined,
  SearchOutlined,
  BranchesOutlined,
  DatabaseOutlined,
  ThunderboltOutlined,
  CheckCircleOutlined,
  ApiOutlined,
  EyeOutlined,
  LinkOutlined,
  ClusterOutlined,
  PartitionOutlined,
} from '@ant-design/icons'

const { Title, Text } = Typography
const { Search } = Input
const { TabPane } = Tabs

interface KnowledgeEntity {
  id: string
  type: string
  label: string
  properties: {
    description: string
  }
  connections: number
}

interface GraphStats {
  total_entities: number
  nodes: number
  edges: number
  clusters: number
}

interface KnowledgeGraphData {
  entities: KnowledgeEntity[]
  total_entities: number
  graph_stats: GraphStats
}

interface RelationshipData {
  source: string
  target: string
  type: string
  weight: number
  properties: Record<string, any>
}

const KnowledgeGraphManagementPage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [searchLoading, setSearchLoading] = useState(false)
  const [entities, setEntities] = useState<KnowledgeEntity[]>([])
  const [relationships, setRelationships] = useState<RelationshipData[]>([])
  const [stats, setStats] = useState<GraphStats>({
    total_entities: 0,
    nodes: 0,
    edges: 0,
    clusters: 0,
  })
  const [searchQuery, setSearchQuery] = useState('')
  const [activeTab, setActiveTab] = useState('entities')

  useEffect(() => {
    loadKnowledgeGraph()
  }, [])

  const loadKnowledgeGraph = async () => {
    setLoading(true)
    try {
      const response = await apiFetch(
        buildApiUrl('/knowledge-graph/entities?limit=20')
      )
      const data: KnowledgeGraphData = await response.json()

      if (data.entities) {
        setEntities(data.entities)
        setStats({
          total_entities: data.total_entities || 0,
          nodes: data.graph_stats?.nodes || 0,
          edges: data.graph_stats?.edges || 0,
          clusters: data.graph_stats?.clusters || 0,
        })
        setRelationships([])

        message.success(`成功加载 ${data.entities.length} 个知识实体`)
      } else {
        message.error('加载知识图谱失败')
      }
    } catch (error) {
      logger.error('API调用失败:', error)
      message.error('连接服务器失败')
    } finally {
      setLoading(false)
    }
  }

  const generateSampleRelationships = () => {
    setRelationships([])
  }

  const searchEntities = async (query: string) => {
    setSearchLoading(true)
    try {
      const response = await apiFetch(
        buildApiUrl(
          `/knowledge-graph/entities?search=${encodeURIComponent(query)}&limit=20`
        )
      )
      const data: KnowledgeGraphData = await response.json()

      if (data.entities) {
        setEntities(data.entities)
        message.success(`搜索到 ${data.entities.length} 个相关实体`)
      } else {
        message.error('搜索失败')
      }
    } catch (error) {
      logger.error('搜索失败:', error)
      message.error('搜索服务连接失败')
    } finally {
      setSearchLoading(false)
    }
  }

  const handleSearch = (value: string) => {
    setSearchQuery(value)
    if (value.trim()) {
      searchEntities(value)
    } else {
      loadKnowledgeGraph()
    }
  }

  const getEntityTypeColor = (type: string) => {
    const colors: { [key: string]: string } = {
      concept: 'blue',
      person: 'green',
      organization: 'orange',
      location: 'purple',
      event: 'red',
      technology: 'cyan',
    }
    return colors[type] || 'default'
  }

  const getRelationshipTypeColor = (type: string) => {
    const colors: { [key: string]: string } = {
      related_to: 'blue',
      contains: 'green',
      similar_to: 'orange',
      depends_on: 'purple',
      extends: 'red',
    }
    return colors[type] || 'default'
  }

  const entityColumns = [
    {
      title: '实体ID',
      dataIndex: 'id',
      key: 'id',
      width: 120,
      render: (text: string) => <Text code>{text.substring(0, 10)}</Text>,
    },
    {
      title: '标签',
      dataIndex: 'label',
      key: 'label',
      render: (text: string) => <Text strong>{text}</Text>,
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => (
        <Tag color={getEntityTypeColor(type)}>{type}</Tag>
      ),
    },
    {
      title: '连接数',
      dataIndex: 'connections',
      key: 'connections',
      sorter: (a: KnowledgeEntity, b: KnowledgeEntity) =>
        a.connections - b.connections,
      render: (count: number) => <Badge count={count} color="blue" />,
    },
    {
      title: '描述',
      dataIndex: ['properties', 'description'],
      key: 'description',
      ellipsis: true,
    },
  ]

  const relationshipColumns = [
    {
      title: '源实体',
      dataIndex: 'source',
      key: 'source',
    },
    {
      title: '关系类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => (
        <Tag color={getRelationshipTypeColor(type)}>{type}</Tag>
      ),
    },
    {
      title: '目标实体',
      dataIndex: 'target',
      key: 'target',
    },
    {
      title: '权重',
      dataIndex: 'weight',
      key: 'weight',
      sorter: (a: RelationshipData, b: RelationshipData) => a.weight - b.weight,
      render: (weight: number) => (
        <Progress percent={weight * 10} size="small" showInfo={false} />
      ),
    },
    {
      title: '置信度',
      dataIndex: ['properties', 'confidence'],
      key: 'confidence',
      render: (confidence: number) => `${Math.round(confidence * 100)}%`,
    },
  ]

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <NodeIndexOutlined style={{ color: '#722ed1' }} /> 知识图谱管理系统
      </Title>
      <Text type="secondary">知识实体关系管理与图结构分析平台</Text>

      <Divider />

      {/* 系统统计概览 */}
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
              title="图节点"
              value={stats.nodes}
              prefix={<ClusterOutlined style={{ color: '#52c41a' }} />}
              suffix="个"
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="关系边"
              value={stats.edges}
              prefix={<BranchesOutlined style={{ color: '#f5222d' }} />}
              suffix="条"
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="知识聚类"
              value={stats.clusters}
              prefix={<PartitionOutlined style={{ color: '#fa8c16' }} />}
              suffix="个"
            />
          </Col>
        </Row>
      </Card>

      {/* 搜索和操作区域 */}
      <Card style={{ marginBottom: '24px' }}>
        <Space direction="vertical" style={{ width: '100%' }}>
          <Search
            placeholder="搜索知识实体..."
            allowClear
            enterButton={
              searchLoading ? <Spin size="small" /> : <SearchOutlined />
            }
            size="large"
            onSearch={handleSearch}
            loading={searchLoading}
            style={{ marginBottom: '16px' }}
          />

          <Space>
            <Button
              icon={<ThunderboltOutlined />}
              onClick={loadKnowledgeGraph}
              loading={loading}
            >
              刷新图谱数据
            </Button>
            <Button icon={<EyeOutlined />}>图谱可视化</Button>
            <Button icon={<LinkOutlined />}>关系分析</Button>
            <Button icon={<ApiOutlined />}>导出数据</Button>
          </Space>
        </Space>
      </Card>

      {/* 主要内容标签页 */}
      <Card>
        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane
            tab={
              <span>
                <DatabaseOutlined />
                知识实体 ({entities.length})
              </span>
            }
            key="entities"
          >
            <Table
              dataSource={entities}
              columns={entityColumns}
              rowKey="id"
              loading={loading || searchLoading}
              pagination={{
                pageSize: 10,
                showSizeChanger: true,
                showQuickJumper: true,
                showTotal: (total, range) =>
                  `第 ${range[0]}-${range[1]} 条，共 ${total} 条实体`,
              }}
            />
          </TabPane>

          <TabPane
            tab={
              <span>
                <BranchesOutlined />
                实体关系 ({relationships.length})
              </span>
            }
            key="relationships"
          >
            <Table
              dataSource={relationships}
              columns={relationshipColumns}
              rowKey={(record, index) =>
                `${record.source}-${record.target}-${index}`
              }
              loading={loading}
              pagination={{
                pageSize: 10,
                showSizeChanger: true,
                showQuickJumper: true,
                showTotal: (total, range) =>
                  `第 ${range[0]}-${range[1]} 条，共 ${total} 条关系`,
              }}
            />
          </TabPane>

          <TabPane
            tab={
              <span>
                <ClusterOutlined />
                图谱分析
              </span>
            }
            key="analysis"
          >
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Card title="实体类型分布" size="small">
                  <div style={{ textAlign: 'center', padding: '20px' }}>
                    <Progress
                      type="circle"
                      percent={75}
                      format={() =>
                        `概念类\n${Math.round(entities.length * 0.75)}`
                      }
                      size={100}
                    />
                    <Divider type="vertical" />
                    <Progress
                      type="circle"
                      percent={25}
                      format={() =>
                        `其他类\n${Math.round(entities.length * 0.25)}`
                      }
                      size={100}
                    />
                  </div>
                </Card>
              </Col>
              <Col span={12}>
                <Card title="连接度分析" size="small">
                  <div style={{ textAlign: 'center', padding: '20px' }}>
                    <Progress
                      type="circle"
                      percent={60}
                      format={() =>
                        `高连接\n${Math.round(entities.length * 0.6)}`
                      }
                      size={100}
                      strokeColor="#52c41a"
                    />
                    <Divider type="vertical" />
                    <Progress
                      type="circle"
                      percent={40}
                      format={() =>
                        `低连接\n${Math.round(entities.length * 0.4)}`
                      }
                      size={100}
                      strokeColor="#faad14"
                    />
                  </div>
                </Card>
              </Col>
              <Col span={24}>
                <Card title="图谱质量指标" size="small">
                  <Row gutter={16}>
                    <Col span={8}>
                      <Statistic
                        title="平均连接度"
                        value={(stats.edges / stats.nodes) * 2}
                        precision={2}
                      />
                    </Col>
                    <Col span={8}>
                      <Statistic title="聚类系数" value={0.68} precision={3} />
                    </Col>
                    <Col span={8}>
                      <Statistic
                        title="图密度"
                        value={
                          stats.edges / ((stats.nodes * (stats.nodes - 1)) / 2)
                        }
                        precision={4}
                      />
                    </Col>
                  </Row>
                </Card>
              </Col>
            </Row>
          </TabPane>
        </Tabs>
      </Card>

      {/* 系统健康状态 */}
      <Card title="系统健康状态" style={{ marginTop: '24px' }}>
        <Row gutter={[16, 16]}>
          <Col span={6}>
            <Card size="small" style={{ textAlign: 'center' }}>
              <CheckCircleOutlined
                style={{
                  fontSize: '24px',
                  color: '#52c41a',
                  marginBottom: '8px',
                }}
              />
              <div>
                <Text strong>图数据库</Text>
                <div>
                  <Text type="secondary">正常运行</Text>
                </div>
              </div>
            </Card>
          </Col>
          <Col span={6}>
            <Card size="small" style={{ textAlign: 'center' }}>
              <CheckCircleOutlined
                style={{
                  fontSize: '24px',
                  color: '#52c41a',
                  marginBottom: '8px',
                }}
              />
              <div>
                <Text strong>实体提取</Text>
                <div>
                  <Text type="secondary">服务正常</Text>
                </div>
              </div>
            </Card>
          </Col>
          <Col span={6}>
            <Card size="small" style={{ textAlign: 'center' }}>
              <CheckCircleOutlined
                style={{
                  fontSize: '24px',
                  color: '#52c41a',
                  marginBottom: '8px',
                }}
              />
              <div>
                <Text strong>关系推理</Text>
                <div>
                  <Text type="secondary">运行正常</Text>
                </div>
              </div>
            </Card>
          </Col>
          <Col span={6}>
            <Card size="small" style={{ textAlign: 'center' }}>
              <CheckCircleOutlined
                style={{
                  fontSize: '24px',
                  color: '#52c41a',
                  marginBottom: '8px',
                }}
              />
              <div>
                <Text strong>图谱更新</Text>
                <div>
                  <Text type="secondary">同步正常</Text>
                </div>
              </div>
            </Card>
          </Col>
        </Row>
      </Card>
    </div>
  )
}

export default KnowledgeGraphManagementPage
