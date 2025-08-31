import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Button,
  Input,
  Select,
  Space,
  Typography,
  Tabs,
  Slider,
  Switch,
  Tooltip,
  Tag,
  Badge,
  Statistic,
  Alert,
  Form,
  Drawer,
  List,
  Avatar,
  notification
} from 'antd'
import {
  NodeExpandOutlined,
  SearchOutlined,
  SettingOutlined,
  ZoomInOutlined,
  ZoomOutOutlined,
  ReloadOutlined,
  DownloadOutlined,
  FullscreenOutlined,
  FilterOutlined,
  ShareAltOutlined,
  NodeIndexOutlined,
  EyeOutlined,
  PlusOutlined,
  DeleteOutlined
} from '@ant-design/icons'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs
const { Option } = Select

interface GraphNode {
  id: string
  label: string
  type: 'entity' | 'concept'
  category: string
  size: number
  color: string
  properties: Record<string, any>
}

interface GraphEdge {
  id: string
  source: string
  target: string
  label: string
  type: string
  weight: number
  properties: Record<string, any>
}

interface GraphLayout {
  id: string
  name: string
  description: string
  parameters: Record<string, any>
}

const KnowledgeGraphVisualizationPage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('graph')
  const [nodes, setNodes] = useState<GraphNode[]>([])
  const [edges, setEdges] = useState<GraphEdge[]>([])
  const [layouts, setLayouts] = useState<GraphLayout[]>([])
  const [selectedLayout, setSelectedLayout] = useState('force-directed')
  const [showSettings, setShowSettings] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)

  useEffect(() => {
    loadMockData()
  }, [])

  const loadMockData = () => {
    setLoading(true)

    const mockNodes: GraphNode[] = [
      {
        id: 'apple',
        label: '苹果公司',
        type: 'entity',
        category: 'organization',
        size: 30,
        color: '#1890ff',
        properties: {
          founded: '1976',
          headquarters: '加州库比蒂诺',
          industry: '科技'
        }
      },
      {
        id: 'jobs',
        label: '史蒂夫·乔布斯',
        type: 'entity',
        category: 'person',
        size: 25,
        color: '#52c41a',
        properties: {
          born: '1955',
          died: '2011',
          occupation: 'CEO'
        }
      },
      {
        id: 'iphone',
        label: 'iPhone',
        type: 'entity',
        category: 'product',
        size: 20,
        color: '#fa8c16',
        properties: {
          launched: '2007',
          type: '智能手机'
        }
      }
    ]

    const mockEdges: GraphEdge[] = [
      {
        id: 'edge1',
        source: 'jobs',
        target: 'apple',
        label: '创立',
        type: 'founded',
        weight: 1.0,
        properties: { year: '1976' }
      },
      {
        id: 'edge2',
        source: 'apple',
        target: 'iphone',
        label: '开发',
        type: 'developed',
        weight: 0.8,
        properties: { year: '2007' }
      }
    ]

    const mockLayouts: GraphLayout[] = [
      {
        id: 'force-directed',
        name: '力导向布局',
        description: '基于物理模拟的自动布局算法',
        parameters: {
          repulsion: 300,
          attraction: 0.1,
          gravity: 0.05
        }
      },
      {
        id: 'hierarchical',
        name: '层次布局',
        description: '按照层次结构排列节点',
        parameters: {
          direction: 'TB',
          levelSeparation: 150,
          nodeSeparation: 100
        }
      },
      {
        id: 'circular',
        name: '环形布局',
        description: '将节点排列成圆形',
        parameters: {
          radius: 200,
          ordering: 'degree'
        }
      }
    ]

    setTimeout(() => {
      setNodes(mockNodes)
      setEdges(mockEdges)
      setLayouts(mockLayouts)
      setLoading(false)
    }, 1000)
  }

  const handleSearch = (value: string) => {
    setSearchQuery(value)
    if (value) {
      notification.info({ message: `搜索节点: ${value}` })
    }
  }

  const handleLayoutChange = (layoutId: string) => {
    setSelectedLayout(layoutId)
    notification.success({ message: `已切换到${layouts.find(l => l.id === layoutId)?.name}` })
  }

  const renderGraphCanvas = () => (
    <Card 
      title="知识图谱可视化"
      extra={
        <Space>
          <Input.Search
            placeholder="搜索节点或关系"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onSearch={handleSearch}
            style={{ width: 200 }}
          />
          <Select 
            value={selectedLayout}
            onChange={handleLayoutChange}
            style={{ width: 150 }}
          >
            {layouts.map(layout => (
              <Option key={layout.id} value={layout.id}>
                {layout.name}
              </Option>
            ))}
          </Select>
          <Button icon={<SettingOutlined />} onClick={() => setShowSettings(true)}>
            设置
          </Button>
          <Button icon={<ZoomInOutlined />}>
            放大
          </Button>
          <Button icon={<ZoomOutOutlined />}>
            缩小
          </Button>
          <Button icon={<ReloadOutlined />}>
            重置
          </Button>
          <Button icon={<FullscreenOutlined />}>
            全屏
          </Button>
        </Space>
      }
      style={{ height: 600 }}
    >
      <div 
        style={{ 
          height: 500, 
          background: '#f5f5f5', 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          border: '1px dashed #d9d9d9',
          borderRadius: 4
        }}
      >
        <div style={{ textAlign: 'center' }}>
          <NodeExpandOutlined style={{ fontSize: 48, color: '#d9d9d9' }} />
          <div style={{ marginTop: 16, color: '#999' }}>
            <Text>知识图谱可视化区域</Text>
            <br />
            <Text type="secondary">当前显示 {nodes.length} 个节点，{edges.length} 条关系</Text>
          </div>
        </div>
      </div>
    </Card>
  )

  const renderGraphStats = () => (
    <Row gutter={[16, 16]}>
      <Col span={6}>
        <Card>
          <Statistic
            title="节点总数"
            value={nodes.length}
            prefix={<NodeIndexOutlined />}
            valueStyle={{ color: '#1890ff' }}
          />
        </Card>
      </Col>
      <Col span={6}>
        <Card>
          <Statistic
            title="关系总数"
            value={edges.length}
            prefix={<ShareAltOutlined />}
            valueStyle={{ color: '#52c41a' }}
          />
        </Card>
      </Col>
      <Col span={6}>
        <Card>
          <Statistic
            title="实体类型"
            value={new Set(nodes.map(n => n.category)).size}
            prefix={<Badge />}
            valueStyle={{ color: '#fa8c16' }}
          />
        </Card>
      </Col>
      <Col span={6}>
        <Card>
          <Statistic
            title="关系类型"
            value={new Set(edges.map(e => e.type)).size}
            prefix={<Badge />}
            valueStyle={{ color: '#722ed1' }}
          />
        </Card>
      </Col>
    </Row>
  )

  const renderNodeList = () => (
    <Card title="节点列表" extra={
      <Space>
        <Button icon={<PlusOutlined />} type="primary">添加节点</Button>
        <Button icon={<FilterOutlined />}>过滤</Button>
      </Space>
    }>
      <List
        itemLayout="horizontal"
        dataSource={nodes}
        renderItem={(node) => (
          <List.Item
            actions={[
              <Button size="small" type="link" icon={<EyeOutlined />}>查看</Button>,
              <Button size="small" type="link" icon={<DeleteOutlined />} danger>删除</Button>
            ]}
          >
            <List.Item.Meta
              avatar={
                <Avatar 
                  style={{ backgroundColor: node.color }}
                  icon={<NodeIndexOutlined />}
                />
              }
              title={
                <Space>
                  <Text strong>{node.label}</Text>
                  <Tag color="blue">{node.category}</Tag>
                </Space>
              }
              description={
                <Space wrap>
                  {Object.entries(node.properties).map(([key, value]) => (
                    <Tag key={key} size="small">
                      {key}: {String(value)}
                    </Tag>
                  ))}
                </Space>
              }
            />
          </List.Item>
        )}
      />
    </Card>
  )

  const renderEdgeList = () => (
    <Card title="关系列表" extra={
      <Space>
        <Button icon={<PlusOutlined />} type="primary">添加关系</Button>
        <Button icon={<FilterOutlined />}>过滤</Button>
      </Space>
    }>
      <List
        itemLayout="horizontal"
        dataSource={edges}
        renderItem={(edge) => (
          <List.Item
            actions={[
              <Button size="small" type="link" icon={<EyeOutlined />}>查看</Button>,
              <Button size="small" type="link" icon={<DeleteOutlined />} danger>删除</Button>
            ]}
          >
            <List.Item.Meta
              avatar={<Avatar icon={<ShareAltOutlined />} />}
              title={
                <Space>
                  <Text>{nodes.find(n => n.id === edge.source)?.label}</Text>
                  <Text strong>{edge.label}</Text>
                  <Text>{nodes.find(n => n.id === edge.target)?.label}</Text>
                </Space>
              }
              description={
                <Space>
                  <Tag color="green">{edge.type}</Tag>
                  <Text type="secondary">权重: {edge.weight}</Text>
                </Space>
              }
            />
          </List.Item>
        )}
      />
    </Card>
  )

  const renderAnalysis = () => (
    <Row gutter={[16, 16]}>
      <Col span={12}>
        <Card title="图谱分析">
          <Space direction="vertical" style={{ width: '100%' }}>
            <div>
              <Text>图密度</Text>
              <div>0.67</div>
            </div>
            <div>
              <Text>平均度数</Text>
              <div>2.3</div>
            </div>
            <div>
              <Text>连通分量</Text>
              <div>1</div>
            </div>
            <div>
              <Text>最短路径</Text>
              <div>2.1</div>
            </div>
          </Space>
        </Card>
      </Col>
      <Col span={12}>
        <Card title="中心性分析">
          <List
            size="small"
            dataSource={nodes.sort((a, b) => b.size - a.size)}
            renderItem={(node) => (
              <List.Item>
                <Text>{node.label}</Text>
                <Badge count={node.size} />
              </List.Item>
            )}
          />
        </Card>
      </Col>
    </Row>
  )

  return (
    <div style={{ padding: 24 }}>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>
          <NodeExpandOutlined style={{ marginRight: 8 }} />
          知识图谱可视化
        </Title>
        <Paragraph type="secondary">
          交互式知识图谱可视化，支持多种布局算法和图分析功能
        </Paragraph>
      </div>

      <Alert
        message="图谱渲染正常"
        description="当前使用力导向布局，显示效果良好"
        type="success"
        showIcon
        style={{ marginBottom: 24 }}
      />

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="图谱视图" key="graph">
          <Space direction="vertical" style={{ width: '100%' }} size="large">
            {renderGraphCanvas()}
            {renderGraphStats()}
          </Space>
        </TabPane>
        <TabPane tab="节点管理" key="nodes">
          {renderNodeList()}
        </TabPane>
        <TabPane tab="关系管理" key="edges">
          {renderEdgeList()}
        </TabPane>
        <TabPane tab="图分析" key="analysis">
          {renderAnalysis()}
        </TabPane>
      </Tabs>

      {/* 设置抽屉 */}
      <Drawer
        title="可视化设置"
        width={400}
        open={showSettings}
        onClose={() => setShowSettings(false)}
      >
        <Form layout="vertical">
          <Form.Item label="布局算法">
            <Select value={selectedLayout} onChange={handleLayoutChange}>
              {layouts.map(layout => (
                <Option key={layout.id} value={layout.id}>
                  {layout.name}
                </Option>
              ))}
            </Select>
          </Form.Item>
          <Form.Item label="节点大小">
            <Slider defaultValue={20} min={10} max={50} />
          </Form.Item>
          <Form.Item label="边粗细">
            <Slider defaultValue={2} min={1} max={10} />
          </Form.Item>
          <Form.Item label="显示标签">
            <Switch defaultChecked />
          </Form.Item>
          <Form.Item label="显示边标签">
            <Switch defaultChecked />
          </Form.Item>
          <Form.Item label="动画效果">
            <Switch defaultChecked />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary">应用设置</Button>
              <Button>重置</Button>
            </Space>
          </Form.Item>
        </Form>
      </Drawer>
    </div>
  )
}

export default KnowledgeGraphVisualizationPage