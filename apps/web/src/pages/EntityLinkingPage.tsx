import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Button,
  Input,
  Select,
  Table,
  Tag,
  Space,
  Typography,
  Tabs,
  Progress,
  Statistic,
  Alert,
  Tooltip,
  List,
  Badge,
  Tree,
  notification,
  Spin,
  Drawer,
  Form,
  Modal,
  Switch
} from 'antd'
import {
  LinkOutlined,
  SearchOutlined,
  DatabaseOutlined,
  NodeIndexOutlined,
  ThunderboltOutlined,
  CheckCircleOutlined,
  ExclamationTriangleOutlined,
  EyeOutlined,
  EditOutlined,
  DeleteOutlined,
  PlusOutlined,
  SettingOutlined,
  SyncOutlined,
  FilterOutlined,
  DownloadOutlined
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs
const { Option } = Select

interface LinkedEntity {
  id: string
  mention: string
  linkedEntity: string
  confidence: number
  knowledgeBase: string
  context: string
  source: string
  verified: boolean
  alternativeCandidates: string[]
  createdAt: string
}

interface KnowledgeBase {
  id: string
  name: string
  type: 'wikipedia' | 'wikidata' | 'dbpedia' | 'custom'
  language: string
  entityCount: number
  status: 'active' | 'inactive' | 'updating'
  accuracy: number
  coverage: number
  lastUpdated: string
  description: string
}

const EntityLinkingPage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('linking')
  const [linkedEntities, setLinkedEntities] = useState<LinkedEntity[]>([])
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([])
  const [linkingText, setLinkingText] = useState('')
  const [selectedKB, setSelectedKB] = useState<string>('wikipedia-zh')

  useEffect(() => {
    loadMockData()
  }, [])

  const loadMockData = () => {
    setLoading(true)

    const mockLinkedEntities: LinkedEntity[] = [
      {
        id: '1',
        mention: '苹果公司',
        linkedEntity: 'Apple Inc.',
        confidence: 0.95,
        knowledgeBase: 'Wikipedia',
        context: '苹果公司是一家美国科技公司',
        source: 'tech_news.txt',
        verified: true,
        alternativeCandidates: ['Apple (disambiguation)', 'Apple Records'],
        createdAt: '2024-01-20 10:30'
      },
      {
        id: '2',
        mention: '北京',
        linkedEntity: 'Beijing',
        confidence: 0.98,
        knowledgeBase: 'Wikidata',
        context: '北京是中国的首都',
        source: 'geography.txt',
        verified: true,
        alternativeCandidates: ['Beijing University', 'Beijing Opera'],
        createdAt: '2024-01-20 11:15'
      }
    ]

    const mockKnowledgeBases: KnowledgeBase[] = [
      {
        id: 'wikipedia-zh',
        name: '中文维基百科',
        type: 'wikipedia',
        language: 'Chinese',
        entityCount: 1250000,
        status: 'active',
        accuracy: 94.2,
        coverage: 87.5,
        lastUpdated: '2024-01-20',
        description: '中文维基百科知识库，包含大量中文实体信息'
      },
      {
        id: 'wikidata',
        name: 'Wikidata',
        type: 'wikidata',
        language: 'Multi',
        entityCount: 105000000,
        status: 'active',
        accuracy: 96.1,
        coverage: 92.3,
        lastUpdated: '2024-01-20',
        description: '多语言结构化知识库，提供丰富的实体关系信息'
      }
    ]

    setTimeout(() => {
      setLinkedEntities(mockLinkedEntities)
      setKnowledgeBases(mockKnowledgeBases)
      setLoading(false)
    }, 1000)
  }

  const handleLinking = async () => {
    if (!linkingText.trim()) {
      notification.warning({ message: '请输入要链接的文本' })
      return
    }

    setLoading(true)
    setTimeout(() => {
      notification.success({ message: `成功链接到 ${selectedKB}，找到 5 个候选实体` })
      setLoading(false)
    }, 2000)
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'success'
      case 'inactive': return 'default'
      case 'updating': return 'processing'
      default: return 'default'
    }
  }

  const linkedEntityColumns: ColumnsType<LinkedEntity> = [
    {
      title: '实体提及',
      dataIndex: 'mention',
      key: 'mention',
      render: (text) => <Text strong>{text}</Text>
    },
    {
      title: '链接实体',
      dataIndex: 'linkedEntity',
      key: 'linkedEntity',
      render: (entity, record) => (
        <Space direction="vertical" size="small">
          <Tag color="blue">{entity}</Tag>
          <Text type="secondary" style={{ fontSize: 12 }}>
            来源: {record.knowledgeBase}
          </Text>
        </Space>
      )
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      key: 'confidence',
      render: (confidence) => (
        <Progress
          percent={Math.round(confidence * 100)}
          size="small"
          format={percent => `${percent}%`}
        />
      )
    },
    {
      title: '候选数量',
      dataIndex: 'alternativeCandidates',
      key: 'alternativeCandidates',
      render: (candidates) => (
        <Badge count={candidates.length} showZero />
      )
    },
    {
      title: '上下文',
      dataIndex: 'context',
      key: 'context',
      ellipsis: true,
      render: (context) => (
        <Tooltip title={context}>
          <Text ellipsis>{context}</Text>
        </Tooltip>
      )
    },
    {
      title: '验证状态',
      dataIndex: 'verified',
      key: 'verified',
      render: (verified) => (
        <Badge 
          status={verified ? 'success' : 'warning'} 
          text={verified ? '已验证' : '待验证'}
        />
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: () => (
        <Space>
          <Button size="small" type="link" icon={<EyeOutlined />}>
            详情
          </Button>
          <Button size="small" type="link" icon={<EditOutlined />}>
            编辑
          </Button>
          <Button size="small" type="link" danger icon={<DeleteOutlined />}>
            删除
          </Button>
        </Space>
      )
    }
  ]

  const kbColumns: ColumnsType<KnowledgeBase> = [
    {
      title: '知识库名称',
      dataIndex: 'name',
      key: 'name',
      render: (name, record) => (
        <Space>
          <DatabaseOutlined />
          <div>
            <Text strong>{name}</Text>
            <br />
            <Text type="secondary" style={{ fontSize: 12 }}>
              {record.type.toUpperCase()}
            </Text>
          </div>
        </Space>
      )
    },
    {
      title: '语言',
      dataIndex: 'language',
      key: 'language'
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => (
        <Tag color={getStatusColor(status)}>
          {status === 'active' ? '活跃' :
           status === 'inactive' ? '未激活' : '更新中'}
        </Tag>
      )
    },
    {
      title: '实体数量',
      dataIndex: 'entityCount',
      key: 'entityCount',
      render: (count) => count.toLocaleString()
    },
    {
      title: '准确率',
      dataIndex: 'accuracy',
      key: 'accuracy',
      render: (accuracy) => `${accuracy}%`
    },
    {
      title: '覆盖率',
      dataIndex: 'coverage',
      key: 'coverage',
      render: (coverage) => `${coverage}%`
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Switch checked={record.status === 'active'} size="small" />
          <Button size="small" type="link" icon={<SettingOutlined />}>
            配置
          </Button>
          <Button size="small" type="link" icon={<SyncOutlined />}>
            更新
          </Button>
        </Space>
      )
    }
  ]

  const renderLinking = () => (
    <Row gutter={[16, 16]}>
      <Col span={24}>
        <Card title="实体链接工具" extra={
          <Space>
            <Select 
              value={selectedKB} 
              onChange={setSelectedKB}
              style={{ width: 200 }}
            >
              {knowledgeBases.filter(kb => kb.status === 'active').map(kb => (
                <Option key={kb.id} value={kb.id}>
                  {kb.name}
                </Option>
              ))}
            </Select>
            <Button icon={<SettingOutlined />}>
              配置
            </Button>
          </Space>
        }>
          <Space direction="vertical" style={{ width: '100%' }} size="large">
            <Input.TextArea
              rows={6}
              placeholder="请输入包含实体的文本，系统将自动识别并链接到知识库..."
              value={linkingText}
              onChange={(e) => setLinkingText(e.target.value)}
            />
            <Row justify="space-between">
              <Col>
                <Space>
                  <Button>
                    上传文件
                  </Button>
                  <Button>
                    批量链接
                  </Button>
                </Space>
              </Col>
              <Col>
                <Button 
                  type="primary" 
                  icon={<LinkOutlined />}
                  loading={loading}
                  onClick={handleLinking}
                >
                  开始链接
                </Button>
              </Col>
            </Row>
          </Space>
        </Card>
      </Col>
      <Col span={24}>
        <Card 
          title="链接结果"
          extra={
            <Space>
              <Button icon={<DownloadOutlined />}>导出结果</Button>
              <Button icon={<FilterOutlined />}>过滤</Button>
            </Space>
          }
        >
          <Table
            columns={linkedEntityColumns}
            dataSource={linkedEntities}
            rowKey="id"
            loading={loading}
            pagination={{
              showSizeChanger: true,
              showQuickJumper: true,
              showTotal: (total) => `共 ${total} 个链接`
            }}
          />
        </Card>
      </Col>
    </Row>
  )

  const renderKnowledgeBases = () => (
    <div>
      <Card 
        title="知识库管理"
        extra={
          <Space>
            <Button type="primary" icon={<PlusOutlined />}>
              添加知识库
            </Button>
            <Button icon={<SyncOutlined />}>
              同步更新
            </Button>
          </Space>
        }
      >
        <Table
          columns={kbColumns}
          dataSource={knowledgeBases}
          rowKey="id"
          loading={loading}
        />
      </Card>
    </div>
  )

  return (
    <div style={{ padding: 24 }}>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>
          <LinkOutlined style={{ marginRight: 8 }} />
          实体链接管理
        </Title>
        <Paragraph type="secondary">
          将识别出的实体提及链接到结构化知识库，增强实体的语义信息
        </Paragraph>
      </div>

      <Alert
        message="实体链接服务运行正常"
        description="当前连接到 2 个知识库，平均链接准确率 95.2%"
        type="success"
        showIcon
        style={{ marginBottom: 24 }}
      />

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="实体链接" key="linking">
          {renderLinking()}
        </TabPane>
        <TabPane tab="知识库管理" key="knowledge-bases">
          {renderKnowledgeBases()}
        </TabPane>
      </Tabs>
    </div>
  )
}

export default EntityLinkingPage