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
  Form,
  Modal,
  Upload,
  Progress,
  Statistic,
  Alert,
  Tooltip,
  Popconfirm,
  Switch,
  Slider,
  Radio,
  Drawer,
  List,
  Badge,
  Timeline,
  notification,
  Spin
} from 'antd'
import {
  BranchesOutlined,
  PlusOutlined,
  SearchOutlined,
  UploadOutlined,
  DownloadOutlined,
  SettingOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  DeleteOutlined,
  EditOutlined,
  EyeOutlined,
  ExperimentOutlined,
  ThunderboltOutlined,
  CheckCircleOutlined,
  ExclamationTriangleOutlined,
  FileTextOutlined,
  NodeIndexOutlined,
  BarChartOutlined,
  ClockCircleOutlined,
  RobotOutlined,
  DatabaseOutlined
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import type { UploadProps } from 'antd'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs
const { TextArea } = Input
const { Option } = Select

interface Entity {
  id: string
  text: string
  label: string
  start: number
  end: number
  confidence: number
  context: string
  source: string
  verified: boolean
  createdAt: string
}

interface Model {
  id: string
  name: string
  type: 'spacy' | 'transformers' | 'stanza' | 'custom'
  language: string
  accuracy: number
  speed: number
  size: string
  status: 'active' | 'inactive' | 'training' | 'loading'
  entities: string[]
  description: string
  version: string
  lastUpdated: string
}

interface ExtractionJob {
  id: string
  name: string
  status: 'running' | 'completed' | 'failed' | 'pending'
  model: string
  documentsCount: number
  entitiesFound: number
  progress: number
  accuracy: number
  startTime: string
  duration: string
}

const EntityRecognitionPage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('extraction')
  const [entities, setEntities] = useState<Entity[]>([])
  const [models, setModels] = useState<Model[]>([])
  const [jobs, setJobs] = useState<ExtractionJob[]>([])
  const [selectedModel, setSelectedModel] = useState<string>('spacy-zh')
  const [extractionText, setExtractionText] = useState('')
  const [showModelConfig, setShowModelConfig] = useState(false)
  const [showJobDetail, setShowJobDetail] = useState(false)
  const [selectedJob, setSelectedJob] = useState<ExtractionJob | null>(null)
  const [form] = Form.useForm()

  useEffect(() => {
    loadMockData()
  }, [])

  const loadMockData = () => {
    setLoading(true)

    // 模拟实体数据
    const mockEntities: Entity[] = [
      {
        id: '1',
        text: '张三',
        label: 'PERSON',
        start: 0,
        end: 2,
        confidence: 0.95,
        context: '张三是一名优秀的软件工程师',
        source: 'news_001.txt',
        verified: true,
        createdAt: '2024-01-20 10:30'
      },
      {
        id: '2',
        text: '苹果公司',
        label: 'ORG',
        start: 5,
        end: 9,
        confidence: 0.98,
        context: '苹果公司发布了新的iPhone',
        source: 'tech_news_002.txt',
        verified: true,
        createdAt: '2024-01-20 11:15'
      },
      {
        id: '3',
        text: '北京',
        label: 'GPE',
        start: 12,
        end: 14,
        confidence: 0.92,
        context: '北京是中国的首都',
        source: 'geography_003.txt',
        verified: false,
        createdAt: '2024-01-20 09:45'
      }
    ]

    // 模拟模型数据
    const mockModels: Model[] = [
      {
        id: 'spacy-zh',
        name: 'spaCy中文模型',
        type: 'spacy',
        language: 'Chinese',
        accuracy: 94.5,
        speed: 1200,
        size: '50MB',
        status: 'active',
        entities: ['PERSON', 'ORG', 'GPE', 'DATE', 'MONEY'],
        description: '基于spaCy的中文命名实体识别模型',
        version: 'v3.4.0',
        lastUpdated: '2024-01-15'
      },
      {
        id: 'bert-large',
        name: 'BERT-Large-Chinese',
        type: 'transformers',
        language: 'Chinese',
        accuracy: 96.2,
        speed: 450,
        size: '1.2GB',
        status: 'active',
        entities: ['PERSON', 'ORG', 'LOC', 'MISC'],
        description: 'BERT大模型微调的中文NER模型',
        version: 'v2.1.0',
        lastUpdated: '2024-01-18'
      },
      {
        id: 'stanza-multi',
        name: 'Stanza多语言模型',
        type: 'stanza',
        language: 'Multi',
        accuracy: 91.8,
        speed: 800,
        size: '200MB',
        status: 'inactive',
        entities: ['PER', 'ORG', 'LOC'],
        description: '支持多语言的Stanza NER模型',
        version: 'v1.4.2',
        lastUpdated: '2024-01-10'
      }
    ]

    // 模拟任务数据
    const mockJobs: ExtractionJob[] = [
      {
        id: 'job_1',
        name: '新闻文档实体抽取',
        status: 'running',
        model: 'spacy-zh',
        documentsCount: 5000,
        entitiesFound: 12450,
        progress: 68,
        accuracy: 94.2,
        startTime: '2024-01-20 10:00',
        duration: '2h 15m'
      },
      {
        id: 'job_2',
        name: '科技文档处理',
        status: 'completed',
        model: 'bert-large',
        documentsCount: 2800,
        entitiesFound: 8900,
        progress: 100,
        accuracy: 96.8,
        startTime: '2024-01-20 08:00',
        duration: '1h 45m'
      }
    ]

    setTimeout(() => {
      setEntities(mockEntities)
      setModels(mockModels)
      setJobs(mockJobs)
      setLoading(false)
    }, 1000)
  }

  const handleExtraction = async () => {
    if (!extractionText.trim()) {
      notification.warning({ message: '请输入要分析的文本' })
      return
    }

    setLoading(true)
    
    // 模拟实体抽取
    const mockExtractedEntities = [
      { text: '张三', label: 'PERSON', start: 0, end: 2, confidence: 0.95 },
      { text: '苹果公司', label: 'ORG', start: 5, end: 9, confidence: 0.98 }
    ]

    setTimeout(() => {
      notification.success({ message: `使用 ${selectedModel} 成功识别 ${mockExtractedEntities.length} 个实体` })
      setLoading(false)
    }, 2000)
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'success'
      case 'inactive': return 'default'
      case 'training': return 'processing'
      case 'loading': return 'warning'
      case 'running': return 'processing'
      case 'completed': return 'success'
      case 'failed': return 'error'
      case 'pending': return 'default'
      default: return 'default'
    }
  }

  const getLabelColor = (label: string) => {
    const colors: Record<string, string> = {
      'PERSON': 'blue',
      'ORG': 'green',
      'GPE': 'orange',
      'LOC': 'purple',
      'DATE': 'cyan',
      'MONEY': 'gold',
      'MISC': 'magenta'
    }
    return colors[label] || 'default'
  }

  const entityColumns: ColumnsType<Entity> = [
    {
      title: '实体文本',
      dataIndex: 'text',
      key: 'text',
      render: (text) => <Text strong>{text}</Text>
    },
    {
      title: '实体类型',
      dataIndex: 'label',
      key: 'label',
      render: (label) => <Tag color={getLabelColor(label)}>{label}</Tag>
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
      title: '来源',
      dataIndex: 'source',
      key: 'source'
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
      render: (_, record) => (
        <Space>
          <Button size="small" type="link" icon={<EyeOutlined />}>
            详情
          </Button>
          <Button size="small" type="link" icon={<EditOutlined />}>
            编辑
          </Button>
          <Popconfirm title="确定删除吗？">
            <Button size="small" type="link" danger icon={<DeleteOutlined />}>
              删除
            </Button>
          </Popconfirm>
        </Space>
      )
    }
  ]

  const modelColumns: ColumnsType<Model> = [
    {
      title: '模型名称',
      dataIndex: 'name',
      key: 'name',
      render: (name, record) => (
        <Space>
          <NodeIndexOutlined />
          <div>
            <Text strong>{name}</Text>
            <br />
            <Text type="secondary" style={{ fontSize: 12 }}>
              {record.version}
            </Text>
          </div>
        </Space>
      )
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type) => (
        <Tag color="blue">
          {type.toUpperCase()}
        </Tag>
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
           status === 'inactive' ? '未激活' :
           status === 'training' ? '训练中' : '加载中'}
        </Tag>
      )
    },
    {
      title: '准确率',
      dataIndex: 'accuracy',
      key: 'accuracy',
      render: (accuracy) => (
        <Statistic
          value={accuracy}
          precision={1}
          suffix="%"
          valueStyle={{ fontSize: 14 }}
        />
      )
    },
    {
      title: '速度',
      dataIndex: 'speed',
      key: 'speed',
      render: (speed) => `${speed} tokens/s`
    },
    {
      title: '大小',
      dataIndex: 'size',
      key: 'size'
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Button 
            size="small" 
            type="link"
            icon={record.status === 'active' ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
          >
            {record.status === 'active' ? '停用' : '启用'}
          </Button>
          <Button size="small" type="link" icon={<SettingOutlined />}>
            配置
          </Button>
          <Button size="small" type="link" icon={<DownloadOutlined />}>
            导出
          </Button>
        </Space>
      )
    }
  ]

  const jobColumns: ColumnsType<ExtractionJob> = [
    {
      title: '任务名称',
      dataIndex: 'name',
      key: 'name',
      render: (name) => <Text strong>{name}</Text>
    },
    {
      title: '模型',
      dataIndex: 'model',
      key: 'model',
      render: (model) => <Tag>{model}</Tag>
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => (
        <Tag color={getStatusColor(status)}>
          {status === 'running' ? '运行中' :
           status === 'completed' ? '已完成' :
           status === 'failed' ? '失败' : '待处理'}
        </Tag>
      )
    },
    {
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      render: (progress, record) => (
        <Progress
          percent={progress}
          size="small"
          status={record.status === 'failed' ? 'exception' : 'active'}
        />
      )
    },
    {
      title: '文档数',
      dataIndex: 'documentsCount',
      key: 'documentsCount'
    },
    {
      title: '实体数',
      dataIndex: 'entitiesFound',
      key: 'entitiesFound'
    },
    {
      title: '准确率',
      dataIndex: 'accuracy',
      key: 'accuracy',
      render: (accuracy) => `${accuracy}%`
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Button 
            size="small" 
            type="link" 
            icon={<EyeOutlined />}
            onClick={() => {
              setSelectedJob(record)
              setShowJobDetail(true)
            }}
          >
            详情
          </Button>
          {record.status === 'running' && (
            <Button size="small" type="link" danger>
              停止
            </Button>
          )}
        </Space>
      )
    }
  ]

  const renderExtraction = () => (
    <Row gutter={[16, 16]}>
      <Col span={24}>
        <Card title="实体抽取工具" extra={
          <Space>
            <Select 
              value={selectedModel} 
              onChange={setSelectedModel}
              style={{ width: 200 }}
            >
              {models.filter(m => m.status === 'active').map(model => (
                <Option key={model.id} value={model.id}>
                  {model.name}
                </Option>
              ))}
            </Select>
            <Button icon={<SettingOutlined />} onClick={() => setShowModelConfig(true)}>
              配置
            </Button>
          </Space>
        }>
          <Space direction="vertical" style={{ width: '100%' }} size="large">
            <TextArea
              rows={6}
              placeholder="请输入要进行实体识别的文本..."
              value={extractionText}
              onChange={(e) => setExtractionText(e.target.value)}
            />
            <Row justify="space-between">
              <Col>
                <Space>
                  <Upload {...{} as UploadProps}>
                    <Button icon={<UploadOutlined />}>
                      上传文件
                    </Button>
                  </Upload>
                  <Button>
                    批量处理
                  </Button>
                </Space>
              </Col>
              <Col>
                <Button 
                  type="primary" 
                  icon={<RobotOutlined />}
                  loading={loading}
                  onClick={handleExtraction}
                >
                  开始识别
                </Button>
              </Col>
            </Row>
          </Space>
        </Card>
      </Col>
      <Col span={24}>
        <Card 
          title="识别结果"
          extra={
            <Space>
              <Button icon={<DownloadOutlined />}>导出结果</Button>
              <Button icon={<SearchOutlined />}>高级搜索</Button>
            </Space>
          }
        >
          <Table
            columns={entityColumns}
            dataSource={entities}
            rowKey="id"
            loading={loading}
            pagination={{
              showSizeChanger: true,
              showQuickJumper: true,
              showTotal: (total) => `共 ${total} 个实体`
            }}
          />
        </Card>
      </Col>
    </Row>
  )

  const renderModels = () => (
    <div>
      <Card 
        title="模型管理"
        extra={
          <Space>
            <Button type="primary" icon={<PlusOutlined />}>
              添加模型
            </Button>
            <Button icon={<UploadOutlined />}>
              导入模型
            </Button>
          </Space>
        }
      >
        <Table
          columns={modelColumns}
          dataSource={models}
          rowKey="id"
          loading={loading}
        />
      </Card>
    </div>
  )

  const renderJobs = () => (
    <div>
      <Card 
        title="抽取任务"
        extra={
          <Space>
            <Button type="primary" icon={<PlusOutlined />}>
              新建任务
            </Button>
            <Button icon={<SearchOutlined />}>
              搜索
            </Button>
          </Space>
        }
      >
        <Table
          columns={jobColumns}
          dataSource={jobs}
          rowKey="id"
          loading={loading}
          pagination={{
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total) => `共 ${total} 个任务`
          }}
        />
      </Card>
    </div>
  )

  return (
    <div style={{ padding: 24 }}>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>
          <BranchesOutlined style={{ marginRight: 8 }} />
          实体识别管理
        </Title>
        <Paragraph type="secondary">
          管理和配置命名实体识别模型，执行实体抽取任务，监控识别性能
        </Paragraph>
      </div>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="实体抽取" key="extraction">
          {renderExtraction()}
        </TabPane>
        <TabPane tab="模型管理" key="models">
          {renderModels()}
        </TabPane>
        <TabPane tab="任务管理" key="jobs">
          {renderJobs()}
        </TabPane>
      </Tabs>

      {/* 模型配置抽屉 */}
      <Drawer
        title="模型配置"
        width={600}
        open={showModelConfig}
        onClose={() => setShowModelConfig(false)}
      >
        <Form layout="vertical">
          <Form.Item label="模型选择">
            <Select value={selectedModel} onChange={setSelectedModel}>
              {models.map(model => (
                <Option key={model.id} value={model.id}>
                  {model.name}
                </Option>
              ))}
            </Select>
          </Form.Item>
          <Form.Item label="置信度阈值">
            <Slider defaultValue={0.8} min={0} max={1} step={0.1} marks={{0: '0', 0.5: '0.5', 1: '1'}} />
          </Form.Item>
          <Form.Item label="批处理大小">
            <Select defaultValue={32}>
              <Option value={16}>16</Option>
              <Option value={32}>32</Option>
              <Option value={64}>64</Option>
              <Option value={128}>128</Option>
            </Select>
          </Form.Item>
          <Form.Item label="最大序列长度">
            <Select defaultValue={512}>
              <Option value={256}>256</Option>
              <Option value={512}>512</Option>
              <Option value={1024}>1024</Option>
            </Select>
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary">保存配置</Button>
              <Button onClick={() => setShowModelConfig(false)}>取消</Button>
            </Space>
          </Form.Item>
        </Form>
      </Drawer>

      {/* 任务详情模态框 */}
      <Modal
        title="任务详情"
        open={showJobDetail}
        onCancel={() => setShowJobDetail(false)}
        footer={null}
        width={800}
      >
        {selectedJob && (
          <div>
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Statistic title="任务名称" value={selectedJob.name} />
              </Col>
              <Col span={12}>
                <Statistic title="使用模型" value={selectedJob.model} />
              </Col>
              <Col span={12}>
                <Statistic title="文档数量" value={selectedJob.documentsCount} />
              </Col>
              <Col span={12}>
                <Statistic title="识别实体" value={selectedJob.entitiesFound} />
              </Col>
              <Col span={12}>
                <Statistic title="准确率" value={selectedJob.accuracy} suffix="%" />
              </Col>
              <Col span={12}>
                <Statistic title="执行时间" value={selectedJob.duration} />
              </Col>
            </Row>
            <div style={{ marginTop: 24 }}>
              <Progress percent={selectedJob.progress} />
            </div>
          </div>
        )}
      </Modal>
    </div>
  )
}

export default EntityRecognitionPage