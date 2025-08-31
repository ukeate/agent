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
  Tree,
  Graph,
  Divider,
  notification,
  Spin,
  Upload
} from 'antd'
import {
  ShareAltOutlined,
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
  DatabaseOutlined,
  BranchesOutlined,
  ArrowRightOutlined,
  FilterOutlined,
  SyncOutlined,
  QuestionCircleOutlined
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs
const { TextArea } = Input
const { Option } = Select

interface Relation {
  id: string
  subject: string
  predicate: string
  object: string
  confidence: number
  context: string
  source: string
  verified: boolean
  pattern: string
  extractorType: 'pattern' | 'dependency' | 'neural'
  createdAt: string
}

interface RelationPattern {
  id: string
  name: string
  pattern: string
  type: 'regex' | 'dependency' | 'template'
  relationTypes: string[]
  accuracy: number
  usage: number
  enabled: boolean
  description: string
  examples: string[]
}

interface ExtractionModel {
  id: string
  name: string
  type: 'bert' | 'roberta' | 'gpt' | 'custom'
  language: string
  accuracy: number
  speed: number
  size: string
  status: 'active' | 'inactive' | 'training'
  supportedRelations: string[]
  description: string
  version: string
}

interface RelationJob {
  id: string
  name: string
  status: 'running' | 'completed' | 'failed' | 'pending'
  extractor: string
  documentsCount: number
  relationsFound: number
  progress: number
  accuracy: number
  startTime: string
  duration: string
  parameters: Record<string, any>
}

const RelationExtractionPage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('extraction')
  const [relations, setRelations] = useState<Relation[]>([])
  const [patterns, setPatterns] = useState<RelationPattern[]>([])
  const [models, setModels] = useState<ExtractionModel[]>([])
  const [jobs, setJobs] = useState<RelationJob[]>([])
  const [selectedExtractor, setSelectedExtractor] = useState<string>('pattern-based')
  const [extractionText, setExtractionText] = useState('')
  const [showPatternEditor, setShowPatternEditor] = useState(false)
  const [showJobDetail, setShowJobDetail] = useState(false)
  const [selectedJob, setSelectedJob] = useState<RelationJob | null>(null)
  const [form] = Form.useForm()

  useEffect(() => {
    loadMockData()
  }, [])

  const loadMockData = () => {
    setLoading(true)

    // 模拟关系数据
    const mockRelations: Relation[] = [
      {
        id: '1',
        subject: '张三',
        predicate: 'works_for',
        object: '苹果公司',
        confidence: 0.92,
        context: '张三在苹果公司工作已经三年了',
        source: 'hr_document.txt',
        verified: true,
        pattern: 'X在Y工作',
        extractorType: 'pattern',
        createdAt: '2024-01-20 10:30'
      },
      {
        id: '2',
        subject: '苹果公司',
        predicate: 'located_in',
        object: '加州',
        confidence: 0.95,
        context: '苹果公司总部位于加州库比蒂诺',
        source: 'company_info.txt',
        verified: true,
        pattern: 'X位于Y',
        extractorType: 'dependency',
        createdAt: '2024-01-20 11:15'
      },
      {
        id: '3',
        subject: '李四',
        predicate: 'ceo_of',
        object: '科技公司',
        confidence: 0.88,
        context: '李四是这家科技公司的首席执行官',
        source: 'news_article.txt',
        verified: false,
        pattern: 'X是Y的CEO',
        extractorType: 'neural',
        createdAt: '2024-01-20 09:45'
      }
    ]

    // 模拟模式数据
    const mockPatterns: RelationPattern[] = [
      {
        id: 'p1',
        name: '工作关系模式',
        pattern: '(?P<subject>\\w+)在(?P<object>\\w+)工作',
        type: 'regex',
        relationTypes: ['works_for', 'employed_by'],
        accuracy: 89.5,
        usage: 156,
        enabled: true,
        description: '识别人员与公司的工作关系',
        examples: ['张三在苹果公司工作', '李四在谷歌工作']
      },
      {
        id: 'p2',
        name: '位置关系模式',
        pattern: '(?P<subject>\\w+)位于(?P<object>\\w+)',
        type: 'regex',
        relationTypes: ['located_in', 'based_in'],
        accuracy: 92.3,
        usage: 89,
        enabled: true,
        description: '识别地理位置关系',
        examples: ['公司位于北京', '总部位于上海']
      }
    ]

    // 模拟模型数据
    const mockModels: ExtractionModel[] = [
      {
        id: 'bert-relation',
        name: 'BERT关系抽取模型',
        type: 'bert',
        language: 'Chinese',
        accuracy: 94.2,
        speed: 800,
        size: '800MB',
        status: 'active',
        supportedRelations: ['works_for', 'located_in', 'ceo_of', 'founded_by'],
        description: '基于BERT的中文关系抽取模型',
        version: 'v2.1.0'
      },
      {
        id: 'roberta-large',
        name: 'RoBERTa-Large关系模型',
        type: 'roberta',
        language: 'Chinese',
        accuracy: 96.1,
        speed: 450,
        size: '1.3GB',
        status: 'active',
        supportedRelations: ['works_for', 'located_in', 'subsidiary_of', 'acquired_by'],
        description: 'RoBERTa大模型微调的关系抽取器',
        version: 'v1.8.0'
      }
    ]

    // 模拟任务数据
    const mockJobs: RelationJob[] = [
      {
        id: 'job_1',
        name: '企业关系抽取任务',
        status: 'running',
        extractor: 'bert-relation',
        documentsCount: 3000,
        relationsFound: 4560,
        progress: 72,
        accuracy: 93.8,
        startTime: '2024-01-20 10:00',
        duration: '1h 25m',
        parameters: { confidence_threshold: 0.8, max_length: 512 }
      },
      {
        id: 'job_2',
        name: '新闻关系抽取',
        status: 'completed',
        extractor: 'pattern-based',
        documentsCount: 1500,
        relationsFound: 2340,
        progress: 100,
        accuracy: 87.5,
        startTime: '2024-01-20 08:30',
        duration: '45m',
        parameters: { pattern_set: 'comprehensive', min_confidence: 0.7 }
      }
    ]

    setTimeout(() => {
      setRelations(mockRelations)
      setPatterns(mockPatterns)
      setModels(mockModels)
      setJobs(mockJobs)
      setLoading(false)
    }, 1000)
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'success'
      case 'inactive': return 'default'
      case 'training': return 'processing'
      case 'running': return 'processing'
      case 'completed': return 'success'
      case 'failed': return 'error'
      case 'pending': return 'default'
      default: return 'default'
    }
  }

  const getExtractorColor = (type: string) => {
    switch (type) {
      case 'pattern': return 'blue'
      case 'dependency': return 'green'
      case 'neural': return 'purple'
      default: return 'default'
    }
  }

  const relationColumns: ColumnsType<Relation> = [
    {
      title: '主体',
      dataIndex: 'subject',
      key: 'subject',
      render: (text) => <Tag color="blue">{text}</Tag>
    },
    {
      title: '关系',
      dataIndex: 'predicate',
      key: 'predicate',
      render: (text) => (
        <Space>
          <ArrowRightOutlined style={{ color: '#1890ff' }} />
          <Text strong>{text}</Text>
        </Space>
      )
    },
    {
      title: '客体',
      dataIndex: 'object',
      key: 'object',
      render: (text) => <Tag color="green">{text}</Tag>
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
      title: '抽取器',
      dataIndex: 'extractorType',
      key: 'extractorType',
      render: (type) => (
        <Tag color={getExtractorColor(type)}>
          {type === 'pattern' ? '模式' :
           type === 'dependency' ? '依存' : '神经网络'}
        </Tag>
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

  const patternColumns: ColumnsType<RelationPattern> = [
    {
      title: '模式名称',
      dataIndex: 'name',
      key: 'name',
      render: (name) => <Text strong>{name}</Text>
    },
    {
      title: '模式类型',
      dataIndex: 'type',
      key: 'type',
      render: (type) => (
        <Tag color="blue">
          {type === 'regex' ? '正则表达式' :
           type === 'dependency' ? '依存分析' : '模板匹配'}
        </Tag>
      )
    },
    {
      title: '关系类型',
      dataIndex: 'relationTypes',
      key: 'relationTypes',
      render: (types) => (
        <Space wrap>
          {types.map((type: string) => (
            <Tag key={type} size="small">{type}</Tag>
          ))}
        </Space>
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
      title: '使用次数',
      dataIndex: 'usage',
      key: 'usage'
    },
    {
      title: '状态',
      dataIndex: 'enabled',
      key: 'enabled',
      render: (enabled) => (
        <Switch checked={enabled} size="small" />
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Button size="small" type="link" icon={<EditOutlined />}>
            编辑
          </Button>
          <Button size="small" type="link" icon={<ExperimentOutlined />}>
            测试
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

  const modelColumns: ColumnsType<ExtractionModel> = [
    {
      title: '模型名称',
      dataIndex: 'name',
      key: 'name',
      render: (name, record) => (
        <Space>
          <RobotOutlined />
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
      title: '模型类型',
      dataIndex: 'type',
      key: 'type',
      render: (type) => (
        <Tag color="purple">
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
           status === 'inactive' ? '未激活' : '训练中'}
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
      render: (speed) => `${speed} relations/s`
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
        </Space>
      )
    }
  ]

  const jobColumns: ColumnsType<RelationJob> = [
    {
      title: '任务名称',
      dataIndex: 'name',
      key: 'name',
      render: (name) => <Text strong>{name}</Text>
    },
    {
      title: '抽取器',
      dataIndex: 'extractor',
      key: 'extractor',
      render: (extractor) => <Tag>{extractor}</Tag>
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
      title: '关系数',
      dataIndex: 'relationsFound',
      key: 'relationsFound'
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

  const handleExtraction = async () => {
    if (!extractionText.trim()) {
      notification.warning({ message: '请输入要分析的文本' })
      return
    }

    setLoading(true)
    
    // 模拟关系抽取
    setTimeout(() => {
      notification.success({ message: `使用 ${selectedExtractor} 成功抽取 3 个关系` })
      setLoading(false)
    }, 2000)
  }

  const renderExtraction = () => (
    <Row gutter={[16, 16]}>
      <Col span={24}>
        <Card title="关系抽取工具" extra={
          <Space>
            <Select 
              value={selectedExtractor} 
              onChange={setSelectedExtractor}
              style={{ width: 200 }}
            >
              <Option value="pattern-based">模式匹配</Option>
              <Option value="dependency-based">依存分析</Option>
              <Option value="bert-relation">BERT模型</Option>
              <Option value="roberta-large">RoBERTa模型</Option>
            </Select>
            <Button icon={<SettingOutlined />}>
              配置
            </Button>
          </Space>
        }>
          <Space direction="vertical" style={{ width: '100%' }} size="large">
            <TextArea
              rows={6}
              placeholder="请输入要进行关系抽取的文本..."
              value={extractionText}
              onChange={(e) => setExtractionText(e.target.value)}
            />
            <Row justify="space-between">
              <Col>
                <Space>
                  <Upload>
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
                  icon={<ShareAltOutlined />}
                  loading={loading}
                  onClick={handleExtraction}
                >
                  开始抽取
                </Button>
              </Col>
            </Row>
          </Space>
        </Card>
      </Col>
      <Col span={24}>
        <Card 
          title="抽取结果"
          extra={
            <Space>
              <Button icon={<DownloadOutlined />}>导出结果</Button>
              <Button icon={<FilterOutlined />}>过滤</Button>
            </Space>
          }
        >
          <Table
            columns={relationColumns}
            dataSource={relations}
            rowKey="id"
            loading={loading}
            pagination={{
              showSizeChanger: true,
              showQuickJumper: true,
              showTotal: (total) => `共 ${total} 个关系`
            }}
          />
        </Card>
      </Col>
    </Row>
  )

  const renderPatterns = () => (
    <div>
      <Card 
        title="关系模式管理"
        extra={
          <Space>
            <Button type="primary" icon={<PlusOutlined />} onClick={() => setShowPatternEditor(true)}>
              新建模式
            </Button>
            <Button icon={<UploadOutlined />}>
              导入模式
            </Button>
          </Space>
        }
      >
        <Table
          columns={patternColumns}
          dataSource={patterns}
          rowKey="id"
          loading={loading}
        />
      </Card>
    </div>
  )

  const renderModels = () => (
    <div>
      <Card 
        title="关系抽取模型"
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
            <Button icon={<SyncOutlined />}>
              刷新
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
          <ShareAltOutlined style={{ marginRight: 8 }} />
          关系抽取管理
        </Title>
        <Paragraph type="secondary">
          管理关系抽取模型和模式，执行关系抽取任务，监控抽取性能和准确率
        </Paragraph>
      </div>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="关系抽取" key="extraction">
          {renderExtraction()}
        </TabPane>
        <TabPane tab="抽取模式" key="patterns">
          {renderPatterns()}
        </TabPane>
        <TabPane tab="抽取模型" key="models">
          {renderModels()}
        </TabPane>
        <TabPane tab="任务管理" key="jobs">
          {renderJobs()}
        </TabPane>
      </Tabs>

      {/* 模式编辑器 */}
      <Drawer
        title="关系模式编辑器"
        width={700}
        open={showPatternEditor}
        onClose={() => setShowPatternEditor(false)}
      >
        <Form layout="vertical" form={form}>
          <Form.Item label="模式名称" required>
            <Input placeholder="输入模式名称" />
          </Form.Item>
          <Form.Item label="模式类型" required>
            <Radio.Group defaultValue="regex">
              <Radio value="regex">正则表达式</Radio>
              <Radio value="dependency">依存分析</Radio>
              <Radio value="template">模板匹配</Radio>
            </Radio.Group>
          </Form.Item>
          <Form.Item label="模式表达式" required>
            <TextArea 
              rows={4} 
              placeholder="输入模式表达式..."
              style={{ fontFamily: 'monospace' }}
            />
          </Form.Item>
          <Form.Item label="关系类型">
            <Select mode="tags" placeholder="选择或输入关系类型">
              <Option value="works_for">works_for</Option>
              <Option value="located_in">located_in</Option>
              <Option value="ceo_of">ceo_of</Option>
              <Option value="founded_by">founded_by</Option>
            </Select>
          </Form.Item>
          <Form.Item label="描述">
            <TextArea rows={3} placeholder="输入模式描述..." />
          </Form.Item>
          <Form.Item label="测试样例">
            <TextArea rows={3} placeholder="输入测试样例，每行一个..." />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary">保存模式</Button>
              <Button>测试模式</Button>
              <Button onClick={() => setShowPatternEditor(false)}>取消</Button>
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
                <Statistic title="抽取器" value={selectedJob.extractor} />
              </Col>
              <Col span={12}>
                <Statistic title="文档数量" value={selectedJob.documentsCount} />
              </Col>
              <Col span={12}>
                <Statistic title="关系数量" value={selectedJob.relationsFound} />
              </Col>
              <Col span={12}>
                <Statistic title="准确率" value={selectedJob.accuracy} suffix="%" />
              </Col>
              <Col span={12}>
                <Statistic title="执行时间" value={selectedJob.duration} />
              </Col>
            </Row>
            <Divider />
            <Title level={4}>参数配置</Title>
            <List
              dataSource={Object.entries(selectedJob.parameters)}
              renderItem={([key, value]) => (
                <List.Item>
                  <Text strong>{key}: </Text>
                  <Text code>{String(value)}</Text>
                </List.Item>
              )}
            />
            <div style={{ marginTop: 24 }}>
              <Progress percent={selectedJob.progress} />
            </div>
          </div>
        )}
      </Modal>
    </div>
  )
}

export default RelationExtractionPage