import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Statistic,
  Progress,
  Table,
  Tag,
  Button,
  Space,
  Typography,
  Tabs,
  Alert,
  Timeline,
  Spin,
  Select,
  DatePicker,
  Input,
  List,
  Avatar,
  Divider,
  Badge,
  Tooltip,
  notification
} from 'antd'
import {
  NodeIndexOutlined,
  ShareAltOutlined,
  LinkOutlined,
  GlobalOutlined,
  ThunderboltOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  DashboardOutlined,
  BarChartOutlined,
  ClockCircleOutlined,
  RocketOutlined,
  FileTextOutlined,
  BugOutlined,
  SettingOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  SearchOutlined
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs
const { RangePicker } = DatePicker

interface ExtractionTask {
  id: string
  name: string
  type: 'entity' | 'relation' | 'linking' | 'multilingual'
  status: 'running' | 'completed' | 'failed' | 'pending'
  progress: number
  documentsProcessed: number
  totalDocuments: number
  entitiesExtracted: number
  relationsExtracted: number
  accuracy: number
  duration: string
  model: string
  language: string
  createdAt: string
}

interface SystemMetrics {
  totalTasks: number
  activeTasks: number
  completedTasks: number
  failedTasks: number
  totalDocuments: number
  totalEntities: number
  totalRelations: number
  averageAccuracy: number
  systemLoad: number
  memoryUsage: number
  throughput: number
}

const KnowledgeExtractionOverviewPage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('overview')
  const [tasks, setTasks] = useState<ExtractionTask[]>([])
  const [metrics, setMetrics] = useState<SystemMetrics>({
    totalTasks: 156,
    activeTasks: 8,
    completedTasks: 142,
    failedTasks: 6,
    totalDocuments: 45320,
    totalEntities: 89456,
    totalRelations: 23789,
    averageAccuracy: 94.2,
    systemLoad: 67,
    memoryUsage: 72,
    throughput: 1250
  })

  useEffect(() => {
    loadMockData()
  }, [])

  const loadMockData = () => {
    setLoading(true)
    
    // 模拟数据
    const mockTasks: ExtractionTask[] = [
      {
        id: '1',
        name: '新闻文档实体识别',
        type: 'entity',
        status: 'running',
        progress: 68,
        documentsProcessed: 3400,
        totalDocuments: 5000,
        entitiesExtracted: 12450,
        relationsExtracted: 0,
        accuracy: 95.2,
        duration: '2h 15m',
        model: 'spaCy-zh',
        language: 'Chinese',
        createdAt: '2024-01-20 10:30'
      },
      {
        id: '2',
        name: '企业关系抽取',
        type: 'relation',
        status: 'completed',
        progress: 100,
        documentsProcessed: 2800,
        totalDocuments: 2800,
        entitiesExtracted: 8900,
        relationsExtracted: 4560,
        accuracy: 92.8,
        duration: '1h 45m',
        model: 'BERT-large',
        language: 'Chinese',
        createdAt: '2024-01-20 08:15'
      },
      {
        id: '3',
        name: '多语言文档处理',
        type: 'multilingual',
        status: 'running',
        progress: 25,
        documentsProcessed: 500,
        totalDocuments: 2000,
        entitiesExtracted: 1200,
        relationsExtracted: 300,
        accuracy: 89.5,
        duration: '45m',
        model: 'XLM-R',
        language: 'Multi',
        createdAt: '2024-01-20 11:00'
      },
      {
        id: '4',
        name: '实体链接任务',
        type: 'linking',
        status: 'failed',
        progress: 15,
        documentsProcessed: 150,
        totalDocuments: 1000,
        entitiesExtracted: 450,
        relationsExtracted: 0,
        accuracy: 78.3,
        duration: '25m',
        model: 'Entity-Linker-v2',
        language: 'English',
        createdAt: '2024-01-20 09:30'
      }
    ]

    setTimeout(() => {
      setTasks(mockTasks)
      setLoading(false)
    }, 1000)
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'processing'
      case 'completed': return 'success'
      case 'failed': return 'error'
      case 'pending': return 'default'
      default: return 'default'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running': return <PlayCircleOutlined />
      case 'completed': return <CheckCircleOutlined />
      case 'failed': return <ExclamationCircleOutlined />
      case 'pending': return <ClockCircleOutlined />
      default: return <ClockCircleOutlined />
    }
  }

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'entity': return <NodeIndexOutlined />
      case 'relation': return <ShareAltOutlined />
      case 'linking': return <LinkOutlined />
      case 'multilingual': return <GlobalOutlined />
      default: return <FileTextOutlined />
    }
  }

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'entity': return 'blue'
      case 'relation': return 'green'
      case 'linking': return 'orange'
      case 'multilingual': return 'purple'
      default: return 'default'
    }
  }

  const taskColumns: ColumnsType<ExtractionTask> = [
    {
      title: '任务名称',
      dataIndex: 'name',
      key: 'name',
      render: (text, record) => (
        <Space>
          {getTypeIcon(record.type)}
          <Text strong>{text}</Text>
        </Space>
      )
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type) => (
        <Tag color={getTypeColor(type)}>
          {type === 'entity' ? '实体识别' :
           type === 'relation' ? '关系抽取' :
           type === 'linking' ? '实体链接' : '多语言处理'}
        </Tag>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => (
        <Tag color={getStatusColor(status)} icon={getStatusIcon(status)}>
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
        <div style={{ width: 120 }}>
          <Progress 
            percent={progress} 
            size="small" 
            status={record.status === 'failed' ? 'exception' : 'active'}
            format={() => `${record.documentsProcessed}/${record.totalDocuments}`}
          />
        </div>
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
      title: '模型',
      dataIndex: 'model',
      key: 'model',
      render: (model) => <Tag>{model}</Tag>
    },
    {
      title: '语言',
      dataIndex: 'language',
      key: 'language'
    },
    {
      title: '耗时',
      dataIndex: 'duration',
      key: 'duration'
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Button size="small" type="link">详情</Button>
          {record.status === 'running' && 
            <Button size="small" type="link" danger>停止</Button>}
          {record.status === 'failed' && 
            <Button size="small" type="link">重试</Button>}
        </Space>
      )
    }
  ]

  const renderOverview = () => (
    <div>
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="总任务数"
              value={metrics.totalTasks}
              prefix={<DashboardOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="活跃任务"
              value={metrics.activeTasks}
              prefix={<PlayCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="已完成"
              value={metrics.completedTasks}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="失败任务"
              value={metrics.failedTasks}
              prefix={<ExclamationCircleOutlined />}
              valueStyle={{ color: '#ff4d4f' }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={8}>
          <Card>
            <Statistic
              title="文档总数"
              value={metrics.totalDocuments}
              prefix={<FileTextOutlined />}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic
              title="实体总数"
              value={metrics.totalEntities}
              prefix={<NodeIndexOutlined />}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic
              title="关系总数"
              value={metrics.totalRelations}
              prefix={<ShareAltOutlined />}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={8}>
          <Card title="平均准确率">
            <Progress
              type="circle"
              percent={metrics.averageAccuracy}
              format={percent => `${percent}%`}
              strokeColor="#52c41a"
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card title="系统负载">
            <Progress
              type="circle"
              percent={metrics.systemLoad}
              format={percent => `${percent}%`}
              strokeColor={metrics.systemLoad > 80 ? "#ff4d4f" : "#1890ff"}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card title="内存使用">
            <Progress
              type="circle"
              percent={metrics.memoryUsage}
              format={percent => `${percent}%`}
              strokeColor={metrics.memoryUsage > 85 ? "#ff4d4f" : "#52c41a"}
            />
          </Card>
        </Col>
      </Row>

      <Card title="系统吞吐量" style={{ marginBottom: 24 }}>
        <Statistic
          title="每分钟处理文档数"
          value={metrics.throughput}
          suffix="docs/min"
          prefix={<ThunderboltOutlined />}
          valueStyle={{ color: '#1890ff' }}
        />
      </Card>
    </div>
  )

  const renderTaskList = () => (
    <div>
      <Card 
        title="抽取任务列表"
        extra={
          <Space>
            <Input.Search placeholder="搜索任务" style={{ width: 200 }} />
            <Select defaultValue="all" style={{ width: 120 }}>
              <Select.Option value="all">所有状态</Select.Option>
              <Select.Option value="running">运行中</Select.Option>
              <Select.Option value="completed">已完成</Select.Option>
              <Select.Option value="failed">失败</Select.Option>
            </Select>
            <Button type="primary" icon={<ReloadOutlined />}>
              刷新
            </Button>
          </Space>
        }
      >
        <Table
          columns={taskColumns}
          dataSource={tasks}
          rowKey="id"
          loading={loading}
          pagination={{
            total: tasks.length,
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total) => `共 ${total} 个任务`
          }}
        />
      </Card>
    </div>
  )

  const renderSystemHealth = () => (
    <div>
      <Row gutter={[16, 16]}>
        <Col span={12}>
          <Card title="系统状态">
            <List>
              <List.Item>
                <List.Item.Meta
                  avatar={<Badge status="success" />}
                  title="实体识别服务"
                  description="运行正常，响应时间 < 100ms"
                />
              </List.Item>
              <List.Item>
                <List.Item.Meta
                  avatar={<Badge status="success" />}
                  title="关系抽取服务"
                  description="运行正常，响应时间 < 150ms"
                />
              </List.Item>
              <List.Item>
                <List.Item.Meta
                  avatar={<Badge status="warning" />}
                  title="实体链接服务"
                  description="性能下降，响应时间 > 300ms"
                />
              </List.Item>
              <List.Item>
                <List.Item.Meta
                  avatar={<Badge status="success" />}
                  title="多语言处理服务"
                  description="运行正常，响应时间 < 200ms"
                />
              </List.Item>
            </List>
          </Card>
        </Col>
        <Col span={12}>
          <Card title="资源使用情况">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Text>CPU使用率</Text>
                <Progress percent={67} status="active" />
              </div>
              <div>
                <Text>内存使用率</Text>
                <Progress percent={72} status="active" />
              </div>
              <div>
                <Text>GPU使用率</Text>
                <Progress percent={84} status="active" />
              </div>
              <div>
                <Text>磁盘使用率</Text>
                <Progress percent={45} />
              </div>
              <div>
                <Text>网络带宽</Text>
                <Progress percent={35} />
              </div>
            </Space>
          </Card>
        </Col>
      </Row>

      <Card title="最近错误日志" style={{ marginTop: 16 }}>
        <Timeline>
          <Timeline.Item color="red">
            <Text type="danger">2024-01-20 11:25:30</Text>
            <br />
            实体链接任务失败：连接知识库超时
          </Timeline.Item>
          <Timeline.Item color="yellow">
            <Text type="warning">2024-01-20 10:45:15</Text>
            <br />
            关系抽取准确率下降：当前批次准确率 85.2%
          </Timeline.Item>
          <Timeline.Item color="green">
            <Text type="success">2024-01-20 09:30:00</Text>
            <br />
            系统自动扩容：增加2个处理节点
          </Timeline.Item>
        </Timeline>
      </Card>
    </div>
  )

  return (
    <div style={{ padding: 24 }}>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>
          <NodeIndexOutlined style={{ marginRight: 8 }} />
          知识抽取总览
        </Title>
        <Paragraph type="secondary">
          实时监控知识抽取系统的运行状态、任务进度和系统性能指标
        </Paragraph>
      </div>

      <Alert
        message="系统运行正常"
        description="所有核心服务运行稳定，当前系统负载适中，建议关注实体链接服务的性能情况"
        type="success"
        showIcon
        style={{ marginBottom: 24 }}
      />

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="系统概览" key="overview">
          {renderOverview()}
        </TabPane>
        <TabPane tab="任务列表" key="tasks">
          {renderTaskList()}
        </TabPane>
        <TabPane tab="系统健康" key="health">
          {renderSystemHealth()}
        </TabPane>
      </Tabs>
    </div>
  )
}

export default KnowledgeExtractionOverviewPage