import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Button,
  Table,
  Tabs,
  Progress,
  Badge,
  Space,
  Typography,
  Upload,
  Modal,
  Form,
  Input,
  Select,
  InputNumber,
  Switch,
  Alert,
  Tag,
  Statistic,
  Timeline,
  Tooltip,
  Popconfirm,
  Drawer,
  Tree,
  Checkbox,
  Radio,
  DatePicker,
  Divider,
} from 'antd'
import {
  CloudUploadOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  DeleteOutlined,
  DownloadOutlined,
  SettingOutlined,
  MonitorOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  FileTextOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  BarsOutlined,
  LineChartOutlined,
  ReloadOutlined,
  FilterOutlined,
  ExportOutlined,
  ImportOutlined,
  ScheduleOutlined,
  TeamOutlined,
  SyncOutlined,
} from '@ant-design/icons'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, AreaChart, Area, BarChart, Bar } from 'recharts'

const { Title, Text } = Typography
const { TabPane } = Tabs
const { TextArea } = Input
const { Option } = Select
const { RangePicker } = DatePicker
const { Dragger } = Upload

interface BatchJob {
  id: string
  name: string
  status: 'pending' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled'
  progress: number
  totalQueries: number
  processedQueries: number
  successQueries: number
  failedQueries: number
  startTime?: string
  endTime?: string
  estimatedTime?: string
  strategy: string
  priority: 'high' | 'medium' | 'low'
  createdBy: string
  outputFormat: 'json' | 'csv' | 'xml'
  results?: any[]
}

interface QueueStats {
  totalJobs: number
  runningJobs: number
  pendingJobs: number
  completedJobs: number
  failedJobs: number
  totalQueries: number
  averageProcessingTime: number
  systemLoad: number
}

const KGReasoningBatchPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('jobs')
  const [batchJobs, setBatchJobs] = useState<BatchJob[]>([])
  const [queueStats, setQueueStats] = useState<QueueStats>({
    totalJobs: 0,
    runningJobs: 0,
    pendingJobs: 0,
    completedJobs: 0,
    failedJobs: 0,
    totalQueries: 0,
    averageProcessingTime: 0,
    systemLoad: 0
  })
  const [selectedJob, setSelectedJob] = useState<BatchJob | null>(null)
  const [createJobVisible, setCreateJobVisible] = useState(false)
  const [jobDetailVisible, setJobDetailVisible] = useState(false)
  const [configVisible, setConfigVisible] = useState(false)
  const [form] = Form.useForm()

  // 模拟批处理作业数据
  const mockBatchJobs: BatchJob[] = [
    {
      id: 'batch_001',
      name: '产品知识图谱推理',
      status: 'running',
      progress: 68,
      totalQueries: 10000,
      processedQueries: 6800,
      successQueries: 6450,
      failedQueries: 350,
      startTime: '2024-01-20 09:15:00',
      estimatedTime: '12分钟',
      strategy: 'ensemble',
      priority: 'high',
      createdBy: 'admin',
      outputFormat: 'json',
    },
    {
      id: 'batch_002',
      name: '用户关系分析',
      status: 'pending',
      progress: 0,
      totalQueries: 5000,
      processedQueries: 0,
      successQueries: 0,
      failedQueries: 0,
      strategy: 'path_only',
      priority: 'medium',
      createdBy: 'analyst',
      outputFormat: 'csv',
    },
    {
      id: 'batch_003',
      name: '语义相似度批量计算',
      status: 'completed',
      progress: 100,
      totalQueries: 8000,
      processedQueries: 8000,
      successQueries: 7650,
      failedQueries: 350,
      startTime: '2024-01-20 08:00:00',
      endTime: '2024-01-20 08:45:00',
      strategy: 'embedding_only',
      priority: 'low',
      createdBy: 'system',
      outputFormat: 'json',
    },
    {
      id: 'batch_004',
      name: '规则验证批处理',
      status: 'failed',
      progress: 35,
      totalQueries: 3000,
      processedQueries: 1050,
      successQueries: 890,
      failedQueries: 160,
      startTime: '2024-01-20 10:00:00',
      endTime: '2024-01-20 10:15:00',
      strategy: 'rule_only',
      priority: 'high',
      createdBy: 'validator',
      outputFormat: 'xml',
    }
  ]

  // 模拟队列统计数据
  const mockQueueStats: QueueStats = {
    totalJobs: 24,
    runningJobs: 3,
    pendingJobs: 8,
    completedJobs: 11,
    failedJobs: 2,
    totalQueries: 125000,
    averageProcessingTime: 2.8,
    systemLoad: 75
  }

  // 模拟性能数据
  const performanceData = [
    { time: '08:00', throughput: 450, errorRate: 2.1, systemLoad: 65 },
    { time: '09:00', throughput: 520, errorRate: 1.8, systemLoad: 72 },
    { time: '10:00', throughput: 680, errorRate: 3.2, systemLoad: 85 },
    { time: '11:00', throughput: 720, errorRate: 2.5, systemLoad: 78 },
    { time: '12:00', throughput: 600, errorRate: 1.9, systemLoad: 70 },
    { time: '13:00', throughput: 750, errorRate: 2.8, systemLoad: 82 },
  ]

  useEffect(() => {
    setBatchJobs(mockBatchJobs)
    setQueueStats(mockQueueStats)
  }, [])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'processing'
      case 'completed': return 'success'
      case 'failed': return 'error'
      case 'pending': return 'default'
      case 'paused': return 'warning'
      case 'cancelled': return 'default'
      default: return 'default'
    }
  }

  const getStatusText = (status: string) => {
    switch (status) {
      case 'running': return '执行中'
      case 'completed': return '已完成'
      case 'failed': return '失败'
      case 'pending': return '等待中'
      case 'paused': return '已暂停'
      case 'cancelled': return '已取消'
      default: return '未知'
    }
  }

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'red'
      case 'medium': return 'orange'
      case 'low': return 'green'
      default: return 'default'
    }
  }

  const handleJobAction = (action: string, jobId: string) => {
    setBatchJobs(prev => prev.map(job => {
      if (job.id === jobId) {
        switch (action) {
          case 'start':
            return { ...job, status: 'running' as const }
          case 'pause':
            return { ...job, status: 'paused' as const }
          case 'stop':
            return { ...job, status: 'cancelled' as const }
          case 'delete':
            return null
          default:
            return job
        }
      }
      return job
    }).filter(Boolean) as BatchJob[])
  }

  const handleCreateJob = (values: any) => {
    const newJob: BatchJob = {
      id: `batch_${Date.now()}`,
      name: values.name,
      status: 'pending',
      progress: 0,
      totalQueries: values.totalQueries || 0,
      processedQueries: 0,
      successQueries: 0,
      failedQueries: 0,
      strategy: values.strategy,
      priority: values.priority,
      createdBy: 'current_user',
      outputFormat: values.outputFormat,
    }
    
    setBatchJobs(prev => [newJob, ...prev])
    setCreateJobVisible(false)
    form.resetFields()
  }

  const jobColumns = [
    {
      title: '作业ID',
      dataIndex: 'id',
      key: 'id',
      width: 120,
      render: (id: string) => <Text code>{id}</Text>
    },
    {
      title: '作业名称',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: BatchJob) => (
        <Space direction="vertical" size={2}>
          <Text strong>{name}</Text>
          <Space size={4}>
            <Tag color={getPriorityColor(record.priority)}>{record.priority}</Tag>
            <Text type="secondary" style={{ fontSize: '12px' }}>by {record.createdBy}</Text>
          </Space>
        </Space>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 120,
      render: (status: string) => (
        <Badge status={getStatusColor(status)} text={getStatusText(status)} />
      )
    },
    {
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      width: 200,
      render: (progress: number, record: BatchJob) => (
        <Space direction="vertical" size={2} style={{ width: '100%' }}>
          <Progress 
            percent={progress} 
            size="small" 
            status={record.status === 'failed' ? 'exception' : 
                   record.status === 'completed' ? 'success' : 'active'}
          />
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {record.processedQueries.toLocaleString()} / {record.totalQueries.toLocaleString()}
          </Text>
        </Space>
      )
    },
    {
      title: '成功/失败',
      key: 'results',
      width: 120,
      render: (_, record: BatchJob) => (
        <Space direction="vertical" size={2}>
          <Text style={{ color: '#52c41a', fontSize: '12px' }}>
            ✓ {record.successQueries.toLocaleString()}
          </Text>
          <Text style={{ color: '#ff4d4f', fontSize: '12px' }}>
            ✗ {record.failedQueries.toLocaleString()}
          </Text>
        </Space>
      )
    },
    {
      title: '策略',
      dataIndex: 'strategy',
      key: 'strategy',
      width: 100,
      render: (strategy: string) => <Tag color="blue">{strategy}</Tag>
    },
    {
      title: '时间',
      key: 'time',
      width: 150,
      render: (_, record: BatchJob) => (
        <Space direction="vertical" size={2}>
          {record.startTime && (
            <Text type="secondary" style={{ fontSize: '12px' }}>
              开始: {record.startTime.split(' ')[1]}
            </Text>
          )}
          {record.endTime ? (
            <Text type="secondary" style={{ fontSize: '12px' }}>
              结束: {record.endTime.split(' ')[1]}
            </Text>
          ) : record.estimatedTime && (
            <Text type="secondary" style={{ fontSize: '12px' }}>
              预计: {record.estimatedTime}
            </Text>
          )}
        </Space>
      )
    },
    {
      title: '操作',
      key: 'actions',
      width: 200,
      render: (_, record: BatchJob) => (
        <Space size="small" wrap>
          {record.status === 'pending' && (
            <Button 
              size="small" 
              type="primary" 
              icon={<PlayCircleOutlined />}
              onClick={() => handleJobAction('start', record.id)}
            >
              启动
            </Button>
          )}
          {record.status === 'running' && (
            <>
              <Button 
                size="small" 
                icon={<PauseCircleOutlined />}
                onClick={() => handleJobAction('pause', record.id)}
              >
                暂停
              </Button>
              <Button 
                size="small" 
                danger
                icon={<StopOutlined />}
                onClick={() => handleJobAction('stop', record.id)}
              >
                停止
              </Button>
            </>
          )}
          {record.status === 'paused' && (
            <Button 
              size="small" 
              type="primary"
              icon={<PlayCircleOutlined />}
              onClick={() => handleJobAction('start', record.id)}
            >
              继续
            </Button>
          )}
          {record.status === 'completed' && (
            <Button 
              size="small" 
              icon={<DownloadOutlined />}
            >
              下载
            </Button>
          )}
          <Button 
            size="small" 
            onClick={() => {
              setSelectedJob(record)
              setJobDetailVisible(true)
            }}
          >
            详情
          </Button>
          <Popconfirm
            title="确定删除此作业吗?"
            onConfirm={() => handleJobAction('delete', record.id)}
          >
            <Button size="small" danger icon={<DeleteOutlined />} />
          </Popconfirm>
        </Space>
      )
    }
  ]

  const uploadProps = {
    name: 'file',
    multiple: false,
    action: '/api/batch/upload',
    accept: '.json,.csv,.txt',
    beforeUpload: (file: any) => {
      // 文件验证逻辑
      return true
    },
    onChange(info: any) {
      const { status } = info.file
      if (status === 'done') {
        // 处理上传成功
      } else if (status === 'error') {
        // 处理上传失败
      }
    }
  }

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <BarsOutlined /> 批量推理处理
        </Title>
        <Text type="secondary">
          大规模推理查询的批量处理、作业管理和结果分析
        </Text>
      </div>

      {/* 统计概览 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={12} sm={6}>
          <Card>
            <Statistic 
              title="总作业数" 
              value={queueStats.totalJobs} 
              prefix={<DatabaseOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card>
            <Statistic 
              title="运行中" 
              value={queueStats.runningJobs} 
              prefix={<PlayCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card>
            <Statistic 
              title="队列中" 
              value={queueStats.pendingJobs} 
              prefix={<ClockCircleOutlined />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
        <Col xs={12} sm={6}>
          <Card>
            <Statistic 
              title="系统负载" 
              value={queueStats.systemLoad} 
              suffix="%" 
              prefix={<MonitorOutlined />}
              valueStyle={{ color: queueStats.systemLoad > 80 ? '#ff4d4f' : '#52c41a' }}
            />
          </Card>
        </Col>
      </Row>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="作业管理" key="jobs">
          <Card title="批处理作业列表" extra={
            <Space>
              <Button 
                type="primary" 
                icon={<CloudUploadOutlined />}
                onClick={() => setCreateJobVisible(true)}
              >
                创建作业
              </Button>
              <Button icon={<SettingOutlined />} onClick={() => setConfigVisible(true)}>
                队列配置
              </Button>
              <Button icon={<ReloadOutlined />}>刷新</Button>
            </Space>
          }>
            <Table 
              dataSource={batchJobs}
              columns={jobColumns}
              rowKey="id"
              pagination={{ 
                showSizeChanger: true,
                showTotal: (total) => `共 ${total} 个作业`
              }}
            />
          </Card>
        </TabPane>

        <TabPane tab="性能监控" key="performance">
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={12}>
              <Card title="处理吞吐量" extra={<LineChartOutlined />}>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis />
                    <RechartsTooltip />
                    <Area 
                      type="monotone" 
                      dataKey="throughput" 
                      stroke="#1890ff" 
                      fill="#1890ff" 
                      fillOpacity={0.3}
                      name="查询/分钟"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card title="错误率 & 系统负载">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="time" />
                    <YAxis yAxisId="left" />
                    <YAxis yAxisId="right" orientation="right" />
                    <RechartsTooltip />
                    <Line 
                      yAxisId="left"
                      type="monotone" 
                      dataKey="errorRate" 
                      stroke="#ff4d4f" 
                      name="错误率(%)"
                    />
                    <Line 
                      yAxisId="right"
                      type="monotone" 
                      dataKey="systemLoad" 
                      stroke="#faad14" 
                      name="系统负载(%)"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Card>
            </Col>
          </Row>

          <Card title="实时队列状态" style={{ marginTop: '16px' }}>
            <Row gutter={16}>
              <Col span={6}>
                <Progress 
                  type="circle" 
                  percent={queueStats.systemLoad} 
                  format={percent => `${percent}%`}
                  strokeColor={queueStats.systemLoad > 80 ? '#ff4d4f' : '#52c41a'}
                />
                <div style={{ textAlign: 'center', marginTop: '8px' }}>
                  <Text>系统负载</Text>
                </div>
              </Col>
              <Col span={18}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>
                    <Text>处理队列: </Text>
                    <Progress 
                      percent={(queueStats.runningJobs / queueStats.totalJobs) * 100} 
                      format={() => `${queueStats.runningJobs} / ${queueStats.totalJobs}`}
                    />
                  </div>
                  <div>
                    <Text>等待队列: </Text>
                    <Progress 
                      percent={(queueStats.pendingJobs / queueStats.totalJobs) * 100} 
                      format={() => `${queueStats.pendingJobs} / ${queueStats.totalJobs}`}
                      strokeColor="#faad14"
                    />
                  </div>
                  <div>
                    <Text>完成率: </Text>
                    <Progress 
                      percent={(queueStats.completedJobs / queueStats.totalJobs) * 100} 
                      format={() => `${queueStats.completedJobs} / ${queueStats.totalJobs}`}
                      strokeColor="#52c41a"
                    />
                  </div>
                </Space>
              </Col>
            </Row>
          </Card>
        </TabPane>

        <TabPane tab="调度管理" key="scheduler">
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={16}>
              <Card title="调度策略配置" extra={<ScheduleOutlined />}>
                <Form layout="vertical">
                  <Row gutter={16}>
                    <Col span={8}>
                      <Form.Item label="最大并发作业数">
                        <InputNumber min={1} max={20} defaultValue={5} />
                      </Form.Item>
                    </Col>
                    <Col span={8}>
                      <Form.Item label="队列优先级策略">
                        <Select defaultValue="priority_first">
                          <Option value="priority_first">优先级优先</Option>
                          <Option value="fifo">先进先出</Option>
                          <Option value="shortest_first">最短作业优先</Option>
                          <Option value="round_robin">轮询调度</Option>
                        </Select>
                      </Form.Item>
                    </Col>
                    <Col span={8}>
                      <Form.Item label="资源预留比例">
                        <InputNumber 
                          min={10} 
                          max={100} 
                          defaultValue={80} 
                          formatter={value => `${value}%`}
                          parser={value => value!.replace('%', '')}
                        />
                      </Form.Item>
                    </Col>
                  </Row>
                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item label="错误重试策略">
                        <Space direction="vertical" style={{ width: '100%' }}>
                          <Switch defaultChecked /> 启用自动重试
                          <InputNumber addonBefore="最大重试次数" min={1} max={10} defaultValue={3} />
                          <InputNumber addonBefore="重试间隔(秒)" min={1} max={3600} defaultValue={60} />
                        </Space>
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item label="负载均衡">
                        <Space direction="vertical" style={{ width: '100%' }}>
                          <Switch defaultChecked /> 启用动态负载均衡
                          <InputNumber addonBefore="负载阈值%" min={50} max={100} defaultValue={80} />
                          <Switch /> 启用预测性调度
                        </Space>
                      </Form.Item>
                    </Col>
                  </Row>
                </Form>
              </Card>
            </Col>
            <Col xs={24} lg={8}>
              <Card title="调度统计" size="small" style={{ marginBottom: '16px' }}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Text>今日调度次数:</Text>
                    <Text strong>1,247</Text>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Text>平均等待时间:</Text>
                    <Text strong>2.3分钟</Text>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Text>调度成功率:</Text>
                    <Text strong style={{ color: '#52c41a' }}>96.8%</Text>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Text>资源利用率:</Text>
                    <Text strong>78.5%</Text>
                  </div>
                </Space>
              </Card>
              <Card title="活动时间线" size="small">
                <Timeline size="small">
                  <Timeline.Item color="green">
                    <Text style={{ fontSize: '12px' }}>10:32 - 作业 batch_001 开始执行</Text>
                  </Timeline.Item>
                  <Timeline.Item color="blue">
                    <Text style={{ fontSize: '12px' }}>10:30 - 系统负载均衡调整</Text>
                  </Timeline.Item>
                  <Timeline.Item color="orange">
                    <Text style={{ fontSize: '12px' }}>10:28 - 队列达到高负载状态</Text>
                  </Timeline.Item>
                  <Timeline.Item color="green">
                    <Text style={{ fontSize: '12px' }}>10:25 - 作业 batch_003 完成</Text>
                  </Timeline.Item>
                </Timeline>
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>

      {/* 创建作业对话框 */}
      <Modal
        title="创建批量推理作业"
        visible={createJobVisible}
        onCancel={() => setCreateJobVisible(false)}
        onOk={() => form.submit()}
        width={800}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleCreateJob}
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item 
                label="作业名称" 
                name="name" 
                rules={[{ required: true, message: '请输入作业名称' }]}
              >
                <Input placeholder="输入批处理作业名称" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="优先级" name="priority" initialValue="medium">
                <Select>
                  <Option value="high">高优先级</Option>
                  <Option value="medium">中优先级</Option>
                  <Option value="low">低优先级</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item label="推理策略" name="strategy" initialValue="auto">
                <Select>
                  <Option value="auto">自动选择</Option>
                  <Option value="rule_only">规则推理</Option>
                  <Option value="embedding_only">嵌入推理</Option>
                  <Option value="path_only">路径推理</Option>
                  <Option value="ensemble">集成策略</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="输出格式" name="outputFormat" initialValue="json">
                <Select>
                  <Option value="json">JSON格式</Option>
                  <Option value="csv">CSV格式</Option>
                  <Option value="xml">XML格式</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
          <Form.Item label="查询数据上传">
            <Dragger {...uploadProps}>
              <p className="ant-upload-drag-icon">
                <CloudUploadOutlined />
              </p>
              <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
              <p className="ant-upload-hint">
                支持 JSON、CSV、TXT 格式，单个文件不超过100MB
              </p>
            </Dragger>
          </Form.Item>
        </Form>
      </Modal>

      {/* 作业详情抽屉 */}
      <Drawer
        title="作业详情"
        width={600}
        visible={jobDetailVisible}
        onClose={() => setJobDetailVisible(false)}
      >
        {selectedJob && (
          <Space direction="vertical" style={{ width: '100%' }} size="large">
            <Card title="基本信息" size="small">
              <Row gutter={[16, 8]}>
                <Col span={12}><Text type="secondary">作业ID:</Text></Col>
                <Col span={12}><Text code>{selectedJob.id}</Text></Col>
                <Col span={12}><Text type="secondary">作业名称:</Text></Col>
                <Col span={12}><Text>{selectedJob.name}</Text></Col>
                <Col span={12}><Text type="secondary">状态:</Text></Col>
                <Col span={12}><Badge status={getStatusColor(selectedJob.status)} text={getStatusText(selectedJob.status)} /></Col>
                <Col span={12}><Text type="secondary">优先级:</Text></Col>
                <Col span={12}><Tag color={getPriorityColor(selectedJob.priority)}>{selectedJob.priority}</Tag></Col>
              </Row>
            </Card>
            
            <Card title="执行详情" size="small">
              <Row gutter={[16, 8]}>
                <Col span={12}><Text type="secondary">总查询数:</Text></Col>
                <Col span={12}><Text>{selectedJob.totalQueries.toLocaleString()}</Text></Col>
                <Col span={12}><Text type="secondary">已处理:</Text></Col>
                <Col span={12}><Text>{selectedJob.processedQueries.toLocaleString()}</Text></Col>
                <Col span={12}><Text type="secondary">成功数:</Text></Col>
                <Col span={12}><Text style={{ color: '#52c41a' }}>{selectedJob.successQueries.toLocaleString()}</Text></Col>
                <Col span={12}><Text type="secondary">失败数:</Text></Col>
                <Col span={12}><Text style={{ color: '#ff4d4f' }}>{selectedJob.failedQueries.toLocaleString()}</Text></Col>
              </Row>
              <Divider />
              <Progress 
                percent={selectedJob.progress}
                status={selectedJob.status === 'failed' ? 'exception' : 
                       selectedJob.status === 'completed' ? 'success' : 'active'}
                strokeWidth={8}
              />
            </Card>
          </Space>
        )}
      </Drawer>

      {/* 队列配置对话框 */}
      <Modal
        title="队列配置"
        visible={configVisible}
        onCancel={() => setConfigVisible(false)}
        width={600}
        footer={[
          <Button key="cancel" onClick={() => setConfigVisible(false)}>
            取消
          </Button>,
          <Button key="save" type="primary">
            保存配置
          </Button>
        ]}
      >
        <Alert 
          message="配置修改将在下次调度时生效" 
          type="info" 
          showIcon 
          style={{ marginBottom: '16px' }}
        />
        {/* 配置表单内容 */}
      </Modal>
    </div>
  )
}

export default KGReasoningBatchPage