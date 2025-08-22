import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Button,
  Space,
  Table,
  Tag,
  Progress,
  Select,
  Input,
  Typography,
  Statistic,
  Modal,
  Form,
  Switch,
  InputNumber,
  Tabs,
  Timeline,
  Tooltip
} from 'antd'
import {
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  ReloadOutlined,
  PlusOutlined,
  SettingOutlined,
  DeleteOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ThunderboltOutlined
} from '@ant-design/icons'

const { Title, Text } = Typography
const { Option } = Select
const { TextArea } = Input
const { TabPane } = Tabs

interface BatchJob {
  id: string
  name: string
  type: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'paused'
  progress: number
  createdAt: string
  startedAt?: string
  completedAt?: string
  priority: 'low' | 'medium' | 'high'
  workerId?: string
  result?: any
  error?: string
}

interface Worker {
  id: string
  name: string
  status: 'idle' | 'busy' | 'offline'
  currentJob?: string
  totalJobs: number
  cpuUsage: number
  memoryUsage: number
  lastHeartbeat: string
}

const BatchJobsPage: React.FC = () => {
  const [jobs, setJobs] = useState<BatchJob[]>([
    {
      id: '1',
      name: 'AI模型训练作业',
      type: 'model_training',
      status: 'running',
      progress: 65,
      createdAt: '2024-01-16 10:00:00',
      startedAt: '2024-01-16 10:05:00',
      priority: 'high',
      workerId: 'worker-001'
    },
    {
      id: '2',
      name: '数据预处理作业',
      type: 'data_processing',
      status: 'completed',
      progress: 100,
      createdAt: '2024-01-16 09:30:00',
      startedAt: '2024-01-16 09:35:00',
      completedAt: '2024-01-16 10:15:00',
      priority: 'medium',
      workerId: 'worker-002'
    },
    {
      id: '3',
      name: '批量向量化作业',
      type: 'vectorization',
      status: 'pending',
      progress: 0,
      createdAt: '2024-01-16 11:00:00',
      priority: 'low'
    },
    {
      id: '4',
      name: '模型评估作业',
      type: 'evaluation',
      status: 'failed',
      progress: 45,
      createdAt: '2024-01-16 08:00:00',
      startedAt: '2024-01-16 08:05:00',
      priority: 'medium',
      workerId: 'worker-001',
      error: 'Out of memory error'
    }
  ])

  const [workers] = useState<Worker[]>([
    {
      id: 'worker-001',
      name: 'AI训练工作者',
      status: 'busy',
      currentJob: 'AI模型训练作业',
      totalJobs: 156,
      cpuUsage: 85,
      memoryUsage: 78,
      lastHeartbeat: '1分钟前'
    },
    {
      id: 'worker-002',
      name: '数据处理工作者',
      status: 'idle',
      totalJobs: 203,
      cpuUsage: 15,
      memoryUsage: 32,
      lastHeartbeat: '30秒前'
    },
    {
      id: 'worker-003',
      name: '向量化工作者',
      status: 'offline',
      totalJobs: 89,
      cpuUsage: 0,
      memoryUsage: 0,
      lastHeartbeat: '5分钟前'
    }
  ])

  const [showCreateModal, setShowCreateModal] = useState(false)
  const [showConfigModal, setShowConfigModal] = useState(false)
  const [selectedFilter, setSelectedFilter] = useState<string>('all')
  const [autoRefresh, setAutoRefresh] = useState(true)

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(() => {
        setJobs(prev => prev.map(job => 
          job.status === 'running' 
            ? { ...job, progress: Math.min(100, job.progress + Math.random() * 5) }
            : job
        ))
      }, 2000)
      return () => clearInterval(interval)
    }
  }, [autoRefresh])

  const getStatusColor = (status: string) => {
    const colors = {
      pending: 'default',
      running: 'processing',
      completed: 'success',
      failed: 'error',
      cancelled: 'warning',
      paused: 'warning'
    }
    return colors[status as keyof typeof colors] || 'default'
  }

  const getStatusIcon = (status: string) => {
    const icons = {
      pending: <ClockCircleOutlined />,
      running: <PlayCircleOutlined />,
      completed: <CheckCircleOutlined />,
      failed: <CloseCircleOutlined />,
      cancelled: <StopOutlined />,
      paused: <PauseCircleOutlined />
    }
    return icons[status as keyof typeof icons]
  }

  const getPriorityColor = (priority: string) => {
    const colors = { low: 'green', medium: 'orange', high: 'red' }
    return colors[priority as keyof typeof colors]
  }

  const getWorkerStatusColor = (status: string) => {
    const colors = { idle: 'green', busy: 'blue', offline: 'red' }
    return colors[status as keyof typeof colors]
  }

  const jobColumns = [
    {
      title: '作业名称',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: BatchJob) => (
        <div>
          <Text strong>{name}</Text>
          <br />
          <Text type="secondary" className="text-xs">ID: {record.id}</Text>
        </div>
      )
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => {
        const typeNames = {
          model_training: 'AI模型训练',
          data_processing: '数据处理',
          vectorization: '向量化',
          evaluation: '模型评估'
        }
        return typeNames[type as keyof typeof typeNames] || type
      }
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={getStatusColor(status)} icon={getStatusIcon(status)}>
          {status === 'pending' && '等待中'}
          {status === 'running' && '运行中'}
          {status === 'completed' && '已完成'}
          {status === 'failed' && '失败'}
          {status === 'cancelled' && '已取消'}
          {status === 'paused' && '已暂停'}
        </Tag>
      )
    },
    {
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      render: (progress: number, record: BatchJob) => (
        <div>
          <Progress 
            percent={progress} 
            size="small"
            status={record.status === 'failed' ? 'exception' : 'normal'}
          />
          <Text className="text-xs">{progress}%</Text>
        </div>
      )
    },
    {
      title: '优先级',
      dataIndex: 'priority',
      key: 'priority',
      render: (priority: string) => (
        <Tag color={getPriorityColor(priority)}>
          {priority === 'low' && '低'}
          {priority === 'medium' && '中'}
          {priority === 'high' && '高'}
        </Tag>
      )
    },
    {
      title: '工作者',
      dataIndex: 'workerId',
      key: 'workerId',
      render: (workerId?: string) => workerId || '-'
    },
    {
      title: '创建时间',
      dataIndex: 'createdAt',
      key: 'createdAt'
    },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: BatchJob) => (
        <Space>
          {record.status === 'running' && (
            <Tooltip title="暂停">
              <Button size="small" icon={<PauseCircleOutlined />} />
            </Tooltip>
          )}
          {record.status === 'paused' && (
            <Tooltip title="继续">
              <Button size="small" icon={<PlayCircleOutlined />} />
            </Tooltip>
          )}
          {(record.status === 'pending' || record.status === 'running') && (
            <Tooltip title="取消">
              <Button size="small" danger icon={<StopOutlined />} />
            </Tooltip>
          )}
          {record.status === 'failed' && (
            <Tooltip title="重试">
              <Button size="small" icon={<ReloadOutlined />} />
            </Tooltip>
          )}
          <Tooltip title="删除">
            <Button size="small" danger icon={<DeleteOutlined />} />
          </Tooltip>
        </Space>
      )
    }
  ]

  const workerColumns = [
    {
      title: '工作者',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: Worker) => (
        <div>
          <Text strong>{name}</Text>
          <br />
          <Text type="secondary" className="text-xs">ID: {record.id}</Text>
        </div>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={getWorkerStatusColor(status)}>
          {status === 'idle' && '空闲'}
          {status === 'busy' && '忙碌'}
          {status === 'offline' && '离线'}
        </Tag>
      )
    },
    {
      title: '当前作业',
      dataIndex: 'currentJob',
      key: 'currentJob',
      render: (job?: string) => job || '-'
    },
    {
      title: 'CPU使用率',
      dataIndex: 'cpuUsage',
      key: 'cpuUsage',
      render: (usage: number) => (
        <div>
          <Progress 
            percent={usage} 
            size="small"
            strokeColor={usage > 80 ? '#ff4d4f' : '#1890ff'}
          />
          <Text className="text-xs">{usage}%</Text>
        </div>
      )
    },
    {
      title: '内存使用率',
      dataIndex: 'memoryUsage',
      key: 'memoryUsage',
      render: (usage: number) => (
        <div>
          <Progress 
            percent={usage} 
            size="small"
            strokeColor={usage > 80 ? '#ff4d4f' : '#52c41a'}
          />
          <Text className="text-xs">{usage}%</Text>
        </div>
      )
    },
    {
      title: '总作业数',
      dataIndex: 'totalJobs',
      key: 'totalJobs'
    },
    {
      title: '最后心跳',
      dataIndex: 'lastHeartbeat',
      key: 'lastHeartbeat'
    }
  ]

  const filteredJobs = selectedFilter === 'all' 
    ? jobs 
    : jobs.filter(job => job.status === selectedFilter)

  const statsData = {
    total: jobs.length,
    running: jobs.filter(j => j.status === 'running').length,
    completed: jobs.filter(j => j.status === 'completed').length,
    failed: jobs.filter(j => j.status === 'failed').length,
    pending: jobs.filter(j => j.status === 'pending').length
  }

  return (
    <div className="p-6">
      <div className="mb-6">
        <div className="flex justify-between items-center mb-4">
          <Title level={2}>批处理作业管理</Title>
          <Space>
            <Button 
              type="primary" 
              icon={<PlusOutlined />}
              onClick={() => setShowCreateModal(true)}
            >
              创建作业
            </Button>
            <Button 
              icon={<SettingOutlined />}
              onClick={() => setShowConfigModal(true)}
            >
              配置管理
            </Button>
            <Button 
              icon={<ReloadOutlined />}
              onClick={() => setAutoRefresh(!autoRefresh)}
              type={autoRefresh ? 'primary' : 'default'}
            >
              {autoRefresh ? '停止' : '开始'}自动刷新
            </Button>
          </Space>
        </div>

        <Row gutter={16} className="mb-6">
          <Col span={6}>
            <Card>
              <Statistic
                title="总作业数"
                value={statsData.total}
                prefix={<ThunderboltOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="运行中"
                value={statsData.running}
                valueStyle={{ color: '#1890ff' }}
                prefix={<PlayCircleOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="已完成"
                value={statsData.completed}
                valueStyle={{ color: '#3f8600' }}
                prefix={<CheckCircleOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="失败"
                value={statsData.failed}
                valueStyle={{ color: '#cf1322' }}
                prefix={<CloseCircleOutlined />}
              />
            </Card>
          </Col>
        </Row>

        <div className="mb-4">
          <Space>
            <Text>状态筛选: </Text>
            <Select 
              value={selectedFilter} 
              onChange={setSelectedFilter}
              style={{ width: 120 }}
            >
              <Option value="all">全部</Option>
              <Option value="pending">等待中</Option>
              <Option value="running">运行中</Option>
              <Option value="completed">已完成</Option>
              <Option value="failed">失败</Option>
              <Option value="paused">已暂停</Option>
            </Select>
          </Space>
        </div>
      </div>

      <Tabs defaultActiveKey="jobs">
        <TabPane tab={`作业列表 (${filteredJobs.length})`} key="jobs">
          <Card>
            <Table
              columns={jobColumns}
              dataSource={filteredJobs}
              rowKey="id"
              pagination={{ pageSize: 10 }}
              size="small"
            />
          </Card>
        </TabPane>

        <TabPane tab={`工作者 (${workers.length})`} key="workers">
          <Card title="工作者状态监控">
            <Table
              columns={workerColumns}
              dataSource={workers}
              rowKey="id"
              pagination={false}
              size="small"
            />
          </Card>
        </TabPane>

        <TabPane tab="配置" key="config">
          <Row gutter={16}>
            <Col span={12}>
              <Card title="队列配置">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>
                    <Text strong>最大并发作业数: </Text>
                    <Text>10</Text>
                  </div>
                  <div>
                    <Text strong>队列大小限制: </Text>
                    <Text>1000</Text>
                  </div>
                  <div>
                    <Text strong>作业超时时间: </Text>
                    <Text>3600秒</Text>
                  </div>
                  <div>
                    <Text strong>重试次数: </Text>
                    <Text>3次</Text>
                  </div>
                  <div>
                    <Text strong>调度策略: </Text>
                    <Text>优先级 + FIFO</Text>
                  </div>
                </Space>
              </Card>
            </Col>
            <Col span={12}>
              <Card title="系统监控">
                <Timeline
                  items={[
                    {
                      color: 'green',
                      children: (
                        <div>
                          <Text strong>作业完成</Text>
                          <br />
                          <Text type="secondary">数据预处理作业 - 5分钟前</Text>
                        </div>
                      )
                    },
                    {
                      color: 'blue',
                      children: (
                        <div>
                          <Text strong>作业开始</Text>
                          <br />
                          <Text type="secondary">AI模型训练作业 - 1小时前</Text>
                        </div>
                      )
                    },
                    {
                      color: 'red',
                      children: (
                        <div>
                          <Text strong>作业失败</Text>
                          <br />
                          <Text type="secondary">模型评估作业 - 2小时前</Text>
                        </div>
                      )
                    }
                  ]}
                />
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>

      {/* 创建作业Modal */}
      <Modal
        title="创建新的批处理作业"
        open={showCreateModal}
        onCancel={() => setShowCreateModal(false)}
        footer={null}
        width={600}
      >
        <Form layout="vertical">
          <Form.Item label="作业名称" required>
            <Input placeholder="输入作业名称" />
          </Form.Item>
          <Form.Item label="作业类型" required>
            <Select placeholder="选择作业类型">
              <Option value="model_training">AI模型训练</Option>
              <Option value="data_processing">数据处理</Option>
              <Option value="vectorization">向量化</Option>
              <Option value="evaluation">模型评估</Option>
            </Select>
          </Form.Item>
          <Form.Item label="优先级">
            <Select defaultValue="medium">
              <Option value="low">低</Option>
              <Option value="medium">中</Option>
              <Option value="high">高</Option>
            </Select>
          </Form.Item>
          <Form.Item label="作业配置">
            <TextArea rows={4} placeholder="JSON格式的作业配置参数" />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary">创建作业</Button>
              <Button onClick={() => setShowCreateModal(false)}>取消</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* 配置管理Modal */}
      <Modal
        title="批处理系统配置"
        open={showConfigModal}
        onCancel={() => setShowConfigModal(false)}
        footer={null}
        width={600}
      >
        <Form layout="vertical">
          <Form.Item label="最大并发作业数">
            <InputNumber min={1} max={50} defaultValue={10} />
          </Form.Item>
          <Form.Item label="队列大小限制">
            <InputNumber min={100} max={10000} defaultValue={1000} />
          </Form.Item>
          <Form.Item label="作业超时时间(秒)">
            <InputNumber min={60} max={86400} defaultValue={3600} />
          </Form.Item>
          <Form.Item label="启用自动重试">
            <Switch defaultChecked />
          </Form.Item>
          <Form.Item label="最大重试次数">
            <InputNumber min={0} max={10} defaultValue={3} />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary">保存配置</Button>
              <Button onClick={() => setShowConfigModal(false)}>取消</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default BatchJobsPage