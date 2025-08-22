import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Table,
  Button,
  Space,
  Tag,
  Progress,
  Statistic,
  Modal,
  Form,
  Input,
  Select,
  message,
  Tabs,
  Typography
} from 'antd'
import {
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  ReloadOutlined,
  PlusOutlined,
  DeleteOutlined,
  EditOutlined,
  CloudServerOutlined,
  ThunderboltOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined
} from '@ant-design/icons'

const { Title } = Typography
const { TabPane } = Tabs
const { Option } = Select

interface BatchJob {
  id: string
  name: string
  description: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'paused'
  created_at: string
  started_at?: string
  completed_at?: string
  progress: number
  total_tasks: number
  completed_tasks: number
  failed_tasks: number
  duration: number
  priority: 'low' | 'medium' | 'high' | 'critical'
  resource_usage: {
    cpu: number
    memory: number
    storage: number
  }
}

const BatchJobsPageFixed: React.FC = () => {
  const [jobs, setJobs] = useState<BatchJob[]>([])
  const [loading, setLoading] = useState(false)
  const [createModalVisible, setCreateModalVisible] = useState(false)
  const [form] = Form.useForm()

  useEffect(() => {
    loadJobs()
  }, [])

  const loadJobs = async () => {
    setLoading(true)
    try {
      // 模拟数据
      const mockJobs: BatchJob[] = [
        {
          id: 'job-001',
          name: 'AI模型训练任务',
          description: '大规模语言模型训练作业',
          status: 'running',
          created_at: '2024-01-15T10:00:00Z',
          started_at: '2024-01-15T10:05:00Z',
          progress: 75,
          total_tasks: 1000,
          completed_tasks: 750,
          failed_tasks: 0,
          duration: 14400,
          priority: 'high',
          resource_usage: {
            cpu: 85,
            memory: 78,
            storage: 65
          }
        },
        {
          id: 'job-002',
          name: '数据预处理流水线',
          description: '大规模数据清洗和特征工程',
          status: 'completed',
          created_at: '2024-01-14T08:00:00Z',
          started_at: '2024-01-14T08:05:00Z',
          completed_at: '2024-01-14T12:30:00Z',
          progress: 100,
          total_tasks: 500,
          completed_tasks: 500,
          failed_tasks: 0,
          duration: 16200,
          priority: 'medium',
          resource_usage: {
            cpu: 0,
            memory: 0,
            storage: 0
          }
        }
      ]
      setJobs(mockJobs)
    } catch (error) {
      console.error('加载批处理作业失败:', error)
      message.error('加载批处理作业失败')
    } finally {
      setLoading(false)
    }
  }

  const getStatusColor = (status: string) => {
    const colors = {
      pending: 'default',
      running: 'processing',
      completed: 'success',
      failed: 'error',
      paused: 'warning'
    }
    return colors[status as keyof typeof colors] || 'default'
  }

  const getPriorityColor = (priority: string) => {
    const colors = {
      low: 'green',
      medium: 'blue',
      high: 'orange',
      critical: 'red'
    }
    return colors[priority as keyof typeof colors] || 'default'
  }

  const columns = [
    {
      title: '作业名称',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: BatchJob) => (
        <div>
          <div style={{ fontWeight: 'bold' }}>{name}</div>
          <div style={{ fontSize: '12px', color: '#666' }}>{record.description}</div>
        </div>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={getStatusColor(status)}>
          {status.toUpperCase()}
        </Tag>
      )
    },
    {
      title: '优先级',
      dataIndex: 'priority',
      key: 'priority',
      render: (priority: string) => (
        <Tag color={getPriorityColor(priority)}>
          {priority.toUpperCase()}
        </Tag>
      )
    },
    {
      title: '进度',
      key: 'progress',
      render: (_: any, record: BatchJob) => (
        <div style={{ width: '150px' }}>
          <Progress
            percent={record.progress}
            status={record.status === 'failed' ? 'exception' : undefined}
            format={() => `${record.completed_tasks}/${record.total_tasks}`}
          />
        </div>
      )
    },
    {
      title: '资源使用',
      key: 'resources',
      render: (_: any, record: BatchJob) => (
        <div>
          <div>CPU: {record.resource_usage.cpu}%</div>
          <div>内存: {record.resource_usage.memory}%</div>
        </div>
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (_: any, record: BatchJob) => (
        <Space>
          {record.status === 'running' ? (
            <Button icon={<PauseCircleOutlined />} />
          ) : record.status === 'paused' ? (
            <Button icon={<PlayCircleOutlined />} type="primary" />
          ) : (
            <Button icon={<PlayCircleOutlined />} />
          )}
          <Button icon={<StopOutlined />} danger />
          <Button icon={<EditOutlined />} />
          <Button icon={<DeleteOutlined />} danger />
        </Space>
      )
    }
  ]

  return (
    <div className="p-6">
      <div className="mb-6">
        <div className="flex justify-between items-center mb-4">
          <Title level={2}>批处理作业 (大规模并行)</Title>
          <Space>
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={() => setCreateModalVisible(true)}
            >
              创建作业
            </Button>
            <Button
              icon={<ReloadOutlined />}
              onClick={loadJobs}
              loading={loading}
            >
              刷新
            </Button>
          </Space>
        </div>

        <Row gutter={16} className="mb-6">
          <Col span={6}>
            <Card>
              <Statistic
                title="总作业数"
                value={jobs.length}
                prefix={<CloudServerOutlined />}
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="运行中"
                value={jobs.filter(j => j.status === 'running').length}
                prefix={<ThunderboltOutlined />}
                valueStyle={{ color: '#52c41a' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="已完成"
                value={jobs.filter(j => j.status === 'completed').length}
                prefix={<CheckCircleOutlined />}
                valueStyle={{ color: '#52c41a' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="失败"
                value={jobs.filter(j => j.status === 'failed').length}
                prefix={<ExclamationCircleOutlined />}
                valueStyle={{ color: '#ff4d4f' }}
              />
            </Card>
          </Col>
        </Row>
      </div>

      <Tabs defaultActiveKey="jobs">
        <TabPane tab="作业列表" key="jobs">
          <Card title="批处理作业管理">
            <Table
              columns={columns}
              dataSource={jobs}
              rowKey="id"
              loading={loading}
              pagination={{ pageSize: 10 }}
            />
          </Card>
        </TabPane>

        <TabPane tab="系统监控" key="monitoring">
          <Card title="系统资源监控">
            <Row gutter={16}>
              <Col span={8}>
                <Card>
                  <Statistic
                    title="CPU使用率"
                    value={68}
                    suffix="%"
                    valueStyle={{ color: '#faad14' }}
                  />
                </Card>
              </Col>
              <Col span={8}>
                <Card>
                  <Statistic
                    title="内存使用率"
                    value={72}
                    suffix="%"
                    valueStyle={{ color: '#52c41a' }}
                  />
                </Card>
              </Col>
              <Col span={8}>
                <Card>
                  <Statistic
                    title="存储使用率"
                    value={45}
                    suffix="%"
                    valueStyle={{ color: '#1890ff' }}
                  />
                </Card>
              </Col>
            </Row>
          </Card>
        </TabPane>
      </Tabs>

      <Modal
        title="创建批处理作业"
        open={createModalVisible}
        onCancel={() => setCreateModalVisible(false)}
        footer={null}
      >
        <Form form={form} layout="vertical">
          <Form.Item
            name="name"
            label="作业名称"
            rules={[{ required: true, message: '请输入作业名称' }]}
          >
            <Input placeholder="输入作业名称" />
          </Form.Item>
          <Form.Item
            name="description"
            label="描述"
            rules={[{ required: true, message: '请输入描述' }]}
          >
            <Input.TextArea placeholder="输入作业描述" rows={3} />
          </Form.Item>
          <Form.Item
            name="priority"
            label="优先级"
            rules={[{ required: true, message: '请选择优先级' }]}
          >
            <Select placeholder="选择优先级">
              <Option value="low">低</Option>
              <Option value="medium">中</Option>
              <Option value="high">高</Option>
              <Option value="critical">紧急</Option>
            </Select>
          </Form.Item>
          <Form.Item className="mb-0">
            <Space>
              <Button type="primary" htmlType="submit">
                创建
              </Button>
              <Button onClick={() => setCreateModalVisible(false)}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default BatchJobsPageFixed