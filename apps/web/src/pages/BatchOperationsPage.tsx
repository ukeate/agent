import React, { useState, useEffect } from 'react'
import {
import { logger } from '../utils/logger'
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
  Typography,
  Alert,
  Tabs,
  message
} from 'antd'
import {
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  ReloadOutlined,
  PlusOutlined,
  DeleteOutlined,
  CloudServerOutlined,
  ThunderboltOutlined,
  CheckCircleOutlined
} from '@ant-design/icons'
import { batchService, BatchJob, BatchStatsSummary, BatchJobCreate } from '../services/batchService'

const { Title } = Typography
const { Option } = Select

const BatchOperationsPage: React.FC = () => {
  const [jobs, setJobs] = useState<BatchJob[]>([])
  const [stats, setStats] = useState<BatchStatsSummary | null>(null)
  const [loading, setLoading] = useState(false)
  const [isModalVisible, setIsModalVisible] = useState(false)
  const [form] = Form.useForm()
  const safeStats = stats || {
    total_jobs: 0,
    active_jobs: 0,
    completed_jobs: 0,
    failed_jobs: 0
  }
  const pendingJobs = Math.max(0, safeStats.total_jobs - safeStats.active_jobs - safeStats.completed_jobs - safeStats.failed_jobs)

  // 加载数据
  const loadData = async () => {
    try {
      setLoading(true)
      const [statsData, jobsData] = await Promise.all([
        batchService.getStatsSummary(),
        batchService.getJobs()
      ])
      setStats(statsData)
      setJobs(jobsData)
    } catch (error) {
      logger.error('加载批量操作数据失败:', error)
      message.error('加载批量操作数据失败')
      setStats(null)
      setJobs([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
  }, [])

  // 取消任务
  const handleCancelJob = async (jobId: string) => {
    try {
      await batchService.cancelJob(jobId)
      message.success('任务已取消')
      loadData()
    } catch (error) {
      message.error('取消任务失败')
    }
  }

  // 重启任务
  const handleRestartJob = async (jobId: string) => {
    try {
      await batchService.restartJob(jobId)
      message.success('任务已重启')
      loadData()
    } catch (error) {
      message.error('重启任务失败')
    }
  }

  // 创建新任务
  const handleCreateJob = async (values: any) => {
    try {
      const jobData: BatchJobCreate = {
        name: values.name,
        task_type: values.task_type,
        parameters: values.parameters || {},
        priority: values.priority || 1
      }
      await batchService.createJob(jobData)
      message.success('任务创建成功')
      setIsModalVisible(false)
      form.resetFields()
      loadData()
    } catch (error) {
      message.error('创建任务失败')
    }
  }

  const statusColors = {
    pending: 'blue',
    running: 'orange',
    completed: 'green',
    failed: 'red'
  }

  const columns = [
    {
      title: '任务名称',
      dataIndex: 'name',
      key: 'name'
    },
    {
      title: 'ID',
      dataIndex: 'job_id',
      key: 'job_id'
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={statusColors[status as keyof typeof statusColors]}>
          {status.toUpperCase()}
        </Tag>
      )
    },
    {
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      render: (progress: number) => <Progress percent={progress} size="small" />
    },
    {
      title: '处理进度',
      key: 'items',
      render: (record: any) => (
        `${record.processed_items || 0}/${record.total_items || 0}`
      )
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (time: string) => new Date(time).toLocaleString()
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: any) => (
        <Space>
          {record.status === 'running' && (
            <Button size="small" icon={<PauseCircleOutlined />} onClick={() => handleCancelJob(record.job_id)}>取消</Button>
          )}
          {record.status === 'paused' && (
            <Button size="small" icon={<PlayCircleOutlined />}>恢复</Button>
          )}
          <Button size="small" icon={<StopOutlined />} danger>停止</Button>
          <Button size="small" icon={<DeleteOutlined />}>删除</Button>
        </Space>
      )
    }
  ]

  const handleCreateOperation = () => {
    setIsModalVisible(true)
  }

  const handleModalOk = () => {
    form.validateFields().then(handleCreateJob)
  }

  const handleModalCancel = () => {
    setIsModalVisible(false)
    form.resetFields()
  }

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <CloudServerOutlined /> 批处理操作管理
      </Title>

      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="运行中任务"
              value={safeStats.active_jobs}
              prefix={<PlayCircleOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="已完成任务"
              value={safeStats.completed_jobs}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="等待中任务"
              value={pendingJobs}
              prefix={<ThunderboltOutlined />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="总任务数"
              value={safeStats.total_jobs}
              prefix={<CloudServerOutlined />}
            />
          </Card>
        </Col>
      </Row>

      <Alert
        message="批处理操作管理"
        description="管理系统中的各种批处理任务，包括数据同步、模型训练、文件处理等操作。"
        type="info"
        showIcon
        style={{ marginBottom: '24px' }}
      />

      <Tabs
        items={[
          {
            key: '1',
            label: '任务列表',
            children: (
              <Card>
                <Space style={{ marginBottom: '16px' }}>
                  <Button 
                    type="primary" 
                    icon={<PlusOutlined />}
                    onClick={handleCreateOperation}
                  >
                    创建任务
                  </Button>
                  <Button icon={<ReloadOutlined />}>刷新</Button>
                </Space>
                <Table
                  columns={columns}
                  dataSource={jobs}
                  rowKey="job_id"
                  loading={loading}
                />
              </Card>
            )
          },
          {
            key: '2',
            label: '系统监控',
            children: (
              <Card title="系统资源使用情况">
                <Row gutter={[16, 16]}>
                  <Col span={12}>
                    <Card size="small" title="CPU 使用率">
                      <Progress percent={45} status="active" />
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card size="small" title="内存使用率">
                      <Progress percent={67} status="active" />
                    </Card>
                  </Col>
                </Row>
              </Card>
            )
          }
        ]}
      />

      <Modal
        title="创建批处理任务"
        open={isModalVisible}
        onOk={handleModalOk}
        onCancel={handleModalCancel}
        okText="创建"
        cancelText="取消"
      >
        <Form form={form} layout="vertical">
          <Form.Item
            name="name"
            label="任务名称"
            rules={[{ required: true, message: '请输入任务名称' }]}
          >
            <Input placeholder="请输入任务名称" />
          </Form.Item>
          <Form.Item
            name="task_type"
            label="任务类型"
            rules={[{ required: true, message: '请选择任务类型' }]}
          >
            <Select placeholder="请选择任务类型">
              <Option value="data_sync">数据同步</Option>
              <Option value="model_training">模型训练</Option>
              <Option value="file_processing">文件处理</Option>
              <Option value="data_export">数据导出</Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default BatchOperationsPage
