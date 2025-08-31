import React, { useState, useEffect } from 'react'
import {
  Card,
  Table,
  Button,
  Modal,
  Form,
  Input,
  Select,
  Steps,
  Progress,
  Tag,
  Space,
  Popconfirm,
  message,
  Drawer,
  Descriptions,
  Timeline,
  Row,
  Col,
  Statistic,
  Typography,
  Alert,
  Spin,
  Tooltip
} from 'antd'
import {
  PlusOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  EyeOutlined,
  ReloadOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  RocketOutlined,
  SettingOutlined,
  ApiOutlined
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'

const { Title, Text } = Typography
const { Option } = Select
const { TextArea } = Input
const { Step } = Steps

interface Workflow {
  workflow_id: string
  name: string
  description: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  progress: number
  current_step?: string
  steps: WorkflowStep[]
  created_at: string
  started_at?: string
  completed_at?: string
  error_message?: string
  metadata: Record<string, any>
}

interface WorkflowStep {
  step_id: string
  name: string
  component_id: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  started_at?: string
  completed_at?: string
  error_message?: string
  output?: any
}

interface WorkflowTemplate {
  id: string
  name: string
  description: string
  steps: Array<{
    name: string
    component_type: string
    required_params: string[]
  }>
}

const WorkflowOrchestrationPage: React.FC = () => {
  const [workflows, setWorkflows] = useState<Workflow[]>([])
  const [templates, setTemplates] = useState<WorkflowTemplate[]>([])
  const [loading, setLoading] = useState(false)
  const [modalVisible, setModalVisible] = useState(false)
  const [detailDrawerVisible, setDetailDrawerVisible] = useState(false)
  const [selectedWorkflow, setSelectedWorkflow] = useState<Workflow | null>(null)
  const [form] = Form.useForm()

  useEffect(() => {
    fetchWorkflows()
    fetchTemplates()
    const interval = setInterval(fetchWorkflows, 5000)
    return () => clearInterval(interval)
  }, [])

  const fetchWorkflows = async () => {
    setLoading(true)
    try {
      const response = await fetch('/api/v1/platform-integration/workflows')
      if (response.ok) {
        const data = await response.json()
        setWorkflows(data.workflows || [])
      } else {
        message.error('获取工作流列表失败')
      }
    } catch (error) {
      message.error('获取工作流列表失败')
    } finally {
      setLoading(false)
    }
  }

  const fetchTemplates = async () => {
    try {
      const response = await fetch('/api/v1/platform-integration/workflow-templates')
      if (response.ok) {
        const data = await response.json()
        setTemplates(data.templates || [])
      }
    } catch (error) {
      console.error('获取模板失败:', error)
    }
  }

  const handleCreateWorkflow = async (values: any) => {
    try {
      const payload = {
        name: values.name,
        description: values.description,
        workflow_type: values.workflow_type,
        parameters: values.parameters ? JSON.parse(values.parameters) : {},
        metadata: values.metadata ? JSON.parse(values.metadata) : {}
      }

      const response = await fetch('/api/v1/platform-integration/workflows/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })

      if (response.ok) {
        message.success('工作流创建成功')
        setModalVisible(false)
        form.resetFields()
        fetchWorkflows()
      } else {
        const error = await response.json()
        message.error(`创建失败: ${error.detail}`)
      }
    } catch (error) {
      message.error('创建失败')
    }
  }

  const handleExecuteWorkflow = async (workflowId: string) => {
    try {
      const response = await fetch(`/api/v1/platform-integration/workflows/${workflowId}/execute`, {
        method: 'POST'
      })

      if (response.ok) {
        message.success('工作流已开始执行')
        fetchWorkflows()
      } else {
        message.error('执行失败')
      }
    } catch (error) {
      message.error('执行失败')
    }
  }

  const handleStopWorkflow = async (workflowId: string) => {
    try {
      const response = await fetch(`/api/v1/platform-integration/workflows/${workflowId}/stop`, {
        method: 'POST'
      })

      if (response.ok) {
        message.success('工作流已停止')
        fetchWorkflows()
      } else {
        message.error('停止失败')
      }
    } catch (error) {
      message.error('停止失败')
    }
  }

  const showWorkflowDetail = (workflow: Workflow) => {
    setSelectedWorkflow(workflow)
    setDetailDrawerVisible(true)
  }

  const getStatusConfig = (status: string) => {
    const configs = {
      pending: { color: 'default', text: '待执行', icon: <ClockCircleOutlined /> },
      running: { color: 'processing', text: '运行中', icon: <RocketOutlined spin /> },
      completed: { color: 'success', text: '已完成', icon: <CheckCircleOutlined /> },
      failed: { color: 'error', text: '失败', icon: <ExclamationCircleOutlined /> },
      cancelled: { color: 'default', text: '已取消', icon: <StopOutlined /> }
    }
    return configs[status] || configs.pending
  }

  const getStepStatus = (step: WorkflowStep) => {
    switch (step.status) {
      case 'completed':
        return 'finish'
      case 'running':
        return 'process'
      case 'failed':
        return 'error'
      default:
        return 'wait'
    }
  }

  const columns: ColumnsType<Workflow> = [
    {
      title: '工作流名称',
      dataIndex: 'name',
      key: 'name',
      render: (text, record) => (
        <Space>
          <RocketOutlined />
          <div>
            <div>{text}</div>
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {record.workflow_id}
            </Text>
          </div>
        </Space>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => {
        const config = getStatusConfig(status)
        return <Tag color={config.color} icon={config.icon}>{config.text}</Tag>
      }
    },
    {
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      render: (progress, record) => (
        <div>
          <Progress 
            percent={progress} 
            size="small"
            status={record.status === 'failed' ? 'exception' : 
                   record.status === 'completed' ? 'success' : 'active'}
          />
          {record.current_step && (
            <Text type="secondary" style={{ fontSize: '12px' }}>
              当前: {record.current_step}
            </Text>
          )}
        </div>
      )
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (time) => new Date(time).toLocaleString()
    },
    {
      title: '执行时间',
      dataIndex: 'started_at',
      key: 'started_at',
      render: (time) => time ? new Date(time).toLocaleString() : '-'
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Tooltip title="查看详情">
            <Button
              type="text"
              icon={<EyeOutlined />}
              onClick={() => showWorkflowDetail(record)}
            />
          </Tooltip>
          {record.status === 'pending' && (
            <Tooltip title="执行">
              <Button
                type="text"
                icon={<PlayCircleOutlined />}
                onClick={() => handleExecuteWorkflow(record.workflow_id)}
              />
            </Tooltip>
          )}
          {record.status === 'running' && (
            <Tooltip title="停止">
              <Popconfirm
                title="确认停止此工作流？"
                onConfirm={() => handleStopWorkflow(record.workflow_id)}
              >
                <Button type="text" danger icon={<StopOutlined />} />
              </Popconfirm>
            </Tooltip>
          )}
        </Space>
      )
    }
  ]

  const summary = workflows.reduce((acc, workflow) => {
    acc.total++
    acc[workflow.status] = (acc[workflow.status] || 0) + 1
    return acc
  }, { total: 0, pending: 0, running: 0, completed: 0, failed: 0, cancelled: 0 } as any)

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: 24, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Title level={2}>工作流编排</Title>
        <Space>
          <Button icon={<ReloadOutlined />} onClick={fetchWorkflows}>
            刷新
          </Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setModalVisible(true)}>
            创建工作流
          </Button>
        </Space>
      </div>

      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={4}>
          <Card>
            <Statistic
              title="总工作流"
              value={summary.total}
              prefix={<RocketOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card>
            <Statistic
              title="待执行"
              value={summary.pending}
              prefix={<ClockCircleOutlined />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card>
            <Statistic
              title="运行中"
              value={summary.running}
              prefix={<RocketOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card>
            <Statistic
              title="已完成"
              value={summary.completed}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card>
            <Statistic
              title="失败"
              value={summary.failed}
              prefix={<ExclamationCircleOutlined />}
              valueStyle={{ color: '#f5222d' }}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card>
            <Statistic
              title="已取消"
              value={summary.cancelled}
              prefix={<StopOutlined />}
              valueStyle={{ color: '#8c8c8c' }}
            />
          </Card>
        </Col>
      </Row>

      <Card>
        <Table
          columns={columns}
          dataSource={workflows}
          rowKey="workflow_id"
          loading={loading}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total) => `共 ${total} 个工作流`
          }}
        />
      </Card>

      <Modal
        title="创建工作流"
        open={modalVisible}
        onCancel={() => {
          setModalVisible(false)
          form.resetFields()
        }}
        footer={null}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleCreateWorkflow}
        >
          <Form.Item
            name="name"
            label="工作流名称"
            rules={[{ required: true, message: '请输入工作流名称' }]}
          >
            <Input placeholder="工作流名称" />
          </Form.Item>

          <Form.Item
            name="description"
            label="描述"
          >
            <TextArea rows={2} placeholder="工作流描述" />
          </Form.Item>

          <Form.Item
            name="workflow_type"
            label="工作流类型"
            rules={[{ required: true, message: '请选择工作流类型' }]}
          >
            <Select placeholder="选择工作流类型">
              <Option value="data_processing">数据处理流程</Option>
              <Option value="model_training">模型训练流程</Option>
              <Option value="inference_pipeline">推理管道</Option>
              <Option value="batch_processing">批处理流程</Option>
              <Option value="custom">自定义流程</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="parameters"
            label="执行参数 (JSON格式)"
          >
            <TextArea
              rows={4}
              placeholder='{"input_data": "path/to/data", "batch_size": 100}'
            />
          </Form.Item>

          <Form.Item
            name="metadata"
            label="元数据 (JSON格式)"
          >
            <TextArea
              rows={3}
              placeholder='{"owner": "user", "priority": "high"}'
            />
          </Form.Item>

          <Form.Item style={{ marginBottom: 0 }}>
            <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
              <Button onClick={() => {
                setModalVisible(false)
                form.resetFields()
              }}>
                取消
              </Button>
              <Button type="primary" htmlType="submit">
                创建
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      <Drawer
        title="工作流详情"
        placement="right"
        onClose={() => setDetailDrawerVisible(false)}
        open={detailDrawerVisible}
        width={800}
      >
        {selectedWorkflow && (
          <div>
            <Descriptions title="基本信息" bordered column={1} size="small">
              <Descriptions.Item label="工作流ID">
                {selectedWorkflow.workflow_id}
              </Descriptions.Item>
              <Descriptions.Item label="名称">
                {selectedWorkflow.name}
              </Descriptions.Item>
              <Descriptions.Item label="描述">
                {selectedWorkflow.description}
              </Descriptions.Item>
              <Descriptions.Item label="状态">
                {(() => {
                  const config = getStatusConfig(selectedWorkflow.status)
                  return <Tag color={config.color} icon={config.icon}>{config.text}</Tag>
                })()}
              </Descriptions.Item>
              <Descriptions.Item label="进度">
                <Progress percent={selectedWorkflow.progress} />
              </Descriptions.Item>
              <Descriptions.Item label="创建时间">
                {new Date(selectedWorkflow.created_at).toLocaleString()}
              </Descriptions.Item>
              {selectedWorkflow.started_at && (
                <Descriptions.Item label="开始时间">
                  {new Date(selectedWorkflow.started_at).toLocaleString()}
                </Descriptions.Item>
              )}
              {selectedWorkflow.completed_at && (
                <Descriptions.Item label="完成时间">
                  {new Date(selectedWorkflow.completed_at).toLocaleString()}
                </Descriptions.Item>
              )}
            </Descriptions>

            {selectedWorkflow.error_message && (
              <Alert
                type="error"
                message="执行错误"
                description={selectedWorkflow.error_message}
                style={{ margin: '16px 0' }}
              />
            )}

            <div style={{ marginTop: 24 }}>
              <Title level={4}>执行步骤</Title>
              <Steps
                direction="vertical"
                current={selectedWorkflow.steps.findIndex(step => step.status === 'running')}
                size="small"
              >
                {selectedWorkflow.steps.map((step, index) => (
                  <Step
                    key={step.step_id}
                    title={step.name}
                    status={getStepStatus(step)}
                    description={
                      <div>
                        <div>组件: {step.component_id}</div>
                        {step.progress > 0 && (
                          <Progress percent={step.progress} size="small" />
                        )}
                        {step.started_at && (
                          <div>开始: {new Date(step.started_at).toLocaleString()}</div>
                        )}
                        {step.completed_at && (
                          <div>完成: {new Date(step.completed_at).toLocaleString()}</div>
                        )}
                        {step.error_message && (
                          <div style={{ color: '#f5222d' }}>错误: {step.error_message}</div>
                        )}
                      </div>
                    }
                  />
                ))}
              </Steps>
            </div>

            {selectedWorkflow.metadata && Object.keys(selectedWorkflow.metadata).length > 0 && (
              <div style={{ marginTop: 24 }}>
                <Title level={4}>元数据</Title>
                <pre style={{ background: '#f5f5f5', padding: 12, borderRadius: 4 }}>
                  {JSON.stringify(selectedWorkflow.metadata, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}
      </Drawer>
    </div>
  )
}

export default WorkflowOrchestrationPage