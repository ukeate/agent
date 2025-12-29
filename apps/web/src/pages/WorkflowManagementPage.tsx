import React, { useState, useEffect } from 'react'
import {
  Card, Button, Table, Tag, Space, Modal, Form, Input, Select, Steps,
  Row, Col, Statistic, Timeline, Alert, Badge, message, Drawer, List,
  Divider, Progress, Descriptions
} from 'antd'
import {
  BranchesOutlined, PlayCircleOutlined,
  CheckCircleOutlined, SyncOutlined,
  NodeIndexOutlined, ForkOutlined, ReloadOutlined, PlusOutlined,
  DeleteOutlined, EditOutlined
} from '@ant-design/icons'
import apiClient from '../services/apiClient'

const { Step } = Steps
const { TextArea } = Input

interface WorkflowStep {
  id: string
  name: string
  type: 'start' | 'action' | 'condition' | 'parallel' | 'end'
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped'
  config?: any
}

interface Workflow {
  id: string
  name: string
  status: 'draft' | 'active' | 'running' | 'completed' | 'failed'
  steps: WorkflowStep[]
  created_at: string
  updated_at: string
  executions: number
}

interface Execution {
  id: string
  workflow_id: string
  status: string
  started_at: string
  completed_at?: string
  current_step: number
  logs: string[]
}

const WorkflowManagementPage: React.FC = () => {
  const [workflows, setWorkflows] = useState<Workflow[]>([])
  const [executions, setExecutions] = useState<Execution[]>([])
  const [selectedWorkflow, setSelectedWorkflow] = useState<Workflow | null>(null)
  const [selectedExecution, setSelectedExecution] = useState<Execution | null>(null)
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [showExecutionDrawer, setShowExecutionDrawer] = useState(false)
  const [loading, setLoading] = useState(false)
  const [form] = Form.useForm()

  useEffect(() => {
    fetchWorkflows()
  }, [])

  // 获取工作流列表
  const fetchWorkflows = async () => {
    setLoading(true)
    try {
      const response = await apiClient.get('/workflows')
      const payload: any = response.data || {}
      const list = Array.isArray(payload) ? payload : (payload.workflows || [])
      const normalized = list.map((wf: any) => ({
        ...wf,
        status: wf.status || 'draft',
        steps: wf.steps || [],
        executions: wf.executions || 0,
        created_at: wf.created_at || new Date().toISOString(),
        updated_at: wf.updated_at || wf.created_at || new Date().toISOString()
      }))
      setWorkflows(normalized)
    } catch (error) {
      message.error('获取工作流列表失败')
      setWorkflows([])
    } finally {
      setLoading(false)
    }
  }

  // 启动工作流
  const startWorkflow = async (workflowId: string) => {
    try {
      const response = await apiClient.post(`/workflows-simple/${workflowId}/start`)
      const execution: Execution = {
        id: response.data.execution_id,
        workflow_id: workflowId,
        status: 'running',
        started_at: response.data.started_at,
        current_step: 0,
        logs: response.data.started_at ? [`启动时间: ${new Date(response.data.started_at).toLocaleString()}`] : []
      }
      setExecutions(prev => [execution, ...prev])
      setSelectedExecution(execution)
      setShowExecutionDrawer(true)
      message.success('工作流已启动')
      fetchWorkflows()
    } catch (error) {
      message.error('启动工作流失败')
    }
  }

  // 创建新工作流
  const handleCreateWorkflow = async (values: any) => {
    try {
      await apiClient.post('/workflows', values)
      message.success('工作流创建成功')
      setShowCreateModal(false)
      form.resetFields()
      fetchWorkflows()
    } catch (error) {
      message.error('创建工作流失败')
    }
  }

  // 工作流表格列定义
  const columns = [
    {
      title: '工作流名称',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: Workflow) => (
        <a onClick={() => setSelectedWorkflow(record)}>{text}</a>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const statusMap: any = {
          draft: { color: 'default', text: '草稿' },
          active: { color: 'success', text: '活动' },
          running: { color: 'processing', text: '运行中' },
          completed: { color: 'success', text: '已完成' },
          failed: { color: 'error', text: '失败' }
        }
        return <Tag color={statusMap[status]?.color}>{statusMap[status]?.text}</Tag>
      }
    },
    {
      title: '步骤数',
      dataIndex: 'steps',
      key: 'steps',
      render: (steps: WorkflowStep[]) => steps.length
    },
    {
      title: '执行次数',
      dataIndex: 'executions',
      key: 'executions'
    },
    {
      title: '更新时间',
      dataIndex: 'updated_at',
      key: 'updated_at',
      render: (time: string) => new Date(time).toLocaleString()
    },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: Workflow) => (
        <Space size="middle">
          <Button
            type="link"
            icon={<PlayCircleOutlined />}
            onClick={() => startWorkflow(record.id)}
            disabled={record.status === 'running'}
          >
            启动
          </Button>
          <Button type="link" icon={<EditOutlined />}>
            编辑
          </Button>
          <Button type="link" danger icon={<DeleteOutlined />}>
            删除
          </Button>
        </Space>
      )
    }
  ]

  // 渲染工作流步骤图标
  const getStepIcon = (type: string) => {
    const iconMap: any = {
      start: <PlayCircleOutlined />,
      action: <NodeIndexOutlined />,
      condition: <ForkOutlined />,
      parallel: <BranchesOutlined />,
      end: <CheckCircleOutlined />
    }
    return iconMap[type] || <NodeIndexOutlined />
  }

  const getStepProgress = (workflow: Workflow) => {
    const total = workflow.steps?.length || 0
    if (!total) return 0
    const completed = workflow.steps.filter(step => step.status === 'completed').length
    return Math.round((completed / total) * 100)
  }

  const getCurrentStepIndex = (workflow: Workflow) => {
    const runningIndex = workflow.steps.findIndex(step => step.status === 'running')
    if (runningIndex >= 0) return runningIndex
    const completedCount = workflow.steps.filter(step => step.status === 'completed').length
    return completedCount
  }

  const successRate = workflows.length
    ? Number(((workflows.filter(w => w.status === 'completed').length / workflows.length) * 100).toFixed(1))
    : 0

  const workflowExecutions = selectedWorkflow
    ? executions.filter(e => e.workflow_id === selectedWorkflow.id)
    : []

  return (
    <div style={{ padding: '24px' }}>
      <Card
        title={
          <Space>
            <BranchesOutlined />
            <span>工作流管理系统</span>
          </Space>
        }
        extra={
          <Space>
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={() => setShowCreateModal(true)}
            >
              创建工作流
            </Button>
            <Button icon={<ReloadOutlined />} onClick={fetchWorkflows}>
              刷新
            </Button>
          </Space>
        }
      >
        <Row gutter={16} style={{ marginBottom: 16 }}>
          <Col span={6}>
            <Card>
              <Statistic
                title="总工作流"
                value={workflows.length}
                prefix={<BranchesOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="运行中"
                value={workflows.filter(w => w.status === 'running').length}
                prefix={<SyncOutlined spin />}
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="总执行次数"
                value={workflows.reduce((sum, w) => sum + w.executions, 0)}
                prefix={<PlayCircleOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="成功率"
                value={successRate}
                suffix="%"
                prefix={<CheckCircleOutlined />}
                valueStyle={{ color: '#52c41a' }}
              />
            </Card>
          </Col>
        </Row>

        {selectedWorkflow ? (
          <Card title={`工作流详情: ${selectedWorkflow.name}`} style={{ marginBottom: 16 }}>
            <Steps current={getCurrentStepIndex(selectedWorkflow)}>
              {selectedWorkflow.steps.map(step => (
                <Step
                  key={step.id}
                  title={step.name}
                  icon={getStepIcon(step.type)}
                  status={
                    step.status === 'completed' ? 'finish' :
                    step.status === 'running' ? 'process' :
                    step.status === 'failed' ? 'error' :
                    'wait'
                  }
                />
              ))}
            </Steps>

            <Divider />

            <Row gutter={16}>
              <Col span={8}>
                <h4>步骤配置</h4>
                <List
                  dataSource={selectedWorkflow.steps}
                  renderItem={step => (
                    <List.Item>
                      <List.Item.Meta
                        avatar={getStepIcon(step.type)}
                        title={step.name}
                        description={`类型: ${step.type} | 状态: ${step.status}`}
                      />
                    </List.Item>
                  )}
                />
              </Col>
              <Col span={8}>
                <h4>执行历史</h4>
                {workflowExecutions.length ? (
                  <Timeline>
                    {workflowExecutions
                      .slice(0, 5)
                      .map(execution => (
                        <Timeline.Item
                          key={execution.id}
                          color={execution.status === 'completed' ? 'green' : 'blue'}
                        >
                          <p>{new Date(execution.started_at).toLocaleString()}</p>
                          <p>状态: {execution.status}</p>
                        </Timeline.Item>
                      ))}
                  </Timeline>
                ) : (
                  <Alert message="暂无执行记录" type="info" showIcon />
                )}
              </Col>
              <Col span={8}>
                <h4>工作流统计</h4>
                <Statistic title="总执行次数" value={selectedWorkflow.executions} />
                <Statistic title="步骤完成度" value={getStepProgress(selectedWorkflow)} suffix="%" />
                <Progress percent={getStepProgress(selectedWorkflow)} status="active" />
              </Col>
            </Row>
          </Card>
        ) : null}

        <Table
          columns={columns}
          dataSource={workflows}
          loading={loading}
          rowKey="id"
          pagination={{ pageSize: 10 }}
        />
      </Card>

      <Modal
        title="创建新工作流"
        visible={showCreateModal}
        onOk={() => form.submit()}
        onCancel={() => {
          setShowCreateModal(false)
          form.resetFields()
        }}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleCreateWorkflow}
        >
          <Form.Item
            label="工作流名称"
            name="name"
            rules={[{ required: true, message: '请输入工作流名称' }]}
          >
            <Input placeholder="输入工作流名称" />
          </Form.Item>

          <Form.Item
            label="描述"
            name="description"
          >
            <TextArea rows={3} placeholder="输入工作流描述" />
          </Form.Item>

          <Form.Item
            label="触发方式"
            name="trigger"
            initialValue="manual"
          >
            <Select>
              <Select.Option value="manual">手动触发</Select.Option>
              <Select.Option value="schedule">定时触发</Select.Option>
              <Select.Option value="event">事件触发</Select.Option>
            </Select>
          </Form.Item>

          <Form.Item
            label="超时时间（分钟）"
            name="timeout"
            initialValue={30}
          >
            <Input type="number" />
          </Form.Item>
        </Form>
      </Modal>

      <Drawer
        title="工作流执行详情"
        placement="right"
        onClose={() => setShowExecutionDrawer(false)}
        visible={showExecutionDrawer}
        width={600}
      >
        {selectedExecution && (
          <div>
            <Descriptions column={1}>
              <Descriptions.Item label="执行ID">{selectedExecution.id}</Descriptions.Item>
              <Descriptions.Item label="状态">
                <Badge status={selectedExecution.status === 'running' ? 'processing' : 'success'} text={selectedExecution.status} />
              </Descriptions.Item>
              <Descriptions.Item label="开始时间">{new Date(selectedExecution.started_at).toLocaleString()}</Descriptions.Item>
              {selectedExecution.completed_at && (
                <Descriptions.Item label="完成时间">{new Date(selectedExecution.completed_at).toLocaleString()}</Descriptions.Item>
              )}
              <Descriptions.Item label="当前步骤">{selectedExecution.current_step + 1}</Descriptions.Item>
            </Descriptions>

            <Divider />

            <h4>执行日志</h4>
            <div style={{ background: '#f5f5f5', padding: 12, borderRadius: 4, maxHeight: 400, overflow: 'auto' }}>
              {selectedExecution.logs.length ? (
                selectedExecution.logs.map((log, index) => (
                  <div key={index} style={{ marginBottom: 8, fontFamily: 'monospace' }}>
                    [{index + 1}] {log}
                  </div>
                ))
              ) : (
                <div style={{ color: '#888' }}>暂无日志</div>
              )}
            </div>
          </div>
        )}
      </Drawer>
    </div>
  )
}

export default WorkflowManagementPage
