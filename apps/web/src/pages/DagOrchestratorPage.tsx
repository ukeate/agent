import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Button,
  Space,
  Table,
  Tag,
  Modal,
  Form,
  Input,
  Select,
  message,
  Statistic,
  Progress,
  Typography,
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
  NodeIndexOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  DeleteOutlined,
  EditOutlined
} from '@ant-design/icons'

const { Title, Text } = Typography
const { TabPane } = Tabs
const { Option } = Select

interface DAGNode {
  id: string
  name: string
  type: 'task' | 'condition' | 'start' | 'end'
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped'
  dependencies: string[]
  duration?: number
  start_time?: string
  end_time?: string
  metadata?: Record<string, any>
}

interface DAGWorkflow {
  id: string
  name: string
  description: string
  status: 'draft' | 'active' | 'paused' | 'completed' | 'failed'
  created_at: string
  nodes: DAGNode[]
  total_nodes: number
  completed_nodes: number
  failed_nodes: number
  execution_time: number
  progress: number
}

interface DAGExecution {
  id: string
  workflow_id: string
  workflow_name: string
  status: string
  start_time: string
  end_time?: string
  duration: number
  total_tasks: number
  completed_tasks: number
  failed_tasks: number
  execution_log: string[]
}

const DagOrchestratorPage: React.FC = () => {
  const [workflows, setWorkflows] = useState<DAGWorkflow[]>([])
  const [executions, setExecutions] = useState<DAGExecution[]>([])
  const [, setSelectedWorkflow] = useState<DAGWorkflow | null>(null)
  const [loading, setLoading] = useState(false)
  const [createModalVisible, setCreateModalVisible] = useState(false)
  const [form] = Form.useForm()

  useEffect(() => {
    loadWorkflows()
    loadExecutions()
  }, [])

  const loadWorkflows = async () => {
    setLoading(true)
    try {
      // 模拟数据 - 实际应调用后端API
      const mockWorkflows: DAGWorkflow[] = [
        {
          id: 'dag-001',
          name: 'AI模型训练流水线',
          description: '包含数据预处理、模型训练、验证和部署的完整流水线',
          status: 'active',
          created_at: '2024-01-15T10:00:00Z',
          nodes: [],
          total_nodes: 8,
          completed_nodes: 6,
          failed_nodes: 0,
          execution_time: 1800,
          progress: 75
        },
        {
          id: 'dag-002',
          name: '数据ETL处理流程',
          description: '从多个数据源抽取、转换和加载数据',
          status: 'completed',
          created_at: '2024-01-14T08:30:00Z',
          nodes: [],
          total_nodes: 12,
          completed_nodes: 12,
          failed_nodes: 0,
          execution_time: 3600,
          progress: 100
        },
        {
          id: 'dag-003',
          name: '自动化测试套件',
          description: '包含单元测试、集成测试和性能测试的自动化流程',
          status: 'failed',
          created_at: '2024-01-16T14:20:00Z',
          nodes: [],
          total_nodes: 15,
          completed_nodes: 10,
          failed_nodes: 2,
          execution_time: 900,
          progress: 67
        }
      ]
      setWorkflows(mockWorkflows)
    } catch (error) {
      console.error('加载工作流失败:', error)
      message.error('加载工作流失败')
    } finally {
      setLoading(false)
    }
  }

  const loadExecutions = async () => {
    try {
      // 模拟执行历史数据
      const mockExecutions: DAGExecution[] = [
        {
          id: 'exec-001',
          workflow_id: 'dag-001',
          workflow_name: 'AI模型训练流水线',
          status: 'running',
          start_time: '2024-01-16T15:30:00Z',
          duration: 1800,
          total_tasks: 8,
          completed_tasks: 6,
          failed_tasks: 0,
          execution_log: [
            '15:30:00 - 开始执行工作流',
            '15:32:15 - 数据预处理任务完成',
            '15:45:30 - 特征工程任务完成',
            '16:00:00 - 模型训练任务进行中...'
          ]
        },
        {
          id: 'exec-002',
          workflow_id: 'dag-002',
          workflow_name: '数据ETL处理流程',
          status: 'completed',
          start_time: '2024-01-16T08:00:00Z',
          end_time: '2024-01-16T09:00:00Z',
          duration: 3600,
          total_tasks: 12,
          completed_tasks: 12,
          failed_tasks: 0,
          execution_log: [
            '08:00:00 - 工作流执行开始',
            '08:15:00 - 数据源连接建立',
            '08:30:00 - 数据抽取完成',
            '08:45:00 - 数据转换完成',
            '09:00:00 - 数据加载完成'
          ]
        }
      ]
      setExecutions(mockExecutions)
    } catch (error) {
      console.error('加载执行历史失败:', error)
    }
  }

  const executeWorkflow = async (workflowId: string) => {
    try {
      message.success(`开始执行工作流: ${workflowId}`)
      // 实际应调用后端API执行工作流
      await loadWorkflows()
      await loadExecutions()
    } catch (error) {
      console.error('执行工作流失败:', error)
      message.error('执行工作流失败')
    }
  }

  const pauseWorkflow = async (workflowId: string) => {
    try {
      message.success(`暂停工作流: ${workflowId}`)
      await loadWorkflows()
    } catch (error) {
      console.error('暂停工作流失败:', error)
      message.error('暂停工作流失败')
    }
  }

  const stopWorkflow = async (workflowId: string) => {
    try {
      message.success(`停止工作流: ${workflowId}`)
      await loadWorkflows()
    } catch (error) {
      console.error('停止工作流失败:', error)
      message.error('停止工作流失败')
    }
  }

  const getStatusColor = (status: string) => {
    const colors = {
      draft: 'default',
      active: 'processing',
      paused: 'warning',
      completed: 'success',
      failed: 'error',
      running: 'processing'
    }
    return colors[status as keyof typeof colors] || 'default'
  }

  const getStatusIcon = (status: string) => {
    const icons = {
      active: <PlayCircleOutlined />,
      paused: <PauseCircleOutlined />,
      completed: <CheckCircleOutlined />,
      failed: <ExclamationCircleOutlined />,
      running: <ClockCircleOutlined />
    }
    return icons[status as keyof typeof icons]
  }

  const workflowColumns = [
    {
      title: '工作流名称',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: DAGWorkflow) => (
        <div>
          <Text strong>{name}</Text>
          <br />
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {record.description}
          </Text>
        </div>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={getStatusColor(status)} icon={getStatusIcon(status)}>
          {status.toUpperCase()}
        </Tag>
      )
    },
    {
      title: '进度',
      key: 'progress',
      render: (_, record: DAGWorkflow) => (
        <div style={{ width: '120px' }}>
          <Progress 
            percent={record.progress} 
            size="small"
            format={() => `${record.completed_nodes}/${record.total_nodes}`}
          />
        </div>
      )
    },
    {
      title: '执行时间',
      dataIndex: 'execution_time',
      key: 'execution_time',
      render: (time: number) => {
        const minutes = Math.floor(time / 60)
        const seconds = time % 60
        return `${minutes}m ${seconds}s`
      }
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
      render: (_, record: DAGWorkflow) => (
        <Space>
          {record.status === 'draft' || record.status === 'paused' ? (
            <Tooltip title="执行">
              <Button 
                size="small" 
                type="primary"
                icon={<PlayCircleOutlined />}
                onClick={() => executeWorkflow(record.id)}
              />
            </Tooltip>
          ) : null}
          {record.status === 'active' ? (
            <>
              <Tooltip title="暂停">
                <Button 
                  size="small" 
                  icon={<PauseCircleOutlined />}
                  onClick={() => pauseWorkflow(record.id)}
                />
              </Tooltip>
              <Tooltip title="停止">
                <Button 
                  size="small" 
                  danger
                  icon={<StopOutlined />}
                  onClick={() => stopWorkflow(record.id)}
                />
              </Tooltip>
            </>
          ) : null}
          <Tooltip title="编辑">
            <Button 
              size="small" 
              icon={<EditOutlined />}
              onClick={() => setSelectedWorkflow(record)}
            />
          </Tooltip>
          <Tooltip title="删除">
            <Button 
              size="small" 
              danger
              icon={<DeleteOutlined />}
            />
          </Tooltip>
        </Space>
      )
    }
  ]

  const executionColumns = [
    {
      title: '执行ID',
      dataIndex: 'id',
      key: 'id',
      render: (id: string) => <code>{id}</code>
    },
    {
      title: '工作流',
      dataIndex: 'workflow_name',
      key: 'workflow_name'
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
      title: '进度',
      key: 'progress',
      render: (_, record: DAGExecution) => (
        <div style={{ width: '120px' }}>
          <Progress 
            percent={Math.round((record.completed_tasks / record.total_tasks) * 100)} 
            size="small"
            format={() => `${record.completed_tasks}/${record.total_tasks}`}
          />
        </div>
      )
    },
    {
      title: '持续时间',
      dataIndex: 'duration',
      key: 'duration',
      render: (time: number) => {
        const minutes = Math.floor(time / 60)
        const seconds = time % 60
        return `${minutes}m ${seconds}s`
      }
    },
    {
      title: '开始时间',
      dataIndex: 'start_time',
      key: 'start_time',
      render: (time: string) => new Date(time).toLocaleString()
    }
  ]

  return (
    <div className="p-6">
      <div className="mb-6">
        <div className="flex justify-between items-center mb-4">
          <Title level={2}>DAG编排器 (任务依赖)</Title>
          <Space>
            <Button 
              type="primary"
              icon={<PlusOutlined />}
              onClick={() => setCreateModalVisible(true)}
            >
              创建工作流
            </Button>
            <Button 
              icon={<ReloadOutlined />}
              onClick={() => {
                loadWorkflows()
                loadExecutions()
              }}
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
                title="活跃工作流"
                value={workflows.filter(w => w.status === 'active').length}
                prefix={<NodeIndexOutlined />}
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="总执行次数"
                value={executions.length}
                prefix={<PlayCircleOutlined />}
                valueStyle={{ color: '#52c41a' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="成功率"
                value={85.7}
                suffix="%"
                prefix={<CheckCircleOutlined />}
                valueStyle={{ color: '#52c41a' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="平均执行时间"
                value="28m 30s"
                prefix={<ClockCircleOutlined />}
                valueStyle={{ color: '#faad14' }}
              />
            </Card>
          </Col>
        </Row>
      </div>

      <Tabs defaultActiveKey="workflows">
        <TabPane tab="工作流管理" key="workflows">
          <Card title="DAG工作流列表">
            <Table
              columns={workflowColumns}
              dataSource={workflows}
              rowKey="id"
              loading={loading}
              pagination={{ pageSize: 10 }}
            />
          </Card>
        </TabPane>

        <TabPane tab="执行历史" key="executions">
          <Card title="执行历史记录">
            <Table
              columns={executionColumns}
              dataSource={executions}
              rowKey="id"
              pagination={{ pageSize: 10 }}
              expandable={{
                expandedRowRender: (record) => (
                  <div className="p-4">
                    <Title level={5}>执行日志</Title>
                    <Timeline
                      items={record.execution_log.map((log, index) => ({
                        children: log,
                        color: index === record.execution_log.length - 1 ? 'blue' : 'green'
                      }))}
                    />
                  </div>
                )
              }}
            />
          </Card>
        </TabPane>

        <TabPane tab="可视化编辑器" key="visual-editor">
          <Card title="DAG可视化编辑器">
            <div className="text-center py-12">
              <NodeIndexOutlined style={{ fontSize: '48px', color: '#d9d9d9' }} />
              <div className="mt-4">
                <Title level={4} type="secondary">可视化编辑器</Title>
                <Text type="secondary">
                  拖拽式DAG工作流设计器，支持节点依赖关系可视化编辑
                </Text>
              </div>
              <Button type="primary" className="mt-4">
                打开编辑器
              </Button>
            </div>
          </Card>
        </TabPane>
      </Tabs>

      <Modal
        title="创建新工作流"
        open={createModalVisible}
        onCancel={() => setCreateModalVisible(false)}
        footer={null}
      >
        <Form form={form} layout="vertical">
          <Form.Item
            name="name"
            label="工作流名称"
            rules={[{ required: true, message: '请输入工作流名称' }]}
          >
            <Input placeholder="输入工作流名称" />
          </Form.Item>
          <Form.Item
            name="description"
            label="描述"
            rules={[{ required: true, message: '请输入描述' }]}
          >
            <Input.TextArea placeholder="输入工作流描述" rows={3} />
          </Form.Item>
          <Form.Item
            name="template"
            label="模板"
          >
            <Select placeholder="选择工作流模板（可选）">
              <Option value="ml-pipeline">机器学习流水线</Option>
              <Option value="etl-process">数据ETL处理</Option>
              <Option value="test-suite">自动化测试套件</Option>
              <Option value="deployment">部署流程</Option>
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

export default DagOrchestratorPage