import React, { useState } from 'react'
import { 
  Card, 
  Table, 
  Tag, 
  Button, 
  Space, 
  Progress, 
  Statistic, 
  Row, 
  Col,
  Badge,
  Tooltip,
  Modal,
  Form,
  Input,
  Select,
  message
} from 'antd'
import { 
  PlayCircleOutlined, 
  PauseCircleOutlined, 
  StopOutlined,
  ReloadOutlined,
  PlusOutlined,
  SettingOutlined,
  DeleteOutlined,
  ExclamationCircleOutlined
} from '@ant-design/icons'

const { Option } = Select
const { confirm } = Modal

interface Agent {
  id: string
  name: string
  type: string
  status: 'running' | 'paused' | 'stopped' | 'error'
  progress: number
  tasksCompleted: number
  tasksTotal: number
  lastActivity: string
  cpu: number
  memory: number
}

const AsyncAgentPage: React.FC = () => {
  const [agents, setAgents] = useState<Agent[]>([
    {
      id: '1',
      name: 'RAG处理器',
      type: 'retrieval',
      status: 'running',
      progress: 75,
      tasksCompleted: 15,
      tasksTotal: 20,
      lastActivity: '2024-01-15 14:30:25',
      cpu: 45,
      memory: 128
    },
    {
      id: '2', 
      name: '文档分析器',
      type: 'analysis',
      status: 'paused',
      progress: 30,
      tasksCompleted: 6,
      tasksTotal: 20,
      lastActivity: '2024-01-15 14:25:10',
      cpu: 0,
      memory: 64
    },
    {
      id: '3',
      name: '代码生成器',
      type: 'generation',
      status: 'running',
      progress: 90,
      tasksCompleted: 18,
      tasksTotal: 20,
      lastActivity: '2024-01-15 14:31:45',
      cpu: 78,
      memory: 256
    }
  ])

  const [isCreateModalVisible, setIsCreateModalVisible] = useState(false)
  const [form] = Form.useForm()

  const statusColors = {
    running: 'success',
    paused: 'warning', 
    stopped: 'default',
    error: 'error'
  }

  const statusTexts = {
    running: '运行中',
    paused: '已暂停',
    stopped: '已停止',
    error: '错误'
  }

  const handleAgentAction = (agentId: string, action: string) => {
    const agent = agents.find(a => a.id === agentId)
    if (!agent) return

    switch (action) {
      case 'start':
        setAgents(prev => prev.map(a => 
          a.id === agentId ? { ...a, status: 'running' as const } : a
        ))
        message.success(`已启动智能体: ${agent.name}`)
        break
      case 'pause':
        setAgents(prev => prev.map(a => 
          a.id === agentId ? { ...a, status: 'paused' as const } : a
        ))
        message.success(`已暂停智能体: ${agent.name}`)
        break
      case 'stop':
        confirm({
          title: '确认停止智能体?',
          icon: <ExclamationCircleOutlined />,
          content: `这将停止智能体 "${agent.name}" 的所有任务`,
          onOk() {
            setAgents(prev => prev.map(a => 
              a.id === agentId ? { ...a, status: 'stopped' as const, progress: 0 } : a
            ))
            message.success(`已停止智能体: ${agent.name}`)
          }
        })
        break
      case 'delete':
        confirm({
          title: '确认删除智能体?',
          icon: <ExclamationCircleOutlined />,
          content: `这将永久删除智能体 "${agent.name}"`,
          okType: 'danger',
          onOk() {
            setAgents(prev => prev.filter(a => a.id !== agentId))
            message.success(`已删除智能体: ${agent.name}`)
          }
        })
        break
    }
  }

  const handleCreateAgent = (values: any) => {
    const newAgent: Agent = {
      id: Date.now().toString(),
      name: values.name,
      type: values.type,
      status: 'stopped',
      progress: 0,
      tasksCompleted: 0,
      tasksTotal: 0,
      lastActivity: new Date().toLocaleString('zh-CN'),
      cpu: 0,
      memory: 0
    }
    
    setAgents(prev => [...prev, newAgent])
    setIsCreateModalVisible(false)
    form.resetFields()
    message.success(`已创建智能体: ${newAgent.name}`)
  }

  const columns = [
    {
      title: '智能体名称',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: Agent) => (
        <Space>
          <Badge status={statusColors[record.status] as any} />
          <strong>{text}</strong>
        </Space>
      )
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => {
        const typeMap = {
          retrieval: { color: 'blue', text: '检索型' },
          analysis: { color: 'green', text: '分析型' },
          generation: { color: 'purple', text: '生成型' }
        }
        const config = typeMap[type as keyof typeof typeMap] || { color: 'default', text: type }
        return <Tag color={config.color}>{config.text}</Tag>
      }
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: keyof typeof statusColors) => (
        <Tag color={statusColors[status]}>{statusTexts[status]}</Tag>
      )
    },
    {
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      render: (progress: number, record: Agent) => (
        <div>
          <Progress 
            percent={progress} 
            size="small" 
            status={record.status === 'error' ? 'exception' : 'active'}
          />
          <div className="text-xs text-gray-500 mt-1">
            {record.tasksCompleted}/{record.tasksTotal} 任务
          </div>
        </div>
      )
    },
    {
      title: '资源使用',
      key: 'resources',
      render: (record: Agent) => (
        <Space direction="vertical" size="small">
          <div className="text-xs">
            CPU: <Progress 
              percent={record.cpu} 
              size="small" 
              showInfo={false}
              strokeColor={record.cpu > 80 ? '#ff4d4f' : '#1890ff'}
            /> {record.cpu}%
          </div>
          <div className="text-xs">
            内存: {record.memory}MB
          </div>
        </Space>
      )
    },
    {
      title: '最后活动',
      dataIndex: 'lastActivity',
      key: 'lastActivity',
      render: (time: string) => (
        <div className="text-xs text-gray-500">{time}</div>
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: Agent) => (
        <Space>
          {record.status === 'running' ? (
            <Tooltip title="暂停">
              <Button 
                size="small" 
                icon={<PauseCircleOutlined />}
                onClick={() => handleAgentAction(record.id, 'pause')}
              />
            </Tooltip>
          ) : (
            <Tooltip title="启动">
              <Button 
                size="small" 
                type="primary"
                icon={<PlayCircleOutlined />}
                onClick={() => handleAgentAction(record.id, 'start')}
              />
            </Tooltip>
          )}
          <Tooltip title="停止">
            <Button 
              size="small" 
              danger
              icon={<StopOutlined />}
              onClick={() => handleAgentAction(record.id, 'stop')}
            />
          </Tooltip>
          <Tooltip title="设置">
            <Button 
              size="small" 
              icon={<SettingOutlined />}
            />
          </Tooltip>
          <Tooltip title="删除">
            <Button 
              size="small" 
              danger
              icon={<DeleteOutlined />}
              onClick={() => handleAgentAction(record.id, 'delete')}
            />
          </Tooltip>
        </Space>
      )
    }
  ]

  const runningAgents = agents.filter(a => a.status === 'running').length
  const totalTasks = agents.reduce((sum, a) => sum + a.tasksTotal, 0)
  const completedTasks = agents.reduce((sum, a) => sum + a.tasksCompleted, 0)
  const avgCpu = agents.length > 0 ? Math.round(agents.reduce((sum, a) => sum + a.cpu, 0) / agents.length) : 0

  return (
    <div className="p-6">
      <div className="mb-6">
        <div className="flex justify-between items-center mb-4">
          <h1 className="text-2xl font-bold">异步智能体管理</h1>
          <Space>
            <Button icon={<ReloadOutlined />} onClick={() => window.location.reload()}>
              刷新
            </Button>
            <Button 
              type="primary" 
              icon={<PlusOutlined />}
              onClick={() => setIsCreateModalVisible(true)}
            >
              创建智能体
            </Button>
          </Space>
        </div>

        <Row gutter={16} className="mb-6">
          <Col span={6}>
            <Card>
              <Statistic
                title="运行中的智能体"
                value={runningAgents}
                suffix={`/ ${agents.length}`}
                valueStyle={{ color: '#3f8600' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="任务进度"
                value={totalTasks > 0 ? Math.round((completedTasks / totalTasks) * 100) : 0}
                suffix="%"
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="平均CPU使用率"
                value={avgCpu}
                suffix="%"
                valueStyle={{ color: avgCpu > 80 ? '#cf1322' : '#1890ff' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="总完成任务"
                value={completedTasks}
                valueStyle={{ color: '#3f8600' }}
              />
            </Card>
          </Col>
        </Row>
      </div>

      <Card title="智能体列表">
        <Table
          columns={columns}
          dataSource={agents}
          rowKey="id"
          pagination={false}
          size="middle"
        />
      </Card>

      <Modal
        title="创建新智能体"
        open={isCreateModalVisible}
        onCancel={() => setIsCreateModalVisible(false)}
        onOk={form.submit}
        okText="创建"
        cancelText="取消"
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleCreateAgent}
        >
          <Form.Item
            name="name"
            label="智能体名称"
            rules={[{ required: true, message: '请输入智能体名称' }]}
          >
            <Input placeholder="请输入智能体名称" />
          </Form.Item>
          <Form.Item
            name="type"
            label="智能体类型"
            rules={[{ required: true, message: '请选择智能体类型' }]}
          >
            <Select placeholder="请选择智能体类型">
              <Option value="retrieval">检索型</Option>
              <Option value="analysis">分析型</Option>
              <Option value="generation">生成型</Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default AsyncAgentPage