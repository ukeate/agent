import { buildApiUrl, apiFetch } from '../utils/apiBase'
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
  message,
} from 'antd'
import {
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  ReloadOutlined,
  PlusOutlined,
  SettingOutlined,
  DeleteOutlined,
  ExclamationCircleOutlined,
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
  const [agents, setAgents] = useState<Agent[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const [isCreateModalVisible, setIsCreateModalVisible] = useState(false)
  const [form] = Form.useForm()

  const statusColors = {
    running: 'success',
    paused: 'warning',
    stopped: 'default',
    error: 'error',
  }

  const statusTexts = {
    running: '运行中',
    paused: '已暂停',
    stopped: '已停止',
    error: '错误',
  }

  const normalizeStatus = (status: string): Agent['status'] => {
    const mapping: Record<string, Agent['status']> = {
      active: 'running',
      running: 'running',
      paused: 'paused',
      idle: 'paused',
      stopped: 'stopped',
      error: 'error',
    }
    return mapping[status] || 'running'
  }

  const fetchAgents = async () => {
    setLoading(true)
    setError('')
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/async-agents/agents'))
      const body = await res.json()
      const agentList: Agent[] = (body.data?.agents || []).map(
        (agent: any, idx: number) => {
          const tasksTotal =
            Number(agent.total_tasks ?? agent.state?.total_tasks ?? 0) || 0
          const tasksCompleted =
            Number(
              agent.completed_tasks ?? agent.state?.completed_tasks ?? 0
            ) || 0
          const progress =
            tasksTotal > 0
              ? Math.round((tasksCompleted / tasksTotal) * 100)
              : Math.round(agent.state?.progress ?? 0)
          return {
            id: agent.id || `agent-${idx}`,
            name: agent.name || agent.role || `agent-${idx}`,
            type: agent.role || 'unknown',
            status: normalizeStatus(agent.status),
            progress: Math.max(0, Math.min(100, progress)),
            tasksCompleted,
            tasksTotal,
            lastActivity: agent.last_activity
              ? new Date(agent.last_activity).toLocaleString('zh-CN')
              : '',
            cpu: Math.round((agent.state?.load ?? 0) * 100),
            memory: Math.round(agent.state?.memory_mb ?? 0),
          }
        }
      )
      setAgents(agentList)
    } catch (e) {
      setError('加载异步智能体失败')
      setAgents([])
    } finally {
      setLoading(false)
    }
  }

  const handleAgentAction = (agentId: string, action: string) => {
    if (action === 'delete') {
      confirm({
        title: '确认删除智能体?',
        icon: <ExclamationCircleOutlined />,
        content: `这将永久删除智能体 ${agentId}`,
        okType: 'danger',
        onOk() {
          apiFetch(buildApiUrl(`/api/v1/async-agents/agents/${agentId}`), {
            method: 'DELETE',
          })
            .then(res => {
              message.success('删除成功')
              fetchAgents()
            })
            .catch(() => message.error('删除智能体失败'))
        },
      })
      return
    }
    message.info('请使用后端任务接口控制运行状态')
  }

  const handleCreateAgent = async (values: any) => {
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/async-agents/agents'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          role: values.type,
          name: values.name || undefined,
        }),
      })
      await res.json().catch(() => null)
      message.success('已创建智能体')
      setIsCreateModalVisible(false)
      form.resetFields()
      fetchAgents()
    } catch (e) {
      message.error('创建智能体失败')
    }
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
      ),
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => {
        const typeMap = {
          retrieval: { color: 'blue', text: '检索型' },
          analysis: { color: 'green', text: '分析型' },
          generation: { color: 'purple', text: '生成型' },
        }
        const config = typeMap[type as keyof typeof typeMap] || {
          color: 'default',
          text: type,
        }
        return <Tag color={config.color}>{config.text}</Tag>
      },
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: keyof typeof statusColors) => (
        <Tag color={statusColors[status]}>{statusTexts[status]}</Tag>
      ),
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
      ),
    },
    {
      title: '资源使用',
      key: 'resources',
      render: (record: Agent) => (
        <Space direction="vertical" size="small">
          <div className="text-xs">
            CPU:{' '}
            <Progress
              percent={record.cpu}
              size="small"
              showInfo={false}
              strokeColor={record.cpu > 80 ? '#ff4d4f' : '#1890ff'}
            />{' '}
            {record.cpu}%
          </div>
          <div className="text-xs">内存: {record.memory}MB</div>
        </Space>
      ),
    },
    {
      title: '最后活动',
      dataIndex: 'lastActivity',
      key: 'lastActivity',
      render: (time: string) => (
        <div className="text-xs text-gray-500">{time}</div>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: Agent) => (
        <Space>
          <Tooltip title="暂停/启动请通过任务接口控制">
            <Button
              size="small"
              icon={
                record.status === 'running' ? (
                  <PauseCircleOutlined />
                ) : (
                  <PlayCircleOutlined />
                )
              }
              disabled
            />
          </Tooltip>
          <Tooltip title="停止请通过任务接口控制">
            <Button size="small" danger icon={<StopOutlined />} disabled />
          </Tooltip>
          <Tooltip title="设置">
            <Button size="small" icon={<SettingOutlined />} />
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
      ),
    },
  ]

  React.useEffect(() => {
    fetchAgents()
  }, [])

  const runningAgents = agents.filter(a => a.status === 'running').length
  const totalTasks = agents.reduce((sum, a) => sum + a.tasksTotal, 0)
  const completedTasks = agents.reduce((sum, a) => sum + a.tasksCompleted, 0)
  const avgCpu =
    agents.length > 0
      ? Math.round(agents.reduce((sum, a) => sum + a.cpu, 0) / agents.length)
      : 0

  return (
    <div className="p-6">
      <div className="mb-6">
        <div className="flex justify-between items-center mb-4">
          <h1 className="text-2xl font-bold">异步智能体管理</h1>
          <Space>
            <Button
              icon={<ReloadOutlined />}
              onClick={fetchAgents}
              loading={loading}
            >
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
                value={
                  totalTasks > 0
                    ? Math.round((completedTasks / totalTasks) * 100)
                    : 0
                }
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

      {error && (
        <Card style={{ marginBottom: 16 }}>
          <Space direction="vertical">
            <span>{error}</span>
          </Space>
        </Card>
      )}

      <Card title="智能体列表">
        <Table
          columns={columns}
          dataSource={agents}
          rowKey="id"
          pagination={false}
          loading={loading}
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
        <Form form={form} layout="vertical" onFinish={handleCreateAgent}>
          <Form.Item
            name="name"
            label="智能体名称"
            rules={[{ required: true, message: '请输入智能体名称' }]}
          >
            <Input placeholder="请输入智能体名称" />
          </Form.Item>
          <Form.Item
            name="type"
            label="智能体角色"
            rules={[{ required: true, message: '请选择智能体角色' }]}
          >
            <Select placeholder="请选择智能体角色">
              <Option value="code_expert">代码专家</Option>
              <Option value="architect">架构师</Option>
              <Option value="doc_expert">文档专家</Option>
              <Option value="knowledge_retrieval">知识检索</Option>
              <Option value="assistant">通用助手</Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default AsyncAgentPage
