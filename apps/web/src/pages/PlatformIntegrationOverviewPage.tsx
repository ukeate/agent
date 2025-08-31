import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Statistic,
  Table,
  Tag,
  Button,
  Space,
  Progress,
  Alert,
  Typography,
  Tabs,
  Timeline,
  Avatar,
  Spin,
  message
} from 'antd'
import {
  RocketOutlined,
  ThunderboltOutlined,
  MonitorOutlined,
  FileTextOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  ExclamationCircleOutlined,
  ApiOutlined,
  SettingOutlined,
  TrendingUpOutlined
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'

const { Title, Text } = Typography
const { TabPane } = Tabs

interface PlatformStatus {
  total_components: number
  healthy_components: number
  unhealthy_components: number
  active_workflows: number
  system_health: 'healthy' | 'degraded' | 'unhealthy'
  cpu_usage: number
  memory_usage: number
  disk_usage: number
}

interface Component {
  component_id: string
  name: string
  component_type: string
  version: string
  status: 'healthy' | 'unhealthy'
  last_check: string
  uptime: number
}

interface Workflow {
  workflow_id: string
  name: string
  status: 'running' | 'completed' | 'failed' | 'pending'
  progress: number
  started_at: string
  estimated_completion?: string
}

interface RecentActivity {
  id: string
  type: 'component_registered' | 'workflow_started' | 'workflow_completed' | 'alert_generated'
  message: string
  timestamp: string
}

const PlatformIntegrationOverviewPage: React.FC = () => {
  const [loading, setLoading] = useState(true)
  const [status, setStatus] = useState<PlatformStatus | null>(null)
  const [components, setComponents] = useState<Component[]>([])
  const [workflows, setWorkflows] = useState<Workflow[]>([])
  const [activities, setActivities] = useState<RecentActivity[]>([])

  useEffect(() => {
    fetchOverviewData()
    const interval = setInterval(fetchOverviewData, 30000)
    return () => clearInterval(interval)
  }, [])

  const fetchOverviewData = async () => {
    try {
      const [statusRes, componentsRes, workflowsRes, activitiesRes] = await Promise.all([
        fetch('/api/v1/platform-integration/status'),
        fetch('/api/v1/platform-integration/components'),
        fetch('/api/v1/platform-integration/workflows?limit=10'),
        fetch('/api/v1/platform-integration/activities?limit=20')
      ])

      if (statusRes.ok) {
        const statusData = await statusRes.json()
        setStatus(statusData)
      }

      if (componentsRes.ok) {
        const componentsData = await componentsRes.json()
        setComponents(componentsData.components || [])
      }

      if (workflowsRes.ok) {
        const workflowsData = await workflowsRes.json()
        setWorkflows(workflowsData.workflows || [])
      }

      if (activitiesRes.ok) {
        const activitiesData = await activitiesRes.json()
        setActivities(activitiesData.activities || [])
      }
    } catch (error) {
      message.error('获取平台状态失败')
    } finally {
      setLoading(false)
    }
  }

  const componentColumns: ColumnsType<Component> = [
    {
      title: '组件名称',
      dataIndex: 'name',
      key: 'name',
      render: (text, record) => (
        <Space>
          <Avatar icon={<ApiOutlined />} size="small" />
          <div>
            <div>{text}</div>
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {record.component_id}
            </Text>
          </div>
        </Space>
      )
    },
    {
      title: '类型',
      dataIndex: 'component_type',
      key: 'component_type',
      render: (type) => <Tag color="blue">{type}</Tag>
    },
    {
      title: '版本',
      dataIndex: 'version',
      key: 'version'
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => (
        <Tag
          color={status === 'healthy' ? 'green' : 'red'}
          icon={status === 'healthy' ? <CheckCircleOutlined /> : <ExclamationCircleOutlined />}
        >
          {status === 'healthy' ? '健康' : '异常'}
        </Tag>
      )
    },
    {
      title: '运行时间',
      dataIndex: 'uptime',
      key: 'uptime',
      render: (uptime) => `${Math.floor(uptime / 3600)}h ${Math.floor((uptime % 3600) / 60)}m`
    }
  ]

  const workflowColumns: ColumnsType<Workflow> = [
    {
      title: '工作流名称',
      dataIndex: 'name',
      key: 'name'
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => {
        const statusConfig = {
          running: { color: 'processing', text: '运行中', icon: <ClockCircleOutlined /> },
          completed: { color: 'success', text: '已完成', icon: <CheckCircleOutlined /> },
          failed: { color: 'error', text: '失败', icon: <ExclamationCircleOutlined /> },
          pending: { color: 'default', text: '待执行', icon: <ClockCircleOutlined /> }
        }
        const config = statusConfig[status]
        return <Tag color={config.color} icon={config.icon}>{config.text}</Tag>
      }
    },
    {
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      render: (progress) => <Progress percent={progress} size="small" />
    },
    {
      title: '开始时间',
      dataIndex: 'started_at',
      key: 'started_at',
      render: (time) => new Date(time).toLocaleString()
    }
  ]

  const getSystemHealthColor = (health: string) => {
    switch (health) {
      case 'healthy': return 'success'
      case 'degraded': return 'warning'
      case 'unhealthy': return 'error'
      default: return 'default'
    }
  }

  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'component_registered': return <ApiOutlined />
      case 'workflow_started': return <RocketOutlined />
      case 'workflow_completed': return <CheckCircleOutlined />
      case 'alert_generated': return <ExclamationCircleOutlined />
      default: return <SettingOutlined />
    }
  }

  if (loading) {
    return <Spin size="large" style={{ display: 'block', textAlign: 'center', marginTop: 100 }} />
  }

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>平台集成总览</Title>
      
      {status && (
        <>
          <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
            <Col span={6}>
              <Card>
                <Statistic
                  title="总组件数"
                  value={status.total_components}
                  prefix={<ApiOutlined />}
                  valueStyle={{ color: '#1890ff' }}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="健康组件"
                  value={status.healthy_components}
                  prefix={<CheckCircleOutlined />}
                  valueStyle={{ color: '#52c41a' }}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="活跃工作流"
                  value={status.active_workflows}
                  prefix={<RocketOutlined />}
                  valueStyle={{ color: '#722ed1' }}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="系统健康度"
                  value={status.system_health}
                  prefix={<MonitorOutlined />}
                  valueStyle={{ color: status.system_health === 'healthy' ? '#52c41a' : '#f5222d' }}
                />
              </Card>
            </Col>
          </Row>

          <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
            <Col span={8}>
              <Card title="CPU使用率" size="small">
                <Progress 
                  percent={status.cpu_usage} 
                  status={status.cpu_usage > 80 ? 'exception' : status.cpu_usage > 60 ? 'active' : 'success'}
                />
              </Card>
            </Col>
            <Col span={8}>
              <Card title="内存使用率" size="small">
                <Progress 
                  percent={status.memory_usage}
                  status={status.memory_usage > 80 ? 'exception' : status.memory_usage > 60 ? 'active' : 'success'}
                />
              </Card>
            </Col>
            <Col span={8}>
              <Card title="磁盘使用率" size="small">
                <Progress 
                  percent={status.disk_usage}
                  status={status.disk_usage > 80 ? 'exception' : status.disk_usage > 60 ? 'active' : 'success'}
                />
              </Card>
            </Col>
          </Row>
        </>
      )}

      <Row gutter={[16, 16]}>
        <Col span={16}>
          <Card>
            <Tabs defaultActiveKey="components">
              <TabPane tab="组件状态" key="components">
                <Table
                  columns={componentColumns}
                  dataSource={components}
                  rowKey="component_id"
                  pagination={{ pageSize: 10 }}
                  size="small"
                />
              </TabPane>
              <TabPane tab="工作流执行" key="workflows">
                <Table
                  columns={workflowColumns}
                  dataSource={workflows}
                  rowKey="workflow_id"
                  pagination={{ pageSize: 10 }}
                  size="small"
                />
              </TabPane>
            </Tabs>
          </Card>
        </Col>
        <Col span={8}>
          <Card title="最近活动" size="small">
            <Timeline size="small">
              {activities.slice(0, 10).map(activity => (
                <Timeline.Item
                  key={activity.id}
                  dot={getActivityIcon(activity.type)}
                  color={activity.type === 'alert_generated' ? 'red' : 'blue'}
                >
                  <div>
                    <div>{activity.message}</div>
                    <Text type="secondary" style={{ fontSize: '12px' }}>
                      {new Date(activity.timestamp).toLocaleString()}
                    </Text>
                  </div>
                </Timeline.Item>
              ))}
            </Timeline>
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default PlatformIntegrationOverviewPage