import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Statistic,
  Table,
  Tag,
  Space,
  Progress,
  Typography,
  Tabs,
  Timeline,
  Avatar,
  Spin,
  message,
} from 'antd'
import {
  RocketOutlined,
  MonitorOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  ExclamationCircleOutlined,
  ApiOutlined,
  SettingOutlined,
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'

const { Title, Text } = Typography
const { TabPane } = Tabs

interface PlatformStatus {
  total_components: number
  healthy_components: number
  unhealthy_components: number
  active_workflows: number
  system_health: 'healthy' | 'degraded' | 'unhealthy' | 'critical'
  cpu_usage: number
  memory_usage: number
  disk_usage: number
}

interface Component {
  component_id: string
  name: string
  component_type: string
  version: string
  status: 'healthy' | 'unhealthy' | 'starting' | 'stopping' | 'error'
  last_check: string
  uptime: number
}

interface Workflow {
  workflow_id: string
  name: string
  status: 'running' | 'completed' | 'failed' | 'pending' | 'cancelled' | 'error'
  progress: number
  started_at: string
  estimated_completion?: string
}

interface RecentActivity {
  id: string
  type:
    | 'component_registered'
    | 'workflow_started'
    | 'workflow_completed'
    | 'alert_generated'
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
      const [healthRes, componentsRes, workflowsRes, metricsRes] =
        await Promise.allSettled([
          apiFetch(buildApiUrl('/api/v1/platform/health')),
          apiFetch(buildApiUrl('/api/v1/platform/components')),
          apiFetch(buildApiUrl('/api/v1/platform/workflows?limit=10')),
          apiFetch(buildApiUrl('/api/v1/platform/optimization/metrics')),
        ])

      let nextComponents: Component[] = []
      let nextWorkflows: Workflow[] = []

      if (componentsRes.status === 'fulfilled') {
        const componentsData = await componentsRes.value.json()
        nextComponents = Object.values(componentsData.components || {}).map(
          (c: any) => ({
            component_id: String(c.component_id || ''),
            name: String(c.name || ''),
            component_type: String(c.component_type || ''),
            version: String(c.version || ''),
            status: c.status,
            last_check: String(c.last_heartbeat || ''),
            uptime: c.registered_at
              ? Math.max(
                  0,
                  (Date.now() - new Date(c.registered_at).getTime()) / 1000
                )
              : 0,
          })
        )
        setComponents(nextComponents)
      }

      if (workflowsRes.status === 'fulfilled') {
        const workflowsData = await workflowsRes.value.json()
        const list = Array.isArray(workflowsData.workflows)
          ? workflowsData.workflows
          : []
        nextWorkflows = list.map((w: any) => {
          const steps = Array.isArray(w.steps) ? w.steps : []
          const totalSteps = steps.length || Number(w.total_steps || 0)
          const completedSteps = steps.filter(
            (s: any) => s.status === 'completed'
          ).length
          const progress =
            totalSteps > 0 ? Math.round((completedSteps / totalSteps) * 100) : 0
          return {
            workflow_id: String(w.workflow_id || ''),
            name: String(w.workflow_type || ''),
            status: w.status,
            progress,
            started_at: String(w.started_at || ''),
          }
        })
        setWorkflows(nextWorkflows)
      }

      if (healthRes.status === 'fulfilled') {
        const healthData = await healthRes.value.json()
        const total = Number(healthData.total_components || 0)
        const healthy = Number(healthData.healthy_components || 0)

        let cpu = 0
        let mem = 0
        let disk = 0
        if (metricsRes.status === 'fulfilled') {
          const metricsData = await metricsRes.value.json()
          cpu = Number(metricsData.metrics?.cpu_percent || 0)
          mem = Number(metricsData.metrics?.memory_percent || 0)
          disk = Number(metricsData.metrics?.disk_usage?.percent || 0)
        }

        setStatus({
          total_components: total,
          healthy_components: healthy,
          unhealthy_components: Math.max(0, total - healthy),
          active_workflows: nextWorkflows.filter(w => w.status === 'running')
            .length,
          system_health: healthData.overall_status,
          cpu_usage: cpu,
          memory_usage: mem,
          disk_usage: disk,
        })
      }

      const nextActivities: RecentActivity[] = []
      nextComponents.forEach(c => {
        if (!c.last_check) return
        nextActivities.push({
          id: `component:${c.component_id}`,
          type: 'component_registered',
          message: `组件已注册: ${c.name}`,
          timestamp: c.last_check,
        })
      })
      nextWorkflows.forEach(w => {
        if (!w.started_at) return
        nextActivities.push({
          id: `workflow:${w.workflow_id}`,
          type: 'workflow_started',
          message: `工作流启动: ${w.name || w.workflow_id}`,
          timestamp: w.started_at,
        })
      })
      nextActivities.sort(
        (a, b) =>
          new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      )
      setActivities(nextActivities.slice(0, 20))
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
      ),
    },
    {
      title: '类型',
      dataIndex: 'component_type',
      key: 'component_type',
      render: type => <Tag color="blue">{type}</Tag>,
    },
    {
      title: '版本',
      dataIndex: 'version',
      key: 'version',
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: status => (
        <Tag
          color={status === 'healthy' ? 'green' : 'red'}
          icon={
            status === 'healthy' ? (
              <CheckCircleOutlined />
            ) : (
              <ExclamationCircleOutlined />
            )
          }
        >
          {status === 'healthy' ? '健康' : '异常'}
        </Tag>
      ),
    },
    {
      title: '运行时间',
      dataIndex: 'uptime',
      key: 'uptime',
      render: uptime =>
        `${Math.floor(uptime / 3600)}h ${Math.floor((uptime % 3600) / 60)}m`,
    },
  ]

  const workflowColumns: ColumnsType<Workflow> = [
    {
      title: '工作流名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: status => {
        const statusConfig = {
          running: {
            color: 'processing',
            text: '运行中',
            icon: <ClockCircleOutlined />,
          },
          completed: {
            color: 'success',
            text: '已完成',
            icon: <CheckCircleOutlined />,
          },
          failed: {
            color: 'error',
            text: '失败',
            icon: <ExclamationCircleOutlined />,
          },
          pending: {
            color: 'default',
            text: '待执行',
            icon: <ClockCircleOutlined />,
          },
          cancelled: {
            color: 'default',
            text: '已取消',
            icon: <ClockCircleOutlined />,
          },
          error: {
            color: 'error',
            text: '错误',
            icon: <ExclamationCircleOutlined />,
          },
        }
        const config = statusConfig[status] || {
          color: 'default',
          text: String(status),
          icon: <ClockCircleOutlined />,
        }
        return (
          <Tag color={config.color} icon={config.icon}>
            {config.text}
          </Tag>
        )
      },
    },
    {
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      render: progress => <Progress percent={progress} size="small" />,
    },
    {
      title: '开始时间',
      dataIndex: 'started_at',
      key: 'started_at',
      render: time => new Date(time).toLocaleString(),
    },
  ]

  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'component_registered':
        return <ApiOutlined />
      case 'workflow_started':
        return <RocketOutlined />
      case 'workflow_completed':
        return <CheckCircleOutlined />
      case 'alert_generated':
        return <ExclamationCircleOutlined />
      default:
        return <SettingOutlined />
    }
  }

  if (loading) {
    return (
      <Spin
        size="large"
        style={{ display: 'block', textAlign: 'center', marginTop: 100 }}
      />
    )
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
                  valueStyle={{
                    color:
                      status.system_health === 'healthy'
                        ? '#52c41a'
                        : '#f5222d',
                  }}
                />
              </Card>
            </Col>
          </Row>

          <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
            <Col span={8}>
              <Card title="CPU使用率" size="small">
                <Progress
                  percent={status.cpu_usage}
                  status={
                    status.cpu_usage > 80
                      ? 'exception'
                      : status.cpu_usage > 60
                        ? 'active'
                        : 'success'
                  }
                />
              </Card>
            </Col>
            <Col span={8}>
              <Card title="内存使用率" size="small">
                <Progress
                  percent={status.memory_usage}
                  status={
                    status.memory_usage > 80
                      ? 'exception'
                      : status.memory_usage > 60
                        ? 'active'
                        : 'success'
                  }
                />
              </Card>
            </Col>
            <Col span={8}>
              <Card title="磁盘使用率" size="small">
                <Progress
                  percent={status.disk_usage}
                  status={
                    status.disk_usage > 80
                      ? 'exception'
                      : status.disk_usage > 60
                        ? 'active'
                        : 'success'
                  }
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
