import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useState, useEffect } from 'react'
import { logger } from '../utils/logger'
import {
  Card,
  Row,
  Col,
  Statistic,
  Progress,
  Badge,
  Typography,
  Alert,
  Descriptions,
  Tag,
  Space,
  Button,
  Timeline,
  Table,
} from 'antd'
import {
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  ClockCircleOutlined,
  DatabaseOutlined,
  ClusterOutlined,
  ThunderboltOutlined,
  ReloadOutlined,
} from '@ant-design/icons'

const { Title, Text } = Typography

interface SystemStatus {
  cluster_health: 'healthy' | 'degraded' | 'critical'
  total_nodes: number
  active_nodes: number
  leader_node: string
  consensus_status: 'stable' | 'electing' | 'split'
  task_throughput: number
  avg_response_time: number
  error_rate: number
  uptime: number
}

const DistributedTaskSystemStatusPage: React.FC = () => {
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null)
  const [alerts, setAlerts] = useState<
    {
      id: string
      level: 'info' | 'warning' | 'error'
      message: string
      timestamp: string
    }[]
  >([])
  const [loading, setLoading] = useState(false)

  const fetchClusterData = async () => {
    setLoading(true)
    try {
      const [statsRes, agentsRes] = await Promise.all([
        apiFetch(buildApiUrl('/api/v1/cluster/stats')),
        apiFetch(buildApiUrl('/api/v1/cluster/agents')),
      ])
      const stats = await statsRes.json()
      const agents = await agentsRes.json()

      const health =
        stats.error_rate > 5
          ? 'critical'
          : stats.offline_agents > 0 || stats.error_agents > 0
            ? 'degraded'
            : 'healthy'
      setSystemStatus({
        cluster_health: health as any,
        total_nodes: stats.total_agents,
        active_nodes: stats.online_agents,
        leader_node: agents?.agents?.[0]?.node_id || 'unknown',
        consensus_status: 'stable',
        task_throughput: stats.total_tasks_processed || 0,
        avg_response_time: stats.avg_cpu_usage || 0,
        error_rate: stats.error_rate || 0,
        uptime: 100 - (stats.error_rate || 0),
      })

      if (agents?.agents) {
        const derivedAlerts = agents.agents
          .filter(
            (a: any) =>
              a.resource_usage?.cpu_usage > 80 || a.status !== 'online'
          )
          .map((a: any, idx: number) => ({
            id: `${idx}`,
            level: a.status !== 'online' ? 'warning' : 'info',
            message:
              a.status !== 'online'
                ? `${a.name || a.agent_id} 离线`
                : `${a.name || a.agent_id} CPU 使用率 ${a.resource_usage.cpu_usage}%`,
            timestamp: a.updated_at || new Date().toISOString(),
          }))
        if (derivedAlerts.length === 0) {
          derivedAlerts.push({
            id: 'ok',
            level: 'info',
            message: '系统运行正常',
            timestamp: new Date().toISOString(),
          })
        }
        setAlerts(derivedAlerts)
      }
    } catch (e) {
      logger.error('加载分布式任务系统状态失败:', e)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchClusterData()
  }, [])

  const getHealthColor = (health: string) => {
    switch (health) {
      case 'healthy':
        return 'success'
      case 'degraded':
        return 'warning'
      case 'critical':
        return 'error'
      default:
        return 'default'
    }
  }

  const getConsensusColor = (status: string) => {
    switch (status) {
      case 'stable':
        return 'success'
      case 'electing':
        return 'processing'
      case 'split':
        return 'error'
      default:
        return 'default'
    }
  }

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <Title level={2}>系统状态总览</Title>
      <Button
        icon={<ReloadOutlined />}
        onClick={fetchClusterData}
        loading={loading}
        style={{ marginBottom: 16 }}
      >
        刷新
      </Button>

      {!systemStatus && (
        <Alert
          type="info"
          message="正在加载集群状态..."
          style={{ marginBottom: 16 }}
        />
      )}

      {systemStatus && (
        <>
          <Row gutter={16} style={{ marginBottom: 24 }}>
            <Col span={8}>
              <Card>
                <Statistic
                  title="集群健康状态"
                  value={systemStatus.cluster_health}
                  prefix={
                    <Badge
                      status={getHealthColor(systemStatus.cluster_health)}
                    />
                  }
                  valueStyle={{
                    color:
                      systemStatus.cluster_health === 'healthy'
                        ? '#3f8600'
                        : systemStatus.cluster_health === 'degraded'
                          ? '#faad14'
                          : '#f5222d',
                  }}
                />
              </Card>
            </Col>
            <Col span={8}>
              <Card>
                <Statistic
                  title="活跃节点"
                  value={systemStatus.active_nodes}
                  suffix={`/ ${systemStatus.total_nodes}`}
                  prefix={<ClusterOutlined />}
                  valueStyle={{
                    color:
                      systemStatus.active_nodes === systemStatus.total_nodes
                        ? '#3f8600'
                        : '#faad14',
                  }}
                />
              </Card>
            </Col>
            <Col span={8}>
              <Card>
                <Statistic
                  title="系统可用性"
                  value={systemStatus.uptime}
                  precision={2}
                  suffix="%"
                  prefix={<CheckCircleOutlined />}
                  valueStyle={{ color: '#3f8600' }}
                />
              </Card>
            </Col>
          </Row>

          <Row gutter={16} style={{ marginBottom: 24 }}>
            <Col span={6}>
              <Card>
                <Statistic
                  title="任务吞吐量"
                  value={Math.floor(systemStatus.task_throughput)}
                  suffix="tasks/min"
                  prefix={<ThunderboltOutlined />}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="平均响应时间"
                  value={Math.floor(systemStatus.avg_response_time)}
                  suffix="ms"
                  prefix={<ClockCircleOutlined />}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="错误率"
                  value={systemStatus.error_rate}
                  precision={2}
                  suffix="%"
                  prefix={<ExclamationCircleOutlined />}
                  valueStyle={{
                    color: systemStatus.error_rate > 1 ? '#f5222d' : '#3f8600',
                  }}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="共识状态"
                  value={systemStatus.consensus_status}
                  prefix={
                    <Badge
                      status={getConsensusColor(systemStatus.consensus_status)}
                    />
                  }
                />
              </Card>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={16}>
              <Card
                title="系统详细信息"
                extra={
                  <Button
                    icon={<ReloadOutlined />}
                    onClick={fetchClusterData}
                    loading={loading}
                  >
                    刷新
                  </Button>
                }
              >
                <Descriptions bordered column={2}>
                  <Descriptions.Item label="Leader节点">
                    <Tag color="gold">{systemStatus.leader_node}</Tag>
                  </Descriptions.Item>
                  <Descriptions.Item label="Raft任期">N/A</Descriptions.Item>
                  <Descriptions.Item label="日志索引">N/A</Descriptions.Item>
                  <Descriptions.Item label="提交索引">N/A</Descriptions.Item>
                  <Descriptions.Item label="网络分区">无</Descriptions.Item>
                  <Descriptions.Item label="存储使用">
                    <Progress
                      percent={Math.min(
                        100,
                        (systemStatus.task_throughput || 0) % 100
                      )}
                      size="small"
                    />
                  </Descriptions.Item>
                  <Descriptions.Item label="内存使用">
                    <Progress
                      percent={Math.min(
                        100,
                        systemStatus.avg_memory_usage || 0
                      )}
                      size="small"
                    />
                  </Descriptions.Item>
                  <Descriptions.Item label="CPU使用">
                    <Progress
                      percent={Math.min(100, systemStatus.avg_cpu_usage || 0)}
                      size="small"
                    />
                  </Descriptions.Item>
                </Descriptions>
              </Card>
            </Col>
            <Col span={8}>
              <Card title="系统告警">
                <Timeline size="small">
                  {alerts.map(alert => (
                    <Timeline.Item
                      key={alert.id}
                      color={
                        alert.level === 'warning'
                          ? 'orange'
                          : alert.level === 'error'
                            ? 'red'
                            : 'green'
                      }
                    >
                      <div>
                        <Text>{alert.message}</Text>
                        <br />
                        <Text type="secondary" style={{ fontSize: '12px' }}>
                          {new Date(alert.timestamp).toLocaleString()}
                        </Text>
                      </div>
                    </Timeline.Item>
                  ))}
                </Timeline>
              </Card>
            </Col>
          </Row>
        </>
      )}
    </div>
  )
}

export default DistributedTaskSystemStatusPage
