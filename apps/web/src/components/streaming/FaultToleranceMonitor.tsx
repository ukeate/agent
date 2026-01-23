import { buildApiUrl, apiFetch } from '../../utils/apiBase'
import React, { useState, useEffect } from 'react'
import {
  Card,
  Table,
  Tag,
  Button,
  Progress,
  Space,
  Alert,
  Row,
  Col,
  Statistic,
  Timeline,
} from 'antd'
import { logger } from '../../utils/logger'
import {
  ReloadOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
} from '@ant-design/icons'

interface ConnectionInfo {
  session_id: string
  state:
    | 'connected'
    | 'disconnected'
    | 'reconnecting'
    | 'failed'
    | 'permanently_failed'
  retry_count: number
  uptime_seconds: number
  total_reconnections: number
  heartbeat_alive: boolean
  buffered_messages: number
  metrics: {
    total_connections: number
    successful_connections: number
    failed_connections: number
    last_failure_reason?: string
  }
}

interface FaultToleranceStats {
  total_active_connections: number
  healthy_connections: number
  failed_connections: number
  average_uptime: number
  total_reconnections: number
  connections: ConnectionInfo[]
}

const FaultToleranceMonitor: React.FC = () => {
  const [stats, setStats] = useState<FaultToleranceStats>({
    total_active_connections: 0,
    healthy_connections: 0,
    failed_connections: 0,
    average_uptime: 0,
    total_reconnections: 0,
    connections: [],
  })
  const [loading, setLoading] = useState(false)
  const [lastUpdate, setLastUpdate] = useState<Date>()

  const fetchFaultToleranceStats = async () => {
    setLoading(true)
    try {
      const response = await apiFetch(
        buildApiUrl('/api/v1/streaming/fault-tolerance/stats')
      )
      const data = await response.json()
      setStats(data)
      setLastUpdate(new Date())
    } catch (error) {
      logger.error('获取容错统计失败:', error)
    } finally {
      setLoading(false)
    }
  }

  const forceReconnect = async (sessionId: string) => {
    try {
      const response = await apiFetch(
        buildApiUrl(`/api/v1/streaming/fault-tolerance/reconnect/${sessionId}`),
        {
          method: 'POST',
        }
      )
      await response.json().catch(() => null)
      fetchFaultToleranceStats()
    } catch (error) {
      logger.error('强制重连失败:', error)
    }
  }

  useEffect(() => {
    fetchFaultToleranceStats()
    const interval = setInterval(fetchFaultToleranceStats, 5000)
    return () => clearInterval(interval)
  }, [])

  const getStateColor = (state: string) => {
    switch (state) {
      case 'connected':
        return 'green'
      case 'reconnecting':
        return 'orange'
      case 'failed':
        return 'red'
      case 'permanently_failed':
        return 'red'
      default:
        return 'default'
    }
  }

  const getStateIcon = (state: string) => {
    switch (state) {
      case 'connected':
        return <CheckCircleOutlined />
      case 'reconnecting':
        return <SyncOutlined spin />
      case 'failed':
      case 'permanently_failed':
        return <CloseCircleOutlined />
      default:
        return <ExclamationCircleOutlined />
    }
  }

  const formatUptime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    return `${hours}h ${minutes}m`
  }

  const connectionColumns = [
    {
      title: '会话ID',
      dataIndex: 'session_id',
      key: 'session_id',
      render: (id: string) => <code>{id}</code>,
    },
    {
      title: '连接状态',
      dataIndex: 'state',
      key: 'state',
      render: (state: string) => (
        <Tag color={getStateColor(state)} icon={getStateIcon(state)}>
          {state.toUpperCase()}
        </Tag>
      ),
    },
    {
      title: '运行时间',
      dataIndex: 'uptime_seconds',
      key: 'uptime',
      render: (seconds: number) => formatUptime(seconds),
    },
    {
      title: '重连次数',
      dataIndex: 'retry_count',
      key: 'retry_count',
    },
    {
      title: '历史重连',
      dataIndex: 'total_reconnections',
      key: 'total_reconnections',
    },
    {
      title: '心跳状态',
      dataIndex: 'heartbeat_alive',
      key: 'heartbeat_alive',
      render: (alive: boolean) => (
        <Tag color={alive ? 'green' : 'red'}>{alive ? '正常' : '异常'}</Tag>
      ),
    },
    {
      title: '缓存消息',
      dataIndex: 'buffered_messages',
      key: 'buffered_messages',
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record: ConnectionInfo) => (
        <Space>
          <Button
            size="small"
            icon={<ReloadOutlined />}
            onClick={() => forceReconnect(record.session_id)}
            disabled={record.state === 'connected'}
          >
            重连
          </Button>
        </Space>
      ),
    },
  ]

  const healthPercentage =
    stats.total_active_connections > 0
      ? Math.round(
          (stats.healthy_connections / stats.total_active_connections) * 100
        )
      : 100

  return (
    <div className="fault-tolerance-monitor">
      <Row gutter={[16, 16]} className="mb-4">
        <Col span={24}>
          <Alert
            message="容错连接监控"
            description="实时监控流式处理连接状态、重连统计和错误恢复情况"
            type="info"
            showIcon
          />
        </Col>
      </Row>

      <Row gutter={[16, 16]} className="mb-4">
        <Col span={6}>
          <Card>
            <Statistic
              title="活跃连接"
              value={stats.total_active_connections}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="健康连接"
              value={stats.healthy_connections}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="失败连接"
              value={stats.failed_connections}
              valueStyle={{ color: '#ff4d4f' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="总重连次数"
              value={stats.total_reconnections}
              valueStyle={{ color: '#fa8c16' }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} className="mb-4">
        <Col span={12}>
          <Card title="连接健康度" size="small">
            <Progress
              type="circle"
              percent={healthPercentage}
              format={() => `${healthPercentage}%`}
              status={
                healthPercentage >= 90
                  ? 'success'
                  : healthPercentage >= 70
                    ? 'active'
                    : 'exception'
              }
            />
            <div className="mt-2 text-center text-gray-600">
              {stats.healthy_connections}/{stats.total_active_connections}{' '}
              连接健康
            </div>
          </Card>
        </Col>
        <Col span={12}>
          <Card title="系统状态" size="small">
            <div className="space-y-2">
              <div className="flex justify-between">
                <span>平均运行时间:</span>
                <span>{formatUptime(stats.average_uptime)}</span>
              </div>
              <div className="flex justify-between">
                <span>最后更新:</span>
                <span>{lastUpdate?.toLocaleTimeString()}</span>
              </div>
              <div className="flex justify-between">
                <span>监控状态:</span>
                <Tag color={loading ? 'orange' : 'green'}>
                  {loading ? '更新中' : '正常'}
                </Tag>
              </div>
            </div>
          </Card>
        </Col>
      </Row>

      <Card
        title="连接详情"
        extra={
          <Button
            icon={<ReloadOutlined />}
            onClick={fetchFaultToleranceStats}
            loading={loading}
          >
            刷新
          </Button>
        }
      >
        <Table
          columns={connectionColumns}
          dataSource={stats.connections}
          rowKey="session_id"
          loading={loading}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: total => `共 ${total} 个连接`,
          }}
          scroll={{ x: 'max-content' }}
        />
      </Card>

      {stats.connections.some(conn => conn.metrics.last_failure_reason) && (
        <Card title="最近错误" className="mt-4">
          <Timeline>
            {stats.connections
              .filter(conn => conn.metrics.last_failure_reason)
              .map(conn => (
                <Timeline.Item
                  key={conn.session_id}
                  color="red"
                  dot={<CloseCircleOutlined />}
                >
                  <div>
                    <strong>会话 {conn.session_id.slice(0, 8)}:</strong>
                    <br />
                    {conn.metrics.last_failure_reason}
                  </div>
                </Timeline.Item>
              ))}
          </Timeline>
        </Card>
      )}
    </div>
  )
}

export default FaultToleranceMonitor
