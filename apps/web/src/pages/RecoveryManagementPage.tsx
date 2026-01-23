import React, { useEffect, useState } from 'react'
import {
  Card,
  Table,
  Tag,
  Space,
  Statistic,
  Row,
  Col,
  Button,
  Typography,
  message,
} from 'antd'
import {
  SyncOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  WarningOutlined,
  ReloadOutlined,
  HistoryOutlined,
} from '@ant-design/icons'
import {
  faultToleranceService,
  type FaultEvent,
} from '../services/faultToleranceService'

interface RecoveryRecord {
  fault_id: string
  fault_type: string
  recovery_success: boolean
  recovery_time: number
  recovery_actions: Array<{
    strategy: string
    success: boolean
    timestamp: string
    details?: string
  }>
  started_at: string
  completed_at?: string
}

interface RecoveryStatistics {
  total_recoveries: number
  success_rate: number
  avg_recovery_time: number
  strategy_success_rates: Record<string, number>
  recent_recoveries?: RecoveryRecord[]
}

const statusColor: Record<string, string> = {
  true: 'green',
  false: 'red',
}

const RecoveryManagementPage: React.FC = () => {
  const [stats, setStats] = useState<RecoveryStatistics | null>(null)
  const [events, setEvents] = useState<FaultEvent[]>([])
  const [loading, setLoading] = useState(false)

  const loadData = async () => {
    setLoading(true)
    try {
      const [statData, activeEvents] = await Promise.all([
        faultToleranceService.getRecoveryStatistics(),
        faultToleranceService.listFaultEvents(undefined, undefined, false),
      ])
      setStats(statData)
      setEvents(activeEvents)
    } catch (error: any) {
      message.error(error?.message || '加载数据失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
    const timer = setInterval(loadData, 10000)
    return () => clearInterval(timer)
  }, [])

  const recoveryColumns = [
    { title: '故障ID', dataIndex: 'fault_id', key: 'fault_id', width: 140 },
    {
      title: '类型',
      dataIndex: 'fault_type',
      key: 'fault_type',
      render: (t: string) => <Tag>{t}</Tag>,
    },
    {
      title: '成功',
      dataIndex: 'recovery_success',
      key: 'recovery_success',
      width: 90,
      render: (s: boolean) => (
        <Tag color={statusColor[String(s)]}>{s ? '成功' : '失败'}</Tag>
      ),
    },
    {
      title: '耗时(s)',
      dataIndex: 'recovery_time',
      key: 'recovery_time',
      width: 100,
    },
    {
      title: '开始时间',
      dataIndex: 'started_at',
      key: 'started_at',
      width: 180,
      render: (t: string) => new Date(t).toLocaleString(),
    },
    {
      title: '完成时间',
      dataIndex: 'completed_at',
      key: 'completed_at',
      width: 180,
      render: (t?: string) => (t ? new Date(t).toLocaleString() : '进行中'),
    },
    {
      title: '策略',
      dataIndex: 'recovery_actions',
      key: 'recovery_actions',
      render: (actions: RecoveryRecord['recovery_actions']) =>
        actions && actions.length
          ? actions.map(a => (
              <Tag key={a.timestamp} color={a.success ? 'green' : 'red'}>
                {a.strategy}
              </Tag>
            ))
          : '-',
    },
  ]

  const activeColumns = [
    { title: '故障ID', dataIndex: 'fault_id', key: 'fault_id', width: 140 },
    {
      title: '类型',
      dataIndex: 'fault_type',
      key: 'fault_type',
      render: (t: string) => <Tag color="orange">{t}</Tag>,
    },
    {
      title: '严重级别',
      dataIndex: 'severity',
      key: 'severity',
      render: (s: string) => <Tag color="red">{s}</Tag>,
    },
    {
      title: '检测时间',
      dataIndex: 'detected_at',
      key: 'detected_at',
      width: 180,
      render: (t: string) => new Date(t).toLocaleString(),
    },
    {
      title: '影响组件',
      dataIndex: 'affected_components',
      key: 'affected_components',
      render: (arr: string[]) => arr?.map(c => <Tag key={c}>{c}</Tag>),
    },
  ]

  return (
    <div style={{ padding: 24 }}>
      <Space align="center" style={{ marginBottom: 16 }}>
        <SyncOutlined />
        <Typography.Title level={3} style={{ margin: 0 }}>
          恢复管理
        </Typography.Title>
      </Space>

      <Card
        style={{ marginBottom: 16 }}
        title="统计概览"
        extra={
          <Button
            icon={<ReloadOutlined />}
            onClick={loadData}
            loading={loading}
          >
            刷新
          </Button>
        }
      >
        <Row gutter={16}>
          <Col xs={24} sm={12} md={6}>
            <Statistic
              title="总恢复次数"
              value={stats?.total_recoveries || 0}
              prefix={<HistoryOutlined />}
            />
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Statistic
              title="成功率"
              value={((stats?.success_rate || 0) * 100).toFixed(1)}
              suffix="%"
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Statistic
              title="平均恢复时间(秒)"
              value={(stats?.avg_recovery_time || 0).toFixed(1)}
              prefix={<WarningOutlined />}
            />
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Statistic
              title="活跃故障"
              value={events.length}
              prefix={<CloseCircleOutlined />}
              valueStyle={{ color: events.length ? '#fa541c' : '#52c41a' }}
            />
          </Col>
        </Row>
      </Card>

      <Card title="活跃故障" style={{ marginBottom: 16 }}>
        <Table
          rowKey="fault_id"
          columns={activeColumns}
          dataSource={events}
          loading={loading}
          pagination={{ pageSize: 5 }}
        />
      </Card>

      <Card title="最近恢复记录">
        <Table
          rowKey="fault_id"
          columns={recoveryColumns}
          dataSource={stats?.recent_recoveries || []}
          loading={loading}
          pagination={{ pageSize: 8 }}
        />
      </Card>
    </div>
  )
}

export default RecoveryManagementPage
