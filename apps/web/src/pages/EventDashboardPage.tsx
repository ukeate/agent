import React, { useState, useEffect } from 'react'
import { logger } from '../utils/logger'
import {
  Card,
  Table,
  Tag,
  Button,
  Space,
  Row,
  Col,
  Statistic,
  Timeline,
  Select,
  Badge,
  Progress,
  Alert,
  Tooltip,
  Empty,
  message,
  Dropdown,
} from 'antd'
import type { MenuProps } from 'antd'
import {
  ReloadOutlined,
  ExportOutlined,
  BellOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  InfoCircleOutlined,
  WifiOutlined,
} from '@ant-design/icons'
import dayjs from 'dayjs'
import eventService, {
  Event,
  EventStats,
  ClusterStatus,
  EventStreamStatus,
} from '../services/eventService'
import { analyticsServiceEnhanced } from '../services/analyticsServiceEnhanced'

const { Option } = Select

const EventDashboardPage: React.FC = () => {
  const [events, setEvents] = useState<Event[]>([])
  const [filteredEvents, setFilteredEvents] = useState<Event[]>([])
  const [filterType, setFilterType] = useState<string>('all')
  const [filterSeverity, setFilterSeverity] = useState<string>('all')
  const [autoRefresh, setAutoRefresh] = useState(false)
  const [loading, setLoading] = useState(false)
  const [stats, setStats] = useState<EventStats>({
    total: 0,
    info: 0,
    warning: 0,
    error: 0,
    success: 0,
    critical: 0,
    by_source: {},
    by_type: {},
  })
  const [clusterStatus, setClusterStatus] = useState<ClusterStatus | null>(null)
  const [connectionStatus, setConnectionStatus] =
    useState<EventStreamStatus | null>(null)
  const [exporting, setExporting] = useState(false)

  // 加载事件数据
  const loadEvents = async () => {
    setLoading(true)
    try {
      const [eventsData, statsData, clusterData] = await Promise.all([
        eventService.getEvents({ limit: 100 }),
        eventService.getEventStats(24),
        eventService.getClusterStatus(),
      ])

      setEvents(eventsData)
      setStats(statsData)
      setClusterStatus(clusterData)
    } catch (error) {
      message.error('加载事件数据失败')
    } finally {
      setLoading(false)
    }
  }

  // 初始加载
  useEffect(() => {
    loadEvents()
  }, [])

  // 过滤事件
  useEffect(() => {
    let filtered = events

    if (filterType !== 'all') {
      filtered = filtered.filter(event => event.type === filterType)
    }

    if (filterSeverity !== 'all') {
      filtered = filtered.filter(event => event.severity === filterSeverity)
    }

    setFilteredEvents(filtered)
  }, [events, filterType, filterSeverity])

  // WebSocket连接管理
  useEffect(() => {
    if (!autoRefresh) {
      setConnectionStatus(null)
      return
    }
    // 连接WebSocket事件流
    const handleNewEvent = (event: Event) => {
      setEvents(prev => [event, ...prev].slice(0, 100)) // 保留最新100条

      // 更新统计信息
      setStats(prev => ({
        ...prev,
        total: prev.total + 1,
        [event.type]:
          ((prev[event.type as keyof EventStats] as number) || 0) + 1,
        critical:
          event.severity === 'critical' ? prev.critical + 1 : prev.critical,
      }))
    }

    const unsubscribe = eventService.subscribeConnectionStatus(status => {
      setConnectionStatus(status)
    })
    eventService.connectEventStream(handleNewEvent)

    // 定期刷新统计和集群状态
    const refreshInterval = setInterval(async () => {
      try {
        const [statsData, clusterData] = await Promise.all([
          eventService.getEventStats(24),
          eventService.getClusterStatus(),
        ])
        setStats(statsData)
        setClusterStatus(clusterData)
      } catch (error) {
        logger.error('刷新数据失败:', error)
      }
    }, 30000) // 每30秒刷新一次

    return () => {
      eventService.removeEventHandler(handleNewEvent)
      unsubscribe()
      clearInterval(refreshInterval)
    }
  }, [autoRefresh])

  const typeColors = {
    info: 'blue',
    warning: 'orange',
    error: 'red',
    success: 'green',
  }

  const typeIcons = {
    info: <InfoCircleOutlined />,
    warning: <WarningOutlined />,
    error: <CloseCircleOutlined />,
    success: <CheckCircleOutlined />,
  }

  const severityColors = {
    low: 'default',
    medium: 'warning',
    high: 'error',
    critical: 'error',
  }

  const severityTexts = {
    low: '低',
    medium: '中',
    high: '高',
    critical: '严重',
  }

  const renderConnectionTag = () => {
    if (!autoRefresh || !connectionStatus) return null
    const { state, attempts, lastError } = connectionStatus
    const config: Record<
      EventStreamStatus['state'],
      { color: 'success' | 'processing' | 'warning' | 'default'; text: string; icon: React.ReactNode }
    > = {
      connected: { color: 'success', text: '实时连接', icon: <WifiOutlined /> },
      connecting: { color: 'processing', text: '连接中', icon: <WifiOutlined /> },
      reconnecting: {
        color: 'warning',
        text: attempts > 0 ? `重连中 ${attempts}次` : '重连中',
        icon: <ReloadOutlined />,
      },
      offline: { color: 'warning', text: '网络离线', icon: <WarningOutlined /> },
      disconnected: { color: 'default', text: '已断开', icon: <CloseCircleOutlined /> },
    }
    const { color, text, icon } = config[state]
    const tag = (
      <Tag icon={icon} color={color}>
        {text}
      </Tag>
    )
    return lastError ? <Tooltip title={lastError}>{tag}</Tooltip> : tag
  }

  const exportMenuItems: MenuProps['items'] = [
    { key: 'csv', label: '导出 CSV' },
    { key: 'json', label: '导出 JSON' },
    { key: 'xlsx', label: '导出 XLSX' },
  ]

  const handleExport = async (format: 'csv' | 'json' | 'xlsx') => {
    if (exporting) return
    setExporting(true)
    try {
      const blob = await analyticsServiceEnhanced.exportEventData({ format })
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = `events_${dayjs().format('YYYYMMDD_HHmmss')}.${format}`
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.URL.revokeObjectURL(url)
      message.success('事件导出已开始')
    } catch (error) {
      message.error('导出事件失败')
    } finally {
      setExporting(false)
    }
  }

  const columns = [
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 160,
      render: (time: string) => (
        <div className="text-xs text-gray-600">
          {dayjs(time).format('YYYY-MM-DD HH:mm:ss')}
        </div>
      ),
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      width: 80,
      render: (type: keyof typeof typeColors) => (
        <Tag color={typeColors[type]} icon={typeIcons[type]}>
          {type.toUpperCase()}
        </Tag>
      ),
    },
    {
      title: '严重程度',
      dataIndex: 'severity',
      key: 'severity',
      width: 100,
      render: (severity: keyof typeof severityColors) => (
        <Tag color={severityColors[severity]}>{severityTexts[severity]}</Tag>
      ),
    },
    {
      title: '来源',
      dataIndex: 'source',
      key: 'source',
      width: 120,
    },
    {
      title: '智能体',
      dataIndex: 'agent',
      key: 'agent',
      width: 120,
      render: (agent: string) => (agent ? <Tag>{agent}</Tag> : '-'),
    },
    {
      title: '标题',
      dataIndex: 'title',
      key: 'title',
      width: 200,
      render: (title: string) => <strong>{title}</strong>,
    },
    {
      title: '消息',
      dataIndex: 'message',
      key: 'message',
      ellipsis: {
        showTitle: false,
      },
      render: (message: string) => (
        <Tooltip title={message}>
          <div className="text-gray-600">{message}</div>
        </Tooltip>
      ),
    },
  ]

  const recentEvents = events.slice(0, 10)

  // 手动提交测试事件
  const submitTestEvent = async () => {
    try {
      await eventService.submitEvent({
        event_type: 'MESSAGE_SENT',
        source: 'Test',
        message: '这是一个测试事件',
        priority: 'normal',
      })
      message.success('测试事件已提交')
      loadEvents()
    } catch (error) {
      message.error('提交测试事件失败')
    }
  }

  return (
    <div className="p-6">
      <div className="mb-6">
        <div className="flex justify-between items-center mb-4">
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold">事件监控仪表板</h1>
            {renderConnectionTag()}
          </div>
          <Space>
            <Button
              icon={<BellOutlined />}
              onClick={() => setAutoRefresh(!autoRefresh)}
              type={autoRefresh ? 'primary' : 'default'}
            >
              {autoRefresh ? '关闭' : '开启'}实时监控
            </Button>
            <Button
              icon={<ReloadOutlined />}
              onClick={loadEvents}
              loading={loading}
            >
              刷新
            </Button>
            <Button onClick={submitTestEvent}>发送测试事件</Button>
            <Dropdown
              menu={{
                items: exportMenuItems,
                onClick: ({ key }) =>
                  handleExport(key as 'csv' | 'json' | 'xlsx'),
              }}
              trigger={['click']}
            >
              <Button icon={<ExportOutlined />} loading={exporting}>
                导出日志
              </Button>
            </Dropdown>
          </Space>
        </div>

        {autoRefresh && (
          <Alert
            message="实时监控已开启"
            description="系统正在通过WebSocket接收实时事件，统计数据每30秒自动更新"
            type="info"
            showIcon
            closable
            className="mb-4"
          />
        )}

        <Row gutter={16} className="mb-6">
          <Col span={4}>
            <Card>
              <Statistic
                title="总事件数"
                value={stats.total}
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
          <Col span={4}>
            <Card>
              <Statistic
                title="成功事件"
                value={stats.success}
                valueStyle={{ color: '#3f8600' }}
              />
            </Card>
          </Col>
          <Col span={4}>
            <Card>
              <Statistic
                title="警告事件"
                value={stats.warning}
                valueStyle={{ color: '#fa8c16' }}
              />
            </Card>
          </Col>
          <Col span={4}>
            <Card>
              <Statistic
                title="错误事件"
                value={stats.error}
                valueStyle={{ color: '#cf1322' }}
              />
            </Card>
          </Col>
          <Col span={4}>
            <Card>
              <Statistic
                title="严重事件"
                value={stats.critical}
                valueStyle={{ color: '#cf1322' }}
              />
            </Card>
          </Col>
          <Col span={4}>
            <Card>
              <Statistic
                title="信息事件"
                value={stats.info}
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
        </Row>
      </div>

      <Row gutter={16}>
        <Col span={16}>
          <Card
            title="事件列表"
            loading={loading}
            extra={
              <Space>
                <Select
                  placeholder="筛选类型"
                  style={{ width: 120 }}
                  value={filterType}
                  onChange={setFilterType}
                >
                  <Option value="all">全部类型</Option>
                  <Option value="info">信息</Option>
                  <Option value="warning">警告</Option>
                  <Option value="error">错误</Option>
                  <Option value="success">成功</Option>
                </Select>
                <Select
                  placeholder="筛选严重程度"
                  style={{ width: 120 }}
                  value={filterSeverity}
                  onChange={setFilterSeverity}
                >
                  <Option value="all">全部程度</Option>
                  <Option value="low">低</Option>
                  <Option value="medium">中</Option>
                  <Option value="high">高</Option>
                  <Option value="critical">严重</Option>
                </Select>
              </Space>
            }
          >
            <Table
              columns={columns}
              dataSource={filteredEvents}
              rowKey="id"
              pagination={{
                pageSize: 10,
                showSizeChanger: true,
                showQuickJumper: true,
                showTotal: total => `共 ${total} 条事件`,
              }}
              size="small"
            />
          </Card>
        </Col>

        <Col span={8}>
          <Card title="最近事件时间线" className="mb-4">
            {recentEvents.length > 0 ? (
              <Timeline
                items={recentEvents.map(event => ({
                  color: typeColors[event.type as keyof typeof typeColors],
                  children: (
                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-medium">{event.title}</span>
                        <Tag
                          color={
                            severityColors[
                              event.severity as keyof typeof severityColors
                            ]
                          }
                        >
                          {
                            severityTexts[
                              event.severity as keyof typeof severityTexts
                            ]
                          }
                        </Tag>
                      </div>
                      <div className="text-xs text-gray-500 mb-1">
                        {dayjs(event.timestamp).format('YYYY-MM-DD HH:mm:ss')}
                      </div>
                      <div className="text-sm text-gray-600">
                        {event.message}
                      </div>
                    </div>
                  ),
                }))}
              />
            ) : (
              <Empty description="暂无事件" />
            )}
          </Card>

          {clusterStatus && (
            <Card title="集群状态" className="mb-4">
              <Space direction="vertical" style={{ width: '100%' }}>
                <div className="flex justify-between">
                  <span>节点ID</span>
                  <Tag>{clusterStatus.node_id}</Tag>
                </div>
                <div className="flex justify-between">
                  <span>角色</span>
                  <Tag
                    color={clusterStatus.role === 'leader' ? 'gold' : 'default'}
                  >
                    {clusterStatus.role}
                  </Tag>
                </div>
                <div className="flex justify-between">
                  <span>状态</span>
                  <Badge status="processing" text={clusterStatus.status} />
                </div>
                <div className="flex justify-between">
                  <span>活跃节点</span>
                  <span>{clusterStatus.active_nodes}</span>
                </div>
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span>负载</span>
                    <span>{(clusterStatus.load * 100).toFixed(1)}%</span>
                  </div>
                  <Progress
                    percent={clusterStatus.load * 100}
                    size="small"
                    status="active"
                  />
                </div>
              </Space>
            </Card>
          )}

          <Card title="系统健康状态">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span>系统状态</span>
                  <Badge status="processing" text="运行中" />
                </div>
                <Progress percent={95} size="small" status="active" />
              </div>

              <div>
                <div className="flex justify-between items-center mb-2">
                  <span>错误率</span>
                  <span className="text-red-500">
                    {stats.total > 0
                      ? ((stats.error / stats.total) * 100).toFixed(1)
                      : 0}
                    %
                  </span>
                </div>
                <Progress
                  percent={
                    stats.total > 0 ? (stats.error / stats.total) * 100 : 0
                  }
                  size="small"
                  strokeColor="#ff4d4f"
                />
              </div>

              <div>
                <div className="flex justify-between items-center mb-2">
                  <span>成功率</span>
                  <span className="text-green-500">
                    {stats.total > 0
                      ? ((stats.success / stats.total) * 100).toFixed(1)
                      : 0}
                    %
                  </span>
                </div>
                <Progress
                  percent={
                    stats.total > 0 ? (stats.success / stats.total) * 100 : 0
                  }
                  size="small"
                  strokeColor="#52c41a"
                />
              </div>
            </Space>
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default EventDashboardPage
