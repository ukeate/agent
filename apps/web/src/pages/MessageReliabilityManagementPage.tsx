import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Table,
  Button,
  Form,
  Input,
  Select,
  Space,
  Typography,
  Alert,
  Tag,
  Modal,
  Tabs,
  Badge,
  Progress,
  Statistic,
  Timeline,
  Switch,
  Slider,
  Tooltip,
  Divider,
  notification,
  Radio,
  Drawer,
  Steps,
  Collapse,
  Descriptions,
} from 'antd'
import {
  SafetyCertificateOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ExclamationCircleOutlined,
  ReloadOutlined,
  SettingOutlined,
  MonitorOutlined,
  AlertOutlined,
  SyncOutlined,
  ClockCircleOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  RestOutlined,
  CarryOutOutlined,
  MessageOutlined,
  EyeOutlined,
  EditOutlined,
  DeleteOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  RedoOutlined,
  StopOutlined,
  WarningOutlined,
  SecurityScanOutlined,
  ApiOutlined,
  LineChartOutlined,
  BellOutlined,
  FileTextOutlined,
} from '@ant-design/icons'

const { Title, Text, Paragraph } = Typography
const { Option } = Select
const { TextArea } = Input
const { TabPane } = Tabs
const { Panel } = Collapse
const { Step } = Steps

interface ReliabilityMetrics {
  deliveryRate: number
  messageIntegrity: number
  averageRetries: number
  duplicateRate: number
  orderingAccuracy: number
  deadLetterCount: number
  backupStatus: 'healthy' | 'warning' | 'error'
  lastBackupTime: string
}

interface DeliveryGuarantee {
  id: string
  name: string
  description: string
  level: 'at-least-once' | 'at-most-once' | 'exactly-once'
  subjects: string[]
  enabled: boolean
  config: {
    acknowledgmentTimeout: number
    maxRetries: number
    retryDelay: number
    duplicateDetectionWindow: number
    persistentStorage: boolean
  }
  metrics: {
    messagesProcessed: number
    successRate: number
    duplicatesDetected: number
    retriesPerformed: number
  }
  status: 'active' | 'inactive' | 'error'
  lastUpdated: string
}

interface MessageBackup {
  id: string
  name: string
  description: string
  type: 'full' | 'incremental' | 'stream-specific'
  schedule: string
  retention: number
  compression: boolean
  encryption: boolean
  status: 'running' | 'completed' | 'failed' | 'scheduled'
  lastRun: string
  nextRun: string
  backupSize: number
  messagesCount: number
  destination: string
}

interface DeadLetterMessage {
  id: string
  originalMessageId: string
  subject: string
  payload: any
  sender: string
  receiver: string
  failureReason: string
  failureCount: number
  firstFailureTime: string
  lastFailureTime: string
  status: 'quarantined' | 'reprocessing' | 'discarded'
  tags: string[]
}

interface CircuitBreaker {
  id: string
  name: string
  description: string
  subject: string
  state: 'closed' | 'open' | 'half-open'
  failureThreshold: number
  recoveryTimeout: number
  requestTimeout: number
  currentFailures: number
  totalRequests: number
  failureRate: number
  lastFailureTime?: string
  nextRetryTime?: string
  enabled: boolean
}

const MessageReliabilityManagementPage: React.FC = () => {
  const [form] = Form.useForm()
  const [activeTab, setActiveTab] = useState('overview')
  const [loading, setLoading] = useState(false)
  const [guaranteeModalVisible, setGuaranteeModalVisible] = useState(false)
  const [backupModalVisible, setBackupModalVisible] = useState(false)
  const [deadLetterDrawerVisible, setDeadLetterDrawerVisible] = useState(false)
  const [selectedGuarantee, setSelectedGuarantee] =
    useState<DeliveryGuarantee | null>(null)

  const [reliabilityMetrics, setReliabilityMetrics] =
    useState<ReliabilityMetrics>({
      deliveryRate: 99.7,
      messageIntegrity: 99.95,
      averageRetries: 0.8,
      duplicateRate: 0.03,
      orderingAccuracy: 98.9,
      deadLetterCount: 23,
      backupStatus: 'healthy',
      lastBackupTime: '2025-08-26 02:00:00',
    })

  const [deliveryGuarantees, setDeliveryGuarantees] = useState<
    DeliveryGuarantee[]
  >([
    {
      id: 'guarantee-001',
      name: '任务消息精确传递',
      description: '确保任务相关消息精确传递一次，避免重复处理',
      level: 'exactly-once',
      subjects: ['agents.tasks.>', 'agents.workflows.>'],
      enabled: true,
      config: {
        acknowledgmentTimeout: 5000,
        maxRetries: 3,
        retryDelay: 1000,
        duplicateDetectionWindow: 300000,
        persistentStorage: true,
      },
      metrics: {
        messagesProcessed: 15642,
        successRate: 99.8,
        duplicatesDetected: 12,
        retriesPerformed: 89,
      },
      status: 'active',
      lastUpdated: '2025-08-26 12:30:00',
    },
    {
      id: 'guarantee-002',
      name: '事件通知至少一次',
      description: '确保事件通知至少传递一次，允许重复',
      level: 'at-least-once',
      subjects: ['system.events.>', 'agents.notifications.>'],
      enabled: true,
      config: {
        acknowledgmentTimeout: 3000,
        maxRetries: 5,
        retryDelay: 500,
        duplicateDetectionWindow: 0,
        persistentStorage: true,
      },
      metrics: {
        messagesProcessed: 28945,
        successRate: 99.5,
        duplicatesDetected: 145,
        retriesPerformed: 234,
      },
      status: 'active',
      lastUpdated: '2025-08-26 12:15:00',
    },
    {
      id: 'guarantee-003',
      name: '状态更新最多一次',
      description: '状态更新消息最多传递一次，避免状态冲突',
      level: 'at-most-once',
      subjects: ['agents.status.>', 'system.heartbeat.>'],
      enabled: true,
      config: {
        acknowledgmentTimeout: 1000,
        maxRetries: 0,
        retryDelay: 0,
        duplicateDetectionWindow: 0,
        persistentStorage: false,
      },
      metrics: {
        messagesProcessed: 45632,
        successRate: 97.8,
        duplicatesDetected: 0,
        retriesPerformed: 0,
      },
      status: 'active',
      lastUpdated: '2025-08-26 12:00:00',
    },
  ])

  const [messageBackups, setMessageBackups] = useState<MessageBackup[]>([
    {
      id: 'backup-001',
      name: '全量消息备份',
      description: '每日全量备份所有消息数据',
      type: 'full',
      schedule: '0 2 * * *',
      retention: 30,
      compression: true,
      encryption: true,
      status: 'completed',
      lastRun: '2025-08-26 02:00:00',
      nextRun: '2025-08-27 02:00:00',
      backupSize: 2048576000,
      messagesCount: 1250847,
      destination: 's3://message-backups/full/',
    },
    {
      id: 'backup-002',
      name: '增量消息备份',
      description: '每小时增量备份新消息',
      type: 'incremental',
      schedule: '0 * * * *',
      retention: 7,
      compression: true,
      encryption: false,
      status: 'completed',
      lastRun: '2025-08-26 12:00:00',
      nextRun: '2025-08-26 13:00:00',
      backupSize: 67108864,
      messagesCount: 4562,
      destination: 's3://message-backups/incremental/',
    },
    {
      id: 'backup-003',
      name: '任务流备份',
      description: '专门备份任务相关消息流',
      type: 'stream-specific',
      schedule: '*/15 * * * *',
      retention: 14,
      compression: false,
      encryption: true,
      status: 'running',
      lastRun: '2025-08-26 12:30:00',
      nextRun: '2025-08-26 12:45:00',
      backupSize: 33554432,
      messagesCount: 892,
      destination: 's3://message-backups/tasks/',
    },
  ])

  const [deadLetterMessages, setDeadLetterMessages] = useState<
    DeadLetterMessage[]
  >([
    {
      id: 'dl-001',
      originalMessageId: 'msg-abc123',
      subject: 'agents.tasks.process',
      payload: { taskId: 'task-failed-001', data: {} },
      sender: 'client-agent',
      receiver: 'worker-agent-offline',
      failureReason: 'Receiver agent offline - connection timeout',
      failureCount: 3,
      firstFailureTime: '2025-08-26 10:30:00',
      lastFailureTime: '2025-08-26 11:45:00',
      status: 'quarantined',
      tags: ['timeout', 'offline-agent'],
    },
    {
      id: 'dl-002',
      originalMessageId: 'msg-def456',
      subject: 'agents.analytics.report',
      payload: { reportId: 'report-invalid', format: 'unknown' },
      sender: 'report-agent',
      receiver: 'analytics-agent',
      failureReason: 'Invalid message format - unknown report format',
      failureCount: 5,
      firstFailureTime: '2025-08-26 09:15:00',
      lastFailureTime: '2025-08-26 12:00:00',
      status: 'reprocessing',
      tags: ['format-error', 'validation-failed'],
    },
  ])

  const [circuitBreakers, setCircuitBreakers] = useState<CircuitBreaker[]>([
    {
      id: 'cb-001',
      name: '分析服务熔断器',
      description: '保护分析服务免受过载',
      subject: 'agents.analytics.>',
      state: 'closed',
      failureThreshold: 5,
      recoveryTimeout: 60000,
      requestTimeout: 10000,
      currentFailures: 2,
      totalRequests: 1567,
      failureRate: 0.13,
      enabled: true,
    },
    {
      id: 'cb-002',
      name: '外部API熔断器',
      description: '保护外部API调用',
      subject: 'agents.external.>',
      state: 'half-open',
      failureThreshold: 3,
      recoveryTimeout: 30000,
      requestTimeout: 5000,
      currentFailures: 3,
      totalRequests: 234,
      failureRate: 1.28,
      lastFailureTime: '2025-08-26 12:30:00',
      nextRetryTime: '2025-08-26 12:46:00',
      enabled: true,
    },
  ])

  const guaranteeColumns = [
    {
      title: '传递保证',
      key: 'guarantee',
      width: 300,
      render: (record: DeliveryGuarantee) => (
        <div>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              marginBottom: '4px',
            }}
          >
            <Badge
              status={
                record.status === 'active'
                  ? 'success'
                  : record.status === 'error'
                    ? 'error'
                    : 'default'
              }
            />
            <Text strong style={{ marginLeft: '8px', fontSize: '13px' }}>
              {record.name}
            </Text>
            <Tag
              color={
                record.level === 'exactly-once'
                  ? 'red'
                  : record.level === 'at-least-once'
                    ? 'orange'
                    : 'blue'
              }
              style={{ marginLeft: '8px', fontSize: '10px' }}
            >
              {record.level === 'exactly-once'
                ? '精确一次'
                : record.level === 'at-least-once'
                  ? '至少一次'
                  : '最多一次'}
            </Tag>
          </div>
          <Text type="secondary" style={{ fontSize: '11px' }}>
            {record.description}
          </Text>
          <div style={{ marginTop: '4px' }}>
            {record.subjects.slice(0, 2).map((subject, index) => (
              <Tag
                key={index}
                style={{ fontSize: '10px', marginBottom: '2px' }}
              >
                {subject}
              </Tag>
            ))}
            {record.subjects.length > 2 && (
              <Text type="secondary" style={{ fontSize: '10px' }}>
                +{record.subjects.length - 2}
              </Text>
            )}
          </div>
        </div>
      ),
    },
    {
      title: '配置参数',
      key: 'config',
      width: 200,
      render: (record: DeliveryGuarantee) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>
              ACK超时: {record.config.acknowledgmentTimeout}ms
            </Text>
          </div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>
              最大重试: {record.config.maxRetries}
            </Text>
          </div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>
              重试延迟: {record.config.retryDelay}ms
            </Text>
          </div>
          <div>
            <Badge
              status={record.config.persistentStorage ? 'success' : 'default'}
              text={record.config.persistentStorage ? '持久存储' : '内存存储'}
            />
          </div>
        </div>
      ),
    },
    {
      title: '性能指标',
      key: 'metrics',
      width: 150,
      render: (record: DeliveryGuarantee) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>
              处理: {record.metrics.messagesProcessed.toLocaleString()}
            </Text>
          </div>
          <div style={{ marginBottom: '2px' }}>
            <Text
              style={{
                fontSize: '11px',
                color: record.metrics.successRate > 99 ? '#52c41a' : '#ff4d4f',
              }}
            >
              成功率: {record.metrics.successRate}%
            </Text>
          </div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>
              重试: {record.metrics.retriesPerformed}
            </Text>
          </div>
          <div>
            <Text style={{ fontSize: '11px' }}>
              重复: {record.metrics.duplicatesDetected}
            </Text>
          </div>
        </div>
      ),
    },
    {
      title: '最后更新',
      dataIndex: 'lastUpdated',
      key: 'lastUpdated',
      width: 120,
      render: (text: string) => (
        <Text style={{ fontSize: '11px' }}>{text}</Text>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      width: 120,
      render: (record: DeliveryGuarantee) => (
        <Space>
          <Tooltip title="查看详情">
            <Button
              type="text"
              size="small"
              icon={<EyeOutlined />}
              onClick={() => handleViewGuarantee(record)}
            />
          </Tooltip>
          <Tooltip title="编辑配置">
            <Button
              type="text"
              size="small"
              icon={<EditOutlined />}
              onClick={() => handleEditGuarantee(record)}
            />
          </Tooltip>
          <Switch
            size="small"
            checked={record.enabled}
            onChange={checked => handleToggleGuarantee(record.id, checked)}
          />
        </Space>
      ),
    },
  ]

  const backupColumns = [
    {
      title: '备份任务',
      key: 'backup',
      render: (record: MessageBackup) => (
        <div>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              marginBottom: '4px',
            }}
          >
            <Badge
              status={
                record.status === 'completed'
                  ? 'success'
                  : record.status === 'running'
                    ? 'processing'
                    : record.status === 'failed'
                      ? 'error'
                      : 'default'
              }
            />
            <Text strong style={{ marginLeft: '8px' }}>
              {record.name}
            </Text>
            <Tag
              color={
                record.type === 'full'
                  ? 'red'
                  : record.type === 'incremental'
                    ? 'orange'
                    : 'blue'
              }
              style={{ marginLeft: '8px', fontSize: '10px' }}
            >
              {record.type === 'full'
                ? '全量'
                : record.type === 'incremental'
                  ? '增量'
                  : '流专用'}
            </Tag>
          </div>
          <Text type="secondary" style={{ fontSize: '11px' }}>
            {record.description}
          </Text>
        </div>
      ),
    },
    {
      title: '调度配置',
      key: 'schedule',
      render: (record: MessageBackup) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>
              <ClockCircleOutlined style={{ marginRight: '4px' }} />
              {record.schedule}
            </Text>
          </div>
          <div>
            <Text style={{ fontSize: '11px' }}>保留: {record.retention}天</Text>
          </div>
        </div>
      ),
    },
    {
      title: '备份统计',
      key: 'stats',
      render: (record: MessageBackup) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>
              大小: {(record.backupSize / 1024 / 1024).toFixed(1)}MB
            </Text>
          </div>
          <div>
            <Text style={{ fontSize: '11px' }}>
              消息: {record.messagesCount.toLocaleString()}
            </Text>
          </div>
        </div>
      ),
    },
    {
      title: '时间信息',
      key: 'timing',
      render: (record: MessageBackup) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>上次: {record.lastRun}</Text>
          </div>
          <div>
            <Text style={{ fontSize: '11px' }}>下次: {record.nextRun}</Text>
          </div>
        </div>
      ),
    },
    {
      title: '特性',
      key: 'features',
      render: (record: MessageBackup) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Badge
              status={record.compression ? 'success' : 'default'}
              text="压缩"
            />
          </div>
          <div>
            <Badge
              status={record.encryption ? 'success' : 'default'}
              text="加密"
            />
          </div>
        </div>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: MessageBackup) => (
        <Space>
          <Button type="text" size="small" icon={<EyeOutlined />} />
          <Button type="text" size="small" icon={<PlayCircleOutlined />} />
          <Button type="text" size="small" icon={<EditOutlined />} />
        </Space>
      ),
    },
  ]

  const deadLetterColumns = [
    {
      title: '消息信息',
      key: 'message',
      render: (record: DeadLetterMessage) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text strong style={{ fontSize: '12px' }}>
              {record.subject}
            </Text>
          </div>
          <Text code style={{ fontSize: '11px' }}>
            {record.originalMessageId}
          </Text>
          <div style={{ marginTop: '4px' }}>
            {record.tags.map((tag, index) => (
              <Tag
                key={index}
                style={{ fontSize: '10px', marginBottom: '2px' }}
              >
                {tag}
              </Tag>
            ))}
          </div>
        </div>
      ),
    },
    {
      title: '通信方',
      key: 'parties',
      render: (record: DeadLetterMessage) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>发送: {record.sender}</Text>
          </div>
          <div>
            <Text style={{ fontSize: '11px' }}>接收: {record.receiver}</Text>
          </div>
        </div>
      ),
    },
    {
      title: '失败信息',
      key: 'failure',
      render: (record: DeadLetterMessage) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>
              次数: {record.failureCount}
            </Text>
          </div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>
              首次: {record.firstFailureTime}
            </Text>
          </div>
          <div>
            <Text style={{ fontSize: '11px' }}>
              最后: {record.lastFailureTime}
            </Text>
          </div>
        </div>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Badge
          status={
            status === 'quarantined'
              ? 'error'
              : status === 'reprocessing'
                ? 'processing'
                : 'default'
          }
          text={
            status === 'quarantined'
              ? '隔离'
              : status === 'reprocessing'
                ? '重处理'
                : '丢弃'
          }
        />
      ),
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: DeadLetterMessage) => (
        <Space>
          <Button type="text" size="small" icon={<EyeOutlined />} />
          <Button type="text" size="small" icon={<RedoOutlined />} />
          <Button type="text" size="small" icon={<DeleteOutlined />} danger />
        </Space>
      ),
    },
  ]

  const circuitBreakerColumns = [
    {
      title: '熔断器',
      key: 'breaker',
      render: (record: CircuitBreaker) => (
        <div>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              marginBottom: '4px',
            }}
          >
            <Badge
              status={
                record.state === 'closed'
                  ? 'success'
                  : record.state === 'open'
                    ? 'error'
                    : 'warning'
              }
            />
            <Text strong style={{ marginLeft: '8px' }}>
              {record.name}
            </Text>
            <Tag
              color={
                record.state === 'closed'
                  ? 'green'
                  : record.state === 'open'
                    ? 'red'
                    : 'orange'
              }
              style={{ marginLeft: '8px', fontSize: '10px' }}
            >
              {record.state === 'closed'
                ? '关闭'
                : record.state === 'open'
                  ? '开启'
                  : '半开'}
            </Tag>
          </div>
          <Text type="secondary" style={{ fontSize: '11px' }}>
            {record.description}
          </Text>
          <br />
          <Text code style={{ fontSize: '11px' }}>
            {record.subject}
          </Text>
        </div>
      ),
    },
    {
      title: '阈值配置',
      key: 'config',
      render: (record: CircuitBreaker) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>
              失败阈值: {record.failureThreshold}
            </Text>
          </div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>
              恢复超时: {record.recoveryTimeout / 1000}s
            </Text>
          </div>
          <div>
            <Text style={{ fontSize: '11px' }}>
              请求超时: {record.requestTimeout / 1000}s
            </Text>
          </div>
        </div>
      ),
    },
    {
      title: '当前状态',
      key: 'status',
      render: (record: CircuitBreaker) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>
              失败: {record.currentFailures}/{record.failureThreshold}
            </Text>
          </div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>
              总请求: {record.totalRequests}
            </Text>
          </div>
          <div>
            <Text
              style={{
                fontSize: '11px',
                color: record.failureRate > 1 ? '#ff4d4f' : '#52c41a',
              }}
            >
              失败率: {record.failureRate.toFixed(2)}%
            </Text>
          </div>
        </div>
      ),
    },
    {
      title: '时间信息',
      key: 'timing',
      render: (record: CircuitBreaker) => (
        <div>
          {record.lastFailureTime && (
            <div style={{ marginBottom: '2px' }}>
              <Text style={{ fontSize: '11px' }}>
                上次失败: {record.lastFailureTime}
              </Text>
            </div>
          )}
          {record.nextRetryTime && (
            <div>
              <Text style={{ fontSize: '11px' }}>
                下次重试: {record.nextRetryTime}
              </Text>
            </div>
          )}
        </div>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: CircuitBreaker) => (
        <Space>
          <Button type="text" size="small" icon={<MonitorOutlined />} />
          <Button type="text" size="small" icon={<SettingOutlined />} />
          <Switch size="small" checked={record.enabled} />
        </Space>
      ),
    },
  ]

  const handleViewGuarantee = (guarantee: DeliveryGuarantee) => {
    Modal.info({
      title: '传递保证详情',
      width: 800,
      content: (
        <div>
          <Descriptions column={2} size="small">
            <Descriptions.Item label="保证ID">{guarantee.id}</Descriptions.Item>
            <Descriptions.Item label="名称">{guarantee.name}</Descriptions.Item>
            <Descriptions.Item label="保证级别">
              <Tag
                color={
                  guarantee.level === 'exactly-once'
                    ? 'red'
                    : guarantee.level === 'at-least-once'
                      ? 'orange'
                      : 'blue'
                }
              >
                {guarantee.level === 'exactly-once'
                  ? '精确一次'
                  : guarantee.level === 'at-least-once'
                    ? '至少一次'
                    : '最多一次'}
              </Tag>
            </Descriptions.Item>
            <Descriptions.Item label="状态">
              <Badge
                status={guarantee.status === 'active' ? 'success' : 'error'}
                text={guarantee.status}
              />
            </Descriptions.Item>
            <Descriptions.Item label="描述" span={2}>
              {guarantee.description}
            </Descriptions.Item>
          </Descriptions>

          <Divider>应用主题</Divider>
          <div>
            {guarantee.subjects.map((subject, index) => (
              <Tag key={index} style={{ marginBottom: '4px' }}>
                {subject}
              </Tag>
            ))}
          </div>

          <Divider>配置参数</Divider>
          <Row gutter={[16, 8]}>
            <Col span={12}>
              <Text strong>确认超时: </Text>
              <Text>{guarantee.config.acknowledgmentTimeout}ms</Text>
            </Col>
            <Col span={12}>
              <Text strong>最大重试: </Text>
              <Text>{guarantee.config.maxRetries}</Text>
            </Col>
            <Col span={12}>
              <Text strong>重试延迟: </Text>
              <Text>{guarantee.config.retryDelay}ms</Text>
            </Col>
            <Col span={12}>
              <Text strong>重复检测窗口: </Text>
              <Text>{guarantee.config.duplicateDetectionWindow}ms</Text>
            </Col>
            <Col span={12}>
              <Text strong>持久存储: </Text>
              <Badge
                status={
                  guarantee.config.persistentStorage ? 'success' : 'default'
                }
                text={guarantee.config.persistentStorage ? '启用' : '禁用'}
              />
            </Col>
          </Row>

          <Divider>性能指标</Divider>
          <Row gutter={[16, 8]}>
            <Col span={6}>
              <Statistic
                title="处理消息"
                value={guarantee.metrics.messagesProcessed}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="成功率"
                value={guarantee.metrics.successRate}
                suffix="%"
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="重复检测"
                value={guarantee.metrics.duplicatesDetected}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="重试执行"
                value={guarantee.metrics.retriesPerformed}
              />
            </Col>
          </Row>
        </div>
      ),
    })
  }

  const handleEditGuarantee = (guarantee: DeliveryGuarantee) => {
    setSelectedGuarantee(guarantee)
    form.setFieldsValue(guarantee)
    setGuaranteeModalVisible(true)
  }

  const handleToggleGuarantee = (guaranteeId: string, enabled: boolean) => {
    setDeliveryGuarantees(prev =>
      prev.map(g => (g.id === guaranteeId ? { ...g, enabled } : g))
    )
    notification.success({
      message: enabled ? '保证已启用' : '保证已禁用',
      description: '传递保证状态已更新',
    })
  }

  const refreshData = () => {
    setLoading(true)
    notification.success({
      message: '刷新成功',
      description: '已请求最新可靠性数据',
    })
    setLoading(false)
  }

  const getReliabilityColor = (value: number) => {
    if (value >= 99) return '#52c41a'
    if (value >= 95) return '#faad14'
    return '#ff4d4f'
  }

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <SafetyCertificateOutlined
            style={{ marginRight: '12px', color: '#1890ff' }}
          />
          消息可靠性管理
        </Title>
        <Paragraph>
          智能体消息传递可靠性保证机制，包括传递保证、消息备份、死信处理、熔断保护等功能
        </Paragraph>
      </div>

      {/* 可靠性状态告警 */}
      {reliabilityMetrics.deliveryRate < 99 && (
        <Alert
          message="可靠性警告"
          description={`消息传递成功率为 ${reliabilityMetrics.deliveryRate.toFixed(2)}%，低于预期阈值 99%`}
          type="warning"
          showIcon
          style={{ marginBottom: '24px' }}
        />
      )}

      {reliabilityMetrics.deadLetterCount > 20 && (
        <Alert
          message="死信队列警告"
          description={`死信队列中有 ${reliabilityMetrics.deadLetterCount} 条消息待处理，建议及时处理`}
          type="error"
          showIcon
          action={
            <Button
              size="small"
              onClick={() => setDeadLetterDrawerVisible(true)}
            >
              查看详情
            </Button>
          }
          style={{ marginBottom: '24px' }}
        />
      )}

      {/* 可靠性指标概览 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={12} sm={6}>
          <Card>
            <Statistic
              title="传递成功率"
              value={reliabilityMetrics.deliveryRate}
              precision={2}
              suffix="%"
              valueStyle={{
                color: getReliabilityColor(reliabilityMetrics.deliveryRate),
              }}
              prefix={<CheckCircleOutlined />}
            />
            <div style={{ marginTop: '8px' }}>
              <Progress
                percent={reliabilityMetrics.deliveryRate}
                size="small"
                showInfo={false}
                status={
                  reliabilityMetrics.deliveryRate >= 99
                    ? 'success'
                    : 'exception'
                }
              />
            </div>
          </Card>
        </Col>

        <Col xs={12} sm={6}>
          <Card>
            <Statistic
              title="消息完整性"
              value={reliabilityMetrics.messageIntegrity}
              precision={2}
              suffix="%"
              valueStyle={{
                color: getReliabilityColor(reliabilityMetrics.messageIntegrity),
              }}
              prefix={<SecurityScanOutlined />}
            />
          </Card>
        </Col>

        <Col xs={12} sm={6}>
          <Card>
            <Statistic
              title="平均重试次数"
              value={reliabilityMetrics.averageRetries}
              precision={1}
              valueStyle={{
                color:
                  reliabilityMetrics.averageRetries < 1 ? '#52c41a' : '#ff4d4f',
              }}
              prefix={<SyncOutlined />}
            />
          </Card>
        </Col>

        <Col xs={12} sm={6}>
          <Card>
            <Statistic
              title="死信消息"
              value={reliabilityMetrics.deadLetterCount}
              valueStyle={{
                color:
                  reliabilityMetrics.deadLetterCount < 10
                    ? '#52c41a'
                    : '#ff4d4f',
              }}
              prefix={<AlertOutlined />}
            />
            <div style={{ marginTop: '8px' }}>
              <Button
                type="link"
                size="small"
                onClick={() => setDeadLetterDrawerVisible(true)}
              >
                查看详情
              </Button>
            </div>
          </Card>
        </Col>
      </Row>

      {/* 主管理界面 */}
      <Card>
        <div
          style={{
            marginBottom: '16px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <Space>
            <Button
              type="primary"
              icon={<SafetyCertificateOutlined />}
              onClick={() => {
                setSelectedGuarantee(null)
                form.resetFields()
                setGuaranteeModalVisible(true)
              }}
            >
              创建传递保证
            </Button>
            <Button
              icon={<ReloadOutlined />}
              loading={loading}
              onClick={refreshData}
            >
              刷新状态
            </Button>
          </Space>

          <Space>
            <Button
              icon={<DatabaseOutlined />}
              onClick={() => setBackupModalVisible(true)}
            >
              备份管理
            </Button>
            <Button
              icon={<AlertOutlined />}
              onClick={() => setDeadLetterDrawerVisible(true)}
            >
              死信处理
            </Button>
            <Button icon={<SettingOutlined />}>可靠性配置</Button>
          </Space>
        </div>

        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane
            tab="传递保证"
            key="guarantees"
            icon={<CheckCircleOutlined />}
          >
            <Table
              columns={guaranteeColumns}
              dataSource={deliveryGuarantees}
              rowKey="id"
              size="small"
              pagination={false}
              scroll={{ x: 1000 }}
            />
          </TabPane>

          <TabPane tab="消息备份" key="backups" icon={<DatabaseOutlined />}>
            <div style={{ marginBottom: '16px' }}>
              <Space>
                <Button type="primary" icon={<DatabaseOutlined />}>
                  创建备份任务
                </Button>
                <Text type="secondary">
                  备份状态:
                  <Badge
                    status={
                      reliabilityMetrics.backupStatus === 'healthy'
                        ? 'success'
                        : 'error'
                    }
                    text={reliabilityMetrics.backupStatus}
                    style={{ marginLeft: '8px' }}
                  />
                </Text>
              </Space>
            </div>
            <Table
              columns={backupColumns}
              dataSource={messageBackups}
              rowKey="id"
              size="small"
              pagination={false}
            />
          </TabPane>

          <TabPane
            tab="熔断保护"
            key="circuit-breakers"
            icon={<RestOutlined />}
          >
            <div style={{ marginBottom: '16px' }}>
              <Button type="primary" icon={<RestOutlined />}>
                创建熔断器
              </Button>
            </div>
            <Table
              columns={circuitBreakerColumns}
              dataSource={circuitBreakers}
              rowKey="id"
              size="small"
              pagination={false}
            />
          </TabPane>

          <TabPane tab="质量监控" key="quality" icon={<MonitorOutlined />}>
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Card title="传递质量趋势" size="small">
                  <div style={{ textAlign: 'center', padding: '40px 0' }}>
                    <Text type="secondary">传递质量趋势图表开发中...</Text>
                  </div>
                </Card>
              </Col>
              <Col span={12}>
                <Card title="重试分析" size="small">
                  <div>
                    <div style={{ marginBottom: '8px' }}>
                      <Text strong>重试原因分布：</Text>
                    </div>
                    <div style={{ marginBottom: '4px' }}>
                      <Tag color="red">网络超时: 45%</Tag>
                    </div>
                    <div style={{ marginBottom: '4px' }}>
                      <Tag color="orange">服务忙碌: 30%</Tag>
                    </div>
                    <div style={{ marginBottom: '4px' }}>
                      <Tag color="yellow">消息格式错误: 15%</Tag>
                    </div>
                    <div>
                      <Tag color="blue">其他原因: 10%</Tag>
                    </div>
                  </div>
                </Card>
              </Col>

              <Col span={24}>
                <Card title="可靠性健康检查" size="small">
                  <Steps current={2} status="process" size="small">
                    <Step
                      title="消息传递"
                      description={`${reliabilityMetrics.deliveryRate.toFixed(1)}% 成功率`}
                      status={
                        reliabilityMetrics.deliveryRate >= 99
                          ? 'finish'
                          : 'error'
                      }
                    />
                    <Step
                      title="完整性验证"
                      description={`${reliabilityMetrics.messageIntegrity.toFixed(2)}% 完整性`}
                      status={
                        reliabilityMetrics.messageIntegrity >= 99.9
                          ? 'finish'
                          : 'error'
                      }
                    />
                    <Step
                      title="顺序保证"
                      description={`${reliabilityMetrics.orderingAccuracy.toFixed(1)}% 顺序准确`}
                      status="process"
                    />
                    <Step
                      title="备份恢复"
                      description={`${reliabilityMetrics.backupStatus} 状态`}
                      status={
                        reliabilityMetrics.backupStatus === 'healthy'
                          ? 'finish'
                          : 'wait'
                      }
                    />
                  </Steps>
                </Card>
              </Col>
            </Row>
          </TabPane>
        </Tabs>
      </Card>

      {/* 死信消息抽屉 */}
      <Drawer
        title="死信队列管理"
        placement="right"
        width={1200}
        visible={deadLetterDrawerVisible}
        onClose={() => setDeadLetterDrawerVisible(false)}
      >
        <div style={{ marginBottom: '16px' }}>
          <Alert
            message={`当前死信队列中有 ${deadLetterMessages.length} 条消息`}
            description="这些消息因各种原因无法正常处理，需要人工介入处理"
            type="info"
            showIcon
          />
        </div>

        <Table
          columns={deadLetterColumns}
          dataSource={deadLetterMessages}
          rowKey="id"
          size="small"
          pagination={{ pageSize: 10 }}
        />
      </Drawer>
    </div>
  )
}

export default MessageReliabilityManagementPage
