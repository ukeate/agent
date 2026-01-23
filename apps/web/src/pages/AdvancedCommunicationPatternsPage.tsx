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
  Collapse,
  Tree,
  InputNumber,
  Upload,
  Descriptions,
  Empty,
} from 'antd'
import {
  ShareAltOutlined,
  TeamOutlined,
  BranchesOutlined,
  NodeIndexOutlined,
  GlobalOutlined,
  RocketOutlined,
  ThunderboltOutlined,
  SendOutlined,
  ReceiveOutlined,
  SwapOutlined,
  MessageOutlined,
  ApiOutlined,
  ClusterOutlined,
  ShareAltOutlined as NetworkOutlined,
  EyeOutlined,
  EditOutlined,
  DeleteOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  SettingOutlined,
  ReloadOutlined,
  PlusOutlined,
  MonitorOutlined,
  LineChartOutlined,
  PartitionOutlined,
  RouterOutlined,
  CloudServerOutlined,
  ForkOutlined,
  MergeCellsOutlined,
  FilterOutlined,
  SortAscendingOutlined,
  ApartmentOutlined,
  RadarChartOutlined,
  SearchOutlined,
} from '@ant-design/icons'

const { Title, Text, Paragraph } = Typography
const { Option } = Select
const { TextArea } = Input
const { TabPane } = Tabs
const { Panel } = Collapse
const { TreeNode } = Tree

interface CommunicationPattern {
  id: string
  name: string
  type:
    | 'multicast'
    | 'broadcast'
    | 'anycast'
    | 'workflow'
    | 'pipeline'
    | 'pubsub'
    | 'scatter-gather'
    | 'routing'
  description: string
  subjects: string[]
  participants: string[]
  config: {
    routingKey?: string
    partitionStrategy?: 'round-robin' | 'hash' | 'sticky' | 'random'
    loadBalancing?: boolean
    failover?: boolean
    timeout?: number
    batchSize?: number
    parallelism?: number
  }
  metrics: {
    messagesProcessed: number
    averageLatency: number
    throughput: number
    successRate: number
    participantCount: number
  }
  status: 'active' | 'inactive' | 'error'
  enabled: boolean
  createdAt: string
  lastUsed: string
}

interface WorkflowDefinition {
  id: string
  name: string
  description: string
  steps: WorkflowStep[]
  triggers: string[]
  variables: Record<string, any>
  config: {
    timeout: number
    retryPolicy: 'none' | 'linear' | 'exponential'
    maxRetries: number
    errorHandling: 'fail-fast' | 'continue' | 'compensate'
  }
  metrics: {
    executions: number
    successRate: number
    averageExecutionTime: number
    failurePoints: Record<string, number>
  }
  status: 'draft' | 'active' | 'paused' | 'archived'
  version: string
}

interface WorkflowStep {
  id: string
  name: string
  type: 'message' | 'condition' | 'parallel' | 'loop' | 'wait' | 'transform'
  config: {
    agent?: string
    subject?: string
    condition?: string
    timeout?: number
    maxIterations?: number
  }
  position: { x: number; y: number }
  connections: string[]
}

interface RoutingRule {
  id: string
  name: string
  description: string
  priority: number
  condition: string
  sourcePattern: string
  targetRoutes: {
    agent: string
    weight: number
    condition?: string
  }[]
  transformations: {
    headerMappings: Record<string, string>
    payloadTransform?: string
    filterExpression?: string
  }
  metrics: {
    matchCount: number
    routeDistribution: Record<string, number>
    averageLatency: number
  }
  enabled: boolean
  createdAt: string
}

interface StreamingPipeline {
  id: string
  name: string
  description: string
  stages: PipelineStage[]
  config: {
    bufferSize: number
    batchTimeout: number
    backpressureStrategy: 'block' | 'drop' | 'buffer'
    parallelism: number
  }
  metrics: {
    throughput: number
    backpressureEvents: number
    processedMessages: number
    droppedMessages: number
    averageLatency: number
  }
  status: 'running' | 'stopped' | 'error'
  createdAt: string
}

interface PipelineStage {
  id: string
  name: string
  type: 'filter' | 'transform' | 'enrich' | 'aggregate' | 'route'
  config: any
  metrics: {
    inputCount: number
    outputCount: number
    processingTime: number
    errorCount: number
  }
}

const AdvancedCommunicationPatternsPage: React.FC = () => {
  const [form] = Form.useForm()
  const [activeTab, setActiveTab] = useState('patterns')
  const [loading, setLoading] = useState(false)
  const [patternModalVisible, setPatternModalVisible] = useState(false)
  const [workflowModalVisible, setWorkflowModalVisible] = useState(false)
  const [routingModalVisible, setRoutingModalVisible] = useState(false)
  const [pipelineModalVisible, setPipelineModalVisible] = useState(false)
  const [selectedPattern, setSelectedPattern] =
    useState<CommunicationPattern | null>(null)

  const [communicationPatterns, setCommunicationPatterns] = useState<
    CommunicationPattern[]
  >([
    {
      id: 'pattern-001',
      name: '任务分发多播',
      type: 'multicast',
      description: '将任务消息同时发送给多个工作智能体',
      subjects: ['agents.tasks.distribute'],
      participants: ['worker-agent-01', 'worker-agent-02', 'worker-agent-03'],
      config: {
        loadBalancing: true,
        failover: true,
        timeout: 10000,
        batchSize: 10,
      },
      metrics: {
        messagesProcessed: 5234,
        averageLatency: 156,
        throughput: 78.5,
        successRate: 98.2,
        participantCount: 3,
      },
      status: 'active',
      enabled: true,
      createdAt: '2025-08-20 10:30:00',
      lastUsed: '2025-08-26 12:45:00',
    },
    {
      id: 'pattern-002',
      name: '系统广播通知',
      type: 'broadcast',
      description: '向所有在线智能体广播系统通知',
      subjects: ['system.broadcast.>', 'agents.announcements.>'],
      participants: ['*'],
      config: {
        timeout: 5000,
        batchSize: 100,
      },
      metrics: {
        messagesProcessed: 1247,
        averageLatency: 89,
        throughput: 145.2,
        successRate: 99.8,
        participantCount: 12,
      },
      status: 'active',
      enabled: true,
      createdAt: '2025-08-18 14:20:00',
      lastUsed: '2025-08-26 11:30:00',
    },
    {
      id: 'pattern-003',
      name: '负载均衡任播',
      type: 'anycast',
      description: '智能负载均衡，选择最优智能体处理请求',
      subjects: ['agents.services.>'],
      participants: [
        'service-agent-01',
        'service-agent-02',
        'service-agent-03',
      ],
      config: {
        partitionStrategy: 'round-robin',
        loadBalancing: true,
        failover: true,
        timeout: 8000,
      },
      metrics: {
        messagesProcessed: 8945,
        averageLatency: 234,
        throughput: 56.8,
        successRate: 97.5,
        participantCount: 3,
      },
      status: 'active',
      enabled: true,
      createdAt: '2025-08-19 09:15:00',
      lastUsed: '2025-08-26 12:30:00',
    },
    {
      id: 'pattern-004',
      name: '分散收集模式',
      type: 'scatter-gather',
      description: '分散任务到多个智能体，收集汇总结果',
      subjects: ['agents.analytics.scatter'],
      participants: [
        'analytics-agent-01',
        'analytics-agent-02',
        'analytics-agent-03',
      ],
      config: {
        timeout: 15000,
        parallelism: 3,
        batchSize: 5,
      },
      metrics: {
        messagesProcessed: 2156,
        averageLatency: 2850,
        throughput: 23.4,
        successRate: 94.8,
        participantCount: 3,
      },
      status: 'active',
      enabled: true,
      createdAt: '2025-08-21 16:45:00',
      lastUsed: '2025-08-26 10:15:00',
    },
  ])

  const [workflowDefinitions, setWorkflowDefinitions] = useState<
    WorkflowDefinition[]
  >([
    {
      id: 'workflow-001',
      name: '订单处理工作流',
      description: '完整的订单处理流程，包括验证、支付、库存、发货等步骤',
      steps: [
        {
          id: 'step-001',
          name: '订单验证',
          type: 'message',
          config: { agent: 'validation-agent', subject: 'orders.validate' },
          position: { x: 100, y: 100 },
          connections: ['step-002'],
        },
        {
          id: 'step-002',
          name: '支付处理',
          type: 'message',
          config: { agent: 'payment-agent', subject: 'payments.process' },
          position: { x: 300, y: 100 },
          connections: ['step-003'],
        },
        {
          id: 'step-003',
          name: '库存检查',
          type: 'message',
          config: { agent: 'inventory-agent', subject: 'inventory.check' },
          position: { x: 500, y: 100 },
          connections: ['step-004'],
        },
      ],
      triggers: ['orders.new', 'orders.updated'],
      variables: { timeout: 30000, maxRetries: 3 },
      config: {
        timeout: 300000,
        retryPolicy: 'exponential',
        maxRetries: 3,
        errorHandling: 'compensate',
      },
      metrics: {
        executions: 1456,
        successRate: 96.8,
        averageExecutionTime: 45000,
        failurePoints: { 'step-002': 32, 'step-003': 18 },
      },
      status: 'active',
      version: '1.2.0',
    },
  ])

  const [routingRules, setRoutingRules] = useState<RoutingRule[]>([
    {
      id: 'routing-001',
      name: '基于优先级路由',
      description: '根据消息优先级路由到不同处理队列',
      priority: 100,
      condition: 'message.priority >= 8',
      sourcePattern: 'agents.tasks.>',
      targetRoutes: [
        { agent: 'high-priority-worker', weight: 100 },
        {
          agent: 'fallback-worker',
          weight: 0,
          condition: 'high-priority-worker.offline',
        },
      ],
      transformations: {
        headerMappings: { 'X-Priority': 'high' },
        filterExpression: 'payload.urgent == true',
      },
      metrics: {
        matchCount: 2456,
        routeDistribution: {
          'high-priority-worker': 2398,
          'fallback-worker': 58,
        },
        averageLatency: 123,
      },
      enabled: true,
      createdAt: '2025-08-20 11:30:00',
    },
    {
      id: 'routing-002',
      name: '地理位置路由',
      description: '基于用户地理位置路由到就近的处理节点',
      priority: 80,
      condition: 'message.headers.region != null',
      sourcePattern: 'users.requests.>',
      targetRoutes: [
        {
          agent: 'us-west-processor',
          weight: 50,
          condition: 'message.headers.region == "us-west"',
        },
        {
          agent: 'us-east-processor',
          weight: 30,
          condition: 'message.headers.region == "us-east"',
        },
        {
          agent: 'eu-processor',
          weight: 20,
          condition: 'message.headers.region == "eu"',
        },
      ],
      transformations: {
        headerMappings: { 'X-Region': 'message.headers.region' },
        payloadTransform: 'payload.region = message.headers.region',
      },
      metrics: {
        matchCount: 8945,
        routeDistribution: {
          'us-west-processor': 4472,
          'us-east-processor': 2684,
          'eu-processor': 1789,
        },
        averageLatency: 89,
      },
      enabled: true,
      createdAt: '2025-08-19 15:20:00',
    },
  ])

  const [streamingPipelines, setStreamingPipelines] = useState<
    StreamingPipeline[]
  >([
    {
      id: 'pipeline-001',
      name: '实时事件处理管道',
      description: '实时处理系统事件，包括过滤、转换、富化和路由',
      stages: [
        {
          id: 'stage-001',
          name: '事件过滤',
          type: 'filter',
          config: { condition: 'event.level >= "WARN"' },
          metrics: {
            inputCount: 15642,
            outputCount: 3456,
            processingTime: 2,
            errorCount: 0,
          },
        },
        {
          id: 'stage-002',
          name: '数据转换',
          type: 'transform',
          config: { mapping: { timestamp: 'ISO8601', level: 'uppercase' } },
          metrics: {
            inputCount: 3456,
            outputCount: 3456,
            processingTime: 5,
            errorCount: 12,
          },
        },
        {
          id: 'stage-003',
          name: '上下文富化',
          type: 'enrich',
          config: { lookup: 'user-context', keys: ['userId'] },
          metrics: {
            inputCount: 3456,
            outputCount: 3444,
            processingTime: 15,
            errorCount: 12,
          },
        },
      ],
      config: {
        bufferSize: 1000,
        batchTimeout: 100,
        backpressureStrategy: 'buffer',
        parallelism: 4,
      },
      metrics: {
        throughput: 1250.5,
        backpressureEvents: 23,
        processedMessages: 156420,
        droppedMessages: 45,
        averageLatency: 22,
      },
      status: 'running',
      createdAt: '2025-08-20 09:00:00',
    },
  ])

  const patternColumns = [
    {
      title: '通信模式',
      key: 'pattern',
      width: 300,
      render: (record: CommunicationPattern) => (
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
                record.type === 'multicast'
                  ? 'blue'
                  : record.type === 'broadcast'
                    ? 'green'
                    : record.type === 'anycast'
                      ? 'orange'
                      : record.type === 'workflow'
                        ? 'purple'
                        : record.type === 'pipeline'
                          ? 'red'
                          : record.type === 'pubsub'
                            ? 'cyan'
                            : record.type === 'scatter-gather'
                              ? 'magenta'
                              : 'gold'
              }
              style={{ marginLeft: '8px', fontSize: '10px' }}
            >
              {record.type === 'multicast'
                ? '多播'
                : record.type === 'broadcast'
                  ? '广播'
                  : record.type === 'anycast'
                    ? '任播'
                    : record.type === 'workflow'
                      ? '工作流'
                      : record.type === 'pipeline'
                        ? '管道'
                        : record.type === 'pubsub'
                          ? '发布订阅'
                          : record.type === 'scatter-gather'
                            ? '分散收集'
                            : '智能路由'}
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
      title: '参与者',
      key: 'participants',
      width: 150,
      render: (record: CommunicationPattern) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>
              数量: {record.metrics.participantCount}
            </Text>
          </div>
          <div>
            {record.participants.slice(0, 2).map((participant, index) => (
              <Tag
                key={index}
                style={{ fontSize: '10px', marginBottom: '2px' }}
              >
                {participant === '*' ? '全部' : participant}
              </Tag>
            ))}
            {record.participants.length > 2 && (
              <Text type="secondary" style={{ fontSize: '10px' }}>
                +{record.participants.length - 2}
              </Text>
            )}
          </div>
        </div>
      ),
    },
    {
      title: '性能指标',
      key: 'metrics',
      width: 150,
      render: (record: CommunicationPattern) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>
              吞吐: {record.metrics.throughput.toFixed(1)} msg/s
            </Text>
          </div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>
              延迟: {record.metrics.averageLatency}ms
            </Text>
          </div>
          <div>
            <Text
              style={{
                fontSize: '11px',
                color: record.metrics.successRate > 95 ? '#52c41a' : '#ff4d4f',
              }}
            >
              成功率: {record.metrics.successRate}%
            </Text>
          </div>
        </div>
      ),
    },
    {
      title: '配置',
      key: 'config',
      width: 120,
      render: (record: CommunicationPattern) => (
        <div>
          {record.config.loadBalancing && (
            <Tag color="blue" style={{ fontSize: '10px', marginBottom: '2px' }}>
              负载均衡
            </Tag>
          )}
          {record.config.failover && (
            <Tag
              color="green"
              style={{ fontSize: '10px', marginBottom: '2px' }}
            >
              故障转移
            </Tag>
          )}
          {record.config.partitionStrategy && (
            <Tag
              color="orange"
              style={{ fontSize: '10px', marginBottom: '2px' }}
            >
              {record.config.partitionStrategy}
            </Tag>
          )}
        </div>
      ),
    },
    {
      title: '最后使用',
      dataIndex: 'lastUsed',
      key: 'lastUsed',
      width: 120,
      render: (text: string) => (
        <Text style={{ fontSize: '11px' }}>{text}</Text>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      width: 120,
      render: (record: CommunicationPattern) => (
        <Space>
          <Tooltip title="查看详情">
            <Button
              type="text"
              size="small"
              icon={<EyeOutlined />}
              onClick={() => handleViewPattern(record)}
            />
          </Tooltip>
          <Tooltip title="编辑模式">
            <Button
              type="text"
              size="small"
              icon={<EditOutlined />}
              onClick={() => handleEditPattern(record)}
            />
          </Tooltip>
          <Switch
            size="small"
            checked={record.enabled}
            onChange={checked => handleTogglePattern(record.id, checked)}
          />
        </Space>
      ),
    },
  ]

  const workflowColumns = [
    {
      title: '工作流',
      key: 'workflow',
      render: (record: WorkflowDefinition) => (
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
                  : record.status === 'paused'
                    ? 'warning'
                    : record.status === 'draft'
                      ? 'processing'
                      : 'default'
              }
            />
            <Text strong style={{ marginLeft: '8px' }}>
              {record.name}
            </Text>
            <Tag color="blue" style={{ marginLeft: '8px', fontSize: '10px' }}>
              v{record.version}
            </Tag>
          </div>
          <Text type="secondary" style={{ fontSize: '11px' }}>
            {record.description}
          </Text>
        </div>
      ),
    },
    {
      title: '步骤',
      key: 'steps',
      render: (record: WorkflowDefinition) => (
        <div>
          <Text style={{ fontSize: '11px' }}>{record.steps.length} 个步骤</Text>
          <br />
          <Text style={{ fontSize: '11px' }}>
            触发器: {record.triggers.length}
          </Text>
        </div>
      ),
    },
    {
      title: '执行指标',
      key: 'metrics',
      render: (record: WorkflowDefinition) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>
              执行: {record.metrics.executions}
            </Text>
          </div>
          <div style={{ marginBottom: '2px' }}>
            <Text
              style={{
                fontSize: '11px',
                color: record.metrics.successRate > 95 ? '#52c41a' : '#ff4d4f',
              }}
            >
              成功率: {record.metrics.successRate}%
            </Text>
          </div>
          <div>
            <Text style={{ fontSize: '11px' }}>
              平均时长:{' '}
              {(record.metrics.averageExecutionTime / 1000).toFixed(1)}s
            </Text>
          </div>
        </div>
      ),
    },
    {
      title: '配置',
      key: 'config',
      render: (record: WorkflowDefinition) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>
              超时: {record.config.timeout / 1000}s
            </Text>
          </div>
          <div>
            <Tag color="orange" style={{ fontSize: '10px' }}>
              {record.config.retryPolicy}重试
            </Tag>
          </div>
        </div>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: WorkflowDefinition) => (
        <Space>
          <Button type="text" size="small" icon={<EyeOutlined />} />
          <Button type="text" size="small" icon={<EditOutlined />} />
          {record.status === 'active' ? (
            <Button type="text" size="small" icon={<PauseCircleOutlined />} />
          ) : (
            <Button type="text" size="small" icon={<PlayCircleOutlined />} />
          )}
        </Space>
      ),
    },
  ]

  const routingColumns = [
    {
      title: '路由规则',
      key: 'rule',
      render: (record: RoutingRule) => (
        <div>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              marginBottom: '4px',
            }}
          >
            <Badge status={record.enabled ? 'success' : 'default'} />
            <Text strong style={{ marginLeft: '8px' }}>
              {record.name}
            </Text>
            <Tag color="red" style={{ marginLeft: '8px', fontSize: '10px' }}>
              优先级: {record.priority}
            </Tag>
          </div>
          <Text type="secondary" style={{ fontSize: '11px' }}>
            {record.description}
          </Text>
        </div>
      ),
    },
    {
      title: '匹配条件',
      key: 'condition',
      render: (record: RoutingRule) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>源: {record.sourcePattern}</Text>
          </div>
          <div>
            <Text code style={{ fontSize: '10px' }}>
              {record.condition}
            </Text>
          </div>
        </div>
      ),
    },
    {
      title: '目标路由',
      key: 'targets',
      render: (record: RoutingRule) => (
        <div>
          {record.targetRoutes.map((route, index) => (
            <div key={index} style={{ marginBottom: '2px' }}>
              <Tag style={{ fontSize: '10px' }}>
                {route.agent} ({route.weight}%)
              </Tag>
            </div>
          ))}
        </div>
      ),
    },
    {
      title: '匹配统计',
      key: 'stats',
      render: (record: RoutingRule) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>
              匹配: {record.metrics.matchCount}
            </Text>
          </div>
          <div>
            <Text style={{ fontSize: '11px' }}>
              延迟: {record.metrics.averageLatency}ms
            </Text>
          </div>
        </div>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: RoutingRule) => (
        <Space>
          <Button type="text" size="small" icon={<EyeOutlined />} />
          <Button type="text" size="small" icon={<EditOutlined />} />
          <Switch size="small" checked={record.enabled} />
        </Space>
      ),
    },
  ]

  const pipelineColumns = [
    {
      title: '流处理管道',
      key: 'pipeline',
      render: (record: StreamingPipeline) => (
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
                record.status === 'running'
                  ? 'success'
                  : record.status === 'error'
                    ? 'error'
                    : 'default'
              }
            />
            <Text strong style={{ marginLeft: '8px' }}>
              {record.name}
            </Text>
          </div>
          <Text type="secondary" style={{ fontSize: '11px' }}>
            {record.description}
          </Text>
        </div>
      ),
    },
    {
      title: '处理阶段',
      key: 'stages',
      render: (record: StreamingPipeline) => (
        <div>
          <Text style={{ fontSize: '11px' }}>
            {record.stages.length} 个阶段
          </Text>
          <div style={{ marginTop: '4px' }}>
            {record.stages.slice(0, 3).map((stage, index) => (
              <Tag
                key={index}
                style={{ fontSize: '10px', marginBottom: '2px' }}
              >
                {stage.name}
              </Tag>
            ))}
          </div>
        </div>
      ),
    },
    {
      title: '吞吐量',
      key: 'throughput',
      render: (record: StreamingPipeline) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>
              {record.metrics.throughput.toFixed(1)} msg/s
            </Text>
          </div>
          <div>
            <Text style={{ fontSize: '11px' }}>
              延迟: {record.metrics.averageLatency}ms
            </Text>
          </div>
        </div>
      ),
    },
    {
      title: '背压状态',
      key: 'backpressure',
      render: (record: StreamingPipeline) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>
              事件: {record.metrics.backpressureEvents}
            </Text>
          </div>
          <div>
            <Text
              style={{
                fontSize: '11px',
                color:
                  record.metrics.droppedMessages > 0 ? '#ff4d4f' : '#52c41a',
              }}
            >
              丢弃: {record.metrics.droppedMessages}
            </Text>
          </div>
        </div>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: StreamingPipeline) => (
        <Space>
          <Button type="text" size="small" icon={<EyeOutlined />} />
          <Button type="text" size="small" icon={<LineChartOutlined />} />
          {record.status === 'running' ? (
            <Button type="text" size="small" icon={<StopOutlined />} />
          ) : (
            <Button type="text" size="small" icon={<PlayCircleOutlined />} />
          )}
        </Space>
      ),
    },
  ]

  const handleViewPattern = (pattern: CommunicationPattern) => {
    Modal.info({
      title: '通信模式详情',
      width: 800,
      content: (
        <div>
          <Descriptions column={2} size="small">
            <Descriptions.Item label="模式ID">{pattern.id}</Descriptions.Item>
            <Descriptions.Item label="模式名称">
              {pattern.name}
            </Descriptions.Item>
            <Descriptions.Item label="模式类型">
              <Tag color="blue">{pattern.type}</Tag>
            </Descriptions.Item>
            <Descriptions.Item label="状态">
              <Badge
                status={pattern.status === 'active' ? 'success' : 'error'}
                text={pattern.status}
              />
            </Descriptions.Item>
            <Descriptions.Item label="创建时间">
              {pattern.createdAt}
            </Descriptions.Item>
            <Descriptions.Item label="最后使用">
              {pattern.lastUsed}
            </Descriptions.Item>
            <Descriptions.Item label="描述" span={2}>
              {pattern.description}
            </Descriptions.Item>
          </Descriptions>

          <Divider>应用主题</Divider>
          <div>
            {pattern.subjects.map((subject, index) => (
              <Tag key={index} style={{ marginBottom: '4px' }}>
                {subject}
              </Tag>
            ))}
          </div>

          <Divider>参与智能体</Divider>
          <div>
            {pattern.participants.map((participant, index) => (
              <Tag key={index} color="blue" style={{ marginBottom: '4px' }}>
                {participant === '*' ? '全部智能体' : participant}
              </Tag>
            ))}
          </div>

          <Divider>配置参数</Divider>
          <Row gutter={[16, 8]}>
            {pattern.config.timeout && (
              <Col span={12}>
                <Text strong>超时时间: </Text>
                <Text>{pattern.config.timeout}ms</Text>
              </Col>
            )}
            {pattern.config.batchSize && (
              <Col span={12}>
                <Text strong>批处理大小: </Text>
                <Text>{pattern.config.batchSize}</Text>
              </Col>
            )}
            {pattern.config.partitionStrategy && (
              <Col span={12}>
                <Text strong>分区策略: </Text>
                <Text>{pattern.config.partitionStrategy}</Text>
              </Col>
            )}
            <Col span={12}>
              <Text strong>负载均衡: </Text>
              <Badge
                status={pattern.config.loadBalancing ? 'success' : 'default'}
                text={pattern.config.loadBalancing ? '启用' : '禁用'}
              />
            </Col>
            <Col span={12}>
              <Text strong>故障转移: </Text>
              <Badge
                status={pattern.config.failover ? 'success' : 'default'}
                text={pattern.config.failover ? '启用' : '禁用'}
              />
            </Col>
          </Row>

          <Divider>性能指标</Divider>
          <Row gutter={[16, 16]}>
            <Col span={6}>
              <Statistic
                title="处理消息"
                value={pattern.metrics.messagesProcessed}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="平均延迟"
                value={pattern.metrics.averageLatency}
                suffix="ms"
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="吞吐量"
                value={pattern.metrics.throughput}
                suffix="msg/s"
                precision={1}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="成功率"
                value={pattern.metrics.successRate}
                suffix="%"
                precision={1}
              />
            </Col>
          </Row>
        </div>
      ),
    })
  }

  const handleEditPattern = (pattern: CommunicationPattern) => {
    setSelectedPattern(pattern)
    form.setFieldsValue(pattern)
    setPatternModalVisible(true)
  }

  const handleTogglePattern = (patternId: string, enabled: boolean) => {
    setCommunicationPatterns(prev =>
      prev.map(p => (p.id === patternId ? { ...p, enabled } : p))
    )
    notification.success({
      message: enabled ? '模式已启用' : '模式已禁用',
      description: '通信模式状态已更新',
    })
  }

  const refreshData = () => {
    setLoading(true)
    setTimeout(() => {
      notification.success({
        message: '刷新成功',
        description: '通信模式数据已更新',
      })
      setLoading(false)
    }, 1000)
  }

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <ShareAltOutlined style={{ marginRight: '12px', color: '#1890ff' }} />
          高级通信模式
        </Title>
        <Paragraph>
          智能体高级通信模式管理，包括多播、广播、工作流、流处理管道、智能路由等复杂通信场景
        </Paragraph>
      </div>

      {/* 模式统计概览 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={12} sm={6}>
          <Card>
            <Statistic
              title="活跃模式"
              value={
                communicationPatterns.filter(
                  p => p.enabled && p.status === 'active'
                ).length
              }
              suffix={`/ ${communicationPatterns.length}`}
              valueStyle={{ color: '#52c41a' }}
              prefix={<ShareAltOutlined />}
            />
          </Card>
        </Col>

        <Col xs={12} sm={6}>
          <Card>
            <Statistic
              title="总吞吐量"
              value={communicationPatterns.reduce(
                (sum, p) => sum + p.metrics.throughput,
                0
              )}
              precision={1}
              suffix="msg/s"
              valueStyle={{ color: '#1890ff' }}
              prefix={<ThunderboltOutlined />}
            />
          </Card>
        </Col>

        <Col xs={12} sm={6}>
          <Card>
            <Statistic
              title="平均成功率"
              value={
                communicationPatterns.reduce(
                  (sum, p) => sum + p.metrics.successRate,
                  0
                ) / communicationPatterns.length
              }
              precision={1}
              suffix="%"
              valueStyle={{ color: '#722ed1' }}
              prefix={<RocketOutlined />}
            />
          </Card>
        </Col>

        <Col xs={12} sm={6}>
          <Card>
            <Statistic
              title="参与智能体"
              value={
                new Set(communicationPatterns.flatMap(p => p.participants)).size
              }
              valueStyle={{ color: '#fa8c16' }}
              prefix={<TeamOutlined />}
            />
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
              icon={<PlusOutlined />}
              onClick={() => {
                setSelectedPattern(null)
                form.resetFields()
                setPatternModalVisible(true)
              }}
            >
              创建通信模式
            </Button>
            <Button
              icon={<ReloadOutlined />}
              loading={loading}
              onClick={refreshData}
            >
              刷新数据
            </Button>
          </Space>

          <Space>
            <Button
              icon={<BranchesOutlined />}
              onClick={() => setWorkflowModalVisible(true)}
            >
              工作流设计
            </Button>
            <Button
              icon={<RouterOutlined />}
              onClick={() => setRoutingModalVisible(true)}
            >
              智能路由
            </Button>
            <Button
              icon={<PartitionOutlined />}
              onClick={() => setPipelineModalVisible(true)}
            >
              流处理管道
            </Button>
          </Space>
        </div>

        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane tab="通信模式" key="patterns" icon={<ShareAltOutlined />}>
            <Table
              columns={patternColumns}
              dataSource={communicationPatterns}
              rowKey="id"
              size="small"
              pagination={{ pageSize: 10 }}
              scroll={{ x: 1200 }}
            />
          </TabPane>

          <TabPane tab="工作流定义" key="workflows" icon={<BranchesOutlined />}>
            <div style={{ marginBottom: '16px' }}>
              <Button type="primary" icon={<PlusOutlined />}>
                创建工作流
              </Button>
            </div>
            <Table
              columns={workflowColumns}
              dataSource={workflowDefinitions}
              rowKey="id"
              size="small"
              pagination={false}
            />
          </TabPane>

          <TabPane tab="路由规则" key="routing" icon={<RouterOutlined />}>
            <div style={{ marginBottom: '16px' }}>
              <Button type="primary" icon={<PlusOutlined />}>
                创建路由规则
              </Button>
            </div>
            <Table
              columns={routingColumns}
              dataSource={routingRules}
              rowKey="id"
              size="small"
              pagination={false}
            />
          </TabPane>

          <TabPane
            tab="流处理管道"
            key="pipelines"
            icon={<PartitionOutlined />}
          >
            <div style={{ marginBottom: '16px' }}>
              <Button type="primary" icon={<PlusOutlined />}>
                创建处理管道
              </Button>
            </div>
            <Table
              columns={pipelineColumns}
              dataSource={streamingPipelines}
              rowKey="id"
              size="small"
              pagination={false}
            />
          </TabPane>

          <TabPane tab="模式分析" key="analysis" icon={<RadarChartOutlined />}>
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Card title="模式使用分布" size="small">
                  <div>
                    <div style={{ marginBottom: '8px' }}>
                      <Text strong>按类型分布：</Text>
                    </div>
                    {[
                      'multicast',
                      'broadcast',
                      'anycast',
                      'scatter-gather',
                    ].map(type => {
                      const count = communicationPatterns.filter(
                        p => p.type === type
                      ).length
                      return count > 0 ? (
                        <div key={type} style={{ marginBottom: '4px' }}>
                          <Tag color="blue">
                            {type}: {count}
                          </Tag>
                        </div>
                      ) : null
                    })}
                  </div>
                </Card>
              </Col>

              <Col span={12}>
                <Card title="性能分析" size="small">
                  <div>
                    <div style={{ marginBottom: '8px' }}>
                      <Text strong>性能排名：</Text>
                    </div>
                    {communicationPatterns
                      .sort(
                        (a, b) => b.metrics.throughput - a.metrics.throughput
                      )
                      .slice(0, 3)
                      .map((pattern, index) => (
                        <div key={pattern.id} style={{ marginBottom: '4px' }}>
                          <Tag
                            color={
                              index === 0
                                ? 'gold'
                                : index === 1
                                  ? 'orange'
                                  : 'blue'
                            }
                          >
                            {index + 1}. {pattern.name}:{' '}
                            {pattern.metrics.throughput.toFixed(1)} msg/s
                          </Tag>
                        </div>
                      ))}
                  </div>
                </Card>
              </Col>

              <Col span={24}>
                <Card title="通信拓扑图" size="small">
                  <div style={{ textAlign: 'center', padding: '60px 0' }}>
                    <Text type="secondary">
                      通信模式拓扑图可视化组件开发中...
                    </Text>
                  </div>
                </Card>
              </Col>
            </Row>
          </TabPane>
        </Tabs>
      </Card>
    </div>
  )
}

export default AdvancedCommunicationPatternsPage
