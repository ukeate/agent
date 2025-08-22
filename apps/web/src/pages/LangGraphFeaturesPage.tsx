import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Button,
  Table,
  Form,
  Input,
  Select,
  Switch,
  Modal,
  Tag,
  Space,
  Statistic,
  Alert,
  Typography,
  Tabs,
  List,
  Descriptions,
  Badge,
  Progress,
  Timeline,
  Tree,
  Drawer
} from 'antd';
import {
  NodeIndexOutlined,
  ApiOutlined,
  ClockCircleOutlined,
  DatabaseOutlined,
  ThunderboltOutlined,
  SettingOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  BranchesOutlined,
  LinkOutlined,
  CheckCircleOutlined,
  WarningOutlined,
  InfoCircleOutlined,
  BugOutlined,
  RocketOutlined,
  CodeOutlined,
  FileTextOutlined,
  ToolOutlined,
  MonitorOutlined
} from '@ant-design/icons';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, Cell, PieChart, Pie } from 'recharts';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;
const { TreeNode } = Tree;

// 接口定义
interface LangGraphNode {
  node_id: string;
  name: string;
  type: 'start' | 'end' | 'tool' | 'conditional' | 'llm' | 'function';
  status: 'idle' | 'running' | 'completed' | 'failed' | 'cached';
  execution_time: number;
  cache_hit: boolean;
  retry_count: number;
  created_at: string;
  hooks: string[];
}

interface WorkflowExecution {
  execution_id: string;
  workflow_name: string;
  status: 'running' | 'completed' | 'failed' | 'paused';
  start_time: string;
  end_time?: string;
  total_nodes: number;
  completed_nodes: number;
  failed_nodes: number;
  cached_nodes: number;
  context_data: any;
  durability_enabled: boolean;
}

interface ContextItem {
  key: string;
  value: any;
  type: 'string' | 'number' | 'object' | 'array' | 'boolean';
  persistent: boolean;
  last_updated: string;
}

interface CacheMetrics {
  total_requests: number;
  cache_hits: number;
  cache_misses: number;
  hit_rate: number;
  cache_size: number;
  evictions: number;
}

const LangGraphFeaturesPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('context');
  const [loading, setLoading] = useState(false);
  const [nodes, setNodes] = useState<LangGraphNode[]>([]);
  const [executions, setExecutions] = useState<WorkflowExecution[]>([]);
  const [contextData, setContextData] = useState<ContextItem[]>([]);
  const [cacheMetrics, setCacheMetrics] = useState<CacheMetrics | null>(null);
  const [selectedExecution, setSelectedExecution] = useState<WorkflowExecution | null>(null);
  const [contextDrawerVisible, setContextDrawerVisible] = useState(false);
  const [nodeDetailVisible, setNodeDetailVisible] = useState(false);
  const [selectedNode, setSelectedNode] = useState<LangGraphNode | null>(null);
  const [form] = Form.useForm();

  // 生成模拟数据
  useEffect(() => {
    generateMockData();
  }, []);

  const generateMockData = () => {
    // 生成节点数据
    const mockNodes: LangGraphNode[] = [
      {
        node_id: 'start_001',
        name: 'workflow_start',
        type: 'start',
        status: 'completed',
        execution_time: 2,
        cache_hit: false,
        retry_count: 0,
        created_at: '2025-08-20T10:00:00Z',
        hooks: ['pre_execution', 'post_execution']
      },
      {
        node_id: 'llm_001',
        name: 'analyze_query',
        type: 'llm',
        status: 'running',
        execution_time: 1250,
        cache_hit: true,
        retry_count: 0,
        created_at: '2025-08-20T10:00:02Z',
        hooks: ['pre_execution', 'cache_check', 'post_execution']
      },
      {
        node_id: 'tool_001',
        name: 'vector_search',
        type: 'tool',
        status: 'completed',
        execution_time: 89,
        cache_hit: false,
        retry_count: 1,
        created_at: '2025-08-20T10:01:15Z',
        hooks: ['pre_execution', 'error_handling', 'post_execution']
      },
      {
        node_id: 'cond_001',
        name: 'decision_gate',
        type: 'conditional',
        status: 'completed',
        execution_time: 15,
        cache_hit: false,
        retry_count: 0,
        created_at: '2025-08-20T10:01:45Z',
        hooks: ['pre_execution']
      },
      {
        node_id: 'func_001',
        name: 'format_response',
        type: 'function',
        status: 'cached',
        execution_time: 0,
        cache_hit: true,
        retry_count: 0,
        created_at: '2025-08-20T10:02:00Z',
        hooks: ['cache_check']
      }
    ];

    // 生成执行记录
    const mockExecutions: WorkflowExecution[] = [
      {
        execution_id: 'exec_001',
        workflow_name: 'intelligent_qa_workflow',
        status: 'running',
        start_time: '2025-08-20T10:00:00Z',
        total_nodes: 8,
        completed_nodes: 5,
        failed_nodes: 0,
        cached_nodes: 2,
        context_data: {
          user_query: '什么是LangGraph?',
          session_id: 'sess_12345',
          model_name: 'claude-3.5-sonnet'
        },
        durability_enabled: true
      },
      {
        execution_id: 'exec_002',
        workflow_name: 'document_analysis_workflow',
        status: 'completed',
        start_time: '2025-08-20T09:45:00Z',
        end_time: '2025-08-20T09:47:30Z',
        total_nodes: 12,
        completed_nodes: 12,
        failed_nodes: 0,
        cached_nodes: 4,
        context_data: {
          document_id: 'doc_67890',
          analysis_type: 'summary',
          language: 'zh-CN'
        },
        durability_enabled: true
      },
      {
        execution_id: 'exec_003',
        workflow_name: 'multi_agent_debate',
        status: 'failed',
        start_time: '2025-08-20T09:30:00Z',
        end_time: '2025-08-20T09:32:15Z',
        total_nodes: 15,
        completed_nodes: 8,
        failed_nodes: 2,
        cached_nodes: 1,
        context_data: {
          topic: '人工智能的未来发展',
          agent_count: 3,
          max_rounds: 5
        },
        durability_enabled: false
      }
    ];

    // 生成上下文数据
    const mockContextData: ContextItem[] = [
      {
        key: 'user_session',
        value: {
          user_id: 'user_12345',
          session_id: 'sess_abcdef',
          preferences: {
            language: 'zh-CN',
            response_format: 'detailed'
          }
        },
        type: 'object',
        persistent: true,
        last_updated: '2025-08-20T10:00:00Z'
      },
      {
        key: 'workflow_config',
        value: {
          max_iterations: 10,
          timeout_seconds: 300,
          retry_policy: 'exponential_backoff'
        },
        type: 'object',
        persistent: true,
        last_updated: '2025-08-20T09:55:00Z'
      },
      {
        key: 'intermediate_results',
        value: ['分析完成', '检索到5个相关文档', '生成候选答案'],
        type: 'array',
        persistent: false,
        last_updated: '2025-08-20T10:01:30Z'
      },
      {
        key: 'current_step',
        value: 3,
        type: 'number',
        persistent: false,
        last_updated: '2025-08-20T10:01:45Z'
      },
      {
        key: 'debug_mode',
        value: true,
        type: 'boolean',
        persistent: true,
        last_updated: '2025-08-20T09:50:00Z'
      }
    ];

    // 生成缓存指标
    const mockCacheMetrics: CacheMetrics = {
      total_requests: 2847,
      cache_hits: 1923,
      cache_misses: 924,
      hit_rate: 67.5,
      cache_size: 128.5, // MB
      evictions: 45
    };

    setNodes(mockNodes);
    setExecutions(mockExecutions);
    setContextData(mockContextData);
    setCacheMetrics(mockCacheMetrics);
  };

  // 节点表格列定义
  const nodeColumns = [
    {
      title: '节点名称',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: LangGraphNode) => (
        <Space>
          <Badge status={record.status === 'completed' ? 'success' : record.status === 'failed' ? 'error' : 'processing'} />
          <Text strong>{text}</Text>
          <Tag color={getNodeTypeColor(record.type)}>{record.type}</Tag>
        </Space>
      )
    },
    {
      title: '执行状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const colors = {
          idle: 'default',
          running: 'processing',
          completed: 'success',
          failed: 'error',
          cached: 'warning'
        };
        return <Badge status={colors[status as keyof typeof colors]} text={status} />;
      }
    },
    {
      title: '执行时间',
      dataIndex: 'execution_time',
      key: 'execution_time',
      render: (time: number) => `${time}ms`
    },
    {
      title: '缓存命中',
      dataIndex: 'cache_hit',
      key: 'cache_hit',
      render: (hit: boolean) => (
        hit ? <CheckCircleOutlined style={{ color: '#52c41a' }} /> : <ClockCircleOutlined style={{ color: '#d9d9d9' }} />
      )
    },
    {
      title: 'Hooks',
      dataIndex: 'hooks',
      key: 'hooks',
      render: (hooks: string[]) => (
        <Space size={4}>
          {hooks.map(hook => (
            <Tag key={hook} size="small">{hook}</Tag>
          ))}
        </Space>
      )
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record: LangGraphNode) => (
        <Space>
          <Button size="small" icon={<InfoCircleOutlined />} onClick={() => handleViewNode(record)}>
            详情
          </Button>
          <Button size="small" icon={<ReloadOutlined />}>
            重试
          </Button>
        </Space>
      )
    }
  ];

  // 执行记录表格列定义
  const executionColumns = [
    {
      title: '工作流名称',
      dataIndex: 'workflow_name',
      key: 'workflow_name'
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const colors = {
          running: 'processing',
          completed: 'success',
          failed: 'error',
          paused: 'warning'
        };
        return <Badge status={colors[status as keyof typeof colors]} text={status} />;
      }
    },
    {
      title: '进度',
      key: 'progress',
      render: (_, record: WorkflowExecution) => {
        const percent = (record.completed_nodes / record.total_nodes) * 100;
        return (
          <Space direction="vertical" size={0}>
            <Progress percent={percent} size="small" />
            <Text style={{ fontSize: '12px' }}>
              {record.completed_nodes}/{record.total_nodes} 节点
            </Text>
          </Space>
        );
      }
    },
    {
      title: '持久化',
      dataIndex: 'durability_enabled',
      key: 'durability_enabled',
      render: (enabled: boolean) => (
        enabled ? <DatabaseOutlined style={{ color: '#52c41a' }} /> : <ClockCircleOutlined style={{ color: '#d9d9d9' }} />
      )
    },
    {
      title: '缓存节点',
      dataIndex: 'cached_nodes',
      key: 'cached_nodes',
      render: (count: number) => (
        <Tag color="orange">{count} 缓存</Tag>
      )
    },
    {
      title: '开始时间',
      dataIndex: 'start_time',
      key: 'start_time',
      render: (time: string) => new Date(time).toLocaleString()
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record: WorkflowExecution) => (
        <Space>
          <Button size="small" icon={<InfoCircleOutlined />} onClick={() => handleViewExecution(record)}>
            详情
          </Button>
          {record.status === 'running' && (
            <Button size="small" icon={<PauseCircleOutlined />}>
              暂停
            </Button>
          )}
        </Space>
      )
    }
  ];

  const getNodeTypeColor = (type: string) => {
    const colors = {
      start: 'green',
      end: 'red',
      tool: 'blue',
      conditional: 'orange',
      llm: 'purple',
      function: 'cyan'
    };
    return colors[type as keyof typeof colors] || 'default';
  };

  const handleViewNode = (node: LangGraphNode) => {
    setSelectedNode(node);
    setNodeDetailVisible(true);
  };

  const handleViewExecution = (execution: WorkflowExecution) => {
    setSelectedExecution(execution);
    setContextDrawerVisible(true);
  };

  // 图表数据
  const performanceData = Array.from({ length: 24 }, (_, i) => ({
    hour: i,
    executions: Math.floor(Math.random() * 50) + 10,
    cache_hits: Math.floor(Math.random() * 30) + 5,
    avg_time: Math.floor(Math.random() * 1000) + 200
  }));

  const nodeTypeData = [
    { name: 'LLM节点', value: nodes.filter(n => n.type === 'llm').length, color: '#8884d8' },
    { name: '工具节点', value: nodes.filter(n => n.type === 'tool').length, color: '#82ca9d' },
    { name: '条件节点', value: nodes.filter(n => n.type === 'conditional').length, color: '#ffc658' },
    { name: '函数节点', value: nodes.filter(n => n.type === 'function').length, color: '#ff7300' }
  ];

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        {/* 页面标题 */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <Title level={2}>
              <NodeIndexOutlined /> LangGraph 0.6.5 新特性中心
            </Title>
            <Paragraph>
              Context API、Durability控制、Node Caching、Pre/Post Hooks演示平台
            </Paragraph>
          </div>
          <Space>
            <Button type="primary" icon={<PlayCircleOutlined />}>
              创建工作流
            </Button>
            <Button icon={<SettingOutlined />}>
              配置管理
            </Button>
            <Button icon={<ReloadOutlined />} onClick={generateMockData}>
              刷新数据
            </Button>
          </Space>
        </div>

        {/* 统计卡片 */}
        <Row gutter={[16, 16]}>
          <Col span={6}>
            <Card>
              <Statistic
                title="活跃工作流"
                value={executions.filter(e => e.status === 'running').length}
                prefix={<RocketOutlined />}
                valueStyle={{ color: '#1677ff' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="缓存命中率"
                value={cacheMetrics?.hit_rate || 0}
                suffix="%"
                prefix={<DatabaseOutlined />}
                valueStyle={{ color: '#52c41a' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="总节点数"
                value={nodes.length}
                prefix={<NodeIndexOutlined />}
                valueStyle={{ color: '#faad14' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="平均执行时间"
                value={nodes.reduce((acc, n) => acc + n.execution_time, 0) / nodes.length}
                suffix="ms"
                prefix={<ClockCircleOutlined />}
                valueStyle={{ color: '#ff4d4f' }}
              />
            </Card>
          </Col>
        </Row>

        {/* 主要功能标签页 */}
        <Card>
          <Tabs activeKey={activeTab} onChange={setActiveTab} items={[
            {
              key: 'context',
              label: (
                <span>
                  <DatabaseOutlined />
                  Context API
                </span>
              ),
              children: (
                <div>
                  <Alert
                    message="LangGraph 0.6.5 Context API 特性"
                    description="支持跨节点共享状态、持久化存储、类型安全的上下文传递"
                    variant="default"
                    showIcon
                    style={{ marginBottom: 16 }}
                  />
                  <Row gutter={16}>
                    <Col span={16}>
                      <List
                        header={<div style={{ fontWeight: 'bold' }}>当前上下文数据</div>}
                        bordered
                        dataSource={contextData}
                        renderItem={(item: ContextItem) => (
                          <List.Item
                            actions={[
                              <Button size="small" type="link">编辑</Button>,
                              <Button size="small" type="link" danger>删除</Button>
                            ]}
                          >
                            <List.Item.Meta
                              title={
                                <Space>
                                  <Text strong>{item.key}</Text>
                                  <Tag color={item.persistent ? 'green' : 'orange'}>
                                    {item.persistent ? '持久化' : '临时'}
                                  </Tag>
                                  <Tag color="blue">{item.type}</Tag>
                                </Space>
                              }
                              description={
                                <div>
                                  <Text code>{JSON.stringify(item.value, null, 2)}</Text>
                                  <br />
                                  <Text type="secondary">更新时间: {new Date(item.last_updated).toLocaleString()}</Text>
                                </div>
                              }
                            />
                          </List.Item>
                        )}
                      />
                    </Col>
                    <Col span={8}>
                      <Card size="small" title="Context 操作">
                        <Space direction="vertical" style={{ width: '100%' }}>
                          <Button block icon={<DatabaseOutlined />}>
                            查看全局状态
                          </Button>
                          <Button block icon={<CodeOutlined />}>
                            导出Context Schema
                          </Button>
                          <Button block icon={<ToolOutlined />}>
                            类型验证工具
                          </Button>
                          <Button block icon={<BugOutlined />}>
                            Context调试器
                          </Button>
                        </Space>
                      </Card>
                    </Col>
                  </Row>
                </div>
              )
            },
            {
              key: 'durability',
              label: (
                <span>
                  <DatabaseOutlined />
                  Durability 控制
                </span>
              ),
              children: (
                <div>
                  <Alert
                    message="工作流持久化控制"
                    description="支持检查点保存、故障恢复、状态持久化存储"
                    type="success"
                    showIcon
                    style={{ marginBottom: 16 }}
                  />
                  <Table
                    columns={executionColumns}
                    dataSource={executions}
                    rowKey="execution_id"
                    pagination={{ pageSize: 10 }}
                  />
                </div>
              )
            },
            {
              key: 'caching',
              label: (
                <span>
                  <ThunderboltOutlined />
                  Node Caching
                </span>
              ),
              children: (
                <div>
                  <Row gutter={16} style={{ marginBottom: 16 }}>
                    <Col span={12}>
                      <Card title="缓存性能指标" size="small">
                        <ResponsiveContainer width="100%" height={200}>
                          <LineChart data={performanceData.slice(-12)}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="hour" />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            <Line type="monotone" dataKey="cache_hits" stroke="#52c41a" name="缓存命中" />
                            <Line type="monotone" dataKey="executions" stroke="#1677ff" name="总执行" />
                          </LineChart>
                        </ResponsiveContainer>
                      </Card>
                    </Col>
                    <Col span={12}>
                      <Card title="缓存统计" size="small">
                        {cacheMetrics && (
                          <Descriptions column={1} size="small">
                            <Descriptions.Item label="总请求">{cacheMetrics.total_requests.toLocaleString()}</Descriptions.Item>
                            <Descriptions.Item label="缓存命中">{cacheMetrics.cache_hits.toLocaleString()}</Descriptions.Item>
                            <Descriptions.Item label="缓存未命中">{cacheMetrics.cache_misses.toLocaleString()}</Descriptions.Item>
                            <Descriptions.Item label="命中率">{cacheMetrics.hit_rate}%</Descriptions.Item>
                            <Descriptions.Item label="缓存大小">{cacheMetrics.cache_size} MB</Descriptions.Item>
                            <Descriptions.Item label="淘汰次数">{cacheMetrics.evictions}</Descriptions.Item>
                          </Descriptions>
                        )}
                      </Card>
                    </Col>
                  </Row>
                  <Table
                    columns={nodeColumns}
                    dataSource={nodes}
                    rowKey="node_id"
                    pagination={{ pageSize: 10 }}
                  />
                </div>
              )
            },
            {
              key: 'hooks',
              label: (
                <span>
                  <LinkOutlined />
                  Pre/Post Hooks
                </span>
              ),
              children: (
                <div>
                  <Alert
                    message="节点钩子系统"
                    description="支持节点执行前后的自定义逻辑、错误处理、性能监控"
                    variant="warning"
                    showIcon
                    style={{ marginBottom: 16 }}
                  />
                  <Row gutter={16}>
                    <Col span={16}>
                      <Timeline mode="left">
                        {nodes.map((node, index) => (
                          <Timeline.Item
                            key={node.node_id}
                            color={node.status === 'completed' ? 'green' : node.status === 'failed' ? 'red' : 'blue'}
                            label={new Date(node.created_at).toLocaleTimeString()}
                          >
                            <Card size="small" style={{ marginBottom: 8 }}>
                              <Space direction="vertical" size={4}>
                                <Space>
                                  <Text strong>{node.name}</Text>
                                  <Tag color={getNodeTypeColor(node.type)}>{node.type}</Tag>
                                  <Badge status={node.status === 'completed' ? 'success' : node.status === 'failed' ? 'error' : 'processing'} text={node.status} />
                                </Space>
                                <Space size={4}>
                                  {node.hooks.map(hook => (
                                    <Tag key={hook} size="small" color="blue">{hook}</Tag>
                                  ))}
                                </Space>
                                <Text type="secondary">执行时间: {node.execution_time}ms</Text>
                              </Space>
                            </Card>
                          </Timeline.Item>
                        ))}
                      </Timeline>
                    </Col>
                    <Col span={8}>
                      <Card title="Hook 类型统计" size="small">
                        <ResponsiveContainer width="100%" height={250}>
                          <PieChart>
                            <Pie
                              data={nodeTypeData}
                              cx="50%"
                              cy="50%"
                              innerRadius={40}
                              outerRadius={80}
                              dataKey="value"
                            >
                              {nodeTypeData.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.color} />
                              ))}
                            </Pie>
                            <Tooltip />
                          </PieChart>
                        </ResponsiveContainer>
                      </Card>
                    </Col>
                  </Row>
                </div>
              )
            },
            {
              key: 'monitoring',
              label: (
                <span>
                  <MonitorOutlined />
                  性能监控
                </span>
              ),
              children: (
                <div>
                  <Row gutter={[16, 16]}>
                    <Col span={24}>
                      <Card title="24小时执行趋势">
                        <ResponsiveContainer width="100%" height={300}>
                          <BarChart data={performanceData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="hour" />
                            <YAxis yAxisId="left" />
                            <YAxis yAxisId="right" orientation="right" />
                            <Tooltip />
                            <Legend />
                            <Bar yAxisId="left" dataKey="executions" fill="#1677ff" name="执行次数" />
                            <Bar yAxisId="left" dataKey="cache_hits" fill="#52c41a" name="缓存命中" />
                            <Line yAxisId="right" type="monotone" dataKey="avg_time" stroke="#ff7300" name="平均时间(ms)" />
                          </BarChart>
                        </ResponsiveContainer>
                      </Card>
                    </Col>
                  </Row>
                </div>
              )
            }
          ]} />
        </Card>

        {/* 上下文详情抽屉 */}
        <Drawer
          title="工作流执行详情"
          open={contextDrawerVisible}
          onClose={() => setContextDrawerVisible(false)}
          width={600}
        >
          {selectedExecution && (
            <Space direction="vertical" size="large" style={{ width: '100%' }}>
              <Descriptions title="基本信息" bordered column={1}>
                <Descriptions.Item label="执行ID">{selectedExecution.execution_id}</Descriptions.Item>
                <Descriptions.Item label="工作流名称">{selectedExecution.workflow_name}</Descriptions.Item>
                <Descriptions.Item label="状态">
                  <Badge status={selectedExecution.status === 'completed' ? 'success' : 'error'} text={selectedExecution.status} />
                </Descriptions.Item>
                <Descriptions.Item label="开始时间">{new Date(selectedExecution.start_time).toLocaleString()}</Descriptions.Item>
                {selectedExecution.end_time && (
                  <Descriptions.Item label="结束时间">{new Date(selectedExecution.end_time).toLocaleString()}</Descriptions.Item>
                )}
                <Descriptions.Item label="持久化">
                  {selectedExecution.durability_enabled ? '已启用' : '未启用'}
                </Descriptions.Item>
              </Descriptions>
              
              <Card title="Context 数据" size="small">
                <pre style={{ background: '#f5f5f5', padding: '12px', borderRadius: '4px' }}>
                  {JSON.stringify(selectedExecution.context_data, null, 2)}
                </pre>
              </Card>
            </Space>
          )}
        </Drawer>

        {/* 节点详情模态框 */}
        <Modal
          title="节点详情"
          open={nodeDetailVisible}
          onCancel={() => setNodeDetailVisible(false)}
          footer={null}
          width={700}
        >
          {selectedNode && (
            <Space direction="vertical" size="large" style={{ width: '100%' }}>
              <Descriptions bordered column={2}>
                <Descriptions.Item label="节点ID">{selectedNode.node_id}</Descriptions.Item>
                <Descriptions.Item label="节点名称">{selectedNode.name}</Descriptions.Item>
                <Descriptions.Item label="节点类型">
                  <Tag color={getNodeTypeColor(selectedNode.type)}>{selectedNode.type}</Tag>
                </Descriptions.Item>
                <Descriptions.Item label="执行状态">
                  <Badge status={selectedNode.status === 'completed' ? 'success' : 'error'} text={selectedNode.status} />
                </Descriptions.Item>
                <Descriptions.Item label="执行时间">{selectedNode.execution_time}ms</Descriptions.Item>
                <Descriptions.Item label="缓存命中">
                  {selectedNode.cache_hit ? '是' : '否'}
                </Descriptions.Item>
                <Descriptions.Item label="重试次数">{selectedNode.retry_count}</Descriptions.Item>
                <Descriptions.Item label="创建时间">{new Date(selectedNode.created_at).toLocaleString()}</Descriptions.Item>
              </Descriptions>
              
              <Card title="Hook 配置" size="small">
                <Space size={8}>
                  {selectedNode.hooks.map(hook => (
                    <Tag key={hook} color="blue">{hook}</Tag>
                  ))}
                </Space>
              </Card>
            </Space>
          )}
        </Modal>
      </Space>
    </div>
  );
};

export default LangGraphFeaturesPage;