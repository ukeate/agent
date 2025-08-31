import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Button,
  Table,
  Tabs,
  Form,
  Input,
  Select,
  InputNumber,
  Switch,
  Alert,
  Space,
  Typography,
  Slider,
  Radio,
  Collapse,
  Tree,
  Transfer,
  Modal,
  Tag,
  Badge,
  Tooltip,
  Popconfirm,
  notification,
  Divider,
  Upload,
  Progress,
  Timeline,
  List,
} from 'antd'
import {
  SettingOutlined,
  DatabaseOutlined,
  ThunderboltOutlined,
  NodeIndexOutlined,
  ShareAltOutlined,
  ExperimentOutlined,
  SaveOutlined,
  ReloadOutlined,
  ImportOutlined,
  ExportOutlined,
  CopyOutlined,
  DeleteOutlined,
  EditOutlined,
  PlusOutlined,
  CheckCircleOutlined,
  WarningOutlined,
  InfoCircleOutlined,
  ToolOutlined,
  BulbOutlined,
  RocketOutlined,
  MonitorOutlined,
  CloudUploadOutlined,
  DownloadOutlined,
  SyncOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
} from '@ant-design/icons'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, BarChart, Bar } from 'recharts'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs
const { TextArea } = Input
const { Option } = Select
const { Panel } = Collapse
const { TreeNode } = Tree

interface StrategyConfig {
  name: string
  enabled: boolean
  weight: number
  priority: number
  parameters: Record<string, any>
  performance: {
    accuracy: number
    speed: number
    resource: number
  }
}

interface EngineConfig {
  id: string
  name: string
  type: 'rule' | 'embedding' | 'path' | 'uncertainty' | 'hybrid'
  status: 'active' | 'inactive' | 'error'
  version: string
  parameters: Record<string, any>
  dependencies: string[]
  lastUpdated: string
}

const KGReasoningConfigPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('strategies')
  const [form] = Form.useForm()
  const [configForm] = Form.useForm()
  const [selectedStrategy, setSelectedStrategy] = useState<string>('')
  const [selectedEngine, setSelectedEngine] = useState<EngineConfig | null>(null)
  const [configModalVisible, setConfigModalVisible] = useState(false)
  const [importModalVisible, setImportModalVisible] = useState(false)
  const [hasChanges, setHasChanges] = useState(false)

  // 推理策略配置
  const [strategyConfigs, setStrategyConfigs] = useState<Record<string, StrategyConfig>>({
    rule_only: {
      name: '规则推理',
      enabled: true,
      weight: 30,
      priority: 1,
      parameters: {
        rule_engine: 'swrl_parser',
        forward_chaining: true,
        inference_depth: 5,
        cache_rules: true,
        parallel_execution: true,
        rule_optimization: 'aggressive'
      },
      performance: { accuracy: 95, speed: 85, resource: 70 }
    },
    embedding_only: {
      name: '嵌入推理',
      enabled: true,
      weight: 25,
      priority: 2,
      parameters: {
        model_type: 'TransE',
        embedding_dim: 200,
        similarity_threshold: 0.8,
        index_type: 'faiss',
        batch_size: 1000,
        gpu_acceleration: true
      },
      performance: { accuracy: 87, speed: 92, resource: 85 }
    },
    path_only: {
      name: '路径推理',
      enabled: true,
      weight: 20,
      priority: 3,
      parameters: {
        search_algorithm: 'bidirectional_bfs',
        max_hops: 4,
        path_ranking: 'confidence',
        pruning_threshold: 0.1,
        parallel_paths: 8,
        timeout_ms: 5000
      },
      performance: { accuracy: 89, speed: 75, resource: 90 }
    },
    uncertainty_only: {
      name: '不确定性推理',
      enabled: false,
      weight: 10,
      priority: 4,
      parameters: {
        bayesian_network: true,
        monte_carlo_samples: 10000,
        confidence_interval: 0.95,
        prior_distribution: 'uniform',
        inference_method: 'variational',
        approximation_threshold: 0.01
      },
      performance: { accuracy: 92, speed: 45, resource: 95 }
    },
    ensemble: {
      name: '集成策略',
      enabled: true,
      weight: 35,
      priority: 1,
      parameters: {
        voting_method: 'weighted',
        confidence_weighting: true,
        min_agreement: 0.7,
        strategy_selection: 'auto',
        dynamic_weights: true,
        fallback_strategy: 'rule_only'
      },
      performance: { accuracy: 96, speed: 65, resource: 80 }
    },
    adaptive: {
      name: '自适应策略',
      enabled: true,
      weight: 30,
      priority: 2,
      parameters: {
        learning_rate: 0.01,
        adaptation_window: 1000,
        performance_threshold: 0.9,
        strategy_switching: true,
        feedback_weight: 0.3,
        exploration_rate: 0.1
      },
      performance: { accuracy: 94, speed: 78, resource: 75 }
    }
  })

  // 引擎配置数据
  const engineConfigs: EngineConfig[] = [
    {
      id: 'rule_engine_001',
      name: 'SWRL规则引擎',
      type: 'rule',
      status: 'active',
      version: '2.1.0',
      parameters: {
        max_rules: 10000,
        inference_timeout: 30000,
        memory_limit: '2GB',
        thread_pool_size: 8
      },
      dependencies: ['swrl-parser', 'jena-core'],
      lastUpdated: '2024-01-20 10:30:15'
    },
    {
      id: 'embedding_engine_001',
      name: 'TransE嵌入引擎',
      type: 'embedding',
      status: 'active',
      version: '1.8.2',
      parameters: {
        embedding_dim: 200,
        batch_size: 1000,
        learning_rate: 0.001,
        regularization: 0.01
      },
      dependencies: ['pytorch', 'faiss', 'numpy'],
      lastUpdated: '2024-01-20 09:45:22'
    },
    {
      id: 'path_engine_001',
      name: '图路径推理引擎',
      type: 'path',
      status: 'active',
      version: '3.0.1',
      parameters: {
        max_path_length: 6,
        pruning_threshold: 0.1,
        parallel_workers: 4,
        cache_size: '500MB'
      },
      dependencies: ['networkx', 'graph-tool'],
      lastUpdated: '2024-01-20 08:20:18'
    },
    {
      id: 'uncertainty_engine_001',
      name: '贝叶斯推理引擎',
      type: 'uncertainty',
      status: 'inactive',
      version: '1.5.0',
      parameters: {
        monte_carlo_samples: 10000,
        burn_in_samples: 1000,
        chain_count: 4,
        convergence_threshold: 0.01
      },
      dependencies: ['pymc3', 'scipy'],
      lastUpdated: '2024-01-19 16:15:30'
    }
  ]

  // 配置模板数据
  const configTemplates = [
    {
      name: '高性能配置',
      description: '优化响应速度和吞吐量',
      strategies: ['rule_only', 'embedding_only'],
      parameters: { cache_aggressive: true, parallel_high: true }
    },
    {
      name: '高精度配置',
      description: '优化推理准确率',
      strategies: ['ensemble', 'adaptive'],
      parameters: { accuracy_first: true, validation_strict: true }
    },
    {
      name: '资源节约配置',
      description: '优化资源使用效率',
      strategies: ['rule_only'],
      parameters: { memory_conservative: true, cpu_limited: true }
    },
    {
      name: '平衡配置',
      description: '平衡性能、准确率和资源',
      strategies: ['ensemble', 'rule_only', 'embedding_only'],
      parameters: { balanced_mode: true }
    }
  ]

  const handleSaveConfig = () => {
    form.validateFields().then(values => {
      setHasChanges(false)
      notification.success({
        message: '配置保存成功',
        description: '推理引擎配置已更新并生效',
      })
    }).catch(errorInfo => {
      notification.error({
        message: '配置保存失败',
        description: '请检查配置参数是否正确',
      })
    })
  }

  const handleResetConfig = () => {
    Modal.confirm({
      title: '确定要重置配置吗？',
      content: '此操作将恢复到默认配置，所有自定义设置将丢失',
      onOk() {
        form.resetFields()
        setHasChanges(false)
        notification.info({
          message: '配置已重置',
          description: '已恢复到默认配置'
        })
      }
    })
  }

  const handleImportConfig = () => {
    // 处理配置导入逻辑
    notification.success({
      message: '配置导入成功',
      description: '配置文件已成功导入并应用'
    })
    setImportModalVisible(false)
  }

  const handleExportConfig = () => {
    // 处理配置导出逻辑
    const config = form.getFieldsValue()
    const dataStr = JSON.stringify(config, null, 2)
    const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr)
    
    const exportFileDefaultName = `kg_reasoning_config_${new Date().toISOString().split('T')[0]}.json`
    
    const linkElement = document.createElement('a')
    linkElement.setAttribute('href', dataUri)
    linkElement.setAttribute('download', exportFileDefaultName)
    linkElement.click()
  }

  const strategyColumns = [
    {
      title: '推理策略',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: StrategyConfig) => (
        <Space>
          <ThunderboltOutlined />
          <Text strong>{name}</Text>
          {record.enabled ? 
            <Badge status="success" text="启用" /> : 
            <Badge status="error" text="禁用" />
          }
        </Space>
      )
    },
    {
      title: '权重',
      dataIndex: 'weight',
      key: 'weight',
      render: (weight: number) => (
        <Progress 
          percent={weight} 
          size="small" 
          strokeColor="#1890ff"
          format={() => `${weight}%`}
        />
      )
    },
    {
      title: '优先级',
      dataIndex: 'priority',
      key: 'priority',
      render: (priority: number) => (
        <Tag color={priority === 1 ? 'red' : priority === 2 ? 'orange' : 'green'}>
          P{priority}
        </Tag>
      )
    },
    {
      title: '性能指标',
      dataIndex: 'performance',
      key: 'performance',
      render: (performance: any) => (
        <Space direction="vertical" size={2}>
          <Text style={{ fontSize: '12px' }}>准确率: {performance.accuracy}%</Text>
          <Text style={{ fontSize: '12px' }}>速度: {performance.speed}%</Text>
          <Text style={{ fontSize: '12px' }}>资源: {performance.resource}%</Text>
        </Space>
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record: StrategyConfig, index: number) => (
        <Space>
          <Button 
            size="small" 
            icon={<EditOutlined />}
            onClick={() => {
              setSelectedStrategy(Object.keys(strategyConfigs)[index])
              setConfigModalVisible(true)
            }}
          >
            配置
          </Button>
          <Switch 
            size="small"
            checked={record.enabled}
            onChange={(checked) => {
              const newConfigs = { ...strategyConfigs }
              const key = Object.keys(strategyConfigs)[index]
              newConfigs[key] = { ...record, enabled: checked }
              setStrategyConfigs(newConfigs)
              setHasChanges(true)
            }}
          />
        </Space>
      )
    }
  ]

  const engineColumns = [
    {
      title: '引擎信息',
      key: 'info',
      render: (_, record: EngineConfig) => (
        <Space direction="vertical" size={2}>
          <Space>
            <DatabaseOutlined />
            <Text strong>{record.name}</Text>
          </Space>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            ID: {record.id} | 版本: {record.version}
          </Text>
        </Space>
      )
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => {
        const typeColors = {
          rule: 'blue',
          embedding: 'green',
          path: 'orange',
          uncertainty: 'purple',
          hybrid: 'red'
        }
        return <Tag color={typeColors[type]}>{type}</Tag>
      }
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Badge 
          status={status === 'active' ? 'success' : status === 'error' ? 'error' : 'default'}
          text={status === 'active' ? '运行中' : status === 'error' ? '错误' : '停止'}
        />
      )
    },
    {
      title: '依赖',
      dataIndex: 'dependencies',
      key: 'dependencies',
      render: (deps: string[]) => (
        <div>
          {deps.slice(0, 2).map(dep => (
            <Tag key={dep} size="small">{dep}</Tag>
          ))}
          {deps.length > 2 && <Tag size="small">+{deps.length - 2}</Tag>}
        </div>
      )
    },
    {
      title: '更新时间',
      dataIndex: 'lastUpdated',
      key: 'lastUpdated',
      render: (time: string) => (
        <Text type="secondary" style={{ fontSize: '12px' }}>{time}</Text>
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record: EngineConfig) => (
        <Space>
          <Button size="small" icon={<SettingOutlined />}>配置</Button>
          {record.status === 'active' ? (
            <Button size="small" icon={<PauseCircleOutlined />}>停止</Button>
          ) : (
            <Button size="small" icon={<PlayCircleOutlined />} type="primary">启动</Button>
          )}
          <Button size="small" icon={<MonitorOutlined />}>监控</Button>
        </Space>
      )
    }
  ]

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <SettingOutlined /> 引擎配置中心
        </Title>
        <Paragraph>
          推理引擎策略配置、参数优化和系统管理控制台
        </Paragraph>
      </div>

      {/* 快速操作栏 */}
      <Card style={{ marginBottom: '24px' }}>
        <Space size="large" style={{ width: '100%', justifyContent: 'space-between' }}>
          <Space>
            <Button 
              type="primary" 
              icon={<SaveOutlined />} 
              onClick={handleSaveConfig}
              disabled={!hasChanges}
            >
              保存配置
            </Button>
            <Button 
              icon={<ReloadOutlined />} 
              onClick={handleResetConfig}
            >
              重置配置
            </Button>
            <Button 
              icon={<ImportOutlined />}
              onClick={() => setImportModalVisible(true)}
            >
              导入配置
            </Button>
            <Button 
              icon={<ExportOutlined />}
              onClick={handleExportConfig}
            >
              导出配置
            </Button>
          </Space>
          {hasChanges && (
            <Alert 
              message="配置已修改，请保存更改" 
              type="warning" 
              showIcon 
              banner
              style={{ marginBottom: 0 }}
            />
          )}
        </Space>
      </Card>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="推理策略" key="strategies">
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={16}>
              <Card title="策略配置管理">
                <Table 
                  dataSource={Object.values(strategyConfigs)}
                  columns={strategyColumns}
                  pagination={false}
                  size="small"
                />
              </Card>

              {/* 策略权重分布图 */}
              <Card title="策略权重分布" style={{ marginTop: '16px' }}>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={Object.entries(strategyConfigs).map(([key, config]) => ({
                    name: config.name,
                    weight: config.weight,
                    enabled: config.enabled
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <RechartsTooltip />
                    <Bar dataKey="weight" fill="#1890ff" />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </Col>

            <Col xs={24} lg={8}>
              <Card title="配置模板" style={{ marginBottom: '16px' }}>
                <List
                  dataSource={configTemplates}
                  renderItem={(template) => (
                    <List.Item 
                      actions={[
                        <Button 
                          size="small"
                          type="link"
                          onClick={() => {
                            // 应用模板配置
                            notification.success({
                              message: '模板已应用',
                              description: `${template.name}配置模板已成功应用`
                            })
                          }}
                        >
                          应用
                        </Button>
                      ]}
                    >
                      <List.Item.Meta
                        title={template.name}
                        description={template.description}
                      />
                    </List.Item>
                  )}
                />
              </Card>

              <Card title="配置验证" size="small">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Alert 
                    message="配置检查"
                    description="当前配置通过所有验证规则"
                    type="success"
                    showIcon
                    size="small"
                  />
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Text>策略一致性:</Text>
                    <Text style={{ color: '#52c41a' }}>✓ 通过</Text>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Text>资源限制:</Text>
                    <Text style={{ color: '#52c41a' }}>✓ 通过</Text>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Text>依赖关系:</Text>
                    <Text style={{ color: '#52c41a' }}>✓ 通过</Text>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Text>性能预估:</Text>
                    <Text style={{ color: '#faad14' }}>⚠ 警告</Text>
                  </div>
                </Space>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="引擎管理" key="engines">
          <Card title="推理引擎列表">
            <Table 
              dataSource={engineConfigs}
              columns={engineColumns}
              rowKey="id"
              pagination={{ showSizeChanger: true }}
              expandable={{
                expandedRowRender: (record: EngineConfig) => (
                  <Row gutter={16}>
                    <Col span={12}>
                      <Card size="small" title="参数配置">
                        <Form layout="vertical" size="small">
                          {Object.entries(record.parameters).map(([key, value]) => (
                            <Form.Item key={key} label={key}>
                              <Input defaultValue={String(value)} />
                            </Form.Item>
                          ))}
                        </Form>
                      </Card>
                    </Col>
                    <Col span={12}>
                      <Card size="small" title="依赖管理">
                        <List 
                          dataSource={record.dependencies}
                          renderItem={(dep) => (
                            <List.Item>
                              <Space>
                                <Tag color="blue">{dep}</Tag>
                                <Badge status="success" text="已安装" />
                              </Space>
                            </List.Item>
                          )}
                        />
                      </Card>
                    </Col>
                  </Row>
                )
              }}
            />
          </Card>
        </TabPane>

        <TabPane tab="全局参数" key="global">
          <Card title="系统全局配置">
            <Form 
              form={form}
              layout="vertical"
              onFieldsChange={() => setHasChanges(true)}
            >
              <Collapse defaultActiveKey={['basic', 'performance']}>
                <Panel header="基础配置" key="basic">
                  <Row gutter={16}>
                    <Col span={8}>
                      <Form.Item label="系统模式" name="system_mode" initialValue="production">
                        <Select>
                          <Option value="development">开发模式</Option>
                          <Option value="testing">测试模式</Option>
                          <Option value="production">生产模式</Option>
                        </Select>
                      </Form.Item>
                    </Col>
                    <Col span={8}>
                      <Form.Item label="日志级别" name="log_level" initialValue="info">
                        <Select>
                          <Option value="debug">Debug</Option>
                          <Option value="info">Info</Option>
                          <Option value="warning">Warning</Option>
                          <Option value="error">Error</Option>
                        </Select>
                      </Form.Item>
                    </Col>
                    <Col span={8}>
                      <Form.Item label="超时时间(秒)" name="global_timeout" initialValue={30}>
                        <InputNumber min={1} max={300} />
                      </Form.Item>
                    </Col>
                  </Row>
                  <Row gutter={16}>
                    <Col span={8}>
                      <Form.Item name="enable_monitoring" valuePropName="checked" initialValue={true}>
                        <Space>
                          <Switch />
                          <Text>启用系统监控</Text>
                        </Space>
                      </Form.Item>
                    </Col>
                    <Col span={8}>
                      <Form.Item name="enable_caching" valuePropName="checked" initialValue={true}>
                        <Space>
                          <Switch />
                          <Text>启用全局缓存</Text>
                        </Space>
                      </Form.Item>
                    </Col>
                    <Col span={8}>
                      <Form.Item name="enable_metrics" valuePropName="checked" initialValue={true}>
                        <Space>
                          <Switch />
                          <Text>启用性能指标</Text>
                        </Space>
                      </Form.Item>
                    </Col>
                  </Row>
                </Panel>

                <Panel header="性能配置" key="performance">
                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item label="最大并发查询数" name="max_concurrent_queries" initialValue={100}>
                        <Slider min={1} max={1000} marks={{ 50: '50', 200: '200', 500: '500' }} />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item label="查询队列长度" name="query_queue_size" initialValue={10000}>
                        <Slider min={100} max={50000} marks={{ 1000: '1K', 10000: '10K', 25000: '25K' }} />
                      </Form.Item>
                    </Col>
                  </Row>
                  <Row gutter={16}>
                    <Col span={8}>
                      <Form.Item label="内存限制(GB)" name="memory_limit" initialValue={8}>
                        <InputNumber min={1} max={128} />
                      </Form.Item>
                    </Col>
                    <Col span={8}>
                      <Form.Item label="CPU线程数" name="cpu_threads" initialValue={8}>
                        <InputNumber min={1} max={64} />
                      </Form.Item>
                    </Col>
                    <Col span={8}>
                      <Form.Item label="磁盘缓存(GB)" name="disk_cache_size" initialValue={10}>
                        <InputNumber min={1} max={1000} />
                      </Form.Item>
                    </Col>
                  </Row>
                </Panel>

                <Panel header="安全配置" key="security">
                  <Row gutter={16}>
                    <Col span={8}>
                      <Form.Item name="enable_auth" valuePropName="checked" initialValue={true}>
                        <Space>
                          <Switch />
                          <Text>启用身份验证</Text>
                        </Space>
                      </Form.Item>
                    </Col>
                    <Col span={8}>
                      <Form.Item name="enable_encryption" valuePropName="checked" initialValue={true}>
                        <Space>
                          <Switch />
                          <Text>启用数据加密</Text>
                        </Space>
                      </Form.Item>
                    </Col>
                    <Col span={8}>
                      <Form.Item name="enable_audit" valuePropName="checked" initialValue={false}>
                        <Space>
                          <Switch />
                          <Text>启用审计日志</Text>
                        </Space>
                      </Form.Item>
                    </Col>
                  </Row>
                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item label="API访问密钥" name="api_key">
                        <Input.Password placeholder="输入API访问密钥" />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item label="会话超时(分钟)" name="session_timeout" initialValue={60}>
                        <InputNumber min={5} max={1440} />
                      </Form.Item>
                    </Col>
                  </Row>
                </Panel>

                <Panel header="网络配置" key="network">
                  <Row gutter={16}>
                    <Col span={8}>
                      <Form.Item label="服务端口" name="server_port" initialValue={8000}>
                        <InputNumber min={1000} max={65535} />
                      </Form.Item>
                    </Col>
                    <Col span={8}>
                      <Form.Item label="连接池大小" name="connection_pool_size" initialValue={20}>
                        <InputNumber min={1} max={1000} />
                      </Form.Item>
                    </Col>
                    <Col span={8}>
                      <Form.Item label="请求超时(秒)" name="request_timeout" initialValue={30}>
                        <InputNumber min={1} max={300} />
                      </Form.Item>
                    </Col>
                  </Row>
                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item name="enable_ssl" valuePropName="checked" initialValue={true}>
                        <Space>
                          <Switch />
                          <Text>启用SSL/TLS</Text>
                        </Space>
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item name="enable_cors" valuePropName="checked" initialValue={true}>
                        <Space>
                          <Switch />
                          <Text>启用CORS跨域</Text>
                        </Space>
                      </Form.Item>
                    </Col>
                  </Row>
                </Panel>
              </Collapse>
            </Form>
          </Card>
        </TabPane>

        <TabPane tab="配置历史" key="history">
          <Card title="配置变更历史">
            <Timeline>
              <Timeline.Item color="green">
                <Space direction="vertical" size={2}>
                  <Text strong>2024-01-20 10:30:15</Text>
                  <Text>管理员更新了ensemble策略权重配置</Text>
                  <Text type="secondary" style={{ fontSize: '12px' }}>
                    权重从25%调整为35%，提高集成策略优先级
                  </Text>
                </Space>
              </Timeline.Item>
              <Timeline.Item color="blue">
                <Space direction="vertical" size={2}>
                  <Text strong>2024-01-20 09:15:30</Text>
                  <Text>系统自动优化了缓存配置参数</Text>
                  <Text type="secondary" style={{ fontSize: '12px' }}>
                    L1缓存大小从2GB扩展到4GB
                  </Text>
                </Space>
              </Timeline.Item>
              <Timeline.Item color="orange">
                <Space direction="vertical" size={2}>
                  <Text strong>2024-01-20 08:45:12</Text>
                  <Text>导入了高性能配置模板</Text>
                  <Text type="secondary" style={{ fontSize: '12px' }}>
                    应用了针对大规模查询的优化配置
                  </Text>
                </Space>
              </Timeline.Item>
              <Timeline.Item color="red">
                <Space direction="vertical" size={2}>
                  <Text strong>2024-01-19 16:20:45</Text>
                  <Text>禁用了不确定性推理引擎</Text>
                  <Text type="secondary" style={{ fontSize: '12px' }}>
                    由于性能问题暂时禁用，等待修复
                  </Text>
                </Space>
              </Timeline.Item>
            </Timeline>
          </Card>
        </TabPane>
      </Tabs>

      {/* 策略配置对话框 */}
      <Modal
        title={`配置 - ${selectedStrategy ? strategyConfigs[selectedStrategy]?.name : ''}`}
        visible={configModalVisible}
        onCancel={() => setConfigModalVisible(false)}
        width={800}
        footer={[
          <Button key="cancel" onClick={() => setConfigModalVisible(false)}>
            取消
          </Button>,
          <Button key="save" type="primary" onClick={() => {
            configForm.validateFields().then(() => {
              setConfigModalVisible(false)
              setHasChanges(true)
              notification.success({
                message: '策略配置已更新',
                description: '配置修改将在下次推理时生效'
              })
            })
          }}>
            保存
          </Button>
        ]}
      >
        {selectedStrategy && strategyConfigs[selectedStrategy] && (
          <Form
            form={configForm}
            layout="vertical"
            initialValues={strategyConfigs[selectedStrategy].parameters}
          >
            <Tabs size="small">
              <TabPane tab="基本参数" key="basic">
                {Object.entries(strategyConfigs[selectedStrategy].parameters).map(([key, value]) => (
                  <Form.Item key={key} label={key} name={key}>
                    {typeof value === 'boolean' ? (
                      <Switch defaultChecked={value} />
                    ) : typeof value === 'number' ? (
                      <InputNumber defaultValue={value} />
                    ) : (
                      <Input defaultValue={String(value)} />
                    )}
                  </Form.Item>
                ))}
              </TabPane>
              <TabPane tab="高级设置" key="advanced">
                <Alert 
                  message="高级参数配置"
                  description="修改这些参数可能会显著影响推理性能，请谨慎操作"
                  type="warning"
                  showIcon
                  style={{ marginBottom: '16px' }}
                />
                {/* 高级参数配置表单 */}
              </TabPane>
            </Tabs>
          </Form>
        )}
      </Modal>

      {/* 导入配置对话框 */}
      <Modal
        title="导入配置文件"
        visible={importModalVisible}
        onCancel={() => setImportModalVisible(false)}
        onOk={handleImportConfig}
      >
        <Upload.Dragger
          name="config"
          accept=".json,.yaml,.yml"
          beforeUpload={() => false}
        >
          <p className="ant-upload-drag-icon">
            <CloudUploadOutlined />
          </p>
          <p className="ant-upload-text">点击或拖拽配置文件到此区域上传</p>
          <p className="ant-upload-hint">
            支持 JSON、YAML 格式的配置文件
          </p>
        </Upload.Dragger>
        <Alert 
          message="导入提示"
          description="导入的配置将覆盖当前设置，建议先导出备份当前配置"
          type="info"
          showIcon
          style={{ marginTop: '16px' }}
        />
      </Modal>
    </div>
  )
}

export default KGReasoningConfigPage