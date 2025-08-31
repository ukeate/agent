import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Input,
  Button,
  Select,
  Form,
  Table,
  Tabs,
  Alert,
  Tag,
  Space,
  Typography,
  Divider,
  Timeline,
  Progress,
  Modal,
  Tooltip,
  Badge,
  Radio,
  Switch,
  Slider,
  InputNumber,
  Tree,
  Collapse,
} from 'antd'
import {
  SearchOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  SaveOutlined,
  LoadingOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  InfoCircleOutlined,
  SettingOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  NodeIndexOutlined,
  ShareAltOutlined,
  ExperimentOutlined,
  BulbOutlined,
  FileTextOutlined,
  HistoryOutlined,
  FilterOutlined,
  ExportOutlined,
} from '@ant-design/icons'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs
const { TextArea } = Input
const { Option } = Select
const { Panel } = Collapse

interface QueryResult {
  id: string
  query: string
  strategy: string
  status: 'running' | 'completed' | 'failed' | 'pending'
  confidence: number
  responseTime: number
  timestamp: string
  results: any[]
  reasoning_path?: string[]
  evidence?: string[]
}

const KGReasoningQueryPage: React.FC = () => {
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('interactive')
  const [queryHistory, setQueryHistory] = useState<QueryResult[]>([])
  const [currentQuery, setCurrentQuery] = useState<QueryResult | null>(null)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [realTimeMode, setRealTimeMode] = useState(true)

  // 查询策略配置
  const strategyOptions = [
    { value: 'auto', label: '自动选择', description: '系统根据查询特征自动选择最优策略' },
    { value: 'rule_only', label: '规则推理', description: '基于逻辑规则的推理，适用于明确的因果关系' },
    { value: 'embedding_only', label: '嵌入推理', description: '基于向量相似度的推理，适用于语义匹配' },
    { value: 'path_only', label: '路径推理', description: '基于图路径的推理，适用于多跳关系查询' },
    { value: 'uncertainty_only', label: '不确定性推理', description: '基于概率的推理，适用于不确定信息' },
    { value: 'ensemble', label: '集成策略', description: '综合多种策略的结果，提高准确性' },
    { value: 'adaptive', label: '自适应策略', description: '根据实时反馈动态调整推理方式' },
    { value: 'cascading', label: '级联策略', description: '按优先级逐级尝试不同推理方式' },
    { value: 'voting', label: '投票策略', description: '多种策略投票决定最终结果' },
  ]

  // 模拟查询历史数据
  const mockQueryHistory: QueryResult[] = [
    {
      id: '1',
      query: 'person(X) → human(X)',
      strategy: 'rule_only',
      status: 'completed',
      confidence: 0.95,
      responseTime: 0.8,
      timestamp: '2024-01-20 10:30:15',
      results: [
        { entity: 'Alice', type: 'person', confidence: 0.98 },
        { entity: 'Bob', type: 'person', confidence: 0.92 },
        { entity: 'Charlie', type: 'person', confidence: 0.87 }
      ],
      reasoning_path: ['规则匹配', '实体识别', '结果验证'],
      evidence: ['知识库规则: person(X) → human(X)', '实体类型验证', '置信度计算']
    },
    {
      id: '2',
      query: 'similar_to(company, organization)',
      strategy: 'embedding_only',
      status: 'completed',
      confidence: 0.87,
      responseTime: 1.2,
      timestamp: '2024-01-20 10:28:42',
      results: [
        { entity: 'Microsoft', similarity: 0.91, type: 'company' },
        { entity: 'Google', similarity: 0.89, type: 'company' },
        { entity: 'Apple', similarity: 0.86, type: 'company' }
      ],
      reasoning_path: ['向量检索', '相似度计算', '结果排序'],
      evidence: ['TransE嵌入模型', '余弦相似度: 0.87', 'Top-K检索结果']
    },
    {
      id: '3',
      query: 'path(Alice, works_at, ?Company)',
      strategy: 'path_only',
      status: 'running',
      confidence: 0.0,
      responseTime: 0,
      timestamp: '2024-01-20 10:32:10',
      results: [],
      reasoning_path: ['路径搜索中...'],
      evidence: ['BFS算法执行', '已访问节点: 1,247', '待探索路径: 3']
    }
  ]

  useEffect(() => {
    setQueryHistory(mockQueryHistory)
  }, [])

  const handleQuerySubmit = (values: any) => {
    setLoading(true)
    
    const newQuery: QueryResult = {
      id: Date.now().toString(),
      query: values.query,
      strategy: values.strategy || 'auto',
      status: 'running',
      confidence: 0,
      responseTime: 0,
      timestamp: new Date().toLocaleString(),
      results: [],
      reasoning_path: ['查询解析', '策略选择', '推理执行中...'],
      evidence: [`策略: ${values.strategy || 'auto'}`, '开始时间: ' + new Date().toLocaleString()]
    }

    setCurrentQuery(newQuery)
    setQueryHistory(prev => [newQuery, ...prev])
    
    // 模拟查询执行
    setTimeout(() => {
      const updatedQuery = {
        ...newQuery,
        status: 'completed' as const,
        confidence: 0.85 + Math.random() * 0.15,
        responseTime: 0.5 + Math.random() * 2,
        results: [
          { entity: 'Result1', confidence: 0.92, type: 'inferred' },
          { entity: 'Result2', confidence: 0.87, type: 'inferred' },
        ],
        reasoning_path: ['查询解析', '策略选择', '推理执行', '结果验证', '置信度计算'],
        evidence: [`策略: ${values.strategy || 'auto'}`, '推理路径长度: 3', '置信度阈值: 0.8', '验证通过']
      }
      
      setCurrentQuery(updatedQuery)
      setQueryHistory(prev => prev.map(q => q.id === newQuery.id ? updatedQuery : q))
      setLoading(false)
    }, 2000 + Math.random() * 3000)
  }

  const handleQueryStop = () => {
    if (currentQuery && currentQuery.status === 'running') {
      const stoppedQuery = {
        ...currentQuery,
        status: 'failed' as const,
        reasoning_path: [...(currentQuery.reasoning_path || []), '用户中止']
      }
      setCurrentQuery(stoppedQuery)
      setQueryHistory(prev => prev.map(q => q.id === currentQuery.id ? stoppedQuery : q))
    }
    setLoading(false)
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'running': return <LoadingOutlined spin style={{ color: '#1890ff' }} />
      case 'completed': return <CheckCircleOutlined style={{ color: '#52c41a' }} />
      case 'failed': return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />
      case 'pending': return <InfoCircleOutlined style={{ color: '#faad14' }} />
      default: return null
    }
  }

  const getStatusText = (status: string) => {
    switch (status) {
      case 'running': return '执行中'
      case 'completed': return '已完成'
      case 'failed': return '已失败'
      case 'pending': return '等待中'
      default: return '未知'
    }
  }

  // 预置查询模板
  const queryTemplates = [
    {
      name: '实体类型推理',
      query: 'person(X) → human(X)',
      description: '推理实体的类型关系',
      strategy: 'rule_only'
    },
    {
      name: '语义相似度查询',
      query: 'similar_to(?, "artificial intelligence")',
      description: '查找与AI相关的实体',
      strategy: 'embedding_only'
    },
    {
      name: '多跳关系查询',
      query: 'path(?, works_at, ?, located_in, "San Francisco")',
      description: '查找在旧金山工作的人员',
      strategy: 'path_only'
    },
    {
      name: '概率推理',
      query: 'probability(rain | weather_forecast)',
      description: '基于天气预报的降雨概率',
      strategy: 'uncertainty_only'
    }
  ]

  const queryColumns = [
    {
      title: '查询ID',
      dataIndex: 'id',
      key: 'id',
      width: 100,
      render: (id: string) => <Text code>#{id}</Text>
    },
    {
      title: '查询表达式',
      dataIndex: 'query',
      key: 'query',
      render: (query: string) => (
        <Tooltip title={query}>
          <Text code style={{ maxWidth: 300, display: 'block', overflow: 'hidden', textOverflow: 'ellipsis' }}>
            {query}
          </Text>
        </Tooltip>
      )
    },
    {
      title: '策略',
      dataIndex: 'strategy',
      key: 'strategy',
      render: (strategy: string) => {
        const strategyConfig = strategyOptions.find(s => s.value === strategy)
        return <Tag color="blue">{strategyConfig?.label || strategy}</Tag>
      }
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Space>
          {getStatusIcon(status)}
          <Text>{getStatusText(status)}</Text>
        </Space>
      )
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      key: 'confidence',
      render: (confidence: number) => (
        <Progress 
          percent={confidence * 100} 
          size="small" 
          format={() => confidence.toFixed(2)}
          strokeColor={confidence > 0.8 ? '#52c41a' : confidence > 0.6 ? '#faad14' : '#ff4d4f'}
        />
      )
    },
    {
      title: '响应时间',
      dataIndex: 'responseTime',
      key: 'responseTime',
      render: (time: number) => <Text>{time > 0 ? `${time.toFixed(1)}s` : '-'}</Text>
    },
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (timestamp: string) => <Text type="secondary">{timestamp}</Text>
    },
  ]

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <SearchOutlined /> 推理查询中心
        </Title>
        <Paragraph>
          交互式知识图推理查询工具，支持多种推理策略和实时结果分析
        </Paragraph>
      </div>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="交互查询" key="interactive">
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={16}>
              <Card title="查询构建器" extra={
                <Space>
                  <Switch 
                    checked={realTimeMode} 
                    onChange={setRealTimeMode}
                    checkedChildren="实时"
                    unCheckedChildren="批量"
                  />
                  <Button 
                    type={showAdvanced ? 'primary' : 'default'} 
                    size="small" 
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    icon={<SettingOutlined />}
                  >
                    高级设置
                  </Button>
                </Space>
              }>
                <Form
                  form={form}
                  layout="vertical"
                  onFinish={handleQuerySubmit}
                  initialValues={{ strategy: 'auto', confidence_threshold: 0.8 }}
                >
                  <Row gutter={16}>
                    <Col span={18}>
                      <Form.Item 
                        label="推理查询" 
                        name="query" 
                        rules={[{ required: true, message: '请输入查询表达式' }]}
                      >
                        <TextArea 
                          rows={3}
                          placeholder="输入推理查询表达式，例如: person(X) → human(X)"
                          style={{ fontFamily: 'monospace' }}
                        />
                      </Form.Item>
                    </Col>
                    <Col span={6}>
                      <Form.Item label="推理策略" name="strategy">
                        <Select>
                          {strategyOptions.map(option => (
                            <Option key={option.value} value={option.value}>
                              <Tooltip title={option.description} placement="left">
                                {option.label}
                              </Tooltip>
                            </Option>
                          ))}
                        </Select>
                      </Form.Item>
                    </Col>
                  </Row>

                  {showAdvanced && (
                    <Collapse ghost>
                      <Panel header="高级参数配置" key="advanced">
                        <Row gutter={16}>
                          <Col span={8}>
                            <Form.Item label="置信度阈值" name="confidence_threshold">
                              <Slider min={0.1} max={1.0} step={0.1} marks={{ 0.5: '0.5', 0.8: '0.8', 1.0: '1.0' }} />
                            </Form.Item>
                          </Col>
                          <Col span={8}>
                            <Form.Item label="最大结果数" name="max_results">
                              <InputNumber min={1} max={1000} defaultValue={50} />
                            </Form.Item>
                          </Col>
                          <Col span={8}>
                            <Form.Item label="超时时间(秒)" name="timeout">
                              <InputNumber min={1} max={300} defaultValue={30} />
                            </Form.Item>
                          </Col>
                        </Row>
                        <Row gutter={16}>
                          <Col span={8}>
                            <Form.Item label="推理深度" name="max_depth">
                              <InputNumber min={1} max={10} defaultValue={3} />
                            </Form.Item>
                          </Col>
                          <Col span={8}>
                            <Form.Item name="enable_explanation" valuePropName="checked">
                              <Space>
                                <Switch size="small" />
                                <Text>启用解释生成</Text>
                              </Space>
                            </Form.Item>
                          </Col>
                          <Col span={8}>
                            <Form.Item name="cache_results" valuePropName="checked">
                              <Space>
                                <Switch size="small" defaultChecked />
                                <Text>缓存查询结果</Text>
                              </Space>
                            </Form.Item>
                          </Col>
                        </Row>
                      </Panel>
                    </Collapse>
                  )}

                  <Form.Item>
                    <Space>
                      <Button 
                        type="primary" 
                        htmlType="submit" 
                        loading={loading}
                        icon={<PlayCircleOutlined />}
                      >
                        执行查询
                      </Button>
                      {loading && (
                        <Button 
                          danger 
                          onClick={handleQueryStop}
                          icon={<StopOutlined />}
                        >
                          停止执行
                        </Button>
                      )}
                      <Button icon={<SaveOutlined />}>保存模板</Button>
                      <Button icon={<HistoryOutlined />}>查看历史</Button>
                    </Space>
                  </Form.Item>
                </Form>

                <Divider orientation="left">查询模板</Divider>
                <Row gutter={[8, 8]}>
                  {queryTemplates.map((template, index) => (
                    <Col key={index}>
                      <Button 
                        size="small" 
                        onClick={() => form.setFieldsValue({ 
                          query: template.query, 
                          strategy: template.strategy 
                        })}
                      >
                        {template.name}
                      </Button>
                    </Col>
                  ))}
                </Row>
              </Card>

              {/* 实时查询结果 */}
              {currentQuery && (
                <Card 
                  title={
                    <Space>
                      <Text>实时结果</Text>
                      {getStatusIcon(currentQuery.status)}
                      <Badge 
                        status={currentQuery.status === 'completed' ? 'success' : 
                               currentQuery.status === 'running' ? 'processing' : 'error'} 
                        text={getStatusText(currentQuery.status)} 
                      />
                    </Space>
                  }
                  extra={currentQuery.confidence > 0 && (
                    <Text>置信度: {(currentQuery.confidence * 100).toFixed(1)}%</Text>
                  )}
                  style={{ marginTop: '16px' }}
                >
                  <Tabs size="small">
                    <TabPane tab="查询结果" key="results">
                      {currentQuery.results.length > 0 ? (
                        <Table 
                          dataSource={currentQuery.results}
                          columns={[
                            { title: '实体', dataIndex: 'entity', key: 'entity' },
                            { title: '类型', dataIndex: 'type', key: 'type', 
                              render: (type: string) => <Tag>{type}</Tag> },
                            { title: '置信度', dataIndex: 'confidence', key: 'confidence',
                              render: (conf: number) => <Progress percent={conf * 100} size="small" /> }
                          ]}
                          pagination={false}
                          size="small"
                        />
                      ) : (
                        <Text type="secondary">
                          {currentQuery.status === 'running' ? '推理执行中...' : '暂无结果'}
                        </Text>
                      )}
                    </TabPane>
                    <TabPane tab="推理路径" key="reasoning">
                      <Timeline size="small">
                        {currentQuery.reasoning_path?.map((step, index) => (
                          <Timeline.Item 
                            key={index}
                            color={index === (currentQuery.reasoning_path?.length || 0) - 1 && 
                                   currentQuery.status === 'running' ? 'blue' : 'green'}
                          >
                            {step}
                            {index === (currentQuery.reasoning_path?.length || 0) - 1 && 
                             currentQuery.status === 'running' && <LoadingOutlined style={{ marginLeft: 8 }} />}
                          </Timeline.Item>
                        ))}
                      </Timeline>
                    </TabPane>
                    <TabPane tab="推理证据" key="evidence">
                      {currentQuery.evidence?.map((evidence, index) => (
                        <Alert 
                          key={index}
                          message={evidence}
                          type="info" 
                          size="small"
                          style={{ marginBottom: '8px' }}
                        />
                      ))}
                    </TabPane>
                  </Tabs>
                </Card>
              )}
            </Col>

            <Col xs={24} lg={8}>
              <Card title="策略建议" size="small" style={{ marginBottom: '16px' }}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Alert 
                    message="规则推理"
                    description="当前查询包含明确的逻辑关系，建议使用规则推理策略"
                    type="success"
                    showIcon
                    size="small"
                  />
                  <Alert 
                    message="性能提示"
                    description="实时模式下建议设置合理的超时时间"
                    type="info"
                    showIcon
                    size="small"
                  />
                </Space>
              </Card>

              <Card title="查询统计" size="small">
                <Space direction="vertical" style={{ width: '100%' }} size="small">
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Text>今日查询:</Text>
                    <Text strong>1,247</Text>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Text>成功率:</Text>
                    <Text strong style={{ color: '#52c41a' }}>94.2%</Text>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Text>平均响应:</Text>
                    <Text strong>1.8s</Text>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Text>正在执行:</Text>
                    <Text strong style={{ color: '#1890ff' }}>12</Text>
                  </div>
                </Space>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="查询历史" key="history">
          <Card title="历史查询记录" extra={
            <Space>
              <Button icon={<FilterOutlined />} size="small">筛选</Button>
              <Button icon={<ExportOutlined />} size="small">导出</Button>
            </Space>
          }>
            <Table 
              dataSource={queryHistory}
              columns={queryColumns}
              rowKey="id"
              pagination={{ 
                showSizeChanger: true, 
                showQuickJumper: true,
                showTotal: (total) => `共 ${total} 条记录`
              }}
              expandable={{
                expandedRowRender: (record: QueryResult) => (
                  <Row gutter={16}>
                    <Col span={12}>
                      <Card size="small" title="推理路径">
                        <Timeline size="small">
                          {record.reasoning_path?.map((step, index) => (
                            <Timeline.Item key={index}>{step}</Timeline.Item>
                          ))}
                        </Timeline>
                      </Card>
                    </Col>
                    <Col span={12}>
                      <Card size="small" title="证据信息">
                        {record.evidence?.map((evidence, index) => (
                          <div key={index} style={{ marginBottom: '4px' }}>
                            <Text type="secondary">• {evidence}</Text>
                          </div>
                        ))}
                      </Card>
                    </Col>
                  </Row>
                )
              }}
            />
          </Card>
        </TabPane>
      </Tabs>
    </div>
  )
}

export default KGReasoningQueryPage