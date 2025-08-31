import React, { useState, useEffect } from 'react'
import {
  Card,
  Input,
  Button,
  Space,
  Table,
  Alert,
  Tooltip,
  Row,
  Col,
  Statistic,
  Tabs,
  Typography,
  Divider,
  Tag,
  Progress,
  Select,
  message,
  Spin,
  Empty
} from 'antd'
import {
  PlayCircleOutlined,
  SaveOutlined,
  HistoryOutlined,
  DatabaseOutlined,
  NodeIndexOutlined,
  ShareAltOutlined,
  ClockCircleOutlined,
  ThunderboltOutlined,
  FileTextOutlined,
  CopyOutlined,
  DeleteOutlined,
  ExportOutlined
} from '@ant-design/icons'

const { Title, Text, Paragraph } = Typography
const { TextArea } = Input
const { TabPane } = Tabs

interface QueryResult {
  columns: string[]
  data: any[][]
  stats: {
    nodes_created: number
    nodes_deleted: number
    relationships_created: number
    relationships_deleted: number
    properties_set: number
    execution_time: number
    records_returned: number
  }
}

interface QueryHistory {
  id: string
  query: string
  timestamp: string
  execution_time: number
  status: 'success' | 'error'
  error?: string
  result_count: number
}

interface QueryTemplate {
  id: string
  name: string
  description: string
  query: string
  category: string
  parameters?: string[]
}

const KnowledgeGraphQueryEngine: React.FC = () => {
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null)
  const [queryHistory, setQueryHistory] = useState<QueryHistory[]>([])
  const [queryTemplates, setQueryTemplates] = useState<QueryTemplate[]>([])
  const [selectedTemplate, setSelectedTemplate] = useState<string>('')
  const [activeTab, setActiveTab] = useState('query')

  // 模拟查询模板
  const mockTemplates: QueryTemplate[] = [
    {
      id: 'template_001',
      name: '查找所有实体',
      description: '获取知识图谱中的所有实体节点',
      query: 'MATCH (n) RETURN n LIMIT 100',
      category: '基础查询'
    },
    {
      id: 'template_002',
      name: '查找实体关系',
      description: '查找两个实体之间的所有关系',
      query: 'MATCH (a)-[r]->(b) WHERE a.name = $entity1 AND b.name = $entity2 RETURN a, r, b',
      category: '关系查询',
      parameters: ['entity1', 'entity2']
    },
    {
      id: 'template_003',
      name: '度中心性分析',
      description: '分析节点的连接度',
      query: 'MATCH (n) RETURN n.name as entity, size((n)--()) as degree ORDER BY degree DESC LIMIT 20',
      category: '图分析'
    },
    {
      id: 'template_004',
      name: '路径查询',
      description: '查找两个实体之间的最短路径',
      query: 'MATCH path = shortestPath((a {name: $start})-[*]-(b {name: $end})) RETURN path',
      category: '路径分析',
      parameters: ['start', 'end']
    },
    {
      id: 'template_005',
      name: '子图提取',
      description: '提取特定类型实体的子图',
      query: 'MATCH (n:Person)-[r]-(m) RETURN n, r, m LIMIT 50',
      category: '子图查询'
    }
  ]

  // 模拟查询历史
  const mockHistory: QueryHistory[] = [
    {
      id: 'hist_001',
      query: 'MATCH (n:Person) RETURN n.name, n.age ORDER BY n.age DESC LIMIT 10',
      timestamp: '2025-01-22T14:30:00Z',
      execution_time: 156,
      status: 'success',
      result_count: 10
    },
    {
      id: 'hist_002',
      query: 'MATCH (a:Company)-[r:EMPLOYS]->(b:Person) RETURN a.name, count(b) as employees ORDER BY employees DESC',
      timestamp: '2025-01-22T14:25:00Z',
      execution_time: 245,
      status: 'success',
      result_count: 15
    },
    {
      id: 'hist_003',
      query: 'MATCH (n:Person {name: "张三"})-[r*1..3]-(m) RETURN n, r, m',
      timestamp: '2025-01-22T14:20:00Z',
      execution_time: 892,
      status: 'error',
      error: 'Node with name "张三" not found',
      result_count: 0
    }
  ]

  useEffect(() => {
    setQueryTemplates(mockTemplates)
    setQueryHistory(mockHistory)
  }, [])

  const executeQuery = async () => {
    if (!query.trim()) {
      message.error('请输入查询语句')
      return
    }

    setLoading(true)
    try {
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1500))
      
      // 模拟查询结果
      const mockResult: QueryResult = {
        columns: ['entity.name', 'entity.type', 'entity.confidence'],
        data: [
          ['张三', 'PERSON', 0.95],
          ['苹果公司', 'ORGANIZATION', 0.98],
          ['北京', 'LOCATION', 0.99],
          ['Python', 'TECHNOLOGY', 0.92]
        ],
        stats: {
          nodes_created: 0,
          nodes_deleted: 0,
          relationships_created: 0,
          relationships_deleted: 0,
          properties_set: 0,
          execution_time: Math.floor(Math.random() * 1000) + 100,
          records_returned: 4
        }
      }

      setQueryResult(mockResult)

      // 添加到历史记录
      const newHistory: QueryHistory = {
        id: `hist_${Date.now()}`,
        query,
        timestamp: new Date().toISOString(),
        execution_time: mockResult.stats.execution_time,
        status: 'success',
        result_count: mockResult.stats.records_returned
      }
      setQueryHistory([newHistory, ...queryHistory])

      message.success('查询执行成功')
    } catch (error) {
      message.error('查询执行失败')
    } finally {
      setLoading(false)
    }
  }

  const loadTemplate = (templateId: string) => {
    const template = queryTemplates.find(t => t.id === templateId)
    if (template) {
      setQuery(template.query)
      setSelectedTemplate(templateId)
      message.success(`已加载模板: ${template.name}`)
    }
  }

  const loadHistoryQuery = (historyQuery: string) => {
    setQuery(historyQuery)
    setActiveTab('query')
    message.success('已加载历史查询')
  }

  const exportResults = () => {
    if (!queryResult) {
      message.error('没有可导出的结果')
      return
    }
    // 模拟导出功能
    message.success('结果已导出到CSV文件')
  }

  const clearHistory = () => {
    setQueryHistory([])
    message.success('历史记录已清空')
  }

  const getStatusColor = (status: string) => {
    return status === 'success' ? 'success' : 'error'
  }

  const resultColumns = queryResult ? queryResult.columns.map((col, index) => ({
    title: col,
    dataIndex: index,
    key: col,
    render: (value: any) => {
      if (typeof value === 'number' && value < 1 && value > 0) {
        return <Progress percent={Math.round(value * 100)} size="small" />
      }
      return <Text>{String(value)}</Text>
    }
  })) : []

  const resultData = queryResult ? queryResult.data.map((row, index) => ({
    key: index,
    ...row.reduce((obj, val, idx) => ({ ...obj, [idx]: val }), {})
  })) : []

  const historyColumns = [
    {
      title: '查询语句',
      dataIndex: 'query',
      key: 'query',
      render: (text: string) => (
        <div style={{ maxWidth: '300px' }}>
          <Text ellipsis={{ tooltip: text }} code>{text}</Text>
        </div>
      ),
    },
    {
      title: '执行时间',
      dataIndex: 'execution_time',
      key: 'execution_time',
      render: (time: number) => <Text>{time}ms</Text>,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string, record: QueryHistory) => (
        <Tag color={getStatusColor(status)}>
          {status === 'success' ? '成功' : '失败'}
          {record.error && (
            <Tooltip title={record.error}>
              <Text type="danger" style={{ marginLeft: 4 }}>⚠</Text>
            </Tooltip>
          )}
        </Tag>
      ),
    },
    {
      title: '结果数',
      dataIndex: 'result_count',
      key: 'result_count',
    },
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (time: string) => new Date(time).toLocaleString(),
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record: QueryHistory) => (
        <Space>
          <Tooltip title="重新执行">
            <Button 
              type="text" 
              icon={<CopyOutlined />}
              onClick={() => loadHistoryQuery(record.query)}
            />
          </Tooltip>
        </Space>
      ),
    },
  ]

  const templatesByCategory = queryTemplates.reduce((acc, template) => {
    if (!acc[template.category]) acc[template.category] = []
    acc[template.category].push(template)
    return acc
  }, {} as Record<string, QueryTemplate[]>)

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <DatabaseOutlined style={{ marginRight: '8px' }} />
          图查询引擎
        </Title>
        <Paragraph type="secondary">
          使用Cypher查询语言对知识图谱进行复杂查询和分析
        </Paragraph>
      </div>

      {/* 统计信息 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="总查询次数"
              value={queryHistory.length}
              prefix={<DatabaseOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="成功查询"
              value={queryHistory.filter(h => h.status === 'success').length}
              valueStyle={{ color: '#3f8600' }}
              prefix={<ThunderboltOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均执行时间"
              value={queryHistory.length > 0 
                ? Math.round(queryHistory.reduce((sum, h) => sum + h.execution_time, 0) / queryHistory.length)
                : 0}
              suffix="ms"
              valueStyle={{ color: '#1890ff' }}
              prefix={<ClockCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="可用模板"
              value={queryTemplates.length}
              prefix={<FileTextOutlined />}
            />
          </Card>
        </Col>
      </Row>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="查询执行" key="query">
          <Row gutter={16}>
            <Col span={16}>
              <Card title="Cypher查询编辑器" style={{ marginBottom: '16px' }}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <TextArea
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="输入Cypher查询语句..."
                    rows={8}
                    style={{ fontFamily: 'Monaco, monospace' }}
                  />
                  <Row justify="space-between">
                    <Col>
                      <Space>
                        <Button 
                          type="primary" 
                          icon={<PlayCircleOutlined />}
                          onClick={executeQuery}
                          loading={loading}
                        >
                          执行查询
                        </Button>
                        <Button icon={<SaveOutlined />}>保存查询</Button>
                        <Button onClick={() => setQuery('')}>清空</Button>
                      </Space>
                    </Col>
                    <Col>
                      <Text type="secondary">支持Cypher语法高亮和自动补全</Text>
                    </Col>
                  </Row>
                </Space>
              </Card>

              {/* 查询结果 */}
              {queryResult && (
                <Card 
                  title="查询结果" 
                  extra={
                    <Space>
                      <Button icon={<ExportOutlined />} onClick={exportResults}>导出</Button>
                    </Space>
                  }
                >
                  <div style={{ marginBottom: '16px' }}>
                    <Row gutter={16}>
                      <Col span={6}>
                        <Statistic
                          title="返回记录"
                          value={queryResult.stats.records_returned}
                          prefix={<NodeIndexOutlined />}
                        />
                      </Col>
                      <Col span={6}>
                        <Statistic
                          title="执行时间"
                          value={queryResult.stats.execution_time}
                          suffix="ms"
                          prefix={<ClockCircleOutlined />}
                        />
                      </Col>
                      <Col span={6}>
                        <Statistic
                          title="创建节点"
                          value={queryResult.stats.nodes_created}
                        />
                      </Col>
                      <Col span={6}>
                        <Statistic
                          title="创建关系"
                          value={queryResult.stats.relationships_created}
                        />
                      </Col>
                    </Row>
                  </div>
                  
                  <Table
                    columns={resultColumns}
                    dataSource={resultData}
                    pagination={{
                      pageSize: 10,
                      showSizeChanger: true,
                      showTotal: (total) => `共 ${total} 条记录`
                    }}
                    scroll={{ x: true }}
                  />
                </Card>
              )}

              {loading && (
                <Card>
                  <div style={{ textAlign: 'center', padding: '50px' }}>
                    <Spin size="large" />
                    <div style={{ marginTop: '16px' }}>
                      <Text>正在执行查询...</Text>
                    </div>
                  </div>
                </Card>
              )}
            </Col>

            <Col span={8}>
              <Card title="查询模板" style={{ marginBottom: '16px' }}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  {Object.entries(templatesByCategory).map(([category, templates]) => (
                    <div key={category}>
                      <Divider orientation="left" style={{ margin: '12px 0 8px 0' }}>
                        <Text strong>{category}</Text>
                      </Divider>
                      {templates.map(template => (
                        <Card 
                          key={template.id}
                          size="small" 
                          style={{ marginBottom: '8px', cursor: 'pointer' }}
                          onClick={() => loadTemplate(template.id)}
                          hoverable
                        >
                          <Text strong>{template.name}</Text>
                          <br />
                          <Text type="secondary" style={{ fontSize: '12px' }}>
                            {template.description}
                          </Text>
                          {template.parameters && (
                            <div style={{ marginTop: '8px' }}>
                              {template.parameters.map(param => (
                                <Tag key={param} size="small">${param}</Tag>
                              ))}
                            </div>
                          )}
                        </Card>
                      ))}
                    </div>
                  ))}
                </Space>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="查询历史" key="history">
          <Card 
            title="查询历史记录"
            extra={
              <Button 
                danger 
                icon={<DeleteOutlined />}
                onClick={clearHistory}
              >
                清空历史
              </Button>
            }
          >
            {queryHistory.length > 0 ? (
              <Table
                columns={historyColumns}
                dataSource={queryHistory}
                rowKey="id"
                pagination={{
                  pageSize: 10,
                  showTotal: (total) => `共 ${total} 条历史记录`
                }}
              />
            ) : (
              <Empty description="暂无查询历史" />
            )}
          </Card>
        </TabPane>

        <TabPane tab="图可视化" key="visualization">
          <Card title="图可视化">
            <Alert
              message="图可视化功能"
              description="此区域将显示查询结果的图形化表示，包括节点和边的可视化布局。支持交互式探索和子图展开。"
              type="info"
              showIcon
              style={{ marginBottom: '16px' }}
            />
            <div style={{ 
              height: '400px', 
              border: '1px dashed #d9d9d9', 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              backgroundColor: '#fafafa'
            }}>
              <div style={{ textAlign: 'center' }}>
                <NodeIndexOutlined style={{ fontSize: '48px', color: '#d9d9d9' }} />
                <div style={{ marginTop: '16px' }}>
                  <Text type="secondary">执行查询后将在此显示图可视化结果</Text>
                </div>
              </div>
            </div>
          </Card>
        </TabPane>
      </Tabs>
    </div>
  )
}

export default KnowledgeGraphQueryEngine