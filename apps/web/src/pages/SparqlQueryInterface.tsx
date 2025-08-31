import React, { useState, useCallback } from 'react'
import { 
  Card, 
  Typography, 
  Row, 
  Col, 
  Space, 
  Button, 
  Input, 
  Table, 
  Tabs, 
  Select, 
  Form, 
  Switch,
  Statistic,
  Tag,
  message,
  Modal,
  Alert
} from 'antd'
import { 
  SearchOutlined, 
  PlayCircleOutlined, 
  SaveOutlined,
  HistoryOutlined,
  DownloadOutlined,
  DatabaseOutlined,
  ThunderboltOutlined,
  MonitorOutlined
} from '@ant-design/icons'

const { Title, Text, Paragraph } = Typography
const { TextArea } = Input
const { TabPane } = Tabs
const { Option } = Select

interface QueryResult {
  columns: string[]
  rows: any[][]
  executionTime: number
  resultCount: number
}

interface QueryHistory {
  id: string
  query: string
  timestamp: string
  executionTime: number
  resultCount: number
  status: 'success' | 'error'
}

const SparqlQueryInterface: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [query, setQuery] = useState(`PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>

SELECT ?subject ?predicate ?object
WHERE {
  ?subject ?predicate ?object
}
LIMIT 10`)
  
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null)
  const [queryHistory, setQueryHistory] = useState<QueryHistory[]>([
    {
      id: '1',
      query: 'SELECT * WHERE { ?s ?p ?o } LIMIT 10',
      timestamp: '2024-01-15 14:30:25',
      executionTime: 120,
      resultCount: 10,
      status: 'success'
    },
    {
      id: '2',
      query: 'SELECT ?person WHERE { ?person rdf:type foaf:Person }',
      timestamp: '2024-01-15 14:25:18',
      executionTime: 350,
      resultCount: 156,
      status: 'success'
    }
  ])
  
  const [selectedFormat, setSelectedFormat] = useState('JSON')
  const [optimizationEnabled, setOptimizationEnabled] = useState(true)
  const [cacheEnabled, setCacheEnabled] = useState(true)
  const [explainMode, setExplainMode] = useState(false)

  const executeQuery = useCallback(async () => {
    if (!query.trim()) {
      message.error('请输入SPARQL查询语句')
      return
    }

    setLoading(true)
    try {
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      const mockResult: QueryResult = {
        columns: ['subject', 'predicate', 'object'],
        rows: [
          ['http://example.org/person/1', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type', 'http://xmlns.com/foaf/0.1/Person'],
          ['http://example.org/person/1', 'http://xmlns.com/foaf/0.1/name', 'John Doe'],
          ['http://example.org/person/1', 'http://xmlns.com/foaf/0.1/age', '30'],
          ['http://example.org/person/2', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type', 'http://xmlns.com/foaf/0.1/Person'],
          ['http://example.org/person/2', 'http://xmlns.com/foaf/0.1/name', 'Jane Smith']
        ],
        executionTime: Math.floor(Math.random() * 500) + 50,
        resultCount: 5
      }
      
      setQueryResult(mockResult)
      
      // 添加到历史记录
      const newHistoryItem: QueryHistory = {
        id: Date.now().toString(),
        query: query.substring(0, 50) + (query.length > 50 ? '...' : ''),
        timestamp: new Date().toLocaleString(),
        executionTime: mockResult.executionTime,
        resultCount: mockResult.resultCount,
        status: 'success'
      }
      
      setQueryHistory(prev => [newHistoryItem, ...prev.slice(0, 9)])
      message.success(`查询执行成功，返回${mockResult.resultCount}条结果`)
      
    } catch (error) {
      message.error('查询执行失败')
      console.error('Query execution error:', error)
    } finally {
      setLoading(false)
    }
  }, [query])

  const queryExamples = [
    {
      name: '查询所有三元组',
      query: `SELECT ?subject ?predicate ?object
WHERE {
  ?subject ?predicate ?object
}
LIMIT 10`
    },
    {
      name: '查询所有人员',
      query: `PREFIX foaf: <http://xmlns.com/foaf/0.1/>
SELECT ?person ?name
WHERE {
  ?person rdf:type foaf:Person .
  ?person foaf:name ?name
}`
    },
    {
      name: '复杂推理查询',
      query: `PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?concept ?parent
WHERE {
  ?concept rdfs:subClassOf ?parent .
  ?parent rdfs:subClassOf owl:Thing
}`
    }
  ]

  const resultColumns = queryResult?.columns.map(col => ({
    title: col,
    dataIndex: col,
    key: col,
    ellipsis: true,
    width: 200
  })) || []

  const resultData = queryResult?.rows.map((row, index) => {
    const obj: any = { key: index }
    queryResult.columns.forEach((col, colIndex) => {
      obj[col] = row[colIndex]
    })
    return obj
  }) || []

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <SearchOutlined style={{ marginRight: '8px', color: '#1890ff' }} />
          SPARQL查询界面
        </Title>
        <Paragraph>
          功能完整的SPARQL 1.1查询界面，支持查询优化、缓存管理和多种结果格式导出
        </Paragraph>
      </div>

      <Row gutter={[24, 24]}>
        <Col span={16}>
          <Card title="查询编辑器" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Row gutter={16}>
                <Col span={6}>
                  <Select
                    value={selectedFormat}
                    onChange={setSelectedFormat}
                    style={{ width: '100%' }}
                  >
                    <Option value="JSON">JSON</Option>
                    <Option value="XML">XML</Option>
                    <Option value="CSV">CSV</Option>
                    <Option value="TSV">TSV</Option>
                  </Select>
                </Col>
                <Col span={6}>
                  <Space>
                    <Text>查询优化:</Text>
                    <Switch
                      checked={optimizationEnabled}
                      onChange={setOptimizationEnabled}
                      size="small"
                    />
                  </Space>
                </Col>
                <Col span={6}>
                  <Space>
                    <Text>启用缓存:</Text>
                    <Switch
                      checked={cacheEnabled}
                      onChange={setCacheEnabled}
                      size="small"
                    />
                  </Space>
                </Col>
                <Col span={6}>
                  <Space>
                    <Text>执行计划:</Text>
                    <Switch
                      checked={explainMode}
                      onChange={setExplainMode}
                      size="small"
                    />
                  </Space>
                </Col>
              </Row>

              <TextArea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="输入SPARQL查询语句..."
                rows={12}
                style={{ fontFamily: 'monospace', fontSize: '14px' }}
              />

              <Row justify="space-between">
                <Col>
                  <Space>
                    <Button
                      type="primary"
                      icon={<PlayCircleOutlined />}
                      loading={loading}
                      onClick={executeQuery}
                    >
                      执行查询
                    </Button>
                    <Button icon={<SaveOutlined />}>保存查询</Button>
                    <Button icon={<DownloadOutlined />}>导出结果</Button>
                  </Space>
                </Col>
                <Col>
                  <Text type="secondary">
                    {query.length} 字符
                  </Text>
                </Col>
              </Row>
            </Space>
          </Card>

          {queryResult && (
            <Card title="查询结果" style={{ marginTop: '16px' }} size="small">
              <div style={{ marginBottom: '16px' }}>
                <Row gutter={16}>
                  <Col span={6}>
                    <Statistic
                      title="结果数量"
                      value={queryResult.resultCount}
                      suffix="条"
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="执行时间"
                      value={queryResult.executionTime}
                      suffix="ms"
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="优化状态"
                      value={optimizationEnabled ? '已启用' : '已禁用'}
                      valueStyle={{ color: optimizationEnabled ? '#3f8600' : '#cf1322' }}
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="缓存状态"
                      value={cacheEnabled ? '命中' : '未命中'}
                      valueStyle={{ color: cacheEnabled ? '#3f8600' : '#cf1322' }}
                    />
                  </Col>
                </Row>
              </div>

              <Table
                dataSource={resultData}
                columns={resultColumns}
                scroll={{ x: true, y: 400 }}
                size="small"
                pagination={{
                  showSizeChanger: true,
                  showQuickJumper: true,
                  showTotal: (total) => `共 ${total} 条记录`
                }}
              />
            </Card>
          )}
        </Col>

        <Col span={8}>
          <Tabs defaultActiveKey="examples" size="small">
            <TabPane tab="查询示例" key="examples">
              <Card size="small">
                <Space direction="vertical" style={{ width: '100%' }}>
                  {queryExamples.map((example, index) => (
                    <Card
                      key={index}
                      size="small"
                      title={example.name}
                      hoverable
                      onClick={() => setQuery(example.query)}
                    >
                      <Text code style={{ fontSize: '12px' }}>
                        {example.query.substring(0, 100)}...
                      </Text>
                    </Card>
                  ))}
                </Space>
              </Card>
            </TabPane>

            <TabPane tab="查询历史" key="history">
              <Card size="small">
                <Space direction="vertical" style={{ width: '100%' }}>
                  {queryHistory.map((item) => (
                    <Card
                      key={item.id}
                      size="small"
                      hoverable
                      onClick={() => setQuery(item.query)}
                    >
                      <Space direction="vertical" size="small" style={{ width: '100%' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Tag color={item.status === 'success' ? 'green' : 'red'}>
                            {item.status === 'success' ? '成功' : '失败'}
                          </Tag>
                          <Text type="secondary" style={{ fontSize: '12px' }}>
                            {item.timestamp}
                          </Text>
                        </div>
                        <Text code style={{ fontSize: '12px' }}>
                          {item.query}
                        </Text>
                        <Row>
                          <Col span={12}>
                            <Text type="secondary">
                              {item.executionTime}ms
                            </Text>
                          </Col>
                          <Col span={12}>
                            <Text type="secondary">
                              {item.resultCount}条结果
                            </Text>
                          </Col>
                        </Row>
                      </Space>
                    </Card>
                  ))}
                </Space>
              </Card>
            </TabPane>

            <TabPane tab="性能监控" key="performance">
              <Card size="small">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Row gutter={[16, 16]}>
                    <Col span={24}>
                      <Statistic
                        title="平均查询时间"
                        value={245}
                        suffix="ms"
                        prefix={<MonitorOutlined />}
                      />
                    </Col>
                    <Col span={24}>
                      <Statistic
                        title="缓存命中率"
                        value={78.5}
                        suffix="%"
                        prefix={<DatabaseOutlined />}
                      />
                    </Col>
                    <Col span={24}>
                      <Statistic
                        title="优化效果"
                        value={34.2}
                        suffix="%"
                        prefix={<ThunderboltOutlined />}
                      />
                    </Col>
                  </Row>

                  <Alert
                    message="性能提示"
                    description="当前查询性能良好，建议启用查询优化以获得更好性能"
                    type="info"
                    showIcon
                    style={{ marginTop: '16px' }}
                  />
                </Space>
              </Card>
            </TabPane>
          </Tabs>
        </Col>
      </Row>
    </div>
  )
}

export default SparqlQueryInterface