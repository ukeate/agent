import React, { useState } from 'react'
import {
  Card,
  Row,
  Col,
  Button,
  Space,
  Table,
  Tag,
  Typography,
  Tabs,
  Alert,
  Progress,
  Timeline,
  Statistic,
  List,
  Tooltip,
  Modal,
  Form,
  Input,
  Select,
  InputNumber
} from 'antd'
import {
  PlayCircleOutlined,
  StopOutlined,
  ReloadOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ClockCircleOutlined,
  DatabaseOutlined,
  CloudOutlined,
  ThunderboltOutlined,
  SettingOutlined,
  BugOutlined,
  CodeOutlined
} from '@ant-design/icons'

const { Title, Text } = Typography
const { TabPane } = Tabs
const { Option } = Select
const { TextArea } = Input

interface TestCase {
  id: string
  name: string
  category: 'async-db' | 'redis' | 'concurrent' | 'mixed'
  status: 'pending' | 'running' | 'passed' | 'failed'
  duration: number
  lastRun: string
  description: string
  result?: any
  error?: string
}

interface TestMetric {
  name: string
  value: number
  unit: string
  status: 'good' | 'warning' | 'critical'
}

const IntegrationTestPage: React.FC = () => {
  const [testCases, setTestCases] = useState<TestCase[]>([
    {
      id: 'test-1',
      name: '异步数据库连接测试',
      category: 'async-db',
      status: 'passed',
      duration: 150,
      lastRun: '5分钟前',
      description: '测试PostgreSQL异步连接池的稳定性和性能',
      result: { connections: 10, avgResponseTime: 15 }
    },
    {
      id: 'test-2',
      name: '数据库事务隔离测试',
      category: 'async-db',
      status: 'passed',
      duration: 230,
      lastRun: '5分钟前',
      description: '验证数据库事务的ACID特性在异步环境下的表现'
    },
    {
      id: 'test-3',
      name: 'Redis缓存一致性测试',
      category: 'redis',
      status: 'passed',
      duration: 89,
      lastRun: '5分钟前',
      description: '测试Redis缓存在高并发场景下的数据一致性',
      result: { cacheHitRate: 95.2, operations: 1000 }
    },
    {
      id: 'test-4',
      name: 'Redis集群故障转移测试',
      category: 'redis',
      status: 'failed',
      duration: 180,
      lastRun: '5分钟前',
      description: '测试Redis集群节点故障时的自动切换能力',
      error: 'Master节点切换超时'
    },
    {
      id: 'test-5',
      name: '高并发请求处理测试',
      category: 'concurrent',
      status: 'running',
      duration: 0,
      lastRun: '进行中',
      description: '测试系统在1000并发请求下的稳定性'
    },
    {
      id: 'test-6',
      name: '混合异步操作测试',
      category: 'mixed',
      status: 'pending',
      duration: 0,
      lastRun: '未运行',
      description: '综合测试数据库、缓存、API调用的异步协调'
    }
  ])

  const [testMetrics] = useState<TestMetric[]>([
    { name: '测试覆盖率', value: 85, unit: '%', status: 'good' },
    { name: '平均响应时间', value: 150, unit: 'ms', status: 'good' },
    { name: '成功率', value: 75, unit: '%', status: 'warning' },
    { name: '并发处理能力', value: 1000, unit: 'req/s', status: 'good' }
  ])

  const [showConfigModal, setShowConfigModal] = useState(false)
  const [runningTests, setRunningTests] = useState<Set<string>>(new Set(['test-5']))

  const getStatusColor = (status: string) => {
    const colors = {
      pending: 'default',
      running: 'processing',
      passed: 'success',
      failed: 'error'
    }
    return colors[status as keyof typeof colors] || 'default'
  }

  const getStatusIcon = (status: string) => {
    const icons = {
      pending: <ClockCircleOutlined />,
      running: <PlayCircleOutlined />,
      passed: <CheckCircleOutlined />,
      failed: <CloseCircleOutlined />
    }
    return icons[status as keyof typeof icons]
  }

  const getCategoryIcon = (category: string) => {
    const icons = {
      'async-db': <DatabaseOutlined />,
      'redis': <CloudOutlined />,
      'concurrent': <ThunderboltOutlined />,
      'mixed': <CodeOutlined />
    }
    return icons[category as keyof typeof icons]
  }

  const getCategoryName = (category: string) => {
    const names = {
      'async-db': '异步数据库',
      'redis': 'Redis缓存',
      'concurrent': '并发测试',
      'mixed': '混合测试'
    }
    return names[category as keyof typeof names] || category
  }

  const getMetricStatusColor = (status: string) => {
    const colors = {
      good: '#52c41a',
      warning: '#faad14',
      critical: '#ff4d4f'
    }
    return colors[status as keyof typeof colors]
  }

  const runTest = (testId: string) => {
    setRunningTests(prev => new Set([...prev, testId]))
    setTestCases(prev => prev.map(test => 
      test.id === testId 
        ? { ...test, status: 'running', lastRun: '进行中' }
        : test
    ))

    // 模拟测试执行
    setTimeout(() => {
      const success = Math.random() > 0.3
      setTestCases(prev => prev.map(test => 
        test.id === testId 
          ? { 
              ...test, 
              status: success ? 'passed' : 'failed',
              duration: Math.floor(Math.random() * 300) + 50,
              lastRun: '刚刚',
              error: success ? undefined : '测试执行失败',
              result: success ? { success: true } : undefined
            }
          : test
      ))
      setRunningTests(prev => {
        const newSet = new Set(prev)
        newSet.delete(testId)
        return newSet
      })
    }, 3000)
  }

  const runAllTests = () => {
    testCases.forEach(test => {
      if (test.status !== 'running') {
        runTest(test.id)
      }
    })
  }

  const columns = [
    {
      title: '测试用例',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: TestCase) => (
        <div>
          <Space>
            {getCategoryIcon(record.category)}
            <Text strong>{name}</Text>
          </Space>
          <br />
          <Text type="secondary" className="text-xs">{record.description}</Text>
        </div>
      )
    },
    {
      title: '分类',
      dataIndex: 'category',
      key: 'category',
      render: (category: string) => (
        <Tag icon={getCategoryIcon(category)}>
          {getCategoryName(category)}
        </Tag>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={getStatusColor(status)} icon={getStatusIcon(status)}>
          {status === 'pending' && '等待'}
          {status === 'running' && '运行中'}
          {status === 'passed' && '通过'}
          {status === 'failed' && '失败'}
        </Tag>
      )
    },
    {
      title: '耗时',
      dataIndex: 'duration',
      key: 'duration',
      render: (duration: number, record: TestCase) => 
        record.status === 'running' ? (
          <Progress type="circle" size={30} />
        ) : (
          duration > 0 ? `${duration}ms` : '-'
        )
    },
    {
      title: '最后运行',
      dataIndex: 'lastRun',
      key: 'lastRun'
    },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: TestCase) => (
        <Space>
          <Tooltip title="运行测试">
            <Button 
              size="small" 
              icon={<PlayCircleOutlined />}
              onClick={() => runTest(record.id)}
              disabled={record.status === 'running'}
              loading={record.status === 'running'}
            />
          </Tooltip>
          {record.status === 'running' && (
            <Tooltip title="停止测试">
              <Button 
                size="small" 
                danger 
                icon={<StopOutlined />}
              />
            </Tooltip>
          )}
        </Space>
      )
    }
  ]

  const passedTests = testCases.filter(t => t.status === 'passed').length
  const totalTests = testCases.length
  const successRate = (passedTests / totalTests) * 100

  return (
    <div className="p-6">
      <div className="mb-6">
        <div className="flex justify-between items-center mb-4">
          <Title level={2}>集成测试 (异步数据库/Redis)</Title>
          <Space>
            <Button 
              type="primary" 
              icon={<PlayCircleOutlined />}
              onClick={runAllTests}
              disabled={runningTests.size > 0}
            >
              运行所有测试
            </Button>
            <Button 
              icon={<SettingOutlined />}
              onClick={() => setShowConfigModal(true)}
            >
              测试配置
            </Button>
            <Button 
              icon={<BugOutlined />}
              onClick={() => console.log('生成测试报告')}
            >
              生成报告
            </Button>
          </Space>
        </div>

        {successRate < 80 && (
          <Alert
            message="测试成功率偏低"
            description={`当前测试成功率为 ${successRate.toFixed(1)}%，建议检查失败的测试用例`}
            variant="warning"
            showIcon
            closable
            className="mb-4"
          />
        )}

        <Row gutter={16} className="mb-6">
          <Col span={6}>
            <Card>
              <Statistic
                title="测试成功率"
                value={successRate}
                precision={1}
                suffix="%"
                valueStyle={{ 
                  color: successRate > 90 ? '#3f8600' : successRate > 70 ? '#faad14' : '#cf1322' 
                }}
                prefix={<CheckCircleOutlined />}
              />
              <div className="mt-2 text-xs text-gray-500">
                {passedTests} / {totalTests} 个测试通过
              </div>
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="运行中测试"
                value={runningTests.size}
                valueStyle={{ color: runningTests.size > 0 ? '#1890ff' : '#666' }}
                prefix={<PlayCircleOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="失败测试"
                value={testCases.filter(t => t.status === 'failed').length}
                valueStyle={{ color: '#cf1322' }}
                prefix={<CloseCircleOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="平均耗时"
                value={Math.round(testCases.filter(t => t.duration > 0).reduce((sum, t) => sum + t.duration, 0) / testCases.filter(t => t.duration > 0).length || 0)}
                suffix="ms"
                prefix={<ClockCircleOutlined />}
              />
            </Card>
          </Col>
        </Row>
      </div>

      <Tabs defaultActiveKey="tests">
        <TabPane tab="测试用例" key="tests">
          <Card>
            <Table
              columns={columns}
              dataSource={testCases}
              rowKey="id"
              pagination={false}
              size="small"
            />
          </Card>
        </TabPane>

        <TabPane tab="性能指标" key="metrics">
          <Row gutter={16}>
            {testMetrics.map((metric, index) => (
              <Col span={6} key={index} className="mb-4">
                <Card>
                  <div className="text-center">
                    <div 
                      className="text-3xl font-bold mb-2"
                      style={{ color: getMetricStatusColor(metric.status) }}
                    >
                      {metric.value}{metric.unit}
                    </div>
                    <Text>{metric.name}</Text>
                    <br />
                    <Tag color={metric.status === 'good' ? 'green' : metric.status === 'warning' ? 'orange' : 'red'}>
                      {metric.status === 'good' && '良好'}
                      {metric.status === 'warning' && '警告'}
                      {metric.status === 'critical' && '严重'}
                    </Tag>
                  </div>
                </Card>
              </Col>
            ))}
          </Row>
        </TabPane>

        <TabPane tab="测试结果" key="results">
          <Row gutter={16}>
            <Col span={12}>
              <Card title="测试详情">
                <List
                  dataSource={testCases.filter(t => t.status !== 'pending')}
                  renderItem={(test) => (
                    <List.Item>
                      <List.Item.Meta
                        avatar={getStatusIcon(test.status)}
                        title={
                          <Space>
                            <Text strong={test.status === 'failed'}>{test.name}</Text>
                            <Tag color={getStatusColor(test.status)}>
                              {test.status === 'passed' && '通过'}
                              {test.status === 'failed' && '失败'}
                              {test.status === 'running' && '运行中'}
                            </Tag>
                          </Space>
                        }
                        description={
                          <div>
                            <div>耗时: {test.duration}ms</div>
                            {test.error && (
                              <Text type="danger">错误: {test.error}</Text>
                            )}
                            {test.result && (
                              <Text type="success">
                                结果: {JSON.stringify(test.result)}
                              </Text>
                            )}
                          </div>
                        }
                      />
                    </List.Item>
                  )}
                />
              </Card>
            </Col>
            <Col span={12}>
              <Card title="执行日志">
                <Timeline
                  items={[
                    {
                      color: 'green',
                      children: (
                        <div>
                          <Text strong>异步数据库连接测试完成</Text>
                          <br />
                          <Text type="secondary">连接池测试通过，平均响应15ms - 5分钟前</Text>
                        </div>
                      )
                    },
                    {
                      color: 'red',
                      children: (
                        <div>
                          <Text strong>Redis集群故障转移测试失败</Text>
                          <br />
                          <Text type="secondary">Master节点切换超时 - 5分钟前</Text>
                        </div>
                      )
                    },
                    {
                      color: 'blue',
                      children: (
                        <div>
                          <Text strong>开始高并发请求处理测试</Text>
                          <br />
                          <Text type="secondary">启动1000并发请求测试 - 刚刚</Text>
                        </div>
                      )
                    }
                  ]}
                />
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="测试分类" key="categories">
          <Row gutter={16}>
            <Col span={8}>
              <Card title="异步数据库测试" extra={<DatabaseOutlined />}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>连接池稳定性测试</div>
                  <div>事务隔离级别测试</div>
                  <div>数据一致性验证</div>
                  <div>异步查询性能测试</div>
                </Space>
              </Card>
            </Col>
            <Col span={8}>
              <Card title="Redis缓存测试" extra={<CloudOutlined />}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>缓存一致性测试</div>
                  <div>集群故障转移测试</div>
                  <div>内存使用优化测试</div>
                  <div>键过期策略测试</div>
                </Space>
              </Card>
            </Col>
            <Col span={8}>
              <Card title="并发与混合测试" extra={<ThunderboltOutlined />}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>高并发请求处理</div>
                  <div>混合异步操作协调</div>
                  <div>资源竞争处理</div>
                  <div>系统稳定性验证</div>
                </Space>
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>

      {/* 测试配置Modal */}
      <Modal
        title="测试配置"
        open={showConfigModal}
        onCancel={() => setShowConfigModal(false)}
        footer={null}
        width={600}
      >
        <Form layout="vertical">
          <Form.Item label="并发数量">
            <InputNumber min={1} max={10000} defaultValue={1000} />
          </Form.Item>
          <Form.Item label="测试超时时间(秒)">
            <InputNumber min={1} max={3600} defaultValue={300} />
          </Form.Item>
          <Form.Item label="数据库连接池大小">
            <InputNumber min={1} max={100} defaultValue={10} />
          </Form.Item>
          <Form.Item label="Redis连接配置">
            <TextArea rows={3} defaultValue="host=localhost\nport=6379\ntimeout=5000" />
          </Form.Item>
          <Form.Item label="测试环境">
            <Select defaultValue="development">
              <Option value="development">开发环境</Option>
              <Option value="staging">测试环境</Option>
              <Option value="production">生产环境</Option>
            </Select>
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary">保存配置</Button>
              <Button onClick={() => setShowConfigModal(false)}>取消</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default IntegrationTestPage