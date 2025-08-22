import React, { useState, useEffect } from 'react'
import { 
  Card, 
  Row, 
  Col, 
  Button, 
  Space, 
  Table, 
  Progress,
  Tag,
  Statistic,
  Alert,
  Typography,
  Divider,
  Tabs,
  List,
  Timeline,
  Badge,
  Tooltip,
  Modal,
  Form,
  Input,
  Select,
  Switch,
  Result,
  Collapse
} from 'antd'
import { 
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  ExperimentOutlined,
  BugOutlined,
  SafetyOutlined,
  ThunderboltOutlined,
  RocketOutlined,
  FieldTimeOutlined,
  DashboardOutlined,
  WarningOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined
} from '@ant-design/icons'
import { 
  LineChart, 
  Line, 
  BarChart,
  Bar,
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Legend, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Tooltip as RechartsTooltip
} from 'recharts'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs
const { Panel } = Collapse
const { Option } = Select

// 测试类型枚举
enum TestType {
  UNIT = 'unit',
  INTEGRATION = 'integration',
  PERFORMANCE = 'performance',
  LOAD = 'load',
  STRESS = 'stress',
  SECURITY = 'security',
  PENETRATION = 'penetration',
  E2E = 'e2e'
}

// 测试状态枚举
enum TestStatus {
  PENDING = 'pending',
  RUNNING = 'running',
  PASSED = 'passed',
  FAILED = 'failed',
  SKIPPED = 'skipped'
}

// 生成测试套件数据
const generateTestSuites = () => {
  const suites = []
  const types = Object.values(TestType)
  
  for (let i = 0; i < 12; i++) {
    const totalTests = Math.floor(Math.random() * 100) + 20
    const passedTests = Math.floor(totalTests * (0.7 + Math.random() * 0.3))
    const failedTests = Math.floor((totalTests - passedTests) * Math.random())
    const skippedTests = totalTests - passedTests - failedTests
    
    suites.push({
      id: `suite_${i + 1}`,
      name: `测试套件 ${i + 1}`,
      type: types[Math.floor(Math.random() * types.length)],
      totalTests,
      passedTests,
      failedTests,
      skippedTests,
      coverage: Math.random() * 30 + 70,
      duration: Math.floor(Math.random() * 300) + 10,
      lastRun: new Date(Date.now() - Math.random() * 7 * 24 * 3600 * 1000),
      status: passedTests === totalTests ? TestStatus.PASSED : 
              failedTests > 0 ? TestStatus.FAILED : TestStatus.PASSED
    })
  }
  
  return suites
}

// 生成性能测试数据
const generatePerformanceData = () => {
  const data = []
  
  for (let i = 0; i < 24; i++) {
    data.push({
      time: `${i}:00`,
      responseTime: Math.random() * 200 + 50,
      throughput: Math.random() * 1000 + 500,
      errorRate: Math.random() * 5,
      cpuUsage: Math.random() * 80 + 10,
      memoryUsage: Math.random() * 70 + 20
    })
  }
  
  return data
}

// 生成测试结果历史
const generateTestHistory = () => {
  const history = []
  
  for (let i = 0; i < 10; i++) {
    history.push({
      id: `run_${i + 1}`,
      timestamp: new Date(Date.now() - i * 24 * 3600 * 1000),
      totalTests: Math.floor(Math.random() * 500) + 200,
      passRate: Math.random() * 20 + 80,
      coverage: Math.random() * 20 + 75,
      duration: Math.floor(Math.random() * 600) + 60
    })
  }
  
  return history
}

const TestingSuitePage: React.FC = () => {
  const [testSuites, setTestSuites] = useState(() => generateTestSuites())
  const [performanceData] = useState(() => generatePerformanceData())
  const [testHistory] = useState(() => generateTestHistory())
  const [selectedSuite, setSelectedSuite] = useState<any>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [modalVisible, setModalVisible] = useState(false)

  // 获取测试类型颜色
  const getTestTypeColor = (type: TestType): string => {
    const colors = {
      [TestType.UNIT]: '#1890ff',
      [TestType.INTEGRATION]: '#52c41a',
      [TestType.PERFORMANCE]: '#fa8c16',
      [TestType.LOAD]: '#722ed1',
      [TestType.STRESS]: '#eb2f96',
      [TestType.SECURITY]: '#ff4d4f',
      [TestType.PENETRATION]: '#f5222d',
      [TestType.E2E]: '#13c2c2'
    }
    return colors[type] || '#666'
  }

  // 获取测试类型名称
  const getTestTypeName = (type: TestType): string => {
    const names = {
      [TestType.UNIT]: '单元测试',
      [TestType.INTEGRATION]: '集成测试',
      [TestType.PERFORMANCE]: '性能测试',
      [TestType.LOAD]: '负载测试',
      [TestType.STRESS]: '压力测试',
      [TestType.SECURITY]: '安全测试',
      [TestType.PENETRATION]: '渗透测试',
      [TestType.E2E]: '端到端测试'
    }
    return names[type] || type
  }

  // 计算总体统计
  const calculateStats = () => {
    const totalTests = testSuites.reduce((sum, suite) => sum + suite.totalTests, 0)
    const passedTests = testSuites.reduce((sum, suite) => sum + suite.passedTests, 0)
    const failedTests = testSuites.reduce((sum, suite) => sum + suite.failedTests, 0)
    const avgCoverage = testSuites.reduce((sum, suite) => sum + suite.coverage, 0) / testSuites.length
    
    return {
      totalTests,
      passedTests,
      failedTests,
      passRate: (passedTests / totalTests * 100).toFixed(1),
      avgCoverage: avgCoverage.toFixed(1)
    }
  }

  const stats = calculateStats()

  // 统计卡片
  const StatsCards = () => (
    <Row gutter={16}>
      <Col span={6}>
        <Card>
          <Statistic
            title="测试总数"
            value={stats.totalTests}
            prefix={<ExperimentOutlined />}
          />
        </Card>
      </Col>
      <Col span={6}>
        <Card>
          <Statistic
            title="通过率"
            value={stats.passRate}
            suffix="%"
            prefix={<CheckCircleOutlined />}
            valueStyle={{ color: '#52c41a' }}
          />
        </Card>
      </Col>
      <Col span={6}>
        <Card>
          <Statistic
            title="平均覆盖率"
            value={stats.avgCoverage}
            suffix="%"
            prefix={<SafetyOutlined />}
            valueStyle={{ color: '#1890ff' }}
          />
        </Card>
      </Col>
      <Col span={6}>
        <Card>
          <Statistic
            title="失败测试"
            value={stats.failedTests}
            prefix={<CloseCircleOutlined />}
            valueStyle={{ color: '#ff4d4f' }}
          />
        </Card>
      </Col>
    </Row>
  )

  // 测试套件表格
  const TestSuitesTable = () => {
    const columns = [
      {
        title: '套件名称',
        dataIndex: 'name',
        key: 'name',
        render: (name: string, record: any) => (
          <Space>
            <Badge status={record.status === TestStatus.PASSED ? 'success' : 
                         record.status === TestStatus.FAILED ? 'error' : 'default'} />
            <Text strong>{name}</Text>
          </Space>
        )
      },
      {
        title: '类型',
        dataIndex: 'type',
        key: 'type',
        render: (type: TestType) => (
          <Tag color={getTestTypeColor(type)}>{getTestTypeName(type)}</Tag>
        )
      },
      {
        title: '测试结果',
        key: 'results',
        render: (record: any) => (
          <Space>
            <Text style={{ color: '#52c41a' }}>{record.passedTests} 通过</Text>
            <Text style={{ color: '#ff4d4f' }}>{record.failedTests} 失败</Text>
            <Text type="secondary">{record.skippedTests} 跳过</Text>
          </Space>
        )
      },
      {
        title: '覆盖率',
        dataIndex: 'coverage',
        key: 'coverage',
        render: (coverage: number) => (
          <Progress 
            percent={coverage} 
            size="small" 
            format={percent => `${percent?.toFixed(1)}%`}
            strokeColor={coverage > 80 ? '#52c41a' : coverage > 60 ? '#faad14' : '#ff4d4f'}
          />
        )
      },
      {
        title: '执行时间',
        dataIndex: 'duration',
        key: 'duration',
        render: (duration: number) => `${duration}s`
      },
      {
        title: '最后运行',
        dataIndex: 'lastRun',
        key: 'lastRun',
        render: (time: Date) => time.toLocaleString()
      },
      {
        title: '操作',
        key: 'actions',
        render: (record: any) => (
          <Space>
            <Button 
              icon={<PlayCircleOutlined />} 
              size="small"
              onClick={() => handleRunTest(record)}
            >
              运行
            </Button>
            <Button 
              icon={<BugOutlined />} 
              size="small"
              onClick={() => {
                setSelectedSuite(record)
                setModalVisible(true)
              }}
            >
              详情
            </Button>
          </Space>
        )
      }
    ]

    return (
      <Card title="测试套件管理" size="small">
        <Table
          columns={columns}
          dataSource={testSuites}
          rowKey="id"
          size="small"
          pagination={{ pageSize: 10 }}
        />
      </Card>
    )
  }

  // 运行测试
  const handleRunTest = async (suite: any) => {
    setIsRunning(true)
    // 模拟测试运行
    setTimeout(() => {
      setIsRunning(false)
      // 更新测试结果
      const updatedSuites = testSuites.map(s => {
        if (s.id === suite.id) {
          return {
            ...s,
            lastRun: new Date(),
            status: Math.random() > 0.2 ? TestStatus.PASSED : TestStatus.FAILED
          }
        }
        return s
      })
      setTestSuites(updatedSuites)
    }, 3000)
  }

  // 性能监控图表
  const PerformanceCharts = () => (
    <Row gutter={16}>
      <Col span={12}>
        <Card title="响应时间趋势" size="small">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <RechartsTooltip />
              <Legend />
              <Line type="monotone" dataKey="responseTime" stroke="#1890ff" name="响应时间(ms)" />
              <Line type="monotone" dataKey="errorRate" stroke="#ff4d4f" name="错误率(%)" />
            </LineChart>
          </ResponsiveContainer>
        </Card>
      </Col>
      <Col span={12}>
        <Card title="系统资源使用" size="small">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <RechartsTooltip />
              <Legend />
              <Line type="monotone" dataKey="cpuUsage" stroke="#52c41a" name="CPU使用率(%)" />
              <Line type="monotone" dataKey="memoryUsage" stroke="#fa8c16" name="内存使用率(%)" />
            </LineChart>
          </ResponsiveContainer>
        </Card>
      </Col>
    </Row>
  )

  // 测试历史趋势
  const TestHistoryChart = () => (
    <Card title="测试历史趋势" size="small">
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={testHistory}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="timestamp" 
            tickFormatter={(time) => new Date(time).toLocaleDateString()}
          />
          <YAxis yAxisId="left" />
          <YAxis yAxisId="right" orientation="right" />
          <RechartsTooltip 
            labelFormatter={(time) => new Date(time).toLocaleString()}
          />
          <Legend />
          <Line 
            yAxisId="left"
            type="monotone" 
            dataKey="passRate" 
            stroke="#52c41a" 
            name="通过率(%)"
          />
          <Line 
            yAxisId="left"
            type="monotone" 
            dataKey="coverage" 
            stroke="#1890ff" 
            name="覆盖率(%)"
          />
          <Line 
            yAxisId="right"
            type="monotone" 
            dataKey="totalTests" 
            stroke="#722ed1" 
            name="测试数量"
          />
        </LineChart>
      </ResponsiveContainer>
    </Card>
  )

  // 测试类型分布
  const TestTypeDistribution = () => {
    const typeData = Object.values(TestType).map(type => ({
      name: getTestTypeName(type),
      value: testSuites.filter(s => s.type === type).length,
      color: getTestTypeColor(type)
    }))

    return (
      <Card title="测试类型分布" size="small">
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={typeData}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              outerRadius={80}
              fill="#8884d8"
              dataKey="value"
            >
              {typeData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <RechartsTooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>
    )
  }

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <ExperimentOutlined /> 测试套件管理中心
      </Title>
      <Paragraph type="secondary">
        全面的测试管理平台，支持单元测试、集成测试、性能测试、安全测试等多种测试类型
      </Paragraph>
      
      <Divider />

      <Tabs defaultActiveKey="1">
        <TabPane tab="测试总览" key="1">
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            <StatsCards />
            
            {isRunning && (
              <Alert
                message="测试运行中"
                description="正在执行测试套件，请稍候..."
                variant="default"
                showIcon
                icon={<SyncOutlined spin />}
              />
            )}

            <TestSuitesTable />
          </Space>
        </TabPane>

        <TabPane tab="性能监控" key="2">
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            <PerformanceCharts />
            
            <Card title="负载测试配置" size="small">
              <Form layout="inline">
                <Form.Item label="并发用户数">
                  <Input placeholder="100" style={{ width: 100 }} />
                </Form.Item>
                <Form.Item label="持续时间">
                  <Input placeholder="60" suffix="秒" style={{ width: 100 }} />
                </Form.Item>
                <Form.Item label="请求速率">
                  <Input placeholder="10" suffix="req/s" style={{ width: 100 }} />
                </Form.Item>
                <Form.Item>
                  <Button type="primary" icon={<ThunderboltOutlined />}>
                    开始负载测试
                  </Button>
                </Form.Item>
              </Form>
            </Card>
          </Space>
        </TabPane>

        <TabPane tab="测试历史" key="3">
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            <TestHistoryChart />
            
            <Card title="测试执行记录" size="small">
              <Timeline>
                {testHistory.slice(0, 5).map((history, index) => (
                  <Timeline.Item
                    key={history.id}
                    color={history.passRate > 90 ? 'green' : history.passRate > 70 ? 'orange' : 'red'}
                  >
                    <Space direction="vertical">
                      <Text strong>{history.timestamp.toLocaleString()}</Text>
                      <Space>
                        <Text>测试数: {history.totalTests}</Text>
                        <Text>通过率: {history.passRate.toFixed(1)}%</Text>
                        <Text>覆盖率: {history.coverage.toFixed(1)}%</Text>
                        <Text>耗时: {history.duration}s</Text>
                      </Space>
                    </Space>
                  </Timeline.Item>
                ))}
              </Timeline>
            </Card>
          </Space>
        </TabPane>

        <TabPane tab="测试分析" key="4">
          <Row gutter={16}>
            <Col span={12}>
              <TestTypeDistribution />
            </Col>
            <Col span={12}>
              <Card title="测试质量指标" size="small">
                <List>
                  <List.Item>
                    <List.Item.Meta
                      title="代码覆盖率"
                      description={
                        <Progress 
                          percent={parseFloat(stats.avgCoverage)} 
                          strokeColor="#52c41a"
                        />
                      }
                    />
                  </List.Item>
                  <List.Item>
                    <List.Item.Meta
                      title="测试通过率"
                      description={
                        <Progress 
                          percent={parseFloat(stats.passRate)} 
                          strokeColor="#1890ff"
                        />
                      }
                    />
                  </List.Item>
                  <List.Item>
                    <List.Item.Meta
                      title="缺陷密度"
                      description={
                        <Space>
                          <Text>{(stats.failedTests / stats.totalTests * 100).toFixed(2)}%</Text>
                          <Tag color="red">{stats.failedTests} 个缺陷</Tag>
                        </Space>
                      }
                    />
                  </List.Item>
                </List>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="安全测试" key="5">
          <Card title="安全扫描结果" size="small">
            <Alert
              message="安全扫描完成"
              description="发现 3 个高危漏洞，5 个中危漏洞，12 个低危漏洞"
              variant="warning"
              showIcon
              style={{ marginBottom: 16 }}
            />
            
            <List>
              <List.Item>
                <List.Item.Meta
                  avatar={<Badge status="error" />}
                  title={<Text strong>SQL注入漏洞</Text>}
                  description="在用户输入参数中发现潜在的SQL注入风险"
                />
                <Tag color="red">高危</Tag>
              </List.Item>
              <List.Item>
                <List.Item.Meta
                  avatar={<Badge status="warning" />}
                  title={<Text strong>XSS跨站脚本</Text>}
                  description="输出内容未进行适当的编码处理"
                />
                <Tag color="orange">中危</Tag>
              </List.Item>
              <List.Item>
                <List.Item.Meta
                  avatar={<Badge status="default" />}
                  title={<Text strong>敏感信息泄露</Text>}
                  description="错误信息中包含系统内部信息"
                />
                <Tag color="yellow">低危</Tag>
              </List.Item>
            </List>
          </Card>
        </TabPane>
      </Tabs>

      {/* 测试详情模态框 */}
      <Modal
        title="测试套件详情"
        visible={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={null}
        width={800}
      >
        {selectedSuite && (
          <div>
            <Descriptions bordered>
              <Descriptions.Item label="套件名称">{selectedSuite.name}</Descriptions.Item>
              <Descriptions.Item label="测试类型">
                <Tag color={getTestTypeColor(selectedSuite.type)}>
                  {getTestTypeName(selectedSuite.type)}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="最后运行">
                {selectedSuite.lastRun.toLocaleString()}
              </Descriptions.Item>
              <Descriptions.Item label="测试总数">{selectedSuite.totalTests}</Descriptions.Item>
              <Descriptions.Item label="通过数">{selectedSuite.passedTests}</Descriptions.Item>
              <Descriptions.Item label="失败数">{selectedSuite.failedTests}</Descriptions.Item>
              <Descriptions.Item label="覆盖率">
                <Progress percent={selectedSuite.coverage} />
              </Descriptions.Item>
              <Descriptions.Item label="执行时间">{selectedSuite.duration}s</Descriptions.Item>
            </Descriptions>
          </div>
        )}
      </Modal>
    </div>
  )
}

export default TestingSuitePage