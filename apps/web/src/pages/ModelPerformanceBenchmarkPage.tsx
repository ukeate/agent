import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Button,
  Table,
  Space,
  Typography,
  Tabs,
  Progress,
  Tag,
  Statistic,
  Alert,
  Select,
  Input,
  Form,
  Switch,
  Divider,
  List,
  Avatar,
  Badge,
  message,
  Modal,
  Tooltip,
  Radio,
  InputNumber,
  Checkbox,
  Tree,
  Empty
} from 'antd'
import {
  ThunderboltOutlined,
  BarChartOutlined,
  LineChartOutlined,
  TrophyOutlined,
  ClockCircleOutlined,
  DatabaseOutlined,
  PlayCircleOutlined,
  StopOutlined,
  ReloadOutlined,
  SettingOutlined,
  EyeOutlined,
  DownloadOutlined,
  SwapOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  LoadingOutlined,
  RocketOutlined,
  BugOutlined,
  FileTextOutlined,
  HddOutlined
} from '@ant-design/icons'
import { Line, Column, Radar } from '@ant-design/plots'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs
const { Option } = Select
const { TextArea } = Input

interface BenchmarkSuite {
  id: string
  name: string
  description: string
  category: 'latency' | 'throughput' | 'memory' | 'accuracy' | 'energy'
  tests: BenchmarkTest[]
  enabled: boolean
}

interface BenchmarkTest {
  id: string
  name: string
  description: string
  inputSize: string
  batchSize: number
  iterations: number
  warmupIterations: number
  enabled: boolean
}

interface BenchmarkResult {
  id: string
  name: string
  modelName: string
  suiteId: string
  testId: string
  status: 'running' | 'completed' | 'failed' | 'pending'
  progress: number
  startTime: string
  endTime?: string
  metrics: {
    avgLatency: number
    p50Latency: number
    p95Latency: number
    p99Latency: number
    throughput: number
    memoryUsage: number
    cpuUsage: number
    gpuUsage: number
    energyConsumption: number
    accuracy?: number
  }
  logs: string[]
}

interface SystemProfile {
  cpu: string
  memory: string
  gpu: string
  framework: string
  version: string
  os: string
}

const ModelPerformanceBenchmarkPage: React.FC = () => {
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('overview')
  const [benchmarkSuites, setBenchmarkSuites] = useState<BenchmarkSuite[]>([])
  const [benchmarkResults, setBenchmarkResults] = useState<BenchmarkResult[]>([])
  const [selectedSuites, setSelectedSuites] = useState<string[]>([])
  const [runningTests, setRunningTests] = useState<string[]>([])
  const [systemProfile, setSystemProfile] = useState<SystemProfile | null>(null)
  const [resultModalVisible, setResultModalVisible] = useState(false)
  const [selectedResult, setSelectedResult] = useState<BenchmarkResult | null>(null)
  const [configModalVisible, setConfigModalVisible] = useState(false)

  useEffect(() => {
    // 模拟基准测试套件
    setBenchmarkSuites([
      {
        id: 'latency-suite',
        name: '延迟性能测试',
        description: '测试模型推理延迟和响应时间',
        category: 'latency',
        enabled: true,
        tests: [
          {
            id: 'single-inference',
            name: '单次推理延迟',
            description: '测试单个样本的推理时间',
            inputSize: '224x224x3',
            batchSize: 1,
            iterations: 1000,
            warmupIterations: 100,
            enabled: true
          },
          {
            id: 'batch-inference',
            name: '批量推理延迟',
            description: '测试批量样本的推理时间',
            inputSize: '224x224x3',
            batchSize: 32,
            iterations: 100,
            warmupIterations: 10,
            enabled: true
          }
        ]
      },
      {
        id: 'throughput-suite',
        name: '吞吐量测试',
        description: '测试模型的处理能力和吞吐量',
        category: 'throughput',
        enabled: true,
        tests: [
          {
            id: 'max-throughput',
            name: '最大吞吐量',
            description: '测试模型的最大处理能力',
            inputSize: '224x224x3',
            batchSize: 64,
            iterations: 50,
            warmupIterations: 5,
            enabled: true
          },
          {
            id: 'sustained-throughput',
            name: '持续吞吐量',
            description: '测试长时间运行的吞吐量',
            inputSize: '224x224x3',
            batchSize: 32,
            iterations: 500,
            warmupIterations: 50,
            enabled: true
          }
        ]
      },
      {
        id: 'memory-suite',
        name: '内存使用测试',
        description: '测试模型的内存占用和效率',
        category: 'memory',
        enabled: true,
        tests: [
          {
            id: 'memory-usage',
            name: '内存占用测试',
            description: '测试模型加载和推理时的内存使用',
            inputSize: '224x224x3',
            batchSize: 1,
            iterations: 100,
            warmupIterations: 10,
            enabled: true
          },
          {
            id: 'memory-leak',
            name: '内存泄漏测试',
            description: '测试长时间运行是否存在内存泄漏',
            inputSize: '224x224x3',
            batchSize: 1,
            iterations: 1000,
            warmupIterations: 100,
            enabled: true
          }
        ]
      },
      {
        id: 'accuracy-suite',
        name: '精度验证测试',
        description: '验证压缩后模型的精度保持',
        category: 'accuracy',
        enabled: true,
        tests: [
          {
            id: 'validation-accuracy',
            name: '验证集精度',
            description: '在验证集上测试模型精度',
            inputSize: '224x224x3',
            batchSize: 32,
            iterations: 100,
            warmupIterations: 0,
            enabled: true
          },
          {
            id: 'robustness-test',
            name: '鲁棒性测试',
            description: '测试模型对噪声的鲁棒性',
            inputSize: '224x224x3',
            batchSize: 16,
            iterations: 200,
            warmupIterations: 0,
            enabled: false
          }
        ]
      }
    ])

    // 模拟基准测试结果
    setBenchmarkResults([
      {
        id: 'result-001',
        name: 'ResNet50量化模型基准测试',
        modelName: 'ResNet50-INT8',
        suiteId: 'latency-suite',
        testId: 'single-inference',
        status: 'completed',
        progress: 100,
        startTime: '2024-08-24 10:30:00',
        endTime: '2024-08-24 10:45:00',
        metrics: {
          avgLatency: 12.8,
          p50Latency: 12.1,
          p95Latency: 15.2,
          p99Latency: 18.9,
          throughput: 78.1,
          memoryUsage: 64.2,
          cpuUsage: 45.6,
          gpuUsage: 72.3,
          energyConsumption: 2.1,
          accuracy: 75.8
        },
        logs: [
          '开始基准测试...',
          '加载ResNet50-INT8模型...',
          '执行warmup iterations (100)...',
          '开始性能测试...',
          'Iteration 500/1000: 平均延迟 12.5ms',
          'Iteration 1000/1000: 平均延迟 12.8ms',
          '测试完成！'
        ]
      },
      {
        id: 'result-002',
        name: 'MobileNet剪枝模型吞吐量测试',
        modelName: 'MobileNet-V2-Pruned',
        suiteId: 'throughput-suite',
        testId: 'max-throughput',
        status: 'running',
        progress: 65,
        startTime: '2024-08-24 11:00:00',
        metrics: {
          avgLatency: 18.9,
          p50Latency: 18.2,
          p95Latency: 22.1,
          p99Latency: 26.4,
          throughput: 52.9,
          memoryUsage: 42.1,
          cpuUsage: 38.2,
          gpuUsage: 55.7,
          energyConsumption: 1.8
        },
        logs: [
          '开始吞吐量测试...',
          '加载MobileNet-V2-Pruned模型...',
          '执行warmup iterations (5)...',
          '开始吞吐量测试...',
          'Iteration 30/50: 当前吞吐量 54.2 samples/s',
          '测试进行中...'
        ]
      }
    ])

    // 模拟系统配置
    setSystemProfile({
      cpu: 'Intel Core i9-12900K',
      memory: '32GB DDR4-3200',
      gpu: 'NVIDIA RTX 4090 24GB',
      framework: 'PyTorch 2.0.1',
      version: 'CUDA 11.8',
      os: 'Ubuntu 22.04 LTS'
    })
  }, [])

  const getCategoryColor = (category: BenchmarkSuite['category']) => {
    switch (category) {
      case 'latency': return '#1890ff'
      case 'throughput': return '#52c41a'
      case 'memory': return '#722ed1'
      case 'accuracy': return '#fa8c16'
      case 'energy': return '#13c2c2'
      default: return '#d9d9d9'
    }
  }

  const getCategoryIcon = (category: BenchmarkSuite['category']) => {
    switch (category) {
      case 'latency': return <ClockCircleOutlined />
      case 'throughput': return <ThunderboltOutlined />
      case 'memory': return <DatabaseOutlined />
      case 'accuracy': return <TrophyOutlined />
      case 'energy': return <BugOutlined />
      default: return <SettingOutlined />
    }
  }

  const getStatusColor = (status: BenchmarkResult['status']) => {
    switch (status) {
      case 'pending': return '#faad14'
      case 'running': return '#1890ff'
      case 'completed': return '#52c41a'
      case 'failed': return '#ff4d4f'
      default: return '#d9d9d9'
    }
  }

  const getStatusText = (status: BenchmarkResult['status']) => {
    switch (status) {
      case 'pending': return '等待中'
      case 'running': return '运行中'
      case 'completed': return '已完成'
      case 'failed': return '失败'
      default: return '未知'
    }
  }

  const handleRunBenchmark = async () => {
    if (selectedSuites.length === 0) {
      message.warning('请选择至少一个测试套件')
      return
    }

    setLoading(true)
    try {
      // 模拟启动基准测试
      const newResults: BenchmarkResult[] = []
      
      selectedSuites.forEach(suiteId => {
        const suite = benchmarkSuites.find(s => s.id === suiteId)
        if (suite) {
          suite.tests.filter(t => t.enabled).forEach(test => {
            const resultId = `result-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
            newResults.push({
              id: resultId,
              name: `${suite.name} - ${test.name}`,
              modelName: '待测试模型',
              suiteId: suite.id,
              testId: test.id,
              status: 'pending',
              progress: 0,
              startTime: new Date().toLocaleString(),
              metrics: {
                avgLatency: 0,
                p50Latency: 0,
                p95Latency: 0,
                p99Latency: 0,
                throughput: 0,
                memoryUsage: 0,
                cpuUsage: 0,
                gpuUsage: 0,
                energyConsumption: 0
              },
              logs: ['测试已创建，等待启动...']
            })
          })
        }
      })

      setBenchmarkResults(prev => [...newResults, ...prev])
      setRunningTests(newResults.map(r => r.id))
      message.success(`已创建 ${newResults.length} 个基准测试任务`)

      // 模拟测试执行
      newResults.forEach((result, index) => {
        setTimeout(() => {
          setBenchmarkResults(prev => prev.map(r => 
            r.id === result.id 
              ? { ...r, status: 'running' as const, logs: [...r.logs, '开始执行测试...'] }
              : r
          ))
        }, (index + 1) * 1000)
      })

      setActiveTab('results')
    } catch (error) {
      console.error('启动基准测试失败:', error)
      message.error('启动基准测试失败')
    } finally {
      setLoading(false)
    }
  }

  const suiteColumns = [
    {
      title: '测试套件',
      key: 'suite',
      render: (record: BenchmarkSuite) => (
        <div>
          <Space>
            {getCategoryIcon(record.category)}
            <Text strong>{record.name}</Text>
          </Space>
          <div style={{ fontSize: '12px', color: '#666', marginTop: '4px' }}>
            {record.description}
          </div>
        </div>
      )
    },
    {
      title: '类别',
      key: 'category',
      render: (record: BenchmarkSuite) => (
        <Tag color={getCategoryColor(record.category)}>
          {record.category.toUpperCase()}
        </Tag>
      )
    },
    {
      title: '测试数量',
      key: 'testCount',
      render: (record: BenchmarkSuite) => (
        <Badge
          count={record.tests.filter(t => t.enabled).length}
          style={{ backgroundColor: '#52c41a' }}
        />
      )
    },
    {
      title: '状态',
      key: 'enabled',
      render: (record: BenchmarkSuite) => (
        <Switch
          checked={record.enabled}
          onChange={(checked) => {
            setBenchmarkSuites(prev => prev.map(suite =>
              suite.id === record.id ? { ...suite, enabled: checked } : suite
            ))
          }}
        />
      )
    }
  ]

  const resultColumns = [
    {
      title: '测试名称',
      key: 'name',
      render: (record: BenchmarkResult) => (
        <div>
          <Text strong>{record.name}</Text>
          <div style={{ fontSize: '12px', color: '#666' }}>
            模型: {record.modelName}
          </div>
        </div>
      )
    },
    {
      title: '状态',
      key: 'status',
      render: (record: BenchmarkResult) => (
        <div>
          <Tag color={getStatusColor(record.status)}>
            {getStatusText(record.status)}
          </Tag>
          {record.status === 'running' && (
            <div style={{ marginTop: '4px' }}>
              <Progress percent={record.progress} size="small" />
            </div>
          )}
        </div>
      )
    },
    {
      title: '平均延迟',
      key: 'avgLatency',
      render: (record: BenchmarkResult) => (
        record.metrics.avgLatency > 0 ? (
          <Text strong style={{ color: '#1890ff' }}>
            {record.metrics.avgLatency.toFixed(1)}ms
          </Text>
        ) : (
          <Text type="secondary">-</Text>
        )
      )
    },
    {
      title: '吞吐量',
      key: 'throughput',
      render: (record: BenchmarkResult) => (
        record.metrics.throughput > 0 ? (
          <Text strong style={{ color: '#52c41a' }}>
            {record.metrics.throughput.toFixed(1)} samples/s
          </Text>
        ) : (
          <Text type="secondary">-</Text>
        )
      )
    },
    {
      title: '内存使用',
      key: 'memoryUsage',
      render: (record: BenchmarkResult) => (
        record.metrics.memoryUsage > 0 ? (
          <Text strong style={{ color: '#722ed1' }}>
            {record.metrics.memoryUsage.toFixed(1)}MB
          </Text>
        ) : (
          <Text type="secondary">-</Text>
        )
      )
    },
    {
      title: '开始时间',
      key: 'startTime',
      render: (record: BenchmarkResult) => (
        <Text type="secondary">{record.startTime}</Text>
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: BenchmarkResult) => (
        <Space>
          <Button
            type="link"
            size="small"
            icon={<EyeOutlined />}
            onClick={() => {
              setSelectedResult(record)
              setResultModalVisible(true)
            }}
          >
            详情
          </Button>
          {record.status === 'running' && (
            <Button
              type="link"
              size="small"
              danger
              icon={<StopOutlined />}
            >
              停止
            </Button>
          )}
          {record.status === 'completed' && (
            <Button
              type="link"
              size="small"
              icon={<DownloadOutlined />}
            >
              导出
            </Button>
          )}
        </Space>
      )
    }
  ]

  const renderOverview = () => {
    const completedResults = benchmarkResults.filter(r => r.status === 'completed')
    const avgLatency = completedResults.length > 0 
      ? completedResults.reduce((sum, r) => sum + r.metrics.avgLatency, 0) / completedResults.length 
      : 0
    const avgThroughput = completedResults.length > 0 
      ? completedResults.reduce((sum, r) => sum + r.metrics.throughput, 0) / completedResults.length 
      : 0

    // 准备雷达图数据
    const radarData = completedResults.length > 0 ? [
      { item: '延迟性能', score: Math.max(0, 100 - (avgLatency / 50) * 100) },
      { item: '吞吐量', score: Math.min(100, (avgThroughput / 100) * 100) },
      { item: '内存效率', score: 85 },
      { item: 'CPU使用', score: 75 },
      { item: 'GPU使用', score: 80 },
      { item: '能耗效率', score: 90 }
    ] : []

    return (
      <div>
        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          <Col span={6}>
            <Card>
              <Statistic
                title="总测试数"
                value={benchmarkResults.length}
                prefix={<ThunderboltOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="已完成"
                value={benchmarkResults.filter(r => r.status === 'completed').length}
                prefix={<CheckCircleOutlined />}
                valueStyle={{ color: '#52c41a' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="运行中"
                value={benchmarkResults.filter(r => r.status === 'running').length}
                prefix={<LoadingOutlined />}
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="平均延迟"
                value={avgLatency}
                precision={1}
                suffix="ms"
                prefix={<ClockCircleOutlined />}
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
        </Row>

        <Row gutter={[16, 16]}>
          <Col span={12}>
            <Card title="系统配置" extra={<ThunderboltOutlined />}>
              {systemProfile && (
                <List size="small">
                  <List.Item>
                    <Text type="secondary">CPU:</Text>
                    <Text strong style={{ marginLeft: '8px' }}>
                      {systemProfile.cpu}
                    </Text>
                  </List.Item>
                  <List.Item>
                    <Text type="secondary">内存:</Text>
                    <Text strong style={{ marginLeft: '8px' }}>
                      {systemProfile.memory}
                    </Text>
                  </List.Item>
                  <List.Item>
                    <Text type="secondary">GPU:</Text>
                    <Text strong style={{ marginLeft: '8px' }}>
                      {systemProfile.gpu}
                    </Text>
                  </List.Item>
                  <List.Item>
                    <Text type="secondary">框架:</Text>
                    <Text strong style={{ marginLeft: '8px' }}>
                      {systemProfile.framework}
                    </Text>
                  </List.Item>
                  <List.Item>
                    <Text type="secondary">版本:</Text>
                    <Text strong style={{ marginLeft: '8px' }}>
                      {systemProfile.version}
                    </Text>
                  </List.Item>
                  <List.Item>
                    <Text type="secondary">操作系统:</Text>
                    <Text strong style={{ marginLeft: '8px' }}>
                      {systemProfile.os}
                    </Text>
                  </List.Item>
                </List>
              )}
            </Card>
          </Col>
          <Col span={12}>
            <Card title="性能概览" extra={<BarChartOutlined />}>
              {radarData.length > 0 ? (
                <Radar
                  data={radarData}
                  xField="item"
                  yField="score"
                  height={280}
                  area={{}}
                  point={{
                    size: 2,
                  }}
                />
              ) : (
                <Empty description="暂无完成的测试结果" />
              )}
            </Card>
          </Col>
        </Row>
      </div>
    )
  }

  const renderConfiguration = () => (
    <Row gutter={[16, 16]}>
      <Col span={14}>
        <Card 
          title="基准测试套件" 
          extra={
            <Space>
              <Text type="secondary">
                已选择: {selectedSuites.length} 个套件
              </Text>
              <Button
                type="primary"
                loading={loading}
                disabled={selectedSuites.length === 0}
                onClick={handleRunBenchmark}
                icon={<PlayCircleOutlined />}
              >
                开始测试
              </Button>
            </Space>
          }
        >
          <Table
            columns={suiteColumns}
            dataSource={benchmarkSuites}
            rowKey="id"
            pagination={false}
            rowSelection={{
              type: 'checkbox',
              selectedRowKeys: selectedSuites,
              onChange: setSelectedSuites
            }}
          />
        </Card>
      </Col>
      
      <Col span={10}>
        <Card title="测试配置" extra={<SettingOutlined />}>
          <Form form={form} layout="vertical">
            <Form.Item label="模型路径">
              <Input placeholder="输入模型文件路径" />
            </Form.Item>
            
            <Form.Item label="测试模式">
              <Radio.Group defaultValue="standard">
                <Radio value="quick">快速测试</Radio>
                <Radio value="standard">标准测试</Radio>
                <Radio value="comprehensive">全面测试</Radio>
              </Radio.Group>
            </Form.Item>

            <Form.Item label="并发线程数">
              <InputNumber min={1} max={16} defaultValue={1} style={{ width: '100%' }} />
            </Form.Item>

            <Form.Item>
              <Checkbox defaultChecked>启用详细日志</Checkbox>
            </Form.Item>

            <Form.Item>
              <Checkbox>测试完成后自动导出报告</Checkbox>
            </Form.Item>
          </Form>

          <Divider />
          
          <Alert
            message="测试提示"
            description="基准测试将消耗系统资源，请确保系统处于空闲状态以获得准确结果。"
            type="info"
            showIcon
          />
        </Card>
      </Col>
    </Row>
  )

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2} style={{ margin: 0, color: '#1a1a1a' }}>
          <ThunderboltOutlined style={{ marginRight: '12px', color: '#1890ff' }} />
          性能基准测试
        </Title>
        <Paragraph style={{ marginTop: '8px', color: '#666', fontSize: '16px' }}>
          全面测试和评估模型性能，包括延迟、吞吐量、内存使用等关键指标
        </Paragraph>
      </div>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="概览仪表板" key="overview">
          {renderOverview()}
        </TabPane>

        <TabPane tab="测试配置" key="configuration">
          {renderConfiguration()}
        </TabPane>

        <TabPane tab="测试结果" key="results">
          <Card
            title="基准测试结果"
            extra={
              <Space>
                <Button
                  icon={<ReloadOutlined />}
                  onClick={() => message.info('刷新测试结果')}
                >
                  刷新
                </Button>
                <Button
                  type="primary"
                  onClick={() => setActiveTab('configuration')}
                  icon={<PlayCircleOutlined />}
                >
                  新建测试
                </Button>
              </Space>
            }
          >
            <Table
              columns={resultColumns}
              dataSource={benchmarkResults}
              rowKey="id"
              pagination={{ pageSize: 10 }}
            />
          </Card>
        </TabPane>
      </Tabs>

      {/* 结果详情模态框 */}
      <Modal
        title="基准测试详情"
        open={resultModalVisible}
        onCancel={() => setResultModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setResultModalVisible(false)}>
            关闭
          </Button>,
          <Button key="export" type="primary" icon={<DownloadOutlined />}>
            导出报告
          </Button>
        ]}
        width={1000}
      >
        {selectedResult && (
          <div>
            <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
              <Col span={6}>
                <Statistic
                  title="平均延迟"
                  value={selectedResult.metrics.avgLatency}
                  precision={1}
                  suffix="ms"
                  valueStyle={{ color: '#1890ff' }}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="P95延迟"
                  value={selectedResult.metrics.p95Latency}
                  precision={1}
                  suffix="ms"
                  valueStyle={{ color: '#722ed1' }}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="吞吐量"
                  value={selectedResult.metrics.throughput}
                  precision={1}
                  suffix="samples/s"
                  valueStyle={{ color: '#52c41a' }}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="内存使用"
                  value={selectedResult.metrics.memoryUsage}
                  precision={1}
                  suffix="MB"
                  valueStyle={{ color: '#fa8c16' }}
                />
              </Col>
            </Row>

            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Card title="延迟分布" size="small">
                  <List size="small">
                    <List.Item>
                      <Text type="secondary">平均延迟:</Text>
                      <Text strong style={{ marginLeft: '8px' }}>
                        {selectedResult.metrics.avgLatency.toFixed(2)}ms
                      </Text>
                    </List.Item>
                    <List.Item>
                      <Text type="secondary">P50延迟:</Text>
                      <Text strong style={{ marginLeft: '8px' }}>
                        {selectedResult.metrics.p50Latency.toFixed(2)}ms
                      </Text>
                    </List.Item>
                    <List.Item>
                      <Text type="secondary">P95延迟:</Text>
                      <Text strong style={{ marginLeft: '8px' }}>
                        {selectedResult.metrics.p95Latency.toFixed(2)}ms
                      </Text>
                    </List.Item>
                    <List.Item>
                      <Text type="secondary">P99延迟:</Text>
                      <Text strong style={{ marginLeft: '8px' }}>
                        {selectedResult.metrics.p99Latency.toFixed(2)}ms
                      </Text>
                    </List.Item>
                  </List>
                </Card>
              </Col>

              <Col span={12}>
                <Card title="资源使用" size="small">
                  <List size="small">
                    <List.Item>
                      <Text type="secondary">CPU使用率:</Text>
                      <Text strong style={{ marginLeft: '8px' }}>
                        {selectedResult.metrics.cpuUsage.toFixed(1)}%
                      </Text>
                    </List.Item>
                    <List.Item>
                      <Text type="secondary">GPU使用率:</Text>
                      <Text strong style={{ marginLeft: '8px' }}>
                        {selectedResult.metrics.gpuUsage.toFixed(1)}%
                      </Text>
                    </List.Item>
                    <List.Item>
                      <Text type="secondary">内存使用:</Text>
                      <Text strong style={{ marginLeft: '8px' }}>
                        {selectedResult.metrics.memoryUsage.toFixed(1)}MB
                      </Text>
                    </List.Item>
                    <List.Item>
                      <Text type="secondary">能耗:</Text>
                      <Text strong style={{ marginLeft: '8px' }}>
                        {selectedResult.metrics.energyConsumption.toFixed(1)}J
                      </Text>
                    </List.Item>
                  </List>
                </Card>
              </Col>
            </Row>

            <Divider />
            <Title level={4}>执行日志</Title>
            <div style={{
              background: '#f5f5f5',
              padding: '12px',
              borderRadius: '4px',
              maxHeight: '200px',
              overflowY: 'auto',
              fontFamily: 'monospace'
            }}>
              {selectedResult.logs.map((log, index) => (
                <div key={index} style={{ margin: '4px 0', fontSize: '12px' }}>
                  <Text type="secondary">{new Date().toLocaleTimeString()}</Text>
                  <span style={{ margin: '0 8px' }}>|</span>
                  <Text>{log}</Text>
                </div>
              ))}
            </div>
          </div>
        )}
      </Modal>
    </div>
  )
}

export default ModelPerformanceBenchmarkPage