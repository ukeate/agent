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
  DatePicker,
  Radio,
  Tooltip,
  Modal,
  Form,
  Switch,
  Divider,
  List,
  Avatar,
  Badge,
  message,
  Empty
} from 'antd'
import {
  BarChartOutlined,
  LineChartOutlined,
  PieChartOutlined,
  SwapOutlined,
  TrophyOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  EyeOutlined,
  DownloadOutlined,
  ShareAltOutlined,
  FilterOutlined,
  ReloadOutlined,
  StarOutlined,
  BulbOutlined
} from '@ant-design/icons'
import { Line, Column, Pie } from '@ant-design/plots'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs
const { Option } = Select
const { RangePicker } = DatePicker

interface CompressionResult {
  id: string
  name: string
  modelName: string
  compressionType: 'quantization' | 'pruning' | 'distillation' | 'mixed'
  baselineAccuracy: number
  compressedAccuracy: number
  accuracyLoss: number
  originalSize: number
  compressedSize: number
  compressionRatio: number
  originalInference: number
  compressedInference: number
  speedup: number
  energyReduction: number
  memoryReduction: number
  score: number
  rank: number
  createTime: string
  tags: string[]
}

interface BenchmarkMetric {
  name: string
  unit: string
  baseline: number
  current: number
  improvement: number
  target: number
  status: 'excellent' | 'good' | 'fair' | 'poor'
}

const ModelCompressionEvaluationPage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('overview')
  const [selectedModels, setSelectedModels] = useState<string[]>([])
  const [compressionResults, setCompressionResults] = useState<CompressionResult[]>([])
  const [compareModalVisible, setCompareModalVisible] = useState(false)
  const [detailModalVisible, setDetailModalVisible] = useState(false)
  const [selectedResult, setSelectedResult] = useState<CompressionResult | null>(null)
  const [filterType, setFilterType] = useState('all')
  const [sortBy, setSortBy] = useState('score')
  const [benchmarkMetrics, setBenchmarkMetrics] = useState<BenchmarkMetric[]>([])

  useEffect(() => {
    // 模拟压缩结果数据
    setCompressionResults([
      {
        id: 'comp-001',
        name: 'ResNet50量化INT8',
        modelName: 'ResNet50',
        compressionType: 'quantization',
        baselineAccuracy: 76.2,
        compressedAccuracy: 75.8,
        accuracyLoss: 0.4,
        originalSize: 102.5,
        compressedSize: 25.6,
        compressionRatio: 4.0,
        originalInference: 45.2,
        compressedInference: 12.8,
        speedup: 3.5,
        energyReduction: 65,
        memoryReduction: 75,
        score: 95.2,
        rank: 1,
        createTime: '2024-08-24 10:30:00',
        tags: ['生产就绪', '高性能', 'INT8']
      },
      {
        id: 'comp-002',
        name: 'MobileNet结构化剪枝',
        modelName: 'MobileNet-V2',
        compressionType: 'pruning',
        baselineAccuracy: 72.1,
        compressedAccuracy: 70.3,
        accuracyLoss: 1.8,
        originalSize: 14.2,
        compressedSize: 8.5,
        compressionRatio: 1.7,
        originalInference: 28.5,
        compressedInference: 18.9,
        speedup: 1.5,
        energyReduction: 40,
        memoryReduction: 40,
        score: 82.4,
        rank: 2,
        createTime: '2024-08-24 11:15:00',
        tags: ['移动端', '结构化', '实时推理']
      },
      {
        id: 'comp-003',
        name: 'BERT知识蒸馏',
        modelName: 'BERT-Base',
        compressionType: 'distillation',
        baselineAccuracy: 84.5,
        compressedAccuracy: 82.1,
        accuracyLoss: 2.4,
        originalSize: 438.2,
        compressedSize: 109.6,
        compressionRatio: 4.0,
        originalInference: 125.6,
        compressedInference: 31.4,
        speedup: 4.0,
        energyReduction: 70,
        memoryReduction: 75,
        score: 88.7,
        rank: 1,
        createTime: '2024-08-24 09:45:00',
        tags: ['NLP', '知识蒸馏', '学生模型']
      },
      {
        id: 'comp-004',
        name: '混合压缩方案',
        modelName: 'EfficientNet-B0',
        compressionType: 'mixed',
        baselineAccuracy: 77.3,
        compressedAccuracy: 75.9,
        accuracyLoss: 1.4,
        originalSize: 20.3,
        compressedSize: 5.1,
        compressionRatio: 4.0,
        originalInference: 35.8,
        compressedInference: 9.2,
        speedup: 3.9,
        energyReduction: 72,
        memoryReduction: 75,
        score: 91.5,
        rank: 1,
        createTime: '2024-08-24 14:20:00',
        tags: ['混合方案', '最优压缩', '平衡性能']
      }
    ])

    // 模拟基准测试指标
    setBenchmarkMetrics([
      {
        name: '模型精度',
        unit: '%',
        baseline: 76.2,
        current: 75.8,
        improvement: -0.4,
        target: 75.0,
        status: 'excellent'
      },
      {
        name: '推理延迟',
        unit: 'ms',
        baseline: 45.2,
        current: 12.8,
        improvement: -71.7,
        target: 15.0,
        status: 'excellent'
      },
      {
        name: '模型大小',
        unit: 'MB',
        baseline: 102.5,
        current: 25.6,
        improvement: -75.0,
        target: 30.0,
        status: 'excellent'
      },
      {
        name: '内存占用',
        unit: 'MB',
        baseline: 256.0,
        current: 64.0,
        improvement: -75.0,
        target: 80.0,
        status: 'excellent'
      },
      {
        name: 'FLOPs',
        unit: 'G',
        baseline: 4.1,
        current: 1.2,
        improvement: -70.7,
        target: 1.5,
        status: 'excellent'
      },
      {
        name: '能耗',
        unit: 'J',
        baseline: 2.8,
        current: 1.0,
        improvement: -64.3,
        target: 1.2,
        status: 'excellent'
      }
    ])
  }, [])

  const getCompressionTypeText = (type: CompressionResult['compressionType']) => {
    switch (type) {
      case 'quantization': return '量化'
      case 'pruning': return '剪枝'
      case 'distillation': return '蒸馏'
      case 'mixed': return '混合'
      default: return type
    }
  }

  const getCompressionTypeColor = (type: CompressionResult['compressionType']) => {
    switch (type) {
      case 'quantization': return '#1890ff'
      case 'pruning': return '#722ed1'
      case 'distillation': return '#52c41a'
      case 'mixed': return '#fa8c16'
      default: return '#d9d9d9'
    }
  }

  const getScoreLevel = (score: number) => {
    if (score >= 90) return { level: '优秀', color: '#52c41a' }
    if (score >= 80) return { level: '良好', color: '#1890ff' }
    if (score >= 70) return { level: '一般', color: '#faad14' }
    return { level: '较差', color: '#ff4d4f' }
  }

  const getMetricStatus = (status: BenchmarkMetric['status']) => {
    switch (status) {
      case 'excellent': return { text: '优秀', color: '#52c41a' }
      case 'good': return { text: '良好', color: '#1890ff' }
      case 'fair': return { text: '一般', color: '#faad14' }
      case 'poor': return { text: '较差', color: '#ff4d4f' }
    }
  }

  const filteredResults = compressionResults.filter(result => {
    if (filterType === 'all') return true
    return result.compressionType === filterType
  }).sort((a, b) => {
    switch (sortBy) {
      case 'score': return b.score - a.score
      case 'speedup': return b.speedup - a.speedup
      case 'compression': return b.compressionRatio - a.compressionRatio
      case 'accuracy': return b.compressedAccuracy - a.compressedAccuracy
      default: return 0
    }
  })

  const resultColumns = [
    {
      title: '模型名称',
      key: 'name',
      render: (record: CompressionResult) => (
        <div>
          <Text strong>{record.name}</Text>
          <div style={{ fontSize: '12px', color: '#666' }}>
            基于 {record.modelName}
          </div>
        </div>
      )
    },
    {
      title: '压缩类型',
      key: 'type',
      render: (record: CompressionResult) => (
        <Tag color={getCompressionTypeColor(record.compressionType)}>
          {getCompressionTypeText(record.compressionType)}
        </Tag>
      )
    },
    {
      title: '精度',
      key: 'accuracy',
      render: (record: CompressionResult) => (
        <div>
          <Text strong>{record.compressedAccuracy.toFixed(1)}%</Text>
          <div style={{ fontSize: '12px', color: record.accuracyLoss > 2 ? '#ff4d4f' : '#52c41a' }}>
            损失: {record.accuracyLoss.toFixed(1)}%
          </div>
        </div>
      )
    },
    {
      title: '压缩比',
      key: 'compression',
      render: (record: CompressionResult) => (
        <div>
          <Text strong style={{ color: '#722ed1' }}>
            {record.compressionRatio.toFixed(1)}x
          </Text>
          <div style={{ fontSize: '12px', color: '#666' }}>
            {record.originalSize.toFixed(1)}MB → {record.compressedSize.toFixed(1)}MB
          </div>
        </div>
      )
    },
    {
      title: '加速比',
      key: 'speedup',
      render: (record: CompressionResult) => (
        <div>
          <Text strong style={{ color: '#1890ff' }}>
            {record.speedup.toFixed(1)}x
          </Text>
          <div style={{ fontSize: '12px', color: '#666' }}>
            {record.originalInference.toFixed(1)}ms → {record.compressedInference.toFixed(1)}ms
          </div>
        </div>
      )
    },
    {
      title: '综合评分',
      key: 'score',
      render: (record: CompressionResult) => {
        const { level, color } = getScoreLevel(record.score)
        return (
          <div>
            <Text strong style={{ color }}>
              {record.score.toFixed(1)}
            </Text>
            <div style={{ fontSize: '12px', color }}>
              {level}
            </div>
          </div>
        )
      }
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: CompressionResult) => (
        <Space>
          <Button
            type="link"
            size="small"
            icon={<EyeOutlined />}
            onClick={() => {
              setSelectedResult(record)
              setDetailModalVisible(true)
            }}
          >
            详情
          </Button>
          <Button
            type="link"
            size="small"
            icon={<SwapOutlined />}
            onClick={() => {
              if (selectedModels.includes(record.id)) {
                setSelectedModels(selectedModels.filter(id => id !== record.id))
              } else {
                setSelectedModels([...selectedModels, record.id])
              }
            }}
          >
            {selectedModels.includes(record.id) ? '取消' : '对比'}
          </Button>
        </Space>
      )
    }
  ]

  const renderOverview = () => {
    // 准备图表数据
    const accuracyData = compressionResults.map(result => ({
      model: result.name,
      baseline: result.baselineAccuracy,
      compressed: result.compressedAccuracy
    }))

    const performanceData = compressionResults.map(result => ({
      model: result.name,
      compression: result.compressionRatio,
      speedup: result.speedup,
      score: result.score
    }))

    const typeDistribution = compressionResults.reduce((acc, result) => {
      const type = getCompressionTypeText(result.compressionType)
      acc[type] = (acc[type] || 0) + 1
      return acc
    }, {} as Record<string, number>)

    const pieData = Object.entries(typeDistribution).map(([type, count]) => ({
      type,
      value: count
    }))

    return (
      <div>
        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          <Col span={6}>
            <Card>
              <Statistic
                title="压缩任务总数"
                value={compressionResults.length}
                prefix={<DatabaseOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="平均压缩比"
                value={compressionResults.reduce((sum, r) => sum + r.compressionRatio, 0) / compressionResults.length}
                precision={1}
                suffix="x"
                prefix={<SwapOutlined />}
                valueStyle={{ color: '#722ed1' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="平均加速比"
                value={compressionResults.reduce((sum, r) => sum + r.speedup, 0) / compressionResults.length}
                precision={1}
                suffix="x"
                prefix={<ThunderboltOutlined />}
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="平均评分"
                value={compressionResults.reduce((sum, r) => sum + r.score, 0) / compressionResults.length}
                precision={1}
                prefix={<TrophyOutlined />}
                valueStyle={{ color: '#52c41a' }}
              />
            </Card>
          </Col>
        </Row>

        <Row gutter={[16, 16]}>
          <Col span={12}>
            <Card title="精度对比" extra={<BarChartOutlined />}>
              <Column
                data={accuracyData.flatMap(item => [
                  { model: item.model, type: '原始精度', value: item.baseline },
                  { model: item.model, type: '压缩后精度', value: item.compressed }
                ])}
                xField="model"
                yField="value"
                seriesField="type"
                height={300}
                columnStyle={{
                  radius: [4, 4, 0, 0],
                }}
              />
            </Card>
          </Col>
          <Col span={12}>
            <Card title="性能分布" extra={<LineChartOutlined />}>
              <Line
                data={performanceData.flatMap(item => [
                  { model: item.model, metric: '压缩比', value: item.compression },
                  { model: item.model, metric: '加速比', value: item.speedup },
                  { model: item.model, metric: '综合评分/10', value: item.score / 10 }
                ])}
                xField="model"
                yField="value"
                seriesField="metric"
                height={300}
                smooth={true}
              />
            </Card>
          </Col>
        </Row>

        <Row gutter={[16, 16]} style={{ marginTop: '16px' }}>
          <Col span={12}>
            <Card title="压缩类型分布" extra={<PieChartOutlined />}>
              <Pie
                data={pieData}
                angleField="value"
                colorField="type"
                radius={0.8}
                height={300}
                label={{
                  type: 'outer',
                  content: '{name} {percentage}',
                }}
                interactions={[{ type: 'element-active' }]}
              />
            </Card>
          </Col>
          <Col span={12}>
            <Card title="基准测试指标" extra={<BarChartOutlined />}>
              <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                {benchmarkMetrics.map(metric => {
                  const status = getMetricStatus(metric.status)
                  return (
                    <div key={metric.name} style={{ marginBottom: '12px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                        <Text>{metric.name}</Text>
                        <Tag color={status.color}>{status.text}</Tag>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
                        <Text type="secondary" style={{ minWidth: '60px' }}>
                          当前: {metric.current}{metric.unit}
                        </Text>
                        <Progress
                          percent={Math.abs(metric.improvement)}
                          size="small"
                          status={metric.improvement < 0 ? 'success' : 'exception'}
                          showInfo={false}
                          style={{ flex: 1, margin: '0 8px' }}
                        />
                        <Text style={{ 
                          color: metric.improvement < 0 ? '#52c41a' : '#ff4d4f',
                          minWidth: '60px'
                        }}>
                          {metric.improvement > 0 ? '+' : ''}{metric.improvement.toFixed(1)}%
                        </Text>
                      </div>
                    </div>
                  )
                })}
              </div>
            </Card>
          </Col>
        </Row>
      </div>
    )
  }

  const renderComparison = () => (
    <div>
      {selectedModels.length === 0 ? (
        <Empty
          description="请从结果列表中选择要对比的模型"
          image={Empty.PRESENTED_IMAGE_SIMPLE}
        />
      ) : (
        <div>
          <Alert
            message={`已选择 ${selectedModels.length} 个模型进行对比`}
            type="info"
            showIcon
            style={{ marginBottom: '24px' }}
            action={
              <Button size="small" onClick={() => setSelectedModels([])}>
                清除选择
              </Button>
            }
          />

          <Row gutter={[16, 16]}>
            {selectedModels.map(modelId => {
              const result = compressionResults.find(r => r.id === modelId)
              if (!result) return null

              const { level, color } = getScoreLevel(result.score)

              return (
                <Col span={8} key={modelId}>
                  <Card
                    title={result.name}
                    extra={
                      <Tag color={getCompressionTypeColor(result.compressionType)}>
                        {getCompressionTypeText(result.compressionType)}
                      </Tag>
                    }
                    style={{ height: '100%' }}
                  >
                    <div style={{ textAlign: 'center', marginBottom: '16px' }}>
                      <Avatar
                        size={64}
                        icon={<DatabaseOutlined />}
                        style={{ backgroundColor: getCompressionTypeColor(result.compressionType) }}
                      />
                      <div style={{ marginTop: '8px' }}>
                        <Text strong style={{ fontSize: '24px', color }}>
                          {result.score.toFixed(1)}
                        </Text>
                        <div style={{ color, fontSize: '12px' }}>{level}</div>
                      </div>
                    </div>

                    <List size="small">
                      <List.Item>
                        <Text type="secondary">基础精度:</Text>
                        <Text strong style={{ marginLeft: '8px' }}>
                          {result.baselineAccuracy.toFixed(1)}%
                        </Text>
                      </List.Item>
                      <List.Item>
                        <Text type="secondary">压缩精度:</Text>
                        <Text strong style={{ marginLeft: '8px' }}>
                          {result.compressedAccuracy.toFixed(1)}%
                        </Text>
                      </List.Item>
                      <List.Item>
                        <Text type="secondary">精度损失:</Text>
                        <Text strong style={{ 
                          marginLeft: '8px',
                          color: result.accuracyLoss > 2 ? '#ff4d4f' : '#52c41a'
                        }}>
                          {result.accuracyLoss.toFixed(1)}%
                        </Text>
                      </List.Item>
                      <List.Item>
                        <Text type="secondary">模型大小:</Text>
                        <Text strong style={{ marginLeft: '8px' }}>
                          {result.originalSize.toFixed(1)}MB → {result.compressedSize.toFixed(1)}MB
                        </Text>
                      </List.Item>
                      <List.Item>
                        <Text type="secondary">压缩比:</Text>
                        <Text strong style={{ marginLeft: '8px', color: '#722ed1' }}>
                          {result.compressionRatio.toFixed(1)}x
                        </Text>
                      </List.Item>
                      <List.Item>
                        <Text type="secondary">推理时间:</Text>
                        <Text strong style={{ marginLeft: '8px' }}>
                          {result.originalInference.toFixed(1)}ms → {result.compressedInference.toFixed(1)}ms
                        </Text>
                      </List.Item>
                      <List.Item>
                        <Text type="secondary">加速比:</Text>
                        <Text strong style={{ marginLeft: '8px', color: '#1890ff' }}>
                          {result.speedup.toFixed(1)}x
                        </Text>
                      </List.Item>
                      <List.Item>
                        <Text type="secondary">内存减少:</Text>
                        <Text strong style={{ marginLeft: '8px', color: '#52c41a' }}>
                          {result.memoryReduction}%
                        </Text>
                      </List.Item>
                      <List.Item>
                        <Text type="secondary">能耗减少:</Text>
                        <Text strong style={{ marginLeft: '8px', color: '#52c41a' }}>
                          {result.energyReduction}%
                        </Text>
                      </List.Item>
                    </List>

                    <div style={{ marginTop: '16px' }}>
                      <Text type="secondary">标签:</Text>
                      <div style={{ marginTop: '4px' }}>
                        {result.tags.map(tag => (
                          <Tag key={tag} size="small">
                            {tag}
                          </Tag>
                        ))}
                      </div>
                    </div>
                  </Card>
                </Col>
              )
            })}
          </Row>
        </div>
      )}
    </div>
  )

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2} style={{ margin: 0, color: '#1a1a1a' }}>
          <BarChartOutlined style={{ marginRight: '12px', color: '#1890ff' }} />
          压缩评估与对比
        </Title>
        <Paragraph style={{ marginTop: '8px', color: '#666', fontSize: '16px' }}>
          分析和对比模型压缩结果，评估压缩效果和性能表现
        </Paragraph>
      </div>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="总览仪表板" key="overview">
          {renderOverview()}
        </TabPane>

        <TabPane tab="详细结果" key="results">
          <Card
            title="压缩结果"
            extra={
              <Space>
                <Select
                  value={filterType}
                  onChange={setFilterType}
                  style={{ width: 120 }}
                  size="small"
                >
                  <Option value="all">全部类型</Option>
                  <Option value="quantization">量化</Option>
                  <Option value="pruning">剪枝</Option>
                  <Option value="distillation">蒸馏</Option>
                  <Option value="mixed">混合</Option>
                </Select>
                <Select
                  value={sortBy}
                  onChange={setSortBy}
                  style={{ width: 120 }}
                  size="small"
                >
                  <Option value="score">综合评分</Option>
                  <Option value="speedup">加速比</Option>
                  <Option value="compression">压缩比</Option>
                  <Option value="accuracy">精度</Option>
                </Select>
                <Button
                  size="small"
                  icon={<SwapOutlined />}
                  disabled={selectedModels.length === 0}
                  onClick={() => setActiveTab('comparison')}
                >
                  对比 ({selectedModels.length})
                </Button>
              </Space>
            }
          >
            <Table
              columns={resultColumns}
              dataSource={filteredResults}
              rowKey="id"
              pagination={{ pageSize: 10 }}
              rowSelection={{
                type: 'checkbox',
                selectedRowKeys: selectedModels,
                onChange: setSelectedModels
              }}
            />
          </Card>
        </TabPane>

        <TabPane tab="模型对比" key="comparison">
          {renderComparison()}
        </TabPane>
      </Tabs>

      {/* 详情模态框 */}
      <Modal
        title="压缩结果详情"
        open={detailModalVisible}
        onCancel={() => setDetailModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setDetailModalVisible(false)}>
            关闭
          </Button>,
          <Button key="download" type="primary" icon={<DownloadOutlined />}>
            下载报告
          </Button>
        ]}
        width={1000}
      >
        {selectedResult && (
          <div>
            <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
              <Col span={8}>
                <Card title="基础信息" size="small">
                  <List size="small">
                    <List.Item>
                      <Text type="secondary">任务名称:</Text>
                      <Text strong style={{ marginLeft: '8px' }}>
                        {selectedResult.name}
                      </Text>
                    </List.Item>
                    <List.Item>
                      <Text type="secondary">基础模型:</Text>
                      <Text strong style={{ marginLeft: '8px' }}>
                        {selectedResult.modelName}
                      </Text>
                    </List.Item>
                    <List.Item>
                      <Text type="secondary">压缩类型:</Text>
                      <Tag
                        color={getCompressionTypeColor(selectedResult.compressionType)}
                        style={{ marginLeft: '8px' }}
                      >
                        {getCompressionTypeText(selectedResult.compressionType)}
                      </Tag>
                    </List.Item>
                    <List.Item>
                      <Text type="secondary">创建时间:</Text>
                      <Text strong style={{ marginLeft: '8px' }}>
                        {selectedResult.createTime}
                      </Text>
                    </List.Item>
                  </List>
                </Card>
              </Col>

              <Col span={8}>
                <Card title="精度指标" size="small">
                  <List size="small">
                    <List.Item>
                      <Text type="secondary">基础精度:</Text>
                      <Text strong style={{ marginLeft: '8px' }}>
                        {selectedResult.baselineAccuracy.toFixed(2)}%
                      </Text>
                    </List.Item>
                    <List.Item>
                      <Text type="secondary">压缩后精度:</Text>
                      <Text strong style={{ marginLeft: '8px' }}>
                        {selectedResult.compressedAccuracy.toFixed(2)}%
                      </Text>
                    </List.Item>
                    <List.Item>
                      <Text type="secondary">精度损失:</Text>
                      <Text strong style={{ 
                        marginLeft: '8px',
                        color: selectedResult.accuracyLoss > 2 ? '#ff4d4f' : '#52c41a'
                      }}>
                        {selectedResult.accuracyLoss.toFixed(2)}%
                      </Text>
                    </List.Item>
                    <List.Item>
                      <Text type="secondary">精度保持率:</Text>
                      <Text strong style={{ marginLeft: '8px', color: '#52c41a' }}>
                        {((selectedResult.compressedAccuracy / selectedResult.baselineAccuracy) * 100).toFixed(1)}%
                      </Text>
                    </List.Item>
                  </List>
                </Card>
              </Col>

              <Col span={8}>
                <Card title="性能指标" size="small">
                  <List size="small">
                    <List.Item>
                      <Text type="secondary">模型压缩:</Text>
                      <Text strong style={{ marginLeft: '8px', color: '#722ed1' }}>
                        {selectedResult.compressionRatio.toFixed(1)}x
                      </Text>
                    </List.Item>
                    <List.Item>
                      <Text type="secondary">推理加速:</Text>
                      <Text strong style={{ marginLeft: '8px', color: '#1890ff' }}>
                        {selectedResult.speedup.toFixed(1)}x
                      </Text>
                    </List.Item>
                    <List.Item>
                      <Text type="secondary">内存减少:</Text>
                      <Text strong style={{ marginLeft: '8px', color: '#52c41a' }}>
                        {selectedResult.memoryReduction}%
                      </Text>
                    </List.Item>
                    <List.Item>
                      <Text type="secondary">能耗减少:</Text>
                      <Text strong style={{ marginLeft: '8px', color: '#52c41a' }}>
                        {selectedResult.energyReduction}%
                      </Text>
                    </List.Item>
                  </List>
                </Card>
              </Col>
            </Row>

            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Card title="大小对比" size="small">
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-around', alignItems: 'center' }}>
                      <div>
                        <Avatar size={64} icon={<DatabaseOutlined />} style={{ backgroundColor: '#ff4d4f' }} />
                        <div style={{ marginTop: '8px' }}>
                          <Text strong>原始模型</Text>
                          <div style={{ fontSize: '12px', color: '#666' }}>
                            {selectedResult.originalSize.toFixed(1)} MB
                          </div>
                        </div>
                      </div>
                      <div style={{ fontSize: '24px', color: '#722ed1' }}>
                        →
                      </div>
                      <div>
                        <Avatar size={64} icon={<SwapOutlined />} style={{ backgroundColor: '#52c41a' }} />
                        <div style={{ marginTop: '8px' }}>
                          <Text strong>压缩模型</Text>
                          <div style={{ fontSize: '12px', color: '#666' }}>
                            {selectedResult.compressedSize.toFixed(1)} MB
                          </div>
                        </div>
                      </div>
                    </div>
                    <Divider />
                    <Statistic
                      title="压缩比"
                      value={selectedResult.compressionRatio}
                      precision={1}
                      suffix="x"
                      valueStyle={{ color: '#722ed1' }}
                    />
                  </div>
                </Card>
              </Col>

              <Col span={12}>
                <Card title="性能对比" size="small">
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-around', alignItems: 'center' }}>
                      <div>
                        <Avatar size={64} icon={<ClockCircleOutlined />} style={{ backgroundColor: '#ff4d4f' }} />
                        <div style={{ marginTop: '8px' }}>
                          <Text strong>原始推理</Text>
                          <div style={{ fontSize: '12px', color: '#666' }}>
                            {selectedResult.originalInference.toFixed(1)} ms
                          </div>
                        </div>
                      </div>
                      <div style={{ fontSize: '24px', color: '#1890ff' }}>
                        →
                      </div>
                      <div>
                        <Avatar size={64} icon={<ThunderboltOutlined />} style={{ backgroundColor: '#52c41a' }} />
                        <div style={{ marginTop: '8px' }}>
                          <Text strong>压缩推理</Text>
                          <div style={{ fontSize: '12px', color: '#666' }}>
                            {selectedResult.compressedInference.toFixed(1)} ms
                          </div>
                        </div>
                      </div>
                    </div>
                    <Divider />
                    <Statistic
                      title="加速比"
                      value={selectedResult.speedup}
                      precision={1}
                      suffix="x"
                      valueStyle={{ color: '#1890ff' }}
                    />
                  </div>
                </Card>
              </Col>
            </Row>
          </div>
        )}
      </Modal>
    </div>
  )
}

export default ModelCompressionEvaluationPage