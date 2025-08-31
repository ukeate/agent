import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Statistic,
  Progress,
  Table,
  Button,
  Space,
  Typography,
  Tabs,
  List,
  Avatar,
  Tag,
  Badge,
  Alert,
  Spin,
  Divider
} from 'antd'
import {
  CompressOutlined,
  ScissorOutlined,
  ExperimentFilled,
  ThunderboltOutlined,
  BarChartOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  SyncOutlined,
  ExclamationCircleOutlined,
  PlayCircleOutlined,
  RightOutlined,
  LineChartOutlined,
  DatabaseOutlined,
  RocketOutlined,
  BulbOutlined
} from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs

interface CompressionJob {
  id: string
  name: string
  type: 'quantization' | 'pruning' | 'distillation' | 'mixed'
  status: 'running' | 'completed' | 'failed' | 'pending'
  progress: number
  originalSize: number
  compressedSize: number
  compressionRatio: number
  accuracyLoss: number
  startTime: string
  estimatedTime?: number
}

interface SystemStats {
  totalJobs: number
  activeJobs: number
  completedJobs: number
  failedJobs: number
  avgCompressionRatio: number
  avgAccuracyLoss: number
  totalModelsSaved: number
  storageReduced: number
}

const ModelCompressionOverviewPage: React.FC = () => {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(true)
  const [stats, setStats] = useState<SystemStats | null>(null)
  const [recentJobs, setRecentJobs] = useState<CompressionJob[]>([])
  const [activeTab, setActiveTab] = useState('overview')

  useEffect(() => {
    // 模拟数据加载
    setTimeout(() => {
      setStats({
        totalJobs: 156,
        activeJobs: 8,
        completedJobs: 142,
        failedJobs: 6,
        avgCompressionRatio: 3.2,
        avgAccuracyLoss: 2.1,
        totalModelsSaved: 89,
        storageReduced: 2.4
      })

      setRecentJobs([
        {
          id: 'job-001',
          name: 'BERT-Base INT8量化',
          type: 'quantization',
          status: 'running',
          progress: 65,
          originalSize: 438,
          compressedSize: 110,
          compressionRatio: 4.0,
          accuracyLoss: 1.2,
          startTime: '2024-08-24 10:30:00',
          estimatedTime: 15
        },
        {
          id: 'job-002', 
          name: 'ResNet50知识蒸馏',
          type: 'distillation',
          status: 'completed',
          progress: 100,
          originalSize: 102,
          compressedSize: 28,
          compressionRatio: 3.6,
          accuracyLoss: 0.8,
          startTime: '2024-08-24 09:15:00'
        },
        {
          id: 'job-003',
          name: 'GPT-2结构化剪枝',
          type: 'pruning', 
          status: 'running',
          progress: 32,
          originalSize: 548,
          compressedSize: 274,
          compressionRatio: 2.0,
          accuracyLoss: 3.1,
          startTime: '2024-08-24 11:00:00',
          estimatedTime: 45
        },
        {
          id: 'job-004',
          name: 'MobileNet混合压缩',
          type: 'mixed',
          status: 'completed',
          progress: 100,
          originalSize: 16,
          compressedSize: 3.2,
          compressionRatio: 5.0,
          accuracyLoss: 1.5,
          startTime: '2024-08-24 08:20:00'
        },
        {
          id: 'job-005',
          name: 'Transformer量化训练',
          type: 'quantization',
          status: 'failed',
          progress: 0,
          originalSize: 267,
          compressedSize: 0,
          compressionRatio: 0,
          accuracyLoss: 0,
          startTime: '2024-08-24 12:45:00'
        }
      ])

      setLoading(false)
    }, 1000)
  }, [])

  const getStatusIcon = (status: CompressionJob['status']) => {
    switch (status) {
      case 'running': return <SyncOutlined spin style={{ color: '#1890ff' }} />
      case 'completed': return <CheckCircleOutlined style={{ color: '#52c41a' }} />
      case 'failed': return <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />
      case 'pending': return <ClockCircleOutlined style={{ color: '#faad14' }} />
      default: return null
    }
  }

  const getStatusText = (status: CompressionJob['status']) => {
    switch (status) {
      case 'running': return '运行中'
      case 'completed': return '已完成'
      case 'failed': return '失败'
      case 'pending': return '等待中'
      default: return '未知'
    }
  }

  const getTypeIcon = (type: CompressionJob['type']) => {
    switch (type) {
      case 'quantization': return <CompressOutlined style={{ color: '#1890ff' }} />
      case 'pruning': return <ScissorOutlined style={{ color: '#722ed1' }} />
      case 'distillation': return <ExperimentFilled style={{ color: '#fa8c16' }} />
      case 'mixed': return <ThunderboltOutlined style={{ color: '#52c41a' }} />
      default: return null
    }
  }

  const getTypeText = (type: CompressionJob['type']) => {
    switch (type) {
      case 'quantization': return '量化'
      case 'pruning': return '剪枝'
      case 'distillation': return '蒸馏'
      case 'mixed': return '混合'
      default: return '未知'
    }
  }

  const quickActions = [
    {
      title: '新建量化任务',
      description: '创建模型量化压缩任务',
      icon: <CompressOutlined />,
      color: '#1890ff',
      path: '/quantization-manager'
    },
    {
      title: '知识蒸馏',
      description: '启动Teacher-Student蒸馏',
      icon: <ExperimentFilled />,
      color: '#fa8c16',
      path: '/knowledge-distillation'
    },
    {
      title: '模型剪枝',
      description: '配置结构化/非结构化剪枝',
      icon: <ScissorOutlined />,
      color: '#722ed1',
      path: '/model-pruning'
    },
    {
      title: '性能基准',
      description: '运行硬件性能测试',
      icon: <RocketOutlined />,
      color: '#52c41a',
      path: '/hardware-benchmark'
    }
  ]

  const recentJobsColumns = [
    {
      title: '任务名称',
      key: 'name',
      render: (record: CompressionJob) => (
        <Space>
          {getTypeIcon(record.type)}
          <div>
            <div style={{ fontWeight: 500 }}>{record.name}</div>
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {record.id}
            </Text>
          </div>
        </Space>
      )
    },
    {
      title: '类型',
      key: 'type',
      render: (record: CompressionJob) => (
        <Tag color={record.type === 'quantization' ? 'blue' : 
                   record.type === 'pruning' ? 'purple' :
                   record.type === 'distillation' ? 'orange' : 'green'}>
          {getTypeText(record.type)}
        </Tag>
      )
    },
    {
      title: '状态',
      key: 'status',
      render: (record: CompressionJob) => (
        <Space>
          {getStatusIcon(record.status)}
          <span>{getStatusText(record.status)}</span>
        </Space>
      )
    },
    {
      title: '进度',
      key: 'progress',
      render: (record: CompressionJob) => (
        <div style={{ width: 80 }}>
          <Progress 
            percent={record.progress} 
            size="small" 
            status={record.status === 'failed' ? 'exception' : undefined}
          />
        </div>
      )
    },
    {
      title: '压缩比',
      key: 'compressionRatio',
      render: (record: CompressionJob) => (
        record.compressionRatio > 0 ? 
          <Text strong style={{ color: '#52c41a' }}>
            {record.compressionRatio}x
          </Text> : 
          <Text type="secondary">-</Text>
      )
    },
    {
      title: '精度损失',
      key: 'accuracyLoss',
      render: (record: CompressionJob) => (
        record.accuracyLoss > 0 ? 
          <Text style={{ color: record.accuracyLoss > 5 ? '#ff4d4f' : '#faad14' }}>
            {record.accuracyLoss}%
          </Text> : 
          <Text type="secondary">-</Text>
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: CompressionJob) => (
        <Space>
          <Button 
            type="link" 
            size="small"
            onClick={() => navigate(`/compression-jobs/${record.id}`)}
          >
            查看详情
          </Button>
          {record.status === 'running' && (
            <Button 
              type="link" 
              size="small" 
              danger
            >
              暂停
            </Button>
          )}
        </Space>
      )
    }
  ]

  if (loading) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '400px' 
      }}>
        <Spin size="large" tip="加载模型压缩数据..." />
      </div>
    )
  }

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2} style={{ margin: 0, color: '#1a1a1a' }}>
          <CompressOutlined style={{ marginRight: '12px', color: '#1890ff' }} />
          模型压缩和量化工具总览
        </Title>
        <Paragraph style={{ marginTop: '8px', color: '#666', fontSize: '16px' }}>
          全面管理和监控模型压缩任务，提升AI模型部署效率
        </Paragraph>
      </div>

      {/* 系统统计卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="总任务数"
              value={stats?.totalJobs}
              prefix={<DatabaseOutlined style={{ color: '#1890ff' }} />}
              suffix="个"
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="活跃任务"
              value={stats?.activeJobs}
              prefix={<SyncOutlined style={{ color: '#faad14' }} />}
              suffix="个"
            />
            <Progress 
              percent={(stats?.activeJobs || 0) * 10} 
              size="small" 
              showInfo={false}
              style={{ marginTop: '8px' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="平均压缩比"
              value={stats?.avgCompressionRatio}
              prefix={<CompressOutlined style={{ color: '#52c41a' }} />}
              suffix="x"
              precision={1}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="存储节省"
              value={stats?.storageReduced}
              prefix={<ThunderboltOutlined style={{ color: '#722ed1' }} />}
              suffix="GB"
              precision={1}
            />
          </Card>
        </Col>
      </Row>

      {/* 快捷操作 */}
      <Card title="快捷操作" style={{ marginBottom: '24px' }}>
        <Row gutter={[16, 16]}>
          {quickActions.map((action, index) => (
            <Col xs={24} sm={12} lg={6} key={index}>
              <Card 
                hoverable
                style={{ textAlign: 'center', cursor: 'pointer' }}
                onClick={() => navigate(action.path)}
                bodyStyle={{ padding: '24px 16px' }}
              >
                <Avatar 
                  size={48} 
                  style={{ backgroundColor: action.color, marginBottom: '16px' }}
                  icon={action.icon}
                />
                <div>
                  <Text strong style={{ display: 'block', marginBottom: '8px' }}>
                    {action.title}
                  </Text>
                  <Text type="secondary" style={{ fontSize: '12px' }}>
                    {action.description}
                  </Text>
                </div>
                <Button 
                  type="primary" 
                  size="small" 
                  style={{ marginTop: '12px' }}
                  icon={<RightOutlined />}
                >
                  立即开始
                </Button>
              </Card>
            </Col>
          ))}
        </Row>
      </Card>

      {/* 详细信息标签页 */}
      <Card>
        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane tab="最近任务" key="recent">
            <div style={{ marginBottom: '16px' }}>
              <Space>
                <Button type="primary" onClick={() => navigate('/compression-jobs')}>
                  查看所有任务
                </Button>
                <Button onClick={() => navigate('/quantization-manager')}>
                  新建任务
                </Button>
                <Button onClick={() => navigate('/compression-monitor')}>
                  监控面板
                </Button>
              </Space>
            </div>
            
            <Table
              columns={recentJobsColumns}
              dataSource={recentJobs}
              rowKey="id"
              size="middle"
              pagination={{ 
                pageSize: 5,
                showSizeChanger: false,
                showQuickJumper: true
              }}
            />
          </TabPane>
          
          <TabPane tab="性能统计" key="stats">
            <Row gutter={[16, 16]}>
              <Col xs={24} lg={12}>
                <Card title="压缩效果分布" size="small">
                  <div style={{ textAlign: 'center', padding: '20px' }}>
                    <Progress 
                      type="circle" 
                      percent={75}
                      format={() => '75%'}
                      width={120}
                    />
                    <div style={{ marginTop: '16px' }}>
                      <Text type="secondary">任务成功率</Text>
                    </div>
                  </div>
                  
                  <Divider />
                  
                  <Row gutter={16}>
                    <Col span={12}>
                      <Statistic
                        title="最佳压缩比"
                        value={5.2}
                        suffix="x"
                        precision={1}
                      />
                    </Col>
                    <Col span={12}>
                      <Statistic
                        title="最低精度损失"
                        value={0.3}
                        suffix="%"
                        precision={1}
                      />
                    </Col>
                  </Row>
                </Card>
              </Col>
              
              <Col xs={24} lg={12}>
                <Card title="系统资源使用" size="small">
                  <div style={{ marginBottom: '16px' }}>
                    <Text type="secondary">GPU使用率</Text>
                    <Progress percent={68} />
                  </div>
                  <div style={{ marginBottom: '16px' }}>
                    <Text type="secondary">内存使用率</Text>
                    <Progress percent={45} />
                  </div>
                  <div style={{ marginBottom: '16px' }}>
                    <Text type="secondary">存储使用率</Text>
                    <Progress percent={32} />
                  </div>
                  
                  <Alert
                    message="系统运行正常"
                    description="所有资源使用率在正常范围内"
                    type="success"
                    showIcon
                    style={{ marginTop: '16px' }}
                  />
                </Card>
              </Col>
            </Row>
          </TabPane>
          
          <TabPane tab="系统配置" key="config">
            <Row gutter={[16, 16]}>
              <Col xs={24} lg={8}>
                <Card title="量化引擎配置" size="small">
                  <List size="small">
                    <List.Item>
                      <List.Item.Meta
                        avatar={<CompressOutlined style={{ color: '#1890ff' }} />}
                        title="支持的精度"
                        description="INT8, INT4, FP16"
                      />
                    </List.Item>
                    <List.Item>
                      <List.Item.Meta
                        avatar={<BulbOutlined style={{ color: '#52c41a' }} />}
                        title="量化算法"
                        description="PTQ, QAT, GPTQ, AWQ"
                      />
                    </List.Item>
                  </List>
                </Card>
              </Col>
              
              <Col xs={24} lg={8}>
                <Card title="蒸馏引擎配置" size="small">
                  <List size="small">
                    <List.Item>
                      <List.Item.Meta
                        avatar={<ExperimentFilled style={{ color: '#fa8c16' }} />}
                        title="蒸馏策略"
                        description="Response, Feature, Attention"
                      />
                    </List.Item>
                    <List.Item>
                      <List.Item.Meta
                        avatar={<LineChartOutlined style={{ color: '#722ed1' }} />}
                        title="温度范围"
                        description="1.0 - 10.0"
                      />
                    </List.Item>
                  </List>
                </Card>
              </Col>
              
              <Col xs={24} lg={8}>
                <Card title="剪枝引擎配置" size="small">
                  <List size="small">
                    <List.Item>
                      <List.Item.Meta
                        avatar={<ScissorOutlined style={{ color: '#722ed1' }} />}
                        title="剪枝类型"
                        description="结构化, 非结构化"
                      />
                    </List.Item>
                    <List.Item>
                      <List.Item.Meta
                        avatar={<BarChartOutlined style={{ color: '#1890ff' }} />}
                        title="稀疏度范围"
                        description="0.1 - 0.9"
                      />
                    </List.Item>
                  </List>
                </Card>
              </Col>
            </Row>
          </TabPane>
        </Tabs>
      </Card>
    </div>
  )
}

export default ModelCompressionOverviewPage