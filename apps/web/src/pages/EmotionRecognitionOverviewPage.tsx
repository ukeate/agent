/**
 * 多模态情感识别引擎总览页面
 * Story 11.1: 多模态情感识别引擎
 */

import React, { useState, useEffect } from 'react'
import { 
  Card, 
  Row, 
  Col, 
  Typography, 
  Space, 
  Tabs, 
  Statistic, 
  Progress,
  Tag,
  Badge,
  Alert,
  Button,
  Divider,
  Timeline,
  Avatar,
  List,
  Tooltip,
  message
} from 'antd'
import {
  SmileOutlined,
  HeartOutlined,
  ThunderboltOutlined,
  BulbOutlined,
  SoundOutlined,
  CameraOutlined,
  FileTextOutlined,
  DashboardOutlined,
  BarChartOutlined,
  LineChartOutlined,
  PieChartOutlined,
  RadarChartOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  SyncOutlined,
  RobotOutlined,
  UserOutlined,
  GlobalOutlined,
  ApiOutlined,
  CloudServerOutlined,
  DatabaseOutlined
} from '@ant-design/icons'
import { Line, Column, Pie, Radar, Gauge } from '@ant-design/plots'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs

const EmotionRecognitionOverviewPage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [systemStats, setSystemStats] = useState({
    totalRequests: 152847,
    averageLatency: 342,
    accuracyRate: 95.3,
    activeModels: 3,
    processedToday: 8421,
    emotionDistribution: {
      happiness: 35,
      sadness: 15,
      anger: 10,
      fear: 8,
      surprise: 12,
      neutral: 20
    }
  })

  // 模拟实时数据更新
  useEffect(() => {
    const interval = setInterval(() => {
      setSystemStats(prev => ({
        ...prev,
        totalRequests: prev.totalRequests + Math.floor(Math.random() * 10),
        processedToday: prev.processedToday + Math.floor(Math.random() * 5)
      }))
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  // 情感分布饼图配置
  const emotionPieConfig = {
    appendPadding: 10,
    data: Object.entries(systemStats.emotionDistribution).map(([emotion, value]) => ({
      type: emotion,
      value
    })),
    angleField: 'value',
    colorField: 'type',
    radius: 0.8,
    label: {
      type: 'outer',
      content: '{name} {percentage}'
    },
    interactions: [{ type: 'pie-legend-active' }, { type: 'element-active' }],
    color: ['#52c41a', '#1890ff', '#f5222d', '#faad14', '#722ed1', '#8c8c8c']
  }

  // 准确率仪表盘配置
  const accuracyGaugeConfig = {
    percent: systemStats.accuracyRate / 100,
    range: {
      color: 'l(0) 0:#B8E1FF 1:#3D76DD'
    },
    startAngle: Math.PI,
    endAngle: 2 * Math.PI,
    indicator: null,
    statistic: {
      title: {
        offsetY: -36,
        style: {
          fontSize: '20px',
          color: '#4B535E'
        },
        formatter: () => '综合准确率'
      },
      content: {
        style: {
          fontSize: '24px',
          lineHeight: '44px',
          color: '#4B535E'
        },
        formatter: () => `${systemStats.accuracyRate}%`
      }
    }
  }

  // 模态性能雷达图配置
  const modalityRadarConfig = {
    data: [
      { item: '文本', score: 92 },
      { item: '语音', score: 88 },
      { item: '图像', score: 85 },
      { item: '响应速度', score: 90 },
      { item: '稳定性', score: 95 },
      { item: '准确度', score: 93 }
    ],
    xField: 'item',
    yField: 'score',
    meta: {
      score: {
        alias: '分数',
        min: 0,
        max: 100
      }
    },
    xAxis: {
      line: null,
      tickLine: null,
      grid: {
        line: {
          style: {
            lineDash: null
          }
        }
      }
    },
    point: {
      size: 2
    },
    area: {}
  }

  // 实时处理时间线图配置
  const timelineConfig = {
    data: Array.from({ length: 24 }, (_, i) => ({
      time: `${i}:00`,
      value: Math.floor(Math.random() * 500) + 200,
      type: '处理量'
    })),
    xField: 'time',
    yField: 'value',
    seriesField: 'type',
    smooth: true,
    animation: {
      appear: {
        animation: 'path-in',
        duration: 2000
      }
    }
  }

  const recentAnalysis = [
    {
      id: 1,
      modality: 'multimodal',
      emotion: 'happiness',
      confidence: 0.92,
      timestamp: '2分钟前',
      user: 'User_A1B2',
      details: '文本+语音+图像融合分析'
    },
    {
      id: 2,
      modality: 'text',
      emotion: 'sadness',
      confidence: 0.87,
      timestamp: '5分钟前',
      user: 'User_C3D4',
      details: '文本情感分析'
    },
    {
      id: 3,
      modality: 'audio',
      emotion: 'excitement',
      confidence: 0.91,
      timestamp: '8分钟前',
      user: 'User_E5F6',
      details: '语音情感识别'
    }
  ]

  const getEmotionIcon = (emotion: string) => {
    const icons: Record<string, React.ReactNode> = {
      happiness: <SmileOutlined style={{ color: '#52c41a' }} />,
      sadness: <SmileOutlined style={{ color: '#1890ff', transform: 'rotate(180deg)' }} />,
      anger: <ThunderboltOutlined style={{ color: '#f5222d' }} />,
      fear: <HeartOutlined style={{ color: '#faad14' }} />,
      surprise: <SmileOutlined style={{ color: '#722ed1' }} />,
      excitement: <SmileOutlined style={{ color: '#fa8c16' }} />,
      neutral: <SmileOutlined style={{ color: '#8c8c8c' }} />
    }
    return icons[emotion] || <SmileOutlined />
  }

  const getModalityIcon = (modality: string) => {
    const icons: Record<string, React.ReactNode> = {
      text: <FileTextOutlined />,
      audio: <SoundOutlined />,
      visual: <CameraOutlined />,
      multimodal: <BulbOutlined />
    }
    return icons[modality] || <BulbOutlined />
  }

  return (
    <div style={{ padding: '24px' }}>
      {/* 页面标题 */}
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>
          <Space>
            <BulbOutlined />
            多模态情感识别引擎
          </Space>
        </Title>
        <Paragraph type="secondary">
          实时分析文本、语音、图像中的情感状态，提供细粒度情感分类和多模态融合分析
        </Paragraph>
      </div>

      {/* 系统状态提示 */}
      <Alert
        message="系统运行正常"
        description={`当前有 ${systemStats.activeModels} 个模型在线，平均响应时间 ${systemStats.averageLatency}ms`}
        type="success"
        showIcon
        closable
        style={{ marginBottom: 24 }}
        action={
          <Space>
            <Button size="small" type="primary">
              查看详情
            </Button>
            <Button size="small">优化建议</Button>
          </Space>
        }
      />

      {/* 核心指标卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card hoverable>
            <Statistic
              title="总处理请求"
              value={systemStats.totalRequests}
              prefix={<ApiOutlined />}
              suffix="次"
              valueStyle={{ color: '#3f8600' }}
            />
            <div style={{ marginTop: 8 }}>
              <Text type="secondary">今日: {systemStats.processedToday}</Text>
            </div>
            <Progress percent={75} strokeColor="#52c41a" />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card hoverable>
            <Statistic
              title="平均延迟"
              value={systemStats.averageLatency}
              prefix={<ClockCircleOutlined />}
              suffix="ms"
              valueStyle={{ color: '#1890ff' }}
            />
            <div style={{ marginTop: 8 }}>
              <Text type="secondary">目标: &lt;500ms</Text>
            </div>
            <Progress percent={68} strokeColor="#1890ff" />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card hoverable>
            <Statistic
              title="综合准确率"
              value={systemStats.accuracyRate}
              prefix={<CheckCircleOutlined />}
              suffix="%"
              valueStyle={{ color: '#cf1322' }}
            />
            <div style={{ marginTop: 8 }}>
              <Text type="secondary">目标: &gt;95%</Text>
            </div>
            <Progress percent={systemStats.accuracyRate} strokeColor="#ff4d4f" />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card hoverable>
            <Statistic
              title="活跃模型"
              value={systemStats.activeModels}
              prefix={<RobotOutlined />}
              suffix="个"
              valueStyle={{ color: '#722ed1' }}
            />
            <div style={{ marginTop: 8 }}>
              <Space>
                <Badge status="success" text="文本" />
                <Badge status="success" text="语音" />
                <Badge status="success" text="视觉" />
              </Space>
            </div>
          </Card>
        </Col>
      </Row>

      {/* 主要内容标签页 */}
      <Tabs defaultActiveKey="1" type="card">
        <TabPane tab={<Space><DashboardOutlined />实时监控</Space>} key="1">
          <Row gutter={[16, 16]}>
            <Col xs={24} md={12}>
              <Card title="情感分布统计" bordered={false}>
                <Pie {...emotionPieConfig} />
              </Card>
            </Col>
            <Col xs={24} md={12}>
              <Card title="模态性能指标" bordered={false}>
                <Radar {...modalityRadarConfig} />
              </Card>
            </Col>
            <Col xs={24}>
              <Card title="24小时处理趋势" bordered={false}>
                <Line {...timelineConfig} />
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab={<Space><BarChartOutlined />分析记录</Space>} key="2">
          <Card bordered={false}>
            <List
              itemLayout="horizontal"
              dataSource={recentAnalysis}
              renderItem={item => (
                <List.Item
                  actions={[
                    <Button type="link">详情</Button>,
                    <Button type="link">重播</Button>
                  ]}
                >
                  <List.Item.Meta
                    avatar={
                      <Avatar icon={getModalityIcon(item.modality)} size={48} />
                    }
                    title={
                      <Space>
                        <Text strong>{item.user}</Text>
                        <Tag icon={getEmotionIcon(item.emotion)} color="success">
                          {item.emotion}
                        </Tag>
                        <Badge 
                          count={`${(item.confidence * 100).toFixed(1)}%`} 
                          style={{ backgroundColor: '#52c41a' }} 
                        />
                      </Space>
                    }
                    description={
                      <Space direction="vertical" size="small">
                        <Text type="secondary">{item.details}</Text>
                        <Text type="secondary">
                          <ClockCircleOutlined /> {item.timestamp}
                        </Text>
                      </Space>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </TabPane>

        <TabPane tab={<Space><LineChartOutlined />性能分析</Space>} key="3">
          <Row gutter={[16, 16]}>
            <Col xs={24} md={12}>
              <Card title="综合准确率仪表盘" bordered={false}>
                <Gauge {...accuracyGaugeConfig} />
              </Card>
            </Col>
            <Col xs={24} md={12}>
              <Card title="模型状态" bordered={false}>
                <Timeline>
                  <Timeline.Item color="green" dot={<CheckCircleOutlined />}>
                    文本情感分析器 - 运行正常 (准确率: 92%)
                  </Timeline.Item>
                  <Timeline.Item color="green" dot={<CheckCircleOutlined />}>
                    语音情感识别器 - 运行正常 (准确率: 88%)
                  </Timeline.Item>
                  <Timeline.Item color="green" dot={<CheckCircleOutlined />}>
                    视觉情感分析器 - 运行正常 (准确率: 85%)
                  </Timeline.Item>
                  <Timeline.Item color="blue" dot={<SyncOutlined spin />}>
                    多模态融合引擎 - 处理中 (融合准确率: 95.3%)
                  </Timeline.Item>
                </Timeline>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab={<Space><GlobalOutlined />系统配置</Space>} key="4">
          <Row gutter={[16, 16]}>
            <Col xs={24} md={8}>
              <Card title="融合策略" bordered={false}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Tag color="blue">动态自适应融合</Tag>
                  <Text>权重分配:</Text>
                  <Progress percent={40} strokeColor="#1890ff" format={() => '文本 40%'} />
                  <Progress percent={35} strokeColor="#52c41a" format={() => '语音 35%'} />
                  <Progress percent={25} strokeColor="#fa8c16" format={() => '视觉 25%'} />
                </Space>
              </Card>
            </Col>
            <Col xs={24} md={8}>
              <Card title="处理配置" bordered={false}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Text>最大延迟: 500ms</Text>
                  <Text>批处理大小: 32</Text>
                  <Text>并发线程: 8</Text>
                  <Text>缓存策略: LRU</Text>
                  <Text>GPU加速: 启用</Text>
                </Space>
              </Card>
            </Col>
            <Col xs={24} md={8}>
              <Card title="API配置" bordered={false}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Text>RESTful端点: /api/v1/emotion</Text>
                  <Text>WebSocket: ws://localhost:8000/ws</Text>
                  <Text>批量处理: 启用</Text>
                  <Text>实时流: 启用</Text>
                  <Button type="primary" block>
                    更新配置
                  </Button>
                </Space>
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>

      {/* 底部操作栏 */}
      <Card style={{ marginTop: 24 }}>
        <Space size="large" wrap>
          <Button type="primary" icon={<SyncOutlined />} loading={loading}>
            刷新数据
          </Button>
          <Button icon={<DatabaseOutlined />}>
            导出报告
          </Button>
          <Button icon={<CloudServerOutlined />}>
            系统诊断
          </Button>
          <Button icon={<BarChartOutlined />}>
            性能优化建议
          </Button>
        </Space>
      </Card>
    </div>
  )
}

export default EmotionRecognitionOverviewPage