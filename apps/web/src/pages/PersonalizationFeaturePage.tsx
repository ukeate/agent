import React, { useState, useEffect } from 'react'
import { Card, Row, Col, Table, Progress, Tag, Button, Space, Select, Slider, Switch, Typography, Timeline, Statistic, Alert } from 'antd'
import { 
  BranchesOutlined,
  FieldTimeOutlined,
  FunctionOutlined,
  HistoryOutlined,
  DatabaseOutlined,
  ClockCircleOutlined,
  SyncOutlined,
  SettingOutlined,
  ExperimentOutlined,
  LineChartOutlined,
  CheckCircleOutlined,
  InfoCircleOutlined,
  UserOutlined
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import { Line, Heatmap } from '@ant-design/plots'

const { Title, Text, Paragraph } = Typography
const { Option } = Select

interface Feature {
  key: string
  name: string
  category: 'user' | 'item' | 'context' | 'interaction'
  type: 'realtime' | 'batch' | 'static'
  value: number | string
  importance: number
  updateFrequency: string
  status: 'active' | 'computing' | 'cached'
}

interface FeatureWindow {
  windowSize: number
  aggregationType: 'sum' | 'avg' | 'max' | 'min' | 'count'
  updateInterval: number
}

const PersonalizationFeaturePage: React.FC = () => {
  const [features, setFeatures] = useState<Feature[]>([
    {
      key: '1',
      name: '点击率CTR',
      category: 'interaction',
      type: 'realtime',
      value: 0.125,
      importance: 0.95,
      updateFrequency: '实时',
      status: 'active'
    },
    {
      key: '2',
      name: '用户活跃度',
      category: 'user',
      type: 'realtime',
      value: 0.78,
      importance: 0.88,
      updateFrequency: '5秒',
      status: 'active'
    },
    {
      key: '3',
      name: '内容新鲜度',
      category: 'item',
      type: 'batch',
      value: 0.92,
      importance: 0.72,
      updateFrequency: '1小时',
      status: 'cached'
    },
    {
      key: '4',
      name: '时段偏好',
      category: 'context',
      type: 'static',
      value: 'evening',
      importance: 0.65,
      updateFrequency: '24小时',
      status: 'cached'
    },
    {
      key: '5',
      name: '兴趣相似度',
      category: 'user',
      type: 'batch',
      value: 0.83,
      importance: 0.91,
      updateFrequency: '30分钟',
      status: 'computing'
    },
    {
      key: '6',
      name: '会话时长',
      category: 'interaction',
      type: 'realtime',
      value: 245,
      importance: 0.76,
      updateFrequency: '实时',
      status: 'active'
    }
  ])

  const [windowConfig, setWindowConfig] = useState<FeatureWindow>({
    windowSize: 300,
    aggregationType: 'avg',
    updateInterval: 5
  })

  const [autoCompute, setAutoCompute] = useState(true)
  const [featureVersion, setFeatureVersion] = useState('v2.3.1')

  // 模拟特征重要性热力图数据
  const [heatmapData, setHeatmapData] = useState<any[]>([])

  useEffect(() => {
    // 生成热力图数据
    const categories = ['user', 'item', 'context', 'interaction']
    const times = ['00:00', '06:00', '12:00', '18:00', '24:00']
    const data: any[] = []
    
    categories.forEach(cat => {
      times.forEach(time => {
        data.push({
          category: cat,
          time: time,
          value: Math.random() * 100
        })
      })
    })
    
    setHeatmapData(data)
  }, [])

  // 模拟实时特征更新
  useEffect(() => {
    if (!autoCompute) return

    const interval = setInterval(() => {
      setFeatures(prev => prev.map(f => {
        if (f.type === 'realtime') {
          return {
            ...f,
            value: typeof f.value === 'number' 
              ? Math.max(0, Math.min(1, f.value + (Math.random() - 0.5) * 0.1))
              : f.value,
            status: 'active' as const
          }
        }
        return f
      }))
    }, windowConfig.updateInterval * 1000)

    return () => clearInterval(interval)
  }, [autoCompute, windowConfig.updateInterval])

  const columns: ColumnsType<Feature> = [
    {
      title: '特征名称',
      dataIndex: 'name',
      key: 'name',
      render: (text, record) => (
        <Space>
          {getCategoryIcon(record.category)}
          <Text strong>{text}</Text>
        </Space>
      )
    },
    {
      title: '类别',
      dataIndex: 'category',
      key: 'category',
      render: (category) => {
        const colors = {
          user: 'blue',
          item: 'green',
          context: 'orange',
          interaction: 'purple'
        }
        return <Tag color={colors[category]}>{category.toUpperCase()}</Tag>
      }
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type) => {
        const config = {
          realtime: { color: 'success', text: '实时' },
          batch: { color: 'processing', text: '批量' },
          static: { color: 'default', text: '静态' }
        }
        return <Tag color={config[type].color}>{config[type].text}</Tag>
      }
    },
    {
      title: '特征值',
      dataIndex: 'value',
      key: 'value',
      render: (value) => {
        if (typeof value === 'number') {
          return <Progress percent={value * 100} size="small" format={(v) => `${v?.toFixed(1)}%`} />
        }
        return <Text>{value}</Text>
      }
    },
    {
      title: '重要性',
      dataIndex: 'importance',
      key: 'importance',
      render: (importance) => (
        <Progress 
          percent={importance * 100} 
          size="small" 
          strokeColor={{
            '0%': '#ff4d4f',
            '50%': '#faad14',
            '100%': '#52c41a',
          }}
        />
      )
    },
    {
      title: '更新频率',
      dataIndex: 'updateFrequency',
      key: 'updateFrequency',
      render: (freq) => <Text type="secondary">{freq}</Text>
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => {
        const config = {
          active: { color: 'success', text: '活跃' },
          computing: { color: 'processing', text: '计算中' },
          cached: { color: 'default', text: '已缓存' }
        }
        return <Tag color={config[status].color}>{config[status].text}</Tag>
      }
    }
  ]

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'user': return <UserOutlined style={{ color: '#1890ff' }} />
      case 'item': return <DatabaseOutlined style={{ color: '#52c41a' }} />
      case 'context': return <FieldTimeOutlined style={{ color: '#fa8c16' }} />
      case 'interaction': return <BranchesOutlined style={{ color: '#722ed1' }} />
      default: return null
    }
  }

  // 时间序列数据
  const timeSeriesData = Array.from({ length: 24 }, (_, i) => ({
    hour: `${i}:00`,
    features: Math.floor(50 + Math.random() * 50),
    computed: Math.floor(40 + Math.random() * 40),
    cached: Math.floor(60 + Math.random() * 30)
  }))

  const lineConfig = {
    data: timeSeriesData.flatMap(d => [
      { hour: d.hour, value: d.features, type: '总特征数' },
      { hour: d.hour, value: d.computed, type: '计算特征' },
      { hour: d.hour, value: d.cached, type: '缓存特征' }
    ]),
    xField: 'hour',
    yField: 'value',
    seriesField: 'type',
    smooth: true,
    yAxis: {
      title: { text: '特征数量' }
    }
  }

  const heatmapConfig = {
    data: heatmapData,
    xField: 'time',
    yField: 'category',
    colorField: 'value',
    color: ['#c6dbef', '#6baed6', '#2171b5', '#08306b'],
    label: {
      style: {
        fill: '#fff',
        shadowBlur: 2,
        shadowColor: 'rgba(0, 0, 0, .45)',
      }
    }
  }

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <FunctionOutlined /> 实时特征工程
      </Title>
      <Paragraph type="secondary">
        管理和监控个性化引擎的特征计算、聚合和版本控制
      </Paragraph>

      <Alert
        message="特征计算状态"
        description={`当前版本: ${featureVersion} | 活跃特征: ${features.filter(f => f.status === 'active').length} | 缓存命中率: 82.5%`}
        variant="default"
        showIcon
        icon={<InfoCircleOutlined />}
        style={{ marginBottom: 24 }}
      />

      {/* 控制面板 */}
      <Card title="特征计算配置" style={{ marginBottom: 24 }}>
        <Row gutter={[16, 16]}>
          <Col span={6}>
            <Text>滑动窗口大小</Text>
            <Slider
              min={60}
              max={3600}
              value={windowConfig.windowSize}
              onChange={(value) => setWindowConfig(prev => ({ ...prev, windowSize: value }))}
              marks={{
                60: '1分钟',
                300: '5分钟',
                1800: '30分钟',
                3600: '1小时'
              }}
            />
          </Col>
          <Col span={6}>
            <Text>聚合类型</Text>
            <Select
              style={{ width: '100%', marginTop: 8 }}
              value={windowConfig.aggregationType}
              onChange={(value) => setWindowConfig(prev => ({ ...prev, aggregationType: value }))}
            >
              <Option value="sum">求和</Option>
              <Option value="avg">平均</Option>
              <Option value="max">最大值</Option>
              <Option value="min">最小值</Option>
              <Option value="count">计数</Option>
            </Select>
          </Col>
          <Col span={6}>
            <Text>更新间隔（秒）</Text>
            <Slider
              min={1}
              max={60}
              value={windowConfig.updateInterval}
              onChange={(value) => setWindowConfig(prev => ({ ...prev, updateInterval: value }))}
            />
          </Col>
          <Col span={6}>
            <Space direction="vertical">
              <Text>自动计算</Text>
              <Switch 
                checked={autoCompute} 
                onChange={setAutoCompute}
                checkedChildren="开启"
                unCheckedChildren="关闭"
              />
            </Space>
          </Col>
        </Row>
      </Card>

      {/* 特征统计 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="总特征数"
              value={features.length}
              prefix={<DatabaseOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="实时特征"
              value={features.filter(f => f.type === 'realtime').length}
              prefix={<ClockCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均重要性"
              value={(features.reduce((acc, f) => acc + f.importance, 0) / features.length * 100).toFixed(1)}
              suffix="%"
              prefix={<ExperimentOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="计算中"
              value={features.filter(f => f.status === 'computing').length}
              prefix={<SyncOutlined spin />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 特征表格 */}
      <Card title="特征详情" style={{ marginBottom: 24 }}>
        <Table 
          columns={columns} 
          dataSource={features}
          pagination={false}
          size="middle"
        />
      </Card>

      {/* 特征趋势图 */}
      <Row gutter={[16, 16]}>
        <Col span={12}>
          <Card title="特征计算趋势">
            <Line {...lineConfig} height={300} />
          </Card>
        </Col>
        <Col span={12}>
          <Card title="特征重要性热力图">
            <Heatmap {...heatmapConfig} height={300} />
          </Card>
        </Col>
      </Row>

      {/* 版本历史 */}
      <Card title="特征版本历史" style={{ marginTop: 24 }}>
        <Timeline>
          <Timeline.Item color="green" dot={<CheckCircleOutlined />}>
            <Space direction="vertical" size="small">
              <Text strong>v2.3.1 (当前版本)</Text>
              <Text type="secondary">2024-01-15 10:30</Text>
              <Text>新增用户长期兴趣特征，优化实时计算性能</Text>
            </Space>
          </Timeline.Item>
          <Timeline.Item color="blue">
            <Space direction="vertical" size="small">
              <Text strong>v2.3.0</Text>
              <Text type="secondary">2024-01-10 14:20</Text>
              <Text>引入上下文感知特征，支持多模态输入</Text>
            </Space>
          </Timeline.Item>
          <Timeline.Item>
            <Space direction="vertical" size="small">
              <Text strong>v2.2.5</Text>
              <Text type="secondary">2024-01-05 09:15</Text>
              <Text>修复特征缓存失效问题，提升缓存命中率</Text>
            </Space>
          </Timeline.Item>
        </Timeline>
      </Card>
    </div>
  )
}

export default PersonalizationFeaturePage