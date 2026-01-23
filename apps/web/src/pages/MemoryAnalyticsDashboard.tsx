/**
 * 记忆分析仪表板
 * 展示记忆系统的统计分析、趋势和模式
 */
import React, { useState, useEffect } from 'react'
import { logger } from '../utils/logger'
import {
  Card,
  Row,
  Col,
  Statistic,
  Progress,
  Tag,
  Space,
  Table,
  Tabs,
  Alert,
  Timeline,
  List,
  Badge,
  Tooltip,
} from 'antd'
import {
  DatabaseOutlined,
  LineChartOutlined,
  PieChartOutlined,
  BarChartOutlined,
  HeatMapOutlined,
  ClusterOutlined,
  RiseOutlined,
  FallOutlined,
  ThunderboltOutlined,
  ClockCircleOutlined,
  BookOutlined,
} from '@ant-design/icons'
import { Line, Pie, Column, Area } from '@ant-design/charts'
import {
  MemoryAnalytics,
  MemoryPattern,
  MemoryTrend,
  MemoryType,
} from '@/types/memory'
import { memoryService } from '@/services/memoryService'

const { TabPane } = Tabs

const MemoryAnalyticsDashboard: React.FC = () => {
  const [analytics, setAnalytics] = useState<MemoryAnalytics | null>(null)
  const [patterns, setPatterns] = useState<MemoryPattern | null>(null)
  const [trends, setTrends] = useState<MemoryTrend | null>(null)
  const [graphStats, setGraphStats] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    loadAllData()
  }, [])

  const loadAllData = async () => {
    setLoading(true)
    try {
      const [analyticsData, patternsData, trendsData, graphData] =
        await Promise.all([
          memoryService.getMemoryAnalytics(),
          memoryService.getMemoryPatterns(),
          memoryService.getMemoryTrends(30),
          memoryService.getGraphStatistics(),
        ])

      setAnalytics(analyticsData)
      setPatterns(patternsData)
      setTrends(trendsData)
      setGraphStats(graphData)
    } catch (error) {
      logger.error('加载数据失败:', error)
    } finally {
      setLoading(false)
    }
  }

  // 准备记忆类型分布数据
  const getMemoryTypeData = () => {
    if (!analytics) return []
    return Object.entries(analytics.memories_by_type).map(([type, count]) => ({
      type:
        type === 'working'
          ? '工作记忆'
          : type === 'episodic'
            ? '情景记忆'
            : '语义记忆',
      value: count,
      color:
        type === 'working'
          ? '#52c41a'
          : type === 'episodic'
            ? '#1890ff'
            : '#722ed1',
    }))
  }

  // 准备趋势图数据
  const getTrendData = () => {
    if (!trends) return []
    return Object.entries(trends.daily_trends).map(([date, data]) => ({
      date,
      count: data.memory_count,
    }))
  }

  // 准备记忆状态分布数据
  const getStatusData = () => {
    if (!analytics) return []
    return Object.entries(analytics.memories_by_status).map(
      ([status, count]) => ({
        status:
          status === 'active'
            ? '活跃'
            : status === 'archived'
              ? '归档'
              : status === 'compressed'
                ? '压缩'
                : '删除',
        count,
      })
    )
  }

  // 饼图配置
  const pieConfig = {
    data: getMemoryTypeData(),
    angleField: 'value',
    colorField: 'type',
    radius: 0.8,
    label: {
      type: 'outer',
      content: '{name} {percentage}',
    },
    interactions: [{ type: 'element-active' }],
  }

  // 趋势线图配置
  const lineConfig = {
    data: getTrendData(),
    xField: 'date',
    yField: 'count',
    smooth: true,
    point: {
      size: 3,
      shape: 'circle',
    },
    tooltip: {
      showMarkers: true,
    },
    xAxis: {
      label: {
        autoRotate: true,
        formatter: (text: string) => text.split('-').slice(1).join('/'),
      },
    },
  }

  // 柱状图配置
  const columnConfig = {
    data: getStatusData(),
    xField: 'status',
    yField: 'count',
    label: {
      position: 'middle' as const,
      style: {
        fill: '#FFFFFF',
        opacity: 0.8,
      },
    },
    meta: {
      status: { alias: '状态' },
      count: { alias: '数量' },
    },
  }

  return (
    <div style={{ padding: 24 }}>
      <h1>
        <LineChartOutlined /> 记忆系统分析仪表板
      </h1>
      <p style={{ color: '#666', marginBottom: 24 }}>
        全面展示记忆系统的运行状态、使用模式和性能指标
      </p>

      {/* 核心指标卡片 */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="总记忆数"
              value={analytics?.total_memories || 0}
              prefix={<DatabaseOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
            <Progress
              percent={(analytics?.total_memories || 0) / 100}
              showInfo={false}
              strokeColor="#1890ff"
            />
          </Card>
        </Col>

        <Col span={6}>
          <Card>
            <Statistic
              title="平均重要性"
              value={(analytics?.avg_importance || 0) * 100}
              precision={1}
              suffix="%"
              prefix={<ThunderboltOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
            <Progress
              percent={(analytics?.avg_importance || 0) * 100}
              showInfo={false}
              strokeColor="#52c41a"
              status="active"
            />
          </Card>
        </Col>

        <Col span={6}>
          <Card>
            <Statistic
              title="记忆增长率"
              value={trends?.summary.growth_rate || 0}
              precision={2}
              suffix="条/天"
              prefix={<RiseOutlined />}
              valueStyle={{
                color: '#cf1322',
              }}
            />
            <div style={{ fontSize: 12, color: '#666', marginTop: 8 }}>
              日均新增: {trends?.summary.avg_daily_creation || 0}
            </div>
          </Card>
        </Col>

        <Col span={6}>
          <Card>
            <Statistic
              title="存储使用"
              value={analytics?.storage_usage_mb || 0}
              precision={2}
              suffix="MB"
              prefix={<HeatMapOutlined />}
            />
            <Progress
              percent={(analytics?.storage_usage_mb || 0) / 100}
              showInfo={false}
              strokeColor={{
                '0%': '#108ee9',
                '100%': '#87d068',
              }}
            />
          </Card>
        </Col>
      </Row>

      {/* 详细分析标签页 */}
      <Tabs defaultActiveKey="distribution">
        <TabPane tab="记忆分布" key="distribution">
          <Row gutter={16}>
            <Col span={12}>
              <Card title="记忆类型分布">
                {getMemoryTypeData().length > 0 && <Pie {...pieConfig} />}
              </Card>
            </Col>
            <Col span={12}>
              <Card title="记忆状态统计">
                {getStatusData().length > 0 && <Column {...columnConfig} />}
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="使用趋势" key="trends">
          <Card title="30天记忆增长趋势">
            {getTrendData().length > 0 && <Line {...lineConfig} />}
            <Row gutter={16} style={{ marginTop: 24 }}>
              <Col span={8}>
                <Statistic
                  title="30天总计"
                  value={trends?.summary.total_memories || 0}
                  prefix={<DatabaseOutlined />}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="日均增长"
                  value={trends?.summary.avg_daily_creation || 0}
                  precision={1}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="增长率"
                  value={trends?.summary.growth_rate || 0}
                  suffix="条/天"
                  precision={2}
                  valueStyle={{
                    color: '#3f8600',
                  }}
                />
              </Col>
            </Row>
          </Card>
        </TabPane>

        <TabPane tab="访问模式" key="patterns">
          <Row gutter={16}>
            <Col span={12}>
              <Card title="高频访问记忆">
                <List
                  size="small"
                  dataSource={(analytics?.most_accessed_memories || []).slice(
                    0,
                    5
                  )}
                  renderItem={item => (
                    <List.Item>
                      <div style={{ width: '100%' }}>
                        <div style={{ marginBottom: 4 }}>
                          {item.content.substring(0, 50)}...
                        </div>
                        <Space>
                          <Tag color={getTypeColor(item.type)}>
                            {getTypeLabel(item.type)}
                          </Tag>
                          <span style={{ fontSize: 12, color: '#666' }}>
                            访问: {item.access_count}次
                          </span>
                        </Space>
                      </div>
                    </List.Item>
                  )}
                />
              </Card>
            </Col>

            <Col span={12}>
              <Card title="最近创建记忆">
                <Timeline>
                  {(analytics?.recent_memories || [])
                    .slice(0, 5)
                    .map(memory => (
                      <Timeline.Item
                        key={memory.id}
                        color={getTypeColor(memory.type)}
                      >
                        <div>
                          <div style={{ marginBottom: 4 }}>
                            {memory.content.substring(0, 50)}...
                          </div>
                          <Space size={4}>
                            <Tag
                              color={getTypeColor(memory.type)}
                              style={{ fontSize: 10 }}
                            >
                              {getTypeLabel(memory.type)}
                            </Tag>
                            <span style={{ fontSize: 10, color: '#999' }}>
                              {new Date(memory.created_at).toLocaleString()}
                            </span>
                          </Space>
                        </div>
                      </Timeline.Item>
                    ))}
                </Timeline>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="网络分析" key="network">
          <Row gutter={16}>
            <Col span={8}>
              <Card>
                <Statistic
                  title="节点总数"
                  value={graphStats?.graph_overview?.total_nodes || 0}
                  prefix={<ClusterOutlined />}
                />
              </Card>
            </Col>
            <Col span={8}>
              <Card>
                <Statistic
                  title="关联边数"
                  value={graphStats?.graph_overview?.total_edges || 0}
                  prefix={<LinkOutlined />}
                />
              </Card>
            </Col>
            <Col span={8}>
              <Card>
                <Statistic
                  title="平均连接度"
                  value={graphStats?.node_statistics?.avg_connections || 0}
                  precision={2}
                />
              </Card>
            </Col>
          </Row>

          {patterns && (
            <Card title="记忆访问模式" style={{ marginTop: 16 }}>
              <Row gutter={16}>
                <Col span={12}>
                  <h4>高峰时段</h4>
                  <Space wrap>
                    {patterns.usage_patterns.peak_hours
                      .slice(0, 5)
                      .map(([hour, count]) => (
                        <Tag key={hour} color="blue">
                          {hour}:00 ({count})
                        </Tag>
                      ))}
                  </Space>
                </Col>
                <Col span={12}>
                  <h4>活跃日期</h4>
                  <Space wrap>
                    {patterns.usage_patterns.most_active_days
                      .slice(0, 5)
                      .map(([day, count]) => (
                        <Tag key={day} color="purple">
                          {day} ({count})
                        </Tag>
                      ))}
                  </Space>
                </Col>
              </Row>
            </Card>
          )}
        </TabPane>
      </Tabs>
    </div>
  )
}

// 辅助函数
const getTypeColor = (type: MemoryType) => {
  switch (type) {
    case MemoryType.WORKING:
      return 'green'
    case MemoryType.EPISODIC:
      return 'blue'
    case MemoryType.SEMANTIC:
      return 'purple'
    default:
      return 'default'
  }
}

const getTypeLabel = (type: MemoryType) => {
  switch (type) {
    case MemoryType.WORKING:
      return '工作'
    case MemoryType.EPISODIC:
      return '情景'
    case MemoryType.SEMANTIC:
      return '语义'
    default:
      return '未知'
  }
}

const LinkOutlined = () => <span>🔗</span>

export default MemoryAnalyticsDashboard
