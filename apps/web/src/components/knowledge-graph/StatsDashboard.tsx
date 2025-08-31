/**
 * 知识图谱统计仪表板
 * 
 * 功能包括：
 * - 显示图谱规模统计(实体数量、关系数量、密度等)
 * - 提供实体类型分布和关系类型分布的可视化
 * - 实现图谱质量指标监控(完整性、一致性分数)
 * - 支持图谱增长趋势和更新活跃度统计
 * - 实时数据更新和刷新功能
 */

import React, { useState, useEffect, useCallback } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Progress,
  Typography,
  Space,
  Tooltip,
  Button,
  Select,
  DatePicker,
  Tag,
  Spin,
  Alert
} from 'antd';
import {
  BarChartOutlined,
  PieChartOutlined,
  LineChartOutlined,
  ReloadOutlined,
  InfoCircleOutlined,
  TrophyOutlined,
  RiseOutlined,
  FallOutlined,
  ClockCircleOutlined
} from '@ant-design/icons';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  Area,
  AreaChart
} from 'recharts';
import dayjs from 'dayjs';

const { Title, Text } = Typography;
const { RangePicker } = DatePicker;
const { Option } = Select;

// ==================== 类型定义 ====================

export interface GraphStats {
  summary: {
    nodeCount: number;
    edgeCount: number;
    density: number;
    averageDegree: number;
    components: number;
    diameter?: number;
    clustering?: number;
  };
  distributions: {
    nodeTypeDistribution: Array<{ type: string; count: number; percentage: number }>;
    edgeTypeDistribution: Array<{ type: string; count: number; percentage: number }>;
    degreeDistribution: Array<{ degree: number; count: number }>;
  };
  quality: {
    completenessScore: number;
    consistencyScore: number;
    freshnessScore: number;
    dataIntegrity: number;
  };
  growth: {
    dailyGrowth: Array<{ date: string; nodes: number; edges: number; activity: number }>;
    weeklyActive: number;
    monthlyActive: number;
    trends: {
      nodeGrowthRate: number;
      edgeGrowthRate: number;
      activityTrend: 'rising' | 'falling' | 'stable';
    };
  };
  performance: {
    lastUpdateTime: string;
    indexingStatus: 'healthy' | 'warning' | 'error';
    queryResponseTime: number;
    throughput: number;
  };
}

interface StatsDashboardProps {
  stats?: GraphStats;
  loading?: boolean;
  error?: string;
  onRefresh?: () => void;
  onDateRangeChange?: (dateRange: [string, string]) => void;
  className?: string;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

// ==================== 颜色配置 ====================

const COLORS = [
  '#1890ff', '#52c41a', '#faad14', '#f5222d', '#722ed1', 
  '#fa8c16', '#a0d911', '#13c2c2', '#eb2f96', '#1890ff'
];

const QUALITY_COLORS = {
  excellent: '#52c41a',
  good: '#a0d911', 
  warning: '#faad14',
  poor: '#f5222d'
};

// ==================== 主组件 ====================

const StatsDashboard: React.FC<StatsDashboardProps> = ({
  stats,
  loading = false,
  error,
  onRefresh,
  onDateRangeChange,
  className = '',
  autoRefresh = false,
  refreshInterval = 30000
}) => {
  // ==================== 状态管理 ====================
  
  const [selectedTimeRange, setSelectedTimeRange] = useState<string>('7d');
  const [refreshTimer, setRefreshTimer] = useState<NodeJS.Timeout | null>(null);
  const [lastRefreshTime, setLastRefreshTime] = useState<Date>(new Date());

  // ==================== 自动刷新 ====================
  
  const handleRefresh = useCallback(() => {
    onRefresh?.();
    setLastRefreshTime(new Date());
  }, [onRefresh]);

  useEffect(() => {
    if (autoRefresh && refreshInterval > 0) {
      const timer = setInterval(handleRefresh, refreshInterval);
      setRefreshTimer(timer);
      
      return () => {
        if (timer) {
          clearInterval(timer);
        }
      };
    }
    
    return () => {
      if (refreshTimer) {
        clearInterval(refreshTimer);
        setRefreshTimer(null);
      }
    };
  }, [autoRefresh, refreshInterval, handleRefresh]);

  // ==================== 时间范围处理 ====================
  
  const handleTimeRangeChange = useCallback((range: string) => {
    setSelectedTimeRange(range);
    
    const endDate = dayjs();
    let startDate: dayjs.Dayjs;
    
    switch (range) {
      case '1d':
        startDate = endDate.subtract(1, 'day');
        break;
      case '7d':
        startDate = endDate.subtract(7, 'day');
        break;
      case '30d':
        startDate = endDate.subtract(30, 'day');
        break;
      case '90d':
        startDate = endDate.subtract(90, 'day');
        break;
      default:
        startDate = endDate.subtract(7, 'day');
    }
    
    onDateRangeChange?.([startDate.format('YYYY-MM-DD'), endDate.format('YYYY-MM-DD')]);
  }, [onDateRangeChange]);

  // ==================== 工具函数 ====================
  
  const getQualityColor = (score: number): string => {
    if (score >= 0.9) return QUALITY_COLORS.excellent;
    if (score >= 0.7) return QUALITY_COLORS.good;
    if (score >= 0.5) return QUALITY_COLORS.warning;
    return QUALITY_COLORS.poor;
  };

  const getQualityLabel = (score: number): string => {
    if (score >= 0.9) return '优秀';
    if (score >= 0.7) return '良好';
    if (score >= 0.5) return '一般';
    return '较差';
  };

  const formatNumber = (num: number): string => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };

  const renderTrendIcon = (trend: string) => {
    switch (trend) {
      case 'rising':
        return <RiseOutlined style={{ color: '#52c41a' }} />;
      case 'falling':
        return <FallOutlined style={{ color: '#f5222d' }} />;
      default:
        return <ClockCircleOutlined style={{ color: '#faad14' }} />;
    }
  };

  // ==================== 渲染组件 ====================

  if (loading) {
    return (
      <Card className={className}>
        <div style={{ textAlign: 'center', padding: '50px 0' }}>
          <Spin size="large" tip="加载统计数据..." />
        </div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className={className}>
        <Alert
          message="数据加载失败"
          description={error}
          type="error"
          showIcon
          action={
            <Button size="small" onClick={handleRefresh}>
              重试
            </Button>
          }
        />
      </Card>
    );
  }

  if (!stats) {
    return (
      <Card className={className}>
        <div style={{ textAlign: 'center', padding: '50px 0' }}>
          <Text type="secondary">暂无统计数据</Text>
        </div>
      </Card>
    );
  }

  return (
    <div className={`stats-dashboard ${className}`}>
      
      {/* 头部控制栏 */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Row justify="space-between" align="middle">
          <Col>
            <Space>
              <Title level={4} style={{ margin: 0 }}>
                <BarChartOutlined /> 图谱统计仪表板
              </Title>
              <Text type="secondary" style={{ fontSize: 12 }}>
                最后更新: {lastRefreshTime.toLocaleString()}
              </Text>
            </Space>
          </Col>
          
          <Col>
            <Space>
              <Select
                value={selectedTimeRange}
                onChange={handleTimeRangeChange}
                size="small"
                style={{ width: 80 }}
              >
                <Option value="1d">1天</Option>
                <Option value="7d">7天</Option>
                <Option value="30d">30天</Option>
                <Option value="90d">90天</Option>
              </Select>
              
              <Tooltip title="刷新数据">
                <Button 
                  icon={<ReloadOutlined />} 
                  size="small" 
                  onClick={handleRefresh}
                />
              </Tooltip>
            </Space>
          </Col>
        </Row>
      </Card>

      <Row gutter={[16, 16]}>
        
        {/* 图谱概览统计 */}
        <Col span={24}>
          <Card 
            title={
              <Space>
                <InfoCircleOutlined />
                <Text strong>图谱概览</Text>
              </Space>
            }
            size="small"
          >
            <Row gutter={16}>
              <Col span={4}>
                <Statistic
                  title="实体数量"
                  value={stats.summary.nodeCount}
                  formatter={formatNumber}
                  prefix={<div style={{ color: '#1890ff', fontSize: 16 }}>●</div>}
                />
              </Col>
              
              <Col span={4}>
                <Statistic
                  title="关系数量"
                  value={stats.summary.edgeCount}
                  formatter={formatNumber}
                  prefix={<div style={{ color: '#52c41a', fontSize: 16 }}>●</div>}
                />
              </Col>
              
              <Col span={4}>
                <Statistic
                  title="图密度"
                  value={stats.summary.density}
                  precision={4}
                  suffix="%"
                  prefix={<div style={{ color: '#faad14', fontSize: 16 }}>●</div>}
                />
              </Col>
              
              <Col span={4}>
                <Statistic
                  title="平均度数"
                  value={stats.summary.averageDegree}
                  precision={2}
                  prefix={<div style={{ color: '#722ed1', fontSize: 16 }}>●</div>}
                />
              </Col>
              
              <Col span={4}>
                <Statistic
                  title="连通分量"
                  value={stats.summary.components}
                  prefix={<div style={{ color: '#fa8c16', fontSize: 16 }}>●</div>}
                />
              </Col>
              
              <Col span={4}>
                <div style={{ textAlign: 'center' }}>
                  <Text strong style={{ fontSize: 14 }}>系统状态</Text>
                  <div style={{ marginTop: 4 }}>
                    <Tag 
                      color={stats.performance.indexingStatus === 'healthy' ? 'green' : 
                             stats.performance.indexingStatus === 'warning' ? 'orange' : 'red'}
                    >
                      {stats.performance.indexingStatus === 'healthy' ? '健康' :
                       stats.performance.indexingStatus === 'warning' ? '警告' : '异常'}
                    </Tag>
                  </div>
                  <Text type="secondary" style={{ fontSize: 11 }}>
                    响应时间: {stats.performance.queryResponseTime}ms
                  </Text>
                </div>
              </Col>
            </Row>
          </Card>
        </Col>

        {/* 质量指标 */}
        <Col span={12}>
          <Card 
            title={
              <Space>
                <TrophyOutlined />
                <Text strong>质量指标</Text>
              </Space>
            }
            size="small"
            style={{ height: 300 }}
          >
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
              
              <div>
                <Row justify="space-between" align="middle">
                  <Col><Text strong>完整性</Text></Col>
                  <Col>
                    <Tag color={getQualityColor(stats.quality.completenessScore)}>
                      {getQualityLabel(stats.quality.completenessScore)}
                    </Tag>
                  </Col>
                </Row>
                <Progress 
                  percent={Math.round(stats.quality.completenessScore * 100)} 
                  strokeColor={getQualityColor(stats.quality.completenessScore)}
                  size="small"
                />
              </div>
              
              <div>
                <Row justify="space-between" align="middle">
                  <Col><Text strong>一致性</Text></Col>
                  <Col>
                    <Tag color={getQualityColor(stats.quality.consistencyScore)}>
                      {getQualityLabel(stats.quality.consistencyScore)}
                    </Tag>
                  </Col>
                </Row>
                <Progress 
                  percent={Math.round(stats.quality.consistencyScore * 100)} 
                  strokeColor={getQualityColor(stats.quality.consistencyScore)}
                  size="small"
                />
              </div>
              
              <div>
                <Row justify="space-between" align="middle">
                  <Col><Text strong>新鲜度</Text></Col>
                  <Col>
                    <Tag color={getQualityColor(stats.quality.freshnessScore)}>
                      {getQualityLabel(stats.quality.freshnessScore)}
                    </Tag>
                  </Col>
                </Row>
                <Progress 
                  percent={Math.round(stats.quality.freshnessScore * 100)} 
                  strokeColor={getQualityColor(stats.quality.freshnessScore)}
                  size="small"
                />
              </div>
              
              <div>
                <Row justify="space-between" align="middle">
                  <Col><Text strong>数据完整性</Text></Col>
                  <Col>
                    <Tag color={getQualityColor(stats.quality.dataIntegrity)}>
                      {getQualityLabel(stats.quality.dataIntegrity)}
                    </Tag>
                  </Col>
                </Row>
                <Progress 
                  percent={Math.round(stats.quality.dataIntegrity * 100)} 
                  strokeColor={getQualityColor(stats.quality.dataIntegrity)}
                  size="small"
                />
              </div>

            </Space>
          </Card>
        </Col>

        {/* 增长趋势指标 */}
        <Col span={12}>
          <Card 
            title={
              <Space>
                <RiseOutlined />
                <Text strong>增长趋势</Text>
                {renderTrendIcon(stats.growth.trends.activityTrend)}
              </Space>
            }
            size="small"
            style={{ height: 300 }}
          >
            <Row gutter={[16, 16]}>
              
              <Col span={12}>
                <Statistic
                  title="节点增长率"
                  value={stats.growth.trends.nodeGrowthRate}
                  precision={2}
                  suffix="%"
                  valueStyle={{ 
                    color: stats.growth.trends.nodeGrowthRate >= 0 ? '#3f8600' : '#cf1322' 
                  }}
                  prefix={
                    stats.growth.trends.nodeGrowthRate >= 0 ? 
                    <RiseOutlined /> : <FallOutlined />
                  }
                />
              </Col>
              
              <Col span={12}>
                <Statistic
                  title="关系增长率"
                  value={stats.growth.trends.edgeGrowthRate}
                  precision={2}
                  suffix="%"
                  valueStyle={{ 
                    color: stats.growth.trends.edgeGrowthRate >= 0 ? '#3f8600' : '#cf1322' 
                  }}
                  prefix={
                    stats.growth.trends.edgeGrowthRate >= 0 ? 
                    <RiseOutlined /> : <FallOutlined />
                  }
                />
              </Col>
              
              <Col span={12}>
                <Statistic
                  title="周活跃度"
                  value={stats.growth.weeklyActive}
                  formatter={formatNumber}
                />
              </Col>
              
              <Col span={12}>
                <Statistic
                  title="月活跃度"
                  value={stats.growth.monthlyActive}
                  formatter={formatNumber}
                />
              </Col>
              
              <Col span={24} style={{ textAlign: 'center', marginTop: 16 }}>
                <Tag 
                  color={
                    stats.growth.trends.activityTrend === 'rising' ? 'green' :
                    stats.growth.trends.activityTrend === 'falling' ? 'red' : 'orange'
                  }
                  style={{ fontSize: 12 }}
                >
                  活跃度趋势: {
                    stats.growth.trends.activityTrend === 'rising' ? '上升' :
                    stats.growth.trends.activityTrend === 'falling' ? '下降' : '稳定'
                  }
                </Tag>
              </Col>

            </Row>
          </Card>
        </Col>

        {/* 实体类型分布 */}
        <Col span={12}>
          <Card 
            title={
              <Space>
                <PieChartOutlined />
                <Text strong>实体类型分布</Text>
              </Space>
            }
            size="small"
            style={{ height: 350 }}
          >
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={stats.distributions.nodeTypeDistribution}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  dataKey="count"
                  nameKey="type"
                >
                  {stats.distributions.nodeTypeDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <RechartsTooltip 
                  formatter={(value: number, name: string, props: any) => [
                    `${value} (${props.payload.percentage.toFixed(1)}%)`,
                    name
                  ]}
                />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </Card>
        </Col>

        {/* 关系类型分布 */}
        <Col span={12}>
          <Card 
            title={
              <Space>
                <BarChartOutlined />
                <Text strong>关系类型分布</Text>
              </Space>
            }
            size="small"
            style={{ height: 350 }}
          >
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={stats.distributions.edgeTypeDistribution}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="type" 
                  angle={-45}
                  textAnchor="end"
                  height={60}
                  fontSize={10}
                />
                <YAxis />
                <RechartsTooltip 
                  formatter={(value: number) => [formatNumber(value), '数量']}
                />
                <Bar dataKey="count" fill="#1890ff" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </Col>

        {/* 活跃度趋势图 */}
        <Col span={24}>
          <Card 
            title={
              <Space>
                <LineChartOutlined />
                <Text strong>活跃度趋势 ({selectedTimeRange})</Text>
              </Space>
            }
            size="small"
          >
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={stats.growth.dailyGrowth}>
                <defs>
                  <linearGradient id="colorNodes" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#1890ff" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#1890ff" stopOpacity={0.1}/>
                  </linearGradient>
                  <linearGradient id="colorEdges" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#52c41a" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#52c41a" stopOpacity={0.1}/>
                  </linearGradient>
                  <linearGradient id="colorActivity" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#faad14" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#faad14" stopOpacity={0.1}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date"
                  tick={{ fontSize: 10 }}
                />
                <YAxis />
                <RechartsTooltip 
                  labelFormatter={(value) => `日期: ${value}`}
                  formatter={(value: number, name: string) => [
                    formatNumber(value),
                    name === 'nodes' ? '节点增量' : 
                    name === 'edges' ? '关系增量' : '活跃度'
                  ]}
                />
                <Legend />
                <Area
                  type="monotone"
                  dataKey="nodes"
                  stroke="#1890ff"
                  fillOpacity={1}
                  fill="url(#colorNodes)"
                  name="节点增量"
                />
                <Area
                  type="monotone"
                  dataKey="edges"
                  stroke="#52c41a"
                  fillOpacity={1}
                  fill="url(#colorEdges)"
                  name="关系增量"
                />
                <Area
                  type="monotone"
                  dataKey="activity"
                  stroke="#faad14"
                  fillOpacity={1}
                  fill="url(#colorActivity)"
                  name="活跃度"
                />
              </AreaChart>
            </ResponsiveContainer>
          </Card>
        </Col>

      </Row>
    </div>
  );
};

export default StatsDashboard;