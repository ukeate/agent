import React, { useState, useEffect } from 'react';
import {
import { logger } from '../../utils/logger'
  Card,
  Row,
  Col,
  Statistic,
  Progress,
  Table,
  Alert,
  Button,
  Tag,
  Typography,
  Space,
  Select,
  DatePicker,
  Divider
} from 'antd';
import {
  LineChartOutlined,
  DashboardOutlined,
  ClockCircleOutlined,
  TrophyOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  ThunderboltOutlined
} from '@ant-design/icons';
import { Line, Column } from '@ant-design/plots';
import { pgvectorApi } from '../../services/pgvectorApi';

const { Title, Text } = Typography;
const { RangePicker } = DatePicker;
const { Option } = Select;

interface PerformanceMetrics {
  timestamp: string;
  avg_latency_ms: number;
  p95_latency_ms: number;
  p99_latency_ms: number;
  cache_hit_rate: number;
  quantization_ratio: number;
  search_count: number;
  error_rate: number;
}

interface PerformanceTarget {
  metric: string;
  current: number;
  target: number;
  status: 'achieved' | 'not_achieved' | 'warning';
  improvement: number;
  unit?: 'ms' | 'ratio' | 'count';
}

const PerformanceMonitorPanel: React.FC = () => {
  const [metrics, setMetrics] = useState<PerformanceMetrics[]>([]);
  const [targets, setTargets] = useState<PerformanceTarget[]>([]);
  const [loading, setLoading] = useState(true);
  const [timeRange, setTimeRange] = useState<'1h' | '6h' | '24h' | '7d'>('1h');
  const [autoRefresh, setAutoRefresh] = useState(false);

  useEffect(() => {
    fetchPerformanceData();
    fetchPerformanceTargets();
  }, [timeRange]);

  useEffect(() => {
    let interval: ReturnType<typeof setTimeout>;
    if (autoRefresh) {
      interval = setInterval(fetchPerformanceData, 10000); // 每10秒刷新
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh, timeRange]);

  const fetchPerformanceData = async () => {
    try {
      setLoading(true);
      const data = await pgvectorApi.getPerformanceMetrics(timeRange);
      const qp = data?.query_performance || {};
      const end = data?.report_period?.end_time || new Date().toISOString();
      setMetrics([
        {
          timestamp: end,
          avg_latency_ms: qp.average_execution_time_ms || 0,
          p95_latency_ms: qp.p95_execution_time_ms || qp.max_execution_time_ms || 0,
          p99_latency_ms: qp.p99_execution_time_ms || qp.max_execution_time_ms || 0,
          cache_hit_rate: qp.cache_hit_ratio || 0,
          quantization_ratio: qp.quantization_ratio || 0,
          search_count: qp.total_queries || 0,
          error_rate: qp.error_rate || 0,
        },
      ]);
    } catch (error) {
      logger.error('获取性能数据失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchPerformanceTargets = async () => {
    try {
      const targetData = await pgvectorApi.getPerformanceTargets();
      setTargets(targetData);
    } catch (error) {
      logger.error('获取性能目标失败:', error);
    }
  };

  const latestMetrics = metrics.length > 0 ? metrics[metrics.length - 1] : null;

  const latencyChartData = metrics.map(m => [
    { timestamp: m.timestamp, type: 'Average', latency: m.avg_latency_ms },
    { timestamp: m.timestamp, type: 'P95', latency: m.p95_latency_ms },
    { timestamp: m.timestamp, type: 'P99', latency: m.p99_latency_ms }
  ]).flat();

  const latencyConfig = {
    data: latencyChartData,
    xField: 'timestamp',
    yField: 'latency',
    seriesField: 'type',
    smooth: true,
    animation: {
      appear: {
        animation: 'path-in',
        duration: 1000,
      },
    },
    color: ['#1890ff', '#f0a60a', '#f50'],
    yAxis: {
      label: {
        formatter: (v: string) => `${v}ms`
      }
    }
  };

  const cacheHitChartData = metrics.map(m => ({
    timestamp: m.timestamp,
    hit_rate: m.cache_hit_rate * 100
  }));

  const cacheConfig = {
    data: cacheHitChartData,
    xField: 'timestamp',
    yField: 'hit_rate',
    smooth: true,
    color: '#52c41a',
    yAxis: {
      label: {
        formatter: (v: string) => `${v}%`
      },
      max: 100
    }
  };

  const getTargetStatusIcon = (status: string) => {
    switch (status) {
      case 'achieved':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'warning':
        return <WarningOutlined style={{ color: '#faad14' }} />;
      default:
        return <WarningOutlined style={{ color: '#f5222d' }} />;
    }
  };

  const getTargetStatusColor = (status: string) => {
    switch (status) {
      case 'achieved': return 'success';
      case 'warning': return 'warning';
      default: return 'error';
    }
  };

  const formatTargetValue = (value: number, record: PerformanceTarget) => {
    if (record.unit === 'ratio' || record.metric.includes('率') || record.metric.includes('命中')) {
      return `${(value * 100).toFixed(1)}%`;
    }
    if (record.unit === 'count') {
      return `${Math.round(value)}`;
    }
    return `${value.toFixed(1)}ms`;
  };

  const targetColumns = [
    {
      title: '性能指标',
      dataIndex: 'metric',
      key: 'metric'
    },
    {
      title: '当前值',
      dataIndex: 'current',
      key: 'current',
      render: (value: number, record: PerformanceTarget) => formatTargetValue(value, record)
    },
    {
      title: '目标值',
      dataIndex: 'target',
      key: 'target',
      render: (value: number, record: PerformanceTarget) => formatTargetValue(value, record)
    },
    {
      title: '改善幅度',
      dataIndex: 'improvement',
      key: 'improvement',
      render: (value: number) => (
        <span style={{ color: value > 0 ? '#52c41a' : '#f5222d' }}>
          {value > 0 ? '+' : ''}{(value * 100).toFixed(1)}%
        </span>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Space>
          {getTargetStatusIcon(status)}
          <Tag color={getTargetStatusColor(status)}>
            {status === 'achieved' ? '已达成' : status === 'warning' ? '警告' : '未达成'}
          </Tag>
        </Space>
      )
    }
  ];

  const achievedCount = targets.filter(t => t.status === 'achieved').length;
  const totalTargets = targets.length;
  const achievementRate = totalTargets ? Math.round((achievedCount / totalTargets) * 100) : 0;
  const avgTarget = targets.find(t => t.metric.includes('平均查询延迟'));

  return (
    <div>
      {/* 控制面板 */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Row justify="space-between" align="middle">
          <Col>
            <Space>
              <Text>时间范围:</Text>
              <Select value={timeRange} onChange={setTimeRange} style={{ width: 100 }}>
                <Option value="1h">1小时</Option>
                <Option value="6h">6小时</Option>
                <Option value="24h">24小时</Option>
                <Option value="7d">7天</Option>
              </Select>
              
              <Button
                type={autoRefresh ? 'primary' : 'default'}
                size="small"
                onClick={() => setAutoRefresh(!autoRefresh)}
              >
                自动刷新
              </Button>
            </Space>
          </Col>
          <Col>
            <Button onClick={fetchPerformanceData} loading={loading}>
              刷新数据
            </Button>
          </Col>
        </Row>
      </Card>

      {/* 核心指标 */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均延迟"
              value={latestMetrics?.avg_latency_ms || 0}
              precision={1}
              suffix="ms"
              prefix={<ClockCircleOutlined />}
              valueStyle={{ color: (latestMetrics?.avg_latency_ms || 0) < 50 ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
        
        <Col span={6}>
          <Card>
            <Statistic
              title="P95延迟"
              value={latestMetrics?.p95_latency_ms || 0}
              precision={1}
              suffix="ms"
              prefix={<DashboardOutlined />}
              valueStyle={{ color: (latestMetrics?.p95_latency_ms || 0) < 100 ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>

        <Col span={6}>
          <Card>
            <Statistic
              title="缓存命中率"
              value={(latestMetrics?.cache_hit_rate || 0) * 100}
              precision={1}
              suffix="%"
              prefix={<ThunderboltOutlined />}
              valueStyle={{ color: (latestMetrics?.cache_hit_rate || 0) > 0.7 ? '#3f8600' : '#faad14' }}
            />
          </Card>
        </Col>

        <Col span={6}>
          <Card>
            <Statistic
              title="量化使用率"
              value={(latestMetrics?.quantization_ratio || 0) * 100}
              precision={1}
              suffix="%"
              prefix={<LineChartOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 性能图表 */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={12}>
          <Card title="延迟趋势" extra={<ClockCircleOutlined />}>
            <Line {...latencyConfig} height={300} />
          </Card>
        </Col>
        
        <Col span={12}>
          <Card title="缓存命中率" extra={<ThunderboltOutlined />}>
            <Line {...cacheConfig} height={300} />
          </Card>
        </Col>
      </Row>

      {/* 性能目标达成情况 */}
      <Card title="性能目标达成情况" extra={<TrophyOutlined />}>
        <Table
          dataSource={targets.map((target, index) => ({ ...target, key: index }))}
          columns={targetColumns}
          size="small"
          pagination={false}
        />
        
        <Divider />
        
        {/* 目标达成概览 */}
        <Row gutter={16}>
          <Col span={8}>
            <Progress
              type="circle"
              percent={achievementRate}
              format={() => `${achievedCount}/${totalTargets || 0}`}
              strokeColor="#52c41a"
            />
            <div style={{ textAlign: 'center', marginTop: 8 }}>
              <Text>目标达成率</Text>
            </div>
          </Col>
          
          <Col span={16}>
            <Alert
              message="性能优化建议"
              description={
                <ul>
                  <li>当前平均延迟: {latestMetrics?.avg_latency_ms?.toFixed(1)}ms{avgTarget ? `，目标: <${avgTarget.target.toFixed(1)}ms` : ''}</li>
                  <li>缓存命中率: {((latestMetrics?.cache_hit_rate || 0) * 100).toFixed(1)}%</li>
                  <li>量化使用率: {((latestMetrics?.quantization_ratio || 0) * 100).toFixed(1)}%</li>
                  <li>建议启用INT8量化以平衡性能和精度</li>
                </ul>
              }
              variant="default"
              showIcon
            />
          </Col>
        </Row>
      </Card>
    </div>
  );
};

export default PerformanceMonitorPanel;
