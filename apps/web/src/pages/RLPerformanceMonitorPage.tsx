import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Select, DatePicker, Space, Button, Table, Progress, Statistic, Alert } from 'antd';
import { Line, Area, Bar, Gauge } from '@ant-design/plots';
import { 
  LineChartOutlined, 
  BarChartOutlined, 
  DashboardOutlined,
  ThunderboltOutlined,
  ClockCircleOutlined,
  ApiOutlined,
  FireOutlined
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import { monitoringService } from '../services/monitoringService';

const { RangePicker } = DatePicker;
const { Option } = Select;

interface PerformanceData {
  timestamp: string;
  qps: number;
  latency: number;
  errorRate: number;
  cacheHit: number;
  cpuUsage: number;
  memoryUsage: number;
}

interface AlgorithmMetrics {
  algorithm: string;
  avgLatency: number;
  p95Latency: number;
  p99Latency: number;
  qps: number;
  errorRate: number;
  cacheHitRate: number;
}

const RLPerformanceMonitorPage: React.FC = () => {
  const [timeRange, setTimeRange] = useState('1h');
  const [performanceData, setPerformanceData] = useState<PerformanceData[]>([]);
  const [algorithmMetrics, setAlgorithmMetrics] = useState<AlgorithmMetrics[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      try {
        const dashboard = await monitoringService.getDashboardData();
        const metrics = dashboard.metrics || {};
        const timeseries = metrics.qps?.points || [];
        const mapped: PerformanceData[] = timeseries.map((p: any) => ({
          timestamp: p.timestamp || '',
          qps: p.value || 0,
          latency: metrics.api_response_time?.current_value || 0,
          errorRate: metrics.error_rate?.current_value || 0,
          cacheHit: metrics.cache_hit_rate?.current_value || 0,
          cpuUsage: metrics.cpu_usage?.current_value || 0,
          memoryUsage: metrics.memory_usage?.current_value || 0
        }));
        setPerformanceData(mapped);
        setAlgorithmMetrics([]);
      } catch (e) {
        setPerformanceData([]);
        setAlgorithmMetrics([]);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [timeRange]);

  // QPS趋势图配置
  const qpsConfig = {
    data: performanceData,
    xField: 'timestamp',
    yField: 'qps',
    smooth: true,
    color: '#1890ff',
    point: {
      size: 3,
      shape: 'circle',
    },
    tooltip: {
      formatter: (data: PerformanceData) => ({
        name: 'QPS',
        value: `${data.qps} req/s`
      }),
    },
    annotations: [
      {
        type: 'line',
        start: ['min', 1000],
        end: ['max', 1000],
        style: {
          stroke: '#FF4D4F',
          lineDash: [4, 4],
        },
      },
    ],
  };

  // 延迟分布图配置
  const latencyConfig = {
    data: performanceData,
    xField: 'timestamp',
    yField: 'latency',
    smooth: true,
    color: '#52c41a',
    area: {
      style: {
        fill: 'l(270) 0:#52c41a 0.5:#52c41a 1:#ffffff',
      },
    },
    tooltip: {
      formatter: (data: PerformanceData) => ({
        name: '延迟',
        value: `${data.latency} ms`
      }),
    },
  };

  // 系统资源使用图配置
  const resourceConfig = {
    data: performanceData.map(d => [
      { timestamp: d.timestamp, type: 'CPU使用率', value: d.cpuUsage },
      { timestamp: d.timestamp, type: '内存使用率', value: d.memoryUsage }
    ]).flat(),
    xField: 'timestamp',
    yField: 'value',
    seriesField: 'type',
    color: ['#faad14', '#722ed1'],
    smooth: true,
    legend: {
      position: 'top',
    },
  };

  // 算法性能对比图配置
  const algorithmCompareConfig = {
    data: algorithmMetrics.map(m => [
      { algorithm: m.algorithm, metric: '平均延迟', value: m.avgLatency },
      { algorithm: m.algorithm, metric: 'P95延迟', value: m.p95Latency },
      { algorithm: m.algorithm, metric: 'P99延迟', value: m.p99Latency }
    ]).flat(),
    xField: 'algorithm',
    yField: 'value',
    seriesField: 'metric',
    color: ['#1890ff', '#faad14', '#ff4d4f'],
    legend: {
      position: 'top',
    },
    label: {
      position: 'top',
      formatter: (data: any) => `${data.value}ms`,
    },
  };

  const algorithmColumns: ColumnsType<AlgorithmMetrics> = [
    {
      title: '算法',
      dataIndex: 'algorithm',
      key: 'algorithm',
      render: (text) => <strong>{text}</strong>
    },
    {
      title: '平均延迟',
      dataIndex: 'avgLatency',
      key: 'avgLatency',
      render: (value) => `${value.toFixed(1)}ms`,
      sorter: (a, b) => a.avgLatency - b.avgLatency
    },
    {
      title: 'P95延迟',
      dataIndex: 'p95Latency',
      key: 'p95Latency',
      render: (value) => `${value.toFixed(1)}ms`,
      sorter: (a, b) => a.p95Latency - b.p95Latency
    },
    {
      title: 'P99延迟',
      dataIndex: 'p99Latency',
      key: 'p99Latency',
      render: (value) => `${value.toFixed(1)}ms`,
      sorter: (a, b) => a.p99Latency - b.p99Latency
    },
    {
      title: 'QPS',
      dataIndex: 'qps',
      key: 'qps',
      render: (value) => `${value.toFixed(1)} req/s`,
      sorter: (a, b) => a.qps - b.qps
    },
    {
      title: '错误率',
      dataIndex: 'errorRate',
      key: 'errorRate',
      render: (value) => (
        <Progress
          percent={value * 100}
          size="small"
          status={value < 0.01 ? 'success' : value < 0.05 ? 'normal' : 'exception'}
          format={() => `${(value * 100).toFixed(2)}%`}
        />
      ),
      sorter: (a, b) => a.errorRate - b.errorRate
    },
    {
      title: '缓存命中率',
      dataIndex: 'cacheHitRate',
      key: 'cacheHitRate',
      render: (value) => (
        <Progress
          percent={value}
          size="small"
          status={value > 90 ? 'success' : 'normal'}
        />
      ),
      sorter: (a, b) => a.cacheHitRate - b.cacheHitRate
    }
  ];

  const currentPerf = performanceData[performanceData.length - 1] || {
    qps: 0, latency: 0, errorRate: 0, cacheHit: 0, cpuUsage: 0, memoryUsage: 0
  };

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <h1 style={{ margin: 0, display: 'flex', alignItems: 'center' }}>
          <LineChartOutlined style={{ marginRight: '8px' }} />
          强化学习性能监控
        </h1>
        <Space>
          <Select value={timeRange} onChange={setTimeRange} style={{ width: 120 }}>
            <Option value="1h">最近1小时</Option>
            <Option value="6h">最近6小时</Option>
            <Option value="24h">最近24小时</Option>
          </Select>
          <RangePicker showTime />
          <Button 
            type="primary" 
            icon={<ThunderboltOutlined />}
            loading={loading}
            onClick={() => {
              setPerformanceData(generatePerformanceData());
              setAlgorithmMetrics(generateAlgorithmMetrics());
            }}
          >
            刷新
          </Button>
        </Space>
      </div>

      {/* 性能状态警告 */}
      {currentPerf.latency > 50 && (
        <Alert
          message="性能警告"
          description={`当前平均延迟为 ${currentPerf.latency.toFixed(1)}ms，超过阈值50ms`}
          variant="warning"
          showIcon
          closable
          style={{ marginBottom: '24px' }}
        />
      )}

      {/* 实时性能指标 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={8} md={4}>
          <Card>
            <Statistic
              title="当前QPS"
              value={currentPerf.qps}
              prefix={<ApiOutlined />}
              suffix="req/s"
              valueStyle={{ color: currentPerf.qps > 1000 ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8} md={4}>
          <Card>
            <Statistic
              title="当前延迟"
              value={currentPerf.latency}
              prefix={<ClockCircleOutlined />}
              suffix="ms"
              precision={1}
              valueStyle={{ color: currentPerf.latency < 50 ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8} md={4}>
          <Card>
            <Statistic
              title="错误率"
              value={currentPerf.errorRate}
              suffix="%"
              precision={2}
              valueStyle={{ color: currentPerf.errorRate < 1 ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8} md={4}>
          <Card>
            <Statistic
              title="缓存命中"
              value={currentPerf.cacheHit}
              prefix={<FireOutlined />}
              suffix="%"
              precision={1}
              valueStyle={{ color: currentPerf.cacheHit > 90 ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8} md={4}>
          <Card>
            <Statistic
              title="CPU使用"
              value={currentPerf.cpuUsage}
              suffix="%"
              precision={1}
              valueStyle={{ color: currentPerf.cpuUsage < 70 ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={8} md={4}>
          <Card>
            <Statistic
              title="内存使用"
              value={currentPerf.memoryUsage}
              suffix="%"
              precision={1}
              valueStyle={{ color: currentPerf.memoryUsage < 80 ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
      </Row>

      {/* QPS趋势图 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} lg={12}>
          <Card title="QPS趋势" loading={loading}>
            <Line {...qpsConfig} height={300} />
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="延迟分布" loading={loading}>
            <Area {...latencyConfig} height={300} />
          </Card>
        </Col>
      </Row>

      {/* 系统资源使用 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} lg={12}>
          <Card title="系统资源使用" loading={loading}>
            <Line {...resourceConfig} height={300} />
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="算法延迟对比" loading={loading}>
            <Bar {...algorithmCompareConfig} height={300} />
          </Card>
        </Col>
      </Row>

      {/* 算法性能详细表格 */}
      <Card title="算法性能详细指标" loading={loading}>
        <Table
          dataSource={algorithmMetrics}
          columns={algorithmColumns}
          rowKey="algorithm"
          pagination={false}
          size="middle"
        />
      </Card>
    </div>
  );
};

export default RLPerformanceMonitorPage;
