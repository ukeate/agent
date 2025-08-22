import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Select, DatePicker, Space, Button, Table, Statistic, Alert, Tabs, Progress, Tag } from 'antd';
import { Line, Area, Bar, Pie, Heatmap, Scatter } from '@ant-design/plots';
import { 
  BarChartOutlined, 
  LineChartOutlined, 
  PieChartOutlined,
  HeatMapOutlined,
  ThunderboltOutlined,
  DownloadOutlined,
  FilterOutlined,
  TrophyOutlined,
  RiseOutlined,
  FallOutlined
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';

const { RangePicker } = DatePicker;
const { Option } = Select;
const { TabPane } = Tabs;

interface MetricData {
  timestamp: string;
  algorithm: string;
  metric: string;
  value: number;
}

interface AlgorithmComparison {
  algorithm: string;
  avgReward: number;
  conversionRate: number;
  ctr: number;
  userSatisfaction: number;
  performance: number;
  trend: 'up' | 'down' | 'stable';
}

interface UserSegmentMetrics {
  segment: string;
  users: number;
  avgReward: number;
  engagement: number;
  retention: number;
}

interface PerformanceMetrics {
  metric: string;
  current: number;
  target: number;
  trend: number;
  status: 'good' | 'warning' | 'critical';
}

const RLMetricsAnalysisPage: React.FC = () => {
  const [timeRange, setTimeRange] = useState('24h');
  const [selectedAlgorithms, setSelectedAlgorithms] = useState(['all']);
  const [metricData, setMetricData] = useState<MetricData[]>([]);
  const [algorithmComparison, setAlgorithmComparison] = useState<AlgorithmComparison[]>([]);
  const [userSegments, setUserSegments] = useState<UserSegmentMetrics[]>([]);
  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics[]>([]);
  const [loading, setLoading] = useState(false);

  // 生成模拟数据
  const generateMetricData = () => {
    const algorithms = ['UCB', 'Thompson Sampling', 'Epsilon Greedy', 'Q-Learning'];
    const metrics = ['reward', 'ctr', 'latency', 'accuracy'];
    const data: MetricData[] = [];
    
    const hours = timeRange === '1h' ? 1 : timeRange === '24h' ? 24 : timeRange === '7d' ? 168 : 720;
    const interval = timeRange === '1h' ? 5 : timeRange === '24h' ? 60 : timeRange === '7d' ? 360 : 1440; // 分钟

    for (let i = 0; i < hours; i++) {
      const timestamp = new Date(Date.now() - (hours - i) * interval * 60000);
      
      algorithms.forEach(algorithm => {
        metrics.forEach(metric => {
          let baseValue = 0;
          let variance = 0;
          
          switch (metric) {
            case 'reward':
              baseValue = algorithm === 'Q-Learning' ? 0.75 : 
                         algorithm === 'UCB' ? 0.72 :
                         algorithm === 'Thompson Sampling' ? 0.70 : 0.68;
              variance = 0.1;
              break;
            case 'ctr':
              baseValue = algorithm === 'UCB' ? 0.045 :
                         algorithm === 'Thompson Sampling' ? 0.042 :
                         algorithm === 'Q-Learning' ? 0.041 : 0.038;
              variance = 0.005;
              break;
            case 'latency':
              baseValue = algorithm === 'Epsilon Greedy' ? 8 :
                         algorithm === 'UCB' ? 12 :
                         algorithm === 'Thompson Sampling' ? 15 : 28;
              variance = 5;
              break;
            case 'accuracy':
              baseValue = algorithm === 'Q-Learning' ? 88 :
                         algorithm === 'UCB' ? 85 :
                         algorithm === 'Thompson Sampling' ? 84 : 81;
              variance = 3;
              break;
          }
          
          data.push({
            timestamp: timestamp.toISOString(),
            algorithm,
            metric,
            value: baseValue + (Math.random() - 0.5) * variance * 2
          });
        });
      });
    }
    
    return data;
  };

  const generateAlgorithmComparison = (): AlgorithmComparison[] => [
    {
      algorithm: 'Q-Learning',
      avgReward: 0.751,
      conversionRate: 0.087,
      ctr: 0.041,
      userSatisfaction: 4.2,
      performance: 92,
      trend: 'up'
    },
    {
      algorithm: 'UCB',
      avgReward: 0.723,
      conversionRate: 0.082,
      ctr: 0.045,
      userSatisfaction: 4.0,
      performance: 88,
      trend: 'stable'
    },
    {
      algorithm: 'Thompson Sampling',
      avgReward: 0.698,
      conversionRate: 0.075,
      ctr: 0.042,
      userSatisfaction: 3.9,
      performance: 85,
      trend: 'up'
    },
    {
      algorithm: 'Epsilon Greedy',
      avgReward: 0.675,
      conversionRate: 0.068,
      ctr: 0.038,
      userSatisfaction: 3.7,
      performance: 79,
      trend: 'down'
    }
  ];

  const generateUserSegments = (): UserSegmentMetrics[] => [
    { segment: '新用户', users: 12500, avgReward: 0.65, engagement: 0.34, retention: 0.68 },
    { segment: '活跃用户', users: 8900, avgReward: 0.78, engagement: 0.72, retention: 0.89 },
    { segment: '高价值用户', users: 2100, avgReward: 0.89, engagement: 0.85, retention: 0.94 },
    { segment: '流失风险用户', users: 4200, avgReward: 0.52, engagement: 0.21, retention: 0.43 }
  ];

  const generatePerformanceMetrics = (): PerformanceMetrics[] => [
    { metric: '平均奖励', current: 0.723, target: 0.750, trend: 2.3, status: 'warning' },
    { metric: '点击率', current: 0.042, target: 0.045, trend: -1.2, status: 'warning' },
    { metric: '转化率', current: 0.081, target: 0.080, trend: 5.4, status: 'good' },
    { metric: '用户满意度', current: 4.1, target: 4.2, trend: 1.8, status: 'good' },
    { metric: '响应延迟', current: 15.8, target: 20.0, trend: -8.2, status: 'good' },
    { metric: '模型准确率', current: 0.852, target: 0.850, trend: 3.1, status: 'good' }
  ];

  useEffect(() => {
    setLoading(true);
    setTimeout(() => {
      setMetricData(generateMetricData());
      setAlgorithmComparison(generateAlgorithmComparison());
      setUserSegments(generateUserSegments());
      setPerformanceMetrics(generatePerformanceMetrics());
      setLoading(false);
    }, 1000);
  }, [timeRange, selectedAlgorithms]);

  // 奖励趋势图配置
  const rewardTrendConfig = {
    data: metricData.filter(d => d.metric === 'reward' && 
      (selectedAlgorithms.includes('all') || selectedAlgorithms.includes(d.algorithm))),
    xField: 'timestamp',
    yField: 'value',
    seriesField: 'algorithm',
    smooth: true,
    color: ['#1890ff', '#52c41a', '#faad14', '#f5222d'],
    legend: { position: 'top' },
    tooltip: {
      formatter: (data: MetricData) => ({
        name: data.algorithm,
        value: data.value.toFixed(3)
      }),
    },
  };

  // CTR对比图配置
  const ctrCompareConfig = {
    data: algorithmComparison,
    xField: 'algorithm',
    yField: 'ctr',
    color: '#52c41a',
    label: {
      position: 'top',
      formatter: (data: AlgorithmComparison) => (data.ctr * 100).toFixed(1) + '%',
    },
    meta: {
      ctr: { formatter: (value: number) => (value * 100).toFixed(2) + '%' }
    }
  };

  // 算法性能雷达图数据
  const radarData = algorithmComparison.map(item => [
    { algorithm: item.algorithm, metric: '平均奖励', value: item.avgReward * 100 },
    { algorithm: item.algorithm, metric: '转化率', value: item.conversionRate * 1000 },
    { algorithm: item.algorithm, metric: '点击率', value: item.ctr * 1000 },
    { algorithm: item.algorithm, metric: '用户满意度', value: item.userSatisfaction * 20 },
    { algorithm: item.algorithm, metric: '整体性能', value: item.performance }
  ]).flat();

  // 用户分段饼图配置
  const userSegmentPieConfig = {
    data: userSegments,
    angleField: 'users',
    colorField: 'segment',
    radius: 0.8,
    label: {
      type: 'outer',
      content: '{name} {percentage}',
    },
    interactions: [{ type: 'element-active' }],
  };

  // 热力图数据
  const heatmapData = [];
  const timeSlots = ['00-06', '06-12', '12-18', '18-24'];
  const days = ['周一', '周二', '周三', '周四', '周五', '周六', '周日'];
  
  days.forEach(day => {
    timeSlots.forEach(time => {
      heatmapData.push({
        day,
        time,
        value: Math.random() * 100
      });
    });
  });

  const heatmapConfig = {
    data: heatmapData,
    xField: 'time',
    yField: 'day',
    colorField: 'value',
    color: ['#174c83', '#7eb6d4', '#f1ecc1', '#e9975a', '#d94f00'],
    meta: {
      value: { min: 0, max: 100 }
    }
  };

  const algorithmColumns: ColumnsType<AlgorithmComparison> = [
    {
      title: '算法',
      dataIndex: 'algorithm',
      key: 'algorithm',
      render: (text) => <strong>{text}</strong>
    },
    {
      title: '平均奖励',
      dataIndex: 'avgReward',
      key: 'avgReward',
      render: (value) => value.toFixed(3),
      sorter: (a, b) => a.avgReward - b.avgReward
    },
    {
      title: '转化率',
      dataIndex: 'conversionRate',
      key: 'conversionRate',
      render: (value) => (value * 100).toFixed(2) + '%',
      sorter: (a, b) => a.conversionRate - b.conversionRate
    },
    {
      title: '点击率',
      dataIndex: 'ctr',
      key: 'ctr',
      render: (value) => (value * 100).toFixed(2) + '%',
      sorter: (a, b) => a.ctr - b.ctr
    },
    {
      title: '用户满意度',
      dataIndex: 'userSatisfaction',
      key: 'userSatisfaction',
      render: (value) => value.toFixed(1) + '/5.0',
      sorter: (a, b) => a.userSatisfaction - b.userSatisfaction
    },
    {
      title: '综合性能',
      dataIndex: 'performance',
      key: 'performance',
      render: (value) => (
        <div>
          <Progress percent={value} size="small" />
          <span style={{ marginLeft: '8px' }}>{value}%</span>
        </div>
      ),
      sorter: (a, b) => a.performance - b.performance
    },
    {
      title: '趋势',
      dataIndex: 'trend',
      key: 'trend',
      render: (trend) => {
        const config = {
          up: { color: 'green', icon: <RiseOutlined />, text: '上升' },
          down: { color: 'red', icon: <FallOutlined />, text: '下降' },
          stable: { color: 'blue', icon: <LineChartOutlined />, text: '稳定' }
        };
        return <Tag color={config[trend].color} icon={config[trend].icon}>{config[trend].text}</Tag>;
      }
    }
  ];

  const performanceColumns: ColumnsType<PerformanceMetrics> = [
    {
      title: '指标',
      dataIndex: 'metric',
      key: 'metric',
    },
    {
      title: '当前值',
      dataIndex: 'current',
      key: 'current',
      render: (value, record) => {
        if (record.metric.includes('率')) {
          return (value * 100).toFixed(2) + '%';
        } else if (record.metric.includes('延迟')) {
          return value.toFixed(1) + 'ms';
        } else if (record.metric.includes('满意度')) {
          return value.toFixed(1) + '/5.0';
        }
        return value.toFixed(3);
      }
    },
    {
      title: '目标值',
      dataIndex: 'target',
      key: 'target',
      render: (value, record) => {
        if (record.metric.includes('率')) {
          return (value * 100).toFixed(2) + '%';
        } else if (record.metric.includes('延迟')) {
          return value.toFixed(1) + 'ms';
        } else if (record.metric.includes('满意度')) {
          return value.toFixed(1) + '/5.0';
        }
        return value.toFixed(3);
      }
    },
    {
      title: '趋势',
      dataIndex: 'trend',
      key: 'trend',
      render: (trend) => (
        <span style={{ color: trend > 0 ? '#52c41a' : '#f5222d' }}>
          {trend > 0 ? <RiseOutlined /> : <FallOutlined />}
          {Math.abs(trend).toFixed(1)}%
        </span>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => {
        const config = {
          good: { color: 'green', text: '良好' },
          warning: { color: 'orange', text: '警告' },
          critical: { color: 'red', text: '严重' }
        };
        return <Tag color={config[status].color}>{config[status].text}</Tag>;
      }
    }
  ];

  const criticalMetrics = performanceMetrics.filter(m => m.status === 'critical').length;
  const warningMetrics = performanceMetrics.filter(m => m.status === 'warning').length;

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <h1 style={{ margin: 0, display: 'flex', alignItems: 'center' }}>
          <BarChartOutlined style={{ marginRight: '8px' }} />
          强化学习指标分析
        </h1>
        <Space>
          <Select
            value={timeRange}
            onChange={setTimeRange}
            style={{ width: 120 }}
          >
            <Option value="1h">最近1小时</Option>
            <Option value="24h">最近24小时</Option>
            <Option value="7d">最近7天</Option>
            <Option value="30d">最近30天</Option>
          </Select>
          <Select
            mode="multiple"
            value={selectedAlgorithms}
            onChange={setSelectedAlgorithms}
            style={{ width: 200 }}
            placeholder="选择算法"
          >
            <Option value="all">全部算法</Option>
            <Option value="UCB">UCB</Option>
            <Option value="Thompson Sampling">Thompson Sampling</Option>
            <Option value="Epsilon Greedy">Epsilon Greedy</Option>
            <Option value="Q-Learning">Q-Learning</Option>
          </Select>
          <RangePicker showTime />
          <Button icon={<FilterOutlined />}>高级筛选</Button>
          <Button icon={<DownloadOutlined />}>导出报告</Button>
          <Button 
            type="primary" 
            icon={<ThunderboltOutlined />}
            loading={loading}
            onClick={() => {
              setMetricData(generateMetricData());
              setAlgorithmComparison(generateAlgorithmComparison());
            }}
          >
            刷新
          </Button>
        </Space>
      </div>

      {/* 性能状态概览 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="严重指标"
              value={criticalMetrics}
              prefix={<TrophyOutlined />}
              valueStyle={{ color: criticalMetrics > 0 ? '#cf1322' : '#3f8600' }}
              suffix="个"
            />
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="警告指标"
              value={warningMetrics}
              valueStyle={{ color: warningMetrics > 0 ? '#faad14' : '#3f8600' }}
              suffix="个"
            />
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card>
            <Statistic
              title="最佳算法"
              value="Q-Learning"
              valueStyle={{ color: '#3f8600' }}
              suffix="92% 性能分"
            />
          </Card>
        </Col>
      </Row>

      {/* 关键指标警告 */}
      {(criticalMetrics > 0 || warningMetrics > 0) && (
        <Alert
          message="指标监控警告"
          description={`发现 ${criticalMetrics} 个严重指标和 ${warningMetrics} 个警告指标，建议优化算法配置`}
          type={criticalMetrics > 0 ? 'error' : 'warning'}
          showIcon
          closable
          style={{ marginBottom: '24px' }}
        />
      )}

      {/* 指标分析标签页 */}
      <Tabs defaultActiveKey="overview">
        <TabPane tab="综合概览" key="overview">
          <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
            <Col xs={24} lg={12}>
              <Card title="奖励趋势分析" loading={loading}>
                <Line {...rewardTrendConfig} height={300} />
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card title="算法点击率对比" loading={loading}>
                <Bar {...ctrCompareConfig} height={300} />
              </Card>
            </Col>
          </Row>
          
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={12}>
              <Card title="用户分段分析" loading={loading}>
                <Pie {...userSegmentPieConfig} height={300} />
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card title="性能指标状态" loading={loading}>
                <Table
                  dataSource={performanceMetrics}
                  columns={performanceColumns}
                  rowKey="metric"
                  pagination={false}
                  size="small"
                />
              </Card>
            </Col>
          </Row>
        </TabPane>
        
        <TabPane tab="算法对比" key="algorithms">
          <Card title="算法性能详细对比" loading={loading}>
            <Table
              dataSource={algorithmComparison}
              columns={algorithmColumns}
              rowKey="algorithm"
              pagination={false}
              size="middle"
            />
          </Card>
        </TabPane>
        
        <TabPane tab="用户行为分析" key="user-behavior">
          <Row gutter={[16, 16]}>
            <Col span={24}>
              <Card title="用户活跃度热力图" loading={loading}>
                <Heatmap {...heatmapConfig} height={300} />
              </Card>
            </Col>
          </Row>
        </TabPane>
        
        <TabPane tab="性能诊断" key="performance">
          <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
            {performanceMetrics.map((metric, index) => (
              <Col xs={24} sm={12} md={8} key={index}>
                <Card size="small">
                  <Statistic
                    title={metric.metric}
                    value={metric.current}
                    precision={metric.metric.includes('满意度') ? 1 : 3}
                    prefix={metric.trend > 0 ? <RiseOutlined /> : <FallOutlined />}
                    valueStyle={{ 
                      color: metric.status === 'good' ? '#3f8600' : 
                             metric.status === 'warning' ? '#faad14' : '#cf1322' 
                    }}
                    suffix={
                      metric.metric.includes('率') ? '%' :
                      metric.metric.includes('延迟') ? 'ms' :
                      metric.metric.includes('满意度') ? '/5.0' : ''
                    }
                  />
                  <Progress
                    percent={(metric.current / metric.target) * 100}
                    size="small"
                    status={metric.status === 'good' ? 'success' : 'exception'}
                    style={{ marginTop: '8px' }}
                  />
                </Card>
              </Col>
            ))}
          </Row>
        </TabPane>
      </Tabs>
    </div>
  );
};

export default RLMetricsAnalysisPage;