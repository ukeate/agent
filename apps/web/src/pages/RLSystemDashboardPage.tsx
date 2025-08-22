import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Statistic, Progress, Table, Tag, Alert, Button, Select, DatePicker, Space } from 'antd';
import { 
  DashboardOutlined, 
  ThunderboltOutlined, 
  CheckCircleOutlined, 
  ExclamationCircleOutlined,
  LineChartOutlined,
  ApiOutlined,
  ClockCircleOutlined,
  FireOutlined
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';

const { RangePicker } = DatePicker;
const { Option } = Select;

interface SystemMetrics {
  qps: number;
  avgLatency: number;
  errorRate: number;
  cacheHitRate: number;
  activeUsers: number;
  recommendationAccuracy: number;
}

interface AlgorithmPerformance {
  algorithm: string;
  requests: number;
  avgReward: number;
  latency: number;
  accuracy: number;
  status: 'excellent' | 'good' | 'warning' | 'error';
}

interface RecentActivity {
  id: string;
  timestamp: string;
  event: string;
  user: string;
  algorithm: string;
  result: 'success' | 'warning' | 'error';
  details: string;
}

const RLSystemDashboardPage: React.FC = () => {
  const [metrics, setMetrics] = useState<SystemMetrics>({
    qps: 1247,
    avgLatency: 45,
    errorRate: 0.12,
    cacheHitRate: 94.5,
    activeUsers: 8432,
    recommendationAccuracy: 87.3
  });

  const [algorithmData, setAlgorithmData] = useState<AlgorithmPerformance[]>([
    {
      algorithm: 'UCB',
      requests: 15420,
      avgReward: 0.73,
      latency: 12,
      accuracy: 85.2,
      status: 'excellent'
    },
    {
      algorithm: 'Thompson Sampling',
      requests: 12380,
      avgReward: 0.71,
      latency: 15,
      accuracy: 83.7,
      status: 'good'
    },
    {
      algorithm: 'Epsilon Greedy',
      requests: 9850,
      avgReward: 0.68,
      latency: 8,
      accuracy: 81.4,
      status: 'good'
    },
    {
      algorithm: 'Q-Learning',
      requests: 7240,
      avgReward: 0.75,
      latency: 28,
      accuracy: 88.1,
      status: 'warning'
    }
  ]);

  const [recentActivities, setRecentActivities] = useState<RecentActivity[]>([
    {
      id: '1',
      timestamp: '2025-08-22 14:23:15',
      event: '推荐请求',
      user: 'user_12345',
      algorithm: 'UCB',
      result: 'success',
      details: '返回5个推荐项目，响应时间12ms'
    },
    {
      id: '2',
      timestamp: '2025-08-22 14:22:58',
      event: '模型更新',
      user: 'system',
      algorithm: 'Q-Learning',
      result: 'success',
      details: '更新Q表，新准确率88.1%'
    },
    {
      id: '3',
      timestamp: '2025-08-22 14:22:42',
      event: '缓存刷新',
      user: 'system',
      algorithm: 'Redis',
      result: 'success',
      details: '清理过期推荐缓存，释放内存127MB'
    },
    {
      id: '4',
      timestamp: '2025-08-22 14:22:01',
      event: 'A/B测试分流',
      user: 'user_67890',
      algorithm: 'Thompson Sampling',
      result: 'warning',
      details: '分流延迟较高，响应时间35ms'
    }
  ]);

  const [refreshing, setRefreshing] = useState(false);

  const handleRefresh = async () => {
    setRefreshing(true);
    // 模拟API调用
    setTimeout(() => {
      setMetrics(prev => ({
        ...prev,
        qps: prev.qps + Math.floor(Math.random() * 200 - 100),
        avgLatency: Math.max(10, prev.avgLatency + Math.floor(Math.random() * 20 - 10)),
        errorRate: Math.max(0, prev.errorRate + (Math.random() * 0.1 - 0.05))
      }));
      setRefreshing(false);
    }, 1000);
  };

  const algorithmColumns: ColumnsType<AlgorithmPerformance> = [
    {
      title: '算法',
      dataIndex: 'algorithm',
      key: 'algorithm',
      render: (text) => <strong>{text}</strong>
    },
    {
      title: '请求数',
      dataIndex: 'requests',
      key: 'requests',
      render: (value) => value.toLocaleString()
    },
    {
      title: '平均奖励',
      dataIndex: 'avgReward',
      key: 'avgReward',
      render: (value) => (
        <Statistic
          value={value}
          precision={3}
          valueStyle={{ fontSize: '14px' }}
        />
      )
    },
    {
      title: '延迟 (ms)',
      dataIndex: 'latency',
      key: 'latency',
      render: (value) => (
        <Tag color={value < 15 ? 'green' : value < 30 ? 'orange' : 'red'}>
          {value}ms
        </Tag>
      )
    },
    {
      title: '准确率',
      dataIndex: 'accuracy',
      key: 'accuracy',
      render: (value) => (
        <Progress
          percent={value}
          size="small"
          status={value > 85 ? 'success' : value > 80 ? 'normal' : 'exception'}
        />
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => {
        const config = {
          excellent: { color: 'green', text: '优秀' },
          good: { color: 'blue', text: '良好' },
          warning: { color: 'orange', text: '警告' },
          error: { color: 'red', text: '错误' }
        };
        return <Tag color={config[status].color}>{config[status].text}</Tag>;
      }
    }
  ];

  const activityColumns: ColumnsType<RecentActivity> = [
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 150
    },
    {
      title: '事件',
      dataIndex: 'event',
      key: 'event',
      width: 100
    },
    {
      title: '用户/系统',
      dataIndex: 'user',
      key: 'user',
      width: 120
    },
    {
      title: '算法',
      dataIndex: 'algorithm',
      key: 'algorithm',
      width: 120
    },
    {
      title: '结果',
      dataIndex: 'result',
      key: 'result',
      width: 80,
      render: (result) => {
        const config = {
          success: { color: 'green', icon: <CheckCircleOutlined /> },
          warning: { color: 'orange', icon: <ExclamationCircleOutlined /> },
          error: { color: 'red', icon: <ExclamationCircleOutlined /> }
        };
        return (
          <Tag color={config[result].color} icon={config[result].icon}>
            {result}
          </Tag>
        );
      }
    },
    {
      title: '详情',
      dataIndex: 'details',
      key: 'details',
      ellipsis: true
    }
  ];

  useEffect(() => {
    // 设置自动刷新
    const interval = setInterval(() => {
      handleRefresh();
    }, 30000); // 30秒刷新一次

    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <h1 style={{ margin: 0, display: 'flex', alignItems: 'center' }}>
          <DashboardOutlined style={{ marginRight: '8px' }} />
          强化学习系统仪表板
        </h1>
        <Space>
          <Select defaultValue="realtime" style={{ width: 120 }}>
            <Option value="realtime">实时</Option>
            <Option value="5min">5分钟</Option>
            <Option value="1hour">1小时</Option>
            <Option value="1day">1天</Option>
          </Select>
          <RangePicker showTime />
          <Button 
            type="primary" 
            icon={<ThunderboltOutlined />}
            loading={refreshing}
            onClick={handleRefresh}
          >
            刷新
          </Button>
        </Space>
      </div>

      {/* 系统状态警告 */}
      {metrics.errorRate > 1.0 && (
        <Alert
          message="系统状态异常"
          description={`当前错误率为 ${metrics.errorRate.toFixed(2)}%，建议检查系统配置`}
          variant="destructive"
          showIcon
          closable
          style={{ marginBottom: '24px' }}
        />
      )}

      {/* 核心指标卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="QPS"
              value={metrics.qps}
              prefix={<ApiOutlined />}
              suffix="req/s"
              valueStyle={{ color: metrics.qps > 1000 ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="平均延迟"
              value={metrics.avgLatency}
              prefix={<ClockCircleOutlined />}
              suffix="ms"
              valueStyle={{ color: metrics.avgLatency < 50 ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="错误率"
              value={metrics.errorRate}
              prefix={<ExclamationCircleOutlined />}
              suffix="%"
              precision={2}
              valueStyle={{ color: metrics.errorRate < 1 ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="缓存命中率"
              value={metrics.cacheHitRate}
              prefix={<FireOutlined />}
              suffix="%"
              precision={1}
              valueStyle={{ color: metrics.cacheHitRate > 90 ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={12}>
          <Card>
            <Statistic
              title="活跃用户数"
              value={metrics.activeUsers}
              prefix={<LineChartOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12}>
          <Card>
            <Statistic
              title="推荐准确率"
              value={metrics.recommendationAccuracy}
              prefix={<CheckCircleOutlined />}
              suffix="%"
              precision={1}
              valueStyle={{ color: metrics.recommendationAccuracy > 85 ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 算法性能表格 */}
      <Card 
        title="算法性能概览" 
        style={{ marginBottom: '24px' }}
        extra={<Tag color="blue">实时数据</Tag>}
      >
        <Table
          dataSource={algorithmData}
          columns={algorithmColumns}
          rowKey="algorithm"
          pagination={false}
          size="middle"
        />
      </Card>

      {/* 最近活动 */}
      <Card 
        title="最近活动" 
        extra={<Tag color="green">自动更新</Tag>}
      >
        <Table
          dataSource={recentActivities}
          columns={activityColumns}
          rowKey="id"
          pagination={{ pageSize: 10 }}
          size="small"
        />
      </Card>
    </div>
  );
};

export default RLSystemDashboardPage;