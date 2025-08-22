import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Progress,
  Typography,
  Space,
  Badge,
  Tabs,
  Table,
  Tag,
  Alert,
  Divider,
  Button,
  Select,
  DatePicker,
  Timeline,
  List,
  Avatar,
} from 'antd';
import {
  ExperimentOutlined,
  RiseOutlined,
  FallOutlined,
  TrophyOutlined,
  TeamOutlined,
  ClockCircleOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  SyncOutlined,
  ArrowUpOutlined,
  ArrowDownOutlined,
  FireOutlined,
  EyeOutlined,
  HeartOutlined,
  ShoppingCartOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;
const { RangePicker } = DatePicker;

interface ExperimentMetrics {
  id: string;
  name: string;
  status: 'running' | 'paused' | 'completed';
  conversion_rate: number;
  conversion_change: number;
  users_enrolled: number;
  statistical_significance: boolean;
  confidence_level: number;
  remaining_days: number;
  primary_metric: string;
}

interface ActivityItem {
  id: string;
  type: 'experiment_started' | 'experiment_paused' | 'experiment_completed' | 'alert_triggered';
  experiment_name: string;
  description: string;
  timestamp: string;
  severity?: 'info' | 'warning' | 'error' | 'success';
}

const ExperimentDashboardPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [timeRange, setTimeRange] = useState<string>('7d');
  const [selectedMetric, setSelectedMetric] = useState<string>('conversion_rate');

  // 模拟仪表板数据
  const dashboardStats = {
    total_experiments: 24,
    running_experiments: 8,
    completed_experiments: 12,
    total_users: 45230,
    total_conversions: 6789,
    overall_conversion_rate: 15.0,
    conversion_lift: 12.5,
    revenue_impact: 125000,
    active_alerts: 3,
  };

  const runningExperiments: ExperimentMetrics[] = [
    {
      id: 'exp_001',
      name: '首页改版A/B测试',
      status: 'running',
      conversion_rate: 14.5,
      conversion_change: 12.3,
      users_enrolled: 15420,
      statistical_significance: true,
      confidence_level: 95,
      remaining_days: 8,
      primary_metric: '转化率',
    },
    {
      id: 'exp_002',
      name: '结算页面优化',
      status: 'running',
      conversion_rate: 23.4,
      conversion_change: 8.7,
      users_enrolled: 8930,
      statistical_significance: true,
      confidence_level: 98,
      remaining_days: 12,
      primary_metric: '购买转化',
    },
    {
      id: 'exp_003',
      name: '推荐算法测试',
      status: 'running',
      conversion_rate: 18.9,
      conversion_change: -2.1,
      users_enrolled: 12450,
      statistical_significance: false,
      confidence_level: 78,
      remaining_days: 15,
      primary_metric: '点击率',
    },
    {
      id: 'exp_004',
      name: '定价策略实验',
      status: 'paused',
      conversion_rate: 8.9,
      conversion_change: -15.3,
      users_enrolled: 3245,
      statistical_significance: false,
      confidence_level: 45,
      remaining_days: 20,
      primary_metric: '购买意向',
    },
  ];

  const recentActivities: ActivityItem[] = [
    {
      id: 'act_001',
      type: 'experiment_started',
      experiment_name: '移动端登录优化',
      description: '新实验已启动，预期运行14天',
      timestamp: '2024-01-22 14:30',
      severity: 'success',
    },
    {
      id: 'act_002',
      type: 'alert_triggered',
      experiment_name: '定价策略实验',
      description: '转化率显著下降，建议暂停实验',
      timestamp: '2024-01-22 13:45',
      severity: 'warning',
    },
    {
      id: 'act_003',
      type: 'experiment_completed',
      experiment_name: '搜索功能改版',
      description: '实验已完成，转化率提升15.6%',
      timestamp: '2024-01-22 10:20',
      severity: 'success',
    },
    {
      id: 'act_004',
      type: 'experiment_paused',
      experiment_name: '定价策略实验',
      description: '由于异常指标，实验已暂停',
      timestamp: '2024-01-22 09:15',
      severity: 'warning',
    },
  ];

  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'experiment_started':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'experiment_paused':
        return <WarningOutlined style={{ color: '#faad14' }} />;
      case 'experiment_completed':
        return <TrophyOutlined style={{ color: '#1890ff' }} />;
      case 'alert_triggered':
        return <FireOutlined style={{ color: '#ff4d4f' }} />;
      default:
        return <ClockCircleOutlined />;
    }
  };

  const experimentColumns: ColumnsType<ExperimentMetrics> = [
    {
      title: '实验名称',
      dataIndex: 'name',
      key: 'name',
      width: 200,
      render: (text: string, record: ExperimentMetrics) => (
        <div>
          <Text strong>{text}</Text>
          <div style={{ marginTop: 4 }}>
            <Badge
              status={record.status === 'running' ? 'processing' : 'default'}
              text={record.status === 'running' ? '运行中' : '已暂停'}
            />
          </div>
        </div>
      ),
    },
    {
      title: '主要指标',
      dataIndex: 'primary_metric',
      key: 'primary_metric',
      width: 100,
      render: (metric: string) => <Tag>{metric}</Tag>,
    },
    {
      title: '转化率',
      dataIndex: 'conversion_rate',
      key: 'conversion_rate',
      width: 120,
      render: (rate: number, record: ExperimentMetrics) => (
        <div style={{ textAlign: 'center' }}>
          <Text strong style={{ fontSize: '16px' }}>
            {rate.toFixed(1)}%
          </Text>
          <div style={{ marginTop: 2 }}>
            {record.conversion_change > 0 ? (
              <Text type="success">
                <ArrowUpOutlined /> +{record.conversion_change.toFixed(1)}%
              </Text>
            ) : (
              <Text type="danger">
                <ArrowDownOutlined /> {record.conversion_change.toFixed(1)}%
              </Text>
            )}
          </div>
        </div>
      ),
    },
    {
      title: '参与用户',
      dataIndex: 'users_enrolled',
      key: 'users_enrolled',
      width: 100,
      align: 'right',
      render: (users: number) => <Text>{users.toLocaleString()}</Text>,
    },
    {
      title: '统计显著性',
      key: 'significance',
      width: 120,
      render: (_, record: ExperimentMetrics) => (
        <div style={{ textAlign: 'center' }}>
          <Progress
            type="circle"
            size={50}
            percent={record.confidence_level}
            format={() => `${record.confidence_level}%`}
            strokeColor={record.statistical_significance ? '#52c41a' : '#faad14'}
          />
          {record.statistical_significance && (
            <div style={{ marginTop: 4 }}>
              <Tag color="green" size="small">显著</Tag>
            </div>
          )}
        </div>
      ),
    },
    {
      title: '剩余天数',
      dataIndex: 'remaining_days',
      key: 'remaining_days',
      width: 100,
      align: 'center',
      render: (days: number) => (
        <Text strong style={{ color: days <= 7 ? '#ff4d4f' : '#1890ff' }}>
          {days}天
        </Text>
      ),
    },
  ];

  useEffect(() => {
    setLoading(true);
    // 模拟数据加载
    setTimeout(() => {
      setLoading(false);
    }, 800);
  }, [timeRange, selectedMetric]);

  return (
    <div style={{ padding: '24px' }}>
      {/* 页面标题和筛选 */}
      <div style={{ marginBottom: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <Title level={2} style={{ margin: 0 }}>
              <ExperimentOutlined /> 实验仪表板
            </Title>
            <Text type="secondary">实时监控和分析A/B测试实验效果</Text>
          </div>
          <Space>
            <Select
              value={selectedMetric}
              onChange={setSelectedMetric}
              style={{ width: 140 }}
            >
              <Option value="conversion_rate">转化率</Option>
              <Option value="click_rate">点击率</Option>
              <Option value="revenue">收入</Option>
              <Option value="engagement">用户参与度</Option>
            </Select>
            <Select
              value={timeRange}
              onChange={setTimeRange}
              style={{ width: 120 }}
            >
              <Option value="24h">过去24小时</Option>
              <Option value="7d">过去7天</Option>
              <Option value="30d">过去30天</Option>
              <Option value="90d">过去90天</Option>
            </Select>
            <RangePicker />
          </Space>
        </div>
      </div>

      {/* 关键指标卡片 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="总实验数"
              value={dashboardStats.total_experiments}
              prefix={<ExperimentOutlined />}
              suffix="个"
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="运行中实验"
              value={dashboardStats.running_experiments}
              valueStyle={{ color: '#1890ff' }}
              prefix={<SyncOutlined spin />}
              suffix="个"
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="总参与用户"
              value={dashboardStats.total_users}
              prefix={<TeamOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="总体转化率"
              value={dashboardStats.overall_conversion_rate}
              precision={1}
              valueStyle={{ color: '#52c41a' }}
              prefix={<RiseOutlined />}
              suffix="%"
            />
          </Card>
        </Col>
      </Row>

      {/* 业务影响指标 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={8}>
          <Card>
            <Statistic
              title="转化提升"
              value={dashboardStats.conversion_lift}
              precision={1}
              valueStyle={{ color: '#52c41a' }}
              prefix={<ArrowUpOutlined />}
              suffix="%"
            />
            <Text type="secondary">相比基准版本</Text>
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic
              title="收入影响"
              value={dashboardStats.revenue_impact}
              prefix="¥"
              valueStyle={{ color: '#52c41a' }}
            />
            <Text type="secondary">预计月度增长</Text>
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic
              title="活跃告警"
              value={dashboardStats.active_alerts}
              valueStyle={{ color: '#ff4d4f' }}
              prefix={<WarningOutlined />}
              suffix="条"
            />
            <Text type="secondary">需要处理的异常</Text>
          </Card>
        </Col>
      </Row>

      {/* 主要内容区域 */}
      <Row gutter={16}>
        <Col span={16}>
          <Card title="运行中的实验" style={{ marginBottom: '16px' }}>
            <Table
              columns={experimentColumns}
              dataSource={runningExperiments}
              rowKey="id"
              loading={loading}
              pagination={false}
              size="small"
            />
          </Card>

          {/* 告警信息 */}
          <Card title="系统告警">
            <Alert
              message="定价策略实验异常"
              description="该实验的转化率相比基准版本下降了15.3%，建议立即暂停并检查配置。"
              variant="warning"
              showIcon
              action={
                <Button size="small" type="primary">
                  查看详情
                </Button>
              }
              style={{ marginBottom: '12px' }}
            />
            <Alert
              message="流量分配不均"
              description="首页改版实验的流量分配出现偏差，变体B的流量占比超过预期5%。"
              variant="default"
              showIcon
              action={
                <Button size="small">
                  重新平衡
                </Button>
              }
            />
          </Card>
        </Col>

        <Col span={8}>
          {/* 快速指标 */}
          <Card title="今日快速指标" size="small" style={{ marginBottom: '16px' }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span><EyeOutlined /> 页面访问</span>
                <Text strong>23,456 <Text type="success">(+12.3%)</Text></Text>
              </div>
              <Divider style={{ margin: '8px 0' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span><HeartOutlined /> 用户参与</span>
                <Text strong>18,923 <Text type="success">(+8.7%)</Text></Text>
              </div>
              <Divider style={{ margin: '8px 0' }} />
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span><ShoppingCartOutlined /> 购买转化</span>
                <Text strong>1,234 <Text type="danger">(-2.1%)</Text></Text>
              </div>
            </Space>
          </Card>

          {/* 最近活动 */}
          <Card title="最近活动" size="small">
            <List
              itemLayout="horizontal"
              dataSource={recentActivities}
              renderItem={(item) => (
                <List.Item>
                  <List.Item.Meta
                    avatar={<Avatar icon={getActivityIcon(item.type)} />}
                    title={
                      <div style={{ fontSize: '13px' }}>
                        <Text strong>{item.experiment_name}</Text>
                        <Text type="secondary" style={{ float: 'right', fontSize: '12px' }}>
                          {item.timestamp}
                        </Text>
                      </div>
                    }
                    description={
                      <Text style={{ fontSize: '12px' }} type="secondary">
                        {item.description}
                      </Text>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default ExperimentDashboardPage;