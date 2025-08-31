import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Table,
  Tag,
  Button,
  Space,
  Select,
  DatePicker,
  Typography,
  Progress,
  Timeline,
  Alert,
  Badge,
  Avatar,
  Tooltip,
  Modal,
  Form,
  Input,
  Tabs
} from 'antd';
import {
  HeartIcon,
  TrendingUpIcon,
  TrendingDownIcon,
  ActivityIcon,
  BrainIcon,
  ClockIcon,
  TargetIcon,
  UserIcon,
  AlertCircleIcon,
  CheckCircleIcon,
  BarChart3Icon
} from 'lucide-react';
import { Line, Column, Gauge, Area } from '@ant-design/plots';

const { Title, Text } = Typography;
const { RangePicker } = DatePicker;
const { Option } = Select;
const { TabPane } = Tabs;

interface HealthMetric {
  metric_id: string;
  user_id: string;
  metric_type: 'mood_stability' | 'stress_level' | 'social_interaction' | 'sleep_quality' | 'activity_level';
  value: number;
  timestamp: string;
  trend: 'improving' | 'stable' | 'declining';
}

interface HealthGoal {
  goal_id: string;
  user_id: string;
  goal_type: string;
  target_value: number;
  current_value: number;
  deadline: string;
  status: 'active' | 'achieved' | 'overdue';
  created_at: string;
}

interface WellnessRecommendation {
  recommendation_id: string;
  user_id: string;
  category: 'exercise' | 'mindfulness' | 'social' | 'sleep' | 'nutrition';
  title: string;
  description: string;
  priority: 'low' | 'medium' | 'high';
  estimated_impact: number;
  resource_link?: string;
}

interface HealthAlert {
  alert_id: string;
  user_id: string;
  alert_type: 'decline_detected' | 'goal_missed' | 'anomaly_found' | 'improvement_noted';
  severity: 'info' | 'warning' | 'error';
  message: string;
  timestamp: string;
  acknowledged: boolean;
}

const EmotionalHealthMonitoringDashboardPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [healthMetrics, setHealthMetrics] = useState<HealthMetric[]>([]);
  const [healthGoals, setHealthGoals] = useState<HealthGoal[]>([]);
  const [recommendations, setRecommendations] = useState<WellnessRecommendation[]>([]);
  const [healthAlerts, setHealthAlerts] = useState<HealthAlert[]>([]);
  const [selectedUser, setSelectedUser] = useState<string>('all');
  const [timeRange, setTimeRange] = useState<string>('week');
  const [goalModalVisible, setGoalModalVisible] = useState(false);
  
  const [dashboardStats, setDashboardStats] = useState({
    totalUsers: 0,
    averageWellness: 0,
    activeGoals: 0,
    pendingAlerts: 0
  });

  const [form] = Form.useForm();

  // 模拟数据
  useEffect(() => {
    const mockMetrics: HealthMetric[] = [
      {
        metric_id: '1',
        user_id: 'user_001',
        metric_type: 'mood_stability',
        value: 7.5,
        timestamp: new Date().toISOString(),
        trend: 'improving'
      },
      {
        metric_id: '2',
        user_id: 'user_001',
        metric_type: 'stress_level',
        value: 4.2,
        timestamp: new Date().toISOString(),
        trend: 'declining'
      },
      {
        metric_id: '3',
        user_id: 'user_002',
        metric_type: 'social_interaction',
        value: 6.8,
        timestamp: new Date().toISOString(),
        trend: 'stable'
      }
    ];

    const mockGoals: HealthGoal[] = [
      {
        goal_id: '1',
        user_id: 'user_001',
        goal_type: '情绪稳定性提升',
        target_value: 8.0,
        current_value: 7.5,
        deadline: '2024-04-30',
        status: 'active',
        created_at: '2024-03-01'
      },
      {
        goal_id: '2',
        user_id: 'user_001',
        goal_type: '压力水平降低',
        target_value: 3.0,
        current_value: 4.2,
        deadline: '2024-04-15',
        status: 'active',
        created_at: '2024-03-01'
      },
      {
        goal_id: '3',
        user_id: 'user_002',
        goal_type: '社交活动增加',
        target_value: 8.0,
        current_value: 8.2,
        deadline: '2024-03-31',
        status: 'achieved',
        created_at: '2024-02-15'
      }
    ];

    const mockRecommendations: WellnessRecommendation[] = [
      {
        recommendation_id: '1',
        user_id: 'user_001',
        category: 'mindfulness',
        title: '每日正念练习',
        description: '建议每天进行10-15分钟的正念冥想，帮助提升情绪稳定性',
        priority: 'high',
        estimated_impact: 8.5,
        resource_link: '/resources/mindfulness'
      },
      {
        recommendation_id: '2',
        user_id: 'user_001',
        category: 'exercise',
        title: '适度有氧运动',
        description: '每周3-4次30分钟的有氧运动，有助于减轻压力和改善心情',
        priority: 'medium',
        estimated_impact: 7.2,
        resource_link: '/resources/exercise'
      },
      {
        recommendation_id: '3',
        user_id: 'user_002',
        category: 'social',
        title: '维持社交联系',
        description: '保持当前良好的社交活动水平，继续参与群体活动',
        priority: 'low',
        estimated_impact: 6.8
      }
    ];

    const mockAlerts: HealthAlert[] = [
      {
        alert_id: '1',
        user_id: 'user_001',
        alert_type: 'decline_detected',
        severity: 'warning',
        message: '检测到用户情绪稳定性在过去3天内有所下降',
        timestamp: new Date(Date.now() - 3600000).toISOString(),
        acknowledged: false
      },
      {
        alert_id: '2',
        user_id: 'user_002',
        alert_type: 'improvement_noted',
        severity: 'info',
        message: '用户社交活动水平持续改善，已达成设定目标',
        timestamp: new Date(Date.now() - 7200000).toISOString(),
        acknowledged: true
      }
    ];

    setHealthMetrics(mockMetrics);
    setHealthGoals(mockGoals);
    setRecommendations(mockRecommendations);
    setHealthAlerts(mockAlerts);
    setDashboardStats({
      totalUsers: 2,
      averageWellness: 6.8,
      activeGoals: mockGoals.filter(g => g.status === 'active').length,
      pendingAlerts: mockAlerts.filter(a => !a.acknowledged).length
    });
  }, []);

  const metricTypeColors = {
    'mood_stability': '#52c41a',
    'stress_level': '#f5222d',
    'social_interaction': '#1890ff',
    'sleep_quality': '#722ed1',
    'activity_level': '#faad14'
  };

  const metricTypeNames = {
    'mood_stability': '情绪稳定性',
    'stress_level': '压力水平',
    'social_interaction': '社交互动',
    'sleep_quality': '睡眠质量',
    'activity_level': '活动水平'
  };

  const trendIcons = {
    'improving': <TrendingUpIcon size={14} color="#52c41a" />,
    'stable': <ActivityIcon size={14} color="#faad14" />,
    'declining': <TrendingDownIcon size={14} color="#f5222d" />
  };

  const priorityColors = {
    'low': 'default',
    'medium': 'orange',
    'high': 'red'
  };

  const severityColors = {
    'info': 'info',
    'warning': 'warning',
    'error': 'error'
  };

  // 健康趋势图数据
  const healthTrendData = [
    { date: '2024-03-20', value: 6.5, type: '整体健康' },
    { date: '2024-03-21', value: 6.8, type: '整体健康' },
    { date: '2024-03-22', value: 6.6, type: '整体健康' },
    { date: '2024-03-23', value: 7.1, type: '整体健康' },
    { date: '2024-03-24', value: 7.3, type: '整体健康' },
    { date: '2024-03-25', value: 7.0, type: '整体健康' },
    { date: '2024-03-26', value: 7.4, type: '整体健康' }
  ];

  const healthTrendConfig = {
    data: healthTrendData,
    xField: 'date',
    yField: 'value',
    smooth: true,
    color: '#1890ff',
    point: {
      size: 4,
      shape: 'diamond',
    },
    yAxis: {
      min: 0,
      max: 10,
    },
    tooltip: {
      formatter: (datum: any) => ({
        name: '健康指数',
        value: `${datum.value}/10`
      })
    }
  };

  // 仪表盘配置
  const gaugeConfig = {
    percent: dashboardStats.averageWellness / 10,
    color: ['#F4664A', '#FAAD14', '#30BF78'],
    innerRadius: 0.75,
    radius: 0.95,
    statistic: {
      title: {
        style: {
          fontSize: '14px',
          lineHeight: '14px',
        },
        formatter: () => '平均健康指数',
      },
      content: {
        style: {
          fontSize: '24px',
          lineHeight: '24px',
        },
        formatter: () => `${dashboardStats.averageWellness.toFixed(1)}/10`,
      },
    },
  };

  const handleCreateGoal = async (values: any) => {
    setLoading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const newGoal: HealthGoal = {
        goal_id: Date.now().toString(),
        user_id: values.user_id,
        goal_type: values.goal_type,
        target_value: parseFloat(values.target_value),
        current_value: 0,
        deadline: values.deadline,
        status: 'active',
        created_at: new Date().toISOString().split('T')[0]
      };

      setHealthGoals(prev => [newGoal, ...prev]);
      setGoalModalVisible(false);
      form.resetFields();
    } catch (error) {
      console.error('目标创建失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const metricsColumns = [
    {
      title: '用户',
      dataIndex: 'user_id',
      key: 'user_id',
      render: (userId: string) => (
        <Space>
          <Avatar icon={<UserIcon size={14} />} size="small" />
          {userId}
        </Space>
      )
    },
    {
      title: '指标类型',
      dataIndex: 'metric_type',
      key: 'metric_type',
      render: (type: string) => (
        <Tag color={metricTypeColors[type as keyof typeof metricTypeColors]}>
          {metricTypeNames[type as keyof typeof metricTypeNames]}
        </Tag>
      )
    },
    {
      title: '当前值',
      dataIndex: 'value',
      key: 'value',
      render: (value: number) => (
        <Text strong>{value.toFixed(1)}/10</Text>
      )
    },
    {
      title: '趋势',
      dataIndex: 'trend',
      key: 'trend',
      render: (trend: string) => (
        <Space>
          {trendIcons[trend as keyof typeof trendIcons]}
          <Text>{trend}</Text>
        </Space>
      )
    },
    {
      title: '更新时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (time: string) => (
        <Space>
          <ClockIcon size={14} />
          {new Date(time).toLocaleString()}
        </Space>
      )
    }
  ];

  const goalsColumns = [
    {
      title: '用户',
      dataIndex: 'user_id',
      key: 'user_id',
      render: (userId: string) => (
        <Space>
          <Avatar icon={<UserIcon size={14} />} size="small" />
          {userId}
        </Space>
      )
    },
    {
      title: '目标类型',
      dataIndex: 'goal_type',
      key: 'goal_type'
    },
    {
      title: '进度',
      key: 'progress',
      render: (record: HealthGoal) => {
        const progress = Math.min((record.current_value / record.target_value) * 100, 100);
        return (
          <div>
            <Progress 
              percent={progress} 
              size="small"
              status={record.status === 'achieved' ? 'success' : progress > 80 ? 'active' : 'normal'}
            />
            <Text style={{ fontSize: '12px' }}>
              {record.current_value.toFixed(1)} / {record.target_value.toFixed(1)}
            </Text>
          </div>
        );
      }
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const statusConfig = {
          'active': { color: 'processing', text: '进行中' },
          'achieved': { color: 'success', text: '已完成' },
          'overdue': { color: 'error', text: '已逾期' }
        };
        const config = statusConfig[status as keyof typeof statusConfig];
        return <Badge status={config.color} text={config.text} />;
      }
    },
    {
      title: '截止日期',
      dataIndex: 'deadline',
      key: 'deadline',
      render: (date: string) => {
        const isOverdue = new Date(date) < new Date();
        return (
          <Text type={isOverdue ? 'danger' : undefined}>
            {date}
          </Text>
        );
      }
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: HealthGoal) => (
        <Space>
          <Button size="small">详情</Button>
          {record.status === 'active' && (
            <Button size="small" type="primary" ghost>
              更新
            </Button>
          )}
        </Space>
      )
    }
  ];

  return (
    <div style={{ padding: '24px' }}>
      {/* 页面标题 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={24}>
          <Card>
            <Space align="center">
              <HeartIcon size={32} />
              <div>
                <Title level={2} style={{ margin: 0 }}>情感健康监测仪表盘</Title>
                <Text type="secondary">全面监测情感健康状态，追踪康复进展，提供个性化健康建议</Text>
              </div>
            </Space>
          </Card>
        </Col>
      </Row>

      {/* 统计卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="监测用户"
              value={dashboardStats.totalUsers}
              prefix={<UserIcon size={16} />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="平均健康指数"
              value={dashboardStats.averageWellness.toFixed(1)}
              suffix="/10"
              prefix={<HeartIcon size={16} />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="活跃目标"
              value={dashboardStats.activeGoals}
              prefix={<TargetIcon size={16} />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="待处理提醒"
              value={dashboardStats.pendingAlerts}
              prefix={<AlertCircleIcon size={16} />}
              valueStyle={{ color: '#f5222d' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 健康提醒 */}
      {dashboardStats.pendingAlerts > 0 && (
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          <Col span={24}>
            <Alert
              message={`系统检测到 ${dashboardStats.pendingAlerts} 个健康提醒需要处理`}
              description="发现用户健康状态变化，建议及时查看详情并采取相应措施。"
              type="warning"
              showIcon
              action={
                <Button size="small" type="primary" ghost>
                  查看提醒
                </Button>
              }
            />
          </Col>
        </Row>
      )}

      {/* 过滤器和操作按钮 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={24}>
          <Space>
            <Select 
              value={selectedUser} 
              onChange={setSelectedUser}
              style={{ width: 150 }}
            >
              <Option value="all">全部用户</Option>
              <Option value="user_001">user_001</Option>
              <Option value="user_002">user_002</Option>
            </Select>
            <Select 
              value={timeRange} 
              onChange={setTimeRange}
              style={{ width: 120 }}
            >
              <Option value="day">今日</Option>
              <Option value="week">本周</Option>
              <Option value="month">本月</Option>
            </Select>
            <Button 
              type="primary" 
              icon={<TargetIcon size={16} />}
              onClick={() => setGoalModalVisible(true)}
            >
              设置健康目标
            </Button>
            <Button icon={<BarChart3Icon size={16} />}>
              生成报告
            </Button>
          </Space>
        </Col>
      </Row>

      {/* 主要内容区域 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        {/* 健康趋势图 */}
        <Col xs={24} lg={16}>
          <Card 
            title={
              <Space>
                <TrendingUpIcon size={20} />
                健康趋势分析
              </Space>
            }
          >
            <Line {...healthTrendConfig} />
          </Card>
        </Col>

        {/* 健康指数仪表盘 */}
        <Col xs={24} lg={8}>
          <Card 
            title={
              <Space>
                <HeartIcon size={20} />
                整体健康状况
              </Space>
            }
          >
            <Gauge {...gaugeConfig} />
          </Card>
        </Col>
      </Row>

      {/* 详细数据 */}
      <Tabs defaultActiveKey="metrics" type="card">
        <TabPane tab="健康指标" key="metrics">
          <Card 
            title={
              <Space>
                <ActivityIcon size={20} />
                健康指标监测
              </Space>
            }
            extra={<Button type="link">导出数据</Button>}
          >
            <Table
              dataSource={healthMetrics}
              columns={metricsColumns}
              rowKey="metric_id"
              pagination={{ pageSize: 10 }}
            />
          </Card>
        </TabPane>

        <TabPane tab="健康目标" key="goals">
          <Card 
            title={
              <Space>
                <TargetIcon size={20} />
                健康目标追踪
              </Space>
            }
            extra={
              <Button 
                type="primary" 
                size="small"
                onClick={() => setGoalModalVisible(true)}
              >
                新建目标
              </Button>
            }
          >
            <Table
              dataSource={healthGoals}
              columns={goalsColumns}
              rowKey="goal_id"
              pagination={{ pageSize: 10 }}
            />
          </Card>
        </TabPane>

        <TabPane tab="健康建议" key="recommendations">
          <Card 
            title={
              <Space>
                <BrainIcon size={20} />
                个性化健康建议
              </Space>
            }
          >
            <Row gutter={[16, 16]}>
              {recommendations.map(rec => (
                <Col xs={24} md={12} lg={8} key={rec.recommendation_id}>
                  <Card size="small">
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Space justify="space-between" style={{ width: '100%' }}>
                        <Text strong>{rec.title}</Text>
                        <Tag color={priorityColors[rec.priority]}>
                          {rec.priority.toUpperCase()}
                        </Tag>
                      </Space>
                      <Text style={{ fontSize: '12px' }}>
                        {rec.description}
                      </Text>
                      <Space justify="space-between" style={{ width: '100%' }}>
                        <div>
                          <Text style={{ fontSize: '11px' }}>预期效果: </Text>
                          <Progress 
                            percent={rec.estimated_impact * 10} 
                            size="small"
                            style={{ width: 60 }}
                            format={() => rec.estimated_impact.toFixed(1)}
                          />
                        </div>
                        <Button size="small" type="primary" ghost>
                          查看详情
                        </Button>
                      </Space>
                    </Space>
                  </Card>
                </Col>
              ))}
            </Row>
          </Card>
        </TabPane>

        <TabPane tab="健康提醒" key="alerts">
          <Card 
            title={
              <Space>
                <AlertCircleIcon size={20} />
                健康提醒
              </Space>
            }
          >
            <Timeline>
              {healthAlerts.map(alert => (
                <Timeline.Item 
                  key={alert.alert_id}
                  color={alert.severity === 'error' ? 'red' : alert.severity === 'warning' ? 'orange' : 'blue'}
                  dot={alert.acknowledged ? <CheckCircleIcon size={14} /> : <AlertCircleIcon size={14} />}
                >
                  <Space direction="vertical">
                    <Space>
                      <Text strong>{alert.user_id}</Text>
                      <Tag color={severityColors[alert.severity]}>
                        {alert.severity.toUpperCase()}
                      </Tag>
                      {!alert.acknowledged && <Badge status="error" text="未处理" />}
                    </Space>
                    <Text>{alert.message}</Text>
                    <Text type="secondary" style={{ fontSize: '12px' }}>
                      {new Date(alert.timestamp).toLocaleString()}
                    </Text>
                    {!alert.acknowledged && (
                      <Button size="small" type="primary" ghost>
                        标记为已处理
                      </Button>
                    )}
                  </Space>
                </Timeline.Item>
              ))}
            </Timeline>
          </Card>
        </TabPane>
      </Tabs>

      {/* 健康目标创建模态框 */}
      <Modal
        title="设置健康目标"
        open={goalModalVisible}
        onCancel={() => {
          setGoalModalVisible(false);
          form.resetFields();
        }}
        footer={null}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleCreateGoal}
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="user_id"
                label="目标用户"
                rules={[{ required: true, message: '请选择用户' }]}
              >
                <Select placeholder="选择用户">
                  <Option value="user_001">user_001</Option>
                  <Option value="user_002">user_002</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="goal_type"
                label="目标类型"
                rules={[{ required: true, message: '请输入目标类型' }]}
              >
                <Input placeholder="如：情绪稳定性提升" />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="target_value"
                label="目标值 (1-10)"
                rules={[
                  { required: true, message: '请输入目标值' },
                  { type: 'number', min: 1, max: 10, message: '目标值必须在1-10之间' }
                ]}
              >
                <Input type="number" placeholder="输入目标值" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="deadline"
                label="截止日期"
                rules={[{ required: true, message: '请选择截止日期' }]}
              >
                <DatePicker style={{ width: '100%' }} />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={loading}>
                创建目标
              </Button>
              <Button onClick={() => setGoalModalVisible(false)}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default EmotionalHealthMonitoringDashboardPage;