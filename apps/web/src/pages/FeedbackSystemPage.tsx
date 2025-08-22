/**
 * 用户反馈学习系统主页面
 * 展示反馈系统概览、实时指标和管理功能
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Table,
  Button,
  Space,
  Typography,
  Alert,
  Progress,
  Tag,
  Tabs,
  Switch,
  Select,
  DatePicker,
  message
} from 'antd';
import {
  HeartOutlined,
  LikeOutlined,
  StarOutlined,
  CommentOutlined,
  EyeOutlined,
  UserOutlined,
  TrophyOutlined,
  ClockCircleOutlined,
  DashboardOutlined,
  SettingOutlined,
  BarChartOutlined,
  ThunderboltOutlined
} from '@ant-design/icons';
import { Line, Column, Pie } from '@ant-design/plots';
import { feedbackService } from '../services/feedbackService';
import FeedbackForm from '../components/feedback/FeedbackForm';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;
const { RangePicker } = DatePicker;

interface FeedbackMetrics {
  total_feedbacks: number;
  feedback_types: Record<string, number>;
  unique_users: number;
  average_quality_score: number;
  top_items: Array<{ item_id: string; feedback_count: number; }>;
}

const FeedbackSystemPage: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [metrics, setMetrics] = useState<FeedbackMetrics | null>(null);
  const [realtimeMetrics, setRealtimeMetrics] = useState<any>(null);
  const [trends, setTrends] = useState<any[]>([]);
  const [autoRefresh, setAutoRefresh] = useState(true);

  // 模拟用户和会话ID
  const userId = 'demo-user-123';
  const sessionId = `session-${Date.now()}`;

  useEffect(() => {
    loadFeedbackMetrics();
    loadRealtimeMetrics();
    loadTrends();

    if (autoRefresh) {
      const interval = setInterval(() => {
        loadRealtimeMetrics();
      }, 5000);
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const loadFeedbackMetrics = async () => {
    try {
      setLoading(true);
      const response = await feedbackService.getFeedbackOverview();
      if (response.success) {
        setMetrics(response.data);
      }
    } catch (error) {
      console.error('加载反馈指标失败:', error);
      message.error('加载反馈指标失败');
    } finally {
      setLoading(false);
    }
  };

  const loadRealtimeMetrics = async () => {
    try {
      const response = await feedbackService.getRealTimeFeedbackMetrics();
      if (response.success) {
        setRealtimeMetrics(response.data);
      }
    } catch (error) {
      console.error('加载实时指标失败:', error);
    }
  };

  const loadTrends = async () => {
    try {
      // 模拟趋势数据
      const mockTrends = Array.from({ length: 30 }, (_, i) => ({
        date: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        feedbacks: Math.floor(Math.random() * 100) + 50,
        quality: 0.7 + Math.random() * 0.3
      }));
      setTrends(mockTrends);
    } catch (error) {
      console.error('加载趋势数据失败:', error);
    }
  };

  const handleTestFeedback = async (type: string) => {
    try {
      let response;
      switch (type) {
        case 'rating':
          response = await feedbackService.submitRating(userId, 'test-item-1', 5, sessionId);
          break;
        case 'like':
          response = await feedbackService.submitLike(userId, 'test-item-1', true, sessionId);
          break;
        case 'bookmark':
          response = await feedbackService.submitBookmark(userId, 'test-item-1', true, sessionId);
          break;
        case 'comment':
          response = await feedbackService.submitComment(userId, 'test-item-1', '这个功能很棒！', sessionId);
          break;
        default:
          return;
      }
      
      if (response.success) {
        message.success(`${type} 反馈提交成功`);
        loadFeedbackMetrics();
      }
    } catch (error) {
      message.error(`${type} 反馈提交失败`);
    }
  };

  // 反馈类型分布饼图配置
  const pieConfig = {
    data: metrics ? Object.entries(metrics.feedback_types).map(([type, count]) => ({
      type: type === 'rating' ? '评分' : 
            type === 'like' ? '点赞' : 
            type === 'comment' ? '评论' : 
            type === 'bookmark' ? '收藏' : 
            type === 'click' ? '点击' : 
            type === 'view' ? '浏览' : type,
      value: count
    })) : [],
    angleField: 'value',
    colorField: 'type',
    radius: 0.8,
    label: {
      type: 'outer',
      content: '{name} {percentage}',
    },
    interactions: [
      {
        type: 'element-active',
      },
    ],
  };

  // 趋势图配置
  const lineConfig = {
    data: trends,
    xField: 'date',
    yField: 'feedbacks',
    smooth: true,
    point: {
      size: 3,
    },
    tooltip: {
      showMarkers: false,
    },
    meta: {
      date: {
        alias: '日期',
      },
      feedbacks: {
        alias: '反馈数量',
      },
    },
  };

  // 热门推荐项表格列
  const topItemsColumns = [
    {
      title: '推荐项ID',
      dataIndex: 'item_id',
      key: 'item_id',
    },
    {
      title: '反馈数量',
      dataIndex: 'feedback_count',
      key: 'feedback_count',
      render: (count: number) => <Tag color="blue">{count}</Tag>,
      sorter: (a: any, b: any) => a.feedback_count - b.feedback_count,
    },
  ];

  return (
    <div style={{ padding: '24px' }}>
      {/* 页面头部 */}
      <div style={{ marginBottom: '24px' }}>
        <Space align="center" style={{ width: '100%', justifyContent: 'space-between' }}>
          <div>
            <Title level={2} style={{ margin: 0 }}>
              <HeartOutlined style={{ color: '#ff4d4f', marginRight: '8px' }} />
              用户反馈学习系统
            </Title>
            <Text type="secondary">
              收集、处理和分析用户反馈，为强化学习算法提供高质量奖励信号
            </Text>
          </div>
          <Space>
            <Switch
              checked={autoRefresh}
              onChange={setAutoRefresh}
              checkedChildren="自动刷新"
              unCheckedChildren="手动刷新"
            />
            <Button type="primary" onClick={loadFeedbackMetrics} loading={loading}>
              刷新数据
            </Button>
          </Space>
        </Space>
      </div>

      {/* 实时指标卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="总反馈数"
              value={metrics?.total_feedbacks || 0}
              prefix={<CommentOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="活跃用户"
              value={metrics?.unique_users || 0}
              prefix={<UserOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="平均质量分"
              value={metrics?.average_quality_score || 0}
              precision={2}
              prefix={<TrophyOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="每分钟事件"
              value={realtimeMetrics?.events_per_minute || 0}
              prefix={<ThunderboltOutlined />}
              valueStyle={{ color: '#fa8c16' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 实时状态 */}
      {realtimeMetrics && (
        <Alert
          message="系统状态"
          description={
            <div>
              <Text>活跃会话: {realtimeMetrics.active_sessions} | </Text>
              <Text>处理延迟: {realtimeMetrics.processing_latency}ms | </Text>
              <Text>缓冲区状态: 正常</Text>
            </div>
          }
          variant="default"
          showIcon
          style={{ marginBottom: '24px' }}
        />
      )}

      {/* 主要内容区域 */}
      <Tabs defaultActiveKey="overview" size="large">
        <TabPane tab={<span><DashboardOutlined />概览</span>} key="overview">
          <Row gutter={[16, 16]}>
            {/* 反馈类型分布 */}
            <Col xs={24} lg={12}>
              <Card title="反馈类型分布" extra={<BarChartOutlined />}>
                {metrics && Object.keys(metrics.feedback_types).length > 0 ? (
                  <Pie {...pieConfig} height={300} />
                ) : (
                  <div style={{ textAlign: 'center', padding: '60px 0' }}>
                    <Text type="secondary">暂无数据</Text>
                  </div>
                )}
              </Card>
            </Col>

            {/* 反馈趋势 */}
            <Col xs={24} lg={12}>
              <Card title="反馈趋势" extra={<BarChartOutlined />}>
                <Line {...lineConfig} height={300} />
              </Card>
            </Col>

            {/* 热门推荐项 */}
            <Col xs={24}>
              <Card title="热门推荐项" extra={<TrophyOutlined />}>
                <Table
                  columns={topItemsColumns}
                  dataSource={metrics?.top_items || []}
                  pagination={{ pageSize: 5 }}
                  size="small"
                  rowKey="item_id"
                />
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab={<span><HeartOutlined />反馈测试</span>} key="test">
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={12}>
              <Card title="快速测试反馈" extra={<ThunderboltOutlined />}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Button
                    type="primary"
                    icon={<StarOutlined />}
                    onClick={() => handleTestFeedback('rating')}
                    block
                  >
                    测试评分反馈
                  </Button>
                  <Button
                    type="primary"
                    icon={<LikeOutlined />}
                    onClick={() => handleTestFeedback('like')}
                    block
                  >
                    测试点赞反馈
                  </Button>
                  <Button
                    type="primary"
                    icon={<HeartOutlined />}
                    onClick={() => handleTestFeedback('bookmark')}
                    block
                  >
                    测试收藏反馈
                  </Button>
                  <Button
                    type="primary"
                    icon={<CommentOutlined />}
                    onClick={() => handleTestFeedback('comment')}
                    block
                  >
                    测试评论反馈
                  </Button>
                </Space>
              </Card>
            </Col>

            <Col xs={24} lg={12}>
              <Card title="反馈组件演示" extra={<SettingOutlined />}>
                <div data-testid="feedback-form">
                  <FeedbackForm
                    userId={userId}
                    sessionId={sessionId}
                    itemId="demo-item-123"
                    title="演示反馈组件"
                    onSubmitSuccess={(data) => {
                      message.success(`反馈提交成功: ${data.type}`);
                      loadFeedbackMetrics();
                    }}
                    onSubmitError={(error) => {
                      message.error(`反馈提交失败: ${error.message}`);
                    }}
                  />
                </div>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab={<span><SettingOutlined />系统配置</span>} key="config">
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={12}>
              <Card title="反馈收集配置">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>
                    <Text strong>自动刷新间隔</Text>
                    <Select defaultValue="5" style={{ width: '100%', marginTop: '8px' }}>
                      <Option value="1">1秒</Option>
                      <Option value="5">5秒</Option>
                      <Option value="10">10秒</Option>
                      <Option value="30">30秒</Option>
                    </Select>
                  </div>
                  <div>
                    <Text strong>质量阈值</Text>
                    <Progress
                      percent={70}
                      style={{ marginTop: '8px' }}
                      status="active"
                    />
                  </div>
                  <div>
                    <Text strong>数据保留期</Text>
                    <RangePicker style={{ width: '100%', marginTop: '8px' }} />
                  </div>
                </Space>
              </Card>
            </Col>

            <Col xs={24} lg={12}>
              <Card title="处理管道状态">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>
                    <Text>标准化处理</Text>
                    <Tag color="green" style={{ float: 'right' }}>运行中</Tag>
                  </div>
                  <div>
                    <Text>质量评估</Text>
                    <Tag color="green" style={{ float: 'right' }}>运行中</Tag>
                  </div>
                  <div>
                    <Text>奖励信号生成</Text>
                    <Tag color="green" style={{ float: 'right' }}>运行中</Tag>
                  </div>
                  <div>
                    <Text>情感分析</Text>
                    <Tag color="green" style={{ float: 'right' }}>运行中</Tag>
                  </div>
                </Space>
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>
    </div>
  );
};

export default FeedbackSystemPage;