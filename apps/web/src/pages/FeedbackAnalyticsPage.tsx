/**
 * 反馈数据分析页面
 * 提供详细的反馈数据分析和可视化
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Table,
  Select,
  DatePicker,
  Button,
  Space,
  Typography,
  Statistic,
  Progress,
  Tag,
  Tabs,
  Alert,
  message
} from 'antd';
import {
  LineChartOutlined,
  BarChartOutlined,
  PieChartOutlined,
  TableOutlined,
  DownloadOutlined,
  FilterOutlined,
  ReloadOutlined
} from '@ant-design/icons';
import { Line, Column, Pie, Heatmap } from '@ant-design/plots';
import { feedbackService } from '../services/feedbackService';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;
const { RangePicker } = DatePicker;

const FeedbackAnalyticsPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [timeRange, setTimeRange] = useState<[any, any] | null>(null);
  const [selectedDimension, setSelectedDimension] = useState('overall');
  const [trendData, setTrendData] = useState<any[]>([]);
  const [distributionData, setDistributionData] = useState<any[]>([]);
  const [heatmapData, setHeatmapData] = useState<any[]>([]);
  const [summaryStats, setSummaryStats] = useState<any>(null);

  useEffect(() => {
    loadAnalyticsData();
  }, [timeRange, selectedDimension]);

  const loadAnalyticsData = async () => {
    setLoading(true);
    try {
      // 模拟加载分析数据
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // 生成模拟趋势数据
      const mockTrends = Array.from({ length: 30 }, (_, i) => ({
        date: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        feedbacks: Math.floor(Math.random() * 200) + 100,
        quality_score: 0.6 + Math.random() * 0.4,
        user_satisfaction: 0.7 + Math.random() * 0.3
      }));
      setTrendData(mockTrends);

      // 生成模拟分布数据
      const mockDistribution = [
        { type: '评分', count: 1250, percentage: 35 },
        { type: '点赞', count: 980, percentage: 28 },
        { type: '点击', count: 650, percentage: 18 },
        { type: '评论', count: 430, percentage: 12 },
        { type: '收藏', count: 240, percentage: 7 }
      ];
      setDistributionData(mockDistribution);

      // 生成模拟热力图数据
      const mockHeatmap = [];
      const hours = Array.from({ length: 24 }, (_, i) => i);
      const days = ['周一', '周二', '周三', '周四', '周五', '周六', '周日'];
      
      days.forEach(day => {
        hours.forEach(hour => {
          mockHeatmap.push({
            day,
            hour: `${hour}:00`,
            value: Math.floor(Math.random() * 100) + 10
          });
        });
      });
      setHeatmapData(mockHeatmap);

      // 生成汇总统计
      setSummaryStats({
        total_feedbacks: 3550,
        unique_users: 1240,
        avg_quality_score: 0.82,
        conversion_rate: 0.24,
        top_feedback_type: '评分',
        growth_rate: 0.15
      });

    } catch (error) {
      console.error('加载分析数据失败:', error);
      message.error('加载分析数据失败');
    } finally {
      setLoading(false);
    }
  };

  // 趋势图配置
  const trendConfig = {
    data: trendData,
    xField: 'date',
    yField: 'feedbacks',
    seriesField: 'type',
    smooth: true,
    animation: {
      appear: {
        animation: 'path-in',
        duration: 2000,
      },
    },
    meta: {
      date: { alias: '日期' },
      feedbacks: { alias: '反馈数量' }
    }
  };

  // 分布饼图配置
  const pieConfig = {
    data: distributionData,
    angleField: 'count',
    colorField: 'type',
    radius: 0.8,
    label: {
      type: 'outer',
      content: '{name} {percentage}%',
    },
    interactions: [{ type: 'element-active' }],
  };

  // 热力图配置
  const heatmapConfig = {
    data: heatmapData,
    xField: 'hour',
    yField: 'day',
    colorField: 'value',
    color: ['#BAE7FF', '#1890FF', '#0050B3'],
    meta: {
      hour: { alias: '小时' },
      day: { alias: '星期' },
      value: { alias: '反馈数量' }
    }
  };

  // 反馈详情表格列
  const feedbackColumns = [
    {
      title: '反馈类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => <Tag color="blue">{type}</Tag>
    },
    {
      title: '总数量',
      dataIndex: 'count',
      key: 'count',
      sorter: (a: any, b: any) => a.count - b.count
    },
    {
      title: '占比',
      dataIndex: 'percentage',
      key: 'percentage',
      render: (percentage: number) => (
        <Progress percent={percentage} size="small" />
      )
    },
    {
      title: '平均质量分',
      dataIndex: 'avg_quality',
      key: 'avg_quality',
      render: (score: number) => score?.toFixed(2) || '-'
    }
  ];

  return (
    <div style={{ padding: '24px' }}>
      {/* 页面头部 */}
      <div style={{ marginBottom: '24px' }}>
        <Space align="center" style={{ width: '100%', justifyContent: 'space-between' }}>
          <div>
            <Title level={2} style={{ margin: 0 }}>
              <LineChartOutlined style={{ color: '#1890ff', marginRight: '8px' }} />
              反馈数据分析
            </Title>
            <Text type="secondary">
              深度分析用户反馈数据，挖掘行为模式和趋势洞察
            </Text>
          </div>
          <Space>
            <Select
              value={selectedDimension}
              onChange={setSelectedDimension}
              style={{ width: 120 }}
            >
              <Option value="overall">总体分析</Option>
              <Option value="user">用户维度</Option>
              <Option value="item">推荐项维度</Option>
              <Option value="time">时间维度</Option>
            </Select>
            <RangePicker onChange={setTimeRange} />
            <Button 
              type="primary" 
              icon={<ReloadOutlined />}
              onClick={loadAnalyticsData}
              loading={loading}
            >
              刷新数据
            </Button>
          </Space>
        </Space>
      </div>

      {/* 汇总指标卡片 */}
      {summaryStats && (
        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="总反馈数"
                value={summaryStats.total_feedbacks}
                prefix={<LineChartOutlined />}
                valueStyle={{ color: '#3f8600' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="活跃用户"
                value={summaryStats.unique_users}
                prefix={<LineChartOutlined />}
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="平均质量分"
                value={summaryStats.avg_quality_score}
                precision={2}
                prefix={<LineChartOutlined />}
                valueStyle={{ color: '#722ed1' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="转化率"
                value={summaryStats.conversion_rate * 100}
                precision={1}
                suffix="%"
                prefix={<LineChartOutlined />}
                valueStyle={{ color: '#fa8c16' }}
              />
            </Card>
          </Col>
        </Row>
      )}

      {/* 主要内容区域 */}
      <Tabs defaultActiveKey="trends" size="large">
        <TabPane tab={<span><LineChartOutlined />趋势分析</span>} key="trends">
          <Row gutter={[16, 16]}>
            <Col xs={24}>
              <Card title="反馈趋势" extra={<BarChartOutlined />}>
                <Line {...trendConfig} height={400} />
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab={<span><PieChartOutlined />分布分析</span>} key="distribution">
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={12}>
              <Card title="反馈类型分布" extra={<PieChartOutlined />}>
                <Pie {...pieConfig} height={350} />
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card title="反馈详情统计" extra={<TableOutlined />}>
                <Table
                  columns={feedbackColumns}
                  dataSource={distributionData}
                  pagination={false}
                  size="small"
                  rowKey="type"
                />
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab={<span><BarChartOutlined />时间热力图</span>} key="heatmap">
          <Row gutter={[16, 16]}>
            <Col xs={24}>
              <Card title="用户活跃时间分布" extra={<BarChartOutlined />}>
                <Heatmap {...heatmapConfig} height={300} />
                <Alert
                  message="使用说明"
                  description="热力图显示不同时间段的用户反馈活跃度，颜色越深表示反馈越多"
                  variant="default"
                  showIcon
                  style={{ marginTop: '16px' }}
                />
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab={<span><FilterOutlined />高级分析</span>} key="advanced">
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={12}>
              <Card title="质量分布分析">
                <div style={{ padding: '20px' }}>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div>
                      <Text strong>高质量反馈 (&gt;0.8)</Text>
                      <Progress percent={65} strokeColor="#52c41a" />
                    </div>
                    <div>
                      <Text strong>中质量反馈 (0.5-0.8)</Text>
                      <Progress percent={28} strokeColor="#fa8c16" />
                    </div>
                    <div>
                      <Text strong>低质量反馈 (&lt;0.5)</Text>
                      <Progress percent={7} strokeColor="#ff4d4f" />
                    </div>
                  </Space>
                </div>
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card title="用户参与度分析">
                <div style={{ padding: '20px' }}>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div>
                      <Text>高活跃用户: </Text>
                      <Tag color="green">342人</Tag>
                    </div>
                    <div>
                      <Text>中活跃用户: </Text>
                      <Tag color="orange">567人</Tag>
                    </div>
                    <div>
                      <Text>低活跃用户: </Text>
                      <Tag color="red">331人</Tag>
                    </div>
                    <div>
                      <Text>平均会话时长: </Text>
                      <Tag color="blue">12.5分钟</Tag>
                    </div>
                  </Space>
                </div>
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>

      {/* 导出功能 */}
      <div style={{ marginTop: '24px', textAlign: 'center' }}>
        <Space>
          <Button icon={<DownloadOutlined />}>导出趋势报告</Button>
          <Button icon={<DownloadOutlined />}>导出分布分析</Button>
          <Button icon={<DownloadOutlined />}>导出完整数据</Button>
        </Space>
      </div>
    </div>
  );
};

export default FeedbackAnalyticsPage;