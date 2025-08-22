/**
 * 推荐项反馈分析页面
 * 分析推荐项的反馈数据和表现
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Table,
  Input,
  Button,
  Space,
  Typography,
  Tag,
  Drawer,
  Descriptions,
  Progress,
  Row,
  Col,
  Statistic,
  Select,
  Rate,
  Tooltip,
  message
} from 'antd';
import {
  TrophyOutlined,
  SearchOutlined,
  EyeOutlined,
  StarOutlined,
  HeartOutlined,
  LikeOutlined,
  CommentOutlined,
  ShareAltOutlined,
  RiseOutlined,
  FallOutlined,
  BookOutlined,
  UserOutlined
} from '@ant-design/icons';
import { Line, Column, Gauge } from '@ant-design/plots';
import type { ColumnsType } from 'antd/es/table';

const { Title, Text } = Typography;
const { Search } = Input;
const { Option } = Select;

interface ItemFeedback {
  id: string;
  item_id: string;
  title: string;
  category: string;
  total_feedbacks: number;
  unique_users: number;
  average_rating: number;
  rating_count: number;
  like_count: number;
  dislike_count: number;
  like_ratio: number;
  view_count: number;
  click_count: number;
  bookmark_count: number;
  share_count: number;
  comment_count: number;
  average_dwell_time: number;
  average_scroll_depth: number;
  bounce_rate: number;
  overall_quality_score: number;
  sentiment_score: number;
  first_feedback_time: string;
  last_feedback_time: string;
  trend: 'up' | 'down' | 'stable';
  recommendation_score: number;
}

const ItemFeedbackAnalysisPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [items, setItems] = useState<ItemFeedback[]>([]);
  const [selectedItem, setSelectedItem] = useState<ItemFeedback | null>(null);
  const [drawerVisible, setDrawerVisible] = useState(false);
  const [searchText, setSearchText] = useState('');
  const [filterCategory, setFilterCategory] = useState<string>('all');
  const [sortBy, setSortBy] = useState<string>('recommendation_score');

  useEffect(() => {
    loadItemFeedbacks();
  }, []);

  const loadItemFeedbacks = async () => {
    setLoading(true);
    try {
      // 模拟加载推荐项反馈数据
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const categories = ['科技', '娱乐', '教育', '商业', '生活'];
      const mockItems: ItemFeedback[] = Array.from({ length: 100 }, (_, i) => {
        const rating = 1 + Math.random() * 4;
        const likeCount = Math.floor(Math.random() * 500);
        const dislikeCount = Math.floor(Math.random() * 100);
        const totalUsers = Math.floor(Math.random() * 1000) + 100;
        
        return {
          id: `item-${i + 1}`,
          item_id: `item-${i + 1}`,
          title: `推荐项目 ${i + 1}`,
          category: categories[Math.floor(Math.random() * categories.length)],
          total_feedbacks: Math.floor(Math.random() * 2000) + 100,
          unique_users: totalUsers,
          average_rating: rating,
          rating_count: Math.floor(Math.random() * 200) + 50,
          like_count: likeCount,
          dislike_count: dislikeCount,
          like_ratio: likeCount / (likeCount + dislikeCount),
          view_count: Math.floor(Math.random() * 5000) + 500,
          click_count: Math.floor(Math.random() * 1000) + 100,
          bookmark_count: Math.floor(Math.random() * 200) + 10,
          share_count: Math.floor(Math.random() * 100) + 5,
          comment_count: Math.floor(Math.random() * 150) + 10,
          average_dwell_time: 30 + Math.random() * 300,
          average_scroll_depth: 0.3 + Math.random() * 0.7,
          bounce_rate: 0.1 + Math.random() * 0.4,
          overall_quality_score: 0.3 + Math.random() * 0.7,
          sentiment_score: -0.5 + Math.random(),
          first_feedback_time: new Date(Date.now() - Math.random() * 365 * 24 * 60 * 60 * 1000).toISOString(),
          last_feedback_time: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString(),
          trend: Math.random() > 0.6 ? 'up' : Math.random() > 0.3 ? 'stable' : 'down',
          recommendation_score: 0.4 + Math.random() * 0.6
        };
      });
      
      setItems(mockItems);
    } catch (error) {
      console.error('加载推荐项反馈失败:', error);
      message.error('加载推荐项反馈失败');
    } finally {
      setLoading(false);
    }
  };

  const handleViewItem = (item: ItemFeedback) => {
    setSelectedItem(item);
    setDrawerVisible(true);
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up': return <RiseOutlined style={{ color: '#52c41a' }} />;
      case 'down': return <FallOutlined style={{ color: '#ff4d4f' }} />;
      default: return <span style={{ color: '#faad14' }}>—</span>;
    }
  };

  const getSentimentColor = (score: number) => {
    if (score > 0.2) return '#52c41a';
    if (score > -0.2) return '#faad14';
    return '#ff4d4f';
  };

  const getSentimentText = (score: number) => {
    if (score > 0.2) return '积极';
    if (score > -0.2) return '中性';
    return '消极';
  };

  // 表格列定义
  const columns: ColumnsType<ItemFeedback> = [
    {
      title: '推荐项',
      key: 'item_info',
      render: (_, record: ItemFeedback) => (
        <div>
          <div style={{ fontWeight: 'bold' }}>{record.title}</div>
          <div>
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {record.item_id}
            </Text>
            <Tag color="blue" style={{ marginLeft: '8px' }}>
              {record.category}
            </Tag>
          </div>
        </div>
      ),
      filteredValue: searchText ? [searchText] : null,
      onFilter: (value, record) =>
        record.title.toLowerCase().includes(value.toString().toLowerCase()) ||
        record.item_id.toLowerCase().includes(value.toString().toLowerCase()),
    },
    {
      title: '评分',
      key: 'rating',
      render: (_, record: ItemFeedback) => (
        <div>
          <Rate value={record.average_rating} disabled allowHalf />
          <div>
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {record.rating_count} 个评分
            </Text>
          </div>
        </div>
      ),
      sorter: (a, b) => a.average_rating - b.average_rating,
    },
    {
      title: '参与度',
      key: 'engagement',
      render: (_, record: ItemFeedback) => (
        <div>
          <div><HeartOutlined /> {record.like_count}</div>
          <div><CommentOutlined /> {record.comment_count}</div>
          <div><BookOutlined /> {record.bookmark_count}</div>
        </div>
      ),
      sorter: (a, b) => a.total_feedbacks - b.total_feedbacks,
    },
    {
      title: '点赞率',
      dataIndex: 'like_ratio',
      key: 'like_ratio',
      render: (ratio: number) => (
        <Progress
          percent={ratio * 100}
          size="small"
          format={(percent) => `${percent?.toFixed(1)}%`}
        />
      ),
      sorter: (a, b) => a.like_ratio - b.like_ratio,
    },
    {
      title: '质量分数',
      dataIndex: 'overall_quality_score',
      key: 'overall_quality_score',
      render: (score: number) => (
        <Progress
          percent={score * 100}
          size="small"
          strokeColor={score > 0.7 ? '#52c41a' : score > 0.4 ? '#faad14' : '#ff4d4f'}
          format={(percent) => `${percent?.toFixed(0)}%`}
        />
      ),
      sorter: (a, b) => a.overall_quality_score - b.overall_quality_score,
    },
    {
      title: '情感倾向',
      dataIndex: 'sentiment_score',
      key: 'sentiment_score',
      render: (score: number) => (
        <Tag color={getSentimentColor(score)}>
          {getSentimentText(score)}
        </Tag>
      ),
      sorter: (a, b) => a.sentiment_score - b.sentiment_score,
    },
    {
      title: '趋势',
      dataIndex: 'trend',
      key: 'trend',
      render: (trend: string) => getTrendIcon(trend),
      filters: [
        { text: '上升', value: 'up' },
        { text: '稳定', value: 'stable' },
        { text: '下降', value: 'down' },
      ],
      onFilter: (value, record) => record.trend === value,
    },
    {
      title: '推荐分数',
      dataIndex: 'recommendation_score',
      key: 'recommendation_score',
      render: (score: number) => (
        <Tooltip title={`推荐分数: ${(score * 100).toFixed(1)}%`}>
          <Progress
            type="circle"
            percent={score * 100}
            size={40}
            format={(percent) => `${percent?.toFixed(0)}`}
          />
        </Tooltip>
      ),
      sorter: (a, b) => a.recommendation_score - b.recommendation_score,
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record: ItemFeedback) => (
        <Button
          type="link"
          icon={<EyeOutlined />}
          onClick={() => handleViewItem(record)}
        >
          查看详情
        </Button>
      ),
    },
  ];

  // 生成趋势图数据
  const getTrendData = (item: ItemFeedback | null) => {
    if (!item) return [];
    
    return Array.from({ length: 30 }, (_, i) => ({
      date: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
      feedbacks: Math.floor(Math.random() * 50) + 20 + (item.trend === 'up' ? i : item.trend === 'down' ? 30 - i : 25),
      quality: 0.3 + Math.random() * 0.7
    }));
  };

  // 过滤和排序数据
  const filteredItems = items
    .filter(item => filterCategory === 'all' || item.category === filterCategory)
    .sort((a, b) => {
      switch (sortBy) {
        case 'recommendation_score': return b.recommendation_score - a.recommendation_score;
        case 'total_feedbacks': return b.total_feedbacks - a.total_feedbacks;
        case 'average_rating': return b.average_rating - a.average_rating;
        case 'like_ratio': return b.like_ratio - a.like_ratio;
        default: return 0;
      }
    });

  const categories = Array.from(new Set(items.map(item => item.category)));

  return (
    <div style={{ padding: '24px' }}>
      {/* 页面头部 */}
      <div style={{ marginBottom: '24px' }}>
        <Space align="center" style={{ width: '100%', justifyContent: 'space-between' }}>
          <div>
            <Title level={2} style={{ margin: 0 }}>
              <TrophyOutlined style={{ color: '#faad14', marginRight: '8px' }} />
              推荐项反馈分析
            </Title>
            <Text type="secondary">
              分析推荐项目的用户反馈表现和参与度指标
            </Text>
          </div>
          <Space>
            <Search
              placeholder="搜索推荐项"
              value={searchText}
              onChange={(e) => setSearchText(e.target.value)}
              style={{ width: 200 }}
              allowClear
            />
            <Select
              value={filterCategory}
              onChange={setFilterCategory}
              style={{ width: 120 }}
            >
              <Option value="all">全部分类</Option>
              {categories.map(cat => (
                <Option key={cat} value={cat}>{cat}</Option>
              ))}
            </Select>
            <Select
              value={sortBy}
              onChange={setSortBy}
              style={{ width: 140 }}
            >
              <Option value="recommendation_score">推荐分数</Option>
              <Option value="total_feedbacks">反馈总数</Option>
              <Option value="average_rating">平均评分</Option>
              <Option value="like_ratio">点赞率</Option>
            </Select>
            <Button type="primary" onClick={loadItemFeedbacks} loading={loading}>
              刷新数据
            </Button>
          </Space>
        </Space>
      </div>

      {/* 统计卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="推荐项总数"
              value={items.length}
              prefix={<TrophyOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="高质量项目"
              value={items.filter(item => item.overall_quality_score >= 0.7).length}
              prefix={<StarOutlined />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="热门项目"
              value={items.filter(item => item.total_feedbacks >= 1000).length}
              prefix={<HeartOutlined />}
              valueStyle={{ color: '#ff4d4f' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="上升趋势"
              value={items.filter(item => item.trend === 'up').length}
              prefix={<RiseOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 推荐项列表 */}
      <Card title="推荐项列表" extra={<TrophyOutlined />}>
        <Table
          columns={columns}
          dataSource={filteredItems}
          loading={loading}
          rowKey="id"
          pagination={{
            total: filteredItems.length,
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => `第 ${range[0]}-${range[1]} 条，共 ${total} 条`,
          }}
        />
      </Card>

      {/* 推荐项详情抽屉 */}
      <Drawer
        title="推荐项详细分析"
        width={720}
        visible={drawerVisible}
        onClose={() => setDrawerVisible(false)}
        extra={
          <Space>
            <Button>导出报告</Button>
            <Button type="primary">优化建议</Button>
          </Space>
        }
      >
        {selectedItem && (
          <div>
            {/* 基本信息 */}
            <Card title="基本信息" size="small" style={{ marginBottom: '16px' }}>
              <Descriptions column={2} size="small">
                <Descriptions.Item label="推荐项ID">
                  {selectedItem.item_id}
                </Descriptions.Item>
                <Descriptions.Item label="标题">
                  {selectedItem.title}
                </Descriptions.Item>
                <Descriptions.Item label="分类">
                  <Tag color="blue">{selectedItem.category}</Tag>
                </Descriptions.Item>
                <Descriptions.Item label="趋势">
                  {getTrendIcon(selectedItem.trend)}
                </Descriptions.Item>
                <Descriptions.Item label="首次反馈">
                  {new Date(selectedItem.first_feedback_time).toLocaleDateString()}
                </Descriptions.Item>
                <Descriptions.Item label="最近反馈">
                  {new Date(selectedItem.last_feedback_time).toLocaleDateString()}
                </Descriptions.Item>
              </Descriptions>
            </Card>

            {/* 核心指标 */}
            <Card title="核心指标" size="small" style={{ marginBottom: '16px' }}>
              <Row gutter={16}>
                <Col span={6}>
                  <div style={{ textAlign: 'center' }}>
                    <Gauge
                      percent={selectedItem.overall_quality_score}
                      range={{ color: ['#FF4D4F', '#FAAD14', '#52C41A'] }}
                      height={120}
                    />
                    <div style={{ marginTop: '8px' }}>质量分数</div>
                  </div>
                </Col>
                <Col span={6}>
                  <div style={{ textAlign: 'center' }}>
                    <Gauge
                      percent={selectedItem.recommendation_score}
                      range={{ color: ['#FF4D4F', '#FAAD14', '#52C41A'] }}
                      height={120}
                    />
                    <div style={{ marginTop: '8px' }}>推荐分数</div>
                  </div>
                </Col>
                <Col span={6}>
                  <div style={{ textAlign: 'center' }}>
                    <Gauge
                      percent={selectedItem.like_ratio}
                      range={{ color: ['#FF4D4F', '#FAAD14', '#52C41A'] }}
                      height={120}
                    />
                    <div style={{ marginTop: '8px' }}>点赞率</div>
                  </div>
                </Col>
                <Col span={6}>
                  <div style={{ textAlign: 'center' }}>
                    <Gauge
                      percent={1 - selectedItem.bounce_rate}
                      range={{ color: ['#FF4D4F', '#FAAD14', '#52C41A'] }}
                      height={120}
                    />
                    <div style={{ marginTop: '8px' }}>留存率</div>
                  </div>
                </Col>
              </Row>
            </Card>

            {/* 反馈统计 */}
            <Card title="反馈统计" size="small" style={{ marginBottom: '16px' }}>
              <Row gutter={16}>
                <Col span={12}>
                  <Statistic
                    title="总反馈数"
                    value={selectedItem.total_feedbacks}
                    prefix={<CommentOutlined />}
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="独立用户"
                    value={selectedItem.unique_users}
                    prefix={<UserOutlined />}
                  />
                </Col>
                <Col span={12}>
                  <div style={{ marginTop: '16px' }}>
                    <Text strong>评分: </Text>
                    <Rate value={selectedItem.average_rating} disabled allowHalf />
                    <Text type="secondary"> ({selectedItem.rating_count})</Text>
                  </div>
                </Col>
                <Col span={12}>
                  <div style={{ marginTop: '16px' }}>
                    <Text strong>情感倾向: </Text>
                    <Tag color={getSentimentColor(selectedItem.sentiment_score)}>
                      {getSentimentText(selectedItem.sentiment_score)}
                    </Tag>
                  </div>
                </Col>
              </Row>
            </Card>

            {/* 用户行为指标 */}
            <Card title="用户行为指标" size="small" style={{ marginBottom: '16px' }}>
              <Row gutter={16}>
                <Col span={8}>
                  <Statistic
                    title="浏览量"
                    value={selectedItem.view_count}
                    prefix={<EyeOutlined />}
                  />
                </Col>
                <Col span={8}>
                  <Statistic
                    title="点击量"
                    value={selectedItem.click_count}
                    prefix={<LikeOutlined />}
                  />
                </Col>
                <Col span={8}>
                  <Statistic
                    title="分享数"
                    value={selectedItem.share_count}
                    prefix={<ShareAltOutlined />}
                  />
                </Col>
                <Col span={8} style={{ marginTop: '16px' }}>
                  <div>
                    <Text>平均停留: </Text>
                    <Tag color="blue">{Math.round(selectedItem.average_dwell_time)}秒</Tag>
                  </div>
                </Col>
                <Col span={8} style={{ marginTop: '16px' }}>
                  <div>
                    <Text>滚动深度: </Text>
                    <Tag color="green">{Math.round(selectedItem.average_scroll_depth * 100)}%</Tag>
                  </div>
                </Col>
                <Col span={8} style={{ marginTop: '16px' }}>
                  <div>
                    <Text>跳出率: </Text>
                    <Tag color="orange">{Math.round(selectedItem.bounce_rate * 100)}%</Tag>
                  </div>
                </Col>
              </Row>
            </Card>

            {/* 趋势图 */}
            <Card title="反馈趋势" size="small">
              <Line
                data={getTrendData(selectedItem)}
                xField="date"
                yField="feedbacks"
                smooth
                height={200}
                meta={{
                  date: { alias: '日期' },
                  feedbacks: { alias: '反馈数量' }
                }}
              />
            </Card>
          </div>
        )}
      </Drawer>
    </div>
  );
};

export default ItemFeedbackAnalysisPage;