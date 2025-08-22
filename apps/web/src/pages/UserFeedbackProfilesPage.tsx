/**
 * 用户反馈档案页面
 * 管理和分析用户反馈行为档案
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
  Avatar,
  Drawer,
  Descriptions,
  Progress,
  Row,
  Col,
  Statistic,
  Select,
  Modal,
  message
} from 'antd';
import {
  UserOutlined,
  SearchOutlined,
  EyeOutlined,
  StarOutlined,
  TrophyOutlined,
  HeartOutlined,
  ClockCircleOutlined,
  ThunderboltOutlined,
  WarningOutlined
} from '@ant-design/icons';
import { Line, Radar } from '@ant-design/plots';
import type { ColumnsType } from 'antd/es/table';

const { Title, Text } = Typography;
const { Search } = Input;
const { Option } = Select;

interface UserProfile {
  id: string;
  user_id: string;
  username: string;
  avatar?: string;
  total_feedbacks: number;
  valid_feedbacks: number;
  average_quality_score: number;
  trust_score: number;
  consistency_score: number;
  activity_days: number;
  last_feedback_time: string;
  first_feedback_time: string;
  feedback_distribution: Record<string, number>;
  preference_vector: Record<string, number>;
  status: 'active' | 'inactive' | 'suspicious';
}

const UserFeedbackProfilesPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [profiles, setProfiles] = useState<UserProfile[]>([]);
  const [selectedProfile, setSelectedProfile] = useState<UserProfile | null>(null);
  const [drawerVisible, setDrawerVisible] = useState(false);
  const [searchText, setSearchText] = useState('');
  const [filterStatus, setFilterStatus] = useState<string>('all');

  useEffect(() => {
    loadUserProfiles();
  }, []);

  const loadUserProfiles = async () => {
    setLoading(true);
    try {
      // 模拟加载用户档案数据
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const mockProfiles: UserProfile[] = Array.from({ length: 50 }, (_, i) => ({
        id: `profile-${i + 1}`,
        user_id: `user-${i + 1}`,
        username: `用户${i + 1}`,
        avatar: undefined,
        total_feedbacks: Math.floor(Math.random() * 1000) + 50,
        valid_feedbacks: Math.floor(Math.random() * 800) + 40,
        average_quality_score: 0.3 + Math.random() * 0.7,
        trust_score: 0.5 + Math.random() * 0.5,
        consistency_score: 0.4 + Math.random() * 0.6,
        activity_days: Math.floor(Math.random() * 365) + 1,
        last_feedback_time: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString(),
        first_feedback_time: new Date(Date.now() - Math.random() * 365 * 24 * 60 * 60 * 1000).toISOString(),
        feedback_distribution: {
          rating: Math.floor(Math.random() * 100),
          like: Math.floor(Math.random() * 100),
          comment: Math.floor(Math.random() * 100),
          bookmark: Math.floor(Math.random() * 100),
          click: Math.floor(Math.random() * 100)
        },
        preference_vector: {
          technology: Math.random(),
          entertainment: Math.random(),
          education: Math.random(),
          business: Math.random(),
          lifestyle: Math.random()
        },
        status: Math.random() > 0.8 ? 'suspicious' : Math.random() > 0.3 ? 'active' : 'inactive'
      }));
      
      setProfiles(mockProfiles);
    } catch (error) {
      console.error('加载用户档案失败:', error);
      message.error('加载用户档案失败');
    } finally {
      setLoading(false);
    }
  };

  const handleViewProfile = (profile: UserProfile) => {
    setSelectedProfile(profile);
    setDrawerVisible(true);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'green';
      case 'inactive': return 'orange';
      case 'suspicious': return 'red';
      default: return 'default';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'active': return '活跃';
      case 'inactive': return '不活跃';
      case 'suspicious': return '可疑';
      default: return '未知';
    }
  };

  const getTrustLevel = (score: number) => {
    if (score >= 0.8) return { level: '高信任', color: '#52c41a' };
    if (score >= 0.6) return { level: '中信任', color: '#fa8c16' };
    return { level: '低信任', color: '#ff4d4f' };
  };

  // 表格列定义
  const columns: ColumnsType<UserProfile> = [
    {
      title: '用户',
      dataIndex: 'username',
      key: 'username',
      render: (username: string, record: UserProfile) => (
        <Space>
          <Avatar icon={<UserOutlined />} />
          <div>
            <div>{username}</div>
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {record.user_id}
            </Text>
          </div>
        </Space>
      ),
      filteredValue: searchText ? [searchText] : null,
      onFilter: (value, record) =>
        record.username.toLowerCase().includes(value.toString().toLowerCase()) ||
        record.user_id.toLowerCase().includes(value.toString().toLowerCase()),
    },
    {
      title: '反馈统计',
      key: 'feedback_stats',
      render: (_, record: UserProfile) => (
        <div>
          <div>总数: <Tag color="blue">{record.total_feedbacks}</Tag></div>
          <div>有效: <Tag color="green">{record.valid_feedbacks}</Tag></div>
        </div>
      ),
      sorter: (a, b) => a.total_feedbacks - b.total_feedbacks,
    },
    {
      title: '质量分数',
      dataIndex: 'average_quality_score',
      key: 'average_quality_score',
      render: (score: number) => (
        <Progress
          percent={score * 100}
          size="small"
          format={(percent) => `${percent?.toFixed(0)}%`}
        />
      ),
      sorter: (a, b) => a.average_quality_score - b.average_quality_score,
    },
    {
      title: '信任度',
      dataIndex: 'trust_score',
      key: 'trust_score',
      render: (score: number) => {
        const { level, color } = getTrustLevel(score);
        return <Tag color={color}>{level}</Tag>;
      },
      sorter: (a, b) => a.trust_score - b.trust_score,
    },
    {
      title: '活跃天数',
      dataIndex: 'activity_days',
      key: 'activity_days',
      render: (days: number) => (
        <Space>
          <ClockCircleOutlined />
          <span>{days}天</span>
        </Space>
      ),
      sorter: (a, b) => a.activity_days - b.activity_days,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={getStatusColor(status)}>
          {getStatusText(status)}
        </Tag>
      ),
      filters: [
        { text: '活跃', value: 'active' },
        { text: '不活跃', value: 'inactive' },
        { text: '可疑', value: 'suspicious' },
      ],
      onFilter: (value, record) => record.status === value,
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record: UserProfile) => (
        <Space>
          <Button
            type="link"
            icon={<EyeOutlined />}
            onClick={() => handleViewProfile(record)}
          >
            查看详情
          </Button>
        </Space>
      ),
    },
  ];

  // 雷达图配置
  const getRadarConfig = (profile: UserProfile | null) => {
    if (!profile) return { data: [] };
    
    const data = Object.entries(profile.preference_vector).map(([key, value]) => ({
      item: key === 'technology' ? '技术' :
            key === 'entertainment' ? '娱乐' :
            key === 'education' ? '教育' :
            key === 'business' ? '商业' :
            key === 'lifestyle' ? '生活' : key,
      score: value * 100
    }));

    return {
      data,
      xField: 'item',
      yField: 'score',
      meta: {
        score: {
          alias: '偏好程度',
          min: 0,
          max: 100,
        },
      },
      xAxis: {
        line: null,
        tickLine: null,
        grid: {
          line: {
            style: {
              lineDash: null,
            },
          },
        },
      },
      yAxis: {
        line: null,
        tickLine: null,
        grid: {
          line: {
            type: 'line',
            style: {
              lineDash: null,
            },
          },
        },
      },
      point: {
        size: 2,
      },
    };
  };

  // 过滤后的数据
  const filteredProfiles = profiles.filter(profile => {
    if (filterStatus !== 'all' && profile.status !== filterStatus) return false;
    return true;
  });

  return (
    <div style={{ padding: '24px' }}>
      {/* 页面头部 */}
      <div style={{ marginBottom: '24px' }}>
        <Space align="center" style={{ width: '100%', justifyContent: 'space-between' }}>
          <div>
            <Title level={2} style={{ margin: 0 }}>
              <UserOutlined style={{ color: '#1890ff', marginRight: '8px' }} />
              用户反馈档案
            </Title>
            <Text type="secondary">
              管理用户反馈行为档案，分析用户偏好和信任度
            </Text>
          </div>
          <Space>
            <Search
              placeholder="搜索用户"
              value={searchText}
              onChange={(e) => setSearchText(e.target.value)}
              style={{ width: 200 }}
              allowClear
            />
            <Select
              value={filterStatus}
              onChange={setFilterStatus}
              style={{ width: 120 }}
            >
              <Option value="all">全部状态</Option>
              <Option value="active">活跃</Option>
              <Option value="inactive">不活跃</Option>
              <Option value="suspicious">可疑</Option>
            </Select>
            <Button type="primary" onClick={loadUserProfiles} loading={loading}>
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
              title="总用户数"
              value={profiles.length}
              prefix={<UserOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="活跃用户"
              value={profiles.filter(p => p.status === 'active').length}
              prefix={<ThunderboltOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="高信任用户"
              value={profiles.filter(p => p.trust_score >= 0.8).length}
              prefix={<TrophyOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="可疑用户"
              value={profiles.filter(p => p.status === 'suspicious').length}
              prefix={<WarningOutlined />}
              valueStyle={{ color: '#ff4d4f' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 用户档案表格 */}
      <Card title="用户档案列表" extra={<UserOutlined />}>
        <Table
          columns={columns}
          dataSource={filteredProfiles}
          loading={loading}
          rowKey="id"
          pagination={{
            total: filteredProfiles.length,
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => `第 ${range[0]}-${range[1]} 条，共 ${total} 条`,
          }}
        />
      </Card>

      {/* 用户详情抽屉 */}
      <Drawer
        title="用户档案详情"
        width={720}
        visible={drawerVisible}
        onClose={() => setDrawerVisible(false)}
        extra={
          <Space>
            <Button>导出数据</Button>
            <Button type="primary">编辑档案</Button>
          </Space>
        }
      >
        {selectedProfile && (
          <div>
            {/* 用户基本信息 */}
            <Card title="基本信息" size="small" style={{ marginBottom: '16px' }}>
              <Descriptions column={2} size="small">
                <Descriptions.Item label="用户ID">
                  {selectedProfile.user_id}
                </Descriptions.Item>
                <Descriptions.Item label="用户名">
                  {selectedProfile.username}
                </Descriptions.Item>
                <Descriptions.Item label="总反馈数">
                  {selectedProfile.total_feedbacks}
                </Descriptions.Item>
                <Descriptions.Item label="有效反馈数">
                  {selectedProfile.valid_feedbacks}
                </Descriptions.Item>
                <Descriptions.Item label="首次反馈">
                  {new Date(selectedProfile.first_feedback_time).toLocaleDateString()}
                </Descriptions.Item>
                <Descriptions.Item label="最近反馈">
                  {new Date(selectedProfile.last_feedback_time).toLocaleDateString()}
                </Descriptions.Item>
                <Descriptions.Item label="活跃天数">
                  {selectedProfile.activity_days}天
                </Descriptions.Item>
                <Descriptions.Item label="用户状态">
                  <Tag color={getStatusColor(selectedProfile.status)}>
                    {getStatusText(selectedProfile.status)}
                  </Tag>
                </Descriptions.Item>
              </Descriptions>
            </Card>

            {/* 质量指标 */}
            <Card title="质量指标" size="small" style={{ marginBottom: '16px' }}>
              <Row gutter={16}>
                <Col span={8}>
                  <div style={{ textAlign: 'center' }}>
                    <Progress
                      type="circle"
                      percent={selectedProfile.average_quality_score * 100}
                      format={(percent) => `${percent?.toFixed(0)}%`}
                      strokeColor="#52c41a"
                      size={80}
                    />
                    <div style={{ marginTop: '8px' }}>平均质量分</div>
                  </div>
                </Col>
                <Col span={8}>
                  <div style={{ textAlign: 'center' }}>
                    <Progress
                      type="circle"
                      percent={selectedProfile.trust_score * 100}
                      format={(percent) => `${percent?.toFixed(0)}%`}
                      strokeColor="#1890ff"
                      size={80}
                    />
                    <div style={{ marginTop: '8px' }}>信任度</div>
                  </div>
                </Col>
                <Col span={8}>
                  <div style={{ textAlign: 'center' }}>
                    <Progress
                      type="circle"
                      percent={selectedProfile.consistency_score * 100}
                      format={(percent) => `${percent?.toFixed(0)}%`}
                      strokeColor="#722ed1"
                      size={80}
                    />
                    <div style={{ marginTop: '8px' }}>一致性</div>
                  </div>
                </Col>
              </Row>
            </Card>

            {/* 反馈分布 */}
            <Card title="反馈类型分布" size="small" style={{ marginBottom: '16px' }}>
              <Row gutter={16}>
                {Object.entries(selectedProfile.feedback_distribution).map(([type, count]) => (
                  <Col span={12} key={type} style={{ marginBottom: '8px' }}>
                    <div>
                      <Text strong>
                        {type === 'rating' ? '评分' :
                         type === 'like' ? '点赞' :
                         type === 'comment' ? '评论' :
                         type === 'bookmark' ? '收藏' :
                         type === 'click' ? '点击' : type}
                      </Text>
                      <Tag color="blue" style={{ float: 'right' }}>{count}</Tag>
                    </div>
                  </Col>
                ))}
              </Row>
            </Card>

            {/* 偏好雷达图 */}
            <Card title="用户偏好分析" size="small">
              <Radar {...getRadarConfig(selectedProfile)} height={300} />
            </Card>
          </div>
        )}
      </Drawer>
    </div>
  );
};

export default UserFeedbackProfilesPage;