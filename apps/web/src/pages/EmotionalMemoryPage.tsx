import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useState, useEffect } from 'react';
import {
import { logger } from '../utils/logger'
  Card,
  Row,
  Col,
  Typography,
  Space,
  Button,
  Tag,
  List,
  Avatar,
  Tooltip,
  Divider,
  Input,
  message,
  Spin,
  Alert
} from 'antd';
import {
  HeartOutlined,
  SearchOutlined,
  UserOutlined,
  CalendarOutlined,
  EnvironmentOutlined,
  TagOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { Search } = Input;

interface EmotionalMemoryItem {
  id: string;
  user_id: string;
  content: string;
  emotion: string;
  intensity: number;
  timestamp: string;
  tags: string[];
  context?: {
    location?: string;
    activity?: string;
  };
}

interface MemorySearchResultItem {
  memory: {
    id: string;
    user_id: string;
    timestamp: string;
    emotion_type: string;
    intensity: number;
    content: string;
    tags: string[];
  };
  relevance_score: number;
}

const EmotionalMemoryPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [memories, setMemories] = useState<EmotionalMemoryItem[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [stats, setStats] = useState({
    total: 0,
    emotions: new Map<string, number>()
  });

  // 加载初始数据
  useEffect(() => {
    loadMemories('happiness');
  }, []);

  const loadMemories = async (query: string = 'happiness') => {
    setLoading(true);
    try {
      const response = await apiFetch(buildApiUrl('/emotional-memory/memories/search'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query,
          limit: 10
        })
      });
      
      const data: MemorySearchResultItem[] = await response.json();
      
      if (Array.isArray(data)) {
        const mapped = data.map((item) => ({
          id: item.memory.id,
          user_id: item.memory.user_id,
          content: item.memory.content,
          emotion: item.memory.emotion_type,
          intensity: item.memory.intensity,
          timestamp: item.memory.timestamp,
          tags: item.memory.tags || []
        }));
        setMemories(mapped);
        setStats({
          total: mapped.length,
          emotions: countEmotions(mapped)
        });
        message.success(`找到 ${mapped.length} 条情感记忆`);
      } else {
        message.error('加载情感记忆失败');
      }
    } catch (error) {
      logger.error('API调用失败:', error);
      message.error('连接服务器失败');
    } finally {
      setLoading(false);
    }
  };

  const countEmotions = (memories: EmotionalMemoryItem[]) => {
    const emotionCount = new Map<string, number>();
    memories.forEach(memory => {
      const count = emotionCount.get(memory.emotion) || 0;
      emotionCount.set(memory.emotion, count + 1);
    });
    return emotionCount;
  };

  const handleSearch = (value: string) => {
    setSearchQuery(value);
    if (value.trim()) {
      loadMemories(value);
    }
  };

  const getEmotionColor = (emotion: string) => {
    const colors: { [key: string]: string } = {
      'joy': 'orange',
      'happiness': 'green',
      'sadness': 'blue',
      'anger': 'red',
      'fear': 'purple',
      'surprise': 'cyan'
    };
    return colors[emotion] || 'default';
  };

  const formatIntensity = (intensity: number) => {
    return `${(intensity * 100).toFixed(0)}%`;
  };

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <HeartOutlined style={{ color: '#eb2f96' }} /> 情感记忆管理系统
      </Title>
      <Text type="secondary">智能情感记忆存储、检索和分析平台</Text>

      <Divider />

      {/* 搜索区域 */}
      <Card style={{ marginBottom: '24px' }}>
        <Search
          placeholder="搜索情感记忆..."
          name="emotionalMemorySearch"
          allowClear
          enterButton={<SearchOutlined />}
          size="large"
          onSearch={handleSearch}
          style={{ marginBottom: '16px' }}
        />
        
        {/* 统计信息 */}
        <Row gutter={16}>
          <Col span={8}>
            <Card size="small">
              <div style={{ textAlign: 'center' }}>
                <Title level={3} style={{ margin: 0, color: '#1890ff' }}>
                  {stats.total}
                </Title>
                <Text type="secondary">总记忆数</Text>
              </div>
            </Card>
          </Col>
          <Col span={8}>
            <Card size="small">
              <div style={{ textAlign: 'center' }}>
                <Title level={3} style={{ margin: 0, color: '#52c41a' }}>
                  {stats.emotions.size}
                </Title>
                <Text type="secondary">情感类型</Text>
              </div>
            </Card>
          </Col>
          <Col span={8}>
            <Card size="small">
              <div style={{ textAlign: 'center' }}>
                <Title level={3} style={{ margin: 0, color: '#f5222d' }}>
                  {Array.from(stats.emotions.values()).reduce((a, b) => Math.max(a, b), 0)}
                </Title>
                <Text type="secondary">最频繁情感</Text>
              </div>
            </Card>
          </Col>
        </Row>
      </Card>

      {/* 记忆列表 */}
      <Card title="情感记忆条目" loading={loading}>
        {memories.length === 0 && !loading ? (
          <Alert
            message="暂无记忆数据"
            description="尝试搜索不同的关键词来加载情感记忆数据"
            type="info"
            showIcon
          />
        ) : (
          <List
            itemLayout="vertical"
            dataSource={memories}
            renderItem={(memory) => (
              <List.Item
                key={memory.id}
                extra={
                  <Space direction="vertical" align="end">
                    <Tag color={getEmotionColor(memory.emotion)}>
                      {memory.emotion}
                    </Tag>
                    <Text strong style={{ color: '#f5222d' }}>
                      强度: {formatIntensity(memory.intensity)}
                    </Text>
                  </Space>
                }
              >
                <List.Item.Meta
                  avatar={<Avatar icon={<UserOutlined />} />}
                  title={
                    <Space>
                      <Text strong>{memory.content}</Text>
                      <Tooltip title="记忆ID">
                        <Text type="secondary" style={{ fontSize: '12px' }}>
                          #{memory.id.substring(0, 8)}
                        </Text>
                      </Tooltip>
                    </Space>
                  }
                  description={
                    <Space direction="vertical" size="small">
                      <Space>
                        <CalendarOutlined />
                        <Text type="secondary">
                          {new Date(memory.timestamp).toLocaleString('zh-CN')}
                        </Text>
                      </Space>
                      <Space>
                        <EnvironmentOutlined />
                        {memory.context?.location || memory.context?.activity ? (
                          <Text type="secondary">
                            {memory.context?.location || ''} {memory.context?.activity ? `- ${memory.context.activity}` : ''}
                          </Text>
                        ) : (
                          <Text type="secondary">未提供上下文</Text>
                        )}
                      </Space>
                      <Space>
                        <TagOutlined />
                        {memory.tags.map((tag, index) => (
                          <Tag key={index} size="small">{tag}</Tag>
                        ))}
                      </Space>
                    </Space>
                  }
                />
              </List.Item>
            )}
          />
        )}
      </Card>

      {/* 情感分布 */}
      {stats.emotions.size > 0 && (
        <Card title="情感分布统计" style={{ marginTop: '24px' }}>
          <Row gutter={[16, 16]}>
            {Array.from(stats.emotions.entries()).map(([emotion, count]) => (
              <Col key={emotion} span={6}>
                <Card size="small" style={{ textAlign: 'center' }}>
                  <Tag color={getEmotionColor(emotion)} style={{ marginBottom: '8px' }}>
                    {emotion}
                  </Tag>
                  <div>
                    <Text strong>{count}</Text>
                    <Text type="secondary"> 条记忆</Text>
                  </div>
                </Card>
              </Col>
            ))}
          </Row>
        </Card>
      )}
    </div>
  );
};

export default EmotionalMemoryPage;
