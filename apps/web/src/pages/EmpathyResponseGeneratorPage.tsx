import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useState, useEffect } from 'react';
import { Card, Tabs, Button, Input, Select, Slider, Progress, Row, Col, Typography, Space, Tag, Alert, Timeline, message, Badge, Spin } from 'antd';
import { HeartOutlined, MessageOutlined, UserOutlined, SettingOutlined, BarChartOutlined, GlobalOutlined, ClockCircleOutlined, BulbOutlined } from '@ant-design/icons';

import { logger } from '../utils/logger'
const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;
const { TabPane } = Tabs;
const { Option } = Select;

// 类型定义
interface EmpathyRequest {
  user_id: string;
  message: string;
  emotion_state?: {
    emotion: string;
    intensity: number;
    valence: number;
    arousal: number;
    dominance: number;
  };
  personality_profile?: {
    emotional_traits: Record<string, number>;
    baseline_emotions: Record<string, number>;
    emotion_volatility: number;
    recovery_rate: number;
  };
  preferred_empathy_type?: string;
  cultural_context?: string;
  max_response_length: number;
  urgency_level: number;
}

interface EmpathyResponse {
  id: string;
  response_text: string;
  empathy_type: string;
  emotion_addressed: string;
  comfort_level: number;
  personalization_score: number;
  suggested_actions: string[];
  tone: string;
  confidence: number;
  timestamp: string;
  generation_time_ms: number;
  cultural_adaptation?: string;
  template_used?: string;
  metadata: Record<string, any>;
}

interface Strategy {
  type: string;
  name: string;
  description: string;
  suitable_for: string[];
}

interface AnalyticsData {
  system_performance: {
    total_requests: number;
    successful_responses: number;
    average_response_time: number;
    strategy_usage: Record<string, number>;
    success_rate: number;
  };
  context_statistics: Record<string, any>;
  personalization_stats: Record<string, any>;
}

// API客户端
const empathyApi = {
  async generateResponse(request: EmpathyRequest): Promise<{ success: boolean; data?: EmpathyResponse; error?: string }> {
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/empathy/generate'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      });
      return { success: true, data: await res.json() };
    } catch (e: any) {
      return { success: false, error: e?.message || '请求失败' };
    }
  },

  async getStrategies(): Promise<{ success: boolean; data?: any; error?: string }> {
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/empathy/strategies'));
      return { success: true, data: await res.json() };
    } catch (e: any) {
      return { success: false, error: e?.message || '请求失败' };
    }
  },

  async getAnalytics(): Promise<{ success: boolean; data?: AnalyticsData; error?: string }> {
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/empathy/analytics'));
      return { success: true, data: await res.json() };
    } catch (e: any) {
      return { success: false, error: e?.message || '请求失败' };
    }
  },

  async submitFeedback(responseId: string, rating: number, feedbackText?: string): Promise<{ success: boolean; error?: string }> {
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/empathy/feedback'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ response_id: responseId, rating, feedback_text: feedbackText }),
      });
      await res.json().catch(() => null);
      return { success: true };
    } catch (e: any) {
      return { success: false, error: e?.message || '请求失败' };
    }
  }
};

const EmpathyResponseGeneratorPage: React.FC = () => {
  const [request, setRequest] = useState<EmpathyRequest>({
    user_id: 'user_demo',
    message: '',
    max_response_length: 200,
    urgency_level: 0.5
  });
  
  const [response, setResponse] = useState<EmpathyResponse | null>(null);
  const [strategies, setStrategies] = useState<any>(null);
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(false);
  const [responseHistory, setResponseHistory] = useState<EmpathyResponse[]>([]);
  
  // 表单状态
  const [emotionForm, setEmotionForm] = useState({
    emotion: '',
    intensity: 0.5,
    valence: 0,
    arousal: 0.5,
    dominance: 0.5
  });

  const [personalityForm, setPersonalityForm] = useState({
    extraversion: 0.5,
    neuroticism: 0.3,
    agreeableness: 0.7,
    conscientiousness: 0.6,
    openness: 0.6
  });

  const emotions = [
    'happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral',
    'joy', 'trust', 'anticipation', 'contempt', 'shame', 'guilt', 'pride',
    'envy', 'love', 'gratitude', 'hope', 'anxiety', 'depression'
  ];

  const empathyTypes = ['cognitive', 'affective', 'compassionate'];
  const culturalContexts = ['collectivist', 'individualist', 'high_context', 'low_context'];

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const [strategiesRes, analyticsRes] = await Promise.all([
        empathyApi.getStrategies(),
        empathyApi.getAnalytics()
      ]);
      
      if (strategiesRes.success) setStrategies(strategiesRes.data);
      if (analyticsRes.success) setAnalytics(analyticsRes.data);
    } catch (error) {
      logger.error('加载数据失败:', error);
    }
  };

  const generateResponse = async () => {
    if (!request.message.trim()) {
      message.error('请输入用户消息');
      return;
    }

    setLoading(true);
    try {
      const empathyRequest = {
        ...request,
        emotion_state: emotionForm.emotion ? {
          emotion: emotionForm.emotion,
          intensity: emotionForm.intensity,
          valence: emotionForm.valence,
          arousal: emotionForm.arousal,
          dominance: emotionForm.dominance
        } : undefined,
        personality_profile: {
          emotional_traits: personalityForm,
          baseline_emotions: {},
          emotion_volatility: 0.4,
          recovery_rate: 0.8
        }
      };

      const result = await empathyApi.generateResponse(empathyRequest);
      
      if (result.success && result.data) {
        setResponse(result.data);
        setResponseHistory(prev => [result.data!, ...prev].slice(0, 10));
        message.success('共情响应生成成功');
        
        // 刷新分析数据
        loadData();
      } else {
        throw new Error(result.error || '生成失败');
      }
    } catch (error) {
      logger.error('生成共情响应失败:', error);
      message.error('生成失败，请重试');
    } finally {
      setLoading(false);
    }
  };

  const submitFeedback = async (responseId: string, rating: number) => {
    try {
      const result = await empathyApi.submitFeedback(responseId, rating);
      if (result.success) {
        message.success('反馈提交成功');
      } else {
        throw new Error(result.error);
      }
    } catch (error) {
      message.error('反馈提交失败');
    }
  };

  const getEmpathyTypeColor = (type: string) => {
    const colors = {
      cognitive: 'blue',
      affective: 'orange',
      compassionate: 'green'
    };
    return colors[type as keyof typeof colors] || 'default';
  };

  const renderRequestForm = () => (
    <Row gutter={24}>
      <Col span={12}>
        <Card title="用户消息与情感状态" style={{ height: '100%' }}>
          <Space direction="vertical" style={{ width: '100%' }}>
            <div>
              <Text strong>用户消息</Text>
              <TextArea
                rows={4}
                value={request.message}
                onChange={(e) => setRequest({ ...request, message: e.target.value })}
                placeholder="请输入用户的情感表达或困扰..."
                style={{ marginTop: 8 }}
              />
            </div>
            
            <div>
              <Text strong>情感类型</Text>
              <Select
                style={{ width: '100%', marginTop: 8 }}
                placeholder="选择情感类型"
                value={emotionForm.emotion}
                onChange={(value) => setEmotionForm({ ...emotionForm, emotion: value })}
                allowClear
              >
                {emotions.map((emotion) => (
                  <Option key={emotion} value={emotion}>
                    {emotion}
                  </Option>
                ))}
              </Select>
            </div>

            <div>
              <Text strong>情感强度: {Math.round(emotionForm.intensity * 100)}%</Text>
              <Slider
                style={{ marginTop: 8 }}
                value={emotionForm.intensity}
                onChange={(value) => setEmotionForm({ ...emotionForm, intensity: value })}
                min={0}
                max={1}
                step={0.1}
              />
            </div>
          </Space>
        </Card>
      </Col>

      <Col span={12}>
        <Card title="个性化设置" style={{ height: '100%' }}>
          <Space direction="vertical" style={{ width: '100%' }}>
            <div>
              <Text strong>偏好的共情类型</Text>
              <Select
                style={{ width: '100%', marginTop: 8 }}
                placeholder="选择共情类型"
                value={request.preferred_empathy_type}
                onChange={(value) => setRequest({ ...request, preferred_empathy_type: value })}
                allowClear
              >
                {empathyTypes.map((type) => (
                  <Option key={type} value={type}>
                    {type}
                  </Option>
                ))}
              </Select>
            </div>

            <div>
              <Text strong>文化背景</Text>
              <Select
                style={{ width: '100%', marginTop: 8 }}
                placeholder="选择文化背景"
                value={request.cultural_context}
                onChange={(value) => setRequest({ ...request, cultural_context: value })}
                allowClear
              >
                {culturalContexts.map((context) => (
                  <Option key={context} value={context}>
                    {context}
                  </Option>
                ))}
              </Select>
            </div>

            <div>
              <Text strong>响应长度限制: {request.max_response_length}字</Text>
              <Slider
                style={{ marginTop: 8 }}
                value={request.max_response_length}
                onChange={(value) => setRequest({ ...request, max_response_length: value })}
                min={50}
                max={500}
                step={10}
              />
            </div>

            <div>
              <Text strong>紧急程度: {Math.round(request.urgency_level * 100)}%</Text>
              <Slider
                style={{ marginTop: 8 }}
                value={request.urgency_level}
                onChange={(value) => setRequest({ ...request, urgency_level: value })}
                min={0}
                max={1}
                step={0.1}
              />
            </div>
          </Space>
        </Card>
      </Col>
    </Row>
  );

  const renderPersonalitySettings = () => (
    <Card title="Big Five 个性特质设置">
      <Row gutter={16}>
        {Object.entries(personalityForm).map(([trait, value]) => {
          const traitNames = {
            extraversion: '外向性',
            neuroticism: '神经质',
            agreeableness: '宜人性',
            conscientiousness: '尽责性',
            openness: '开放性'
          };
          
          return (
            <Col key={trait} span={8} style={{ marginBottom: 16 }}>
              <div>
                <Text strong>{traitNames[trait as keyof typeof traitNames]}: {Math.round(value * 100)}%</Text>
                <Slider
                  style={{ marginTop: 8 }}
                  value={value}
                  onChange={(newValue) => setPersonalityForm({ 
                    ...personalityForm, 
                    [trait]: newValue 
                  })}
                  min={0}
                  max={1}
                  step={0.1}
                />
              </div>
            </Col>
          );
        })}
      </Row>
    </Card>
  );

  const renderResponse = () => {
    if (!response) return null;

    return (
      <Card title="共情响应结果">
        <Space direction="vertical" style={{ width: '100%' }}>
          <Alert
            message={
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span>生成时间: {response.generation_time_ms}ms</span>
                <div>
                  <Tag color={getEmpathyTypeColor(response.empathy_type)}>
                    {response.empathy_type}
                  </Tag>
                  <Tag color="blue">
                    置信度: {Math.round(response.confidence * 100)}%
                  </Tag>
                </div>
              </div>
            }
            type="info"
          />

          <Card>
            <Paragraph style={{ fontSize: 16, lineHeight: 1.8 }}>
              {response.response_text}
            </Paragraph>
          </Card>

          <Row gutter={16}>
            <Col span={8}>
              <Card size="small" title="情感指标">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Text>舒适度:</Text>
                    <Progress 
                      percent={Math.round(response.comfort_level * 100)} 
                      size="small" 
                      strokeColor="#52c41a"
                    />
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Text>个性化得分:</Text>
                    <Progress 
                      percent={Math.round(response.personalization_score * 100)} 
                      size="small" 
                      strokeColor="#1890ff"
                    />
                  </div>
                </Space>
              </Card>
            </Col>
            
            <Col span={8}>
              <Card size="small" title="建议行动">
                <Space direction="vertical">
                  {response.suggested_actions.map((action, index) => (
                    <Tag key={index} color="green">
                      {action}
                    </Tag>
                  ))}
                </Space>
              </Card>
            </Col>
            
            <Col span={8}>
              <Card size="small" title="响应反馈">
                <Space>
                  {[1, 2, 3, 4, 5].map(rating => (
                    <Button
                      key={rating}
                      size="small"
                      onClick={() => submitFeedback(response.id, rating)}
                    >
                      {rating}★
                    </Button>
                  ))}
                </Space>
              </Card>
            </Col>
          </Row>
        </Space>
      </Card>
    );
  };

  const renderStrategies = () => {
    if (!strategies) return <Spin />;

    return (
      <Row gutter={16}>
        {strategies.available_strategies.map((strategy: Strategy) => (
          <Col key={strategy.type} span={8}>
            <Card
              title={
                <span>
                  <Tag color={getEmpathyTypeColor(strategy.type)}>{strategy.type}</Tag>
                  {strategy.name}
                </span>
              }
            >
              <Space direction="vertical" style={{ width: '100%' }}>
                <Paragraph>{strategy.description}</Paragraph>
                <div>
                  <Text strong>适用场景:</Text>
                  <div style={{ marginTop: 8 }}>
                    {strategy.suitable_for.map((scenario: string) => (
                      <Tag key={scenario} style={{ marginBottom: 4 }}>
                        {scenario}
                      </Tag>
                    ))}
                  </div>
                </div>
              </Space>
            </Card>
          </Col>
        ))}
      </Row>
    );
  };

  const renderAnalytics = () => {
    if (!analytics) return <Spin />;

    const { system_performance } = analytics;

    return (
      <Row gutter={24}>
        <Col span={12}>
          <Card title="系统性能">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <Text>总请求数:</Text>
                <Text strong>{system_performance.total_requests.toLocaleString()}</Text>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <Text>成功率:</Text>
                <Badge 
                  count={`${Math.round(system_performance.success_rate * 100)}%`} 
                  style={{ backgroundColor: '#52c41a' }} 
                />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <Text>平均响应时间:</Text>
                <Text strong>{Math.round(system_performance.average_response_time)}ms</Text>
              </div>
            </Space>
          </Card>
        </Col>

        <Col span={12}>
          <Card title="策略使用分布">
            <Space direction="vertical" style={{ width: '100%' }}>
              {Object.entries(system_performance.strategy_usage).map(([strategy, count]) => (
                <div key={strategy} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Tag color={getEmpathyTypeColor(strategy)}>
                    {strategy}
                  </Tag>
                  <div style={{ flex: 1, margin: '0 16px' }}>
                    <Progress 
                      percent={Math.round((count as number) / system_performance.total_requests * 100)} 
                      size="small"
                      strokeColor={getEmpathyTypeColor(strategy)}
                    />
                  </div>
                  <Text strong>{(count as number).toLocaleString()}</Text>
                </div>
              ))}
            </Space>
          </Card>
        </Col>
      </Row>
    );
  };

  const renderHistory = () => (
    <Card title="响应历史">
      {responseHistory.length > 0 ? (
        <Timeline>
          {responseHistory.map((resp, index) => (
            <Timeline.Item key={resp.id} color={index === 0 ? 'green' : 'blue'}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                  <Tag color={getEmpathyTypeColor(resp.empathy_type)}>
                    {resp.empathy_type}
                  </Tag>
                  <Text type="secondary">
                    {new Date(resp.timestamp).toLocaleString()}
                  </Text>
                  <Text type="secondary">
                    {resp.generation_time_ms}ms
                  </Text>
                </div>
                <Paragraph style={{ marginBottom: 0 }}>
                  {resp.response_text}
                </Paragraph>
              </div>
            </Timeline.Item>
          ))}
        </Timeline>
      ) : (
        <div style={{ textAlign: 'center', color: '#999', padding: 40 }}>
          暂无响应历史
        </div>
      )}
    </Card>
  );

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>
          <HeartOutlined style={{ marginRight: 12, color: '#ff4d4f' }} />
          共情响应生成器
        </Title>
        <Paragraph>
          基于情感状态和个性特征生成个性化的共情回应，提供情感支持和建设性建议。
        </Paragraph>
      </div>

      <Tabs defaultActiveKey="generate">
        <TabPane 
          tab={
            <span>
              <MessageOutlined />
              生成响应
            </span>
          } 
          key="generate"
        >
          <Space direction="vertical" style={{ width: '100%' }} size="large">
            {renderRequestForm()}
            {renderPersonalitySettings()}
            
            <div style={{ textAlign: 'center' }}>
              <Button
                type="primary"
                size="large"
                loading={loading}
                onClick={generateResponse}
                disabled={!request.message.trim()}
              >
                生成共情响应
              </Button>
            </div>

            {renderResponse()}
          </Space>
        </TabPane>

        <TabPane 
          tab={
            <span>
              <BulbOutlined />
              策略介绍
            </span>
          } 
          key="strategies"
        >
          {renderStrategies()}
        </TabPane>

        <TabPane 
          tab={
            <span>
              <BarChartOutlined />
              系统分析
            </span>
          } 
          key="analytics"
        >
          {renderAnalytics()}
        </TabPane>

        <TabPane 
          tab={
            <span>
              <ClockCircleOutlined />
              历史记录
            </span>
          } 
          key="history"
        >
          {renderHistory()}
        </TabPane>
      </Tabs>
    </div>
  );
};

export default EmpathyResponseGeneratorPage;
