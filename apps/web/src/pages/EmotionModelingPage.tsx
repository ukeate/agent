import React, { useState, useEffect } from 'react';
import { Card, Tabs, Button, Badge, Input, Select, Slider, Progress, Row, Col, Typography, Space, Tag, Alert, Timeline, message } from 'antd';
import { ExperimentOutlined, LineChartOutlined, BarChartOutlined, AimOutlined, ClockCircleOutlined, StarOutlined, HeartOutlined, RadarChartOutlined } from '@ant-design/icons';
const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;
const { TabPane } = Tabs;
const { Option } = Select;

// 类型定义
interface EmotionState {
  id: string;
  emotion: string;
  intensity: number;
  valence: number;
  arousal: number;
  dominance: number;
  timestamp: string;
  confidence: number;
  triggers?: string[];
  context?: Record<string, any>;
}

interface PersonalityProfile {
  user_id: string;
  emotional_traits: Record<string, number>;
  baseline_emotions: Record<string, number>;
  emotion_volatility: number;
  recovery_rate: number;
  dominant_emotions: string[];
  sample_count: number;
  confidence_score: number;
  created_at: string;
  updated_at: string;
}

interface EmotionAnalytics {
  user_id: string;
  period_days: number;
  temporal_patterns: {
    best_hours: Array<[number, number]>;
    worst_hours: Array<[number, number]>;
    weekly_patterns: Record<string, number>;
    monthly_patterns: Record<string, number>;
  };
  emotion_distribution: Record<string, number>;
  volatility: {
    overall_volatility: number;
    valence_volatility: number;
    arousal_volatility: number;
    dominance_volatility: number;
  };
  clusters: Array<{
    name: string;
    emotions: string[];
    frequency: number;
  }>;
  patterns: Array<{
    pattern_type: string;
    description: string;
    frequency: number;
    confidence: number;
  }>;
  recovery_analysis: {
    average_recovery_time: number;
    recovery_rate: number;
    triggers: Record<string, number>;
  };
}

// 临时API客户端实现
const emotionApi = {
  async recordEmotionState(data: any) {
    // 模拟API调用
    console.log('记录情感状态:', data);
    return { success: false, error: '后端服务未连接，使用模拟数据' };
  },
  async getLatestEmotionState() {
    return { success: false, error: '后端服务未连接' };
  },
  async getEmotionHistory(params: any) {
    return { success: false, error: '后端服务未连接' };
  },
  async getPersonalityProfile() {
    return { success: false, error: '后端服务未连接' };
  },
  async getEmotionAnalytics(days: number) {
    return { success: false, error: '后端服务未连接' };
  },
  async predictEmotions(timeHorizon: number) {
    return { success: false, error: '后端服务未连接' };
  },
  connectRealtime(userId: string, callbacks: any): WebSocket {
    console.log('WebSocket连接模拟 - 用户:', userId);
    // 创建一个模拟的WebSocket对象
    return {
      close: () => console.log('模拟WebSocket关闭'),
      send: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => true,
      onopen: null,
      onclose: null,
      onmessage: null,
      onerror: null,
      readyState: 1,
      url: '',
      protocol: '',
      extensions: '',
      binaryType: 'blob' as BinaryType,
      bufferedAmount: 0,
      CONNECTING: 0,
      OPEN: 1,
      CLOSING: 2,
      CLOSED: 3
    } as WebSocket;
  }
};

const EmotionModelingPage: React.FC = () => {
  const [currentEmotion, setCurrentEmotion] = useState<EmotionState | null>(null);
  const [emotionHistory, setEmotionHistory] = useState<EmotionState[]>([]);
  const [personalityProfile, setPersonalityProfile] = useState<PersonalityProfile | null>(null);
  const [prediction, setPrediction] = useState<any>(null);
  const [analytics, setAnalytics] = useState<EmotionAnalytics | null>(null);
  const [loading, setLoading] = useState(false);
  const [wsConnection, setWsConnection] = useState<WebSocket | null>(null);

  // 新状态录入表单
  const [newEmotion, setNewEmotion] = useState({
    emotion: '',
    intensity: 0.5,
    triggers: '',
    context: ''
  });

  const emotions = [
    'happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral',
    'joy', 'trust', 'anticipation', 'contempt', 'shame', 'guilt', 'pride',
    'envy', 'love', 'gratitude', 'hope', 'anxiety', 'depression'
  ];

  useEffect(() => {
    loadData();
    
    // 建立WebSocket连接
    const userId = 'user1'; // 实际应从认证获取
    try {
      const ws = emotionApi.connectRealtime(userId, {
        onOpen: () => {
          console.log('WebSocket连接已建立');
        },
        onMessage: (data) => {
          console.log('收到实时更新:', data);
          // 根据消息类型更新界面
          if (data.type === 'emotion_update') {
            setCurrentEmotion(data.emotion);
            setEmotionHistory(prev => [data.emotion, ...prev].slice(0, 50));
          } else if (data.type === 'profile_update') {
            setPersonalityProfile(data.profile);
          } else if (data.type === 'analytics_update') {
            setAnalytics(data.analytics);
          }
        },
        onClose: () => {
          console.log('WebSocket连接已断开');
        },
        onError: (error) => {
          console.error('WebSocket连接错误:', error);
        }
      });
      
      setWsConnection(ws);
    } catch (error) {
      console.error('建立WebSocket连接失败:', error);
    }

    // 清理函数
    return () => {
      if (wsConnection) {
        wsConnection.close();
      }
    };
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      // 模拟API调用
      await Promise.all([
        loadLatestEmotion(),
        loadEmotionHistory(),
        loadPersonalityProfile(),
        loadAnalytics()
      ]);
    } catch (error) {
      console.error('加载数据失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadLatestEmotion = async () => {
    try {
      const response = await emotionApi.getLatestEmotionState();
      if (response.success && response.data) {
        setCurrentEmotion(response.data);
      }
    } catch (error) {
      console.error('获取最新情感状态失败:', error);
      // 降级到模拟数据
      setCurrentEmotion({
        id: '1',
        emotion: 'happiness',
        intensity: 0.8,
        valence: 0.7,
        arousal: 0.6,
        dominance: 0.7,
        timestamp: new Date().toISOString(),
        confidence: 0.9
      });
    }
  };

  const loadEmotionHistory = async () => {
    try {
      const response = await emotionApi.getEmotionHistory({ limit: 50 });
      if (response.success && response.data) {
        setEmotionHistory(response.data);
      }
    } catch (error) {
      console.error('获取情感历史失败:', error);
      // 降级到模拟数据
      const history = emotions.slice(0, 10).map((emotion, index) => ({
        id: `${index + 1}`,
        emotion,
        intensity: 0.3 + Math.random() * 0.7,
        valence: -1 + Math.random() * 2,
        arousal: Math.random(),
        dominance: Math.random(),
        timestamp: new Date(Date.now() - index * 3600000).toISOString(),
        confidence: 0.7 + Math.random() * 0.3
      }));
      setEmotionHistory(history);
    }
  };

  const loadPersonalityProfile = async () => {
    try {
      const response = await emotionApi.getPersonalityProfile();
      if (response.success && response.data) {
        setPersonalityProfile(response.data);
      }
    } catch (error) {
      console.error('获取个性画像失败:', error);
      // 降级到模拟数据
      setPersonalityProfile({
        user_id: 'user1',
        emotional_traits: {
          extraversion: 0.7,
          neuroticism: 0.3,
          agreeableness: 0.8,
          conscientiousness: 0.6,
          openness: 0.75
        },
        baseline_emotions: {
          happiness: 0.4,
          sadness: 0.1,
          anger: 0.05,
          neutral: 0.35,
          joy: 0.1
        },
        emotion_volatility: 0.4,
        recovery_rate: 0.8,
        dominant_emotions: ['happiness', 'joy', 'neutral'],
        sample_count: 100,
        confidence_score: 0.85,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      });
    }
  };

  const loadAnalytics = async () => {
    try {
      const response = await emotionApi.getEmotionAnalytics(30);
      if (response.success && response.data) {
        setAnalytics(response.data);
      }
    } catch (error) {
      console.error('获取情感分析失败:', error);
      // 降级到模拟数据
      setAnalytics({
        user_id: 'user1',
        period_days: 30,
        temporal_patterns: {
          best_hours: [[9, 0.8], [14, 0.75], [19, 0.7]],
          worst_hours: [[3, 0.2], [23, 0.3], [1, 0.25]],
          weekly_patterns: {},
          monthly_patterns: {}
        },
        emotion_distribution: {
          happiness: 0.35,
          neutral: 0.3,
          joy: 0.15,
          sadness: 0.1,
          other: 0.1
        },
        volatility: {
          overall_volatility: 0.4,
          valence_volatility: 0.3,
          arousal_volatility: 0.25,
          dominance_volatility: 0.2
        },
        clusters: [],
        patterns: [],
        recovery_analysis: {
          average_recovery_time: 2.5,
          recovery_rate: 0.8,
          triggers: {}
        }
      });
    }
  };

  const recordEmotion = async () => {
    if (!newEmotion.emotion) {
      message.error('请选择情感类型');
      return;
    }

    try {
      const emotionData = {
        emotion: newEmotion.emotion,
        intensity: newEmotion.intensity,
        triggers: newEmotion.triggers ? newEmotion.triggers.split(',').map(t => t.trim()).filter(Boolean) : [],
        context: newEmotion.context ? { description: newEmotion.context } : {},
        source: 'manual',
        timestamp: new Date().toISOString()
      };

      const response = await emotionApi.recordEmotionState(emotionData);
      
      if (response.success && response.data) {
        message.success('情感状态记录成功');
        
        // 更新列表
        setEmotionHistory(prev => [response.data, ...prev].slice(0, 50));
        setCurrentEmotion(response.data);
        
        // 重置表单
        setNewEmotion({
          emotion: '',
          intensity: 0.5,
          triggers: '',
          context: ''
        });

        // 重新加载数据
        await Promise.all([
          loadPersonalityProfile(),
          loadAnalytics()
        ]);
      } else {
        throw new Error(response.error || '记录失败');
      }
    } catch (error) {
      console.error('记录情感状态失败:', error);
      message.error('记录失败，请重试');
    }
  };

  const generatePrediction = async (timeHorizon: number = 1) => {
    setLoading(true);
    try {
      const response = await emotionApi.predictEmotions(timeHorizon);
      
      if (response.success && response.data) {
        setPrediction(response.data);
        message.success('预测生成成功');
      } else {
        throw new Error(response.error || '预测失败');
      }
    } catch (error) {
      console.error('生成预测失败:', error);
      // 降级到模拟数据
      setPrediction({
        user_id: 'user1',
        time_horizon_hours: timeHorizon,
        predictions: [
          { emotion: 'happiness', probability: 0.4, intensity_range: [0.6, 0.8] },
          { emotion: 'joy', probability: 0.3, intensity_range: [0.5, 0.7] },
          { emotion: 'neutral', probability: 0.2, intensity_range: [0.4, 0.6] },
          { emotion: 'contentment', probability: 0.1, intensity_range: [0.3, 0.5] }
        ],
        confidence: 0.75,
        factors: {
          current_intensity: 0.8,
          recent_volatility: 0.3,
          personality_influence: 0.6
        },
        timestamp: new Date().toISOString()
      });
      message.warning('使用本地预测数据');
    } finally {
      setLoading(false);
    }
  };

  const getEmotionColor = (emotion: string) => {
    const colorMap: Record<string, string> = {
      happiness: 'gold',
      joy: 'orange',
      sadness: 'blue',
      anger: 'red',
      fear: 'purple',
      neutral: 'default',
      surprise: 'magenta',
      love: 'pink',
      gratitude: 'green'
    };
    return colorMap[emotion] || 'default';
  };

  const renderPersonalityTraits = () => {
    if (!personalityProfile) return null;

    const traits = personalityProfile.emotional_traits;
    const traitNames = {
      extraversion: '外向性',
      neuroticism: '神经质',
      agreeableness: '宜人性',
      conscientiousness: '尽责性',
      openness: '开放性'
    };

    return (
      <Space direction="vertical" style={{ width: '100%' }}>
        {Object.entries(traits).map(([trait, value]) => (
          <div key={trait}>
            <Text strong style={{ display: 'inline-block', width: '80px' }}>
              {traitNames[trait as keyof typeof traitNames]}
            </Text>
            <Progress 
              percent={Math.round((typeof value === 'number' ? value : 0) * 100)} 
              size="small" 
              style={{ flex: 1, marginLeft: 10 }}
              strokeColor="#1890ff"
            />
          </div>
        ))}
      </Space>
    );
  };

  const renderEmotionDistribution = () => {
    if (!analytics?.emotion_distribution) return null;

    return (
      <Space direction="vertical" style={{ width: '100%' }}>
        {Object.entries(analytics.emotion_distribution).map(([emotion, percentage]) => (
          <div key={emotion} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Tag color={getEmotionColor(emotion)}>
              {emotion}
            </Tag>
            <Text strong>
              {(percentage as number * 100).toFixed(1)}%
            </Text>
          </div>
        ))}
      </Space>
    );
  };

  const renderCurrentState = () => (
    <Row gutter={24}>
      <Col span={8}>
        <Card title={
          <span>
            <LineChartOutlined style={{ marginRight: 8 }} />
            当前情感状态
          </span>
        }>
          {currentEmotion ? (
            <Space direction="vertical" style={{ width: '100%' }}>
              <div style={{ textAlign: 'center' }}>
                <Tag 
                  color={getEmotionColor(currentEmotion.emotion)} 
                  style={{ fontSize: '16px', padding: '8px 16px' }}
                >
                  {currentEmotion.emotion}
                </Tag>
              </div>
              <div>
                <Text>强度: </Text>
                <Progress 
                  percent={Math.round(currentEmotion.intensity * 100)} 
                  size="small" 
                  strokeColor="#52c41a"
                />
              </div>
              <div>
                <Text>置信度: </Text>
                <Progress 
                  percent={Math.round(currentEmotion.confidence * 100)} 
                  size="small" 
                  strokeColor="#1890ff"
                />
              </div>
              <Text type="secondary" style={{ fontSize: '12px' }}>
                更新时间: {new Date(currentEmotion.timestamp).toLocaleString()}
              </Text>
            </Space>
          ) : (
            <Text type="secondary">暂无数据</Text>
          )}
        </Card>
      </Col>

      <Col span={8}>
        <Card title={
          <span>
            <AimOutlined style={{ marginRight: 8 }} />
            VAD空间坐标
          </span>
        }>
          {currentEmotion && (
            <Space direction="vertical" style={{ width: '100%' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <Text>效价 (Valence):</Text>
                <Text code>{currentEmotion.valence.toFixed(2)}</Text>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <Text>唤醒度 (Arousal):</Text>
                <Text code>{currentEmotion.arousal.toFixed(2)}</Text>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <Text>支配性 (Dominance):</Text>
                <Text code>{currentEmotion.dominance.toFixed(2)}</Text>
              </div>
            </Space>
          )}
        </Card>
      </Col>

      <Col span={8}>
        <Card title={
          <span>
            <ClockCircleOutlined style={{ marginRight: 8 }} />
            快速操作
          </span>
        }>
          <Space direction="vertical" style={{ width: '100%' }}>
            <Button 
              type="primary" 
              block 
              loading={loading}
              onClick={() => generatePrediction(1)}
            >
              生成情感预测
            </Button>
            <Button 
              block 
              onClick={loadData}
            >
              刷新数据
            </Button>
          </Space>
        </Card>
      </Col>
    </Row>
  );

  const renderEmotionRecord = () => (
    <Card title="记录新的情感状态" style={{ maxWidth: 800, margin: '0 auto' }}>
      <Row gutter={16}>
        <Col span={12}>
          <div style={{ marginBottom: 16 }}>
            <Text strong>情感类型</Text>
            <Select
              style={{ width: '100%', marginTop: 8 }}
              placeholder="选择情感类型"
              value={newEmotion.emotion}
              onChange={(value) => setNewEmotion({ ...newEmotion, emotion: value })}
            >
              {emotions.map((emotion) => (
                <Option key={emotion} value={emotion}>
                  {emotion}
                </Option>
              ))}
            </Select>
          </div>
        </Col>
        <Col span={12}>
          <div style={{ marginBottom: 16 }}>
            <Text strong>强度: {Math.round(newEmotion.intensity * 100)}%</Text>
            <Slider
              style={{ marginTop: 8 }}
              min={0}
              max={1}
              step={0.1}
              value={newEmotion.intensity}
              onChange={(value) => setNewEmotion({ ...newEmotion, intensity: value })}
            />
          </div>
        </Col>
      </Row>

      <div style={{ marginBottom: 16 }}>
        <Text strong>触发因素 (逗号分隔)</Text>
        <Input
          style={{ marginTop: 8 }}
          value={newEmotion.triggers}
          onChange={(e) => setNewEmotion({ ...newEmotion, triggers: e.target.value })}
          placeholder="工作压力, 家庭问题, 运动..."
        />
      </div>

      <div style={{ marginBottom: 16 }}>
        <Text strong>上下文描述</Text>
        <TextArea
          style={{ marginTop: 8 }}
          rows={3}
          value={newEmotion.context}
          onChange={(e) => setNewEmotion({ ...newEmotion, context: e.target.value })}
          placeholder="描述当前情况或相关背景..."
        />
      </div>

      <Button
        type="primary"
        block
        size="large"
        disabled={!newEmotion.emotion}
        onClick={recordEmotion}
      >
        记录情感状态
      </Button>
    </Card>
  );

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>
          <ExperimentOutlined style={{ marginRight: 12, color: '#1890ff' }} />
          情感状态建模系统
        </Title>
      </div>

      <Tabs defaultActiveKey="current">
        <TabPane tab="当前状态" key="current">
          {renderCurrentState()}
        </TabPane>

        <TabPane tab="记录情感" key="record">
          {renderEmotionRecord()}
        </TabPane>

        <TabPane tab="历史轨迹" key="history">
          <Card title="情感历史轨迹">
            <Timeline>
              {emotionHistory.slice(0, 8).map((state, index) => (
                <Timeline.Item 
                  key={state.id} 
                  color={index === 0 ? 'green' : 'blue'}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <Tag color={getEmotionColor(state.emotion)}>
                      {state.emotion}
                    </Tag>
                    <Text>强度: {Math.round(state.intensity * 100)}%</Text>
                    <Text type="secondary">
                      {new Date(state.timestamp).toLocaleString()}
                    </Text>
                  </div>
                </Timeline.Item>
              ))}
            </Timeline>
          </Card>
        </TabPane>

        <TabPane tab="个性画像" key="personality">
          <Row gutter={24}>
            <Col span={12}>
              <Card title="Big Five人格特质">
                {renderPersonalityTraits()}
              </Card>
            </Col>
            <Col span={12}>
              <Card title="情感特征">
                {personalityProfile && (
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Text>情感波动性:</Text>
                      <Badge 
                        color={personalityProfile.emotion_volatility > 0.7 ? 'red' : 'blue'}
                        text={`${Math.round(personalityProfile.emotion_volatility * 100)}%`}
                      />
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Text>恢复速度:</Text>
                      <Badge 
                        color={personalityProfile.recovery_rate > 0.7 ? 'green' : 'blue'}
                        text={`${Math.round(personalityProfile.recovery_rate * 100)}%`}
                      />
                    </div>
                    <div>
                      <Text strong>主导情感:</Text>
                      <div style={{ marginTop: 8 }}>
                        {personalityProfile.dominant_emotions.map((emotion) => (
                          <Tag key={emotion} color={getEmotionColor(emotion)}>
                            {emotion}
                          </Tag>
                        ))}
                      </div>
                    </div>
                  </Space>
                )}
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="情感预测" key="prediction">
          <Card title={
            <span>
              <StarOutlined style={{ marginRight: 8 }} />
              情感预测结果
            </span>
          }>
            {prediction ? (
              <Space direction="vertical" style={{ width: '100%' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Text strong>预测置信度:</Text>
                  <Badge 
                    count={`${Math.round(prediction.confidence * 100)}%`} 
                    style={{ backgroundColor: '#52c41a' }} 
                  />
                </div>
                
                <div>
                  <Title level={4}>预测情感序列 ({prediction.time_horizon_hours}小时内):</Title>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    {prediction.predictions.map((pred: any, index: number) => (
                      <div key={index} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                        <Tag color={getEmotionColor(pred.emotion)}>
                          {pred.emotion}
                        </Tag>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1, marginLeft: 16 }}>
                          <Progress 
                            percent={Math.round(pred.probability * 100)} 
                            size="small" 
                            style={{ flex: 1 }}
                            strokeColor="#1890ff"
                          />
                          <Text style={{ minWidth: '80px' }}>
                            {Math.round(pred.probability * 100)}%
                          </Text>
                          <Text type="secondary" style={{ minWidth: '120px', fontSize: '12px' }}>
                            强度: {pred.intensity_range[0].toFixed(1)}-{pred.intensity_range[1].toFixed(1)}
                          </Text>
                        </div>
                      </div>
                    ))}
                  </Space>
                </div>
              </Space>
            ) : (
              <div style={{ textAlign: 'center', padding: 40 }}>
                <Button 
                  type="primary" 
                  size="large" 
                  loading={loading}
                  onClick={() => generatePrediction(1)}
                >
                  生成情感预测
                </Button>
              </div>
            )}
          </Card>
        </TabPane>

        <TabPane tab="数据分析" key="analytics">
          <Row gutter={24}>
            <Col span={12}>
              <Card title={
                <span>
                  <BarChartOutlined style={{ marginRight: 8 }} />
                  情感分布
                </span>
              }>
                {renderEmotionDistribution()}
              </Card>
            </Col>
            <Col span={12}>
              <Card title="时间模式分析">
                {analytics?.temporal_patterns && (
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div>
                      <Title level={5}>最佳时段:</Title>
                      <Space direction="vertical">
                        {analytics.temporal_patterns.best_hours.map(([hour, score]: [number, number]) => (
                          <div key={hour} style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <Text>{hour}:00</Text>
                            <Text style={{ color: '#52c41a' }}>{Math.round(score * 100)}%</Text>
                          </div>
                        ))}
                      </Space>
                    </div>
                    
                    <div>
                      <Title level={5}>低潮时段:</Title>
                      <Space direction="vertical">
                        {analytics.temporal_patterns.worst_hours.map(([hour, score]: [number, number]) => (
                          <div key={hour} style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <Text>{hour}:00</Text>
                            <Text style={{ color: '#ff4d4f' }}>{Math.round(score * 100)}%</Text>
                          </div>
                        ))}
                      </Space>
                    </div>
                  </Space>
                )}
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>
    </div>
  );
};

export default EmotionModelingPage;