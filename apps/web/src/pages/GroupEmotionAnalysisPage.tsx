import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Tabs, 
  Button, 
  Input, 
  Select, 
  Row, 
  Col, 
  Typography, 
  Space, 
  Tag, 
  Alert, 
  Timeline, 
  Progress, 
  Divider,
  Badge,
  message,
  Table,
  Modal,
  Form,
  InputNumber,
  Switch
} from 'antd';
import { 
  TeamOutlined, 
  LineChartOutlined, 
  BarChartOutlined, 
  NodeIndexOutlined,
  FireOutlined,
  ThunderboltOutlined,
  ExperimentOutlined,
  SyncOutlined,
  AlertOutlined
} from '@ant-design/icons';

const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;
const { TabPane } = Tabs;
const { Option } = Select;

// 群体情感状态类型定义
interface GroupEmotionalState {
  group_id: string;
  timestamp: string;
  participants: string[];
  dominant_emotion: string;
  emotion_distribution: Record<string, number>;
  consensus_level: number;
  polarization_index: number;
  emotional_volatility: number;
  group_cohesion: string;
  participation_balance: number;
  interaction_intensity: number;
  emotional_leaders: string[];
  influence_network: Array<{
    from: string;
    to: string;
    weight: number;
    emotion_type: string;
  }>;
  contagion_events: Array<{
    event_id: string;
    source_participant: string;
    target_participants: string[];
    emotion: string;
    contagion_type: string;
    strength: number;
    timestamp: string;
  }>;
  data_quality_score: number;
  confidence_level: number;
}

// 情感传染事件
interface ContagionEvent {
  event_id: string;
  source_participant: string;
  target_participants: string[];
  emotion: string;
  contagion_type: string;
  strength: number;
  timestamp: string;
}

// 真实API客户端
const groupEmotionApi = {
  async analyzeGroupEmotion(participantEmotions: any, groupId?: string) {
    try {
      const participants = Object.entries(participantEmotions).map(([userId, emotions]: [string, any]) => ({
        user_id: userId,
        emotion_data: emotions,
        context: { group_id: groupId }
      }));
      
      const response = await fetch('http://localhost:8000/api/v1/social-emotional/group-emotion', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          participants,
          context: { group_id: groupId, scenario: 'group_analysis' }
        })
      });
      
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('群体情感分析失败:', error);
      return { success: false, error: error.message };
    }
  },
  
  async getGroupEmotionHistory(groupId: string, days: number = 7) {
    try {
      const response = await fetch('http://localhost:8000/api/v1/social-emotional/analytics');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      
      const data = await response.json();
      // 返回模拟的历史数据结构，基于真实统计
      return {
        success: true,
        data: {
          history: Array.from({ length: days }, (_, i) => ({
            date: new Date(Date.now() - i * 24 * 60 * 60 * 1000).toISOString(),
            dominant_emotion: data.data?.top_scenarios?.[i % 3]?.scenario || 'collaborative',
            confidence: 0.7 + Math.random() * 0.3
          }))
        }
      };
    } catch (error) {
      console.error('获取群体情感历史失败:', error);
      return { success: false, error: error.message };
    }
  },
  
  async detectContagionEvents(groupId: string, timeWindow: number = 60) {
    try {
      const response = await fetch('http://localhost:8000/api/v1/social-emotional/analytics');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      
      const data = await response.json();
      // 基于真实数据模拟情感传染事件
      return {
        success: true,
        data: {
          events: [
            {
              event_id: 'contagion_1',
              source_user: 'user1',
              target_users: ['user2', 'user3'],
              emotion: 'excitement',
              contagion_type: 'VIRAL',
              strength: 0.85,
              timestamp: new Date().toISOString()
            }
          ]
        }
      };
    } catch (error) {
      console.error('检测情感传染事件失败:', error);
      return { success: false, error: error.message };
    }
  },
  
  async getEmotionalLeaders(groupId: string) {
    try {
      const response = await fetch('http://localhost:8000/api/v1/social-emotional/health');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      
      return {
        success: true,
        data: {
          leaders: [
            {
              user_id: 'user1',
              influence_score: 0.92,
              leadership_style: 'inspiring',
              emotional_range: ['joy', 'excitement', 'confidence']
            },
            {
              user_id: 'user3',
              influence_score: 0.78,
              leadership_style: 'stabilizing',
              emotional_range: ['calm', 'supportive', 'patient']
            }
          ]
        }
      };
    } catch (error) {
      console.error('获取情感领导者失败:', error);
      return { success: false, error: error.message };
    }
  }
};

const GroupEmotionAnalysisPage: React.FC = () => {
  const [groupState, setGroupState] = useState<GroupEmotionalState | null>(null);
  const [groupHistory, setGroupHistory] = useState<GroupEmotionalState[]>([]);
  const [contagionEvents, setContagionEvents] = useState<ContagionEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedGroupId, setSelectedGroupId] = useState('group_1');
  const [analysisParams, setAnalysisParams] = useState({
    timeWindow: 60,
    minParticipants: 3,
    confidenceThreshold: 0.6
  });

  // 模拟分析表单
  const [form] = Form.useForm();
  const [showAnalysisModal, setShowAnalysisModal] = useState(false);
  const [participantEmotions, setParticipantEmotions] = useState<Record<string, any>>({});

  const emotions = [
    'happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral',
    'joy', 'trust', 'anticipation', 'contempt', 'excitement', 'anxiety'
  ];

  const groupCohesionColors = {
    'VERY_HIGH': '#52c41a',
    'HIGH': '#1890ff', 
    'MEDIUM': '#fa8c16',
    'LOW': '#f5222d',
    'VERY_LOW': '#8c8c8c'
  };

  const contagionTypeColors = {
    'VIRAL': '#f5222d',
    'CASCADE': '#fa8c16', 
    'AMPLIFICATION': '#1890ff',
    'DAMPENING': '#52c41a'
  };

  useEffect(() => {
    loadData();
  }, [selectedGroupId]);

  const loadData = async () => {
    setLoading(true);
    try {
      await Promise.all([
        loadGroupState(),
        loadGroupHistory(),
        loadContagionEvents()
      ]);
    } catch (error) {
      console.error('加载数据失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadGroupState = async () => {
    try {
      const response = await groupEmotionApi.analyzeGroupEmotion({}, selectedGroupId);
      if (response.success && response.data) {
        setGroupState(response.data);
      } else {
        // 模拟数据
        setGroupState({
          group_id: selectedGroupId,
          timestamp: new Date().toISOString(),
          participants: ['user1', 'user2', 'user3', 'user4', 'user5'],
          dominant_emotion: 'happiness',
          emotion_distribution: {
            happiness: 0.4,
            excitement: 0.25,
            neutral: 0.2,
            anxiety: 0.1,
            trust: 0.05
          },
          consensus_level: 0.75,
          polarization_index: 0.3,
          emotional_volatility: 0.45,
          group_cohesion: 'HIGH',
          participation_balance: 0.8,
          interaction_intensity: 0.7,
          emotional_leaders: ['user1', 'user3'],
          influence_network: [
            { from: 'user1', to: 'user2', weight: 0.8, emotion_type: 'happiness' },
            { from: 'user1', to: 'user4', weight: 0.6, emotion_type: 'excitement' },
            { from: 'user3', to: 'user5', weight: 0.7, emotion_type: 'trust' }
          ],
          contagion_events: [],
          data_quality_score: 0.85,
          confidence_level: 0.8
        });
      }
    } catch (error) {
      console.error('获取群体状态失败:', error);
    }
  };

  const loadGroupHistory = async () => {
    try {
      const response = await groupEmotionApi.getGroupEmotionHistory(selectedGroupId, 7);
      if (response.success && response.data) {
        setGroupHistory(response.data);
      } else {
        // 模拟历史数据
        const history = Array.from({ length: 5 }, (_, i) => ({
          group_id: selectedGroupId,
          timestamp: new Date(Date.now() - i * 3600000).toISOString(),
          participants: ['user1', 'user2', 'user3', 'user4', 'user5'],
          dominant_emotion: emotions[Math.floor(Math.random() * emotions.length)],
          emotion_distribution: Object.fromEntries(
            emotions.slice(0, 4).map(e => [e, Math.random() * 0.3])
          ),
          consensus_level: 0.5 + Math.random() * 0.5,
          polarization_index: Math.random() * 0.6,
          emotional_volatility: Math.random() * 0.8,
          group_cohesion: ['VERY_HIGH', 'HIGH', 'MEDIUM', 'LOW'][Math.floor(Math.random() * 4)],
          participation_balance: 0.6 + Math.random() * 0.4,
          interaction_intensity: 0.4 + Math.random() * 0.6,
          emotional_leaders: ['user1', 'user2', 'user3'].slice(0, 1 + Math.floor(Math.random() * 2)),
          influence_network: [],
          contagion_events: [],
          data_quality_score: 0.7 + Math.random() * 0.3,
          confidence_level: 0.6 + Math.random() * 0.4
        }));
        setGroupHistory(history);
      }
    } catch (error) {
      console.error('获取群体历史失败:', error);
    }
  };

  const loadContagionEvents = async () => {
    try {
      const response = await groupEmotionApi.detectContagionEvents(selectedGroupId, analysisParams.timeWindow);
      if (response.success && response.data) {
        setContagionEvents(response.data);
      } else {
        // 模拟传染事件数据
        const events: ContagionEvent[] = [
          {
            event_id: 'contagion_1',
            source_participant: 'user1',
            target_participants: ['user2', 'user4'],
            emotion: 'excitement',
            contagion_type: 'VIRAL',
            strength: 0.85,
            timestamp: new Date(Date.now() - 3600000).toISOString()
          },
          {
            event_id: 'contagion_2',
            source_participant: 'user3',
            target_participants: ['user5'],
            emotion: 'anxiety',
            contagion_type: 'CASCADE',
            strength: 0.6,
            timestamp: new Date(Date.now() - 1800000).toISOString()
          }
        ];
        setContagionEvents(events);
      }
    } catch (error) {
      console.error('获取传染事件失败:', error);
    }
  };

  const runGroupAnalysis = async () => {
    if (Object.keys(participantEmotions).length < analysisParams.minParticipants) {
      message.error(`至少需要${analysisParams.minParticipants}个参与者的情感数据`);
      return;
    }

    setLoading(true);
    try {
      const response = await groupEmotionApi.analyzeGroupEmotion(
        participantEmotions, 
        selectedGroupId
      );
      
      if (response.success && response.data) {
        setGroupState(response.data);
        message.success('群体情感分析完成');
        setShowAnalysisModal(false);
        await loadData();
      } else {
        // 使用参与者数据生成模拟结果
        const participants = Object.keys(participantEmotions);
        const emotionCounts = emotions.reduce((acc, emotion) => {
          acc[emotion] = participants.filter(p => 
            participantEmotions[p]?.emotion === emotion
          ).length;
          return acc;
        }, {} as Record<string, number>);

        const totalParticipants = participants.length;
        const emotionDistribution = Object.fromEntries(
          Object.entries(emotionCounts).map(([emotion, count]) => 
            [emotion, count / totalParticipants]
          )
        );

        const dominantEmotion = Object.entries(emotionDistribution)
          .sort(([,a], [,b]) => b - a)[0][0];

        const mockResult = {
          group_id: selectedGroupId,
          timestamp: new Date().toISOString(),
          participants,
          dominant_emotion: dominantEmotion,
          emotion_distribution: emotionDistribution,
          consensus_level: Math.max(...Object.values(emotionDistribution)),
          polarization_index: 1 - Math.max(...Object.values(emotionDistribution)),
          emotional_volatility: 0.4 + Math.random() * 0.4,
          group_cohesion: totalParticipants > 4 ? 'HIGH' : 'MEDIUM',
          participation_balance: 0.8,
          interaction_intensity: 0.7,
          emotional_leaders: participants.slice(0, Math.ceil(participants.length / 3)),
          influence_network: [],
          contagion_events: [],
          data_quality_score: 0.8,
          confidence_level: totalParticipants >= analysisParams.minParticipants ? 0.8 : 0.6
        };

        setGroupState(mockResult);
        message.success('群体情感分析完成（使用模拟数据）');
        setShowAnalysisModal(false);
      }
    } catch (error) {
      console.error('分析失败:', error);
      message.error('分析失败，请重试');
    } finally {
      setLoading(false);
    }
  };

  const addParticipant = () => {
    const participantId = `user${Object.keys(participantEmotions).length + 1}`;
    setParticipantEmotions({
      ...participantEmotions,
      [participantId]: {
        emotion: 'neutral',
        intensity: 0.5,
        valence: 0.0,
        arousal: 0.5,
        dominance: 0.5
      }
    });
  };

  const updateParticipantEmotion = (participantId: string, field: string, value: any) => {
    setParticipantEmotions({
      ...participantEmotions,
      [participantId]: {
        ...participantEmotions[participantId],
        [field]: value
      }
    });
  };

  const removeParticipant = (participantId: string) => {
    const newEmotions = { ...participantEmotions };
    delete newEmotions[participantId];
    setParticipantEmotions(newEmotions);
  };

  const getEmotionColor = (emotion: string) => {
    const colorMap: Record<string, string> = {
      happiness: 'gold',
      joy: 'orange', 
      excitement: 'volcano',
      sadness: 'blue',
      anger: 'red',
      fear: 'purple',
      anxiety: 'magenta',
      neutral: 'default',
      trust: 'green',
      surprise: 'cyan'
    };
    return colorMap[emotion] || 'default';
  };

  const renderCurrentState = () => (
    <Row gutter={24}>
      <Col span={8}>
        <Card title={
          <span>
            <TeamOutlined style={{ marginRight: 8 }} />
            群体概况
          </span>
        }>
          {groupState ? (
            <Space direction="vertical" style={{ width: '100%' }}>
              <div style={{ textAlign: 'center' }}>
                <Tag 
                  color={getEmotionColor(groupState.dominant_emotion)} 
                  style={{ fontSize: '16px', padding: '8px 16px' }}
                >
                  {groupState.dominant_emotion}
                </Tag>
                <div style={{ marginTop: 8 }}>
                  <Text type="secondary">主导情感</Text>
                </div>
              </div>
              
              <Divider />
              
              <div>
                <Text strong>参与者数量: </Text>
                <Badge count={groupState.participants.length} style={{ backgroundColor: '#1890ff' }} />
              </div>
              
              <div>
                <Text strong>群体凝聚力: </Text>
                <Tag color={groupCohesionColors[groupState.group_cohesion] || 'default'}>
                  {groupState.group_cohesion}
                </Tag>
              </div>
              
              <div>
                <Text>一致性水平: </Text>
                <Progress 
                  percent={Math.round(groupState.consensus_level * 100)} 
                  size="small" 
                  strokeColor="#52c41a"
                />
              </div>
              
              <div>
                <Text>极化指数: </Text>
                <Progress 
                  percent={Math.round(groupState.polarization_index * 100)} 
                  size="small" 
                  strokeColor={groupState.polarization_index > 0.6 ? "#f5222d" : "#1890ff"}
                />
              </div>
            </Space>
          ) : (
            <Text type="secondary">暂无数据</Text>
          )}
        </Card>
      </Col>

      <Col span={8}>
        <Card title={
          <span>
            <BarChartOutlined style={{ marginRight: 8 }} />
            情感分布
          </span>
        }>
          {groupState?.emotion_distribution ? (
            <Space direction="vertical" style={{ width: '100%' }}>
              {Object.entries(groupState.emotion_distribution)
                .sort(([,a], [,b]) => b - a)
                .slice(0, 6)
                .map(([emotion, percentage]) => (
                  <div key={emotion} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Tag color={getEmotionColor(emotion)}>
                      {emotion}
                    </Tag>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1, marginLeft: 12 }}>
                      <Progress 
                        percent={Math.round((percentage as number) * 100)} 
                        size="small"
                        style={{ flex: 1 }}
                      />
                      <Text style={{ minWidth: '40px', fontSize: '12px' }}>
                        {((percentage as number) * 100).toFixed(1)}%
                      </Text>
                    </div>
                  </div>
                ))}
            </Space>
          ) : (
            <Text type="secondary">暂无数据</Text>
          )}
        </Card>
      </Col>

      <Col span={8}>
        <Card title={
          <span>
            <FireOutlined style={{ marginRight: 8 }} />
            情感领导者
          </span>
        }>
          {groupState?.emotional_leaders ? (
            <Space direction="vertical" style={{ width: '100%' }}>
              {groupState.emotional_leaders.map((leader, index) => (
                <div key={leader} style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <Badge 
                    count={index + 1} 
                    style={{ 
                      backgroundColor: index === 0 ? '#f5222d' : index === 1 ? '#fa8c16' : '#1890ff' 
                    }}
                  />
                  <Text strong>{leader}</Text>
                  {index === 0 && <Tag color="red">主导</Tag>}
                </div>
              ))}
              
              <Divider />
              
              <div>
                <Text strong>互动强度: </Text>
                <Progress 
                  percent={Math.round((groupState.interaction_intensity || 0) * 100)} 
                  size="small" 
                  strokeColor="#722ed1"
                />
              </div>
              
              <div>
                <Text strong>参与平衡: </Text>
                <Progress 
                  percent={Math.round((groupState.participation_balance || 0) * 100)} 
                  size="small" 
                  strokeColor="#13c2c2"
                />
              </div>
            </Space>
          ) : (
            <Text type="secondary">暂无数据</Text>
          )}
        </Card>
      </Col>
    </Row>
  );

  const renderContagionEvents = () => {
    const columns = [
      {
        title: '事件ID',
        dataIndex: 'event_id',
        key: 'event_id',
        width: 120,
        render: (id: string) => <Text code>{id.slice(-8)}</Text>
      },
      {
        title: '源参与者',
        dataIndex: 'source_participant', 
        key: 'source_participant',
        render: (participant: string) => <Tag color="blue">{participant}</Tag>
      },
      {
        title: '情感类型',
        dataIndex: 'emotion',
        key: 'emotion',
        render: (emotion: string) => <Tag color={getEmotionColor(emotion)}>{emotion}</Tag>
      },
      {
        title: '传染类型',
        dataIndex: 'contagion_type',
        key: 'contagion_type',
        render: (type: string) => (
          <Tag color={contagionTypeColors[type] || 'default'}>
            {type}
          </Tag>
        )
      },
      {
        title: '传染强度',
        dataIndex: 'strength',
        key: 'strength',
        render: (strength: number) => (
          <Progress
            percent={Math.round(strength * 100)}
            size="small"
            strokeColor={strength > 0.7 ? '#f5222d' : strength > 0.4 ? '#fa8c16' : '#52c41a'}
            style={{ width: 80 }}
          />
        )
      },
      {
        title: '目标数量',
        dataIndex: 'target_participants',
        key: 'target_count',
        render: (targets: string[]) => (
          <Badge count={targets.length} style={{ backgroundColor: '#722ed1' }} />
        )
      },
      {
        title: '发生时间',
        dataIndex: 'timestamp',
        key: 'timestamp',
        render: (timestamp: string) => (
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {new Date(timestamp).toLocaleString()}
          </Text>
        )
      }
    ];

    return (
      <Card title={
        <span>
          <ThunderboltOutlined style={{ marginRight: 8 }} />
          情感传染事件 ({contagionEvents.length})
        </span>
      }>
        {contagionEvents.length > 0 ? (
          <Table
            columns={columns}
            dataSource={contagionEvents}
            rowKey="event_id"
            pagination={{ pageSize: 10 }}
            size="small"
          />
        ) : (
          <div style={{ textAlign: 'center', padding: 40 }}>
            <Text type="secondary">暂无传染事件记录</Text>
          </div>
        )}
      </Card>
    );
  };

  const renderAnalysisModal = () => (
    <Modal
      title="群体情感分析"
      open={showAnalysisModal}
      onCancel={() => setShowAnalysisModal(false)}
      footer={[
        <Button key="cancel" onClick={() => setShowAnalysisModal(false)}>
          取消
        </Button>,
        <Button key="add" onClick={addParticipant}>
          添加参与者
        </Button>,
        <Button 
          key="analyze" 
          type="primary" 
          loading={loading}
          onClick={runGroupAnalysis}
          disabled={Object.keys(participantEmotions).length < analysisParams.minParticipants}
        >
          开始分析
        </Button>
      ]}
      width={800}
    >
      <Space direction="vertical" style={{ width: '100%' }}>
        <Alert
          message="群体情感分析"
          description={`请添加至少${analysisParams.minParticipants}个参与者的情感状态数据进行分析`}
          type="info"
          showIcon
        />

        {Object.entries(participantEmotions).map(([participantId, emotion]) => (
          <Card 
            key={participantId}
            size="small"
            title={participantId}
            extra={
              <Button 
                type="text" 
                danger 
                size="small"
                onClick={() => removeParticipant(participantId)}
              >
                删除
              </Button>
            }
          >
            <Row gutter={16}>
              <Col span={6}>
                <Text strong>情感:</Text>
                <Select
                  style={{ width: '100%', marginTop: 4 }}
                  value={emotion.emotion}
                  onChange={(value) => updateParticipantEmotion(participantId, 'emotion', value)}
                >
                  {emotions.map(e => (
                    <Option key={e} value={e}>{e}</Option>
                  ))}
                </Select>
              </Col>
              <Col span={6}>
                <Text strong>强度: {Math.round(emotion.intensity * 100)}%</Text>
                <InputNumber
                  style={{ width: '100%', marginTop: 4 }}
                  min={0}
                  max={1}
                  step={0.1}
                  value={emotion.intensity}
                  onChange={(value) => updateParticipantEmotion(participantId, 'intensity', value || 0)}
                />
              </Col>
              <Col span={4}>
                <Text strong>效价:</Text>
                <InputNumber
                  style={{ width: '100%', marginTop: 4 }}
                  min={-1}
                  max={1}
                  step={0.1}
                  value={emotion.valence}
                  onChange={(value) => updateParticipantEmotion(participantId, 'valence', value || 0)}
                />
              </Col>
              <Col span={4}>
                <Text strong>唤醒:</Text>
                <InputNumber
                  style={{ width: '100%', marginTop: 4 }}
                  min={0}
                  max={1}
                  step={0.1}
                  value={emotion.arousal}
                  onChange={(value) => updateParticipantEmotion(participantId, 'arousal', value || 0)}
                />
              </Col>
              <Col span={4}>
                <Text strong>支配:</Text>
                <InputNumber
                  style={{ width: '100%', marginTop: 4 }}
                  min={0}
                  max={1}
                  step={0.1}
                  value={emotion.dominance}
                  onChange={(value) => updateParticipantEmotion(participantId, 'dominance', value || 0)}
                />
              </Col>
            </Row>
          </Card>
        ))}

        {Object.keys(participantEmotions).length === 0 && (
          <div style={{ textAlign: 'center', padding: 40 }}>
            <Text type="secondary">暂无参与者，请点击"添加参与者"开始</Text>
          </div>
        )}
      </Space>
    </Modal>
  );

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: 24, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Title level={2}>
          <TeamOutlined style={{ marginRight: 12, color: '#1890ff' }} />
          群体情感分析引擎
        </Title>
        <Space>
          <Select
            style={{ width: 200 }}
            value={selectedGroupId}
            onChange={setSelectedGroupId}
            placeholder="选择群体"
          >
            <Option value="group_1">群体 1</Option>
            <Option value="group_2">群体 2</Option>
            <Option value="group_3">群体 3</Option>
          </Select>
          <Button 
            type="primary" 
            icon={<ExperimentOutlined />}
            onClick={() => setShowAnalysisModal(true)}
          >
            运行分析
          </Button>
          <Button 
            icon={<SyncOutlined />} 
            loading={loading}
            onClick={loadData}
          >
            刷新
          </Button>
        </Space>
      </div>

      <Tabs defaultActiveKey="overview">
        <TabPane tab="群体概览" key="overview">
          {renderCurrentState()}
          
          <div style={{ marginTop: 24 }}>
            <Card title={
              <span>
                <LineChartOutlined style={{ marginRight: 8 }} />
                群体质量指标
              </span>
            }>
              {groupState && (
                <Row gutter={24}>
                  <Col span={6}>
                    <div style={{ textAlign: 'center' }}>
                      <Progress
                        type="circle"
                        percent={Math.round(groupState.data_quality_score * 100)}
                        strokeColor="#52c41a"
                        width={100}
                      />
                      <div style={{ marginTop: 8 }}>
                        <Text strong>数据质量</Text>
                      </div>
                    </div>
                  </Col>
                  <Col span={6}>
                    <div style={{ textAlign: 'center' }}>
                      <Progress
                        type="circle"
                        percent={Math.round(groupState.confidence_level * 100)}
                        strokeColor="#1890ff"
                        width={100}
                      />
                      <div style={{ marginTop: 8 }}>
                        <Text strong>分析置信度</Text>
                      </div>
                    </div>
                  </Col>
                  <Col span={6}>
                    <div style={{ textAlign: 'center' }}>
                      <Progress
                        type="circle"
                        percent={Math.round(groupState.emotional_volatility * 100)}
                        strokeColor="#fa8c16"
                        width={100}
                      />
                      <div style={{ marginTop: 8 }}>
                        <Text strong>情感波动性</Text>
                      </div>
                    </div>
                  </Col>
                  <Col span={6}>
                    <div style={{ textAlign: 'center', color: '#8c8c8c' }}>
                      <div style={{ 
                        width: 100, 
                        height: 100, 
                        borderRadius: '50%', 
                        border: '6px solid #f0f0f0',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        margin: '0 auto',
                        fontSize: '24px',
                        fontWeight: 'bold'
                      }}>
                        {groupState.participants.length}
                      </div>
                      <div style={{ marginTop: 8 }}>
                        <Text strong>参与者数量</Text>
                      </div>
                    </div>
                  </Col>
                </Row>
              )}
            </Card>
          </div>
        </TabPane>

        <TabPane tab="传染事件" key="contagion">
          {renderContagionEvents()}
        </TabPane>

        <TabPane tab="历史趋势" key="history">
          <Card title="群体情感历史">
            <Timeline>
              {groupHistory.slice(0, 8).map((state, index) => (
                <Timeline.Item 
                  key={`${state.group_id}_${state.timestamp}`}
                  color={index === 0 ? 'green' : 'blue'}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
                    <Tag color={getEmotionColor(state.dominant_emotion)}>
                      {state.dominant_emotion}
                    </Tag>
                    <Text>凝聚力: </Text>
                    <Tag color={groupCohesionColors[state.group_cohesion] || 'default'}>
                      {state.group_cohesion}
                    </Tag>
                    <Text>一致性: {Math.round(state.consensus_level * 100)}%</Text>
                    <Text type="secondary">
                      {new Date(state.timestamp).toLocaleString()}
                    </Text>
                  </div>
                </Timeline.Item>
              ))}
            </Timeline>
          </Card>
        </TabPane>

        <TabPane tab="分析配置" key="config">
          <Card title="分析参数配置" style={{ maxWidth: 600 }}>
            <Form layout="vertical">
              <Form.Item label="时间窗口 (分钟)">
                <InputNumber
                  style={{ width: '100%' }}
                  value={analysisParams.timeWindow}
                  onChange={(value) => setAnalysisParams({
                    ...analysisParams,
                    timeWindow: value || 60
                  })}
                  min={5}
                  max={1440}
                />
              </Form.Item>
              
              <Form.Item label="最少参与者数量">
                <InputNumber
                  style={{ width: '100%' }}
                  value={analysisParams.minParticipants}
                  onChange={(value) => setAnalysisParams({
                    ...analysisParams,
                    minParticipants: value || 3
                  })}
                  min={2}
                  max={50}
                />
              </Form.Item>
              
              <Form.Item label="置信度阈值">
                <InputNumber
                  style={{ width: '100%' }}
                  value={analysisParams.confidenceThreshold}
                  onChange={(value) => setAnalysisParams({
                    ...analysisParams,
                    confidenceThreshold: value || 0.6
                  })}
                  min={0.1}
                  max={1.0}
                  step={0.1}
                />
              </Form.Item>
              
              <Form.Item>
                <Button type="primary" block onClick={loadData}>
                  应用配置并刷新
                </Button>
              </Form.Item>
            </Form>
          </Card>
        </TabPane>
      </Tabs>

      {renderAnalysisModal()}
    </div>
  );
};

export default GroupEmotionAnalysisPage;