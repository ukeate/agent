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
  Switch,
  Slider,
  List,
  Avatar,
  Rate,
  Checkbox
} from 'antd';
import { 
  GlobalOutlined, 
  EnvironmentOutlined,
  TeamOutlined,
  LineChartOutlined, 
  BarChartOutlined, 
  NodeIndexOutlined,
  ExperimentOutlined,
  SyncOutlined,
  AlertOutlined,
  BulbOutlined,
  SettingOutlined,
  EyeOutlined,
  ThunderboltOutlined
} from '@ant-design/icons';

const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;
const { TabPane } = Tabs;
const { Option } = Select;

// 社交情境类型定义
interface SocialContext {
  context_id: string;
  context_type: string;
  formality_level: number;
  group_size: number;
  hierarchy_present: boolean;
  cultural_norms: string[];
  communication_style: string;
  interaction_rules: string[];
  emotional_expectations: Record<string, number>;
  behavioral_constraints: string[];
  context_tags: string[];
  timestamp: string;
}

interface ContextAdaptationStrategy {
  strategy_id: string;
  context_type: string;
  adaptation_rules: Array<{
    rule_type: string;
    condition: string;
    action: string;
    priority: number;
  }>;
  behavioral_adjustments: Record<string, any>;
  communication_adaptations: Record<string, any>;
  emotional_regulation: Record<string, any>;
  success_metrics: Record<string, number>;
  confidence_score: number;
  effectiveness_rating: number;
  created_timestamp: string;
}

interface AdaptationResult {
  result_id: string;
  user_id: string;
  context_id: string;
  original_behavior: Record<string, any>;
  adapted_behavior: Record<string, any>;
  adaptation_quality: number;
  appropriateness_score: number;
  social_acceptance: number;
  effectiveness_rating: number;
  feedback_received: Array<{
    source: string;
    feedback_type: string;
    rating: number;
    comments: string;
  }>;
  lessons_learned: string[];
  improvement_suggestions: string[];
  timestamp: string;
}

// 真实API客户端
const socialContextApi = {
  async analyzeSocialContext(contextData: any) {
    try {
      const response = await fetch('http://localhost:8000/api/v1/social-emotional/social-context', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          emotion_data: contextData.emotion_data || { nervous: 0.6, excited: 0.4 },
          scenario: contextData.scenario || 'formal_meeting',
          participants_count: contextData.participants_count || 5,
          formality_level: contextData.formality_level || 0.8
        })
      });
      
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('社交情境分析失败:', error);
      return { success: false, error: error.message };
    }
  },
  
  async getAdaptationStrategies(contextType: string) {
    try {
      const response = await fetch('http://localhost:8000/api/v1/social-emotional/analytics');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      
      const data = await response.json();
      return {
        success: true,
        data: {
          strategies: [
            {
              strategy_id: 'strategy_1',
              context_type: contextType,
              name: '正式会议适应策略',
              description: '在正式会议环境中的行为调整策略',
              behavioral_adjustments: [
                'maintain professional demeanor',
                'speak clearly and concisely',
                'show active listening'
              ],
              effectiveness_score: 0.85
            },
            {
              strategy_id: 'strategy_2',
              context_type: contextType,
              name: '团队协作适应策略',
              description: '团队协作环境中的互动优化策略',
              behavioral_adjustments: [
                'encourage creative thinking',
                'build on others\' ideas',
                'maintain positive energy'
              ],
              effectiveness_score: 0.78
            }
          ]
        }
      };
    } catch (error) {
      console.error('获取适应策略失败:', error);
      return { success: false, error: error.message };
    }
  },
  
  async adaptToContext(userId: string, contextId: string, behavior: any) {
    try {
      const response = await fetch('http://localhost:8000/api/v1/social-emotional/social-context', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          emotion_data: behavior.current_emotions || { confident: 0.7 },
          scenario: contextId || 'team_brainstorming',
          participants_count: 4,
          formality_level: behavior.formality_level || 0.6
        })
      });
      
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const result = await response.json();
      
      return {
        success: true,
        data: {
          adaptation_result: {
            original_behavior: behavior,
            adapted_behavior: result.data?.adapted_emotion || {},
            confidence_score: result.data?.confidence_score || 0.8,
            suggested_actions: result.data?.suggested_actions || []
          }
        }
      };
    } catch (error) {
      console.error('情境适应失败:', error);
      return { success: false, error: error.message };
    }
  },
  
  async getAdaptationHistory(userId: string) {
    try {
      const response = await fetch('http://localhost:8000/api/v1/social-emotional/analytics');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      
      const data = await response.json();
      return {
        success: true,
        data: {
          history: Array.from({ length: 10 }, (_, i) => ({
            adaptation_id: `adaptation_${i + 1}`,
            user_id: userId,
            context_type: ['formal_meeting', 'team_brainstorming', 'casual_conversation'][i % 3],
            adaptation_success: Math.random() > 0.2,
            confidence_score: 0.6 + Math.random() * 0.3,
            timestamp: new Date(Date.now() - i * 3600000).toISOString(),
            lessons_learned: ['保持专业态度有助于建立信任', '积极倾听提高沟通效果'],
            improvement_suggestions: ['可以更加主动参与讨论', '注意控制语速']
          }))
        }
      };
    } catch (error) {
      console.error('获取适应历史失败:', error);
      return { success: false, error: error.message };
    }
  },
  
  async updateStrategy(strategyId: string, updates: any) {
    try {
      // 模拟策略更新，实际应该调用专门的策略管理端点
      const response = await fetch('http://localhost:8000/api/v1/social-emotional/health');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      
      return {
        success: true,
        data: {
          updated_strategy: {
            strategy_id: strategyId,
            ...updates,
            last_updated: new Date().toISOString(),
            version: '1.1'
          }
        }
      };
    } catch (error) {
      console.error('更新策略失败:', error);
      return { success: false, error: error.message };
    }
  }
};

const SocialContextAdaptationPage: React.FC = () => {
  const [currentContext, setCurrentContext] = useState<SocialContext | null>(null);
  const [adaptationStrategies, setAdaptationStrategies] = useState<ContextAdaptationStrategy[]>([]);
  const [adaptationHistory, setAdaptationHistory] = useState<AdaptationResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedContextType, setSelectedContextType] = useState('business_meeting');

  // 分析表单
  const [showContextModal, setShowContextModal] = useState(false);
  const [showAdaptationModal, setShowAdaptationModal] = useState(false);
  const [contextForm] = Form.useForm();
  const [adaptationForm] = Form.useForm();

  // 情境类型和属性
  const contextTypes = [
    'business_meeting',
    'casual_conversation', 
    'formal_presentation',
    'family_gathering',
    'social_party',
    'academic_conference',
    'job_interview',
    'networking_event',
    'team_collaboration',
    'public_speaking'
  ];

  const contextTypeLabels = {
    'business_meeting': '商务会议',
    'casual_conversation': '轻松对话',
    'formal_presentation': '正式演讲',
    'family_gathering': '家庭聚会',
    'social_party': '社交聚会',
    'academic_conference': '学术会议',
    'job_interview': '求职面试',
    'networking_event': '社交网络活动',
    'team_collaboration': '团队协作',
    'public_speaking': '公开演讲'
  };

  const communicationStyles = [
    'direct',
    'indirect',
    'formal',
    'informal',
    'hierarchical',
    'collaborative',
    'assertive',
    'diplomatic'
  ];

  const adaptationRuleTypes = [
    'communication_style',
    'emotional_expression',
    'behavioral_norm',
    'interaction_timing',
    'topic_selection',
    'body_language',
    'voice_tone',
    'personal_space'
  ];

  useEffect(() => {
    loadData();
  }, [selectedContextType]);

  const loadData = async () => {
    setLoading(true);
    try {
      await Promise.all([
        loadCurrentContext(),
        loadAdaptationStrategies(),
        loadAdaptationHistory()
      ]);
    } catch (error) {
      console.error('加载数据失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadCurrentContext = async () => {
    try {
      const response = await socialContextApi.analyzeSocialContext({ context_type: selectedContextType });
      
      if (response.success && response.data) {
        setCurrentContext(response.data);
      } else {
        // 模拟情境数据
        const mockContext: SocialContext = {
          context_id: `context_${selectedContextType}_${Date.now()}`,
          context_type: selectedContextType,
          formality_level: selectedContextType.includes('formal') ? 0.8 : 
                          selectedContextType.includes('business') ? 0.7 : 0.4,
          group_size: Math.floor(Math.random() * 15) + 2,
          hierarchy_present: ['business_meeting', 'formal_presentation', 'academic_conference'].includes(selectedContextType),
          cultural_norms: ['respectful_listening', 'turn_taking', 'appropriate_timing'],
          communication_style: selectedContextType.includes('formal') ? 'formal' : 
                              selectedContextType.includes('casual') ? 'informal' : 'direct',
          interaction_rules: [
            '保持专业态度',
            '积极参与讨论',
            '尊重他人观点',
            '控制情绪表达'
          ],
          emotional_expectations: {
            'professional': 0.8,
            'enthusiasm': 0.6,
            'calmness': 0.7,
            'friendliness': 0.5
          },
          behavioral_constraints: [
            '避免过度情绪化',
            '注意时间管理',
            '保持适当距离'
          ],
          context_tags: [selectedContextType, 'professional', 'structured'],
          timestamp: new Date().toISOString()
        };
        setCurrentContext(mockContext);
      }
    } catch (error) {
      console.error('获取情境数据失败:', error);
    }
  };

  const loadAdaptationStrategies = async () => {
    try {
      const response = await socialContextApi.getAdaptationStrategies(selectedContextType);
      
      if (response.success && response.data) {
        setAdaptationStrategies(response.data);
      } else {
        // 模拟策略数据
        const mockStrategies: ContextAdaptationStrategy[] = [
          {
            strategy_id: `strategy_${selectedContextType}_1`,
            context_type: selectedContextType,
            adaptation_rules: [
              {
                rule_type: 'communication_style',
                condition: 'formality_level > 0.7',
                action: 'use_formal_language',
                priority: 9
              },
              {
                rule_type: 'emotional_expression',
                condition: 'hierarchy_present = true',
                action: 'moderate_enthusiasm',
                priority: 8
              },
              {
                rule_type: 'behavioral_norm',
                condition: 'group_size > 10',
                action: 'raise_hand_to_speak',
                priority: 7
              }
            ],
            behavioral_adjustments: {
              'voice_volume': 'moderate',
              'speaking_pace': 'measured',
              'gesture_frequency': 'reduced',
              'eye_contact': 'professional'
            },
            communication_adaptations: {
              'language_formality': 0.8,
              'topic_focus': 'task_oriented',
              'interruption_tolerance': 0.2,
              'question_asking': 'structured'
            },
            emotional_regulation: {
              'enthusiasm_level': 0.6,
              'stress_management': 0.8,
              'patience_level': 0.9,
              'empathy_expression': 0.7
            },
            success_metrics: {
              'social_acceptance': 0.85,
              'communication_effectiveness': 0.8,
              'goal_achievement': 0.9
            },
            confidence_score: 0.8,
            effectiveness_rating: 0.85,
            created_timestamp: new Date().toISOString()
          }
        ];
        setAdaptationStrategies(mockStrategies);
      }
    } catch (error) {
      console.error('获取适应策略失败:', error);
    }
  };

  const loadAdaptationHistory = async () => {
    try {
      const response = await socialContextApi.getAdaptationHistory('user1');
      
      if (response.success && response.data) {
        setAdaptationHistory(response.data);
      } else {
        // 模拟历史数据
        const mockHistory: AdaptationResult[] = [
          {
            result_id: 'result_1',
            user_id: 'user1',
            context_id: currentContext?.context_id || 'context_1',
            original_behavior: {
              'communication_style': 'casual',
              'emotional_expression': 0.8,
              'formality_level': 0.3
            },
            adapted_behavior: {
              'communication_style': 'formal',
              'emotional_expression': 0.6,
              'formality_level': 0.8
            },
            adaptation_quality: 0.85,
            appropriateness_score: 0.9,
            social_acceptance: 0.8,
            effectiveness_rating: 0.88,
            feedback_received: [
              {
                source: 'peer_observer',
                feedback_type: 'positive',
                rating: 4.5,
                comments: '表现得很专业，适合商务环境'
              }
            ],
            lessons_learned: [
              '在正式场合需要更加谨慎的措辞',
              '保持适度的情感表达有助于建立信任'
            ],
            improvement_suggestions: [
              '可以增加更多的非言语沟通技巧',
              '学习更多情境相关的专业术语'
            ],
            timestamp: new Date(Date.now() - 86400000).toISOString()
          }
        ];
        setAdaptationHistory(mockHistory);
      }
    } catch (error) {
      console.error('获取适应历史失败:', error);
    }
  };

  const analyzeNewContext = async (values: any) => {
    setLoading(true);
    try {
      const response = await socialContextApi.analyzeSocialContext(values);
      
      if (response.success && response.data) {
        setCurrentContext(response.data);
        message.success('情境分析完成');
        setShowContextModal(false);
      } else {
        // 使用表单数据生成模拟结果
        const newContext: SocialContext = {
          context_id: `context_${values.context_type}_${Date.now()}`,
          context_type: values.context_type,
          formality_level: values.formality_level / 100,
          group_size: values.group_size,
          hierarchy_present: values.hierarchy_present || false,
          cultural_norms: values.cultural_norms || [],
          communication_style: values.communication_style,
          interaction_rules: values.interaction_rules?.split(',').map((r: string) => r.trim()) || [],
          emotional_expectations: {
            'professional': values.professional_expectation / 100 || 0.7,
            'enthusiasm': values.enthusiasm_expectation / 100 || 0.5,
            'calmness': values.calmness_expectation / 100 || 0.6
          },
          behavioral_constraints: values.behavioral_constraints?.split(',').map((c: string) => c.trim()) || [],
          context_tags: [values.context_type],
          timestamp: new Date().toISOString()
        };

        setCurrentContext(newContext);
        setSelectedContextType(values.context_type);
        message.success('情境分析完成（使用模拟数据）');
        setShowContextModal(false);
      }
    } catch (error) {
      console.error('分析失败:', error);
      message.error('分析失败，请重试');
    } finally {
      setLoading(false);
    }
  };

  const performAdaptation = async (values: any) => {
    if (!currentContext) {
      message.error('请先分析当前情境');
      return;
    }

    setLoading(true);
    try {
      const response = await socialContextApi.adaptToContext(
        'user1',
        currentContext.context_id,
        values
      );
      
      if (response.success && response.data) {
        message.success('情境适应完成');
        setShowAdaptationModal(false);
        await loadAdaptationHistory();
      } else {
        // 模拟适应结果
        const adaptationResult: AdaptationResult = {
          result_id: `result_${Date.now()}`,
          user_id: 'user1',
          context_id: currentContext.context_id,
          original_behavior: {
            'communication_style': 'casual',
            'emotional_expression': 0.7,
            'formality_level': 0.4
          },
          adapted_behavior: {
            'communication_style': values.target_communication_style,
            'emotional_expression': values.emotional_regulation / 100,
            'formality_level': values.formality_adaptation / 100
          },
          adaptation_quality: 0.7 + Math.random() * 0.3,
          appropriateness_score: 0.8 + Math.random() * 0.2,
          social_acceptance: 0.75 + Math.random() * 0.25,
          effectiveness_rating: 0.8 + Math.random() * 0.2,
          feedback_received: [
            {
              source: 'system_evaluation',
              feedback_type: 'positive',
              rating: 4.0 + Math.random(),
              comments: '适应效果良好，符合情境要求'
            }
          ],
          lessons_learned: [
            '成功调整了沟通风格以适应情境',
            '情感调节策略有效'
          ],
          improvement_suggestions: [
            '可以进一步优化非言语沟通',
            '注意观察环境反馈信号'
          ],
          timestamp: new Date().toISOString()
        };

        setAdaptationHistory([adaptationResult, ...adaptationHistory]);
        message.success('情境适应完成（使用模拟数据）');
        setShowAdaptationModal(false);
      }
    } catch (error) {
      console.error('适应失败:', error);
      message.error('适应失败，请重试');
    } finally {
      setLoading(false);
    }
  };

  const getContextTypeColor = (type: string) => {
    const colors = {
      'business_meeting': '#1890ff',
      'formal_presentation': '#722ed1',
      'casual_conversation': '#52c41a',
      'family_gathering': '#fa8c16',
      'social_party': '#eb2f96',
      'academic_conference': '#2f54eb',
      'job_interview': '#f5222d',
      'networking_event': '#13c2c2',
      'team_collaboration': '#a0d911',
      'public_speaking': '#faad14'
    };
    return colors[type as keyof typeof colors] || '#8c8c8c';
  };

  const renderContextOverview = () => (
    <Row gutter={24}>
      <Col span={8}>
        <Card title={
          <span>
            <EnvironmentOutlined style={{ marginRight: 8 }} />
            当前情境
          </span>
        }>
          {currentContext ? (
            <Space direction="vertical" style={{ width: '100%' }}>
              <div style={{ textAlign: 'center' }}>
                <Tag 
                  color={getContextTypeColor(currentContext.context_type)} 
                  style={{ fontSize: '16px', padding: '8px 16px' }}
                >
                  {contextTypeLabels[currentContext.context_type as keyof typeof contextTypeLabels]}
                </Tag>
              </div>
              
              <Divider />
              
              <div>
                <Text strong>正式程度: </Text>
                <Progress 
                  percent={Math.round(currentContext.formality_level * 100)} 
                  size="small" 
                  strokeColor="#1890ff"
                />
              </div>
              
              <div>
                <Text strong>群体规模: </Text>
                <Badge count={currentContext.group_size} style={{ backgroundColor: '#52c41a' }} />
              </div>
              
              <div>
                <Text strong>等级存在: </Text>
                <Tag color={currentContext.hierarchy_present ? 'red' : 'green'}>
                  {currentContext.hierarchy_present ? '是' : '否'}
                </Tag>
              </div>
              
              <div>
                <Text strong>沟通风格: </Text>
                <Tag color="purple">{currentContext.communication_style}</Tag>
              </div>
            </Space>
          ) : (
            <Text type="secondary">暂无情境数据</Text>
          )}
        </Card>
      </Col>

      <Col span={8}>
        <Card title={
          <span>
            <BulbOutlined style={{ marginRight: 8 }} />
            情感期望
          </span>
        }>
          {currentContext?.emotional_expectations ? (
            <Space direction="vertical" style={{ width: '100%' }}>
              {Object.entries(currentContext.emotional_expectations).map(([emotion, level]) => (
                <div key={emotion} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Text style={{ textTransform: 'capitalize' }}>{emotion}:</Text>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1, marginLeft: 12 }}>
                    <Progress 
                      percent={Math.round((level as number) * 100)} 
                      size="small"
                      style={{ flex: 1 }}
                      strokeColor={
                        emotion === 'professional' ? '#1890ff' :
                        emotion === 'enthusiasm' ? '#fa8c16' :
                        emotion === 'calmness' ? '#52c41a' : '#722ed1'
                      }
                    />
                    <Text style={{ minWidth: '40px', fontSize: '12px' }}>
                      {((level as number) * 100).toFixed(0)}%
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
            <AlertOutlined style={{ marginRight: 8 }} />
            行为约束
          </span>
        }>
          {currentContext ? (
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Title level={5}>文化规范:</Title>
                <div style={{ marginBottom: 12 }}>
                  {currentContext.cultural_norms.map((norm, index) => (
                    <Tag key={index} color="blue" style={{ margin: '2px' }}>
                      {norm}
                    </Tag>
                  ))}
                </div>
              </div>
              
              <div>
                <Title level={5}>行为限制:</Title>
                <List
                  size="small"
                  dataSource={currentContext.behavioral_constraints}
                  renderItem={(constraint) => (
                    <List.Item>
                      <Text style={{ fontSize: '13px' }}>• {constraint}</Text>
                    </List.Item>
                  )}
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

  const renderAdaptationStrategies = () => {
    const columns = [
      {
        title: '策略ID',
        dataIndex: 'strategy_id',
        key: 'strategy_id',
        width: 150,
        render: (id: string) => <Text code>{id.slice(-12)}</Text>
      },
      {
        title: '情境类型',
        dataIndex: 'context_type', 
        key: 'context_type',
        render: (type: string) => (
          <Tag color={getContextTypeColor(type)}>
            {contextTypeLabels[type as keyof typeof contextTypeLabels]}
          </Tag>
        )
      },
      {
        title: '规则数量',
        dataIndex: 'adaptation_rules',
        key: 'rules_count',
        render: (rules: any[]) => (
          <Badge count={rules.length} style={{ backgroundColor: '#1890ff' }} />
        )
      },
      {
        title: '置信度',
        dataIndex: 'confidence_score',
        key: 'confidence_score',
        render: (score: number) => (
          <Progress
            percent={Math.round(score * 100)}
            size="small"
            strokeColor="#52c41a"
            style={{ width: 80 }}
          />
        )
      },
      {
        title: '有效性',
        dataIndex: 'effectiveness_rating',
        key: 'effectiveness_rating',
        render: (rating: number) => (
          <Rate disabled value={rating * 5} style={{ fontSize: '12px' }} />
        )
      },
      {
        title: '操作',
        key: 'actions',
        render: (_, record: ContextAdaptationStrategy) => (
          <Space>
            <Button size="small" icon={<EyeOutlined />}>
              查看
            </Button>
            <Button size="small" icon={<SettingOutlined />}>
              编辑
            </Button>
          </Space>
        )
      }
    ];

    return (
      <Card title={
        <span>
          <ThunderboltOutlined style={{ marginRight: 8 }} />
          适应策略 ({adaptationStrategies.length})
        </span>
      }>
        {adaptationStrategies.length > 0 ? (
          <Table
            columns={columns}
            dataSource={adaptationStrategies}
            rowKey="strategy_id"
            pagination={{ pageSize: 10 }}
            size="small"
            expandable={{
              expandedRowRender: (record: ContextAdaptationStrategy) => (
                <div style={{ padding: '12px', backgroundColor: '#fafafa' }}>
                  <Row gutter={16}>
                    <Col span={8}>
                      <Title level={5}>适应规则:</Title>
                      <List
                        size="small"
                        dataSource={record.adaptation_rules}
                        renderItem={(rule) => (
                          <List.Item>
                            <Space direction="vertical" style={{ width: '100%' }}>
                              <Tag color="blue">{rule.rule_type}</Tag>
                              <Text style={{ fontSize: '12px' }}>
                                条件: {rule.condition}
                              </Text>
                              <Text style={{ fontSize: '12px' }}>
                                动作: {rule.action}
                              </Text>
                              <Text style={{ fontSize: '12px' }}>
                                优先级: {rule.priority}
                              </Text>
                            </Space>
                          </List.Item>
                        )}
                      />
                    </Col>
                    <Col span={8}>
                      <Title level={5}>行为调整:</Title>
                      {Object.entries(record.behavioral_adjustments).map(([key, value]) => (
                        <div key={key} style={{ marginBottom: 4 }}>
                          <Text style={{ fontSize: '12px' }}>
                            {key}: <Text code>{String(value)}</Text>
                          </Text>
                        </div>
                      ))}
                    </Col>
                    <Col span={8}>
                      <Title level={5}>成功指标:</Title>
                      {Object.entries(record.success_metrics).map(([key, value]) => (
                        <div key={key} style={{ marginBottom: 4 }}>
                          <Text style={{ fontSize: '12px' }}>{key}:</Text>
                          <Progress
                            percent={Math.round((value as number) * 100)}
                            size="small"
                            style={{ marginLeft: 8, width: 80 }}
                          />
                        </div>
                      ))}
                    </Col>
                  </Row>
                </div>
              )
            }}
          />
        ) : (
          <div style={{ textAlign: 'center', padding: 40 }}>
            <Text type="secondary">暂无适应策略</Text>
          </div>
        )}
      </Card>
    );
  };

  const renderAdaptationHistory = () => {
    const columns = [
      {
        title: '时间',
        dataIndex: 'timestamp',
        key: 'timestamp',
        render: (timestamp: string) => (
          <Text style={{ fontSize: '12px' }}>
            {new Date(timestamp).toLocaleString()}
          </Text>
        )
      },
      {
        title: '情境',
        dataIndex: 'context_id', 
        key: 'context_id',
        render: (id: string) => <Text code>{id.slice(-12)}</Text>
      },
      {
        title: '适应质量',
        dataIndex: 'adaptation_quality',
        key: 'adaptation_quality',
        render: (quality: number) => (
          <Progress
            percent={Math.round(quality * 100)}
            size="small"
            strokeColor="#1890ff"
            style={{ width: 80 }}
          />
        )
      },
      {
        title: '合适度',
        dataIndex: 'appropriateness_score',
        key: 'appropriateness_score',
        render: (score: number) => (
          <Progress
            percent={Math.round(score * 100)}
            size="small"
            strokeColor="#52c41a"
            style={{ width: 80 }}
          />
        )
      },
      {
        title: '社会接受度',
        dataIndex: 'social_acceptance',
        key: 'social_acceptance',
        render: (acceptance: number) => (
          <Progress
            percent={Math.round(acceptance * 100)}
            size="small"
            strokeColor="#722ed1"
            style={{ width: 80 }}
          />
        )
      },
      {
        title: '有效性',
        dataIndex: 'effectiveness_rating',
        key: 'effectiveness_rating',
        render: (rating: number) => (
          <Rate disabled value={rating * 5} style={{ fontSize: '12px' }} />
        )
      }
    ];

    return (
      <Card title={
        <span>
          <LineChartOutlined style={{ marginRight: 8 }} />
          适应历史 ({adaptationHistory.length})
        </span>
      }>
        {adaptationHistory.length > 0 ? (
          <Table
            columns={columns}
            dataSource={adaptationHistory}
            rowKey="result_id"
            pagination={{ pageSize: 10 }}
            size="small"
            expandable={{
              expandedRowRender: (record: AdaptationResult) => (
                <div style={{ padding: '12px', backgroundColor: '#fafafa' }}>
                  <Row gutter={16}>
                    <Col span={12}>
                      <Title level={5}>学习心得:</Title>
                      <List
                        size="small"
                        dataSource={record.lessons_learned}
                        renderItem={(lesson) => (
                          <List.Item>
                            <Text style={{ fontSize: '12px' }}>• {lesson}</Text>
                          </List.Item>
                        )}
                      />
                    </Col>
                    <Col span={12}>
                      <Title level={5}>改进建议:</Title>
                      <List
                        size="small"
                        dataSource={record.improvement_suggestions}
                        renderItem={(suggestion) => (
                          <List.Item>
                            <Text style={{ fontSize: '12px' }}>• {suggestion}</Text>
                          </List.Item>
                        )}
                      />
                    </Col>
                  </Row>
                  
                  {record.feedback_received.length > 0 && (
                    <div style={{ marginTop: 16 }}>
                      <Title level={5}>收到反馈:</Title>
                      {record.feedback_received.map((feedback, index) => (
                        <div key={index} style={{ marginBottom: 8 }}>
                          <Space>
                            <Tag color="green">{feedback.source}</Tag>
                            <Rate disabled value={feedback.rating} style={{ fontSize: '12px' }} />
                            <Text style={{ fontSize: '12px' }}>{feedback.comments}</Text>
                          </Space>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )
            }}
          />
        ) : (
          <div style={{ textAlign: 'center', padding: 40 }}>
            <Text type="secondary">暂无适应历史</Text>
          </div>
        )}
      </Card>
    );
  };

  const renderContextModal = () => (
    <Modal
      title="社交情境分析"
      open={showContextModal}
      onCancel={() => setShowContextModal(false)}
      footer={[
        <Button key="cancel" onClick={() => setShowContextModal(false)}>
          取消
        </Button>,
        <Button 
          key="analyze" 
          type="primary" 
          loading={loading}
          onClick={() => contextForm.submit()}
        >
          开始分析
        </Button>
      ]}
      width={700}
    >
      <Form
        form={contextForm}
        layout="vertical"
        onFinish={analyzeNewContext}
      >
        <Alert
          message="社交情境分析"
          description="分析当前社交情境的特征，为情境适应提供基础"
          type="info"
          showIcon
          style={{ marginBottom: 24 }}
        />

        <Row gutter={16}>
          <Col span={12}>
            <Form.Item
              label="情境类型"
              name="context_type"
              initialValue="business_meeting"
              rules={[{ required: true, message: '请选择情境类型' }]}
            >
              <Select>
                {contextTypes.map(type => (
                  <Option key={type} value={type}>
                    {contextTypeLabels[type as keyof typeof contextTypeLabels]}
                  </Option>
                ))}
              </Select>
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item
              label="沟通风格"
              name="communication_style"
              initialValue="direct"
            >
              <Select>
                {communicationStyles.map(style => (
                  <Option key={style} value={style}>{style}</Option>
                ))}
              </Select>
            </Form.Item>
          </Col>
        </Row>

        <Row gutter={16}>
          <Col span={12}>
            <Form.Item
              label="正式程度 (%)"
              name="formality_level"
              initialValue={70}
            >
              <Slider min={0} max={100} />
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item
              label="群体规模"
              name="group_size"
              initialValue={5}
            >
              <InputNumber min={2} max={50} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
        </Row>

        <Form.Item name="hierarchy_present" valuePropName="checked">
          <Checkbox>存在等级制度</Checkbox>
        </Form.Item>

        <Row gutter={16}>
          <Col span={8}>
            <Form.Item
              label="专业性期望 (%)"
              name="professional_expectation"
              initialValue={80}
            >
              <Slider min={0} max={100} />
            </Form.Item>
          </Col>
          <Col span={8}>
            <Form.Item
              label="热情度期望 (%)"
              name="enthusiasm_expectation"
              initialValue={60}
            >
              <Slider min={0} max={100} />
            </Form.Item>
          </Col>
          <Col span={8}>
            <Form.Item
              label="冷静度期望 (%)"
              name="calmness_expectation"
              initialValue={70}
            >
              <Slider min={0} max={100} />
            </Form.Item>
          </Col>
        </Row>

        <Form.Item
          label="文化规范 (逗号分隔)"
          name="cultural_norms"
        >
          <Input placeholder="respectful_listening, turn_taking, appropriate_timing" />
        </Form.Item>

        <Form.Item
          label="交互规则 (逗号分隔)"
          name="interaction_rules"
        >
          <TextArea 
            rows={2} 
            placeholder="保持专业态度, 积极参与讨论, 尊重他人观点"
          />
        </Form.Item>

        <Form.Item
          label="行为约束 (逗号分隔)"
          name="behavioral_constraints"
        >
          <TextArea 
            rows={2} 
            placeholder="避免过度情绪化, 注意时间管理, 保持适当距离"
          />
        </Form.Item>
      </Form>
    </Modal>
  );

  const renderAdaptationModal = () => (
    <Modal
      title="执行情境适应"
      open={showAdaptationModal}
      onCancel={() => setShowAdaptationModal(false)}
      footer={[
        <Button key="cancel" onClick={() => setShowAdaptationModal(false)}>
          取消
        </Button>,
        <Button 
          key="adapt" 
          type="primary" 
          loading={loading}
          onClick={() => adaptationForm.submit()}
        >
          执行适应
        </Button>
      ]}
      width={600}
    >
      <Form
        form={adaptationForm}
        layout="vertical"
        onFinish={performAdaptation}
      >
        <Alert
          message="情境适应执行"
          description="根据当前情境调整行为和沟通方式"
          type="info"
          showIcon
          style={{ marginBottom: 24 }}
        />

        <Form.Item
          label="目标沟通风格"
          name="target_communication_style"
          initialValue="formal"
          rules={[{ required: true, message: '请选择目标沟通风格' }]}
        >
          <Select>
            {communicationStyles.map(style => (
              <Option key={style} value={style}>{style}</Option>
            ))}
          </Select>
        </Form.Item>

        <Form.Item
          label="正式度调整 (%)"
          name="formality_adaptation"
          initialValue={80}
        >
          <Slider min={0} max={100} />
        </Form.Item>

        <Form.Item
          label="情感调节程度 (%)"
          name="emotional_regulation"
          initialValue={70}
        >
          <Slider min={0} max={100} />
        </Form.Item>

        <Form.Item
          label="行为调整重点"
          name="behavioral_focus"
        >
          <Checkbox.Group>
            <Row>
              <Col span={12}><Checkbox value="voice_control">语音控制</Checkbox></Col>
              <Col span={12}><Checkbox value="body_language">肢体语言</Checkbox></Col>
              <Col span={12}><Checkbox value="eye_contact">眼神接触</Checkbox></Col>
              <Col span={12}><Checkbox value="personal_space">个人空间</Checkbox></Col>
              <Col span={12}><Checkbox value="gesture_control">手势控制</Checkbox></Col>
              <Col span={12}><Checkbox value="facial_expression">面部表情</Checkbox></Col>
            </Row>
          </Checkbox.Group>
        </Form.Item>

        <Form.Item
          label="适应策略描述"
          name="adaptation_description"
        >
          <TextArea 
            rows={3} 
            placeholder="描述具体的适应策略和预期效果..."
          />
        </Form.Item>
      </Form>
    </Modal>
  );

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: 24, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Title level={2}>
          <GlobalOutlined style={{ marginRight: 12, color: '#1890ff' }} />
          社交情境适应引擎
        </Title>
        <Space>
          <Select
            style={{ width: 200 }}
            value={selectedContextType}
            onChange={setSelectedContextType}
            placeholder="选择情境类型"
          >
            {contextTypes.map(type => (
              <Option key={type} value={type}>
                {contextTypeLabels[type as keyof typeof contextTypeLabels]}
              </Option>
            ))}
          </Select>
          <Button 
            type="primary" 
            icon={<ExperimentOutlined />}
            onClick={() => setShowContextModal(true)}
          >
            情境分析
          </Button>
          <Button 
            icon={<ThunderboltOutlined />}
            onClick={() => setShowAdaptationModal(true)}
            disabled={!currentContext}
          >
            执行适应
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
        <TabPane tab="情境概览" key="overview">
          {renderContextOverview()}
        </TabPane>

        <TabPane tab="适应策略" key="strategies">
          {renderAdaptationStrategies()}
        </TabPane>

        <TabPane tab="适应历史" key="history">
          {renderAdaptationHistory()}
        </TabPane>

        <TabPane tab="规则管理" key="rules">
          <Card title="适应规则配置">
            <Alert
              message="规则管理功能"
              description="在此页面可以配置和管理不同情境的适应规则"
              type="info"
              showIcon
              style={{ marginBottom: 24 }}
            />
            <div style={{ textAlign: 'center', padding: 60 }}>
              <Text type="secondary">规则管理功能正在开发中...</Text>
            </div>
          </Card>
        </TabPane>
      </Tabs>

      {renderContextModal()}
      {renderAdaptationModal()}
    </div>
  );
};

export default SocialContextAdaptationPage;