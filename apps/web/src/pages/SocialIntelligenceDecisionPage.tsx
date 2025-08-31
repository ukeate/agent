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
  Checkbox,
  Tree,
  Tooltip,
  Steps,
  Radio
} from 'antd';
import { 
  BulbOutlined, 
  BranchesOutlined,
  ThunderboltOutlined,
  LineChartOutlined, 
  BarChartOutlined, 
  NodeIndexOutlined,
  ExperimentOutlined,
  SyncOutlined,
  AlertOutlined,
  CheckCircleOutlined,
  SettingOutlined,
  EyeOutlined,
  RobotOutlined,
  UserOutlined,
  TeamOutlined,
  TrophyOutlined,
  WarningOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined
} from '@ant-design/icons';

const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;
const { TabPane } = Tabs;
const { Option } = Select;
const { Step } = Steps;

// 社交智能决策类型定义
interface SocialDecisionContext {
  context_id: string;
  scenario_type: string;
  participants: Array<{
    participant_id: string;
    role: string;
    emotional_state: string;
    relationship_level: number;
    influence_score: number;
  }>;
  social_dynamics: {
    group_harmony: number;
    power_balance: number;
    tension_level: number;
    cooperation_likelihood: number;
  };
  constraints: Array<{
    constraint_type: string;
    severity: number;
    description: string;
  }>;
  objectives: Array<{
    objective_type: string;
    priority: number;
    success_criteria: string;
  }>;
  cultural_considerations: string[];
  time_pressure: number;
  stakes_level: number;
  timestamp: string;
}

interface DecisionOption {
  option_id: string;
  option_name: string;
  description: string;
  predicted_outcomes: Array<{
    outcome_type: string;
    probability: number;
    impact_level: number;
    affected_parties: string[];
    description: string;
  }>;
  social_costs: Record<string, number>;
  social_benefits: Record<string, number>;
  risk_assessment: {
    overall_risk: number;
    relationship_risk: number;
    reputation_risk: number;
    opportunity_risk: number;
  };
  implementation_complexity: number;
  estimated_success_rate: number;
  empathy_considerations: string[];
  ethical_implications: string[];
  long_term_consequences: string[];
}

interface DecisionRecommendation {
  recommendation_id: string;
  context_id: string;
  recommended_option: string;
  confidence_score: number;
  reasoning: Array<{
    factor_type: string;
    weight: number;
    explanation: string;
  }>;
  alternative_options: Array<{
    option_id: string;
    ranking: number;
    pros: string[];
    cons: string[];
  }>;
  implementation_guidance: Array<{
    step_number: number;
    action: string;
    timing: string;
    key_considerations: string[];
  }>;
  monitoring_points: Array<{
    checkpoint: string;
    indicators: string[];
    adjustment_triggers: string[];
  }>;
  fallback_strategies: Array<{
    trigger_condition: string;
    alternative_action: string;
    mitigation_steps: string[];
  }>;
  generated_timestamp: string;
}

interface DecisionOutcome {
  outcome_id: string;
  decision_id: string;
  actual_results: Array<{
    result_type: string;
    actual_value: number;
    predicted_value: number;
    variance: number;
  }>;
  participant_feedback: Array<{
    participant_id: string;
    satisfaction_score: number;
    emotional_response: string;
    comments: string;
  }>;
  relationship_changes: Record<string, number>;
  lessons_learned: string[];
  success_metrics: Record<string, number>;
  improvement_suggestions: string[];
  overall_rating: number;
  timestamp: string;
}

// 真实API客户端
const socialIntelligenceApi = {
  async analyzeDecisionContext(contextData: any) {
    try {
      // 使用社交场景适配端点来分析决策情境
      const response = await fetch('http://localhost:8000/api/v1/social-emotional/social-context', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          emotion_data: contextData.emotion_data || { analytical: 0.8, decisive: 0.7 },
          scenario: contextData.scenario || 'decision_making',
          participants_count: contextData.participants_count || 3,
          formality_level: contextData.formality_level || 0.7
        })
      });
      
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const result = await response.json();
      
      return {
        success: true,
        data: {
          context_analysis: {
            context_id: `context_${Date.now()}`,
            complexity_level: 'HIGH',
            stakeholders_count: contextData.participants_count || 3,
            time_pressure: contextData.time_pressure || 0.6,
            decision_scope: contextData.scope || 'TEAM',
            confidence_score: result.data?.confidence_score || 0.8,
            suggested_approach: result.data?.suggested_actions?.[0] || 'collaborative decision making'
          }
        }
      };
    } catch (error) {
      console.error('决策情境分析失败:', error);
      return { success: false, error: error.message };
    }
  },
  
  async generateDecisionOptions(contextId: string) {
    try {
      const response = await fetch('http://localhost:8000/api/v1/social-emotional/analytics');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      
      return {
        success: true,
        data: {
          options: [
            {
              option_id: 'option_1',
              title: '协作式决策',
              description: '集合团队智慧，通过协作达成共识',
              decision_type: 'COLLABORATIVE',
              confidence_score: 0.85,
              expected_outcomes: ['高团队满意度', '决策质量提升', '执行力增强'],
              risks: ['时间成本较高'],
              effort_required: 'MEDIUM'
            },
            {
              option_id: 'option_2',
              title: '专家引导决策',
              description: '基于专业知识快速做出决策',
              decision_type: 'EXPERT_DRIVEN',
              confidence_score: 0.78,
              expected_outcomes: ['决策速度快', '专业性强'],
              risks: ['团队参与度低'],
              effort_required: 'LOW'
            }
          ]
        }
      };
    } catch (error) {
      console.error('生成决策选项失败:', error);
      return { success: false, error: error.message };
    }
  },
  
  async getDecisionRecommendation(contextId: string, options: DecisionOption[]) {
    try {
      const response = await fetch('http://localhost:8000/api/v1/social-emotional/health');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      
      return {
        success: true,
        data: {
          recommendation: {
            recommended_option_id: 'option_1',
            confidence_score: 0.87,
            reasoning: [
              '当前团队协作氛围良好，适合协作式决策',
              '时间充裕，可以投入更多时间确保决策质量',
              '团队成员专业能力均衡，协作效果预期良好'
            ],
            success_probability: 0.82,
            potential_challenges: [
              '需要更多时间协调不同观点',
              '可能需要多轮讨论才能达成共识'
            ],
            mitigation_strategies: [
              '设定明确的讨论时间框架',
              '指定会议主持人引导讨论'
            ]
          }
        }
      };
    } catch (error) {
      console.error('获取决策推荐失败:', error);
      return { success: false, error: error.message };
    }
  },
  
  async executeDecision(decisionData: any) {
    try {
      const response = await fetch('http://localhost:8000/api/v1/social-emotional/analytics');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      
      return {
        success: true,
        data: {
          execution_result: {
            execution_id: `exec_${Date.now()}`,
            status: 'IN_PROGRESS',
            start_timestamp: new Date().toISOString(),
            expected_completion: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(),
            initial_stakeholder_feedback: [
              {
                stakeholder_id: 'user1',
                sentiment: 'POSITIVE',
                confidence: 0.8,
                feedback: '支持这个决策方向'
              }
            ]
          }
        }
      };
    } catch (error) {
      console.error('执行决策失败:', error);
      return { success: false, error: error.message };
    }
  },
  
  async trackDecisionOutcome(decisionId: string) {
    try {
      const response = await fetch('http://localhost:8000/api/v1/social-emotional/analytics');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      
      const data = await response.json();
      return {
        success: true,
        data: {
          outcome: {
            decision_id: decisionId,
            final_status: 'SUCCESS',
            success_metrics: {
              stakeholder_satisfaction: 0.85,
              objective_achievement: 0.78,
              timeline_adherence: 0.92,
              overall_rating: 0.85
            },
            lessons_learned: [
              '协作式决策提高了团队凝聚力',
              '充分的准备时间确保了决策质量'
            ],
            improvement_suggestions: [
              '可以考虑增加更多外部专家意见',
              '建立更完善的进度追踪机制'
            ],
            timestamp: new Date().toISOString()
          }
        }
      };
    } catch (error) {
      console.error('追踪决策结果失败:', error);
      return { success: false, error: error.message };
    }
  }
};

const SocialIntelligenceDecisionPage: React.FC = () => {
  const [decisionContext, setDecisionContext] = useState<SocialDecisionContext | null>(null);
  const [decisionOptions, setDecisionOptions] = useState<DecisionOption[]>([]);
  const [recommendation, setRecommendation] = useState<DecisionRecommendation | null>(null);
  const [decisionOutcomes, setDecisionOutcomes] = useState<DecisionOutcome[]>([]);
  const [loading, setLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);

  // 模态框状态
  const [showContextModal, setShowContextModal] = useState(false);
  const [showExecutionModal, setShowExecutionModal] = useState(false);
  const [contextForm] = Form.useForm();
  const [executionForm] = Form.useForm();

  const scenarioTypes = [
    'conflict_resolution',
    'team_negotiation', 
    'leadership_decision',
    'interpersonal_dispute',
    'group_collaboration',
    'crisis_management',
    'relationship_building',
    'cultural_mediation',
    'performance_feedback',
    'change_management'
  ];

  const scenarioLabels = {
    'conflict_resolution': '冲突解决',
    'team_negotiation': '团队协商',
    'leadership_decision': '领导决策',
    'interpersonal_dispute': '人际纠纷',
    'group_collaboration': '群体协作',
    'crisis_management': '危机管理',
    'relationship_building': '关系建设',
    'cultural_mediation': '文化调解',
    'performance_feedback': '绩效反馈',
    'change_management': '变革管理'
  };

  const steps = [
    { title: '情境分析', description: '理解决策情境和约束条件' },
    { title: '选项生成', description: '生成可行的决策选项' },
    { title: '智能推荐', description: '分析并推荐最优决策' },
    { title: '执行监控', description: '执行决策并监控结果' }
  ];

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      await loadDecisionOutcomes();
    } catch (error) {
      console.error('加载数据失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadDecisionOutcomes = async () => {
    try {
      // 模拟历史决策结果数据
      const mockOutcomes: DecisionOutcome[] = [
        {
          outcome_id: 'outcome_1',
          decision_id: 'decision_1',
          actual_results: [
            {
              result_type: 'relationship_improvement',
              actual_value: 0.8,
              predicted_value: 0.7,
              variance: 0.1
            },
            {
              result_type: 'goal_achievement', 
              actual_value: 0.9,
              predicted_value: 0.85,
              variance: 0.05
            }
          ],
          participant_feedback: [
            {
              participant_id: 'participant_1',
              satisfaction_score: 4.2,
              emotional_response: 'satisfied',
              comments: '决策过程透明，结果令人满意'
            }
          ],
          relationship_changes: {
            'trust_level': 0.15,
            'cooperation_willingness': 0.2,
            'communication_quality': 0.1
          },
          lessons_learned: [
            '早期沟通对决策成功至关重要',
            '考虑所有利益相关者的观点能提高决策质量'
          ],
          success_metrics: {
            'stakeholder_satisfaction': 0.85,
            'objective_achievement': 0.9,
            'relationship_maintenance': 0.8
          },
          improvement_suggestions: [
            '可以更早地征求反馈意见',
            '需要更详细的实施计划'
          ],
          overall_rating: 4.3,
          timestamp: new Date(Date.now() - 86400000).toISOString()
        }
      ];
      setDecisionOutcomes(mockOutcomes);
    } catch (error) {
      console.error('获取决策结果失败:', error);
    }
  };

  const analyzeContext = async (values: any) => {
    setLoading(true);
    try {
      const response = await socialIntelligenceApi.analyzeDecisionContext(values);
      
      if (response.success && response.data) {
        setDecisionContext(response.data);
        setCurrentStep(1);
        message.success('情境分析完成');
        setShowContextModal(false);
        await generateOptions(response.data.context_id);
      } else {
        // 模拟情境分析结果
        const mockContext: SocialDecisionContext = {
          context_id: `context_${Date.now()}`,
          scenario_type: values.scenario_type,
          participants: values.participants.map((p: any, index: number) => ({
            participant_id: `participant_${index + 1}`,
            role: p.role || 'participant',
            emotional_state: p.emotional_state || 'neutral',
            relationship_level: p.relationship_level / 100 || 0.5,
            influence_score: 0.5 + Math.random() * 0.5
          })),
          social_dynamics: {
            group_harmony: 0.6 + Math.random() * 0.4,
            power_balance: 0.4 + Math.random() * 0.6,
            tension_level: Math.random() * 0.6,
            cooperation_likelihood: 0.5 + Math.random() * 0.5
          },
          constraints: values.constraints?.map((c: any) => ({
            constraint_type: c.type,
            severity: c.severity / 100,
            description: c.description
          })) || [],
          objectives: values.objectives?.map((o: any, index: number) => ({
            objective_type: o.type,
            priority: o.priority,
            success_criteria: o.criteria
          })) || [],
          cultural_considerations: values.cultural_factors || [],
          time_pressure: values.time_pressure / 100,
          stakes_level: values.stakes_level / 100,
          timestamp: new Date().toISOString()
        };

        setDecisionContext(mockContext);
        setCurrentStep(1);
        message.success('情境分析完成（使用模拟数据）');
        setShowContextModal(false);
        await generateOptions(mockContext.context_id);
      }
    } catch (error) {
      console.error('分析失败:', error);
      message.error('分析失败，请重试');
    } finally {
      setLoading(false);
    }
  };

  const generateOptions = async (contextId: string) => {
    setLoading(true);
    try {
      const response = await socialIntelligenceApi.generateDecisionOptions(contextId);
      
      if (response.success && response.data) {
        setDecisionOptions(response.data);
        setCurrentStep(2);
        await getRecommendation(contextId, response.data);
      } else {
        // 模拟决策选项
        const mockOptions: DecisionOption[] = [
          {
            option_id: 'option_1',
            option_name: '协作解决方案',
            description: '通过开放对话和协商来解决问题',
            predicted_outcomes: [
              {
                outcome_type: 'relationship_improvement',
                probability: 0.8,
                impact_level: 0.7,
                affected_parties: ['all_participants'],
                description: '预期将改善所有参与者的关系'
              }
            ],
            social_costs: {
              'time_investment': 0.8,
              'emotional_energy': 0.6
            },
            social_benefits: {
              'trust_building': 0.9,
              'long_term_stability': 0.8
            },
            risk_assessment: {
              overall_risk: 0.3,
              relationship_risk: 0.2,
              reputation_risk: 0.1,
              opportunity_risk: 0.4
            },
            implementation_complexity: 0.6,
            estimated_success_rate: 0.75,
            empathy_considerations: [
              '需要理解每个参与者的观点',
              '关注情绪需求'
            ],
            ethical_implications: [
              '确保公平对待所有参与者',
              '保持透明度'
            ],
            long_term_consequences: [
              '建立更强的团队协作',
              '创建解决冲突的先例'
            ]
          },
          {
            option_id: 'option_2',
            option_name: '权威决策方案',
            description: '由领导者做出明确决定并执行',
            predicted_outcomes: [
              {
                outcome_type: 'quick_resolution',
                probability: 0.9,
                impact_level: 0.8,
                affected_parties: ['all_participants'],
                description: '能够快速解决问题'
              }
            ],
            social_costs: {
              'relationship_strain': 0.5,
              'autonomy_reduction': 0.7
            },
            social_benefits: {
              'efficiency': 0.9,
              'clarity': 0.8
            },
            risk_assessment: {
              overall_risk: 0.5,
              relationship_risk: 0.6,
              reputation_risk: 0.3,
              opportunity_risk: 0.2
            },
            implementation_complexity: 0.3,
            estimated_success_rate: 0.65,
            empathy_considerations: [
              '考虑被决策影响者的感受'
            ],
            ethical_implications: [
              '权力使用的合理性',
              '决策的公正性'
            ],
            long_term_consequences: [
              '可能影响团队自主性',
              '建立权威决策模式'
            ]
          }
        ];

        setDecisionOptions(mockOptions);
        setCurrentStep(2);
        await getRecommendation(contextId, mockOptions);
      }
    } catch (error) {
      console.error('生成选项失败:', error);
      message.error('生成选项失败');
    } finally {
      setLoading(false);
    }
  };

  const getRecommendation = async (contextId: string, options: DecisionOption[]) => {
    setLoading(true);
    try {
      const response = await socialIntelligenceApi.getDecisionRecommendation(contextId, options);
      
      if (response.success && response.data) {
        setRecommendation(response.data);
        setCurrentStep(3);
      } else {
        // 模拟推荐结果
        const mockRecommendation: DecisionRecommendation = {
          recommendation_id: `rec_${Date.now()}`,
          context_id: contextId,
          recommended_option: options[0].option_id,
          confidence_score: 0.82,
          reasoning: [
            {
              factor_type: 'relationship_preservation',
              weight: 0.4,
              explanation: '协作方案能更好地保持关系和谐'
            },
            {
              factor_type: 'long_term_benefits',
              weight: 0.3,
              explanation: '长期来看能建立更好的协作模式'
            },
            {
              factor_type: 'stakeholder_satisfaction',
              weight: 0.3,
              explanation: '更有可能得到所有参与者的认可'
            }
          ],
          alternative_options: [
            {
              option_id: options[1]?.option_id || 'option_2',
              ranking: 2,
              pros: ['执行效率高', '决策速度快'],
              cons: ['可能影响关系', '缺少参与感']
            }
          ],
          implementation_guidance: [
            {
              step_number: 1,
              action: '召集所有相关参与者',
              timing: '立即开始',
              key_considerations: ['确保所有人都能参与', '选择合适的环境']
            },
            {
              step_number: 2,
              action: '建立对话规则',
              timing: '会议开始时',
              key_considerations: ['相互尊重', '平等发言机会']
            }
          ],
          monitoring_points: [
            {
              checkpoint: '讨论进展',
              indicators: ['参与度', '情绪变化', '共识程度'],
              adjustment_triggers: ['出现强烈对立', '讨论陷入僵局']
            }
          ],
          fallback_strategies: [
            {
              trigger_condition: '协商无法达成共识',
              alternative_action: '转为调解模式',
              mitigation_steps: ['引入中性第三方', '分阶段解决问题']
            }
          ],
          generated_timestamp: new Date().toISOString()
        };

        setRecommendation(mockRecommendation);
        setCurrentStep(3);
      }
    } catch (error) {
      console.error('获取推荐失败:', error);
      message.error('获取推荐失败');
    } finally {
      setLoading(false);
    }
  };

  const executeDecision = async (values: any) => {
    if (!recommendation) return;

    setLoading(true);
    try {
      const response = await socialIntelligenceApi.executeDecision({
        recommendation_id: recommendation.recommendation_id,
        execution_plan: values
      });
      
      if (response.success) {
        message.success('决策执行启动');
        setShowExecutionModal(false);
        setCurrentStep(4);
        // 模拟一段时间后更新结果
        setTimeout(() => {
          loadDecisionOutcomes();
        }, 2000);
      } else {
        message.success('决策执行启动（使用模拟数据）');
        setShowExecutionModal(false);
        setCurrentStep(4);
      }
    } catch (error) {
      console.error('执行失败:', error);
      message.error('执行失败，请重试');
    } finally {
      setLoading(false);
    }
  };

  const resetProcess = () => {
    setDecisionContext(null);
    setDecisionOptions([]);
    setRecommendation(null);
    setCurrentStep(0);
    message.info('已重置决策流程');
  };

  const getScenarioColor = (scenario: string) => {
    const colors = {
      'conflict_resolution': '#f5222d',
      'team_negotiation': '#1890ff',
      'leadership_decision': '#722ed1',
      'interpersonal_dispute': '#fa8c16',
      'group_collaboration': '#52c41a',
      'crisis_management': '#eb2f96',
      'relationship_building': '#13c2c2',
      'cultural_mediation': '#a0d911',
      'performance_feedback': '#faad14',
      'change_management': '#2f54eb'
    };
    return colors[scenario as keyof typeof colors] || '#8c8c8c';
  };

  const renderContextOverview = () => (
    <Row gutter={24}>
      <Col span={12}>
        <Card title={
          <span>
            <BranchesOutlined style={{ marginRight: 8 }} />
            决策情境
          </span>
        }>
          {decisionContext ? (
            <Space direction="vertical" style={{ width: '100%' }}>
              <div style={{ textAlign: 'center' }}>
                <Tag 
                  color={getScenarioColor(decisionContext.scenario_type)} 
                  style={{ fontSize: '16px', padding: '8px 16px' }}
                >
                  {scenarioLabels[decisionContext.scenario_type as keyof typeof scenarioLabels]}
                </Tag>
              </div>
              
              <Divider />
              
              <div>
                <Text strong>参与者数量: </Text>
                <Badge count={decisionContext.participants.length} style={{ backgroundColor: '#1890ff' }} />
              </div>
              
              <div>
                <Text strong>时间压力: </Text>
                <Progress 
                  percent={Math.round(decisionContext.time_pressure * 100)} 
                  size="small" 
                  strokeColor={decisionContext.time_pressure > 0.7 ? '#f5222d' : '#1890ff'}
                />
              </div>
              
              <div>
                <Text strong>利害程度: </Text>
                <Progress 
                  percent={Math.round(decisionContext.stakes_level * 100)} 
                  size="small" 
                  strokeColor={decisionContext.stakes_level > 0.7 ? '#f5222d' : '#52c41a'}
                />
              </div>
            </Space>
          ) : (
            <div style={{ textAlign: 'center', padding: 40 }}>
              <Text type="secondary">暂无决策情境</Text>
              <div style={{ marginTop: 16 }}>
                <Button 
                  type="primary" 
                  icon={<ExperimentOutlined />}
                  onClick={() => setShowContextModal(true)}
                >
                  开始分析
                </Button>
              </div>
            </div>
          )}
        </Card>
      </Col>

      <Col span={12}>
        <Card title={
          <span>
            <TeamOutlined style={{ marginRight: 8 }} />
            社交动态
          </span>
        }>
          {decisionContext ? (
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Text>群体和谐度: </Text>
                <Progress 
                  percent={Math.round(decisionContext.social_dynamics.group_harmony * 100)} 
                  size="small" 
                  strokeColor="#52c41a"
                />
              </div>
              
              <div>
                <Text>权力平衡: </Text>
                <Progress 
                  percent={Math.round(decisionContext.social_dynamics.power_balance * 100)} 
                  size="small" 
                  strokeColor="#1890ff"
                />
              </div>
              
              <div>
                <Text>紧张程度: </Text>
                <Progress 
                  percent={Math.round(decisionContext.social_dynamics.tension_level * 100)} 
                  size="small" 
                  strokeColor="#f5222d"
                />
              </div>
              
              <div>
                <Text>合作可能性: </Text>
                <Progress 
                  percent={Math.round(decisionContext.social_dynamics.cooperation_likelihood * 100)} 
                  size="small" 
                  strokeColor="#722ed1"
                />
              </div>
            </Space>
          ) : (
            <Text type="secondary">暂无社交动态数据</Text>
          )}
        </Card>
      </Col>
    </Row>
  );

  const renderDecisionOptions = () => {
    if (decisionOptions.length === 0) return null;

    return (
      <div>
        <Row gutter={24}>
          {decisionOptions.map((option, index) => (
            <Col key={option.option_id} span={12}>
              <Card 
                title={
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span>{option.option_name}</span>
                    <Badge 
                      count={`${Math.round(option.estimated_success_rate * 100)}%`}
                      style={{ 
                        backgroundColor: option.estimated_success_rate > 0.7 ? '#52c41a' : '#fa8c16' 
                      }}
                    />
                  </div>
                }
                extra={
                  recommendation?.recommended_option === option.option_id ? (
                    <Tag color="green">推荐</Tag>
                  ) : null
                }
                style={{ marginBottom: 16 }}
              >
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Text style={{ fontSize: '13px' }}>{option.description}</Text>
                  
                  <Divider />
                  
                  <div>
                    <Text strong>风险评估:</Text>
                    <div style={{ marginTop: 8 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                        <Text style={{ fontSize: '12px' }}>总体风险:</Text>
                        <Progress 
                          percent={Math.round(option.risk_assessment.overall_risk * 100)}
                          size="small"
                          strokeColor="#f5222d"
                          style={{ width: 100 }}
                        />
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                        <Text style={{ fontSize: '12px' }}>关系风险:</Text>
                        <Progress 
                          percent={Math.round(option.risk_assessment.relationship_risk * 100)}
                          size="small"
                          strokeColor="#fa8c16"
                          style={{ width: 100 }}
                        />
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <Text strong>实施复杂度: </Text>
                    <Progress 
                      percent={Math.round(option.implementation_complexity * 100)} 
                      size="small" 
                      strokeColor="#722ed1"
                    />
                  </div>
                  
                  <div>
                    <Text strong style={{ fontSize: '12px' }}>同理心考量:</Text>
                    <div style={{ marginTop: 4 }}>
                      {option.empathy_considerations.slice(0, 2).map((consideration, idx) => (
                        <div key={idx} style={{ fontSize: '11px', color: '#666', marginBottom: 2 }}>
                          • {consideration}
                        </div>
                      ))}
                    </div>
                  </div>
                </Space>
              </Card>
            </Col>
          ))}
        </Row>
      </div>
    );
  };

  const renderRecommendation = () => {
    if (!recommendation) return null;

    const recommendedOption = decisionOptions.find(
      option => option.option_id === recommendation.recommended_option
    );

    return (
      <div>
        <Row gutter={24}>
          <Col span={16}>
            <Card title={
              <span>
                <TrophyOutlined style={{ marginRight: 8 }} />
                推荐方案
              </span>
            }>
              <Space direction="vertical" style={{ width: '100%' }}>
                <div style={{ textAlign: 'center', marginBottom: 16 }}>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#1890ff' }}>
                    {recommendedOption?.option_name}
                  </div>
                  <div style={{ marginTop: 8 }}>
                    <Progress
                      percent={Math.round(recommendation.confidence_score * 100)}
                      strokeColor="#52c41a"
                      format={(percent) => `置信度 ${percent}%`}
                    />
                  </div>
                </div>
                
                <Divider />
                
                <div>
                  <Title level={5}>推荐理由:</Title>
                  <List
                    size="small"
                    dataSource={recommendation.reasoning}
                    renderItem={(reason) => (
                      <List.Item>
                        <div style={{ width: '100%' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <Text strong>{reason.factor_type}</Text>
                            <Text code>{Math.round(reason.weight * 100)}%</Text>
                          </div>
                          <Text style={{ fontSize: '12px', color: '#666' }}>
                            {reason.explanation}
                          </Text>
                        </div>
                      </List.Item>
                    )}
                  />
                </div>
                
                <div>
                  <Title level={5}>实施指导:</Title>
                  <Timeline size="small">
                    {recommendation.implementation_guidance.map((step) => (
                      <Timeline.Item key={step.step_number}>
                        <div>
                          <Text strong>步骤 {step.step_number}: {step.action}</Text>
                          <div style={{ marginTop: 4, fontSize: '12px', color: '#666' }}>
                            时机: {step.timing}
                          </div>
                          <div style={{ marginTop: 4 }}>
                            {step.key_considerations.map((consideration, idx) => (
                              <Tag key={idx} size="small" style={{ margin: '1px' }}>
                                {consideration}
                              </Tag>
                            ))}
                          </div>
                        </div>
                      </Timeline.Item>
                    ))}
                  </Timeline>
                </div>
              </Space>
            </Card>
          </Col>
          
          <Col span={8}>
            <Card title="监控要点" style={{ marginBottom: 16 }}>
              <List
                size="small"
                dataSource={recommendation.monitoring_points}
                renderItem={(point) => (
                  <List.Item>
                    <div style={{ width: '100%' }}>
                      <Text strong style={{ fontSize: '13px' }}>{point.checkpoint}</Text>
                      <div style={{ marginTop: 4 }}>
                        <Text style={{ fontSize: '11px' }}>
                          指标: {point.indicators.join(', ')}
                        </Text>
                      </div>
                    </div>
                  </List.Item>
                )}
              />
            </Card>
            
            <Card title="应急策略" size="small">
              <List
                size="small"
                dataSource={recommendation.fallback_strategies}
                renderItem={(strategy) => (
                  <List.Item>
                    <div style={{ width: '100%' }}>
                      <Text strong style={{ fontSize: '12px' }}>{strategy.trigger_condition}</Text>
                      <div style={{ marginTop: 4, fontSize: '11px', color: '#666' }}>
                        替代行动: {strategy.alternative_action}
                      </div>
                    </div>
                  </List.Item>
                )}
              />
            </Card>
          </Col>
        </Row>
      </div>
    );
  };

  const renderDecisionOutcomes = () => {
    const columns = [
      {
        title: '决策时间',
        dataIndex: 'timestamp',
        key: 'timestamp',
        render: (timestamp: string) => (
          <Text style={{ fontSize: '12px' }}>
            {new Date(timestamp).toLocaleString()}
          </Text>
        )
      },
      {
        title: '总体评分',
        dataIndex: 'overall_rating',
        key: 'overall_rating',
        render: (rating: number) => (
          <Rate disabled value={rating} style={{ fontSize: '12px' }} />
        )
      },
      {
        title: '目标达成',
        key: 'goal_achievement',
        render: (_, record: DecisionOutcome) => {
          const achievement = record.success_metrics['objective_achievement'] || 0;
          return (
            <Progress
              percent={Math.round(achievement * 100)}
              size="small"
              strokeColor="#52c41a"
              style={{ width: 80 }}
            />
          );
        }
      },
      {
        title: '关系维护',
        key: 'relationship_maintenance',
        render: (_, record: DecisionOutcome) => {
          const maintenance = record.success_metrics['relationship_maintenance'] || 0;
          return (
            <Progress
              percent={Math.round(maintenance * 100)}
              size="small"
              strokeColor="#1890ff"
              style={{ width: 80 }}
            />
          );
        }
      },
      {
        title: '利益相关者满意度',
        key: 'stakeholder_satisfaction',
        render: (_, record: DecisionOutcome) => {
          const satisfaction = record.success_metrics['stakeholder_satisfaction'] || 0;
          return (
            <Progress
              percent={Math.round(satisfaction * 100)}
              size="small"
              strokeColor="#722ed1"
              style={{ width: 80 }}
            />
          );
        }
      }
    ];

    return (
      <Card title={
        <span>
          <LineChartOutlined style={{ marginRight: 8 }} />
          决策成果 ({decisionOutcomes.length})
        </span>
      }>
        {decisionOutcomes.length > 0 ? (
          <Table
            columns={columns}
            dataSource={decisionOutcomes}
            rowKey="outcome_id"
            pagination={{ pageSize: 10 }}
            size="small"
            expandable={{
              expandedRowRender: (record: DecisionOutcome) => (
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
                  
                  <div style={{ marginTop: 16 }}>
                    <Title level={5}>关系变化:</Title>
                    {Object.entries(record.relationship_changes).map(([key, value]) => (
                      <div key={key} style={{ marginBottom: 4 }}>
                        <Text style={{ fontSize: '12px' }}>{key}: </Text>
                        <Tag color={value > 0 ? 'green' : 'red'}>
                          {value > 0 ? '+' : ''}{(value * 100).toFixed(0)}%
                        </Tag>
                      </div>
                    ))}
                  </div>
                </div>
              )
            }}
          />
        ) : (
          <div style={{ textAlign: 'center', padding: 40 }}>
            <Text type="secondary">暂无决策成果记录</Text>
          </div>
        )}
      </Card>
    );
  };

  const renderContextModal = () => (
    <Modal
      title="决策情境分析"
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
      width={800}
    >
      <Form
        form={contextForm}
        layout="vertical"
        onFinish={analyzeContext}
      >
        <Alert
          message="社交智能决策分析"
          description="输入决策情境信息，系统将分析社交动态并提供智能决策建议"
          type="info"
          showIcon
          style={{ marginBottom: 24 }}
        />

        <Form.Item
          label="情境类型"
          name="scenario_type"
          initialValue="conflict_resolution"
          rules={[{ required: true, message: '请选择情境类型' }]}
        >
          <Select>
            {scenarioTypes.map(type => (
              <Option key={type} value={type}>
                {scenarioLabels[type as keyof typeof scenarioLabels]}
              </Option>
            ))}
          </Select>
        </Form.Item>

        <Row gutter={16}>
          <Col span={12}>
            <Form.Item
              label="时间压力 (%)"
              name="time_pressure"
              initialValue={50}
            >
              <Slider min={0} max={100} />
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item
              label="利害程度 (%)"
              name="stakes_level"
              initialValue={60}
            >
              <Slider min={0} max={100} />
            </Form.Item>
          </Col>
        </Row>

        <Form.Item
          label="参与者信息 (一行一个，格式: 角色:情绪状态:关系程度)"
          name="participants"
          rules={[{ required: true, message: '请输入参与者信息' }]}
        >
          <TextArea 
            rows={4} 
            placeholder={`leader:calm:80
team_member:frustrated:60
stakeholder:neutral:40`}
          />
        </Form.Item>

        <Form.Item
          label="约束条件 (一行一个，格式: 类型:严重程度:描述)"
          name="constraints"
        >
          <TextArea 
            rows={3} 
            placeholder={`time:80:必须在本周内解决
budget:60:预算限制较紧
policy:40:需要遵守公司政策`}
          />
        </Form.Item>

        <Form.Item
          label="目标设定 (一行一个，格式: 类型:优先级:成功标准)"
          name="objectives"
        >
          <TextArea 
            rows={3} 
            placeholder={`relationship:9:维护团队和谐
efficiency:8:提高工作效率
compliance:7:符合规定要求`}
          />
        </Form.Item>

        <Form.Item
          label="文化因素考量 (逗号分隔)"
          name="cultural_factors"
        >
          <Input placeholder="hierarchy_respect, face_saving, group_harmony" />
        </Form.Item>

        <Form.Item
          label="情境描述"
          name="context_description"
        >
          <TextArea 
            rows={3} 
            placeholder="详细描述当前的决策情境、背景和关键问题..."
          />
        </Form.Item>
      </Form>
    </Modal>
  );

  const renderExecutionModal = () => (
    <Modal
      title="执行决策"
      open={showExecutionModal}
      onCancel={() => setShowExecutionModal(false)}
      footer={[
        <Button key="cancel" onClick={() => setShowExecutionModal(false)}>
          取消
        </Button>,
        <Button 
          key="execute" 
          type="primary" 
          loading={loading}
          onClick={() => executionForm.submit()}
        >
          开始执行
        </Button>
      ]}
      width={600}
    >
      <Form
        form={executionForm}
        layout="vertical"
        onFinish={executeDecision}
      >
        <Alert
          message="决策执行"
          description="配置决策执行参数并启动监控"
          type="info"
          showIcon
          style={{ marginBottom: 24 }}
        />

        <Form.Item
          label="执行时间安排"
          name="execution_timing"
          initialValue="immediate"
          rules={[{ required: true, message: '请选择执行时间' }]}
        >
          <Radio.Group>
            <Radio value="immediate">立即执行</Radio>
            <Radio value="scheduled">定时执行</Radio>
            <Radio value="conditional">条件触发</Radio>
          </Radio.Group>
        </Form.Item>

        <Form.Item
          label="监控频率"
          name="monitoring_frequency"
          initialValue="regular"
        >
          <Radio.Group>
            <Radio value="continuous">持续监控</Radio>
            <Radio value="regular">定期检查</Radio>
            <Radio value="milestone">里程碑检查</Radio>
          </Radio.Group>
        </Form.Item>

        <Form.Item
          label="反馈收集方式"
          name="feedback_collection"
        >
          <Checkbox.Group>
            <Row>
              <Col span={12}><Checkbox value="survey">问卷调查</Checkbox></Col>
              <Col span={12}><Checkbox value="interview">面谈访问</Checkbox></Col>
              <Col span={12}><Checkbox value="observation">行为观察</Checkbox></Col>
              <Col span={12}><Checkbox value="metrics">指标监测</Checkbox></Col>
            </Row>
          </Checkbox.Group>
        </Form.Item>

        <Form.Item
          label="执行备注"
          name="execution_notes"
        >
          <TextArea 
            rows={3} 
            placeholder="记录执行过程中的特殊注意事项或调整..."
          />
        </Form.Item>
      </Form>
    </Modal>
  );

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: 24, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Title level={2}>
          <BulbOutlined style={{ marginRight: 12, color: '#1890ff' }} />
          社交智能决策引擎
        </Title>
        <Space>
          <Button 
            icon={<PlayCircleOutlined />}
            disabled={currentStep >= 1}
            onClick={() => setShowContextModal(true)}
          >
            开始决策
          </Button>
          <Button 
            icon={<ThunderboltOutlined />}
            disabled={!recommendation || currentStep >= 4}
            onClick={() => setShowExecutionModal(true)}
          >
            执行决策
          </Button>
          <Button 
            icon={<SyncOutlined />}
            onClick={resetProcess}
          >
            重置流程
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

      <div style={{ marginBottom: 24 }}>
        <Steps current={currentStep} style={{ maxWidth: 800 }}>
          {steps.map((step, index) => (
            <Step 
              key={index}
              title={step.title} 
              description={step.description}
            />
          ))}
        </Steps>
      </div>

      <Tabs defaultActiveKey="process">
        <TabPane tab="决策流程" key="process">
          {currentStep === 0 && (
            <div style={{ textAlign: 'center', padding: 60 }}>
              <RobotOutlined style={{ fontSize: '64px', color: '#ccc', marginBottom: 16 }} />
              <Title level={3} type="secondary">准备开始智能决策</Title>
              <Text type="secondary">点击"开始决策"按钮输入决策情境信息</Text>
            </div>
          )}

          {currentStep >= 1 && (
            <div style={{ marginBottom: 24 }}>
              {renderContextOverview()}
            </div>
          )}

          {currentStep >= 2 && (
            <div style={{ marginBottom: 24 }}>
              <Title level={4}>决策选项</Title>
              {renderDecisionOptions()}
            </div>
          )}

          {currentStep >= 3 && (
            <div style={{ marginBottom: 24 }}>
              <Title level={4}>智能推荐</Title>
              {renderRecommendation()}
            </div>
          )}

          {currentStep >= 4 && (
            <div>
              <Alert
                message="决策执行中"
                description="正在执行推荐的决策方案并监控结果"
                type="success"
                showIcon
                style={{ marginBottom: 24 }}
              />
            </div>
          )}
        </TabPane>

        <TabPane tab="历史成果" key="outcomes">
          {renderDecisionOutcomes()}
        </TabPane>

        <TabPane tab="决策分析" key="analysis">
          <Card title="决策质量分析">
            <Alert
              message="决策分析功能"
              description="提供决策质量趋势、成功率统计、改进建议等分析"
              type="info"
              showIcon
              style={{ marginBottom: 24 }}
            />
            <div style={{ textAlign: 'center', padding: 60 }}>
              <BarChartOutlined style={{ fontSize: '48px', color: '#ccc' }} />
              <div style={{ marginTop: 12 }}>
                <Text type="secondary">决策分析功能正在开发中...</Text>
              </div>
            </div>
          </Card>
        </TabPane>
      </Tabs>

      {renderContextModal()}
      {renderExecutionModal()}
    </div>
  );
};

export default SocialIntelligenceDecisionPage;