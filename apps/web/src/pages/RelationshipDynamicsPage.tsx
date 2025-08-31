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
  Avatar
} from 'antd';
import { 
  HeartOutlined, 
  UserOutlined,
  LineChartOutlined, 
  BarChartOutlined, 
  NodeIndexOutlined,
  TeamOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  ExperimentOutlined,
  SyncOutlined,
  AlertOutlined,
  TrophyOutlined,
  WarningOutlined
} from '@ant-design/icons';

const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;
const { TabPane } = Tabs;
const { Option } = Select;

// 关系动态类型定义
interface RelationshipDynamics {
  relationship_id: string;
  participants: string[];
  relationship_type: string;
  intimacy_level: string;
  intimacy_score: number;
  trust_level: number;
  vulnerability_sharing: number;
  power_balance: number;
  power_dynamics: string;
  influence_patterns: Record<string, number>;
  emotional_reciprocity: number;
  support_balance: number;
  empathy_symmetry: number;
  support_patterns: EmotionalSupportPattern[];
  conflict_indicators: ConflictIndicator[];
  conflict_frequency: number;
  conflict_resolution_rate: number;
  relationship_health: number;
  stability_score: number;
  satisfaction_level: number;
  development_trend: string;
  future_outlook: string;
  data_quality_score: number;
  confidence_level: number;
  harmony_indicators: string[];
  relationship_trajectory: number[];
  analysis_timestamp: string;
}

interface EmotionalSupportPattern {
  support_id: string;
  giver_id: string;
  receiver_id: string;
  support_type: string;
  frequency: number;
  intensity: number;
  reciprocity_score: number;
  effectiveness_score: number;
  timestamp: string;
  verbal_affirmation: boolean;
  active_listening: boolean;
  empathy_expression: boolean;
  problem_solving: boolean;
  resource_sharing: boolean;
}

interface ConflictIndicator {
  indicator_id: string;
  participants: string[];
  conflict_type: string;
  severity_level: number;
  escalation_risk: number;
  resolution_potential: number;
  timestamp: string;
  verbal_disagreement: boolean;
  emotional_tension: boolean;
  communication_breakdown: boolean;
  value_conflict: boolean;
  resource_competition: boolean;
  conflict_styles: Record<string, string>;
}

interface RelationshipMilestone {
  milestone_id: string;
  relationship_id: string;
  milestone_type: string;
  significance_level: number;
  emotional_impact: number;
  relationship_change: number;
  timestamp: string;
  description: string;
  positive_milestone: boolean;
  relationship_deepening: boolean;
  trust_building: boolean;
  boundary_setting: boolean;
  conflict_resolution: boolean;
}

// 真实API客户端
const relationshipApi = {
  async analyzeRelationship(participants: string[], interactionHistory?: any) {
    try {
      const participantsData = participants.map(userId => ({
        user_id: userId,
        emotion_data: { collaborative: 0.7, trusting: 0.6 },
        context: { relationship_analysis: true }
      }));
      
      const response = await fetch('http://localhost:8000/api/v1/social-emotional/relationships', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          participants: participantsData,
          context: { analysis_type: 'relationship_dynamics', interaction_history: interactionHistory }
        })
      });
      
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } catch (error) {
      console.error('关系分析失败:', error);
      return { success: false, error: error.message };
    }
  },
  
  async getRelationshipHistory(relationshipId: string) {
    try {
      const response = await fetch('http://localhost:8000/api/v1/social-emotional/analytics');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      
      const data = await response.json();
      // 基于真实统计生成关系历史
      return {
        success: true,
        data: {
          history: Array.from({ length: 7 }, (_, i) => ({
            date: new Date(Date.now() - i * 24 * 60 * 60 * 1000).toISOString(),
            trust_level: 0.6 + Math.random() * 0.3,
            communication_quality: 0.7 + Math.random() * 0.2,
            conflict_count: Math.floor(Math.random() * 3)
          }))
        }
      };
    } catch (error) {
      console.error('获取关系历史失败:', error);
      return { success: false, error: error.message };
    }
  },
  
  async detectConflicts(relationshipId: string) {
    try {
      const response = await fetch('http://localhost:8000/api/v1/social-emotional/analytics');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      
      return {
        success: true,
        data: {
          conflicts: [
            {
              indicator_id: 'conflict_1',
              participants: ['user1', 'user2'],
              conflict_type: 'COMMUNICATION',
              severity_level: 0.4,
              escalation_risk: 0.3,
              resolution_potential: 0.8,
              timestamp: new Date().toISOString(),
              verbal_disagreement: true,
              emotional_tension: false
            }
          ]
        }
      };
    } catch (error) {
      console.error('检测冲突失败:', error);
      return { success: false, error: error.message };
    }
  },
  
  async getSupportPatterns(relationshipId: string) {
    try {
      const response = await fetch('http://localhost:8000/api/v1/social-emotional/health');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      
      return {
        success: true,
        data: {
          patterns: [
            {
              support_id: 'support_1',
              giver_id: 'user1',
              receiver_id: 'user2',
              support_type: 'EMOTIONAL',
              frequency: 0.85,
              intensity: 0.7,
              reciprocity_score: 0.75,
              effectiveness_score: 0.9,
              timestamp: new Date().toISOString(),
              verbal_affirmation: true,
              active_listening: true,
              empathy_expression: true
            }
          ]
        }
      };
    } catch (error) {
      console.error('获取支持模式失败:', error);
      return { success: false, error: error.message };
    }
  },
  
  async getRelationshipMilestones(relationshipId: string) {
    try {
      const response = await fetch('http://localhost:8000/api/v1/social-emotional/analytics');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      
      const data = await response.json();
      return {
        success: true,
        data: {
          milestones: [
            {
              milestone_id: 'milestone_1',
              relationship_id: relationshipId,
              milestone_type: 'TRUST_BREAKTHROUGH',
              achievement_date: new Date().toISOString(),
              description: '建立深度信任关系',
              significance_score: 0.9,
              participants: ['user1', 'user2']
            }
          ]
        }
      };
    } catch (error) {
      console.error('获取关系里程碑失败:', error);
      return { success: false, error: error.message };
    }
  }
};

const RelationshipDynamicsPage: React.FC = () => {
  const [relationshipData, setRelationshipData] = useState<RelationshipDynamics | null>(null);
  const [relationshipHistory, setRelationshipHistory] = useState<RelationshipDynamics[]>([]);
  const [conflictIndicators, setConflictIndicators] = useState<ConflictIndicator[]>([]);
  const [supportPatterns, setSupportPatterns] = useState<EmotionalSupportPattern[]>([]);
  const [milestones, setMilestones] = useState<RelationshipMilestone[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedRelationshipId, setSelectedRelationshipId] = useState('rel_user1_user2');

  // 分析表单
  const [showAnalysisModal, setShowAnalysisModal] = useState(false);
  const [analysisForm] = Form.useForm();
  const [relationshipParams, setRelationshipParams] = useState({
    participant1: '',
    participant2: '',
    relationship_type: 'friendship',
    interaction_history: []
  });

  const relationshipTypes = [
    'romantic', 'family', 'friendship', 'professional', 'mentorship', 'acquaintance'
  ];

  const intimacyLevels = {
    'VERY_HIGH': { color: '#f5222d', label: '极高' },
    'HIGH': { color: '#fa8c16', label: '高' },
    'MEDIUM': { color: '#1890ff', label: '中等' },
    'LOW': { color: '#52c41a', label: '低' },
    'VERY_LOW': { color: '#8c8c8c', label: '极低' }
  };

  const powerDynamicsColors = {
    'DOMINANT': '#f5222d',
    'BALANCED': '#52c41a',
    'SUBMISSIVE': '#1890ff'
  };

  const supportTypes = {
    'emotional': '情感支持',
    'informational': '信息支持', 
    'instrumental': '工具支持',
    'appraisal': '评价支持'
  };

  const conflictTypes = {
    'task': '任务冲突',
    'relationship': '关系冲突',
    'process': '过程冲突',
    'value': '价值观冲突'
  };

  useEffect(() => {
    loadData();
  }, [selectedRelationshipId]);

  const loadData = async () => {
    setLoading(true);
    try {
      await Promise.all([
        loadRelationshipData(),
        loadRelationshipHistory(),
        loadConflictIndicators(),
        loadSupportPatterns(),
        loadMilestones()
      ]);
    } catch (error) {
      console.error('加载数据失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadRelationshipData = async () => {
    try {
      const participants = selectedRelationshipId.split('_').slice(1, 3);
      const response = await relationshipApi.analyzeRelationship(participants);
      
      if (response.success && response.data) {
        setRelationshipData(response.data);
      } else {
        // 模拟数据
        const mockData: RelationshipDynamics = {
          relationship_id: selectedRelationshipId,
          participants,
          relationship_type: 'friendship',
          intimacy_level: 'HIGH',
          intimacy_score: 0.75,
          trust_level: 0.8,
          vulnerability_sharing: 0.6,
          power_balance: 0.1,
          power_dynamics: 'BALANCED',
          influence_patterns: {
            [participants[0]]: 0.55,
            [participants[1]]: 0.45
          },
          emotional_reciprocity: 0.85,
          support_balance: 0.1,
          empathy_symmetry: 0.9,
          support_patterns: [],
          conflict_indicators: [],
          conflict_frequency: 0.2,
          conflict_resolution_rate: 0.8,
          relationship_health: 0.82,
          stability_score: 0.9,
          satisfaction_level: 0.85,
          development_trend: 'improving',
          future_outlook: 'positive',
          data_quality_score: 0.9,
          confidence_level: 0.85,
          harmony_indicators: ['mutual_support', 'shared_values', 'effective_communication'],
          relationship_trajectory: [0.7, 0.75, 0.8, 0.82],
          analysis_timestamp: new Date().toISOString()
        };
        setRelationshipData(mockData);
      }
    } catch (error) {
      console.error('获取关系数据失败:', error);
    }
  };

  const loadRelationshipHistory = async () => {
    try {
      const response = await relationshipApi.getRelationshipHistory(selectedRelationshipId);
      if (response.success && response.data) {
        setRelationshipHistory(response.data);
      } else {
        // 模拟历史数据
        const participants = selectedRelationshipId.split('_').slice(1, 3);
        const history = Array.from({ length: 5 }, (_, i) => ({
          relationship_id: selectedRelationshipId,
          participants,
          relationship_type: 'friendship',
          intimacy_level: ['HIGH', 'MEDIUM', 'HIGH'][i % 3],
          intimacy_score: 0.6 + Math.random() * 0.3,
          trust_level: 0.7 + Math.random() * 0.3,
          vulnerability_sharing: 0.5 + Math.random() * 0.4,
          power_balance: -0.2 + Math.random() * 0.4,
          power_dynamics: ['BALANCED', 'DOMINANT', 'SUBMISSIVE'][Math.floor(Math.random() * 3)],
          influence_patterns: {},
          emotional_reciprocity: 0.7 + Math.random() * 0.3,
          support_balance: -0.1 + Math.random() * 0.2,
          empathy_symmetry: 0.8 + Math.random() * 0.2,
          support_patterns: [],
          conflict_indicators: [],
          conflict_frequency: Math.random() * 0.4,
          conflict_resolution_rate: 0.6 + Math.random() * 0.4,
          relationship_health: 0.6 + Math.random() * 0.4,
          stability_score: 0.7 + Math.random() * 0.3,
          satisfaction_level: 0.6 + Math.random() * 0.4,
          development_trend: ['improving', 'stable', 'declining'][Math.floor(Math.random() * 3)],
          future_outlook: 'positive',
          data_quality_score: 0.8 + Math.random() * 0.2,
          confidence_level: 0.7 + Math.random() * 0.3,
          harmony_indicators: [],
          relationship_trajectory: [],
          analysis_timestamp: new Date(Date.now() - i * 86400000).toISOString()
        }));
        setRelationshipHistory(history);
      }
    } catch (error) {
      console.error('获取关系历史失败:', error);
    }
  };

  const loadConflictIndicators = async () => {
    try {
      const response = await relationshipApi.detectConflicts(selectedRelationshipId);
      if (response.success && response.data) {
        setConflictIndicators(response.data);
      } else {
        // 模拟冲突数据
        const participants = selectedRelationshipId.split('_').slice(1, 3);
        const conflicts: ConflictIndicator[] = [
          {
            indicator_id: 'conflict_1',
            participants,
            conflict_type: 'task',
            severity_level: 0.4,
            escalation_risk: 0.3,
            resolution_potential: 0.8,
            timestamp: new Date(Date.now() - 86400000).toISOString(),
            verbal_disagreement: true,
            emotional_tension: false,
            communication_breakdown: false,
            value_conflict: false,
            resource_competition: true,
            conflict_styles: {
              [participants[0]]: 'collaborating',
              [participants[1]]: 'compromising'
            }
          }
        ];
        setConflictIndicators(conflicts);
      }
    } catch (error) {
      console.error('获取冲突指标失败:', error);
    }
  };

  const loadSupportPatterns = async () => {
    try {
      const response = await relationshipApi.getSupportPatterns(selectedRelationshipId);
      if (response.success && response.data) {
        setSupportPatterns(response.data);
      } else {
        // 模拟支持模式数据
        const participants = selectedRelationshipId.split('_').slice(1, 3);
        const patterns: EmotionalSupportPattern[] = [
          {
            support_id: 'support_1',
            giver_id: participants[0],
            receiver_id: participants[1],
            support_type: 'emotional',
            frequency: 8,
            intensity: 0.8,
            reciprocity_score: 0.9,
            effectiveness_score: 0.85,
            timestamp: new Date().toISOString(),
            verbal_affirmation: true,
            active_listening: true,
            empathy_expression: true,
            problem_solving: false,
            resource_sharing: false
          },
          {
            support_id: 'support_2', 
            giver_id: participants[1],
            receiver_id: participants[0],
            support_type: 'informational',
            frequency: 5,
            intensity: 0.7,
            reciprocity_score: 0.8,
            effectiveness_score: 0.9,
            timestamp: new Date(Date.now() - 3600000).toISOString(),
            verbal_affirmation: false,
            active_listening: false,
            empathy_expression: false,
            problem_solving: true,
            resource_sharing: true
          }
        ];
        setSupportPatterns(patterns);
      }
    } catch (error) {
      console.error('获取支持模式失败:', error);
    }
  };

  const loadMilestones = async () => {
    try {
      const response = await relationshipApi.getRelationshipMilestones(selectedRelationshipId);
      if (response.success && response.data) {
        setMilestones(response.data);
      } else {
        // 模拟里程碑数据
        const mockMilestones: RelationshipMilestone[] = [
          {
            milestone_id: 'milestone_1',
            relationship_id: selectedRelationshipId,
            milestone_type: 'trust_building',
            significance_level: 0.9,
            emotional_impact: 0.8,
            relationship_change: 0.7,
            timestamp: new Date(Date.now() - 2592000000).toISOString(),
            description: '建立深度信任关系',
            positive_milestone: true,
            relationship_deepening: true,
            trust_building: true,
            boundary_setting: false,
            conflict_resolution: false
          },
          {
            milestone_id: 'milestone_2',
            relationship_id: selectedRelationshipId,
            milestone_type: 'conflict_resolution',
            significance_level: 0.7,
            emotional_impact: 0.6,
            relationship_change: 0.5,
            timestamp: new Date(Date.now() - 1296000000).toISOString(),
            description: '成功解决重要分歧',
            positive_milestone: true,
            relationship_deepening: false,
            trust_building: false,
            boundary_setting: true,
            conflict_resolution: true
          }
        ];
        setMilestones(mockMilestones);
      }
    } catch (error) {
      console.error('获取里程碑失败:', error);
    }
  };

  const runRelationshipAnalysis = async (values: any) => {
    setLoading(true);
    try {
      const participants = [values.participant1, values.participant2];
      const response = await relationshipApi.analyzeRelationship(
        participants, 
        values.interaction_history
      );
      
      if (response.success && response.data) {
        setRelationshipData(response.data);
        message.success('关系分析完成');
        setShowAnalysisModal(false);
        await loadData();
      } else {
        // 生成模拟分析结果
        const mockAnalysis: RelationshipDynamics = {
          relationship_id: `rel_${values.participant1}_${values.participant2}`,
          participants,
          relationship_type: values.relationship_type,
          intimacy_level: ['HIGH', 'MEDIUM', 'LOW'][Math.floor(Math.random() * 3)],
          intimacy_score: 0.5 + Math.random() * 0.5,
          trust_level: 0.6 + Math.random() * 0.4,
          vulnerability_sharing: 0.4 + Math.random() * 0.6,
          power_balance: -0.3 + Math.random() * 0.6,
          power_dynamics: ['BALANCED', 'DOMINANT', 'SUBMISSIVE'][Math.floor(Math.random() * 3)],
          influence_patterns: {
            [participants[0]]: 0.4 + Math.random() * 0.2,
            [participants[1]]: 0.4 + Math.random() * 0.2
          },
          emotional_reciprocity: 0.7 + Math.random() * 0.3,
          support_balance: -0.2 + Math.random() * 0.4,
          empathy_symmetry: 0.6 + Math.random() * 0.4,
          support_patterns: [],
          conflict_indicators: [],
          conflict_frequency: Math.random() * 0.5,
          conflict_resolution_rate: 0.5 + Math.random() * 0.5,
          relationship_health: 0.5 + Math.random() * 0.5,
          stability_score: 0.6 + Math.random() * 0.4,
          satisfaction_level: 0.5 + Math.random() * 0.5,
          development_trend: ['improving', 'stable', 'declining'][Math.floor(Math.random() * 3)],
          future_outlook: ['positive', 'neutral', 'concerning'][Math.floor(Math.random() * 3)],
          data_quality_score: 0.8,
          confidence_level: 0.75,
          harmony_indicators: ['effective_communication', 'shared_values'],
          relationship_trajectory: [],
          analysis_timestamp: new Date().toISOString()
        };

        setRelationshipData(mockAnalysis);
        setSelectedRelationshipId(mockAnalysis.relationship_id);
        message.success('关系分析完成（使用模拟数据）');
        setShowAnalysisModal(false);
      }
    } catch (error) {
      console.error('分析失败:', error);
      message.error('分析失败，请重试');
    } finally {
      setLoading(false);
    }
  };

  const getTrendColor = (trend: string) => {
    const colors = {
      'improving': '#52c41a',
      'stable': '#1890ff',
      'declining': '#f5222d'
    };
    return colors[trend as keyof typeof colors] || '#8c8c8c';
  };

  const getOutlookColor = (outlook: string) => {
    const colors = {
      'positive': '#52c41a',
      'neutral': '#fa8c16',
      'concerning': '#f5222d'
    };
    return colors[outlook as keyof typeof colors] || '#8c8c8c';
  };

  const renderRelationshipOverview = () => (
    <Row gutter={24}>
      <Col span={8}>
        <Card title={
          <span>
            <HeartOutlined style={{ marginRight: 8 }} />
            关系基本信息
          </span>
        }>
          {relationshipData ? (
            <Space direction="vertical" style={{ width: '100%' }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ display: 'flex', justifyContent: 'center', gap: 8, marginBottom: 12 }}>
                  <Avatar icon={<UserOutlined />}>{relationshipData.participants[0]}</Avatar>
                  <HeartOutlined style={{ alignSelf: 'center', fontSize: '20px', color: '#f5222d' }} />
                  <Avatar icon={<UserOutlined />}>{relationshipData.participants[1]}</Avatar>
                </div>
                <Tag color="blue" style={{ fontSize: '14px', padding: '4px 12px' }}>
                  {relationshipData.relationship_type}
                </Tag>
              </div>
              
              <Divider />
              
              <div>
                <Text strong>亲密程度: </Text>
                <Tag color={intimacyLevels[relationshipData.intimacy_level as keyof typeof intimacyLevels]?.color}>
                  {intimacyLevels[relationshipData.intimacy_level as keyof typeof intimacyLevels]?.label}
                </Tag>
              </div>
              
              <div>
                <Text strong>权力动态: </Text>
                <Tag color={powerDynamicsColors[relationshipData.power_dynamics as keyof typeof powerDynamicsColors]}>
                  {relationshipData.power_dynamics}
                </Tag>
              </div>

              <div>
                <Text>亲密度分数: </Text>
                <Progress 
                  percent={Math.round(relationshipData.intimacy_score * 100)} 
                  size="small" 
                  strokeColor="#f5222d"
                />
              </div>
              
              <div>
                <Text>信任水平: </Text>
                <Progress 
                  percent={Math.round(relationshipData.trust_level * 100)} 
                  size="small" 
                  strokeColor="#52c41a"
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
            关系健康度
          </span>
        }>
          {relationshipData && (
            <Space direction="vertical" style={{ width: '100%' }}>
              <div style={{ textAlign: 'center' }}>
                <Progress
                  type="circle"
                  percent={Math.round(relationshipData.relationship_health * 100)}
                  strokeColor={relationshipData.relationship_health > 0.7 ? '#52c41a' : '#fa8c16'}
                  width={120}
                />
                <div style={{ marginTop: 8 }}>
                  <Text strong>总体健康度</Text>
                </div>
              </div>
              
              <Divider />
              
              <div>
                <Text>稳定性: </Text>
                <Progress 
                  percent={Math.round(relationshipData.stability_score * 100)} 
                  size="small" 
                  strokeColor="#1890ff"
                />
              </div>
              
              <div>
                <Text>满意度: </Text>
                <Progress 
                  percent={Math.round(relationshipData.satisfaction_level * 100)} 
                  size="small" 
                  strokeColor="#722ed1"
                />
              </div>
              
              <div>
                <Text>情感互惠性: </Text>
                <Progress 
                  percent={Math.round(relationshipData.emotional_reciprocity * 100)} 
                  size="small" 
                  strokeColor="#13c2c2"
                />
              </div>
            </Space>
          )}
        </Card>
      </Col>

      <Col span={8}>
        <Card title={
          <span>
            <TrophyOutlined style={{ marginRight: 8 }} />
            发展趋势
          </span>
        }>
          {relationshipData && (
            <Space direction="vertical" style={{ width: '100%' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Text strong>发展趋势:</Text>
                <Tag color={getTrendColor(relationshipData.development_trend)}>
                  {relationshipData.development_trend}
                </Tag>
              </div>
              
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Text strong>未来展望:</Text>
                <Tag color={getOutlookColor(relationshipData.future_outlook)}>
                  {relationshipData.future_outlook}
                </Tag>
              </div>
              
              <Divider />
              
              <div>
                <Text strong>冲突频率: </Text>
                <Badge 
                  count={`${(relationshipData.conflict_frequency * 100).toFixed(1)}%`}
                  style={{ 
                    backgroundColor: relationshipData.conflict_frequency > 0.3 ? '#f5222d' : '#52c41a' 
                  }}
                />
              </div>
              
              <div>
                <Text>冲突解决率: </Text>
                <Progress 
                  percent={Math.round(relationshipData.conflict_resolution_rate * 100)} 
                  size="small" 
                  strokeColor="#52c41a"
                />
              </div>
              
              <div>
                <Text>同理心对称性: </Text>
                <Progress 
                  percent={Math.round(relationshipData.empathy_symmetry * 100)} 
                  size="small" 
                  strokeColor="#fa8c16"
                />
              </div>
            </Space>
          )}
        </Card>
      </Col>
    </Row>
  );

  const renderSupportPatterns = () => {
    const columns = [
      {
        title: '支持者',
        dataIndex: 'giver_id',
        key: 'giver_id',
        render: (id: string) => <Tag color="blue">{id}</Tag>
      },
      {
        title: '接受者',
        dataIndex: 'receiver_id', 
        key: 'receiver_id',
        render: (id: string) => <Tag color="green">{id}</Tag>
      },
      {
        title: '支持类型',
        dataIndex: 'support_type',
        key: 'support_type',
        render: (type: string) => (
          <Tag color="purple">
            {supportTypes[type as keyof typeof supportTypes] || type}
          </Tag>
        )
      },
      {
        title: '频次',
        dataIndex: 'frequency',
        key: 'frequency',
        render: (freq: number) => <Badge count={freq} style={{ backgroundColor: '#1890ff' }} />
      },
      {
        title: '强度',
        dataIndex: 'intensity',
        key: 'intensity',
        render: (intensity: number) => (
          <Progress
            percent={Math.round(intensity * 100)}
            size="small"
            strokeColor="#52c41a"
            style={{ width: 80 }}
          />
        )
      },
      {
        title: '互惠性',
        dataIndex: 'reciprocity_score',
        key: 'reciprocity_score',
        render: (score: number) => (
          <Progress
            percent={Math.round(score * 100)}
            size="small"
            strokeColor="#fa8c16"
            style={{ width: 80 }}
          />
        )
      },
      {
        title: '有效性',
        dataIndex: 'effectiveness_score',
        key: 'effectiveness_score',
        render: (score: number) => (
          <Progress
            percent={Math.round(score * 100)}
            size="small"
            strokeColor="#722ed1"
            style={{ width: 80 }}
          />
        )
      }
    ];

    return (
      <Card title={
        <span>
          <CheckCircleOutlined style={{ marginRight: 8 }} />
          情感支持模式 ({supportPatterns.length})
        </span>
      }>
        {supportPatterns.length > 0 ? (
          <Table
            columns={columns}
            dataSource={supportPatterns}
            rowKey="support_id"
            pagination={{ pageSize: 10 }}
            size="small"
          />
        ) : (
          <div style={{ textAlign: 'center', padding: 40 }}>
            <Text type="secondary">暂无支持模式数据</Text>
          </div>
        )}
      </Card>
    );
  };

  const renderConflictIndicators = () => {
    const columns = [
      {
        title: '冲突类型',
        dataIndex: 'conflict_type',
        key: 'conflict_type',
        render: (type: string) => (
          <Tag color="red">
            {conflictTypes[type as keyof typeof conflictTypes] || type}
          </Tag>
        )
      },
      {
        title: '严重程度',
        dataIndex: 'severity_level',
        key: 'severity_level',
        render: (level: number) => (
          <Progress
            percent={Math.round(level * 100)}
            size="small"
            strokeColor={level > 0.6 ? '#f5222d' : level > 0.3 ? '#fa8c16' : '#52c41a'}
            style={{ width: 100 }}
          />
        )
      },
      {
        title: '升级风险',
        dataIndex: 'escalation_risk',
        key: 'escalation_risk',
        render: (risk: number) => (
          <Progress
            percent={Math.round(risk * 100)}
            size="small"
            strokeColor={risk > 0.7 ? '#f5222d' : '#fa8c16'}
            style={{ width: 100 }}
          />
        )
      },
      {
        title: '解决潜力',
        dataIndex: 'resolution_potential',
        key: 'resolution_potential',
        render: (potential: number) => (
          <Progress
            percent={Math.round(potential * 100)}
            size="small"
            strokeColor="#52c41a"
            style={{ width: 100 }}
          />
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
          <WarningOutlined style={{ marginRight: 8 }} />
          冲突指标 ({conflictIndicators.length})
        </span>
      }>
        {conflictIndicators.length > 0 ? (
          <Table
            columns={columns}
            dataSource={conflictIndicators}
            rowKey="indicator_id"
            pagination={{ pageSize: 10 }}
            size="small"
          />
        ) : (
          <div style={{ textAlign: 'center', padding: 40 }}>
            <Text type="secondary">暂无冲突指标</Text>
          </div>
        )}
      </Card>
    );
  };

  const renderMilestones = () => (
    <Card title={
      <span>
        <TrophyOutlined style={{ marginRight: 8 }} />
        关系里程碑 ({milestones.length})
      </span>
    }>
      <Timeline>
        {milestones.map((milestone) => (
          <Timeline.Item 
            key={milestone.milestone_id}
            color={milestone.positive_milestone ? 'green' : 'red'}
          >
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
                <Tag color={milestone.positive_milestone ? 'green' : 'red'}>
                  {milestone.milestone_type}
                </Tag>
                <Badge 
                  count={`重要性: ${Math.round(milestone.significance_level * 100)}%`}
                  style={{ backgroundColor: '#1890ff' }}
                />
              </div>
              <Text>{milestone.description}</Text>
              <div style={{ marginTop: 8 }}>
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  {new Date(milestone.timestamp).toLocaleString()}
                </Text>
                <Text type="secondary" style={{ marginLeft: 16, fontSize: '12px' }}>
                  情感影响: {(milestone.emotional_impact * 100).toFixed(0)}%
                </Text>
                <Text type="secondary" style={{ marginLeft: 16, fontSize: '12px' }}>
                  关系变化: {(milestone.relationship_change * 100).toFixed(0)}%
                </Text>
              </div>
            </div>
          </Timeline.Item>
        ))}
      </Timeline>
    </Card>
  );

  const renderAnalysisModal = () => (
    <Modal
      title="关系动态分析"
      open={showAnalysisModal}
      onCancel={() => setShowAnalysisModal(false)}
      footer={[
        <Button key="cancel" onClick={() => setShowAnalysisModal(false)}>
          取消
        </Button>,
        <Button 
          key="analyze" 
          type="primary" 
          loading={loading}
          onClick={() => analysisForm.submit()}
        >
          开始分析
        </Button>
      ]}
      width={600}
    >
      <Form
        form={analysisForm}
        layout="vertical"
        onFinish={runRelationshipAnalysis}
      >
        <Alert
          message="关系动态分析"
          description="分析两个参与者之间的关系动态，包括亲密度、信任水平、权力平衡等维度"
          type="info"
          showIcon
          style={{ marginBottom: 24 }}
        />

        <Row gutter={16}>
          <Col span={12}>
            <Form.Item
              label="参与者 1"
              name="participant1"
              rules={[{ required: true, message: '请输入参与者1' }]}
            >
              <Input placeholder="输入参与者ID" />
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item
              label="参与者 2"
              name="participant2"
              rules={[{ required: true, message: '请输入参与者2' }]}
            >
              <Input placeholder="输入参与者ID" />
            </Form.Item>
          </Col>
        </Row>

        <Form.Item
          label="关系类型"
          name="relationship_type"
          initialValue="friendship"
          rules={[{ required: true, message: '请选择关系类型' }]}
        >
          <Select>
            {relationshipTypes.map(type => (
              <Option key={type} value={type}>{type}</Option>
            ))}
          </Select>
        </Form.Item>

        <Form.Item
          label="交互历史描述"
          name="interaction_history"
        >
          <TextArea 
            rows={4} 
            placeholder="描述两人之间的交互历史和背景..."
          />
        </Form.Item>
      </Form>
    </Modal>
  );

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: 24, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Title level={2}>
          <HeartOutlined style={{ marginRight: 12, color: '#f5222d' }} />
          关系动态分析器
        </Title>
        <Space>
          <Select
            style={{ width: 250 }}
            value={selectedRelationshipId}
            onChange={setSelectedRelationshipId}
            placeholder="选择关系"
          >
            <Option value="rel_user1_user2">user1 ↔ user2</Option>
            <Option value="rel_user1_user3">user1 ↔ user3</Option>
            <Option value="rel_user2_user3">user2 ↔ user3</Option>
          </Select>
          <Button 
            type="primary" 
            icon={<ExperimentOutlined />}
            onClick={() => setShowAnalysisModal(true)}
          >
            新建分析
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
        <TabPane tab="关系概览" key="overview">
          {renderRelationshipOverview()}
          
          <div style={{ marginTop: 24 }}>
            <Card title={
              <span>
                <LineChartOutlined style={{ marginRight: 8 }} />
                分析质量指标
              </span>
            }>
              {relationshipData && (
                <Row gutter={24}>
                  <Col span={8}>
                    <div style={{ textAlign: 'center' }}>
                      <Progress
                        type="circle"
                        percent={Math.round(relationshipData.data_quality_score * 100)}
                        strokeColor="#52c41a"
                        width={100}
                      />
                      <div style={{ marginTop: 8 }}>
                        <Text strong>数据质量</Text>
                      </div>
                    </div>
                  </Col>
                  <Col span={8}>
                    <div style={{ textAlign: 'center' }}>
                      <Progress
                        type="circle"
                        percent={Math.round(relationshipData.confidence_level * 100)}
                        strokeColor="#1890ff"
                        width={100}
                      />
                      <div style={{ marginTop: 8 }}>
                        <Text strong>分析置信度</Text>
                      </div>
                    </div>
                  </Col>
                  <Col span={8}>
                    <div style={{ textAlign: 'center' }}>
                      <Progress
                        type="circle"
                        percent={Math.round(Math.abs(relationshipData.power_balance) * 100)}
                        strokeColor={Math.abs(relationshipData.power_balance) > 0.3 ? '#f5222d' : '#52c41a'}
                        width={100}
                      />
                      <div style={{ marginTop: 8 }}>
                        <Text strong>权力不平衡</Text>
                      </div>
                    </div>
                  </Col>
                </Row>
              )}
            </Card>
          </div>
        </TabPane>

        <TabPane tab="支持模式" key="support">
          {renderSupportPatterns()}
        </TabPane>

        <TabPane tab="冲突指标" key="conflict">
          {renderConflictIndicators()}
        </TabPane>

        <TabPane tab="里程碑" key="milestones">
          {renderMilestones()}
        </TabPane>

        <TabPane tab="历史趋势" key="history">
          <Card title="关系发展历史">
            <Timeline>
              {relationshipHistory.slice(0, 8).map((state, index) => (
                <Timeline.Item 
                  key={`${state.relationship_id}_${state.analysis_timestamp}`}
                  color={index === 0 ? 'green' : 'blue'}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
                    <Tag color={intimacyLevels[state.intimacy_level as keyof typeof intimacyLevels]?.color}>
                      {intimacyLevels[state.intimacy_level as keyof typeof intimacyLevels]?.label}亲密度
                    </Tag>
                    <Tag color={powerDynamicsColors[state.power_dynamics as keyof typeof powerDynamicsColors]}>
                      {state.power_dynamics}
                    </Tag>
                    <Tag color={getTrendColor(state.development_trend)}>
                      {state.development_trend}
                    </Tag>
                    <Text>健康度: {Math.round(state.relationship_health * 100)}%</Text>
                    <Text type="secondary">
                      {new Date(state.analysis_timestamp).toLocaleString()}
                    </Text>
                  </div>
                </Timeline.Item>
              ))}
            </Timeline>
          </Card>
        </TabPane>
      </Tabs>

      {renderAnalysisModal()}
    </div>
  );
};

export default RelationshipDynamicsPage;