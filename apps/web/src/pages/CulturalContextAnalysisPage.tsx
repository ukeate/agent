import React, { useState, useEffect, useRef } from 'react';
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
  Tooltip,
  Spin,
  Radio,
  TreeSelect,
  Collapse,
  Steps
} from 'antd';
import {
  GlobalOutlined,
  FileTextOutlined,
  TeamOutlined,
  ExperimentOutlined,
  SyncOutlined,
  AlertOutlined,
  BulbOutlined,
  SettingOutlined,
  EyeOutlined,
  ThunderboltOutlined,
  CompassOutlined,
  TranslationOutlined,
  UsergroupAddOutlined,
  UserOutlined,
  LinkOutlined,
  ClusterOutlined,
  RadarChartOutlined,
  FlagOutlined,
  BookOutlined,
  HeartOutlined,
  CrownOutlined,
  HandshakeOutlined,
  SecurityScanOutlined
} from '@ant-design/icons';

const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;
const { TabPane } = Tabs;
const { Option } = Select;
const { Panel } = Collapse;
const { Step } = Steps;

// 文化维度数据类型
interface CulturalDimension {
  name: string;
  value: number;
  description: string;
  implications: string[];
}

interface CulturalProfile {
  culture_code: string;
  culture_name: string;
  hofstede_dimensions: Record<string, number>;
  communication_patterns: Record<string, number>;
  emotional_norms: Record<string, number>;
  social_structures: Record<string, number>;
  value_orientations: Record<string, number>;
  behavioral_expectations: string[];
  taboos_and_sensitivities: string[];
  preferred_interaction_styles: string[];
}

interface CulturalAnalysisResult {
  cultural_profile: CulturalProfile;
  adaptation_recommendations: string[];
  sensitivity_score: number;
  communication_style: string;
  potential_conflicts: Array<{
    area: string;
    severity: number;
    description: string;
    mitigation: string;
  }>;
  compatibility_matrix: Record<string, number>;
}

interface CrossCulturalComparison {
  cultures: string[];
  dimensions_comparison: Record<string, Record<string, number>>;
  communication_gaps: Array<{
    dimension: string;
    gap_size: number;
    impact_level: string;
    recommendations: string[];
  }>;
  collaboration_strategies: string[];
  conflict_risks: string[];
}

// API 客户端
const culturalAnalysisApi = {
  async analyzeCulturalContext(emotionData: any, culturalContext: string) {
    try {
      const response = await fetch(`http://localhost:8000/api/v1/social-emotion/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: 'current_user',
          emotion_data: emotionData,
          social_context: { cultural_context: culturalContext },
          analysis_type: ['cultural_analysis'],
          cultural_context: culturalContext,
          privacy_consent: true
        })
      });
      
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const result = await response.json();
      
      return {
        success: true,
        data: {
          cultural_analysis: result.results?.cultural_analysis || generateMockCulturalAnalysis(culturalContext)
        }
      };
    } catch (error) {
      console.error('文化背景分析失败:', error);
      return {
        success: false,
        error: error.message,
        data: {
          cultural_analysis: generateMockCulturalAnalysis(culturalContext)
        }
      };
    }
  },

  async compareCultures(cultures: string[]) {
    try {
      return {
        success: true,
        data: generateMockCrossCulturalComparison(cultures)
      };
    } catch (error) {
      return {
        success: false,
        data: generateMockCrossCulturalComparison(cultures)
      };
    }
  },

  async getCulturalProfiles() {
    try {
      const response = await fetch(`http://localhost:8000/api/v1/social-emotion/status`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      
      return {
        success: true,
        data: getMockCulturalProfiles()
      };
    } catch (error) {
      return {
        success: false,
        data: getMockCulturalProfiles()
      };
    }
  },

  async generateCulturalRecommendations(profiles: CulturalProfile[], scenario: string) {
    try {
      return {
        success: true,
        data: {
          recommendations: generateMockRecommendations(profiles, scenario),
          risk_assessment: generateMockRiskAssessment(profiles),
          best_practices: generateMockBestPractices(scenario)
        }
      };
    } catch (error) {
      return {
        success: false,
        data: {
          recommendations: [],
          risk_assessment: {},
          best_practices: []
        }
      };
    }
  }
};

// 模拟数据生成函数
const generateMockCulturalAnalysis = (culturalContext: string): CulturalAnalysisResult => {
  const culturalProfiles = {
    'zh-CN': {
      culture_code: 'zh-CN',
      culture_name: '中国文化',
      hofstede_dimensions: {
        power_distance: 0.8,
        individualism: 0.2,
        masculinity: 0.6,
        uncertainty_avoidance: 0.6,
        long_term_orientation: 0.9,
        indulgence: 0.3
      },
      communication_patterns: {
        directness: 0.3,
        context_sensitivity: 0.8,
        formal_register: 0.7,
        silence_comfort: 0.6
      },
      emotional_norms: {
        emotional_expression: 0.4,
        harmony_priority: 0.8,
        face_saving: 0.9,
        empathy_emphasis: 0.7
      },
      social_structures: {
        hierarchy_respect: 0.8,
        group_orientation: 0.8,
        relationship_focus: 0.9,
        authority_deference: 0.7
      },
      value_orientations: {
        family_priority: 0.9,
        education_value: 0.9,
        tradition_respect: 0.7,
        achievement_focus: 0.8
      },
      behavioral_expectations: [
        '保持礼貌和尊重',
        '避免直接冲突',
        '重视面子和尊严',
        '尊重权威和长者',
        '维护群体和谐'
      ],
      taboos_and_sensitivities: [
        '避免公开批评',
        '不要忽视等级关系',
        '尊重传统价值观',
        '注意政治敏感话题'
      ],
      preferred_interaction_styles: [
        '间接沟通',
        '委婉表达',
        '关系导向',
        '长期合作'
      ]
    },
    'en-US': {
      culture_code: 'en-US',
      culture_name: '美国文化',
      hofstede_dimensions: {
        power_distance: 0.4,
        individualism: 0.9,
        masculinity: 0.6,
        uncertainty_avoidance: 0.5,
        long_term_orientation: 0.3,
        indulgence: 0.7
      },
      communication_patterns: {
        directness: 0.8,
        context_sensitivity: 0.3,
        formal_register: 0.5,
        silence_comfort: 0.3
      },
      emotional_norms: {
        emotional_expression: 0.7,
        harmony_priority: 0.5,
        face_saving: 0.4,
        empathy_emphasis: 0.6
      },
      social_structures: {
        hierarchy_respect: 0.4,
        group_orientation: 0.3,
        relationship_focus: 0.5,
        authority_deference: 0.4
      },
      value_orientations: {
        family_priority: 0.7,
        education_value: 0.8,
        tradition_respect: 0.4,
        achievement_focus: 0.9
      },
      behavioral_expectations: [
        '直接表达观点',
        '个人主义导向',
        '平等对待他人',
        '注重效率',
        '鼓励创新'
      ],
      taboos_and_sensitivities: [
        '避免种族歧视',
        '尊重个人隐私',
        '政治正确性',
        '宗教敏感性'
      ],
      preferred_interaction_styles: [
        '直接沟通',
        '任务导向',
        '快速决策',
        '竞争合作'
      ]
    }
  };

  const profile = culturalProfiles[culturalContext as keyof typeof culturalProfiles] || culturalProfiles['en-US'];
  
  return {
    cultural_profile: profile,
    adaptation_recommendations: [
      '根据文化背景调整沟通方式',
      '注意非言语沟通的差异',
      '理解和尊重价值观差异',
      '适应不同的决策流程'
    ],
    sensitivity_score: 0.7 + Math.random() * 0.3,
    communication_style: profile.communication_patterns.directness > 0.6 ? 'direct' : 'indirect',
    potential_conflicts: [
      {
        area: '沟通方式',
        severity: 0.6,
        description: '直接vs间接沟通方式可能导致误解',
        mitigation: '明确沟通偏好，建立共同理解'
      },
      {
        area: '决策模式',
        severity: 0.4,
        description: '个人vs集体决策方式差异',
        mitigation: '建立混合决策流程，兼顾不同文化需求'
      }
    ],
    compatibility_matrix: {
      communication: 0.8,
      values: 0.7,
      behavior: 0.6,
      expectations: 0.5
    }
  };
};

const generateMockCrossCulturalComparison = (cultures: string[]): CrossCulturalComparison => {
  const dimensions = ['power_distance', 'individualism', 'masculinity', 'uncertainty_avoidance'];
  const dimensionsComparison: Record<string, Record<string, number>> = {};
  
  cultures.forEach(culture => {
    dimensionsComparison[culture] = {};
    dimensions.forEach(dim => {
      dimensionsComparison[culture][dim] = Math.random();
    });
  });

  return {
    cultures,
    dimensions_comparison: dimensionsComparison,
    communication_gaps: [
      {
        dimension: 'directness',
        gap_size: 0.5,
        impact_level: 'medium',
        recommendations: ['建立明确的沟通规范', '提供跨文化培训']
      },
      {
        dimension: 'hierarchy',
        gap_size: 0.3,
        impact_level: 'low',
        recommendations: ['讨论权力距离差异', '建立扁平化沟通渠道']
      }
    ],
    collaboration_strategies: [
      '建立文化桥梁角色',
      '定期进行文化敏感性培训',
      '创建多元化团队',
      '实施文化导师制度'
    ],
    conflict_risks: [
      '沟通方式误解',
      '价值观冲突',
      '决策速度不匹配',
      '权威认知差异'
    ]
  };
};

const getMockCulturalProfiles = () => ([
  { code: 'zh-CN', name: '中国', flag: '🇨🇳' },
  { code: 'en-US', name: '美国', flag: '🇺🇸' },
  { code: 'ja-JP', name: '日本', flag: '🇯🇵' },
  { code: 'de-DE', name: '德国', flag: '🇩🇪' },
  { code: 'fr-FR', name: '法国', flag: '🇫🇷' },
  { code: 'en-GB', name: '英国', flag: '🇬🇧' },
  { code: 'ko-KR', name: '韩国', flag: '🇰🇷' },
  { code: 'es-ES', name: '西班牙', flag: '🇪🇸' }
]);

const generateMockRecommendations = (profiles: CulturalProfile[], scenario: string) => [
  {
    category: '沟通建议',
    items: [
      '使用中性、尊重的语言',
      '提供多种沟通渠道选择',
      '建立清晰的反馈机制'
    ]
  },
  {
    category: '协作策略',
    items: [
      '建立跨文化工作小组',
      '实施轮换领导制度',
      '创建文化分享活动'
    ]
  }
];

const generateMockRiskAssessment = (profiles: CulturalProfile[]) => ({
  high_risk_areas: ['直接反馈', '权威挑战', '时间观念差异'],
  medium_risk_areas: ['决策流程', '团队合作方式'],
  low_risk_areas: ['技术讨论', '目标设定'],
  mitigation_strategies: [
    '建立文化敏感性指南',
    '提供跨文化沟通培训',
    '设置文化协调员角色'
  ]
});

const generateMockBestPractices = (scenario: string) => [
  '尊重文化差异，避免价值判断',
  '建立包容性的沟通环境',
  '提供文化背景解释',
  '鼓励开放式对话',
  '定期检查理解程度'
];

const CulturalContextAnalysisPage: React.FC = () => {
  const [currentAnalysis, setCurrentAnalysis] = useState<CulturalAnalysisResult | null>(null);
  const [crossCulturalComparison, setCrossCulturalComparison] = useState<CrossCulturalComparison | null>(null);
  const [culturalProfiles, setCulturalProfiles] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedCulture, setSelectedCulture] = useState('zh-CN');
  const [comparisonCultures, setComparisonCultures] = useState<string[]>(['zh-CN', 'en-US']);
  const [showAnalysisModal, setShowAnalysisModal] = useState(false);
  const [showComparisonModal, setShowComparisonModal] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  
  const [analysisForm] = Form.useForm();
  const [comparisonForm] = Form.useForm();

  useEffect(() => {
    loadInitialData();
  }, []);

  useEffect(() => {
    if (selectedCulture) {
      performCulturalAnalysis(selectedCulture);
    }
  }, [selectedCulture]);

  const loadInitialData = async () => {
    setLoading(true);
    try {
      const profilesResult = await culturalAnalysisApi.getCulturalProfiles();
      if (profilesResult.data) {
        setCulturalProfiles(profilesResult.data);
      }
    } catch (error) {
      console.error('加载初始数据失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const performCulturalAnalysis = async (cultureCode: string) => {
    setLoading(true);
    try {
      const result = await culturalAnalysisApi.analyzeCulturalContext(
        { emotions: { neutral: 0.5 }, intensity: 0.5 },
        cultureCode
      );
      
      if (result.data?.cultural_analysis) {
        setCurrentAnalysis(result.data.cultural_analysis);
        if (!result.success) {
          message.warning('使用模拟数据显示');
        }
      }
    } catch (error) {
      console.error('文化分析失败:', error);
      message.error('分析失败');
    } finally {
      setLoading(false);
    }
  };

  const performCrossComparison = async () => {
    if (comparisonCultures.length < 2) {
      message.warning('请至少选择两种文化进行比较');
      return;
    }
    
    setLoading(true);
    try {
      const result = await culturalAnalysisApi.compareCultures(comparisonCultures);
      if (result.data) {
        setCrossCulturalComparison(result.data);
        message.success('跨文化比较完成');
      }
    } catch (error) {
      message.error('比较分析失败');
    } finally {
      setLoading(false);
    }
  };

  const renderOverviewCards = () => (
    <Row gutter={16}>
      <Col span={6}>
        <Card>
          <div style={{ textAlign: 'center' }}>
            <GlobalOutlined style={{ fontSize: 24, color: '#1890ff', marginBottom: 8 }} />
            <div style={{ fontSize: 24, fontWeight: 'bold', color: '#1890ff' }}>
              {currentAnalysis ? Math.round(currentAnalysis.sensitivity_score * 100) : 0}%
            </div>
            <div style={{ color: '#8c8c8c' }}>文化敏感度</div>
          </div>
        </Card>
      </Col>
      <Col span={6}>
        <Card>
          <div style={{ textAlign: 'center' }}>
            <CompassOutlined style={{ fontSize: 24, color: '#52c41a', marginBottom: 8 }} />
            <div style={{ fontSize: 24, fontWeight: 'bold', color: '#52c41a' }}>
              {currentAnalysis ? Object.keys(currentAnalysis.cultural_profile.hofstede_dimensions).length : 0}
            </div>
            <div style={{ color: '#8c8c8c' }}>文化维度</div>
          </div>
        </Card>
      </Col>
      <Col span={6}>
        <Card>
          <div style={{ textAlign: 'center' }}>
            <AlertOutlined style={{ fontSize: 24, color: '#fa8c16', marginBottom: 8 }} />
            <div style={{ fontSize: 24, fontWeight: 'bold', color: '#fa8c16' }}>
              {currentAnalysis ? currentAnalysis.potential_conflicts.length : 0}
            </div>
            <div style={{ color: '#8c8c8c' }}>潜在冲突</div>
          </div>
        </Card>
      </Col>
      <Col span={6}>
        <Card>
          <div style={{ textAlign: 'center' }}>
            <BulbOutlined style={{ fontSize: 24, color: '#722ed1', marginBottom: 8 }} />
            <div style={{ fontSize: 24, fontWeight: 'bold', color: '#722ed1' }}>
              {currentAnalysis ? currentAnalysis.adaptation_recommendations.length : 0}
            </div>
            <div style={{ color: '#8c8c8c' }}>适配建议</div>
          </div>
        </Card>
      </Col>
    </Row>
  );

  const renderCulturalDimensions = () => {
    if (!currentAnalysis) return null;

    const { hofstede_dimensions } = currentAnalysis.cultural_profile;
    
    const dimensionLabels = {
      power_distance: { name: '权力距离', icon: <CrownOutlined />, color: '#1890ff' },
      individualism: { name: '个人主义', icon: <UserOutlined />, color: '#52c41a' },
      masculinity: { name: '男性化', icon: <TeamOutlined />, color: '#fa8c16' },
      uncertainty_avoidance: { name: '不确定性规避', icon: <SecurityScanOutlined />, color: '#722ed1' },
      long_term_orientation: { name: '长期取向', icon: <BookOutlined />, color: '#eb2f96' },
      indulgence: { name: '纵容性', icon: <HeartOutlined />, color: '#13c2c2' }
    };

    return (
      <Card title={
        <span>
          <RadarChartOutlined style={{ marginRight: 8 }} />
          霍夫斯泰德文化维度分析
        </span>
      }>
        <Row gutter={[16, 16]}>
          {Object.entries(hofstede_dimensions).map(([key, value]) => {
            const config = dimensionLabels[key as keyof typeof dimensionLabels];
            if (!config) return null;
            
            return (
              <Col span={12} key={key}>
                <div style={{ marginBottom: 16 }}>
                  <div style={{ display: 'flex', alignItems: 'center', marginBottom: 8 }}>
                    <span style={{ color: config.color, marginRight: 8 }}>
                      {config.icon}
                    </span>
                    <Text strong>{config.name}</Text>
                    <Tooltip title={getCulturalDimensionDescription(key)}>
                      <BulbOutlined style={{ marginLeft: 8, color: '#999' }} />
                    </Tooltip>
                  </div>
                  <Progress
                    percent={Math.round(value * 100)}
                    strokeColor={config.color}
                    trailColor="#f0f0f0"
                  />
                  <div style={{ marginTop: 4, fontSize: '12px', color: '#666' }}>
                    {getCulturalDimensionInterpretation(key, value)}
                  </div>
                </div>
              </Col>
            );
          })}
        </Row>
      </Card>
    );
  };

  const getCulturalDimensionDescription = (dimension: string) => {
    const descriptions = {
      power_distance: '社会对权力不平等分配的接受程度',
      individualism: '个人利益与集体利益的优先级',
      masculinity: '竞争性与合作性价值观的倾向',
      uncertainty_avoidance: '对不确定和模糊情况的容忍度',
      long_term_orientation: '对传统与变革的态度',
      indulgence: '对欲望和冲动的控制程度'
    };
    return descriptions[dimension as keyof typeof descriptions] || '';
  };

  const getCulturalDimensionInterpretation = (dimension: string, value: number) => {
    const interpretations = {
      power_distance: value > 0.6 ? '等级制度明显' : value > 0.4 ? '中等等级观念' : '扁平化结构',
      individualism: value > 0.6 ? '个人主义导向' : value > 0.4 ? '个人集体平衡' : '集体主义导向',
      masculinity: value > 0.6 ? '竞争导向' : value > 0.4 ? '竞合平衡' : '合作导向',
      uncertainty_avoidance: value > 0.6 ? '规避不确定性' : value > 0.4 ? '中等风险容忍' : '拥抱不确定性',
      long_term_orientation: value > 0.6 ? '长期规划导向' : value > 0.4 ? '中期平衡' : '短期结果导向',
      indulgence: value > 0.6 ? '相对自由' : value > 0.4 ? '适度约束' : '严格控制'
    };
    return interpretations[dimension as keyof typeof interpretations] || '';
  };

  const renderCommunicationPatterns = () => {
    if (!currentAnalysis) return null;

    const { communication_patterns } = currentAnalysis.cultural_profile;

    return (
      <Card title={
        <span>
          <TranslationOutlined style={{ marginRight: 8 }} />
          沟通模式分析
        </span>
      }>
        <Row gutter={16}>
          <Col span={12}>
            <div>
              <Text strong>直接性程度:</Text>
              <Progress
                percent={Math.round(communication_patterns.directness * 100)}
                strokeColor={communication_patterns.directness > 0.6 ? '#52c41a' : '#fa8c16'}
                style={{ marginTop: 4 }}
              />
              <Text style={{ fontSize: '12px', color: '#666' }}>
                {communication_patterns.directness > 0.6 ? '倾向直接沟通' : '倾向间接沟通'}
              </Text>
            </div>
            <Divider />
            <div>
              <Text strong>上下文敏感度:</Text>
              <Progress
                percent={Math.round(communication_patterns.context_sensitivity * 100)}
                strokeColor="#1890ff"
                style={{ marginTop: 4 }}
              />
              <Text style={{ fontSize: '12px', color: '#666' }}>
                {communication_patterns.context_sensitivity > 0.6 ? '高语境文化' : '低语境文化'}
              </Text>
            </div>
          </Col>
          <Col span={12}>
            <div>
              <Text strong>正式程度:</Text>
              <Progress
                percent={Math.round(communication_patterns.formal_register * 100)}
                strokeColor="#722ed1"
                style={{ marginTop: 4 }}
              />
              <Text style={{ fontSize: '12px', color: '#666' }}>
                {communication_patterns.formal_register > 0.6 ? '偏好正式交流' : '偏好非正式交流'}
              </Text>
            </div>
            <Divider />
            <div>
              <Text strong>沉默舒适度:</Text>
              <Progress
                percent={Math.round(communication_patterns.silence_comfort * 100)}
                strokeColor="#13c2c2"
                style={{ marginTop: 4 }}
              />
              <Text style={{ fontSize: '12px', color: '#666' }}>
                {communication_patterns.silence_comfort > 0.6 ? '接受沉默' : '偏好连续对话'}
              </Text>
            </div>
          </Col>
        </Row>
      </Card>
    );
  };

  const renderConflictAnalysis = () => {
    if (!currentAnalysis || !currentAnalysis.potential_conflicts.length) return null;

    const columns = [
      {
        title: '冲突领域',
        dataIndex: 'area',
        key: 'area',
        render: (area: string) => <Tag color="orange">{area}</Tag>
      },
      {
        title: '严重程度',
        dataIndex: 'severity',
        key: 'severity',
        render: (severity: number) => (
          <Progress
            percent={Math.round(severity * 100)}
            size="small"
            strokeColor={severity > 0.7 ? '#ff4d4f' : severity > 0.4 ? '#fa8c16' : '#52c41a'}
            style={{ width: 100 }}
          />
        )
      },
      {
        title: '描述',
        dataIndex: 'description',
        key: 'description',
        ellipsis: true
      },
      {
        title: '缓解措施',
        dataIndex: 'mitigation',
        key: 'mitigation',
        ellipsis: true
      }
    ];

    return (
      <Card title={
        <span>
          <AlertOutlined style={{ marginRight: 8 }} />
          潜在冲突分析
        </span>
      }>
        <Table
          columns={columns}
          dataSource={currentAnalysis.potential_conflicts}
          rowKey="area"
          pagination={false}
          size="small"
        />
      </Card>
    );
  };

  const renderAdaptationRecommendations = () => {
    if (!currentAnalysis) return null;

    return (
      <Card title={
        <span>
          <BulbOutlined style={{ marginRight: 8 }} />
          适配建议
        </span>
      }>
        <Steps
          direction="vertical"
          size="small"
          current={-1}
          items={currentAnalysis.adaptation_recommendations.map((recommendation, index) => ({
            title: `建议 ${index + 1}`,
            description: recommendation,
            status: 'process'
          }))}
        />
      </Card>
    );
  };

  const renderCrossComparison = () => {
    if (!crossCulturalComparison) return null;

    const { dimensions_comparison, communication_gaps } = crossCulturalComparison;

    return (
      <div>
        <Card title={
          <span>
            <GlobalOutlined style={{ marginRight: 8 }} />
            跨文化维度对比
          </span>
        } style={{ marginBottom: 16 }}>
          {Object.keys(dimensions_comparison).length > 0 && (
            <div>
              {Object.keys(dimensions_comparison[Object.keys(dimensions_comparison)[0]]).map(dimension => (
                <div key={dimension} style={{ marginBottom: 16 }}>
                  <Text strong style={{ textTransform: 'capitalize', marginBottom: 8, display: 'block' }}>
                    {dimension.replace('_', ' ')}
                  </Text>
                  <Row gutter={8}>
                    {Object.entries(dimensions_comparison).map(([culture, values]) => (
                      <Col span={24 / Object.keys(dimensions_comparison).length} key={culture}>
                        <div style={{ textAlign: 'center', marginBottom: 4 }}>
                          <Tag color="blue">{culture}</Tag>
                        </div>
                        <Progress
                          percent={Math.round((values as any)[dimension] * 100)}
                          size="small"
                          strokeColor={`hsl(${Object.keys(dimensions_comparison).indexOf(culture) * 60}, 70%, 50%)`}
                        />
                      </Col>
                    ))}
                  </Row>
                </div>
              ))}
            </div>
          )}
        </Card>

        <Card title={
          <span>
            <AlertOutlined style={{ marginRight: 8 }} />
            沟通差距分析
          </span>
        }>
          <List
            dataSource={communication_gaps}
            renderItem={(gap) => (
              <List.Item>
                <List.Item.Meta
                  avatar={
                    <Badge
                      color={gap.impact_level === 'high' ? '#ff4d4f' : 
                             gap.impact_level === 'medium' ? '#fa8c16' : '#52c41a'}
                      text={gap.impact_level}
                    />
                  }
                  title={
                    <div>
                      <Text strong>{gap.dimension}</Text>
                      <span style={{ marginLeft: 8 }}>
                        差距: {Math.round(gap.gap_size * 100)}%
                      </span>
                    </div>
                  }
                  description={
                    <div>
                      <div style={{ marginBottom: 8 }}>
                        <Text strong>建议措施:</Text>
                      </div>
                      {gap.recommendations.map((rec, index) => (
                        <Tag key={index} color="green" style={{ margin: '2px' }}>
                          {rec}
                        </Tag>
                      ))}
                    </div>
                  }
                />
              </List.Item>
            )}
          />
        </Card>
      </div>
    );
  };

  const renderAnalysisModal = () => (
    <Modal
      title="文化背景分析"
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
      width={700}
    >
      <Form
        form={analysisForm}
        layout="vertical"
        onFinish={(values) => {
          performCulturalAnalysis(values.culture_code);
          setShowAnalysisModal(false);
        }}
      >
        <Alert
          message="文化背景分析"
          description="深入分析特定文化背景下的情感表达模式和社交规范"
          type="info"
          showIcon
          style={{ marginBottom: 24 }}
        />

        <Form.Item
          label="目标文化"
          name="culture_code"
          rules={[{ required: true, message: '请选择要分析的文化' }]}
        >
          <Select placeholder="选择文化背景">
            {culturalProfiles.map(profile => (
              <Option key={profile.code} value={profile.code}>
                <span style={{ marginRight: 8 }}>{profile.flag}</span>
                {profile.name}
              </Option>
            ))}
          </Select>
        </Form.Item>

        <Form.Item
          label="分析范围"
          name="analysis_scope"
        >
          <Checkbox.Group>
            <Row>
              <Col span={12}><Checkbox value="hofstede">霍夫斯泰德维度</Checkbox></Col>
              <Col span={12}><Checkbox value="communication">沟通模式</Checkbox></Col>
              <Col span={12}><Checkbox value="emotions">情感规范</Checkbox></Col>
              <Col span={12}><Checkbox value="conflicts">冲突分析</Checkbox></Col>
            </Row>
          </Checkbox.Group>
        </Form.Item>

        <Form.Item
          label="应用场景"
          name="scenario"
        >
          <Select placeholder="选择应用场景">
            <Option value="business">商务合作</Option>
            <Option value="education">教育培训</Option>
            <Option value="social">社交互动</Option>
            <Option value="healthcare">医疗健康</Option>
          </Select>
        </Form.Item>
      </Form>
    </Modal>
  );

  const renderComparisonModal = () => (
    <Modal
      title="跨文化对比分析"
      open={showComparisonModal}
      onCancel={() => setShowComparisonModal(false)}
      footer={[
        <Button key="cancel" onClick={() => setShowComparisonModal(false)}>
          取消
        </Button>,
        <Button 
          key="compare" 
          type="primary" 
          loading={loading}
          onClick={() => {
            performCrossComparison();
            setShowComparisonModal(false);
          }}
        >
          开始比较
        </Button>
      ]}
    >
      <Form
        form={comparisonForm}
        layout="vertical"
      >
        <Alert
          message="跨文化对比分析"
          description="比较多种文化背景下的差异，识别潜在的沟通障碍和协作机会"
          type="info"
          showIcon
          style={{ marginBottom: 24 }}
        />

        <Form.Item
          label="比较文化 (请选择2-4种文化)"
          name="cultures"
          rules={[{ required: true, message: '请至少选择2种文化进行比较' }]}
        >
          <Select
            mode="multiple"
            placeholder="选择要比较的文化"
            value={comparisonCultures}
            onChange={setComparisonCultures}
            maxTagCount={4}
          >
            {culturalProfiles.map(profile => (
              <Option key={profile.code} value={profile.code}>
                <span style={{ marginRight: 8 }}>{profile.flag}</span>
                {profile.name}
              </Option>
            ))}
          </Select>
        </Form.Item>

        <Form.Item
          label="比较维度"
          name="comparison_dimensions"
        >
          <Checkbox.Group>
            <Row>
              <Col span={12}><Checkbox value="cultural_dimensions">文化维度</Checkbox></Col>
              <Col span={12}><Checkbox value="communication">沟通方式</Checkbox></Col>
              <Col span={12}><Checkbox value="values">价值观</Checkbox></Col>
              <Col span={12}><Checkbox value="behaviors">行为模式</Checkbox></Col>
            </Row>
          </Checkbox.Group>
        </Form.Item>
      </Form>
    </Modal>
  );

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: 24, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Title level={2}>
          <GlobalOutlined style={{ marginRight: 12, color: '#1890ff' }} />
          文化背景分析
        </Title>
        <Space>
          <Select
            style={{ width: 200 }}
            value={selectedCulture}
            onChange={setSelectedCulture}
            placeholder="选择文化"
          >
            {culturalProfiles.map(profile => (
              <Option key={profile.code} value={profile.code}>
                <span style={{ marginRight: 8 }}>{profile.flag}</span>
                {profile.name}
              </Option>
            ))}
          </Select>
          <Button 
            type="primary" 
            icon={<ExperimentOutlined />}
            onClick={() => setShowAnalysisModal(true)}
          >
            新建分析
          </Button>
          <Button 
            icon={<CompassOutlined />}
            onClick={() => setShowComparisonModal(true)}
          >
            文化对比
          </Button>
          <Button 
            icon={<SyncOutlined />} 
            loading={loading}
            onClick={() => performCulturalAnalysis(selectedCulture)}
          >
            刷新
          </Button>
        </Space>
      </div>

      <div style={{ marginBottom: 24 }}>
        {renderOverviewCards()}
      </div>

      <Tabs defaultActiveKey="dimensions">
        <TabPane tab="文化维度" key="dimensions">
          <Row gutter={24}>
            <Col span={16}>
              {renderCulturalDimensions()}
            </Col>
            <Col span={8}>
              {renderAdaptationRecommendations()}
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="沟通模式" key="communication">
          <Row gutter={24}>
            <Col span={16}>
              {renderCommunicationPatterns()}
            </Col>
            <Col span={8}>
              {renderConflictAnalysis()}
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="跨文化对比" key="cross-comparison">
          {crossCulturalComparison ? renderCrossComparison() : (
            <Card>
              <div style={{ textAlign: 'center', padding: 60 }}>
                <CompassOutlined style={{ fontSize: 48, color: '#d9d9d9', marginBottom: 16 }} />
                <div>
                  <Text type="secondary">点击"文化对比"按钮开始跨文化分析</Text>
                </div>
              </div>
            </Card>
          )}
        </TabPane>

        <TabPane tab="实践指南" key="guidelines">
          <Card title="文化敏感性实践指南">
            <Alert
              message="实践指南功能"
              description="提供基于文化分析结果的具体实践建议和最佳做法"
              type="info"
              showIcon
              style={{ marginBottom: 24 }}
            />
            <div style={{ textAlign: 'center', padding: 60 }}>
              <BookOutlined style={{ fontSize: 48, color: '#d9d9d9', marginBottom: 16 }} />
              <div>
                <Text type="secondary">实践指南功能正在开发中...</Text>
              </div>
            </div>
          </Card>
        </TabPane>
      </Tabs>

      {renderAnalysisModal()}
      {renderComparisonModal()}
    </div>
  );
};

export default CulturalContextAnalysisPage;