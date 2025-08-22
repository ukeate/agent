import React, { useState, useEffect } from 'react';
import { Card, Button, Select, Tabs, Progress, Alert, Badge, Space, Typography, Divider } from 'antd';
import { 
  BrainCircuit, 
  FileText, 
  BarChart3, 
  Download, 
  RefreshCw, 
  Settings,
  HelpCircle,
  TrendingUp,
  Target,
  AlertTriangle,
  CheckCircle
} from 'lucide-react';

const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;

interface ExplanationComponent {
  factor_name: string;
  factor_value: any;
  weight: number;
  impact_score: number;
  evidence_type: string;
  evidence_source: string;
  evidence_content: string;
}

interface ConfidenceMetrics {
  overall_confidence: number;
  uncertainty_score: number;
  confidence_interval_lower?: number;
  confidence_interval_upper?: number;
}

interface CounterfactualScenario {
  scenario_name: string;
  predicted_outcome: string;
  probability: number;
  impact_difference: number;
  explanation: string;
}

interface DecisionExplanation {
  id: string;
  decision_id: string;
  explanation_type: string;
  decision_outcome: string;
  summary_explanation: string;
  detailed_explanation: string;
  components: ExplanationComponent[];
  confidence_metrics: ConfidenceMetrics;
  counterfactuals: CounterfactualScenario[];
}

const ExplainableAiPage: React.FC = () => {
  const [explanation, setExplanation] = useState<DecisionExplanation | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedExplanationType, setSelectedExplanationType] = useState('decision');
  const [selectedExplanationLevel, setSelectedExplanationLevel] = useState('detailed');

  // 模拟数据
  const mockExplanation: DecisionExplanation = {
    id: 'exp-001',
    decision_id: 'loan-001',
    explanation_type: 'decision',
    decision_outcome: '贷款申请批准',
    summary_explanation: '基于申请人的信用评分750分、年收入8万元等因素，系统建议批准此贷款申请。',
    detailed_explanation: '经过综合分析，申请人具备良好的信用记录和稳定的收入来源，风险评估结果为低风险，符合贷款批准标准。',
    components: [
      {
        factor_name: '信用评分',
        factor_value: 750,
        weight: 0.35,
        impact_score: 0.8,
        evidence_type: 'external_source',
        evidence_source: '征信机构',
        evidence_content: '信用评分750分，表现良好'
      },
      {
        factor_name: '年收入',
        factor_value: 80000,
        weight: 0.25,
        impact_score: 0.7,
        evidence_type: 'external_source',
        evidence_source: '财务报表',
        evidence_content: '年收入8万元，收入稳定'
      },
      {
        factor_name: '工作年限',
        factor_value: 5,
        weight: 0.2,
        impact_score: 0.6,
        evidence_type: 'external_source',
        evidence_source: 'HR系统',
        evidence_content: '工作5年，稳定性良好'
      }
    ],
    confidence_metrics: {
      overall_confidence: 0.85,
      uncertainty_score: 0.15,
      confidence_interval_lower: 0.78,
      confidence_interval_upper: 0.92
    },
    counterfactuals: [
      {
        scenario_name: '信用评分降低场景',
        predicted_outcome: '可能被拒绝',
        probability: 0.7,
        impact_difference: -0.3,
        explanation: '如果信用评分降至650分，批准概率将显著降低'
      }
    ]
  };

  const generateExplanation = async () => {
    setLoading(true);
    try {
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1500));
      setExplanation(mockExplanation);
    } catch (error) {
      console.error('Failed to generate explanation:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    generateExplanation();
  }, []);

  const renderConfidenceMetrics = () => {
    if (!explanation) return null;

    const { confidence_metrics } = explanation;
    
    return (
      <Card>
        <div style={{ padding: '16px' }}>
          <Title level={4}>置信度分析</Title>
          <Space direction="vertical" style={{ width: '100%' }}>
            <div>
              <Text>整体置信度</Text>
              <Progress 
                percent={confidence_metrics.overall_confidence * 100} 
                strokeColor={{
                  '0%': '#108ee9',
                  '100%': '#87d068',
                }}
              />
              <Text type="secondary">
                {(confidence_metrics.overall_confidence * 100).toFixed(1)}%
              </Text>
            </div>
            <div>
              <Text>不确定性</Text>
              <Progress 
                percent={confidence_metrics.uncertainty_score * 100} 
                strokeColor="#ff4d4f"
              />
              <Text type="secondary">
                {(confidence_metrics.uncertainty_score * 100).toFixed(1)}%
              </Text>
            </div>
          </Space>
        </div>
      </Card>
    );
  };

  const renderFactorsTable = () => {
    if (!explanation) return null;

    return (
      <Card>
        <div style={{ padding: '16px' }}>
          <Title level={4}>关键因素分析</Title>
          <div style={{ marginTop: '16px' }}>
            {explanation.components.map((component, index) => (
              <div key={index} style={{ marginBottom: '16px', padding: '12px', border: '1px solid #f0f0f0', borderRadius: '6px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                  <Text strong>{component.factor_name}</Text>
                  <Badge 
                    color={component.impact_score > 0.7 ? 'green' : component.impact_score > 0.4 ? 'orange' : 'red'}
                    text={`影响度: ${(component.impact_score * 100).toFixed(0)}%`}
                  />
                </div>
                <div style={{ marginBottom: '8px' }}>
                  <Text>值: </Text>
                  <Text code>{component.factor_value}</Text>
                  <Text style={{ marginLeft: '16px' }}>权重: </Text>
                  <Text code>{(component.weight * 100).toFixed(0)}%</Text>
                </div>
                <Text type="secondary">{component.evidence_content}</Text>
              </div>
            ))}
          </div>
        </div>
      </Card>
    );
  };

  const renderCounterfactuals = () => {
    if (!explanation) return null;

    return (
      <Card>
        <div style={{ padding: '16px' }}>
          <Title level={4}>反事实分析</Title>
          <Paragraph>
            <Text type="secondary">分析在不同假设条件下的可能结果</Text>
          </Paragraph>
          
          {explanation.counterfactuals.map((scenario, index) => (
            <Card key={index} style={{ marginBottom: '16px' }} size="small">
              <div style={{ padding: '12px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                  <Text strong>{scenario.scenario_name}</Text>
                  <Badge 
                    color={scenario.impact_difference > 0 ? 'green' : 'red'}
                    text={`${scenario.impact_difference > 0 ? '+' : ''}${(scenario.impact_difference * 100).toFixed(1)}%`}
                  />
                </div>
                <div style={{ marginBottom: '8px' }}>
                  <Text>预测结果: </Text>
                  <Text strong>{scenario.predicted_outcome}</Text>
                </div>
                <div style={{ marginBottom: '8px' }}>
                  <Text>发生概率: </Text>
                  <Text code>{(scenario.probability * 100).toFixed(0)}%</Text>
                </div>
                <Text type="secondary">{scenario.explanation}</Text>
              </div>
            </Card>
          ))}
        </div>
      </Card>
    );
  };

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
        {/* 页面标题 */}
        <div style={{ marginBottom: '24px' }}>
          <Title level={2}>
            <BrainCircuit style={{ marginRight: '8px', color: '#1890ff' }} />
            可解释AI决策系统
          </Title>
          <Paragraph>
            通过透明化的决策过程分析，帮助理解AI系统的推理逻辑和决策依据。
          </Paragraph>
        </div>

        {/* 控制面板 */}
        <Card style={{ marginBottom: '24px' }}>
          <div style={{ padding: '16px' }}>
            <Title level={4}>解释配置</Title>
            <Space size="large">
              <div>
                <Text>解释类型:</Text>
                <Select 
                  value={selectedExplanationType}
                  onChange={setSelectedExplanationType}
                  style={{ width: 120, marginLeft: '8px' }}
                >
                  <Select.Option value="decision">决策解释</Select.Option>
                  <Select.Option value="reasoning">推理解释</Select.Option>
                  <Select.Option value="workflow">工作流解释</Select.Option>
                </Select>
              </div>
              
              <div>
                <Text>详细程度:</Text>
                <Select 
                  value={selectedExplanationLevel}
                  onChange={setSelectedExplanationLevel}
                  style={{ width: 120, marginLeft: '8px' }}
                >
                  <Select.Option value="summary">概要</Select.Option>
                  <Select.Option value="detailed">详细</Select.Option>
                  <Select.Option value="technical">技术</Select.Option>
                </Select>
              </div>

              <Button 
                type="primary" 
                loading={loading}
                onClick={generateExplanation}
                icon={<RefreshCw size={16} />}
              >
                生成解释
              </Button>
            </Space>
          </div>
        </Card>

        {/* 解释结果 */}
        {explanation && (
          <Card>
            <div style={{ padding: '16px' }}>
              <Title level={3}>决策解释结果</Title>
              
              {/* 基本信息 */}
              <div style={{ marginBottom: '24px', padding: '16px', background: '#f8f9fa', borderRadius: '6px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                  <Title level={4} style={{ margin: 0 }}>
                    决策结果: {explanation.decision_outcome}
                  </Title>
                  <Badge 
                    color="green" 
                    text={`置信度: ${(explanation.confidence_metrics.overall_confidence * 100).toFixed(1)}%`}
                  />
                </div>
                <Paragraph style={{ margin: 0 }}>
                  {explanation.summary_explanation}
                </Paragraph>
              </div>

              {/* 详细分析标签页 */}
              <Tabs defaultActiveKey="factors">
                <TabPane tab="影响因子" key="factors">
                  {renderFactorsTable()}
                </TabPane>
                
                <TabPane tab="置信度分析" key="confidence">
                  {renderConfidenceMetrics()}
                </TabPane>
                
                <TabPane tab="反事实分析" key="counterfactuals">
                  {renderCounterfactuals()}
                </TabPane>
                
                <TabPane tab="技术细节" key="technical">
                  <Card>
                    <div style={{ padding: '16px' }}>
                      <Title level={4}>技术实现细节</Title>
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <div>
                          <Text strong>决策ID: </Text>
                          <Text code>{explanation.decision_id}</Text>
                        </div>
                        <div>
                          <Text strong>解释类型: </Text>
                          <Text code>{explanation.explanation_type}</Text>
                        </div>
                        <div>
                          <Text strong>置信区间: </Text>
                          <Text code>
                            [{(explanation.confidence_metrics.confidence_interval_lower || 0) * 100}%, {(explanation.confidence_metrics.confidence_interval_upper || 0) * 100}%]
                          </Text>
                        </div>
                        <Divider />
                        <div>
                          <Title level={5}>详细解释</Title>
                          <Paragraph>{explanation.detailed_explanation}</Paragraph>
                        </div>
                      </Space>
                    </div>
                  </Card>
                </TabPane>
              </Tabs>
            </div>
          </Card>
        )}

        {!explanation && !loading && (
          <Alert
            message="暂无解释数据"
            description='点击"生成解释"按钮来获取AI决策的详细解释'
            variant="default"
            showIcon
          />
        )}
      </div>
    </div>
  );
};

export default ExplainableAiPage;