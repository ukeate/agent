import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Button,
  Table,
  Tag,
  Progress,
  Alert,
  Space,
  Modal,
  Form,
  Input,
  Select,
  Timeline,
  Divider,
  Typography,
  Spin
} from 'antd';
import {
  BrainCircuitIcon,
  HeartHandshakeIcon,
  ShieldAlertIcon,
  TrendingUpIcon,
  MessageSquareIcon,
  UserCheckIcon,
  AlertTriangleIcon,
  ActivityIcon
} from 'lucide-react';

const { Title, Text } = Typography;
const { Option } = Select;

interface EmotionalDecision {
  decision_id: string;
  user_id: string;
  chosen_strategy: string;
  confidence_score: number;
  reasoning: string[];
  timestamp: string;
  decision_type: string;
}

interface RiskAssessment {
  assessment_id: string;
  user_id: string;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  risk_score: number;
  prediction_confidence: number;
  recommended_actions: string[];
}

interface DecisionContext {
  user_id: string;
  session_id?: string;
  current_emotion_state: any;
  emotion_history?: any[];
  user_input: string;
  environmental_factors?: any;
}

const EmotionalIntelligenceDecisionEnginePage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [decisions, setDecisions] = useState<EmotionalDecision[]>([]);
  const [riskAssessments, setRiskAssessments] = useState<RiskAssessment[]>([]);
  const [decisionModalVisible, setDecisionModalVisible] = useState(false);
  const [riskModalVisible, setRiskModalVisible] = useState(false);
  const [currentDecision, setCurrentDecision] = useState<EmotionalDecision | null>(null);
  const [stats, setStats] = useState({
    totalDecisions: 0,
    averageConfidence: 0,
    highRiskUsers: 0,
    activeInterventions: 0
  });

  const [form] = Form.useForm();

  // 模拟数据
  useEffect(() => {
    const mockDecisions: EmotionalDecision[] = [
      {
        decision_id: '1',
        user_id: 'user_001',
        chosen_strategy: 'supportive_strategy',
        confidence_score: 0.92,
        reasoning: ['用户情感状态稳定', '历史互动积极', '支持策略效果良好'],
        timestamp: new Date().toISOString(),
        decision_type: 'supportive'
      },
      {
        decision_id: '2', 
        user_id: 'user_002',
        chosen_strategy: 'intervention_strategy',
        confidence_score: 0.78,
        reasoning: ['检测到焦虑情绪', '需要主动关怀', '建议专业支持'],
        timestamp: new Date(Date.now() - 3600000).toISOString(),
        decision_type: 'corrective'
      }
    ];

    const mockRiskAssessments: RiskAssessment[] = [
      {
        assessment_id: '1',
        user_id: 'user_001',
        risk_level: 'low',
        risk_score: 0.2,
        prediction_confidence: 0.85,
        recommended_actions: ['继续正面强化', '维持当前互动模式']
      },
      {
        assessment_id: '2',
        user_id: 'user_002', 
        risk_level: 'medium',
        risk_score: 0.6,
        prediction_confidence: 0.73,
        recommended_actions: ['增加关怀频率', '监控情感变化', '提供情感支持资源']
      }
    ];

    setDecisions(mockDecisions);
    setRiskAssessments(mockRiskAssessments);
    setStats({
      totalDecisions: mockDecisions.length,
      averageConfidence: mockDecisions.reduce((acc, d) => acc + d.confidence_score, 0) / mockDecisions.length,
      highRiskUsers: mockRiskAssessments.filter(r => ['high', 'critical'].includes(r.risk_level)).length,
      activeInterventions: 3
    });
  }, []);

  const handleMakeDecision = async (values: any) => {
    setLoading(true);
    try {
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const newDecision: EmotionalDecision = {
        decision_id: Date.now().toString(),
        user_id: values.user_id,
        chosen_strategy: 'supportive_strategy',
        confidence_score: 0.88,
        reasoning: ['基于当前情感状态分析', '历史数据支持此决策', '高置信度推荐'],
        timestamp: new Date().toISOString(),
        decision_type: 'supportive'
      };
      
      setDecisions(prev => [newDecision, ...prev]);
      setDecisionModalVisible(false);
      form.resetFields();
    } catch (error) {
      console.error('决策制定失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const decisionColumns = [
    {
      title: '决策ID',
      dataIndex: 'decision_id',
      key: 'decision_id',
      width: 120
    },
    {
      title: '用户ID',
      dataIndex: 'user_id',
      key: 'user_id',
      width: 120
    },
    {
      title: '选择策略',
      dataIndex: 'chosen_strategy',
      key: 'chosen_strategy',
      render: (strategy: string) => (
        <Tag color={strategy === 'supportive_strategy' ? 'green' : 'orange'}>
          {strategy}
        </Tag>
      )
    },
    {
      title: '置信度',
      dataIndex: 'confidence_score',
      key: 'confidence_score',
      render: (score: number) => (
        <Progress 
          percent={score * 100} 
          size="small"
          status={score > 0.8 ? 'success' : score > 0.6 ? 'normal' : 'exception'}
        />
      )
    },
    {
      title: '决策类型',
      dataIndex: 'decision_type',
      key: 'decision_type',
      render: (type: string) => {
        const colorMap: Record<string, string> = {
          'supportive': 'green',
          'corrective': 'orange',
          'crisis': 'red'
        };
        return <Tag color={colorMap[type] || 'default'}>{type}</Tag>;
      }
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: EmotionalDecision) => (
        <Space>
          <Button 
            size="small" 
            onClick={() => {
              setCurrentDecision(record);
              setDecisionModalVisible(true);
            }}
          >
            详情
          </Button>
        </Space>
      )
    }
  ];

  const riskColumns = [
    {
      title: '评估ID',
      dataIndex: 'assessment_id',
      key: 'assessment_id',
      width: 120
    },
    {
      title: '用户ID', 
      dataIndex: 'user_id',
      key: 'user_id',
      width: 120
    },
    {
      title: '风险等级',
      dataIndex: 'risk_level',
      key: 'risk_level',
      render: (level: string) => {
        const colorMap: Record<string, string> = {
          'low': 'green',
          'medium': 'orange', 
          'high': 'red',
          'critical': 'purple'
        };
        return <Tag color={colorMap[level]}>{level.toUpperCase()}</Tag>;
      }
    },
    {
      title: '风险分数',
      dataIndex: 'risk_score',
      key: 'risk_score',
      render: (score: number) => (
        <Progress 
          percent={score * 100}
          size="small"
          status={score > 0.7 ? 'exception' : score > 0.4 ? 'active' : 'success'}
        />
      )
    },
    {
      title: '预测置信度',
      dataIndex: 'prediction_confidence', 
      key: 'prediction_confidence',
      render: (confidence: number) => `${(confidence * 100).toFixed(1)}%`
    },
    {
      title: '推荐行动',
      dataIndex: 'recommended_actions',
      key: 'recommended_actions',
      render: (actions: string[]) => (
        <div>
          {actions.slice(0, 2).map((action, index) => (
            <Tag key={index} style={{ marginBottom: 4 }}>
              {action}
            </Tag>
          ))}
          {actions.length > 2 && <Text type="secondary">+{actions.length - 2}更多</Text>}
        </div>
      )
    }
  ];

  return (
    <div style={{ padding: '24px' }}>
      {/* 页面标题 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={24}>
          <Card>
            <Space align="center">
              <BrainCircuitIcon size={32} />
              <div>
                <Title level={2} style={{ margin: 0 }}>情感智能决策引擎</Title>
                <Text type="secondary">基于情感状态的智能决策系统，提供个性化情感支持和风险预警</Text>
              </div>
            </Space>
          </Card>
        </Col>
      </Row>

      {/* 统计卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="决策总数"
              value={stats.totalDecisions}
              prefix={<ActivityIcon size={16} />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="平均置信度"
              value={(stats.averageConfidence * 100).toFixed(1)}
              suffix="%"
              prefix={<TrendingUpIcon size={16} />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="高风险用户"
              value={stats.highRiskUsers}
              prefix={<AlertTriangleIcon size={16} />}
              valueStyle={{ color: '#f5222d' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="活跃干预"
              value={stats.activeInterventions}
              prefix={<HeartHandshakeIcon size={16} />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 系统状态提醒 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={24}>
          <Alert
            message="系统运行状态良好"
            description="情感智能决策引擎运行正常，当前处理延迟 < 200ms，所有监控指标正常。"
            type="success"
            showIcon
            style={{ marginBottom: 16 }}
          />
        </Col>
      </Row>

      {/* 操作按钮 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={24}>
          <Space>
            <Button 
              type="primary" 
              icon={<BrainCircuitIcon size={16} />}
              onClick={() => setDecisionModalVisible(true)}
            >
              新建决策
            </Button>
            <Button 
              icon={<ShieldAlertIcon size={16} />}
              onClick={() => setRiskModalVisible(true)}
            >
              风险评估
            </Button>
            <Button icon={<UserCheckIcon size={16} />}>
              健康监测
            </Button>
            <Button icon={<MessageSquareIcon size={16} />}>
              干预管理
            </Button>
          </Space>
        </Col>
      </Row>

      {/* 主要内容区域 */}
      <Row gutter={[16, 16]}>
        {/* 决策历史 */}
        <Col span={24}>
          <Card 
            title={
              <Space>
                <ActivityIcon size={20} />
                最近决策
              </Space>
            }
            extra={<Button type="link">查看全部</Button>}
          >
            <Table
              dataSource={decisions}
              columns={decisionColumns}
              rowKey="decision_id"
              pagination={{ pageSize: 5 }}
              size="small"
            />
          </Card>
        </Col>

        {/* 风险评估 */}
        <Col span={24}>
          <Card 
            title={
              <Space>
                <ShieldAlertIcon size={20} />
                风险评估
              </Space>
            }
            extra={<Button type="link">查看全部</Button>}
          >
            <Table
              dataSource={riskAssessments}
              columns={riskColumns}
              rowKey="assessment_id"
              pagination={{ pageSize: 5 }}
              size="small"
            />
          </Card>
        </Col>
      </Row>

      {/* 新建决策模态框 */}
      <Modal
        title="新建情感决策"
        open={decisionModalVisible}
        onCancel={() => {
          setDecisionModalVisible(false);
          setCurrentDecision(null);
          form.resetFields();
        }}
        footer={null}
        width={600}
      >
        {currentDecision ? (
          // 显示决策详情
          <div>
            <Title level={4}>决策详情</Title>
            <Divider />
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Text strong>决策ID: </Text>
                <Text>{currentDecision.decision_id}</Text>
              </Col>
              <Col span={12}>
                <Text strong>用户ID: </Text>
                <Text>{currentDecision.user_id}</Text>
              </Col>
              <Col span={12}>
                <Text strong>选择策略: </Text>
                <Tag>{currentDecision.chosen_strategy}</Tag>
              </Col>
              <Col span={12}>
                <Text strong>置信度: </Text>
                <Progress 
                  percent={currentDecision.confidence_score * 100} 
                  size="small" 
                  style={{ width: 100 }}
                />
              </Col>
              <Col span={24}>
                <Text strong>决策推理过程: </Text>
                <Timeline style={{ marginTop: 8 }}>
                  {currentDecision.reasoning.map((reason, index) => (
                    <Timeline.Item key={index}>
                      {reason}
                    </Timeline.Item>
                  ))}
                </Timeline>
              </Col>
            </Row>
          </div>
        ) : (
          // 新建决策表单
          <Form
            form={form}
            layout="vertical"
            onFinish={handleMakeDecision}
          >
            <Form.Item
              name="user_id"
              label="用户ID"
              rules={[{ required: true, message: '请输入用户ID' }]}
            >
              <Input placeholder="输入用户ID" />
            </Form.Item>

            <Form.Item
              name="user_input"
              label="用户输入"
              rules={[{ required: true, message: '请输入用户消息' }]}
            >
              <Input.TextArea 
                placeholder="输入用户的消息内容..."
                rows={3}
              />
            </Form.Item>

            <Form.Item
              name="emotion_context"
              label="情感上下文"
            >
              <Select placeholder="选择情感状态">
                <Option value="happy">开心</Option>
                <Option value="sad">难过</Option>
                <Option value="anxious">焦虑</Option>
                <Option value="angry">愤怒</Option>
                <Option value="neutral">中性</Option>
              </Select>
            </Form.Item>

            <Form.Item>
              <Space>
                <Button type="primary" htmlType="submit" loading={loading}>
                  制定决策
                </Button>
                <Button onClick={() => setDecisionModalVisible(false)}>
                  取消
                </Button>
              </Space>
            </Form.Item>
          </Form>
        )}
      </Modal>

      {/* 风险评估模态框 */}
      <Modal
        title="执行风险评估"
        open={riskModalVisible}
        onCancel={() => setRiskModalVisible(false)}
        footer={null}
      >
        <Spin spinning={loading}>
          <Form layout="vertical">
            <Form.Item label="目标用户">
              <Input placeholder="输入用户ID" />
            </Form.Item>
            <Form.Item label="评估类型">
              <Select placeholder="选择评估类型">
                <Option value="comprehensive">综合评估</Option>
                <Option value="crisis">危机评估</Option>
                <Option value="trend">趋势分析</Option>
              </Select>
            </Form.Item>
            <Form.Item>
              <Space>
                <Button type="primary">
                  开始评估
                </Button>
                <Button onClick={() => setRiskModalVisible(false)}>
                  取消
                </Button>
              </Space>
            </Form.Item>
          </Form>
        </Spin>
      </Modal>
    </div>
  );
};

export default EmotionalIntelligenceDecisionEnginePage;