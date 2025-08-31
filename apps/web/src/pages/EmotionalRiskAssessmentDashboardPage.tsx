import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Table,
  Tag,
  Progress,
  Alert,
  Space,
  Button,
  Select,
  DatePicker,
  Typography,
  Divider,
  Timeline,
  Badge,
  Tooltip,
  Modal,
  Form,
  Input
} from 'antd';
import {
  ShieldAlertIcon,
  TrendingUpIcon,
  TrendingDownIcon,
  AlertTriangleIcon,
  UserIcon,
  BarChart3Icon,
  Activity,
  Clock,
  Target,
  Brain
} from 'lucide-react';
import { Line, Bar, Pie } from '@ant-design/plots';

const { Title, Text } = Typography;
const { RangePicker } = DatePicker;
const { Option } = Select;

interface RiskFactor {
  factor_type: string;
  score: number;
  evidence: any;
  weight: number;
  description: string;
}

interface RiskAssessment {
  assessment_id: string;
  user_id: string;
  timestamp: string;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
  risk_score: number;
  risk_factors: RiskFactor[];
  prediction_confidence: number;
  prediction_timeframe: string;
  recommended_actions: string[];
  urgency_level: string;
  alert_triggered: boolean;
}

interface RiskTrend {
  date: string;
  risk_score: number;
  user_count: number;
  alert_count: number;
}

const EmotionalRiskAssessmentDashboardPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [assessments, setAssessments] = useState<RiskAssessment[]>([]);
  const [trends, setTrends] = useState<RiskTrend[]>([]);
  const [selectedTimeRange, setSelectedTimeRange] = useState('week');
  const [assessmentModalVisible, setAssessmentModalVisible] = useState(false);
  const [selectedAssessment, setSelectedAssessment] = useState<RiskAssessment | null>(null);
  
  const [stats, setStats] = useState({
    totalAssessments: 0,
    highRiskUsers: 0,
    averageRiskScore: 0,
    alertsTriggered: 0,
    riskReduction: 0
  });

  const [form] = Form.useForm();

  // 模拟数据
  useEffect(() => {
    const mockAssessments: RiskAssessment[] = [
      {
        assessment_id: '1',
        user_id: 'user_001',
        timestamp: new Date().toISOString(),
        risk_level: 'high',
        risk_score: 0.75,
        risk_factors: [
          {
            factor_type: 'depression_indicators',
            score: 0.8,
            evidence: { negative_emotion_count: 15, avg_valence: -0.7 },
            weight: 0.3,
            description: '检测到显著抑郁症状'
          },
          {
            factor_type: 'social_isolation',
            score: 0.6,
            evidence: { isolation_duration: '7天' },
            weight: 0.15,
            description: '社交活动明显减少'
          }
        ],
        prediction_confidence: 0.82,
        prediction_timeframe: '3天',
        recommended_actions: ['专业心理咨询', '增加社交活动', '情感支持热线'],
        urgency_level: 'high',
        alert_triggered: true
      },
      {
        assessment_id: '2',
        user_id: 'user_002',
        timestamp: new Date(Date.now() - 3600000).toISOString(),
        risk_level: 'medium',
        risk_score: 0.55,
        risk_factors: [
          {
            factor_type: 'anxiety_indicators',
            score: 0.65,
            evidence: { anxiety_frequency: 'daily', intensity_avg: 0.6 },
            weight: 0.25,
            description: '持续性焦虑症状'
          }
        ],
        prediction_confidence: 0.74,
        prediction_timeframe: '1周',
        recommended_actions: ['放松训练', '生活规律调整', '定期情感检查'],
        urgency_level: 'medium',
        alert_triggered: false
      },
      {
        assessment_id: '3',
        user_id: 'user_003',
        timestamp: new Date(Date.now() - 7200000).toISOString(),
        risk_level: 'low',
        risk_score: 0.25,
        risk_factors: [
          {
            factor_type: 'emotional_volatility',
            score: 0.3,
            evidence: { volatility_index: 0.25 },
            weight: 0.2,
            description: '情感波动轻微'
          }
        ],
        prediction_confidence: 0.91,
        prediction_timeframe: '1月',
        recommended_actions: ['保持当前状态', '定期自我检查'],
        urgency_level: 'low',
        alert_triggered: false
      }
    ];

    const mockTrends: RiskTrend[] = [
      { date: '2024-01-01', risk_score: 0.3, user_count: 120, alert_count: 2 },
      { date: '2024-01-02', risk_score: 0.35, user_count: 125, alert_count: 3 },
      { date: '2024-01-03', risk_score: 0.28, user_count: 118, alert_count: 1 },
      { date: '2024-01-04', risk_score: 0.42, user_count: 130, alert_count: 5 },
      { date: '2024-01-05', risk_score: 0.38, user_count: 128, alert_count: 4 },
      { date: '2024-01-06', risk_score: 0.25, user_count: 115, alert_count: 1 },
      { date: '2024-01-07', risk_score: 0.33, user_count: 122, alert_count: 2 }
    ];

    setAssessments(mockAssessments);
    setTrends(mockTrends);
    setStats({
      totalAssessments: mockAssessments.length,
      highRiskUsers: mockAssessments.filter(a => ['high', 'critical'].includes(a.risk_level)).length,
      averageRiskScore: mockAssessments.reduce((acc, a) => acc + a.risk_score, 0) / mockAssessments.length,
      alertsTriggered: mockAssessments.filter(a => a.alert_triggered).length,
      riskReduction: 15.2
    });
  }, []);

  const riskLevelColor = (level: string) => {
    const colors: Record<string, string> = {
      'low': '#52c41a',
      'medium': '#faad14',
      'high': '#f5222d',
      'critical': '#722ed1'
    };
    return colors[level] || '#d9d9d9';
  };

  const urgencyColor = (level: string) => {
    const colors: Record<string, string> = {
      'low': 'green',
      'medium': 'orange',
      'high': 'red',
      'critical': 'purple'
    };
    return colors[level] || 'default';
  };

  const assessmentColumns = [
    {
      title: '用户ID',
      dataIndex: 'user_id',
      key: 'user_id',
      width: 120,
      render: (userId: string) => (
        <Space>
          <UserIcon size={14} />
          {userId}
        </Space>
      )
    },
    {
      title: '风险等级',
      dataIndex: 'risk_level',
      key: 'risk_level',
      render: (level: string, record: RiskAssessment) => (
        <Space>
          <Badge color={riskLevelColor(level)} />
          <Tag color={riskLevelColor(level)}>{level.toUpperCase()}</Tag>
          {record.alert_triggered && <AlertTriangleIcon size={14} color="#f5222d" />}
        </Space>
      )
    },
    {
      title: '风险分数',
      dataIndex: 'risk_score',
      key: 'risk_score',
      render: (score: number) => (
        <Progress 
          percent={score * 100}
          size="small"
          strokeColor={score > 0.7 ? '#f5222d' : score > 0.4 ? '#faad14' : '#52c41a'}
          format={percent => `${percent?.toFixed(1)}%`}
        />
      )
    },
    {
      title: '预测置信度',
      dataIndex: 'prediction_confidence',
      key: 'prediction_confidence',
      render: (confidence: number) => (
        <Tooltip title="AI模型预测的置信度">
          <Tag>{(confidence * 100).toFixed(1)}%</Tag>
        </Tooltip>
      )
    },
    {
      title: '紧急程度',
      dataIndex: 'urgency_level',
      key: 'urgency_level',
      render: (level: string) => (
        <Tag color={urgencyColor(level)}>{level.toUpperCase()}</Tag>
      )
    },
    {
      title: '评估时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (time: string) => (
        <Space>
          <Clock size={14} />
          {new Date(time).toLocaleString()}
        </Space>
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: RiskAssessment) => (
        <Space>
          <Button 
            size="small"
            onClick={() => {
              setSelectedAssessment(record);
              setAssessmentModalVisible(true);
            }}
          >
            详情
          </Button>
          <Button size="small" type="primary" ghost>
            干预
          </Button>
        </Space>
      )
    }
  ];

  // 风险趋势图配置
  const trendConfig = {
    data: trends,
    xField: 'date',
    yField: 'risk_score',
    smooth: true,
    color: '#722ed1',
    point: {
      size: 4,
      shape: 'diamond',
    },
    label: {
      style: {
        fill: '#aaa',
      },
    },
    tooltip: {
      formatter: (datum: any) => {
        return {
          name: '平均风险分数',
          value: (datum.risk_score * 100).toFixed(1) + '%'
        };
      }
    }
  };

  // 风险等级分布配置
  const riskDistributionData = [
    { type: '低风险', value: assessments.filter(a => a.risk_level === 'low').length },
    { type: '中风险', value: assessments.filter(a => a.risk_level === 'medium').length },
    { type: '高风险', value: assessments.filter(a => a.risk_level === 'high').length },
    { type: '严重风险', value: assessments.filter(a => a.risk_level === 'critical').length },
  ];

  const distributionConfig = {
    data: riskDistributionData,
    angleField: 'value',
    colorField: 'type',
    radius: 0.8,
    color: ['#52c41a', '#faad14', '#f5222d', '#722ed1'],
    label: {
      type: 'outer',
      content: '{name} {percentage}',
    },
    interactions: [
      {
        type: 'element-selected',
      },
      {
        type: 'element-active',
      },
    ],
  };

  const handleNewAssessment = async (values: any) => {
    setLoading(true);
    try {
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      const newAssessment: RiskAssessment = {
        assessment_id: Date.now().toString(),
        user_id: values.user_id,
        timestamp: new Date().toISOString(),
        risk_level: 'medium',
        risk_score: 0.5,
        risk_factors: [
          {
            factor_type: 'assessment_requested',
            score: 0.5,
            evidence: { manual_trigger: true },
            weight: 1.0,
            description: '手动触发的风险评估'
          }
        ],
        prediction_confidence: 0.75,
        prediction_timeframe: '1周',
        recommended_actions: ['定期监测', '情感支持'],
        urgency_level: 'medium',
        alert_triggered: false
      };

      setAssessments(prev => [newAssessment, ...prev]);
      setAssessmentModalVisible(false);
      form.resetFields();
    } catch (error) {
      console.error('风险评估失败:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '24px' }}>
      {/* 页面标题 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={24}>
          <Card>
            <Space align="center">
              <ShieldAlertIcon size={32} />
              <div>
                <Title level={2} style={{ margin: 0 }}>情感风险评估仪表盘</Title>
                <Text type="secondary">实时监控用户情感健康风险，预测危机并提供预警机制</Text>
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
              title="总评估数"
              value={stats.totalAssessments}
              prefix={<BarChart3Icon size={16} />}
              valueStyle={{ color: '#1890ff' }}
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
              title="平均风险分数"
              value={(stats.averageRiskScore * 100).toFixed(1)}
              suffix="%"
              prefix={<Activity size={16} />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="风险降低"
              value={stats.riskReduction}
              suffix="%"
              prefix={<TrendingDownIcon size={16} />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 预警提醒 */}
      {stats.alertsTriggered > 0 && (
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          <Col span={24}>
            <Alert
              message={`检测到 ${stats.alertsTriggered} 个高风险预警`}
              description="系统已自动识别需要立即关注的用户，建议及时采取干预措施。"
              type="warning"
              showIcon
              action={
                <Button size="small" type="primary" ghost>
                  查看详情
                </Button>
              }
            />
          </Col>
        </Row>
      )}

      {/* 操作栏 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={24}>
          <Space>
            <Button 
              type="primary" 
              icon={<ShieldAlertIcon size={16} />}
              onClick={() => {
                setSelectedAssessment(null);
                setAssessmentModalVisible(true);
              }}
            >
              新建评估
            </Button>
            <Select 
              value={selectedTimeRange} 
              onChange={setSelectedTimeRange}
              style={{ width: 120 }}
            >
              <Option value="day">今日</Option>
              <Option value="week">本周</Option>
              <Option value="month">本月</Option>
            </Select>
            <Button icon={<Target size={16} />}>
              批量评估
            </Button>
            <Button icon={<Brain size={16} />}>
              AI预测
            </Button>
          </Space>
        </Col>
      </Row>

      {/* 图表和数据 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        {/* 风险趋势图 */}
        <Col xs={24} lg={16}>
          <Card 
            title={
              <Space>
                <TrendingUpIcon size={20} />
                风险趋势分析
              </Space>
            }
          >
            <Line {...trendConfig} />
          </Card>
        </Col>

        {/* 风险等级分布 */}
        <Col xs={24} lg={8}>
          <Card 
            title={
              <Space>
                <BarChart3Icon size={20} />
                风险等级分布
              </Space>
            }
          >
            <Pie {...distributionConfig} />
          </Card>
        </Col>
      </Row>

      {/* 评估记录表格 */}
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Card 
            title={
              <Space>
                <ShieldAlertIcon size={20} />
                风险评估记录
              </Space>
            }
            extra={<Button type="link">导出报告</Button>}
          >
            <Table
              dataSource={assessments}
              columns={assessmentColumns}
              rowKey="assessment_id"
              pagination={{ 
                pageSize: 10,
                showSizeChanger: true,
                showQuickJumper: true,
                showTotal: (total, range) => `显示 ${range[0]}-${range[1]} 条，共 ${total} 条记录`
              }}
              scroll={{ x: 1000 }}
            />
          </Card>
        </Col>
      </Row>

      {/* 评估详情/新建模态框 */}
      <Modal
        title={selectedAssessment ? "风险评估详情" : "新建风险评估"}
        open={assessmentModalVisible}
        onCancel={() => {
          setAssessmentModalVisible(false);
          setSelectedAssessment(null);
          form.resetFields();
        }}
        footer={null}
        width={800}
      >
        {selectedAssessment ? (
          // 显示评估详情
          <div>
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Card size="small" title="基本信息">
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div>
                      <Text strong>评估ID: </Text>
                      <Text>{selectedAssessment.assessment_id}</Text>
                    </div>
                    <div>
                      <Text strong>用户ID: </Text>
                      <Text>{selectedAssessment.user_id}</Text>
                    </div>
                    <div>
                      <Text strong>风险等级: </Text>
                      <Tag color={riskLevelColor(selectedAssessment.risk_level)}>
                        {selectedAssessment.risk_level.toUpperCase()}
                      </Tag>
                    </div>
                    <div>
                      <Text strong>风险分数: </Text>
                      <Progress 
                        percent={selectedAssessment.risk_score * 100} 
                        size="small"
                        style={{ width: '100px' }}
                      />
                    </div>
                  </Space>
                </Card>
              </Col>
              <Col span={12}>
                <Card size="small" title="预测信息">
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div>
                      <Text strong>预测置信度: </Text>
                      <Text>{(selectedAssessment.prediction_confidence * 100).toFixed(1)}%</Text>
                    </div>
                    <div>
                      <Text strong>预测时间框架: </Text>
                      <Text>{selectedAssessment.prediction_timeframe}</Text>
                    </div>
                    <div>
                      <Text strong>紧急程度: </Text>
                      <Tag color={urgencyColor(selectedAssessment.urgency_level)}>
                        {selectedAssessment.urgency_level.toUpperCase()}
                      </Tag>
                    </div>
                    <div>
                      <Text strong>是否触发预警: </Text>
                      <Badge 
                        status={selectedAssessment.alert_triggered ? 'error' : 'default'}
                        text={selectedAssessment.alert_triggered ? '已触发' : '未触发'}
                      />
                    </div>
                  </Space>
                </Card>
              </Col>
            </Row>

            <Divider />

            <Card size="small" title="风险因子分析" style={{ marginBottom: 16 }}>
              {selectedAssessment.risk_factors.map((factor, index) => (
                <Card.Grid key={index} style={{ width: '50%', padding: '16px' }}>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Text strong>{factor.factor_type}</Text>
                    <Progress 
                      percent={factor.score * 100}
                      size="small"
                      strokeColor={factor.score > 0.7 ? '#f5222d' : factor.score > 0.4 ? '#faad14' : '#52c41a'}
                    />
                    <Text type="secondary" style={{ fontSize: '12px' }}>
                      {factor.description}
                    </Text>
                    <Text type="secondary" style={{ fontSize: '11px' }}>
                      权重: {(factor.weight * 100).toFixed(1)}%
                    </Text>
                  </Space>
                </Card.Grid>
              ))}
            </Card>

            <Card size="small" title="推荐行动">
              <Timeline>
                {selectedAssessment.recommended_actions.map((action, index) => (
                  <Timeline.Item key={index}>
                    {action}
                  </Timeline.Item>
                ))}
              </Timeline>
            </Card>
          </div>
        ) : (
          // 新建评估表单
          <Form
            form={form}
            layout="vertical"
            onFinish={handleNewAssessment}
          >
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item
                  name="user_id"
                  label="目标用户"
                  rules={[{ required: true, message: '请输入用户ID' }]}
                >
                  <Input placeholder="输入用户ID" />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item
                  name="assessment_type"
                  label="评估类型"
                  rules={[{ required: true, message: '请选择评估类型' }]}
                >
                  <Select placeholder="选择评估类型">
                    <Option value="comprehensive">综合风险评估</Option>
                    <Option value="crisis">危机风险检测</Option>
                    <Option value="trend">风险趋势分析</Option>
                    <Option value="predictive">预测性评估</Option>
                  </Select>
                </Form.Item>
              </Col>
            </Row>

            <Form.Item
              name="context"
              label="评估上下文"
            >
              <Input.TextArea 
                placeholder="输入相关背景信息（可选）..."
                rows={3}
              />
            </Form.Item>

            <Form.Item>
              <Space>
                <Button type="primary" htmlType="submit" loading={loading}>
                  开始评估
                </Button>
                <Button onClick={() => setAssessmentModalVisible(false)}>
                  取消
                </Button>
              </Space>
            </Form.Item>
          </Form>
        )}
      </Modal>
    </div>
  );
};

export default EmotionalRiskAssessmentDashboardPage;