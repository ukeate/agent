import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Table,
  Tag,
  Button,
  Space,
  Select,
  DatePicker,
  Typography,
  Progress,
  Timeline,
  Badge,
  Tooltip,
  Modal,
  Descriptions,
  Tabs,
  Alert
} from 'antd';
import {
  HistoryIcon,
  BrainIcon,
  TrendingUpIcon,
  BarChart3Icon,
  UserIcon,
  ClockIcon,
  ActivityIcon,
  CheckCircleIcon,
  XCircleIcon,
  AlertTriangleIcon,
  Target
} from 'lucide-react';
import { Line, Column, Pie, Scatter } from '@ant-design/plots';

const { Title, Text } = Typography;
const { RangePicker } = DatePicker;
const { Option } = Select;
const { TabPane } = Tabs;

interface DecisionHistoryRecord {
  decision_id: string;
  user_id: string;
  timestamp: string;
  chosen_strategy: string;
  confidence_score: number;
  reasoning: string[];
  decision_type: 'supportive' | 'corrective' | 'crisis';
  context: {
    emotion_state: string;
    risk_level: string;
    urgency: string;
  };
  outcome?: {
    effectiveness_score: number;
    user_feedback: number;
    duration: number;
    follow_up_required: boolean;
  };
  execution_status: 'pending' | 'in_progress' | 'completed' | 'failed';
}

interface DecisionAnalytics {
  strategy_performance: {
    strategy_name: string;
    usage_count: number;
    avg_confidence: number;
    avg_effectiveness: number;
    success_rate: number;
  }[];
  temporal_patterns: {
    hour: number;
    decision_count: number;
    avg_confidence: number;
  }[];
  user_patterns: {
    user_id: string;
    total_decisions: number;
    avg_confidence: number;
    most_used_strategy: string;
    improvement_trend: 'improving' | 'stable' | 'declining';
  }[];
}

const DecisionHistoryAnalysisPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [decisionHistory, setDecisionHistory] = useState<DecisionHistoryRecord[]>([]);
  const [analytics, setAnalytics] = useState<DecisionAnalytics | null>(null);
  const [selectedDecision, setSelectedDecision] = useState<DecisionHistoryRecord | null>(null);
  const [detailModalVisible, setDetailModalVisible] = useState(false);
  const [timeRange, setTimeRange] = useState<string>('week');
  const [filterStrategy, setFilterStrategy] = useState<string>('all');
  
  const [stats, setStats] = useState({
    totalDecisions: 0,
    averageConfidence: 0,
    successRate: 0,
    pendingDecisions: 0
  });

  // 模拟数据
  useEffect(() => {
    const mockDecisions: DecisionHistoryRecord[] = [
      {
        decision_id: '1',
        user_id: 'user_001',
        timestamp: new Date().toISOString(),
        chosen_strategy: 'cognitive_behavioral_support',
        confidence_score: 0.92,
        reasoning: [
          '用户表现出轻度焦虑症状',
          '历史数据显示对认知疗法响应良好',
          '当前情绪状态适合进行认知重构'
        ],
        decision_type: 'supportive',
        context: {
          emotion_state: 'anxiety',
          risk_level: 'medium',
          urgency: 'normal'
        },
        outcome: {
          effectiveness_score: 8.5,
          user_feedback: 4.2,
          duration: 42,
          follow_up_required: true
        },
        execution_status: 'completed'
      },
      {
        decision_id: '2',
        user_id: 'user_002',
        timestamp: new Date(Date.now() - 3600000).toISOString(),
        chosen_strategy: 'crisis_intervention',
        confidence_score: 0.98,
        reasoning: [
          '检测到高风险自杀倾向语言',
          '情感状态极度负面',
          '需要立即专业干预'
        ],
        decision_type: 'crisis',
        context: {
          emotion_state: 'despair',
          risk_level: 'high',
          urgency: 'immediate'
        },
        outcome: {
          effectiveness_score: 9.2,
          user_feedback: 4.8,
          duration: 12,
          follow_up_required: true
        },
        execution_status: 'completed'
      },
      {
        decision_id: '3',
        user_id: 'user_001',
        timestamp: new Date(Date.now() - 7200000).toISOString(),
        chosen_strategy: 'mindfulness_training',
        confidence_score: 0.78,
        reasoning: [
          '用户压力水平较高',
          '正念练习有助于压力缓解',
          '用户之前尝试过类似方法'
        ],
        decision_type: 'corrective',
        context: {
          emotion_state: 'stress',
          risk_level: 'low',
          urgency: 'normal'
        },
        execution_status: 'in_progress'
      }
    ];

    const mockAnalytics: DecisionAnalytics = {
      strategy_performance: [
        {
          strategy_name: 'cognitive_behavioral_support',
          usage_count: 156,
          avg_confidence: 0.86,
          avg_effectiveness: 8.2,
          success_rate: 84.6
        },
        {
          strategy_name: 'crisis_intervention',
          usage_count: 23,
          avg_confidence: 0.94,
          avg_effectiveness: 9.1,
          success_rate: 95.7
        },
        {
          strategy_name: 'mindfulness_training',
          usage_count: 98,
          avg_confidence: 0.79,
          avg_effectiveness: 7.5,
          success_rate: 76.5
        }
      ],
      temporal_patterns: [
        { hour: 0, decision_count: 5, avg_confidence: 0.78 },
        { hour: 6, decision_count: 12, avg_confidence: 0.82 },
        { hour: 12, decision_count: 25, avg_confidence: 0.85 },
        { hour: 18, decision_count: 35, avg_confidence: 0.88 },
        { hour: 21, decision_count: 18, avg_confidence: 0.80 }
      ],
      user_patterns: [
        {
          user_id: 'user_001',
          total_decisions: 45,
          avg_confidence: 0.82,
          most_used_strategy: 'cognitive_behavioral_support',
          improvement_trend: 'improving'
        },
        {
          user_id: 'user_002',
          total_decisions: 12,
          avg_confidence: 0.91,
          most_used_strategy: 'crisis_intervention',
          improvement_trend: 'stable'
        }
      ]
    };

    setDecisionHistory(mockDecisions);
    setAnalytics(mockAnalytics);
    setStats({
      totalDecisions: mockDecisions.length,
      averageConfidence: mockDecisions.reduce((acc, d) => acc + d.confidence_score, 0) / mockDecisions.length,
      successRate: mockDecisions.filter(d => d.execution_status === 'completed').length / mockDecisions.length * 100,
      pendingDecisions: mockDecisions.filter(d => d.execution_status === 'pending').length
    });
  }, []);

  const decisionTypeColors = {
    'supportive': '#52c41a',
    'corrective': '#faad14',
    'crisis': '#f5222d'
  };

  const statusColors = {
    'pending': 'default',
    'in_progress': 'processing',
    'completed': 'success',
    'failed': 'error'
  };

  const strategyPerformanceConfig = {
    data: analytics?.strategy_performance || [],
    xField: 'strategy_name',
    yField: 'success_rate',
    columnStyle: {
      radius: [4, 4, 0, 0],
    },
    color: '#1890ff',
    label: {
      position: 'top' as const,
      formatter: (datum: any) => `${datum.success_rate}%`,
    },
  };

  const temporalPatternsConfig = {
    data: analytics?.temporal_patterns || [],
    xField: 'hour',
    yField: 'decision_count',
    smooth: true,
    color: '#722ed1',
    point: {
      size: 4,
      shape: 'diamond',
    },
  };

  const confidenceDistributionData = decisionHistory.map(d => ({
    confidence: Math.floor(d.confidence_score * 10) / 10,
    count: 1
  })).reduce((acc: any[], curr) => {
    const existing = acc.find(item => item.confidence === curr.confidence);
    if (existing) {
      existing.count++;
    } else {
      acc.push(curr);
    }
    return acc;
  }, []);

  const confidenceDistributionConfig = {
    data: confidenceDistributionData,
    angleField: 'count',
    colorField: 'confidence',
    radius: 0.8,
    label: {
      type: 'outer',
      content: '{name} {percentage}',
    },
    interactions: [
      {
        type: 'element-selected',
      },
    ],
  };

  const historyColumns = [
    {
      title: '决策ID',
      dataIndex: 'decision_id',
      key: 'decision_id',
      width: 100
    },
    {
      title: '用户',
      dataIndex: 'user_id',
      key: 'user_id',
      render: (userId: string) => (
        <Space>
          <UserIcon size={14} />
          {userId}
        </Space>
      )
    },
    {
      title: '策略',
      dataIndex: 'chosen_strategy',
      key: 'chosen_strategy',
      render: (strategy: string) => (
        <Tooltip title={strategy}>
          <Tag>{strategy.replace(/_/g, ' ')}</Tag>
        </Tooltip>
      )
    },
    {
      title: '类型',
      dataIndex: 'decision_type',
      key: 'decision_type',
      render: (type: string) => (
        <Tag color={decisionTypeColors[type as keyof typeof decisionTypeColors]}>
          {type.toUpperCase()}
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
          format={percent => `${percent?.toFixed(1)}%`}
          strokeColor={score > 0.9 ? '#52c41a' : score > 0.7 ? '#1890ff' : '#faad14'}
        />
      )
    },
    {
      title: '执行状态',
      dataIndex: 'execution_status',
      key: 'execution_status',
      render: (status: string) => (
        <Badge status={statusColors[status as keyof typeof statusColors]} text={status.toUpperCase()} />
      )
    },
    {
      title: '效果评分',
      key: 'effectiveness',
      render: (record: DecisionHistoryRecord) => (
        record.outcome ? (
          <Space direction="vertical" size="small">
            <Progress
              percent={record.outcome.effectiveness_score * 10}
              size="small"
              format={() => `${record.outcome!.effectiveness_score}/10`}
            />
            <Text style={{ fontSize: '11px' }}>
              用户反馈: {record.outcome.user_feedback}/5
            </Text>
          </Space>
        ) : (
          <Text type="secondary">未完成</Text>
        )
      )
    },
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (time: string) => (
        <Space>
          <ClockIcon size={14} />
          {new Date(time).toLocaleString()}
        </Space>
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: DecisionHistoryRecord) => (
        <Space>
          <Button
            size="small"
            onClick={() => {
              setSelectedDecision(record);
              setDetailModalVisible(true);
            }}
          >
            详情
          </Button>
          <Button size="small" type="primary" ghost>
            分析
          </Button>
        </Space>
      )
    }
  ];

  const strategyColumns = [
    {
      title: '策略名称',
      dataIndex: 'strategy_name',
      key: 'strategy_name',
      render: (name: string) => <Text strong>{name.replace(/_/g, ' ')}</Text>
    },
    {
      title: '使用次数',
      dataIndex: 'usage_count',
      key: 'usage_count',
      render: (count: number) => <Badge count={count} overflowCount={999} />
    },
    {
      title: '平均置信度',
      dataIndex: 'avg_confidence',
      key: 'avg_confidence',
      render: (confidence: number) => (
        <Progress
          percent={confidence * 100}
          size="small"
          format={() => `${(confidence * 100).toFixed(1)}%`}
        />
      )
    },
    {
      title: '平均效果',
      dataIndex: 'avg_effectiveness',
      key: 'avg_effectiveness',
      render: (effectiveness: number) => `${effectiveness}/10`
    },
    {
      title: '成功率',
      dataIndex: 'success_rate',
      key: 'success_rate',
      render: (rate: number) => (
        <Progress
          percent={rate}
          size="small"
          strokeColor={rate > 90 ? '#52c41a' : rate > 80 ? '#1890ff' : '#faad14'}
          format={() => `${rate}%`}
        />
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
              <HistoryIcon size={32} />
              <div>
                <Title level={2} style={{ margin: 0 }}>决策历史与效果分析</Title>
                <Text type="secondary">分析决策历史数据，优化策略选择，提升决策质量</Text>
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
              title="总决策数"
              value={stats.totalDecisions}
              prefix={<BrainIcon size={16} />}
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
              prefix={<Target size={16} />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="执行成功率"
              value={stats.successRate.toFixed(1)}
              suffix="%"
              prefix={<CheckCircleIcon size={16} />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="待处理决策"
              value={stats.pendingDecisions}
              prefix={<ClockIcon size={16} />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 过滤器 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={24}>
          <Space>
            <Select 
              value={timeRange} 
              onChange={setTimeRange}
              style={{ width: 120 }}
            >
              <Option value="day">今日</Option>
              <Option value="week">本周</Option>
              <Option value="month">本月</Option>
              <Option value="quarter">本季度</Option>
            </Select>
            <Select 
              value={filterStrategy} 
              onChange={setFilterStrategy}
              style={{ width: 200 }}
            >
              <Option value="all">全部策略</Option>
              <Option value="cognitive_behavioral_support">认知行为支持</Option>
              <Option value="crisis_intervention">危机干预</Option>
              <Option value="mindfulness_training">正念训练</Option>
            </Select>
            <Button icon={<BarChart3Icon size={16} />}>
              生成报告
            </Button>
            <Button icon={<ActivityIcon size={16} />}>
              导出数据
            </Button>
          </Space>
        </Col>
      </Row>

      {/* 主要内容 */}
      <Tabs defaultActiveKey="history" type="card">
        <TabPane tab="决策历史" key="history">
          <Card 
            title={
              <Space>
                <HistoryIcon size={20} />
                历史决策记录
              </Space>
            }
          >
            <Table
              dataSource={decisionHistory}
              columns={historyColumns}
              rowKey="decision_id"
              pagination={{ 
                pageSize: 10,
                showSizeChanger: true,
                showQuickJumper: true
              }}
              scroll={{ x: 1200 }}
            />
          </Card>
        </TabPane>

        <TabPane tab="策略分析" key="strategy">
          <Row gutter={[16, 16]}>
            <Col span={24}>
              <Card 
                title={
                  <Space>
                    <BarChart3Icon size={20} />
                    策略性能分析
                  </Space>
                }
              >
                <Row gutter={[16, 16]}>
                  <Col xs={24} lg={16}>
                    <Column {...strategyPerformanceConfig} />
                  </Col>
                  <Col xs={24} lg={8}>
                    <Table
                      dataSource={analytics?.strategy_performance}
                      columns={strategyColumns}
                      rowKey="strategy_name"
                      pagination={false}
                      size="small"
                    />
                  </Col>
                </Row>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="趋势分析" key="trends">
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={16}>
              <Card 
                title={
                  <Space>
                    <TrendingUpIcon size={20} />
                    时间模式分析
                  </Space>
                }
              >
                <Line {...temporalPatternsConfig} />
              </Card>
            </Col>
            <Col xs={24} lg={8}>
              <Card 
                title={
                  <Space>
                    <Target size={20} />
                    置信度分布
                  </Space>
                }
              >
                <Pie {...confidenceDistributionConfig} />
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="用户洞察" key="users">
          <Row gutter={[16, 16]}>
            {analytics?.user_patterns.map(user => (
              <Col xs={24} md={12} lg={8} key={user.user_id}>
                <Card size="small" title={user.user_id}>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div>
                      <Text>总决策数: </Text>
                      <Badge count={user.total_decisions} />
                    </div>
                    <div>
                      <Text>平均置信度: </Text>
                      <Progress 
                        percent={user.avg_confidence * 100} 
                        size="small"
                        style={{ width: 100 }}
                        format={() => `${(user.avg_confidence * 100).toFixed(1)}%`}
                      />
                    </div>
                    <div>
                      <Text>常用策略: </Text>
                      <Tag size="small">{user.most_used_strategy.replace(/_/g, ' ')}</Tag>
                    </div>
                    <div>
                      <Text>改善趋势: </Text>
                      <Badge 
                        status={
                          user.improvement_trend === 'improving' ? 'success' : 
                          user.improvement_trend === 'stable' ? 'processing' : 'error'
                        }
                        text={user.improvement_trend}
                      />
                    </div>
                  </Space>
                </Card>
              </Col>
            ))}
          </Row>
        </TabPane>
      </Tabs>

      {/* 决策详情模态框 */}
      <Modal
        title="决策详情"
        open={detailModalVisible}
        onCancel={() => {
          setDetailModalVisible(false);
          setSelectedDecision(null);
        }}
        footer={[
          <Button key="close" onClick={() => setDetailModalVisible(false)}>
            关闭
          </Button>,
          <Button key="analyze" type="primary">
            深度分析
          </Button>
        ]}
        width={800}
      >
        {selectedDecision && (
          <div>
            <Descriptions bordered column={2} size="small">
              <Descriptions.Item label="决策ID">{selectedDecision.decision_id}</Descriptions.Item>
              <Descriptions.Item label="用户ID">{selectedDecision.user_id}</Descriptions.Item>
              <Descriptions.Item label="选择策略">{selectedDecision.chosen_strategy}</Descriptions.Item>
              <Descriptions.Item label="决策类型">
                <Tag color={decisionTypeColors[selectedDecision.decision_type]}>
                  {selectedDecision.decision_type.toUpperCase()}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="置信度">
                <Progress 
                  percent={selectedDecision.confidence_score * 100} 
                  size="small"
                  style={{ width: 150 }}
                />
              </Descriptions.Item>
              <Descriptions.Item label="执行状态">
                <Badge 
                  status={statusColors[selectedDecision.execution_status as keyof typeof statusColors]} 
                  text={selectedDecision.execution_status.toUpperCase()} 
                />
              </Descriptions.Item>
            </Descriptions>

            <Card size="small" title="决策上下文" style={{ margin: '16px 0' }}>
              <Row gutter={16}>
                <Col span={8}>
                  <Text strong>情感状态: </Text>
                  <Tag>{selectedDecision.context.emotion_state}</Tag>
                </Col>
                <Col span={8}>
                  <Text strong>风险等级: </Text>
                  <Tag color={
                    selectedDecision.context.risk_level === 'high' ? 'red' :
                    selectedDecision.context.risk_level === 'medium' ? 'orange' : 'green'
                  }>
                    {selectedDecision.context.risk_level}
                  </Tag>
                </Col>
                <Col span={8}>
                  <Text strong>紧急程度: </Text>
                  <Tag color={
                    selectedDecision.context.urgency === 'immediate' ? 'red' :
                    selectedDecision.context.urgency === 'high' ? 'orange' : 'blue'
                  }>
                    {selectedDecision.context.urgency}
                  </Tag>
                </Col>
              </Row>
            </Card>

            <Card size="small" title="决策推理过程" style={{ margin: '16px 0' }}>
              <Timeline>
                {selectedDecision.reasoning.map((reason, index) => (
                  <Timeline.Item key={index}>
                    {reason}
                  </Timeline.Item>
                ))}
              </Timeline>
            </Card>

            {selectedDecision.outcome && (
              <Card size="small" title="执行结果" style={{ margin: '16px 0' }}>
                <Row gutter={16}>
                  <Col span={12}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <div>
                        <Text strong>效果评分: </Text>
                        <Progress 
                          percent={selectedDecision.outcome.effectiveness_score * 10} 
                          size="small"
                          format={() => `${selectedDecision.outcome!.effectiveness_score}/10`}
                        />
                      </div>
                      <div>
                        <Text strong>用户反馈: </Text>
                        <Text>{selectedDecision.outcome.user_feedback}/5</Text>
                      </div>
                    </Space>
                  </Col>
                  <Col span={12}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <div>
                        <Text strong>持续时间: </Text>
                        <Text>{selectedDecision.outcome.duration}天</Text>
                      </div>
                      <div>
                        <Text strong>需要跟进: </Text>
                        <Badge 
                          status={selectedDecision.outcome.follow_up_required ? 'warning' : 'success'}
                          text={selectedDecision.outcome.follow_up_required ? '是' : '否'}
                        />
                      </div>
                    </Space>
                  </Col>
                </Row>
              </Card>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
};

export default DecisionHistoryAnalysisPage;