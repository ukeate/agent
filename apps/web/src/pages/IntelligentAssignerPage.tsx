import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Button,
  Form,
  Input,
  Select,
  Table,
  Tag,
  Alert,
  Space,
  Modal,
  message,
  Statistic,
  Progress,
  Typography,
  Divider,
  Descriptions,
  Radio,
  Slider,
  Switch,
  Tooltip,
  Badge
} from 'antd';
import {
  UserOutlined,
  TeamOutlined,
  ThunderboltOutlined,
  BarChartOutlined,
  ClockCircleOutlined,
  DollarOutlined,
  GlobalOutlined,
  HeartOutlined,
  ReloadOutlined,
  PlayCircleOutlined,
  SettingOutlined,
  EyeOutlined,
  TrophyOutlined
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;

interface Agent {
  agent_id: string;
  name: string;
  capabilities: string[];
  performance_score: number;
  current_load: number;
  max_capacity: number;
  location: string;
  cost_per_hour: number;
  availability: boolean;
  specialization: string;
  success_rate: number;
  average_completion_time: number;
  resource_usage: {
    cpu: number;
    memory: number;
    disk: number;
  };
  task_history: number;
  current_tasks: string[];
}

interface AssignmentStrategy {
  key: string;
  name: string;
  description: string;
  icon: React.ReactNode;
  color: string;
  factors: string[];
  best_for: string[];
}

interface AssignmentResult {
  task_id: string;
  assigned_agent: string;
  strategy_used: string;
  score: number;
  reasoning: string;
  alternatives: { agent_id: string; score: number }[];
  assignment_time: string;
}

const IntelligentAssignerPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [assignmentHistory, setAssignmentHistory] = useState<AssignmentResult[]>([]);
  const [selectedStrategy, setSelectedStrategy] = useState<string>('capability_based');
  const [simulationVisible, setSimulationVisible] = useState(false);
  const [form] = Form.useForm();

  // 分配策略定义
  const assignmentStrategies: Record<string, AssignmentStrategy> = {
    capability_based: {
      key: 'capability_based',
      name: '基于能力分配',
      description: '根据智能体的能力匹配度分配任务',
      icon: <TrophyOutlined />,
      color: 'blue',
      factors: ['能力匹配', '专业技能', '成功率'],
      best_for: ['专业任务', '技能要求高', '质量优先']
    },
    load_balanced: {
      key: 'load_balanced',
      name: '负载均衡分配',
      description: '平衡各智能体的工作负载',
      icon: <BarChartOutlined />,
      color: 'green',
      factors: ['当前负载', '历史负载', '容量利用率'],
      best_for: ['大批量任务', '系统稳定性', '长期运行']
    },
    resource_optimized: {
      key: 'resource_optimized',
      name: '资源优化分配',
      description: '优化资源使用效率',
      icon: <ThunderboltOutlined />,
      color: 'orange',
      factors: ['资源使用率', 'CPU/内存', '成本效益'],
      best_for: ['资源敏感', '成本控制', '高效利用']
    },
    deadline_aware: {
      key: 'deadline_aware',
      name: '截止时间敏感',
      description: '考虑任务截止时间和处理速度',
      icon: <ClockCircleOutlined />,
      color: 'red',
      factors: ['完成时间', '处理速度', '可用时间'],
      best_for: ['时间敏感', '紧急任务', 'SLA要求']
    },
    cost_optimized: {
      key: 'cost_optimized',
      name: '成本优化分配',
      description: '选择成本最优的智能体',
      icon: <DollarOutlined />,
      color: 'gold',
      factors: ['小时成本', '效率比', '总成本'],
      best_for: ['成本敏感', '预算控制', '批量处理']
    },
    locality_aware: {
      key: 'locality_aware',
      name: '位置感知分配',
      description: '考虑地理位置和网络延迟',
      icon: <GlobalOutlined />,
      color: 'purple',
      factors: ['地理位置', '网络延迟', '数据本地性'],
      best_for: ['分布式系统', '数据本地化', '延迟敏感']
    },
    affinity_based: {
      key: 'affinity_based',
      name: '亲和性分配',
      description: '基于任务和智能体的亲和性',
      icon: <HeartOutlined />,
      color: 'magenta',
      factors: ['历史协作', '任务类型', '偏好设置'],
      best_for: ['团队协作', '个性化服务', '用户偏好']
    },
    priority_weighted: {
      key: 'priority_weighted',
      name: '优先级加权分配',
      description: '根据任务优先级加权选择',
      icon: <UserOutlined />,
      color: 'cyan',
      factors: ['任务优先级', '智能体等级', '权重分配'],
      best_for: ['多优先级', '分级处理', '重要性区分']
    }
  };

  // 模拟智能体数据
  const generateMockAgents = (): Agent[] => {
    const locations = ['北京', '上海', '深圳', '杭州', '成都'];
    const specializations = ['数据处理', '图像分析', '自然语言', '机器学习', '系统集成'];
    const capabilities = ['python', 'java', 'ml', 'nlp', 'cv', 'data', 'api', 'db'];

    return Array.from({ length: 12 }, (_, i) => ({
      agent_id: `agent_${String(i + 1).padStart(3, '0')}`,
      name: `智能体-${i + 1}`,
      capabilities: capabilities.slice(Math.floor(Math.random() * 3), Math.floor(Math.random() * 5) + 3),
      performance_score: Number((60 + Math.random() * 40).toFixed(1)),
      current_load: Number((Math.random() * 80).toFixed(1)),
      max_capacity: 100,
      location: locations[Math.floor(Math.random() * locations.length)],
      cost_per_hour: Number((10 + Math.random() * 40).toFixed(2)),
      availability: Math.random() > 0.2,
      specialization: specializations[Math.floor(Math.random() * specializations.length)],
      success_rate: Number((75 + Math.random() * 25).toFixed(1)),
      average_completion_time: Number((30 + Math.random() * 120).toFixed(1)),
      resource_usage: {
        cpu: Number((Math.random() * 60).toFixed(1)),
        memory: Number((Math.random() * 80).toFixed(1)),
        disk: Number((Math.random() * 40).toFixed(1))
      },
      task_history: Math.floor(Math.random() * 500) + 50,
      current_tasks: Array.from({ length: Math.floor(Math.random() * 3) }, (_, j) => `task_${i}_${j}`)
    }));
  };

  // 模拟任务分配
  const simulateAssignment = (strategy: string, taskData: any): AssignmentResult => {
    const availableAgents = agents.filter(agent => agent.availability);
    let bestAgent: Agent;
    let score: number;
    let reasoning: string;

    switch (strategy) {
      case 'capability_based':
        bestAgent = availableAgents.reduce((best, current) => {
          const currentScore = current.performance_score * current.success_rate / 100;
          const bestScore = best.performance_score * best.success_rate / 100;
          return currentScore > bestScore ? current : best;
        });
        score = bestAgent.performance_score * bestAgent.success_rate / 100;
        reasoning = `基于能力评分 ${bestAgent.performance_score} 和成功率 ${bestAgent.success_rate}% 选择`;
        break;

      case 'load_balanced':
        bestAgent = availableAgents.reduce((best, current) => 
          current.current_load < best.current_load ? current : best
        );
        score = 100 - bestAgent.current_load;
        reasoning = `选择负载最低的智能体，当前负载 ${bestAgent.current_load}%`;
        break;

      case 'resource_optimized':
        bestAgent = availableAgents.reduce((best, current) => {
          const currentUtil = (current.resource_usage.cpu + current.resource_usage.memory) / 2;
          const bestUtil = (best.resource_usage.cpu + best.resource_usage.memory) / 2;
          return currentUtil < bestUtil ? current : best;
        });
        score = 100 - (bestAgent.resource_usage.cpu + bestAgent.resource_usage.memory) / 2;
        reasoning = `选择资源利用率最低的智能体`;
        break;

      case 'cost_optimized':
        bestAgent = availableAgents.reduce((best, current) =>
          current.cost_per_hour < best.cost_per_hour ? current : best
        );
        score = 100 - (bestAgent.cost_per_hour / 50) * 100;
        reasoning = `选择成本最低的智能体，每小时 $${bestAgent.cost_per_hour}`;
        break;

      default:
        bestAgent = availableAgents[Math.floor(Math.random() * availableAgents.length)];
        score = Math.floor(Math.random() * 40) + 60;
        reasoning = `使用 ${assignmentStrategies[strategy]?.name} 策略选择`;
    }

    const alternatives = availableAgents
      .filter(agent => agent.agent_id !== bestAgent.agent_id)
      .slice(0, 3)
      .map(agent => ({
        agent_id: agent.agent_id,
        score: Math.floor(Math.random() * 30) + 50
      }));

    return {
      task_id: `task_${Date.now()}`,
      assigned_agent: bestAgent.agent_id,
      strategy_used: strategy,
      score: Number(score.toFixed(1)),
      reasoning,
      alternatives,
      assignment_time: new Date().toISOString()
    };
  };

  // 执行分配模拟
  const executeAssignment = async () => {
    const values = await form.validateFields();
    setLoading(true);

    try {
      const taskData = JSON.parse(values.taskData || '{}');
      const result = simulateAssignment(values.strategy, taskData);
      
      setAssignmentHistory(prev => [result, ...prev.slice(0, 19)]); // 保留最近20条
      message.success(`任务成功分配给 ${result.assigned_agent}，评分: ${result.score}`);
    } catch (error) {
      message.error('分配失败：' + error);
    } finally {
      setLoading(false);
    }
  };

  // 批量分配模拟
  const batchAssignment = async () => {
    setLoading(true);
    const strategies = Object.keys(assignmentStrategies);
    const results: AssignmentResult[] = [];

    for (let i = 0; i < 10; i++) {
      const strategy = strategies[i % strategies.length];
      const result = simulateAssignment(strategy, {});
      results.push(result);
      await new Promise(resolve => setTimeout(resolve, 100)); // 模拟延迟
    }

    setAssignmentHistory(prev => [...results, ...prev.slice(0, 10)]);
    setLoading(false);
    message.success(`批量分配完成，共处理 ${results.length} 个任务`);
  };

  // 初始化数据
  useEffect(() => {
    setAgents(generateMockAgents());
  }, []);

  // 智能体表格列
  const agentColumns: ColumnsType<Agent> = [
    {
      title: '智能体ID',
      dataIndex: 'agent_id',
      key: 'agent_id',
      render: (id: string, record: Agent) => (
        <Space>
          <Badge status={record.availability ? 'success' : 'error'} />
          <Text code>{id}</Text>
        </Space>
      )
    },
    {
      title: '名称',
      dataIndex: 'name',
      key: 'name'
    },
    {
      title: '专业领域',
      dataIndex: 'specialization',
      key: 'specialization',
      render: (spec: string) => <Tag color="blue">{spec}</Tag>
    },
    {
      title: '能力',
      dataIndex: 'capabilities',
      key: 'capabilities',
      render: (caps: string[]) => (
        <Space wrap>
          {caps.slice(0, 3).map(cap => (
            <Tag key={cap} size="small">{cap}</Tag>
          ))}
          {caps.length > 3 && <Tag size="small">+{caps.length - 3}</Tag>}
        </Space>
      )
    },
    {
      title: '性能评分',
      dataIndex: 'performance_score',
      key: 'performance_score',
      render: (score: number) => (
        <Space>
          <Progress
            percent={score}
            size="small"
            status={score > 80 ? 'success' : score > 60 ? 'normal' : 'exception'}
            format={() => score}
          />
        </Space>
      )
    },
    {
      title: '当前负载',
      dataIndex: 'current_load',
      key: 'current_load',
      render: (load: number, record: Agent) => (
        <Progress
          percent={load}
          size="small"
          status={load > 80 ? 'exception' : 'normal'}
          format={() => `${load}%`}
        />
      )
    },
    {
      title: '成功率',
      dataIndex: 'success_rate',
      key: 'success_rate',
      render: (rate: number) => (
        <Text style={{ color: rate > 90 ? '#52c41a' : rate > 80 ? '#faad14' : '#f5222d' }}>
          {rate}%
        </Text>
      )
    },
    {
      title: '成本/小时',
      dataIndex: 'cost_per_hour',
      key: 'cost_per_hour',
      render: (cost: number) => <Text>${cost}</Text>
    },
    {
      title: '位置',
      dataIndex: 'location',
      key: 'location',
      render: (location: string) => <Tag color="geekblue">{location}</Tag>
    }
  ];

  // 分配历史表格列
  const historyColumns: ColumnsType<AssignmentResult> = [
    {
      title: '任务ID',
      dataIndex: 'task_id',
      key: 'task_id',
      render: (id: string) => <Text code>{id.substring(5, 13)}</Text>
    },
    {
      title: '分配给',
      dataIndex: 'assigned_agent',
      key: 'assigned_agent',
      render: (agentId: string) => <Tag color="blue">{agentId}</Tag>
    },
    {
      title: '策略',
      dataIndex: 'strategy_used',
      key: 'strategy_used',
      render: (strategy: string) => (
        <Tag color={assignmentStrategies[strategy]?.color || 'default'}>
          {assignmentStrategies[strategy]?.name || strategy}
        </Tag>
      )
    },
    {
      title: '评分',
      dataIndex: 'score',
      key: 'score',
      render: (score: number) => (
        <Tag color={score > 85 ? 'success' : score > 70 ? 'warning' : 'error'}>
          {score}
        </Tag>
      )
    },
    {
      title: '分配理由',
      dataIndex: 'reasoning',
      key: 'reasoning',
      ellipsis: true
    },
    {
      title: '分配时间',
      dataIndex: 'assignment_time',
      key: 'assignment_time',
      render: (time: string) => new Date(time).toLocaleTimeString()
    }
  ];

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <Title level={2}>智能任务分配器</Title>
      <Paragraph>
        基于多种策略的智能任务分配系统，支持能力匹配、负载均衡、资源优化等8种分配策略。
      </Paragraph>

      {/* 策略选择和配置 */}
      <Card title="分配策略配置" style={{ marginBottom: 24 }}>
        <Row gutter={24}>
          <Col span={16}>
            <Form
              form={form}
              layout="vertical"
              initialValues={{ strategy: 'capability_based' }}
            >
              <Form.Item
                label="分配策略"
                name="strategy"
                rules={[{ required: true, message: '请选择分配策略' }]}
              >
                <Radio.Group onChange={(e) => setSelectedStrategy(e.target.value)}>
                  <Row gutter={[16, 16]}>
                    {Object.entries(assignmentStrategies).map(([key, strategy]) => (
                      <Col span={12} key={key}>
                        <Radio value={key} style={{ width: '100%' }}>
                          <Card
                            size="small"
                            style={{
                              border: selectedStrategy === key ? '2px solid #1890ff' : '1px solid #d9d9d9'
                            }}
                          >
                            <Space>
                              {strategy.icon}
                              <div>
                                <Text strong>{strategy.name}</Text>
                                <br />
                                <Text type="secondary" style={{ fontSize: '12px' }}>
                                  {strategy.description}
                                </Text>
                              </div>
                            </Space>
                          </Card>
                        </Radio>
                      </Col>
                    ))}
                  </Row>
                </Radio.Group>
              </Form.Item>

              <Form.Item
                label="任务数据 (JSON)"
                name="taskData"
                help="可选：提供任务具体数据用于更精准的分配"
              >
                <Input.TextArea
                  rows={3}
                  placeholder='{"type": "data_processing", "priority": "high", "requirements": {"cpu": 0.5}}'
                />
              </Form.Item>

              <Form.Item>
                <Space>
                  <Button
                    type="primary"
                    onClick={executeAssignment}
                    loading={loading}
                    icon={<PlayCircleOutlined />}
                  >
                    执行分配
                  </Button>
                  <Button
                    onClick={batchAssignment}
                    loading={loading}
                    icon={<TeamOutlined />}
                  >
                    批量分配 (10个任务)
                  </Button>
                  <Button
                    onClick={() => setAgents(generateMockAgents())}
                    icon={<ReloadOutlined />}
                  >
                    刷新智能体
                  </Button>
                </Space>
              </Form.Item>
            </Form>
          </Col>

          <Col span={8}>
            {selectedStrategy && (
              <Card title="策略详情" size="small">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>
                    <Text strong>考虑因素：</Text>
                    <ul style={{ margin: '8px 0', paddingLeft: '20px' }}>
                      {assignmentStrategies[selectedStrategy].factors.map((factor, index) => (
                        <li key={index}><Text type="secondary">{factor}</Text></li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <Text strong>适用场景：</Text>
                    <ul style={{ margin: '8px 0', paddingLeft: '20px' }}>
                      {assignmentStrategies[selectedStrategy].best_for.map((scenario, index) => (
                        <li key={index}><Text type="secondary">{scenario}</Text></li>
                      ))}
                    </ul>
                  </div>
                </Space>
              </Card>
            )}
          </Col>
        </Row>
      </Card>

      {/* 智能体状态总览 */}
      <Card title="智能体状态总览" style={{ marginBottom: 24 }}>
        <Row gutter={16}>
          <Col span={6}>
            <Statistic
              title="总智能体数"
              value={agents.length}
              suffix="个"
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="可用智能体"
              value={agents.filter(a => a.availability).length}
              suffix="个"
              valueStyle={{ color: '#3f8600' }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="平均负载"
              value={Number((agents.reduce((sum, a) => sum + a.current_load, 0) / agents.length).toFixed(1))}
              suffix="%"
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="分配历史"
              value={assignmentHistory.length}
              suffix="次"
            />
          </Col>
        </Row>
      </Card>

      {/* 详细信息标签页 */}
      <Row gutter={24}>
        <Col span={14}>
          <Card title="智能体详情" style={{ marginBottom: 16 }}>
            <Table
              columns={agentColumns}
              dataSource={agents}
              rowKey="agent_id"
              pagination={{ pageSize: 8, showSizeChanger: false }}
              size="small"
              scroll={{ x: 'max-content' }}
            />
          </Card>
        </Col>

        <Col span={10}>
          <Card title="分配历史" style={{ marginBottom: 16 }}>
            {assignmentHistory.length === 0 ? (
              <Alert
                message="暂无分配记录"
                description="执行任务分配后将显示分配历史"
                type="info"
                showIcon
              />
            ) : (
              <Table
                columns={historyColumns}
                dataSource={assignmentHistory}
                rowKey="task_id"
                pagination={{ pageSize: 6, showSizeChanger: false }}
                size="small"
              />
            )}
          </Card>

          {/* 分配统计 */}
          {assignmentHistory.length > 0 && (
            <Card title="分配统计" size="small">
              <Row gutter={8}>
                <Col span={12}>
                  <Statistic
                    title="平均评分"
                    value={Number((assignmentHistory.reduce((sum, h) => sum + h.score, 0) / assignmentHistory.length).toFixed(1))}
                    precision={1}
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="最高评分"
                    value={Math.max(...assignmentHistory.map(h => h.score))}
                    precision={1}
                  />
                </Col>
              </Row>
            </Card>
          )}
        </Col>
      </Row>
    </div>
  );
};

export default IntelligentAssignerPage;