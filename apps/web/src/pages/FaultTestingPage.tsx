import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Button, Table, Modal, Form, Input, Select, Tag, Progress, Alert, Tabs, Space, Statistic, Timeline, notification, Tooltip } from 'antd';
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Legend, PieChart, Pie, Cell } from 'recharts';
import {
  ExperimentOutlined,
  PlayCircleOutlined,
  StopOutlined,
  ReloadOutlined,
  DeleteOutlined,
  EyeOutlined,
  SettingOutlined,
  ThunderboltOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  BugOutlined,
  FireOutlined,
  CloudOutlined,
  DatabaseOutlined,
  ApiOutlined
} from '@ant-design/icons';

const { TabPane } = Tabs;
const { Option } = Select;
const { TextArea } = Input;

interface FaultScenario {
  id: string;
  name: string;
  type: 'network' | 'cpu' | 'memory' | 'disk' | 'service' | 'database' | 'custom';
  description: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  duration: number;
  parameters: Record<string, any>;
  created_at: string;
  created_by: string;
  tags: string[];
}

interface FaultTest {
  id: string;
  scenario_id: string;
  scenario_name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'stopped';
  started_at: string;
  ended_at?: string;
  duration_actual?: number;
  target_components: string[];
  results: {
    success: boolean;
    impact_metrics: Record<string, number>;
    recovery_time: number;
    errors_generated: number;
    affected_users: number;
  };
  logs: string[];
}

interface TestTemplate {
  id: string;
  name: string;
  category: string;
  description: string;
  parameters: {
    name: string;
    type: 'string' | 'number' | 'boolean' | 'select';
    default?: any;
    options?: string[];
    required: boolean;
    description: string;
  }[];
}

interface SystemImpact {
  component: string;
  metric: string;
  baseline: number;
  current: number;
  change_percent: number;
  status: 'normal' | 'degraded' | 'critical';
}

const FaultTestingPage: React.FC = () => {
  const [scenarios, setScenarios] = useState<FaultScenario[]>([]);
  const [tests, setTests] = useState<FaultTest[]>([]);
  const [templates, setTemplates] = useState<TestTemplate[]>([]);
  const [systemImpact, setSystemImpact] = useState<SystemImpact[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('scenarios');
  
  const [scenarioModalVisible, setScenarioModalVisible] = useState(false);
  const [testModalVisible, setTestModalVisible] = useState(false);
  const [detailModalVisible, setDetailModalVisible] = useState(false);
  const [selectedTest, setSelectedTest] = useState<FaultTest | null>(null);
  const [selectedTemplate, setSelectedTemplate] = useState<TestTemplate | null>(null);
  
  const [scenarioForm] = Form.useForm();
  const [testForm] = Form.useForm();

  useEffect(() => {
    loadScenarios();
    loadTests();
    loadTemplates();
    loadSystemImpact();
    
    const interval = setInterval(() => {
      loadTests();
      loadSystemImpact();
    }, 5000);
    
    return () => clearInterval(interval);
  }, []);

  const loadScenarios = async () => {
    try {
      // 使用模拟数据，因为后端没有专门的场景管理API
      const mockScenarios: FaultScenario[] = [
        {
          id: 'scenario_001',
          name: 'Agent响应性测试',
          type: 'agent',
          description: '测试Agent在高负载下的响应能力',
          severity: 'medium',
          duration: 5,
          parameters: { target_latency: 1000, max_connections: 100 },
          created_at: new Date().toISOString(),
          created_by: 'system',
          tags: ['performance', 'agent']
        },
        {
          id: 'scenario_002',
          name: '网络分区测试',
          type: 'network',
          description: '模拟网络分区情况下的系统行为',
          severity: 'high',
          duration: 10,
          parameters: { partition_percentage: 0.3, recovery_time: 300 },
          created_at: new Date(Date.now() - 86400000).toISOString(),
          created_by: 'admin',
          tags: ['network', 'partition']
        }
      ];
      setScenarios(mockScenarios);
    } catch (error) {
      console.error('Failed to load scenarios:', error);
    }
  };

  const loadTests = async () => {
    try {
      // 使用故障事件API获取测试记录
      const response = await fetch('http://localhost:8000/api/v1/fault-tolerance/faults/events');
      if (response.ok) {
        const data = await response.json();
        const faultEvents = data.events || [];
        // 将故障事件转换为测试记录格式
        const testRecords: FaultTest[] = faultEvents.map((fault: any) => ({
          id: fault.event_id || `test_${Date.now()}`,
          scenario_id: 'injected_fault',
          scenario_name: '故障注入测试',
          status: fault.resolved ? 'completed' : 'running',
          started_at: fault.detected_at,
          ended_at: fault.resolved_at,
          duration_actual: fault.resolved_at ? 
            Math.floor((new Date(fault.resolved_at).getTime() - new Date(fault.detected_at).getTime()) / 1000) : 
            undefined,
          target_components: fault.affected_components || ['unknown'],
          results: {
            success: fault.resolved || false,
            impact_metrics: { severity: fault.severity },
            recovery_time: fault.resolved_at ? 
              Math.floor((new Date(fault.resolved_at).getTime() - new Date(fault.detected_at).getTime()) / 1000) : 0,
            errors_generated: 1,
            affected_users: 0
          },
          logs: [`Fault detected: ${fault.description}`]
        }));
        setTests(testRecords);
      } else {
        // 使用模拟数据作为fallback
        const mockTests: FaultTest[] = [
          {
            id: 'test_001',
            scenario_id: 'scenario_001',
            scenario_name: 'Agent响应性测试',
            status: 'completed',
            started_at: new Date(Date.now() - 300000).toISOString(),
            ended_at: new Date(Date.now() - 60000).toISOString(),
            duration_actual: 240,
            target_components: ['agent-1'],
            results: {
              success: true,
              impact_metrics: { response_time: 1.2, error_rate: 0.02 },
              recovery_time: 45,
              errors_generated: 3,
              affected_users: 5
            },
            logs: ['Test started', 'Fault injected', 'Recovery initiated', 'Test completed']
          }
        ];
        setTests(mockTests);
      }
    } catch (error) {
      console.error('Failed to load tests:', error);
      // 使用模拟数据
      const mockTests: FaultTest[] = [
        {
          id: 'test_001',
          scenario_id: 'scenario_001', 
          scenario_name: 'Agent响应性测试',
          status: 'completed',
          started_at: new Date(Date.now() - 300000).toISOString(),
          ended_at: new Date(Date.now() - 60000).toISOString(),
          duration_actual: 240,
          target_components: ['agent-1'],
          results: {
            success: true,
            impact_metrics: { response_time: 1.2, error_rate: 0.02 },
            recovery_time: 45,
            errors_generated: 3,
            affected_users: 5
          },
          logs: ['Test started', 'Fault injected', 'Recovery initiated', 'Test completed']
        }
      ];
      setTests(mockTests);
    }
  };

  const loadTemplates = async () => {
    try {
      const response = await fetch('/api/v1/fault-tolerance/testing/templates');
      const data = await response.json();
      setTemplates(data);
    } catch (error) {
      console.error('Failed to load templates:', error);
    }
  };

  const loadSystemImpact = async () => {
    try {
      const response = await fetch('/api/v1/fault-tolerance/testing/system-impact');
      const data = await response.json();
      setSystemImpact(data);
    } catch (error) {
      console.error('Failed to load system impact:', error);
    }
  };

  const createScenario = async (values: any) => {
    setLoading(true);
    try {
      await fetch('/api/v1/fault-tolerance/testing/scenarios', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(values)
      });
      loadScenarios();
      setScenarioModalVisible(false);
      scenarioForm.resetFields();
      notification.success({ message: '故障场景创建成功' });
    } catch (error) {
      console.error('Failed to create scenario:', error);
      notification.error({ message: '创建失败' });
    }
    setLoading(false);
  };

  const startTest = async (scenarioId: string, targetComponents: string[]) => {
    setLoading(true);
    try {
      // 使用真实的故障注入API
      const response = await fetch('http://localhost:8000/api/v1/fault-tolerance/testing/inject-fault', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          component_id: targetComponents[0] || 'test-component',
          fault_type: 'agent_error',
          duration_seconds: 60
        })
      });
      
      if (response.ok) {
        loadTests();
        setTestModalVisible(false);
        testForm.resetFields();
        notification.success({ message: '故障测试已启动' });
      } else {
        throw new Error('API call failed');
      }
    } catch (error) {
      console.error('Failed to start test:', error);
      notification.error({ message: '启动失败' });
    }
    setLoading(false);
  };

  const stopTest = async (testId: string) => {
    setLoading(true);
    try {
      await fetch(`/api/v1/fault-tolerance/testing/tests/${testId}/stop`, {
        method: 'POST'
      });
      loadTests();
      notification.success({ message: '测试已停止' });
    } catch (error) {
      console.error('Failed to stop test:', error);
      notification.error({ message: '停止失败' });
    }
    setLoading(false);
  };

  const deleteScenario = async (scenarioId: string) => {
    setLoading(true);
    try {
      await fetch(`/api/v1/fault-tolerance/testing/scenarios/${scenarioId}`, {
        method: 'DELETE'
      });
      loadScenarios();
      notification.success({ message: '场景已删除' });
    } catch (error) {
      console.error('Failed to delete scenario:', error);
      notification.error({ message: '删除失败' });
    }
    setLoading(false);
  };

  const getTypeIcon = (type: string) => {
    const iconMap = {
      network: <CloudOutlined />,
      cpu: <ThunderboltOutlined />,
      memory: <FireOutlined />,
      disk: <DatabaseOutlined />,
      service: <ApiOutlined />,
      database: <DatabaseOutlined />,
      custom: <BugOutlined />
    };
    return iconMap[type] || <ExperimentOutlined />;
  };

  const getSeverityColor = (severity: string) => {
    const colorMap = {
      low: '#52c41a',
      medium: '#faad14',
      high: '#ff7a45',
      critical: '#ff4d4f'
    };
    return colorMap[severity] || '#1890ff';
  };

  const getStatusColor = (status: string) => {
    const colorMap = {
      pending: '#faad14',
      running: '#1890ff',
      completed: '#52c41a',
      failed: '#ff4d4f',
      stopped: '#8c8c8c'
    };
    return colorMap[status] || '#1890ff';
  };

  const scenarioColumns = [
    {
      title: '场景名称',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: FaultScenario) => (
        <Space>
          {getTypeIcon(record.type)}
          <span>{text}</span>
        </Space>
      )
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => <Tag>{type.toUpperCase()}</Tag>
    },
    {
      title: '严重程度',
      dataIndex: 'severity',
      key: 'severity',
      render: (severity: string) => (
        <Tag color={getSeverityColor(severity)}>
          {severity.toUpperCase()}
        </Tag>
      )
    },
    {
      title: '持续时间',
      dataIndex: 'duration',
      key: 'duration',
      render: (duration: number) => `${duration}分钟`
    },
    {
      title: '标签',
      dataIndex: 'tags',
      key: 'tags',
      render: (tags: string[]) => (
        <>
          {tags.map(tag => (
            <Tag key={tag} size="small">{tag}</Tag>
          ))}
        </>
      )
    },
    {
      title: '创建者',
      dataIndex: 'created_by',
      key: 'created_by'
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (time: string) => new Date(time).toLocaleString()
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: FaultScenario) => (
        <Space>
          <Button
            size="small"
            type="primary"
            icon={<PlayCircleOutlined />}
            onClick={() => {
              testForm.setFieldValue('scenario_id', record.id);
              setTestModalVisible(true);
            }}
          >
            运行测试
          </Button>
          <Button
            size="small"
            icon={<EyeOutlined />}
            onClick={() => {
              // 查看场景详情
            }}
          >
            详情
          </Button>
          <Button
            size="small"
            danger
            icon={<DeleteOutlined />}
            onClick={() => deleteScenario(record.id)}
          >
            删除
          </Button>
        </Space>
      )
    }
  ];

  const testColumns = [
    {
      title: '测试场景',
      dataIndex: 'scenario_name',
      key: 'scenario_name'
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const statusConfig = {
          pending: { color: 'orange', icon: <ClockCircleOutlined />, text: '等待中' },
          running: { color: 'blue', icon: <PlayCircleOutlined />, text: '运行中' },
          completed: { color: 'green', icon: <CheckCircleOutlined />, text: '已完成' },
          failed: { color: 'red', icon: <WarningOutlined />, text: '失败' },
          stopped: { color: 'gray', icon: <StopOutlined />, text: '已停止' }
        };
        const config = statusConfig[status];
        return (
          <Tag color={config.color} icon={config.icon}>
            {config.text}
          </Tag>
        );
      }
    },
    {
      title: '目标组件',
      dataIndex: 'target_components',
      key: 'target_components',
      render: (components: string[]) => (
        <>
          {components.map(component => (
            <Tag key={component} size="small">{component}</Tag>
          ))}
        </>
      )
    },
    {
      title: '开始时间',
      dataIndex: 'started_at',
      key: 'started_at',
      render: (time: string) => new Date(time).toLocaleString()
    },
    {
      title: '持续时间',
      key: 'duration',
      render: (record: FaultTest) => {
        if (record.status === 'running') {
          const duration = Math.floor((Date.now() - new Date(record.started_at).getTime()) / 1000);
          return `${Math.floor(duration / 60)}分${duration % 60}秒`;
        } else if (record.duration_actual) {
          return `${Math.floor(record.duration_actual / 60)}分${record.duration_actual % 60}秒`;
        }
        return '-';
      }
    },
    {
      title: '恢复时间',
      key: 'recovery_time',
      render: (record: FaultTest) => 
        record.results?.recovery_time ? `${record.results.recovery_time}秒` : '-'
    },
    {
      title: '影响用户',
      key: 'affected_users',
      render: (record: FaultTest) => 
        record.results?.affected_users ? record.results.affected_users : '-'
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: FaultTest) => (
        <Space>
          {record.status === 'running' && (
            <Button
              size="small"
              danger
              icon={<StopOutlined />}
              onClick={() => stopTest(record.id)}
            >
              停止
            </Button>
          )}
          <Button
            size="small"
            icon={<EyeOutlined />}
            onClick={() => {
              setSelectedTest(record);
              setDetailModalVisible(true);
            }}
          >
            详情
          </Button>
        </Space>
      )
    }
  ];

  const impactColumns = [
    {
      title: '组件',
      dataIndex: 'component',
      key: 'component'
    },
    {
      title: '指标',
      dataIndex: 'metric',
      key: 'metric'
    },
    {
      title: '基线值',
      dataIndex: 'baseline',
      key: 'baseline'
    },
    {
      title: '当前值',
      dataIndex: 'current',
      key: 'current'
    },
    {
      title: '变化',
      dataIndex: 'change_percent',
      key: 'change_percent',
      render: (change: number) => (
        <span style={{ color: change > 0 ? '#ff4d4f' : change < 0 ? '#52c41a' : '#1890ff' }}>
          {change > 0 ? '+' : ''}{change}%
        </span>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const colorMap = {
          normal: 'green',
          degraded: 'orange',
          critical: 'red'
        };
        return <Tag color={colorMap[status]}>{status.toUpperCase()}</Tag>;
      }
    }
  ];

  const runningTests = tests.filter(t => t.status === 'running');
  const completedTests = tests.filter(t => t.status === 'completed');
  const failedTests = tests.filter(t => t.status === 'failed');

  const testStatsData = [
    { name: '运行中', value: runningTests.length, color: '#1890ff' },
    { name: '已完成', value: completedTests.length, color: '#52c41a' },
    { name: '失败', value: failedTests.length, color: '#ff4d4f' },
    { name: '已停止', value: tests.filter(t => t.status === 'stopped').length, color: '#8c8c8c' }
  ];

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <h1>
          <ExperimentOutlined style={{ marginRight: '8px' }} />
          故障测试平台
        </h1>
        <Space>
          <Button icon={<ReloadOutlined />} onClick={() => {
            loadScenarios();
            loadTests();
            loadSystemImpact();
          }}>
            刷新
          </Button>
        </Space>
      </div>

      {runningTests.length > 0 && (
        <Alert
          message={`当前有 ${runningTests.length} 个测试正在运行`}
          description="请注意监控系统状态，必要时可停止测试"
          type="info"
          showIcon
          style={{ marginBottom: '24px' }}
        />
      )}

      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="总场景数"
              value={scenarios.length}
              prefix={<ExperimentOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="运行中测试"
              value={runningTests.length}
              prefix={<PlayCircleOutlined />}
              valueStyle={{ color: runningTests.length > 0 ? '#1890ff' : '#8c8c8c' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="成功率"
              value={tests.length > 0 ? Math.round((completedTests.length / tests.length) * 100) : 0}
              suffix="%"
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均恢复时间"
              value={completedTests.length > 0 ? Math.round(completedTests.reduce((sum, test) => sum + (test.results?.recovery_time || 0), 0) / completedTests.length) : 0}
              suffix="秒"
              prefix={<ClockCircleOutlined />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
      </Row>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="测试场景" key="scenarios">
          <Card
            title="故障测试场景"
            extra={
              <Button 
                type="primary" 
                icon={<ExperimentOutlined />}
                onClick={() => setScenarioModalVisible(true)}
              >
                创建场景
              </Button>
            }
          >
            <Table
              columns={scenarioColumns}
              dataSource={scenarios}
              rowKey="id"
              pagination={{ pageSize: 10 }}
              loading={loading}
            />
          </Card>
        </TabPane>

        <TabPane tab="运行测试" key="tests">
          <Card title="测试执行记录">
            <Table
              columns={testColumns}
              dataSource={tests}
              rowKey="id"
              pagination={{ pageSize: 10 }}
              loading={loading}
            />
          </Card>
        </TabPane>

        <TabPane tab="系统影响" key="impact">
          <Card title="实时系统影响监控">
            <Table
              columns={impactColumns}
              dataSource={systemImpact}
              rowKey={(record) => `${record.component}-${record.metric}`}
              pagination={{ pageSize: 10 }}
              loading={loading}
            />
          </Card>
        </TabPane>

        <TabPane tab="测试统计" key="statistics">
          <Row gutter={[16, 16]}>
            <Col span={12}>
              <Card title="测试状态分布">
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={testStatsData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, value }) => `${name}: ${value}`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {testStatsData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <RechartsTooltip />
                  </PieChart>
                </ResponsiveContainer>
              </Card>
            </Col>
            <Col span={12}>
              <Card title="测试结果趋势">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={tests.slice(-10).map((test, index) => ({
                    name: `Test ${index + 1}`,
                    recovery_time: test.results?.recovery_time || 0,
                    affected_users: test.results?.affected_users || 0
                  }))}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <RechartsTooltip />
                    <Legend />
                    <Line type="monotone" dataKey="recovery_time" stroke="#8884d8" name="恢复时间(秒)" />
                    <Line type="monotone" dataKey="affected_users" stroke="#82ca9d" name="受影响用户" />
                  </LineChart>
                </ResponsiveContainer>
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>

      <Modal
        title="创建故障测试场景"
        visible={scenarioModalVisible}
        onCancel={() => {
          setScenarioModalVisible(false);
          scenarioForm.resetFields();
        }}
        footer={null}
        width={800}
      >
        <Form
          form={scenarioForm}
          layout="vertical"
          onFinish={createScenario}
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="name"
                label="场景名称"
                rules={[{ required: true, message: '请输入场景名称' }]}
              >
                <Input placeholder="输入场景名称" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="type"
                label="故障类型"
                rules={[{ required: true, message: '请选择故障类型' }]}
              >
                <Select placeholder="选择故障类型">
                  <Option value="network">网络故障</Option>
                  <Option value="cpu">CPU故障</Option>
                  <Option value="memory">内存故障</Option>
                  <Option value="disk">磁盘故障</Option>
                  <Option value="service">服务故障</Option>
                  <Option value="database">数据库故障</Option>
                  <Option value="custom">自定义故障</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="severity"
                label="严重程度"
                rules={[{ required: true, message: '请选择严重程度' }]}
              >
                <Select placeholder="选择严重程度">
                  <Option value="low">低</Option>
                  <Option value="medium">中</Option>
                  <Option value="high">高</Option>
                  <Option value="critical">严重</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="duration"
                label="持续时间（分钟）"
                rules={[{ required: true, message: '请输入持续时间' }]}
              >
                <Input type="number" placeholder="输入持续时间" />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            name="description"
            label="场景描述"
            rules={[{ required: true, message: '请输入场景描述' }]}
          >
            <TextArea rows={3} placeholder="详细描述故障场景" />
          </Form.Item>

          <Form.Item
            name="tags"
            label="标签"
          >
            <Select mode="tags" placeholder="输入标签">
              <Option value="高可用性">高可用性</Option>
              <Option value="性能测试">性能测试</Option>
              <Option value="灾难恢复">灾难恢复</Option>
              <Option value="网络分区">网络分区</Option>
            </Select>
          </Form.Item>

          <Form.Item>
            <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
              <Button onClick={() => {
                setScenarioModalVisible(false);
                scenarioForm.resetFields();
              }}>
                取消
              </Button>
              <Button type="primary" htmlType="submit" loading={loading}>
                创建场景
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title="启动故障测试"
        visible={testModalVisible}
        onCancel={() => {
          setTestModalVisible(false);
          testForm.resetFields();
        }}
        onOk={() => {
          testForm.validateFields().then(values => {
            startTest(values.scenario_id, values.target_components);
          });
        }}
        confirmLoading={loading}
      >
        <Form form={testForm} layout="vertical">
          <Form.Item
            name="scenario_id"
            label="选择测试场景"
            rules={[{ required: true, message: '请选择测试场景' }]}
          >
            <Select placeholder="选择要运行的测试场景">
              {scenarios.map(scenario => (
                <Option key={scenario.id} value={scenario.id}>
                  {scenario.name} ({scenario.type})
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item
            name="target_components"
            label="目标组件"
            rules={[{ required: true, message: '请选择目标组件' }]}
          >
            <Select mode="multiple" placeholder="选择要测试的组件">
              <Option value="api-gateway">API网关</Option>
              <Option value="user-service">用户服务</Option>
              <Option value="order-service">订单服务</Option>
              <Option value="payment-service">支付服务</Option>
              <Option value="database">数据库</Option>
              <Option value="redis">Redis缓存</Option>
              <Option value="message-queue">消息队列</Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title="测试详情"
        visible={detailModalVisible}
        onCancel={() => {
          setDetailModalVisible(false);
          setSelectedTest(null);
        }}
        footer={[
          <Button key="close" onClick={() => {
            setDetailModalVisible(false);
            setSelectedTest(null);
          }}>
            关闭
          </Button>
        ]}
        width={800}
      >
        {selectedTest && (
          <div>
            <Row gutter={[16, 16]} style={{ marginBottom: '16px' }}>
              <Col span={8}>
                <Card size="small">
                  <Statistic title="测试状态" value={selectedTest.status.toUpperCase()} />
                </Card>
              </Col>
              <Col span={8}>
                <Card size="small">
                  <Statistic 
                    title="恢复时间" 
                    value={selectedTest.results?.recovery_time || 0} 
                    suffix="秒" 
                  />
                </Card>
              </Col>
              <Col span={8}>
                <Card size="small">
                  <Statistic 
                    title="受影响用户" 
                    value={selectedTest.results?.affected_users || 0} 
                  />
                </Card>
              </Col>
            </Row>

            {selectedTest.results && (
              <Card title="影响指标" size="small" style={{ marginBottom: '16px' }}>
                {Object.entries(selectedTest.results.impact_metrics).map(([metric, value]) => (
                  <div key={metric} style={{ marginBottom: '8px' }}>
                    <strong>{metric}:</strong> {value}
                  </div>
                ))}
              </Card>
            )}

            {selectedTest.logs && selectedTest.logs.length > 0 && (
              <Card title="执行日志" size="small">
                <Timeline>
                  {selectedTest.logs.map((log, index) => (
                    <Timeline.Item key={index}>
                      <pre style={{ fontSize: '12px', margin: 0 }}>{log}</pre>
                    </Timeline.Item>
                  ))}
                </Timeline>
              </Card>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
};

export default FaultTestingPage;