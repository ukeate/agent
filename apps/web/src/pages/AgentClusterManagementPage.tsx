import React, { useState, useEffect } from 'react';
import {
import { logger } from '../utils/logger'
  Layout,
  Card,
  Row,
  Col,
  Table,
  Button,
  Tabs,
  Statistic,
  Progress,
  Tag,
  Space,
  Modal,
  Form,
  Input,
  InputNumber,
  Select,
  Switch,
  message,
  Descriptions,
  Tooltip,
  Typography,
  Alert,
  Divider,
  Badge,
  Popconfirm
} from 'antd';
import {
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  DeleteOutlined,
  PlusOutlined,
  SettingOutlined,
  MonitorOutlined,
  ClusterOutlined,
  BranchesOutlined,
  ThunderboltOutlined,
  InfoCircleOutlined,
  WarningOutlined,
  CheckCircleOutlined
} from '@ant-design/icons';
import {
  clusterManagementService,
  AgentInfo,
  ClusterStats,
  AgentGroup,
  ScalingPolicy,
  AgentCreateRequest,
  GroupCreateRequest,
  ScalingPolicyRequest,
  ManualScalingRequest
} from '../services/clusterManagementService';

const { Title, Text } = Typography;
const { TabPane } = Tabs;

const AgentClusterManagementPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [groups, setGroups] = useState<AgentGroup[]>([]);
  const [policies, setPolicies] = useState<ScalingPolicy[]>([]);
  const [clusterStats, setClusterStats] = useState<ClusterStats | null>(null);
  const [selectedAgent, setSelectedAgent] = useState<AgentInfo | null>(null);
  const [createAgentVisible, setCreateAgentVisible] = useState(false);
  const [createGroupVisible, setCreateGroupVisible] = useState(false);
  const [createPolicyVisible, setCreatePolicyVisible] = useState(false);
  const [manualScalingVisible, setManualScalingVisible] = useState(false);
  const [selectedGroup, setSelectedGroup] = useState<string>('');
  const [agentDetailsVisible, setAgentDetailsVisible] = useState(false);
  const [activeTab, setActiveTab] = useState('agents');

  const [createAgentForm] = Form.useForm();
  const [createGroupForm] = Form.useForm();
  const [createPolicyForm] = Form.useForm();
  const [manualScalingForm] = Form.useForm();

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [agentsData, groupsData, policiesData, statsData] = await Promise.all([
        clusterManagementService.getAgents(),
        clusterManagementService.getGroups(),
        clusterManagementService.getScalingPolicies(),
        clusterManagementService.getClusterStats()
      ]);

      setAgents(agentsData);
      setGroups(groupsData);
      setPolicies(policiesData);
      setClusterStats(statsData);
    } catch (error) {
      message.error('加载数据失败');
      logger.error(error);
    } finally {
      setLoading(false);
    }
  };

  const handleAgentAction = async (agentId: string, action: 'start' | 'stop' | 'restart' | 'delete') => {
    try {
      let result;
      switch (action) {
        case 'start':
          result = await clusterManagementService.startAgent(agentId);
          break;
        case 'stop':
          result = await clusterManagementService.stopAgent(agentId);
          break;
        case 'restart':
          result = await clusterManagementService.restartAgent(agentId);
          break;
        case 'delete':
          await clusterManagementService.deleteAgent(agentId);
          message.success('智能体删除成功');
          loadData();
          return;
      }

      if (result?.success) {
        message.success(`智能体${action}操作成功`);
        loadData();
      } else {
        message.error(result?.message || `智能体${action}操作失败`);
      }
    } catch (error) {
      const detail = (error as any)?.response?.data?.detail;
      const errorMessage = detail || (error as Error)?.message || '未知错误';
      message.error(`智能体操作失败: ${errorMessage}`);
    }
  };

  const handleCreateAgent = async (values: AgentCreateRequest) => {
    try {
      await clusterManagementService.createAgent(values);
      message.success('智能体创建成功');
      setCreateAgentVisible(false);
      createAgentForm.resetFields();
      loadData();
    } catch (error) {
      message.error('智能体创建失败');
    }
  };

  const handleCreateGroup = async (values: GroupCreateRequest) => {
    try {
      await clusterManagementService.createGroup(values);
      message.success('分组创建成功');
      setCreateGroupVisible(false);
      createGroupForm.resetFields();
      loadData();
    } catch (error) {
      message.error('分组创建失败');
    }
  };

  const handleCreatePolicy = async (values: ScalingPolicyRequest) => {
    try {
      await clusterManagementService.createScalingPolicy(values);
      message.success('扩缩容策略创建成功');
      setCreatePolicyVisible(false);
      createPolicyForm.resetFields();
      loadData();
    } catch (error) {
      message.error('扩缩容策略创建失败');
    }
  };

  const handleManualScaling = async (values: ManualScalingRequest) => {
    try {
      await clusterManagementService.manualScale(selectedGroup, values);
      message.success('手动扩缩容成功');
      setManualScalingVisible(false);
      manualScalingForm.resetFields();
      loadData();
    } catch (error) {
      message.error('手动扩缩容失败');
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online':
      case 'running':
        return 'success';
      case 'offline':
      case 'stopped':
        return 'default';
      case 'error':
      case 'failed':
        return 'error';
      case 'starting':
      case 'stopping':
        return 'processing';
      default:
        return 'default';
    }
  };

  const agentColumns = [
    {
      title: 'Agent ID',
      dataIndex: 'agent_id',
      key: 'agent_id',
      width: 200,
      render: (text: string) => (
        <Text copyable={{ text }}>{(text || '').substring(0, 8)}...</Text>
      ),
    },
    {
      title: '名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={getStatusColor(status)}>{status.toUpperCase()}</Tag>
      ),
    },
    {
      title: '端点',
      dataIndex: 'endpoint',
      key: 'endpoint',
    },
    {
      title: '能力',
      dataIndex: 'capabilities',
      key: 'capabilities',
      render: (capabilities: string[]) => (
        <Space wrap>
          {capabilities?.map((cap, index) => (
            <Tag key={index} color="blue">{cap}</Tag>
          ))}
        </Space>
      ),
    },
    {
      title: 'CPU使用率',
      dataIndex: ['resource_usage', 'cpu_usage'],
      key: 'cpu_usage',
      render: (value: number) => (
        <Progress percent={Math.round(value || 0)} size="small" />
      ),
    },
    {
      title: '内存使用率',
      dataIndex: ['resource_usage', 'memory_usage'],
      key: 'memory_usage',
      render: (value: number) => (
        <Progress percent={Math.round(value || 0)} size="small" />
      ),
    },
    {
      title: '健康状态',
      dataIndex: 'is_healthy',
      key: 'is_healthy',
      render: (healthy: boolean) => (
        <Badge 
          status={healthy ? 'success' : 'error'} 
          text={healthy ? '健康' : '异常'} 
        />
      ),
    },
    {
      title: '操作',
      key: 'actions',
      width: 300,
      render: (_: any, record: AgentInfo) => (
        <Space size="small">
          <Tooltip title="查看详情">
            <Button 
              size="small" 
              icon={<InfoCircleOutlined />}
              onClick={() => {
                setSelectedAgent(record);
                setAgentDetailsVisible(true);
              }}
            />
          </Tooltip>
          <Tooltip title="启动">
            <Button 
              size="small" 
              type="primary" 
              icon={<PlayCircleOutlined />}
              disabled={record.status === 'online'}
              onClick={() => handleAgentAction(record.agent_id, 'start')}
            />
          </Tooltip>
          <Tooltip title="停止">
            <Button 
              size="small" 
              icon={<PauseCircleOutlined />}
              disabled={record.status === 'offline'}
              onClick={() => handleAgentAction(record.agent_id, 'stop')}
            />
          </Tooltip>
          <Tooltip title="重启">
            <Button 
              size="small" 
              icon={<ReloadOutlined />}
              onClick={() => handleAgentAction(record.agent_id, 'restart')}
            />
          </Tooltip>
          <Popconfirm
            title="确定要删除这个智能体吗？"
            onConfirm={() => handleAgentAction(record.agent_id, 'delete')}
          >
            <Tooltip title="删除">
              <Button 
                size="small" 
                danger 
                icon={<DeleteOutlined />}
              />
            </Tooltip>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  const groupColumns = [
    {
      title: '分组名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
    },
    {
      title: '智能体数量',
      dataIndex: 'current_agents',
      key: 'current_agents',
      render: (current: number, record: AgentGroup) => (
        <span>
          {current} / {record.max_agents || '∞'}
          {record.min_agents > 0 && ` (最少: ${record.min_agents})`}
        </span>
      ),
    },
    {
      title: '状态',
      key: 'status',
      render: (_: any, record: AgentGroup) => {
        if (record.current_agents < record.min_agents) {
          return <Tag color="warning">需要扩容</Tag>;
        }
        if (record.max_agents && record.current_agents >= record.max_agents) {
          return <Tag color="error">已满</Tag>;
        }
        return <Tag color="success">正常</Tag>;
      },
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (time: string) => new Date(time).toLocaleString(),
    },
    {
      title: '操作',
      key: 'actions',
      render: (_: any, record: AgentGroup) => (
        <Space>
          <Button 
            size="small" 
            type="primary"
            icon={<ThunderboltOutlined />}
            onClick={() => {
              setSelectedGroup(record.group_id);
              setManualScalingVisible(true);
            }}
          >
            手动扩缩容
          </Button>
        </Space>
      ),
    },
  ];

  const policyColumns = [
    {
      title: '策略名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: 'CPU目标',
      dataIndex: 'target_cpu_percent',
      key: 'target_cpu_percent',
      render: (value: number) => `${value}%`,
    },
    {
      title: '内存目标',
      dataIndex: 'target_memory_percent',
      key: 'target_memory_percent',
      render: (value: number) => `${value}%`,
    },
    {
      title: '实例范围',
      key: 'instance_range',
      render: (_: any, record: ScalingPolicy) => (
        `${record.min_instances} - ${record.max_instances}`
      ),
    },
    {
      title: '冷却时间',
      dataIndex: 'cooldown_period_seconds',
      key: 'cooldown_period_seconds',
      render: (value: number) => `${value}秒`,
    },
    {
      title: '状态',
      dataIndex: 'enabled',
      key: 'enabled',
      render: (enabled: boolean) => (
        <Tag color={enabled ? 'success' : 'default'}>
          {enabled ? '启用' : '禁用'}
        </Tag>
      ),
    },
  ];

  return (
    <Layout style={{ padding: '24px', background: '#f0f2f5' }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto', width: '100%' }}>
        <Title level={2}>智能体集群管理</Title>
        
        {/* 集群概览 */}
        {clusterStats && (
          <Card title="集群概览" style={{ marginBottom: 24 }}>
            <Row gutter={16}>
              <Col span={6}>
                <Statistic 
                  title="总智能体数" 
                  value={clusterStats.total_agents}
                  prefix={<ClusterOutlined />}
                />
              </Col>
              <Col span={6}>
                <Statistic 
                  title="在线智能体" 
                  value={clusterStats.online_agents}
                  valueStyle={{ color: '#3f8600' }}
                  prefix={<CheckCircleOutlined />}
                />
              </Col>
              <Col span={6}>
                <Statistic 
                  title="平均CPU使用率" 
                  value={clusterStats.avg_cpu_usage}
                  suffix="%"
                  prefix={<MonitorOutlined />}
                />
              </Col>
              <Col span={6}>
                <Statistic 
                  title="平均内存使用率" 
                  value={clusterStats.avg_memory_usage}
                  suffix="%"
                  prefix={<MonitorOutlined />}
                />
              </Col>
            </Row>
            
            <Divider />
            
            <Row gutter={16}>
              <Col span={6}>
                <Statistic 
                  title="错误率" 
                  value={clusterStats.error_rate}
                  suffix="%"
                  valueStyle={{ color: clusterStats.error_rate > 5 ? '#cf1322' : '#3f8600' }}
                  prefix={<WarningOutlined />}
                />
              </Col>
              <Col span={6}>
                <Statistic 
                  title="已处理任务" 
                  value={clusterStats.total_tasks_processed}
                />
              </Col>
              <Col span={6}>
                <Statistic 
                  title="容量利用率" 
                  value={Math.round((clusterStats.used_capacity / clusterStats.total_capacity) * 100)}
                  suffix="%"
                />
              </Col>
              <Col span={6}>
                <Text type="secondary">
                  最后更新: {new Date(clusterStats.last_updated).toLocaleString()}
                </Text>
              </Col>
            </Row>
          </Card>
        )}

        {/* 主要内容区域 */}
        <Card>
          <Tabs activeKey={activeTab} onChange={setActiveTab}>
            {/* 智能体管理 */}
            <TabPane tab="智能体管理" key="agents">
              <div style={{ marginBottom: 16 }}>
                <Space>
                  <Button 
                    type="primary" 
                    icon={<PlusOutlined />}
                    onClick={() => setCreateAgentVisible(true)}
                  >
                    创建智能体
                  </Button>
                  <Button icon={<ReloadOutlined />} onClick={loadData}>
                    刷新
                  </Button>
                </Space>
              </div>
              
              <Table
                columns={agentColumns}
                dataSource={agents}
                rowKey="agent_id"
                loading={loading}
                pagination={{ pageSize: 10 }}
                scroll={{ x: 1200 }}
              />
            </TabPane>

            {/* 分组管理 */}
            <TabPane tab="分组管理" key="groups">
              <div style={{ marginBottom: 16 }}>
                <Space>
                  <Button 
                    type="primary" 
                    icon={<PlusOutlined />}
                    onClick={() => setCreateGroupVisible(true)}
                  >
                    创建分组
                  </Button>
                  <Button icon={<ReloadOutlined />} onClick={loadData}>
                    刷新
                  </Button>
                </Space>
              </div>
              
              <Table
                columns={groupColumns}
                dataSource={groups}
                rowKey="group_id"
                loading={loading}
                pagination={{ pageSize: 10 }}
              />
            </TabPane>

            {/* 扩缩容策略 */}
            <TabPane tab="扩缩容策略" key="scaling">
              <div style={{ marginBottom: 16 }}>
                <Space>
                  <Button 
                    type="primary" 
                    icon={<PlusOutlined />}
                    onClick={() => setCreatePolicyVisible(true)}
                  >
                    创建策略
                  </Button>
                  <Button icon={<ReloadOutlined />} onClick={loadData}>
                    刷新
                  </Button>
                </Space>
              </div>
              
              <Table
                columns={policyColumns}
                dataSource={policies}
                rowKey="policy_id"
                loading={loading}
                pagination={{ pageSize: 10 }}
              />
            </TabPane>
          </Tabs>
        </Card>

        {/* 创建智能体模态框 */}
        <Modal
          title="创建智能体"
          open={createAgentVisible}
          onCancel={() => setCreateAgentVisible(false)}
          footer={null}
          width={600}
        >
          <Form
            form={createAgentForm}
            layout="vertical"
            onFinish={handleCreateAgent}
          >
            <Form.Item
              label="名称"
              name="name"
              rules={[{ required: true, message: '请输入智能体名称' }]}
            >
              <Input placeholder="智能体名称" />
            </Form.Item>
            
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item
                  label="主机地址"
                  name="host"
                  rules={[{ required: true, message: '请输入主机地址' }]}
                >
                  <Input placeholder="localhost" />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item
                  label="端口"
                  name="port"
                  rules={[{ required: true, message: '请输入端口号' }]}
                >
                  <InputNumber min={1} max={65535} placeholder="8080" style={{ width: '100%' }} />
                </Form.Item>
              </Col>
            </Row>
            
            <Form.Item label="能力" name="capabilities">
              <Select
                mode="tags"
                placeholder="选择或输入能力标签"
                style={{ width: '100%' }}
              >
                <Select.Option value="nlp">NLP</Select.Option>
                <Select.Option value="cv">计算机视觉</Select.Option>
                <Select.Option value="ml">机器学习</Select.Option>
                <Select.Option value="reasoning">推理</Select.Option>
              </Select>
            </Form.Item>
            
            <Form.Item label="版本" name="version" initialValue="1.0.0">
              <Input placeholder="1.0.0" />
            </Form.Item>
            
            <div style={{ textAlign: 'right' }}>
              <Space>
                <Button onClick={() => setCreateAgentVisible(false)}>取消</Button>
                <Button type="primary" htmlType="submit">创建</Button>
              </Space>
            </div>
          </Form>
        </Modal>

        {/* 创建分组模态框 */}
        <Modal
          title="创建智能体分组"
          open={createGroupVisible}
          onCancel={() => setCreateGroupVisible(false)}
          footer={null}
          width={500}
        >
          <Form
            form={createGroupForm}
            layout="vertical"
            onFinish={handleCreateGroup}
          >
            <Form.Item
              label="分组名称"
              name="name"
              rules={[{ required: true, message: '请输入分组名称' }]}
            >
              <Input placeholder="分组名称" />
            </Form.Item>
            
            <Form.Item label="描述" name="description">
              <Input.TextArea placeholder="分组描述" rows={3} />
            </Form.Item>
            
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item label="最小智能体数" name="min_agents" initialValue={0}>
                  <InputNumber min={0} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item label="最大智能体数" name="max_agents">
                  <InputNumber min={1} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
            </Row>
            
            <div style={{ textAlign: 'right' }}>
              <Space>
                <Button onClick={() => setCreateGroupVisible(false)}>取消</Button>
                <Button type="primary" htmlType="submit">创建</Button>
              </Space>
            </div>
          </Form>
        </Modal>

        {/* 创建扩缩容策略模态框 */}
        <Modal
          title="创建扩缩容策略"
          open={createPolicyVisible}
          onCancel={() => setCreatePolicyVisible(false)}
          footer={null}
          width={600}
        >
          <Form
            form={createPolicyForm}
            layout="vertical"
            onFinish={handleCreatePolicy}
          >
            <Form.Item
              label="策略名称"
              name="name"
              rules={[{ required: true, message: '请输入策略名称' }]}
            >
              <Input placeholder="策略名称" />
            </Form.Item>
            
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item label="目标CPU使用率 (%)" name="target_cpu_percent" initialValue={70}>
                  <InputNumber min={1} max={100} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item label="目标内存使用率 (%)" name="target_memory_percent" initialValue={75}>
                  <InputNumber min={1} max={100} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
            </Row>
            
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item label="最小实例数" name="min_instances" initialValue={1}>
                  <InputNumber min={1} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item label="最大实例数" name="max_instances" initialValue={10}>
                  <InputNumber min={1} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
            </Row>
            
            <Form.Item label="冷却时间 (秒)" name="cooldown_period_seconds" initialValue={180}>
              <InputNumber min={30} style={{ width: '100%' }} />
            </Form.Item>
            
            <Form.Item label="启用策略" name="enabled" valuePropName="checked" initialValue={true}>
              <Switch />
            </Form.Item>
            
            <div style={{ textAlign: 'right' }}>
              <Space>
                <Button onClick={() => setCreatePolicyVisible(false)}>取消</Button>
                <Button type="primary" htmlType="submit">创建</Button>
              </Space>
            </div>
          </Form>
        </Modal>

        {/* 手动扩缩容模态框 */}
        <Modal
          title="手动扩缩容"
          open={manualScalingVisible}
          onCancel={() => setManualScalingVisible(false)}
          footer={null}
        >
          <Form
            form={manualScalingForm}
            layout="vertical"
            onFinish={handleManualScaling}
          >
            <Form.Item
              label="目标实例数"
              name="target_instances"
              rules={[{ required: true, message: '请输入目标实例数' }]}
            >
              <InputNumber min={0} style={{ width: '100%' }} />
            </Form.Item>
            
            <Form.Item label="扩缩容原因" name="reason">
              <Input.TextArea placeholder="请输入扩缩容原因" rows={3} />
            </Form.Item>
            
            <div style={{ textAlign: 'right' }}>
              <Space>
                <Button onClick={() => setManualScalingVisible(false)}>取消</Button>
                <Button type="primary" htmlType="submit">执行</Button>
              </Space>
            </div>
          </Form>
        </Modal>

        {/* 智能体详情模态框 */}
        <Modal
          title="智能体详情"
          open={agentDetailsVisible}
          onCancel={() => setAgentDetailsVisible(false)}
          footer={[
            <Button key="close" onClick={() => setAgentDetailsVisible(false)}>
              关闭
            </Button>
          ]}
          width={800}
        >
          {selectedAgent && (
            <Descriptions bordered column={2}>
              <Descriptions.Item label="Agent ID">{selectedAgent.agent_id}</Descriptions.Item>
              <Descriptions.Item label="名称">{selectedAgent.name}</Descriptions.Item>
              <Descriptions.Item label="状态">
                <Tag color={getStatusColor(selectedAgent.status)}>
                  {selectedAgent.status?.toUpperCase()}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="端点">{selectedAgent.endpoint}</Descriptions.Item>
              <Descriptions.Item label="版本">{selectedAgent.version}</Descriptions.Item>
              <Descriptions.Item label="运行时间">{selectedAgent.uptime}秒</Descriptions.Item>
              <Descriptions.Item label="CPU使用率">
                {selectedAgent.resource_usage?.cpu_usage.toFixed(1)}%
              </Descriptions.Item>
              <Descriptions.Item label="内存使用率">
                {selectedAgent.resource_usage?.memory_usage.toFixed(1)}%
              </Descriptions.Item>
              <Descriptions.Item label="活跃任务数">
                {selectedAgent.resource_usage?.active_tasks}
              </Descriptions.Item>
              <Descriptions.Item label="错误率">
                {selectedAgent.resource_usage?.error_rate.toFixed(2)}%
              </Descriptions.Item>
              <Descriptions.Item label="能力" span={2}>
                <Space wrap>
                  {selectedAgent.capabilities?.map((cap, index) => (
                    <Tag key={index} color="blue">{cap}</Tag>
                  ))}
                </Space>
              </Descriptions.Item>
            </Descriptions>
          )}
        </Modal>
      </div>
    </Layout>
  );
};

export default AgentClusterManagementPage;
