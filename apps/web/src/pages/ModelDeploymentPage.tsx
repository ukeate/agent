import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Table, 
  Button, 
  Modal, 
  Form, 
  Select, 
  Input, 
  Tabs, 
  Progress, 
  Badge, 
  Space,
  Alert,
  Descriptions,
  Row,
  Col,
  Statistic,
  Tooltip,
  Popconfirm,
  message,
  Timeline,
  Tag
} from 'antd';
import { 
  PlayCircleOutlined, 
  PauseCircleOutlined, 
  StopOutlined,
  ReloadOutlined,
  DeleteOutlined,
  SettingOutlined,
  MonitorOutlined,
  CloudServerOutlined,
  RocketOutlined,
  ApiOutlined
} from '@ant-design/icons';

const { TabPane } = Tabs;
const { Option } = Select;
const { TextArea } = Input;

interface Deployment {
  id: string;
  name: string;
  model_id: string;
  model_name: string;
  platform: 'docker' | 'kubernetes' | 'edge';
  status: 'pending' | 'running' | 'stopped' | 'failed' | 'updating';
  endpoint: string;
  replicas: number;
  cpu_usage: number;
  memory_usage: number;
  requests_per_minute: number;
  created_at: string;
  updated_at: string;
  configuration: any;
  health_status: 'healthy' | 'unhealthy' | 'warning';
}

interface DeploymentConfig {
  platform: 'docker' | 'kubernetes' | 'edge';
  replicas: number;
  cpu_limit: string;
  memory_limit: string;
  gpu_enabled: boolean;
  auto_scaling: boolean;
  min_replicas?: number;
  max_replicas?: number;
  environment: Record<string, string>;
  health_check: {
    enabled: boolean;
    path: string;
    interval: number;
  };
}

const ModelDeploymentPage: React.FC = () => {
  const [deployments, setDeployments] = useState<Deployment[]>([]);
  const [models, setModels] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [deployModalVisible, setDeployModalVisible] = useState(false);
  const [configModalVisible, setConfigModalVisible] = useState(false);
  const [selectedDeployment, setSelectedDeployment] = useState<Deployment | null>(null);
  const [form] = Form.useForm();
  const [configForm] = Form.useForm();

  const deploymentStats = {
    total: deployments.length,
    running: deployments.filter(d => d.status === 'running').length,
    failed: deployments.filter(d => d.status === 'failed').length,
    pending: deployments.filter(d => d.status === 'pending').length
  };

  useEffect(() => {
    fetchDeployments();
    fetchModels();
  }, []);

  const fetchDeployments = async () => {
    setLoading(true);
    try {
      // 模拟API调用
      const mockDeployments: Deployment[] = [
        {
          id: '1',
          name: 'text-classifier-prod',
          model_id: 'model_001',
          model_name: 'BERT文本分类器',
          platform: 'kubernetes',
          status: 'running',
          endpoint: 'https://api.example.com/v1/classify',
          replicas: 3,
          cpu_usage: 45,
          memory_usage: 60,
          requests_per_minute: 1250,
          created_at: '2024-01-15T10:30:00Z',
          updated_at: '2024-01-20T15:45:00Z',
          configuration: {},
          health_status: 'healthy'
        },
        {
          id: '2',
          name: 'sentiment-analyzer-staging',
          model_id: 'model_002',
          model_name: '情感分析模型',
          platform: 'docker',
          status: 'running',
          endpoint: 'https://staging.example.com/v1/sentiment',
          replicas: 1,
          cpu_usage: 30,
          memory_usage: 40,
          requests_per_minute: 320,
          created_at: '2024-01-18T14:20:00Z',
          updated_at: '2024-01-18T14:20:00Z',
          configuration: {},
          health_status: 'healthy'
        },
        {
          id: '3',
          name: 'edge-detector-device',
          model_id: 'model_003',
          model_name: '边缘检测模型',
          platform: 'edge',
          status: 'failed',
          endpoint: 'device://edge-001/detect',
          replicas: 1,
          cpu_usage: 0,
          memory_usage: 0,
          requests_per_minute: 0,
          created_at: '2024-01-22T09:15:00Z',
          updated_at: '2024-01-22T12:30:00Z',
          configuration: {},
          health_status: 'unhealthy'
        }
      ];
      setDeployments(mockDeployments);
    } catch (error) {
      message.error('加载部署列表失败');
    } finally {
      setLoading(false);
    }
  };

  const fetchModels = async () => {
    try {
      const mockModels = [
        { id: 'model_001', name: 'BERT文本分类器', version: '1.2.0' },
        { id: 'model_002', name: '情感分析模型', version: '2.1.0' },
        { id: 'model_003', name: '边缘检测模型', version: '1.0.0' },
        { id: 'model_004', name: 'GPT对话模型', version: '3.0.0' }
      ];
      setModels(mockModels);
    } catch (error) {
      message.error('加载模型列表失败');
    }
  };

  const handleDeploy = async (values: any) => {
    try {
      setLoading(true);
      // 模拟部署API调用
      const newDeployment: Deployment = {
        id: Date.now().toString(),
        name: values.name,
        model_id: values.model_id,
        model_name: models.find(m => m.id === values.model_id)?.name || '',
        platform: values.platform,
        status: 'pending',
        endpoint: '',
        replicas: values.replicas || 1,
        cpu_usage: 0,
        memory_usage: 0,
        requests_per_minute: 0,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        configuration: values,
        health_status: 'healthy'
      };
      
      setDeployments([...deployments, newDeployment]);
      setDeployModalVisible(false);
      form.resetFields();
      message.success('部署任务已启动');
      
      // 模拟部署过程
      setTimeout(() => {
        setDeployments(prev => prev.map(d => 
          d.id === newDeployment.id 
            ? { ...d, status: 'running', endpoint: `https://api.example.com/v1/${values.name}` }
            : d
        ));
        message.success('模型部署成功');
      }, 3000);
    } catch (error) {
      message.error('部署失败');
    } finally {
      setLoading(false);
    }
  };

  const handleStatusChange = async (deploymentId: string, action: 'start' | 'stop' | 'restart') => {
    try {
      setLoading(true);
      // 模拟状态变更API调用
      let newStatus: Deployment['status'] = 'pending';
      switch (action) {
        case 'start':
          newStatus = 'running';
          break;
        case 'stop':
          newStatus = 'stopped';
          break;
        case 'restart':
          newStatus = 'running';
          break;
      }
      
      setDeployments(prev => prev.map(d => 
        d.id === deploymentId 
          ? { ...d, status: newStatus, updated_at: new Date().toISOString() }
          : d
      ));
      message.success(`部署${action === 'start' ? '启动' : action === 'stop' ? '停止' : '重启'}成功`);
    } catch (error) {
      message.error('操作失败');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (deploymentId: string) => {
    try {
      setDeployments(prev => prev.filter(d => d.id !== deploymentId));
      message.success('部署已删除');
    } catch (error) {
      message.error('删除失败');
    }
  };

  const getStatusColor = (status: Deployment['status']) => {
    const colors = {
      pending: 'orange',
      running: 'green',
      stopped: 'default',
      failed: 'red',
      updating: 'blue'
    };
    return colors[status];
  };

  const getHealthColor = (health: Deployment['health_status']) => {
    const colors = {
      healthy: 'green',
      unhealthy: 'red',
      warning: 'orange'
    };
    return colors[health];
  };

  const getPlatformIcon = (platform: string) => {
    const icons = {
      docker: <CloudServerOutlined />,
      kubernetes: <ApiOutlined />,
      edge: <MonitorOutlined />
    };
    return icons[platform as keyof typeof icons] || <CloudServerOutlined />;
  };

  const columns = [
    {
      title: '部署名称',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: Deployment) => (
        <Space>
          {getPlatformIcon(record.platform)}
          <span>{text}</span>
        </Space>
      )
    },
    {
      title: '模型',
      dataIndex: 'model_name',
      key: 'model_name',
    },
    {
      title: '平台',
      dataIndex: 'platform',
      key: 'platform',
      render: (platform: string) => (
        <Tag color={platform === 'kubernetes' ? 'blue' : platform === 'docker' ? 'cyan' : 'purple'}>
          {platform.toUpperCase()}
        </Tag>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: Deployment['status']) => (
        <Badge status={getStatusColor(status) as any} text={
          status === 'pending' ? '部署中' :
          status === 'running' ? '运行中' :
          status === 'stopped' ? '已停止' :
          status === 'failed' ? '失败' : '更新中'
        } />
      )
    },
    {
      title: '健康状态',
      dataIndex: 'health_status',
      key: 'health_status',
      render: (health: Deployment['health_status']) => (
        <Badge status={getHealthColor(health) as any} text={
          health === 'healthy' ? '健康' :
          health === 'unhealthy' ? '不健康' : '警告'
        } />
      )
    },
    {
      title: '副本数',
      dataIndex: 'replicas',
      key: 'replicas',
    },
    {
      title: 'CPU使用率',
      dataIndex: 'cpu_usage',
      key: 'cpu_usage',
      render: (usage: number) => (
        <Progress 
          percent={usage} 
          size="small" 
          strokeColor={usage > 80 ? '#ff4d4f' : usage > 60 ? '#faad14' : '#52c41a'}
        />
      )
    },
    {
      title: '请求量/分钟',
      dataIndex: 'requests_per_minute',
      key: 'requests_per_minute',
      render: (rpm: number) => rpm.toLocaleString()
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record: Deployment) => (
        <Space>
          {record.status === 'running' ? (
            <Tooltip title="停止">
              <Button 
                icon={<PauseCircleOutlined />} 
                size="small"
                onClick={() => handleStatusChange(record.id, 'stop')}
              />
            </Tooltip>
          ) : (
            <Tooltip title="启动">
              <Button 
                icon={<PlayCircleOutlined />} 
                size="small" 
                type="primary"
                onClick={() => handleStatusChange(record.id, 'start')}
              />
            </Tooltip>
          )}
          <Tooltip title="重启">
            <Button 
              icon={<ReloadOutlined />} 
              size="small"
              onClick={() => handleStatusChange(record.id, 'restart')}
            />
          </Tooltip>
          <Tooltip title="配置">
            <Button 
              icon={<SettingOutlined />} 
              size="small"
              onClick={() => {
                setSelectedDeployment(record);
                setConfigModalVisible(true);
              }}
            />
          </Tooltip>
          <Popconfirm
            title="确定要删除这个部署吗？"
            onConfirm={() => handleDelete(record.id)}
            okText="确定"
            cancelText="取消"
          >
            <Tooltip title="删除">
              <Button 
                icon={<DeleteOutlined />} 
                size="small" 
                danger
              />
            </Tooltip>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div style={{ padding: '24px' }}>
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="总部署数"
              value={deploymentStats.total}
              prefix={<RocketOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="运行中"
              value={deploymentStats.running}
              valueStyle={{ color: '#3f8600' }}
              prefix={<PlayCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="失败"
              value={deploymentStats.failed}
              valueStyle={{ color: '#cf1322' }}
              prefix={<StopOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="待部署"
              value={deploymentStats.pending}
              valueStyle={{ color: '#fa8c16' }}
              prefix={<MonitorOutlined />}
            />
          </Card>
        </Col>
      </Row>

      <Card
        title="模型部署管理"
        extra={
          <Button 
            type="primary" 
            icon={<RocketOutlined />}
            onClick={() => setDeployModalVisible(true)}
          >
            新建部署
          </Button>
        }
      >
        <Table
          columns={columns}
          dataSource={deployments}
          rowKey="id"
          loading={loading}
          expandable={{
            expandedRowRender: (record) => (
              <Descriptions title="部署详情" bordered column={2}>
                <Descriptions.Item label="端点">{record.endpoint || '未分配'}</Descriptions.Item>
                <Descriptions.Item label="创建时间">{new Date(record.created_at).toLocaleString()}</Descriptions.Item>
                <Descriptions.Item label="更新时间">{new Date(record.updated_at).toLocaleString()}</Descriptions.Item>
                <Descriptions.Item label="内存使用率">{record.memory_usage}%</Descriptions.Item>
              </Descriptions>
            ),
          }}
        />
      </Card>

      <Modal
        title="新建部署"
        visible={deployModalVisible}
        onCancel={() => setDeployModalVisible(false)}
        footer={null}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleDeploy}
        >
          <Form.Item
            name="name"
            label="部署名称"
            rules={[{ required: true, message: '请输入部署名称' }]}
          >
            <Input placeholder="请输入部署名称" />
          </Form.Item>

          <Form.Item
            name="model_id"
            label="选择模型"
            rules={[{ required: true, message: '请选择模型' }]}
          >
            <Select placeholder="请选择要部署的模型">
              {models.map(model => (
                <Option key={model.id} value={model.id}>
                  {model.name} (v{model.version})
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item
            name="platform"
            label="部署平台"
            rules={[{ required: true, message: '请选择部署平台' }]}
          >
            <Select placeholder="请选择部署平台">
              <Option value="docker">Docker</Option>
              <Option value="kubernetes">Kubernetes</Option>
              <Option value="edge">边缘设备</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="replicas"
            label="副本数量"
            initialValue={1}
          >
            <Select>
              <Option value={1}>1个副本</Option>
              <Option value={2}>2个副本</Option>
              <Option value={3}>3个副本</Option>
              <Option value={5}>5个副本</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="cpu_limit"
            label="CPU限制"
            initialValue="1000m"
          >
            <Input placeholder="例如: 1000m, 2" />
          </Form.Item>

          <Form.Item
            name="memory_limit"
            label="内存限制"
            initialValue="2Gi"
          >
            <Input placeholder="例如: 2Gi, 4096Mi" />
          </Form.Item>

          <Form.Item
            name="gpu_enabled"
            label="启用GPU"
            valuePropName="checked"
            initialValue={false}
          >
            <Select>
              <Option value={true}>启用</Option>
              <Option value={false}>禁用</Option>
            </Select>
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={loading}>
                开始部署
              </Button>
              <Button onClick={() => setDeployModalVisible(false)}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title="部署配置"
        visible={configModalVisible}
        onCancel={() => setConfigModalVisible(false)}
        footer={null}
        width={800}
      >
        {selectedDeployment && (
          <Tabs defaultActiveKey="basic">
            <TabPane tab="基础配置" key="basic">
              <Descriptions bordered column={2}>
                <Descriptions.Item label="部署名称">{selectedDeployment.name}</Descriptions.Item>
                <Descriptions.Item label="模型">{selectedDeployment.model_name}</Descriptions.Item>
                <Descriptions.Item label="平台">{selectedDeployment.platform}</Descriptions.Item>
                <Descriptions.Item label="副本数">{selectedDeployment.replicas}</Descriptions.Item>
                <Descriptions.Item label="状态">{selectedDeployment.status}</Descriptions.Item>
                <Descriptions.Item label="健康状态">{selectedDeployment.health_status}</Descriptions.Item>
              </Descriptions>
            </TabPane>
            <TabPane tab="资源监控" key="monitoring">
              <Row gutter={16}>
                <Col span={12}>
                  <Card title="CPU使用率">
                    <Progress 
                      percent={selectedDeployment.cpu_usage} 
                      strokeColor="#52c41a"
                    />
                  </Card>
                </Col>
                <Col span={12}>
                  <Card title="内存使用率">
                    <Progress 
                      percent={selectedDeployment.memory_usage} 
                      strokeColor="#1890ff"
                    />
                  </Card>
                </Col>
              </Row>
              <Card title="请求统计" style={{ marginTop: 16 }}>
                <Statistic
                  title="每分钟请求数"
                  value={selectedDeployment.requests_per_minute}
                  suffix="RPM"
                />
              </Card>
            </TabPane>
            <TabPane tab="部署历史" key="history">
              <Timeline>
                <Timeline.Item color="green">
                  <p>部署成功 {new Date(selectedDeployment.created_at).toLocaleString()}</p>
                  <p>初始版本部署完成</p>
                </Timeline.Item>
                <Timeline.Item color="blue">
                  <p>配置更新 {new Date(selectedDeployment.updated_at).toLocaleString()}</p>
                  <p>更新资源配置</p>
                </Timeline.Item>
              </Timeline>
            </TabPane>
          </Tabs>
        )}
      </Modal>
    </div>
  );
};

export default ModelDeploymentPage;