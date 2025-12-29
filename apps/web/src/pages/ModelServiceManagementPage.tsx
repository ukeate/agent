import React, { useState, useEffect } from 'react';
import {
import { logger } from '../utils/logger'
  Card,
  Tabs,
  Table,
  Button,
  Space,
  Tag,
  Upload,
  Modal,
  Form,
  Input,
  Select,
  InputNumber,
  Switch,
  message,
  Divider,
  Statistic,
  Row,
  Col,
  Progress,
  Alert,
  Spin,
  Descriptions,
  List,
  Avatar
} from 'antd';
import {
  PlusOutlined,
  UploadOutlined,
  DeleteOutlined,
  EyeOutlined,
  PlayCircleOutlined,
  StopOutlined,
  RocketOutlined,
  ExperimentOutlined,
  MonitorOutlined,
  DatabaseOutlined,
  ApiOutlined,
  CloudOutlined,
  LoadingOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  BarChartOutlined,
  LineChartOutlined,
  BulbOutlined,
  SettingOutlined
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import type { UploadProps } from 'antd';
import { Line, Column } from '@ant-design/charts';
import {
  modelService,
  type Model,
  type Deployment,
  type LearningSession,
  type ABTest,
  type MonitoringOverview,
  type Alert as ServiceAlert,
  type ModelStatistics,
  type PredictionRequest,
  type DeploymentConfig
} from '../services/modelService';

const { TabPane } = Tabs;
const { Option } = Select;
const { TextArea } = Input;

const ModelServiceManagementPage: React.FC = () => {
  // ==================== 状态管理 ====================
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('models');
  
  // 数据状态
  const [models, setModels] = useState<Model[]>([]);
  const [deployments, setDeployments] = useState<Deployment[]>([]);
  const [learningSessions, setLearningSessions] = useState<LearningSession[]>([]);
  const [abTests, setABTests] = useState<ABTest[]>([]);
  const [monitoringOverview, setMonitoringOverview] = useState<MonitoringOverview | null>(null);
  const [alerts, setAlerts] = useState<ServiceAlert[]>([]);
  const [statistics, setStatistics] = useState<ModelStatistics | null>(null);
  const [loadedModels, setLoadedModels] = useState<any[]>([]);
  
  // 模态框状态
  const [uploadModalVisible, setUploadModalVisible] = useState(false);
  const [hubModalVisible, setHubModalVisible] = useState(false);
  const [deployModalVisible, setDeployModalVisible] = useState(false);
  const [predictModalVisible, setPredictModalVisible] = useState(false);
  const [learningModalVisible, setLearningModalVisible] = useState(false);
  const [abTestModalVisible, setABTestModalVisible] = useState(false);
  const [modelDetailModalVisible, setModelDetailModalVisible] = useState(false);
  
  // 选中项
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);
  const [selectedDeployment, setSelectedDeployment] = useState<Deployment | null>(null);
  
  // 表单
  const [uploadForm] = Form.useForm();
  const [hubForm] = Form.useForm();
  const [deployForm] = Form.useForm();
  const [predictForm] = Form.useForm();
  const [learningForm] = Form.useForm();
  const [abTestForm] = Form.useForm();
  
  // ==================== 数据获取 ====================
  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    setLoading(true);
    try {
      await Promise.all([
        loadModels(),
        loadDeployments(),
        loadLearningSessions(),
        loadABTests(),
        loadMonitoringData(),
        loadStatistics(),
        loadLoadedModels()
      ]);
    } catch (error) {
      message.error('加载数据失败');
      logger.error('加载数据失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadModels = async () => {
    try {
      const data = await modelService.getModels();
      setModels(data);
    } catch (error) {
      logger.error('加载模型列表失败:', error);
    }
  };

  const loadDeployments = async () => {
    try {
      const data = await modelService.listDeployments();
      setDeployments(data);
    } catch (error) {
      logger.error('加载部署列表失败:', error);
    }
  };

  const loadLearningSessions = async () => {
    try {
      const data = await modelService.getLearningSessions();
      setLearningSessions(data);
    } catch (error) {
      logger.error('加载学习会话失败:', error);
    }
  };

  const loadABTests = async () => {
    try {
      const data = await modelService.listABTests();
      setABTests(data);
    } catch (error) {
      logger.error('加载AB测试失败:', error);
    }
  };

  const loadMonitoringData = async () => {
    try {
      const [overview, alertsData] = await Promise.all([
        modelService.getMonitoringOverview(),
        modelService.getAlerts()
      ]);
      setMonitoringOverview(overview);
      setAlerts(alertsData);
    } catch (error) {
      logger.error('加载监控数据失败:', error);
    }
  };

  const loadStatistics = async () => {
    try {
      const data = await modelService.getModelStatistics();
      setStatistics(data);
    } catch (error) {
      logger.error('加载统计信息失败:', error);
    }
  };

  const loadLoadedModels = async () => {
    try {
      const data = await modelService.getLoadedModels();
      setLoadedModels(data);
    } catch (error) {
      logger.error('加载已加载模型失败:', error);
    }
  };

  // ==================== 模型管理操作 ====================
  const handleUploadModel = async (values: any) => {
    try {
      const formData = new FormData();
      if (values.model_file?.file) {
        formData.append('model_file', values.model_file.file.originFileObj);
        formData.append('name', values.name);
        formData.append('version', values.version);
        formData.append('type', values.type);
        if (values.description) {
          formData.append('description', values.description);
        }
        
        await modelService.uploadModel(formData);
        message.success('模型上传成功');
        setUploadModalVisible(false);
        uploadForm.resetFields();
        loadModels();
      }
    } catch (error) {
      message.error('模型上传失败');
      logger.error('上传模型失败:', error);
    }
  };

  const handleRegisterFromHub = async (values: any) => {
    try {
      await modelService.registerModelFromHub(values);
      message.success('模型注册成功');
      setHubModalVisible(false);
      hubForm.resetFields();
      loadModels();
    } catch (error) {
      message.error('模型注册失败');
      logger.error('注册模型失败:', error);
    }
  };

  const handleDeleteModel = async (modelName: string, version: string) => {
    try {
      await modelService.deleteModelVersion(modelName, version);
      message.success('模型删除成功');
      loadModels();
    } catch (error) {
      message.error('模型删除失败');
      logger.error('删除模型失败:', error);
    }
  };

  const handleValidateModel = async (modelName: string, version: string) => {
    try {
      const result = await modelService.validateModel(modelName, version);
      Modal.info({
        title: '模型验证结果',
        content: (
          <div>
            <p><strong>验证结果:</strong> {result.valid ? '通过' : '失败'}</p>
            <p><strong>验证分数:</strong> {result.validation_score}</p>
            {result.issues.length > 0 && (
              <div>
                <p><strong>问题:</strong></p>
                <ul>
                  {result.issues.map((issue, index) => (
                    <li key={index}>{issue}</li>
                  ))}
                </ul>
              </div>
            )}
            {result.recommendations.length > 0 && (
              <div>
                <p><strong>建议:</strong></p>
                <ul>
                  {result.recommendations.map((rec, index) => (
                    <li key={index}>{rec}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ),
      });
    } catch (error) {
      message.error('模型验证失败');
      logger.error('验证模型失败:', error);
    }
  };

  const handleLoadModel = async (modelName: string, version?: string) => {
    try {
      await modelService.loadModel(modelName, version);
      message.success('模型加载中...');
      loadLoadedModels();
    } catch (error) {
      message.error('模型加载失败');
      logger.error('加载模型失败:', error);
    }
  };

  const handleUnloadModel = async (modelName: string) => {
    try {
      await modelService.unloadModel(modelName);
      message.success('模型卸载成功');
      loadLoadedModels();
    } catch (error) {
      message.error('模型卸载失败');
      logger.error('卸载模型失败:', error);
    }
  };

  // ==================== 部署管理操作 ====================
  const handleDeployModel = async (values: DeploymentConfig & { environment_vars?: string }) => {
    try {
      let envVars: Record<string, string> | undefined = undefined;
      if (typeof values.environment_vars === 'string' && values.environment_vars.trim()) {
        try {
          envVars = JSON.parse(values.environment_vars);
        } catch (error) {
          message.error('环境变量必须是合法的JSON对象');
          return;
        }
      }
      await modelService.deployModel({
        ...values,
        model_version: values.model_version || 'latest',
        environment_vars: envVars,
      });
      message.success('模型部署中...');
      setDeployModalVisible(false);
      deployForm.resetFields();
      loadDeployments();
    } catch (error) {
      message.error('模型部署失败');
      logger.error('部署模型失败:', error);
    }
  };

  const handleDeleteDeployment = async (deploymentId: string) => {
    try {
      await modelService.deleteDeployment(deploymentId);
      message.success('部署删除成功');
      loadDeployments();
    } catch (error) {
      message.error('部署删除失败');
      logger.error('删除部署失败:', error);
    }
  };

  // ==================== 预测操作 ====================
  const handlePredict = async (values: PredictionRequest) => {
    try {
      const result = await modelService.predict(values);
      message.success(`预测完成，置信度: ${result.confidence.toFixed(2)}`);
      Modal.info({
        title: '预测结果',
        content: (
          <div>
            <p><strong>预测结果:</strong> {JSON.stringify(result.prediction, null, 2)}</p>
            <p><strong>置信度:</strong> {result.confidence}</p>
            <p><strong>处理时间:</strong> {result.processing_time_ms}ms</p>
            <p><strong>模型:</strong> {result.model_info.name} v{result.model_info.version}</p>
          </div>
        ),
      });
    } catch (error) {
      message.error('预测失败');
      logger.error('预测失败:', error);
    }
  };

  // ==================== 学习会话操作 ====================
  const handleStartLearning = async (values: any) => {
    try {
      await modelService.startLearningSession(values.model_name, {
        dataset_id: values.dataset_id,
        learning_rate: values.learning_rate,
        batch_size: values.batch_size,
        epochs: values.epochs
      });
      message.success('学习会话开始');
      setLearningModalVisible(false);
      learningForm.resetFields();
      loadLearningSessions();
    } catch (error) {
      message.error('启动学习会话失败');
      logger.error('启动学习会话失败:', error);
    }
  };

  // ==================== AB测试操作 ====================
  const handleCreateABTest = async (values: any) => {
    try {
      await modelService.createABTest(values);
      message.success('AB测试创建成功');
      setABTestModalVisible(false);
      abTestForm.resetFields();
      loadABTests();
    } catch (error) {
      message.error('创建AB测试失败');
      logger.error('创建AB测试失败:', error);
    }
  };

  // ==================== 表格列定义 ====================
  const modelColumns: ColumnsType<Model> = [
    {
      title: '名称',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: Model) => (
        <Button type="link" onClick={() => {
          setSelectedModel(record);
          setModelDetailModalVisible(true);
        }}>
          {text}
        </Button>
      ),
    },
    {
      title: '版本',
      dataIndex: 'version',
      key: 'version',
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => <Tag color="blue">{type}</Tag>,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const statusConfig = {
          active: { color: 'green', text: '活跃' },
          inactive: { color: 'red', text: '非活跃' },
          training: { color: 'orange', text: '训练中' },
          deployed: { color: 'blue', text: '已部署' },
        };
        const config = statusConfig[status as keyof typeof statusConfig] || { color: 'default', text: status };
        return <Tag color={config.color}>{config.text}</Tag>;
      },
    },
    {
      title: '大小 (MB)',
      dataIndex: 'size_mb',
      key: 'size_mb',
      render: (size: number) => size.toFixed(2),
    },
    {
      title: '准确率',
      dataIndex: 'accuracy',
      key: 'accuracy',
      render: (accuracy?: number) => accuracy ? `${(accuracy * 100).toFixed(2)}%` : '-',
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record: Model) => (
        <Space>
          <Button
            size="small"
            icon={<EyeOutlined />}
            onClick={() => handleValidateModel(record.name, record.version)}
          >
            验证
          </Button>
          <Button
            size="small"
            icon={<PlayCircleOutlined />}
            onClick={() => handleLoadModel(record.name, record.version)}
          >
            加载
          </Button>
          <Button
            size="small"
            icon={<RocketOutlined />}
            onClick={() => {
              setSelectedModel(record);
              setDeployModalVisible(true);
            }}
          >
            部署
          </Button>
          <Button
            size="small"
            danger
            icon={<DeleteOutlined />}
            onClick={() => {
              Modal.confirm({
                title: '确认删除',
                content: `确定要删除模型 ${record.name} v${record.version} 吗？`,
                onOk: () => handleDeleteModel(record.name, record.version),
              });
            }}
          >
            删除
          </Button>
        </Space>
      ),
    },
  ];

  const deploymentColumns: ColumnsType<Deployment> = [
    {
      title: '部署ID',
      dataIndex: 'deployment_id',
      key: 'deployment_id',
      ellipsis: true,
    },
    {
      title: '模型',
      key: 'model',
      render: (record: Deployment) => `${record.model_name} v${record.model_version}`,
    },
    {
      title: '类型',
      dataIndex: 'deployment_type',
      key: 'deployment_type',
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const statusConfig: Record<string, { color: string; text: string }> = {
          pending: { color: 'orange', text: '等待中' },
          building: { color: 'processing', text: '构建中' },
          deploying: { color: 'processing', text: '部署中' },
          deployed: { color: 'green', text: '已部署' },
          failed: { color: 'red', text: '失败' },
          stopped: { color: 'default', text: '已停止' },
        };
        const config = statusConfig[status] || { color: 'default', text: status || '-' };
        return <Tag color={config.color}>{config.text}</Tag>;
      },
    },
    {
      title: '访问地址',
      dataIndex: 'endpoint_url',
      key: 'endpoint_url',
      render: (value: string) => value || '-',
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (value: string) => (value ? new Date(value).toLocaleString() : '-'),
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record: Deployment) => (
        <Space>
          <Button
            size="small"
            icon={<EyeOutlined />}
            onClick={() => {
              setSelectedDeployment(record);
            }}
          >
            详情
          </Button>
          <Button
            size="small"
            danger
            icon={<DeleteOutlined />}
            onClick={() => {
              Modal.confirm({
                title: '确认删除',
                content: `确定要删除部署 ${record.deployment_id} 吗？`,
                onOk: () => handleDeleteDeployment(record.deployment_id),
              });
            }}
          >
            删除
          </Button>
        </Space>
      ),
    },
  ];

  const learningSessionColumns: ColumnsType<LearningSession> = [
    {
      title: '会话ID',
      dataIndex: 'session_id',
      key: 'session_id',
      ellipsis: true,
    },
    {
      title: '模型名称',
      dataIndex: 'model_name',
      key: 'model_name',
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const statusConfig = {
          active: { color: 'green', text: '活跃' },
          paused: { color: 'orange', text: '已暂停' },
          completed: { color: 'blue', text: '已完成' },
          failed: { color: 'red', text: '失败' },
        };
        const config = statusConfig[status as keyof typeof statusConfig] || { color: 'default', text: status };
        return <Tag color={config.color}>{config.text}</Tag>;
      },
    },
    {
      title: '反馈数量',
      dataIndex: 'feedback_count',
      key: 'feedback_count',
    },
    {
      title: '当前准确率',
      dataIndex: 'performance_metrics',
      key: 'current_accuracy',
      render: (_: Record<string, number>, record: LearningSession) => {
        const accuracy = record.performance_metrics?.accuracy;
        return typeof accuracy === 'number' ? `${(accuracy * 100).toFixed(2)}%` : '-';
      },
    },
    {
      title: '开始时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (time: string) => (time ? new Date(time).toLocaleString() : '-'),
    },
    {
      title: '完成时间',
      dataIndex: 'updated_at',
      key: 'updated_at',
      render: (time: string, record: LearningSession) => (
        record.status === 'completed' && time ? new Date(time).toLocaleString() : '-'
      ),
    },
  ];

  const abTestColumns: ColumnsType<ABTest> = [
    {
      title: '测试ID',
      dataIndex: 'test_id',
      key: 'test_id',
      ellipsis: true,
    },
    {
      title: '测试名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '对照模型',
      dataIndex: 'control_model',
      key: 'control_model',
    },
    {
      title: '实验模型',
      dataIndex: 'treatment_models',
      key: 'treatment_models',
      render: (models: string[]) => (Array.isArray(models) ? models.join(', ') : '-'),
    },
    {
      title: '流量分配',
      dataIndex: 'traffic_split',
      key: 'traffic_split',
      render: (split: Record<string, number>) => {
        if (!split || typeof split !== 'object') return '-';
        return Object.entries(split)
          .map(([name, ratio]) => `${name}:${Math.round(Number(ratio) * 100)}%`)
          .join(' ');
      },
    },
    {
      title: '用户数',
      dataIndex: 'total_users',
      key: 'total_users',
      render: (value: number) => Number(value || 0).toLocaleString(),
    },
    {
      title: '样本数',
      dataIndex: 'sample_counts',
      key: 'sample_counts',
      render: (counts: Record<string, number>) => {
        if (!counts || typeof counts !== 'object') return '-';
        return Object.entries(counts)
          .map(([name, count]) => `${name}:${count}`)
          .join(' ');
      },
    },
  ];

  // ==================== 渲染函数 ====================
  const renderOverviewStats = () => {
    if (!statistics) return null;

    return (
      <Row gutter={[16, 16]}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="总模型数"
              value={statistics.total_models}
              prefix={<DatabaseOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="活跃部署"
              value={statistics.active_deployments}
              prefix={<CloudOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="预测请求数"
              value={statistics.total_predictions}
              prefix={<ApiOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="平均模型大小"
              value={statistics.average_model_size_mb}
              suffix="MB"
              prefix={<BarChartOutlined />}
            />
          </Card>
        </Col>
      </Row>
    );
  };

  const renderMonitoringDashboard = () => {
    if (!monitoringOverview) return null;

    return (
      <div>
        <Row gutter={[16, 16]}>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="总请求数"
                value={monitoringOverview.total_requests}
                prefix={<ApiOutlined />}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="平均延迟"
                value={monitoringOverview.average_latency}
                suffix="ms"
                prefix={<LineChartOutlined />}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="错误率"
                value={(monitoringOverview.error_rate * 100).toFixed(2)}
                suffix="%"
                prefix={<ExclamationCircleOutlined />}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="CPU使用率"
                value={monitoringOverview.resource_utilization.cpu_percent.toFixed(1)}
                suffix="%"
                prefix={<MonitorOutlined />}
              />
            </Card>
          </Col>
        </Row>

        <Divider />

        <Row gutter={[16, 16]}>
          <Col xs={24} lg={12}>
            <Card title="系统警报" size="small">
              <List
                dataSource={alerts.slice(0, 5)}
                renderItem={(alert) => (
                  <List.Item>
                    <List.Item.Meta
                      avatar={
                        <Avatar
                          icon={alert.severity === 'critical' ? <ExclamationCircleOutlined /> : <BulbOutlined />}
                          style={{
                            backgroundColor: alert.severity === 'critical' ? '#ff4d4f' : 
                                             alert.severity === 'high' ? '#faad14' : '#52c41a'
                          }}
                        />
                      }
                      title={
                        <Space>
                          <Tag color={alert.severity === 'critical' ? 'red' : 
                                     alert.severity === 'high' ? 'orange' : 'blue'}>
                            {alert.severity}
                          </Tag>
                          {alert.type}
                        </Space>
                      }
                      description={alert.message}
                    />
                  </List.Item>
                )}
              />
            </Card>
          </Col>
          <Col xs={24} lg={12}>
            <Card title="已加载模型" size="small">
              <List
                dataSource={loadedModels.slice(0, 5)}
                renderItem={(model) => (
                  <List.Item
                    actions={[
                      <Button
                        size="small"
                        danger
                        icon={<StopOutlined />}
                        onClick={() => handleUnloadModel(model.name)}
                      >
                        卸载
                      </Button>
                    ]}
                  >
                    <List.Item.Meta
                      title={`${model.name} v${model.version}`}
                      description={
                        <Space>
                          <Tag color={model.status === 'ready' ? 'green' : 'orange'}>
                            {model.status}
                          </Tag>
                          <span>{model.memory_usage_mb}MB</span>
                        </Space>
                      }
                    />
                  </List.Item>
                )}
              />
            </Card>
          </Col>
        </Row>
      </div>
    );
  };

  return (
    <div style={{ padding: 24 }}>
      <div style={{ marginBottom: 16 }}>
        <h1>模型服务管理</h1>
        <p>统一管理模型的上传、部署、推理、学习和监控</p>
      </div>

      <Spin spinning={loading}>
        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane tab="概览" key="overview">
            <Space direction="vertical" size="large" style={{ width: '100%' }}>
              {renderOverviewStats()}
              <Divider />
              {renderMonitoringDashboard()}
            </Space>
          </TabPane>

          <TabPane tab="模型管理" key="models">
            <Card
              title="模型列表"
              extra={
                <Space>
                  <Button
                    type="primary"
                    icon={<UploadOutlined />}
                    onClick={() => setUploadModalVisible(true)}
                  >
                    上传模型
                  </Button>
                  <Button
                    icon={<CloudOutlined />}
                    onClick={() => setHubModalVisible(true)}
                  >
                    从Hub注册
                  </Button>
                  <Button icon={<SettingOutlined />} onClick={loadModels}>
                    刷新
                  </Button>
                </Space>
              }
            >
              <Table
                columns={modelColumns}
                dataSource={models}
                rowKey={(record) => `${record.name}-${record.version}`}
                pagination={{ pageSize: 10 }}
              />
            </Card>
          </TabPane>

          <TabPane tab="部署管理" key="deployments">
            <Card
              title="部署列表"
              extra={
                <Button icon={<SettingOutlined />} onClick={loadDeployments}>
                  刷新
                </Button>
              }
            >
              <Table
                columns={deploymentColumns}
                dataSource={deployments}
                rowKey="deployment_id"
                pagination={{ pageSize: 10 }}
              />
            </Card>
          </TabPane>

          <TabPane tab="推理服务" key="inference">
            <Card
              title="模型推理"
              extra={
                <Button
                  type="primary"
                  icon={<ApiOutlined />}
                  onClick={() => setPredictModalVisible(true)}
                >
                  新建预测
                </Button>
              }
            >
              <Alert
                message="推理服务"
                description="在这里可以对已加载的模型进行实时推理，支持单次预测和批量预测。"
                type="info"
                showIcon
                style={{ marginBottom: 16 }}
              />
              {renderMonitoringDashboard()}
            </Card>
          </TabPane>

          <TabPane tab="学习会话" key="learning">
            <Card
              title="学习会话"
              extra={
                <Button
                  type="primary"
                  icon={<ExperimentOutlined />}
                  onClick={() => setLearningModalVisible(true)}
                >
                  开始学习
                </Button>
              }
            >
              <Table
                columns={learningSessionColumns}
                dataSource={learningSessions}
                rowKey="id"
                pagination={{ pageSize: 10 }}
              />
            </Card>
          </TabPane>

          <TabPane tab="AB测试" key="abtest">
            <Card
              title="AB测试"
              extra={
                <Button
                  type="primary"
                  icon={<ExperimentOutlined />}
                  onClick={() => setABTestModalVisible(true)}
                >
                  创建测试
                </Button>
              }
            >
              <Table
                columns={abTestColumns}
                dataSource={abTests}
                rowKey="id"
                pagination={{ pageSize: 10 }}
              />
            </Card>
          </TabPane>
        </Tabs>
      </Spin>

      {/* 上传模型模态框 */}
      <Modal
        title="上传模型"
        visible={uploadModalVisible}
        onCancel={() => setUploadModalVisible(false)}
        footer={null}
      >
        <Form form={uploadForm} onFinish={handleUploadModel} layout="vertical">
          <Form.Item
            name="name"
            label="模型名称"
            rules={[{ required: true, message: '请输入模型名称' }]}
          >
            <Input placeholder="请输入模型名称" />
          </Form.Item>
          <Form.Item
            name="version"
            label="版本"
            rules={[{ required: true, message: '请输入版本' }]}
          >
            <Input placeholder="例如: 1.0.0" />
          </Form.Item>
          <Form.Item
            name="type"
            label="模型类型"
            rules={[{ required: true, message: '请选择模型类型' }]}
          >
            <Select placeholder="选择模型类型">
              <Option value="classification">分类</Option>
              <Option value="regression">回归</Option>
              <Option value="nlp">自然语言处理</Option>
              <Option value="vision">计算机视觉</Option>
              <Option value="multimodal">多模态</Option>
            </Select>
          </Form.Item>
          <Form.Item
            name="model_file"
            label="模型文件"
            rules={[{ required: true, message: '请选择模型文件' }]}
          >
            <Upload beforeUpload={() => false} maxCount={1}>
              <Button icon={<UploadOutlined />}>选择文件</Button>
            </Upload>
          </Form.Item>
          <Form.Item name="description" label="描述">
            <TextArea placeholder="模型描述（可选）" />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                上传
              </Button>
              <Button onClick={() => setUploadModalVisible(false)}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* 从Hub注册模态框 */}
      <Modal
        title="从Hub注册模型"
        visible={hubModalVisible}
        onCancel={() => setHubModalVisible(false)}
        footer={null}
      >
        <Form form={hubForm} onFinish={handleRegisterFromHub} layout="vertical">
          <Form.Item
            name="hub_model_id"
            label="Hub模型ID"
            rules={[{ required: true, message: '请输入Hub模型ID' }]}
          >
            <Input placeholder="例如: huggingface/bert-base-uncased" />
          </Form.Item>
          <Form.Item
            name="name"
            label="本地名称"
            rules={[{ required: true, message: '请输入本地名称' }]}
          >
            <Input placeholder="本地存储的模型名称" />
          </Form.Item>
          <Form.Item
            name="version"
            label="版本"
            rules={[{ required: true, message: '请输入版本' }]}
          >
            <Input placeholder="例如: 1.0.0" />
          </Form.Item>
          <Form.Item name="description" label="描述">
            <TextArea placeholder="模型描述（可选）" />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                注册
              </Button>
              <Button onClick={() => setHubModalVisible(false)}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* 部署模型模态框 */}
      <Modal
        title="部署模型"
        visible={deployModalVisible}
        onCancel={() => setDeployModalVisible(false)}
        footer={null}
        width={600}
      >
        <Form form={deployForm} onFinish={handleDeployModel} layout="vertical">
          <Form.Item
            name="model_name"
            label="模型名称"
            rules={[{ required: true, message: '请输入模型名称' }]}
            initialValue={selectedModel?.name}
          >
            <Input />
          </Form.Item>
          <Form.Item
            name="model_version"
            label="模型版本"
            initialValue={selectedModel?.version || 'latest'}
          >
            <Input />
          </Form.Item>
          <Form.Item
            name="deployment_type"
            label="部署类型"
            rules={[{ required: true, message: '请选择部署类型' }]}
            initialValue="docker"
          >
            <Select>
              <Select.Option value="docker">Docker</Select.Option>
              <Select.Option value="kubernetes">Kubernetes</Select.Option>
              <Select.Option value="edge">Edge</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item
            name="replicas"
            label="副本数"
            rules={[{ required: true, message: '请输入副本数' }]}
            initialValue={1}
          >
            <InputNumber min={1} max={10} />
          </Form.Item>
          <Form.Item label="资源配置">
            <Form.Item
              name="cpu_request"
              label="CPU请求"
              rules={[{ required: true, message: '请输入CPU请求' }]}
              initialValue="200m"
            >
              <Input placeholder="例如: 200m" />
            </Form.Item>
            <Form.Item
              name="cpu_limit"
              label="CPU限制"
              rules={[{ required: true, message: '请输入CPU限制' }]}
              initialValue="1000m"
            >
              <Input placeholder="例如: 1000m" />
            </Form.Item>
            <Form.Item
              name="memory_request"
              label="内存请求"
              rules={[{ required: true, message: '请输入内存请求' }]}
              initialValue="512Mi"
            >
              <Input placeholder="例如: 512Mi" />
            </Form.Item>
            <Form.Item
              name="memory_limit"
              label="内存限制"
              rules={[{ required: true, message: '请输入内存限制' }]}
              initialValue="2Gi"
            >
              <Input placeholder="例如: 2Gi" />
            </Form.Item>
          </Form.Item>
          <Form.Item label="GPU配置">
            <Form.Item name="gpu_required" valuePropName="checked" initialValue={false}>
              <Switch checkedChildren="需要GPU" unCheckedChildren="不需要GPU" />
            </Form.Item>
            <Form.Item name="gpu_count" label="GPU数量" initialValue={0}>
              <InputNumber min={0} max={8} />
            </Form.Item>
          </Form.Item>
          <Form.Item name="port" label="服务端口" initialValue={8080}>
            <InputNumber min={1} max={65535} />
          </Form.Item>
          <Form.Item name="environment_vars" label="环境变量(JSON)">
            <TextArea rows={3} placeholder='{"KEY":"VALUE"}' />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                部署
              </Button>
              <Button onClick={() => setDeployModalVisible(false)}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* 预测模态框 */}
      <Modal
        title="模型预测"
        visible={predictModalVisible}
        onCancel={() => setPredictModalVisible(false)}
        footer={null}
      >
        <Form form={predictForm} onFinish={handlePredict} layout="vertical">
          <Form.Item
            name="model_name"
            label="模型名称"
            rules={[{ required: true, message: '请输入模型名称' }]}
          >
            <Input placeholder="请输入模型名称" />
          </Form.Item>
          <Form.Item name="version" label="版本（可选）">
            <Input placeholder="留空使用最新版本" />
          </Form.Item>
          <Form.Item
            name="input_data"
            label="输入数据"
            rules={[{ required: true, message: '请输入数据' }]}
          >
            <TextArea 
              placeholder="请输入JSON格式的数据" 
              rows={4}
            />
          </Form.Item>
          <Form.Item label="选项（可选）">
            <Form.Item name={['options', 'temperature']} label="Temperature">
              <InputNumber min={0} max={2} step={0.1} placeholder="0.7" />
            </Form.Item>
            <Form.Item name={['options', 'max_tokens']} label="Max Tokens">
              <InputNumber min={1} max={4000} placeholder="100" />
            </Form.Item>
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                预测
              </Button>
              <Button onClick={() => setPredictModalVisible(false)}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* 学习会话模态框 */}
      <Modal
        title="开始学习会话"
        visible={learningModalVisible}
        onCancel={() => setLearningModalVisible(false)}
        footer={null}
      >
        <Form form={learningForm} onFinish={handleStartLearning} layout="vertical">
          <Form.Item
            name="model_name"
            label="模型名称"
            rules={[{ required: true, message: '请输入模型名称' }]}
          >
            <Input placeholder="请输入模型名称" />
          </Form.Item>
          <Form.Item name="dataset_id" label="数据集ID">
            <Input placeholder="训练数据集ID（可选）" />
          </Form.Item>
          <Form.Item
            name="learning_rate"
            label="学习率"
            initialValue={0.001}
          >
            <InputNumber min={0.0001} max={1} step={0.001} />
          </Form.Item>
          <Form.Item
            name="batch_size"
            label="批次大小"
            initialValue={32}
          >
            <InputNumber min={1} max={1000} />
          </Form.Item>
          <Form.Item
            name="epochs"
            label="训练轮数"
            initialValue={10}
          >
            <InputNumber min={1} max={1000} />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                开始
              </Button>
              <Button onClick={() => setLearningModalVisible(false)}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* AB测试模态框 */}
      <Modal
        title="创建AB测试"
        visible={abTestModalVisible}
        onCancel={() => setABTestModalVisible(false)}
        footer={null}
      >
        <Form form={abTestForm} onFinish={handleCreateABTest} layout="vertical">
          <Form.Item
            name="name"
            label="测试名称"
            rules={[{ required: true, message: '请输入测试名称' }]}
          >
            <Input placeholder="请输入测试名称" />
          </Form.Item>
          <Form.Item
            name="model_a"
            label="模型A"
            rules={[{ required: true, message: '请输入模型A' }]}
          >
            <Input placeholder="基准模型" />
          </Form.Item>
          <Form.Item
            name="model_b"
            label="模型B"
            rules={[{ required: true, message: '请输入模型B' }]}
          >
            <Input placeholder="测试模型" />
          </Form.Item>
          <Form.Item
            name="traffic_split"
            label="流量分配（模型A的百分比）"
            rules={[{ required: true, message: '请设置流量分配' }]}
            initialValue={50}
          >
            <InputNumber min={10} max={90} />
          </Form.Item>
          <Form.Item name="duration_hours" label="测试时长（小时）">
            <InputNumber min={1} max={720} placeholder="24" />
          </Form.Item>
          <Form.Item
            name="success_metrics"
            label="成功指标"
            rules={[{ required: true, message: '请选择成功指标' }]}
          >
            <Select mode="multiple" placeholder="选择评估指标">
              <Option value="accuracy">准确率</Option>
              <Option value="latency">延迟</Option>
              <Option value="throughput">吞吐量</Option>
              <Option value="error_rate">错误率</Option>
            </Select>
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                创建
              </Button>
              <Button onClick={() => setABTestModalVisible(false)}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default ModelServiceManagementPage;
