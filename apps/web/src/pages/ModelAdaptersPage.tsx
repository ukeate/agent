import React, { useState, useEffect } from 'react';
import { 
import { logger } from '../utils/logger'
  Card, 
  Table, 
  Button, 
  Space, 
  Typography, 
  Row, 
  Col, 
  Statistic, 
  Tag,
  Progress,
  Tabs,
  Form,
  Input,
  Select,
  Slider,
  Switch,
  Alert,
  Descriptions,
  Modal,
  message
} from 'antd';
import {
  DeploymentUnitOutlined,
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  ExportOutlined,
  LineChartOutlined,
  DatabaseOutlined
} from '@ant-design/icons';
import fineTuningService, { TrainingJob } from '../services/fineTuningService';
import { modelService, type Deployment } from '../services/modelService';
import apiClient from '../services/apiClient';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

const ModelAdaptersPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('list');
  const [form] = Form.useForm();
  const [adapters, setAdapters] = useState<TrainingJob[]>([]);
  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [deployments, setDeployments] = useState<Deployment[]>([]);
  const [resourceMetrics, setResourceMetrics] = useState<{
    cpu: number;
    memory: number;
    disk: number;
    network: number;
    requestsRate: number;
    timestamp?: string;
  } | null>(null);

  const loadAdapters = async () => {
    setLoading(true);
    try {
      const jobs = await fineTuningService.getTrainingJobs();
      setAdapters(jobs || []);
    } catch (error) {
      setAdapters([]);
    } finally {
      setLoading(false);
    }
  };

  const loadDeployments = async () => {
    try {
      const data = await modelService.listDeployments();
      setDeployments(Array.isArray(data) ? data : []);
    } catch (error) {
      setDeployments([]);
    }
  };

  const loadResourceMetrics = async () => {
    try {
      const response = await apiClient.get('/metrics');
      const data = response.data || {};
      setResourceMetrics({
        cpu: Number(data?.cpu?.usage || 0),
        memory: Number(data?.memory?.usage || 0),
        disk: Number(data?.disk?.usage || 0),
        network: Number(data?.network?.throughput || 0),
        requestsRate: Number(data?.requests?.rate || 0),
        timestamp: data?.timestamp
      });
    } catch (error) {
      setResourceMetrics(null);
    }
  };

  useEffect(() => {
    loadAdapters();
    loadDeployments();
    loadResourceMetrics();
    form.setFieldsValue({
      type: 'lora',
      rank: 16,
      alpha: 32,
      dropout: 0.1,
      bias: false
    });
  }, []);

  const handleCreateAdapter = async () => {
    try {
      setCreating(true);
      const values = await form.validateFields();
      const trainingMode = values.type || 'lora';
      const loraConfig = {
        rank: Number(values.rank || 16),
        alpha: Number(values.alpha || 32),
        dropout: Number(values.dropout || 0.1),
        target_modules: values.targetModules?.length ? values.targetModules : undefined,
        bias: values.bias ? 'all' : 'none'
      };
      const payload: any = {
        job_name: values.name,
        model_name: values.baseModel,
        training_mode: trainingMode,
        dataset_path: values.datasetPath,
        lora_config: loraConfig
      };
      if (trainingMode === 'qlora') {
        payload.quantization_config = {
          quantization_type: 'nf4',
          bits: 4,
          use_double_quant: true,
          quant_type: 'nf4',
          compute_dtype: 'bfloat16'
        };
      }
      await fineTuningService.createTrainingJob(payload);
      message.success('训练任务已创建');
      form.resetFields();
      form.setFieldsValue({
        type: 'lora',
        rank: 16,
        alpha: 32,
        dropout: 0.1,
        bias: false
      });
      setActiveTab('list');
      loadAdapters();
    } catch (error: any) {
      const detail = error?.response?.data?.detail || error?.message || '创建失败';
      message.error(detail);
    } finally {
      setCreating(false);
    }
  };

  const handleViewAdapter = (record: TrainingJob) => {
    const config = record.config || {};
    const loraConfig = config.lora_config || {};
    Modal.info({
      title: '训练任务详情',
      width: 520,
      content: (
        <Descriptions column={1} size="small" bordered>
          <Descriptions.Item label="任务名称">{record.job_name}</Descriptions.Item>
          <Descriptions.Item label="任务状态">{record.status}</Descriptions.Item>
          <Descriptions.Item label="模型名称">{config.model_name || '-'}</Descriptions.Item>
          <Descriptions.Item label="训练模式">{config.training_mode || '-'}</Descriptions.Item>
          <Descriptions.Item label="数据集">{config.dataset_path || '-'}</Descriptions.Item>
          <Descriptions.Item label="进度">{record.progress}%</Descriptions.Item>
          <Descriptions.Item label="轮次">
            {record.current_epoch}/{record.total_epochs}
          </Descriptions.Item>
          <Descriptions.Item label="当前损失">{record.current_loss ?? '-'}</Descriptions.Item>
          <Descriptions.Item label="最佳损失">{record.best_loss ?? '-'}</Descriptions.Item>
          <Descriptions.Item label="Rank">{loraConfig.rank ?? '-'}</Descriptions.Item>
          <Descriptions.Item label="Alpha">{loraConfig.alpha ?? '-'}</Descriptions.Item>
        </Descriptions>
      )
    });
  };

  const handleDownloadAdapter = (jobId: string) => {
    window.open(`/api/v1/fine-tuning/jobs/${jobId}/download`, '_blank');
  };

  const handleDeleteAdapter = async (jobId: string) => {
    try {
      await fineTuningService.deleteTrainingJob(jobId);
      message.success('任务已删除');
      loadAdapters();
    } catch (error: any) {
      const detail = error?.response?.data?.detail || error?.message || '删除失败';
      message.error(detail);
    }
  };

  const columns = [
    {
      title: '适配器名称',
      dataIndex: 'job_name',
      key: 'job_name',
      render: (text: string, record: any) => (
        <div>
          <div style={{ fontWeight: 'bold' }}>{text}</div>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {record.config?.model_name || record.config?.model_architecture || ''}
          </Text>
        </div>
      ),
    },
    {
      title: '基座模型',
      dataIndex: 'model_name',
      key: 'model_name',
      render: (_: any, record: any) => record.config?.model_name || record.config?.model_architecture || '-'
    },
    {
      title: '适配器类型',
      dataIndex: 'training_mode',
      key: 'training_mode',
      render: (_: string, record: TrainingJob) => {
        const type = record.config?.training_mode;
        const color = type === 'qlora' ? 'purple' : type === 'lora' ? 'blue' : 'default';
        return (
          <Tag color={color}>
            {type || '-'}
          </Tag>
        );
      }
    },
    {
      title: '参数配置',
      key: 'params',
      render: (record: any) => (
        <div>
          <div>Rank: {record.config?.lora_config?.rank ?? '-'}</div>
          <div>Alpha: {record.config?.lora_config?.alpha ?? '-'}</div>
        </div>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const colorMap: Record<string, string> = {
          '训练完成': 'green',
          '训练中': 'processing',
          '已部署': 'success',
          completed: 'green',
          running: 'processing',
          pending: 'orange',
          failed: 'red'
        };
        return <Tag color={colorMap[status] || 'blue'}>{status || '-'}</Tag>;
      },
    },
    {
      title: '性能',
      dataIndex: 'progress',
      key: 'progress',
      render: (progress: number) => (
        <div>
          <div>{progress ? `${progress}%` : '-'}</div>
          <Progress percent={progress || 0} size="small" showInfo={false} />
        </div>
      ),
    },
    {
      title: '训练轮次',
      key: 'epochs',
      render: (record: TrainingJob) => `${record.current_epoch}/${record.total_epochs}`
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: any) => (
        <Space>
          <Button size="small" icon={<EditOutlined />} onClick={() => handleViewAdapter(record)}>查看</Button>
          <Button size="small" icon={<ExportOutlined />} onClick={() => handleDownloadAdapter(record.job_id)}>导出</Button>
          <Button danger size="small" icon={<DeleteOutlined />} onClick={() => handleDeleteAdapter(record.job_id)}>删除</Button>
        </Space>
      ),
    }
  ];

  const deploymentColumns = [
    {
      title: '部署ID',
      dataIndex: 'deployment_id',
      key: 'deployment_id',
      ellipsis: true,
    },
    {
      title: '模型',
      key: 'model',
      render: (record: Deployment) => `${record.model_name}:${record.model_version}`,
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
        const map: Record<string, { color: string; text: string }> = {
          pending: { color: 'orange', text: '等待中' },
          building: { color: 'processing', text: '构建中' },
          deploying: { color: 'processing', text: '部署中' },
          deployed: { color: 'green', text: '已部署' },
          failed: { color: 'red', text: '失败' },
          stopped: { color: 'default', text: '已停止' },
        };
        const config = map[status] || { color: 'default', text: status || '-' };
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
      key: 'action',
      render: (record: Deployment) => (
        <Button
          size="small"
          danger
          onClick={async () => {
            try {
              await modelService.deleteDeployment(record.deployment_id);
              await loadDeployments();
            } catch (error) {
              logger.error('停止部署失败:', error);
            }
          }}
        >
          停止
        </Button>
      ),
    },
  ];

  const totalDeployments = deployments.length;
  const activeDeployments = deployments.filter((d) => d.status === 'deployed').length;
  const failedDeployments = deployments.filter((d) => d.status === 'failed').length;
  const failedJobs = adapters.filter((a) => a.status === 'failed').length;
  const recommendations: Array<{ type: 'success' | 'info' | 'warning' | 'error'; title: string; detail: string }> = [];
  if (failedJobs > 0) {
    recommendations.push({
      type: 'error',
      title: '存在失败任务',
      detail: `当前有 ${failedJobs} 个训练任务失败，请检查日志与配置。`
    });
  }
  if (resourceMetrics) {
    if (resourceMetrics.cpu >= 85) {
      recommendations.push({
        type: 'warning',
        title: 'CPU负载偏高',
        detail: `CPU使用率 ${resourceMetrics.cpu.toFixed(1)}%，建议降低并发或错峰训练。`
      });
    }
    if (resourceMetrics.memory >= 85) {
      recommendations.push({
        type: 'warning',
        title: '内存占用偏高',
        detail: `内存使用率 ${resourceMetrics.memory.toFixed(1)}%，建议调低批次或启用梯度检查点。`
      });
    }
    if (resourceMetrics.disk >= 90) {
      recommendations.push({
        type: 'warning',
        title: '磁盘空间紧张',
        detail: `磁盘使用率 ${resourceMetrics.disk.toFixed(1)}%，建议清理旧模型与日志。`
      });
    }
  }
  if (!recommendations.length) {
    recommendations.push({
      type: 'success',
      title: '暂无异常',
      detail: '资源使用正常，未发现需要处理的问题。'
    });
  }

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <DeploymentUnitOutlined style={{ marginRight: 8, color: '#1890ff' }} />
          模型适配器管理
        </Title>
        <Text type="secondary">
          管理和部署LoRA/QLoRA适配器，实现模型的快速定制化
        </Text>
      </div>

      {/* 统计概览 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="适配器总数"
              value={adapters.length}
              prefix={<DeploymentUnitOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="已完成"
              value={adapters.filter(a => a.status === 'completed').length}
              prefix={<DatabaseOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均进度"
              value={
                adapters.length
                  ? (
                      adapters.reduce((sum, item) => sum + (Number(item.progress) || 0), 0) /
                      adapters.length
                    ).toFixed(1)
                  : '0'
              }
              suffix="%"
              prefix={<LineChartOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="失败任务"
              value={adapters.filter(a => a.status === 'failed').length}
              prefix={<DatabaseOutlined />}
              valueStyle={{ color: '#ff4d4f' }}
            />
          </Card>
        </Col>
      </Row>

      <Card>
        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane tab="适配器列表" key="list">
            <div style={{ marginBottom: 16 }}>
              <Space>
                <Button type="primary" icon={<PlusOutlined />} onClick={() => setActiveTab('create')}>
                  创建适配器
                </Button>
              </Space>
            </div>
            <Table 
              columns={columns} 
              dataSource={adapters} 
              rowKey="job_id"
              pagination={{ pageSize: 10 }}
              loading={loading}
            />
          </TabPane>

          <TabPane tab="创建适配器" key="create">
            <Row gutter={24}>
              <Col span={12}>
                <Card title="基础配置" size="small">
                  <Form form={form} layout="vertical">
                    <Form.Item label="适配器名称" name="name" rules={[{ required: true }]}>
                      <Input placeholder="输入适配器名称" />
                    </Form.Item>
                    
                    <Form.Item label="基座模型" name="baseModel" rules={[{ required: true }]}>
                      <Input placeholder="例如：Qwen/Qwen2-7B-Instruct" />
                    </Form.Item>

                    <Form.Item label="适配器类型" name="type" rules={[{ required: true }]}>
                      <Select placeholder="选择适配器类型">
                        <Option value="lora">LoRA</Option>
                        <Option value="qlora">QLoRA</Option>
                      </Select>
                    </Form.Item>

                    <Form.Item label="数据集路径" name="datasetPath" rules={[{ required: true }]}>
                      <Input placeholder="例如：./datasets/train.jsonl" />
                    </Form.Item>
                  </Form>
                </Card>
              </Col>

              <Col span={12}>
                <Card title="LoRA参数配置" size="small">
                  <Form form={form} layout="vertical">
                    <Form.Item label="Rank (r)" name="rank">
                      <div>
                        <Slider 
                          min={1} 
                          max={256} 
                          marks={{ 1: '1', 16: '16', 64: '64', 256: '256' }}
                        />
                        <Text type="secondary">
                          控制适配器容量，值越大表达能力越强但参数量越多
                        </Text>
                      </div>
                    </Form.Item>

                    <Form.Item label="Alpha (α)" name="alpha">
                      <div>
                        <Slider 
                          min={1} 
                          max={128} 
                          marks={{ 1: '1', 16: '16', 32: '32', 64: '64', 128: '128' }}
                        />
                        <Text type="secondary">
                          缩放因子，通常设置为Rank的2倍
                        </Text>
                      </div>
                    </Form.Item>

                    <Form.Item label="Dropout" name="dropout">
                      <div>
                        <Slider 
                          min={0} 
                          max={0.5} 
                          step={0.1}
                          marks={{ 0: '0', 0.1: '0.1', 0.3: '0.3', 0.5: '0.5' }}
                        />
                        <Text type="secondary">
                          防止过拟合的正则化参数
                        </Text>
                      </div>
                    </Form.Item>

                    <Form.Item label="目标模块" name="targetModules">
                      <Select mode="multiple" placeholder="选择要应用LoRA的模块">
                        <Option value="q_proj">q_proj (查询投影)</Option>
                        <Option value="v_proj">v_proj (值投影)</Option>
                        <Option value="k_proj">k_proj (键投影)</Option>
                        <Option value="o_proj">o_proj (输出投影)</Option>
                        <Option value="gate_proj">gate_proj (门控投影)</Option>
                        <Option value="up_proj">up_proj (上投影)</Option>
                        <Option value="down_proj">down_proj (下投影)</Option>
                      </Select>
                    </Form.Item>

                    <Form.Item label="启用偏置" name="bias" valuePropName="checked">
                      <Switch />
                    </Form.Item>
                  </Form>
                </Card>
              </Col>
            </Row>

            <div style={{ marginTop: 16, textAlign: 'center' }}>
              <Space>
                <Button type="primary" size="large" onClick={handleCreateAdapter} loading={creating}>
                  创建适配器
                </Button>
                <Button size="large" onClick={() => form.resetFields()}>
                  重置配置
                </Button>
              </Space>
            </div>
          </TabPane>

          <TabPane tab="性能分析" key="performance">
            <Row gutter={16}>
              <Col span={12}>
                <Card title="适配器性能对比" size="small" style={{ marginBottom: 16 }}>
                  <div style={{ marginBottom: 16 }}>
                    <Alert
                      message="训练进度与损失"
                      description="展示当前训练任务的进度与损失指标"
                      type="info"
                      showIcon
                    />
                  </div>
                  
                  {adapters.length === 0 && (
                    <Alert message="暂无训练任务数据" type="warning" showIcon />
                  )}
                  {adapters.map(adapter => {
                    const config = adapter.config || {};
                    const loraConfig = config.lora_config || {};
                    const trainingMode = config.training_mode || '';
                    const modeColor = trainingMode === 'qlora' ? 'purple' : trainingMode === 'lora' ? 'blue' : 'default';
                    return (
                    <div key={adapter.job_id} style={{ marginBottom: 16, padding: 16, border: '1px solid #f0f0f0', borderRadius: 6 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                        <Text strong>{adapter.job_name}</Text>
                        <Tag color={modeColor}>
                          {trainingMode || '-'}
                        </Tag>
                      </div>
                      <div style={{ marginBottom: 8 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                          <Text>训练进度</Text>
                          <Text>{adapter.progress}%</Text>
                        </div>
                        <Progress percent={adapter.progress || 0} size="small" />
                      </div>
                      <Descriptions column={2} size="small" bordered>
                        <Descriptions.Item label="模型">{config.model_name || '-'}</Descriptions.Item>
                        <Descriptions.Item label="轮次">
                          {adapter.current_epoch}/{adapter.total_epochs}
                        </Descriptions.Item>
                        <Descriptions.Item label="当前损失">{adapter.current_loss ?? '-'}</Descriptions.Item>
                        <Descriptions.Item label="最佳损失">{adapter.best_loss ?? '-'}</Descriptions.Item>
                        <Descriptions.Item label="Rank">{loraConfig.rank ?? '-'}</Descriptions.Item>
                        <Descriptions.Item label="Alpha">{loraConfig.alpha ?? '-'}</Descriptions.Item>
                      </Descriptions>
                    </div>
                  )})}
                </Card>
              </Col>

              <Col span={12}>
                <Card title="资源使用分析" size="small" style={{ marginBottom: 16 }}>
                  {!resourceMetrics && (
                    <Alert message="暂无资源监控数据" type="warning" showIcon />
                  )}
                  {resourceMetrics && (
                    <>
                      <div style={{ marginBottom: 16 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                          <Text>CPU使用率</Text>
                          <Text strong>{resourceMetrics.cpu.toFixed(1)}%</Text>
                        </div>
                        <Progress percent={resourceMetrics.cpu} strokeColor="#faad14" />
                      </div>

                      <div style={{ marginBottom: 16 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                          <Text>内存使用率</Text>
                          <Text strong>{resourceMetrics.memory.toFixed(1)}%</Text>
                        </div>
                        <Progress percent={resourceMetrics.memory} strokeColor="#1890ff" />
                      </div>

                      <div style={{ marginBottom: 16 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                          <Text>磁盘使用率</Text>
                          <Text strong>{resourceMetrics.disk.toFixed(1)}%</Text>
                        </div>
                        <Progress percent={resourceMetrics.disk} strokeColor="#52c41a" />
                      </div>

                      <Descriptions bordered size="small">
                        <Descriptions.Item label="网络吞吐">{resourceMetrics.network.toFixed(2)} MB/s</Descriptions.Item>
                        <Descriptions.Item label="请求速率">{resourceMetrics.requestsRate.toFixed(2)} req/s</Descriptions.Item>
                        <Descriptions.Item label="采集时间">
                          {resourceMetrics.timestamp ? new Date(resourceMetrics.timestamp).toLocaleString() : '-'}
                        </Descriptions.Item>
                      </Descriptions>
                    </>
                  )}
                </Card>

                <Card title="优化建议" size="small">
                  <Space direction="vertical" style={{ width: '100%' }}>
                    {recommendations.map((item, index) => (
                      <Alert key={index} type={item.type} message={item.title} description={item.detail} showIcon />
                    ))}
                  </Space>
                </Card>
              </Col>
            </Row>
          </TabPane>

          <TabPane tab="部署管理" key="deployment">
            <Row gutter={16}>
              <Col span={16}>
                <Card title="部署状态" size="small">
                  <Table
                    columns={deploymentColumns}
                    dataSource={deployments}
                    rowKey="deployment_id"
                    size="small"
                  />
                </Card>
              </Col>
              
              <Col span={8}>
                <Card title="快速部署" size="small" style={{ marginBottom: 16 }}>
                  <Form layout="vertical">
                    <Form.Item label="选择适配器">
                      <Select placeholder="选择要部署的适配器">
                        {adapters.map(adapter => (
                          <Option key={adapter.id} value={adapter.id}>
                            {adapter.name}
                          </Option>
                        ))}
                      </Select>
                    </Form.Item>
                    
                    <Form.Item label="部署环境">
                      <Select placeholder="选择部署环境">
                        <Option value="dev">开发环境</Option>
                        <Option value="test">测试环境</Option>
                        <Option value="prod">生产环境</Option>
                      </Select>
                    </Form.Item>
                    
                    <Form.Item label="实例数量">
                      <Slider min={1} max={10} defaultValue={1} marks={{ 1: '1', 5: '5', 10: '10' }} />
                    </Form.Item>
                    
                    <Button type="primary" block>
                      立即部署
                    </Button>
                  </Form>
                </Card>

                <Card title="部署统计" size="small">
                  <Statistic title="总部署数" value={totalDeployments} style={{ marginBottom: 16 }} />
                  <Statistic title="运行中" value={activeDeployments} style={{ marginBottom: 16 }} />
                  <Statistic title="失败" value={failedDeployments} />
                </Card>
              </Col>
            </Row>
          </TabPane>
        </Tabs>
      </Card>
    </div>
  );
};

export default ModelAdaptersPage;
