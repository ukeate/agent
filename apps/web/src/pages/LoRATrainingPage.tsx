import React, { useState, useEffect } from 'react';
import { 
import { logger } from '../utils/logger'
  Card, 
  Row, 
  Col, 
  Typography, 
  Progress, 
  Statistic, 
  Table, 
  Button,
  Space,
  Tag,
  Tabs,
  Alert,
  Select,
  Input,
  Form,
  Slider,
  Switch,
  Tooltip,
  Badge,
  Timeline,
  Descriptions,
  Modal,
  InputNumber,
  message,
  Spin,
  Divider
} from 'antd';
import {
  GoldOutlined,
  ThunderboltOutlined,
  DashboardOutlined,
  LineChartOutlined,
  SettingOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  MonitorOutlined,
  DatabaseOutlined,
  CheckCircleOutlined,
  InfoCircleOutlined,
  FireOutlined,
  DesktopOutlined,
  ClockCircleOutlined,
  DownloadOutlined,
  ExperimentOutlined,
  ReloadOutlined,
  StopOutlined,
  SaveOutlined
} from '@ant-design/icons';
import { Line, Bar, Gauge } from '@ant-design/charts';
import {
  loraTrainingService,
  type TrainingJob,
  type TrainingProgress,
  type TrainingMetric,
  type ModelInfo,
  type CheckpointInfo,
  type LoRAConfig,
  type TrainingConfig
} from '../services/loraTrainingService';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { TextArea } = Input;
const { Option } = Select;

const LoRATrainingPage: React.FC = () => {
  // ==================== 状态管理 ====================
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');
  const [refreshing, setRefreshing] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  
  // 数据状态
  const [trainingJobs, setTrainingJobs] = useState<TrainingJob[]>([]);
  const [selectedJob, setSelectedJob] = useState<TrainingJob | null>(null);
  const [trainingProgress, setTrainingProgress] = useState<TrainingProgress | null>(null);
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetric[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([]);
  
  // 模态框状态
  const [configModalVisible, setConfigModalVisible] = useState(false);
  const [createJobModalVisible, setCreateJobModalVisible] = useState(false);
  const [inferenceModalVisible, setInferenceModalVisible] = useState(false);
  const [inferenceResult, setInferenceResult] = useState<any>(null);

  // ==================== 数据加载 ====================
  
  const loadTrainingJobs = async () => {
    setLoading(true);
    try {
      const jobs = await loraTrainingService.getTrainingJobs();
      setTrainingJobs(jobs);
      
      // 自动选择第一个运行中的任务
      const runningJob = jobs.find(j => j.status === 'running');
      if (runningJob && !selectedJob) {
        setSelectedJob(runningJob);
        await loadJobDetails(runningJob.job_id);
      }
    } catch (error) {
      logger.error('加载训练任务失败:', error);
      message.error('加载训练任务失败');
    } finally {
      setLoading(false);
    }
  };

  const loadJobDetails = async (jobId: string) => {
    try {
      const [progress, metrics, info, ckpts] = await Promise.all([
        loraTrainingService.getTrainingProgress(jobId),
        loraTrainingService.getTrainingMetrics(jobId),
        loraTrainingService.getModelInfo(jobId),
        loraTrainingService.getCheckpoints(jobId)
      ]);
      
      setTrainingProgress(progress);
      setTrainingMetrics(metrics);
      setModelInfo(info);
      setCheckpoints(ckpts);
    } catch (error) {
      logger.error('加载任务详情失败:', error);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadTrainingJobs();
    if (selectedJob) {
      await loadJobDetails(selectedJob.job_id);
    }
    setRefreshing(false);
    message.success('数据已刷新');
  };

  // ==================== 任务控制 ====================
  
  const handleJobControl = async (
    jobId: string,
    action: 'start' | 'pause' | 'resume' | 'stop'
  ) => {
    try {
      let result;
      switch (action) {
        case 'start':
          result = await loraTrainingService.startTraining(jobId);
          break;
        case 'pause':
          result = await loraTrainingService.pauseTraining(jobId);
          break;
        case 'resume':
          result = await loraTrainingService.resumeTraining(jobId);
          break;
        case 'stop':
          result = await loraTrainingService.stopTraining(jobId);
          break;
      }
      
      if (result?.success) {
        message.success(result.message);
        await loadTrainingJobs();
      }
    } catch (error) {
      message.error(`操作失败: ${error}`);
    }
  };

  const handleSaveCheckpoint = async () => {
    if (!selectedJob) return;
    
    try {
      const checkpoint = await loraTrainingService.saveCheckpoint(selectedJob.job_id);
      message.success('检查点保存成功');
      setCheckpoints([...checkpoints, checkpoint]);
    } catch (error) {
      message.error('保存检查点失败');
    }
  };

  const handleTestInference = async (values: any) => {
    if (!selectedJob) return;
    
    try {
      const result = await loraTrainingService.testInference(
        selectedJob.job_id,
        values.input_text,
        {
          max_new_tokens: values.max_new_tokens,
          temperature: values.temperature,
          top_p: values.top_p,
          do_sample: values.do_sample
        }
      );
      setInferenceResult(result);
    } catch (error) {
      message.error('推理测试失败');
    }
  };

  // ==================== 生命周期 ====================
  
  useEffect(() => {
    loadTrainingJobs();
  }, []);

  useEffect(() => {
    if (!autoRefresh || !selectedJob || selectedJob.status !== 'running') return;
    
    const timer = setInterval(() => {
      loadJobDetails(selectedJob.job_id);
    }, 5000);
    
    return () => clearInterval(timer);
  }, [autoRefresh, selectedJob]);

  // ==================== 图表配置 ====================
  
  const lossChartConfig = {
    data: trainingMetrics.map(m => ({
      step: m.step,
      trainLoss: m.train_loss,
      evalLoss: m.eval_loss
    })),
    xField: 'step',
    yField: ['trainLoss', 'evalLoss'],
    smooth: true,
    legend: {
      position: 'top-right',
    },
    yAxis: {
      title: { text: '损失值' }
    },
    tooltip: {
      showCrosshairs: true,
    }
  };

  const learningRateConfig = {
    data: trainingMetrics.map(m => ({
      step: m.step,
      lr: m.learning_rate
    })),
    xField: 'step',
    yField: 'lr',
    smooth: true,
    color: '#faad14',
    yAxis: {
      title: { text: '学习率' }
    }
  };

  const gaugeConfig = {
    percent: (trainingProgress?.progress_percent || 0) / 100,
    range: {
      color: '#52c41a'
    },
    indicator: {
      pointer: { style: { stroke: '#D0D0D0' } },
      pin: { style: { stroke: '#D0D0D0' } }
    },
    statistic: {
      content: {
        style: { fontSize: '24px', lineHeight: '24px' },
        formatter: () => `${trainingProgress?.progress_percent.toFixed(1)}%`
      }
    }
  };

  // ==================== 表格配置 ====================
  
  const jobColumns = [
    {
      title: '任务名称',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: TrainingJob) => (
        <Button type="link" onClick={() => {
          setSelectedJob(record);
          loadJobDetails(record.job_id);
        }}>
          {text}
        </Button>
      )
    },
    {
      title: '基础模型',
      dataIndex: 'base_model',
      key: 'base_model',
      render: (model: string) => (
        <Tooltip title={model}>
          <Text ellipsis style={{ maxWidth: 200 }}>
            {model.split('/').pop()}
          </Text>
        </Tooltip>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const colorMap: Record<string, string> = {
          created: 'default',
          pending: 'warning',
          running: 'processing',
          completed: 'success',
          failed: 'error',
          cancelled: 'default'
        };
        return <Tag color={colorMap[status]}>{status.toUpperCase()}</Tag>;
      }
    },
    {
      title: 'LoRA Rank',
      key: 'lora_rank',
      render: (_, record: TrainingJob) => record.lora_config.r
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (date: string) => new Date(date).toLocaleString()
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record: TrainingJob) => (
        <Space size="small">
          {record.status === 'created' && (
            <Button
              size="small"
              icon={<PlayCircleOutlined />}
              onClick={() => handleJobControl(record.job_id, 'start')}
            >
              启动
            </Button>
          )}
          {record.status === 'running' && (
            <>
              <Button
                size="small"
                icon={<PauseCircleOutlined />}
                onClick={() => handleJobControl(record.job_id, 'pause')}
              />
              <Button
                size="small"
                danger
                icon={<StopOutlined />}
                onClick={() => handleJobControl(record.job_id, 'stop')}
              />
            </>
          )}
          {record.status === 'paused' && (
            <Button
              size="small"
              icon={<PlayCircleOutlined />}
              onClick={() => handleJobControl(record.job_id, 'resume')}
            >
              恢复
            </Button>
          )}
        </Space>
      )
    }
  ];

  const loraLayerColumns = [
    {
      title: '层名称',
      dataIndex: 'layer_name',
      key: 'layer_name',
      render: (name: string) => (
        <Text code style={{ fontSize: 12 }}>{name}</Text>
      )
    },
    {
      title: 'Rank',
      dataIndex: 'rank',
      key: 'rank'
    },
    {
      title: 'Alpha',
      dataIndex: 'alpha',
      key: 'alpha'
    },
    {
      title: 'Dropout',
      dataIndex: 'dropout',
      key: 'dropout'
    },
    {
      title: '参数量',
      dataIndex: 'param_count',
      key: 'param_count',
      render: (count: number) => count.toLocaleString()
    },
    {
      title: '权重范数',
      dataIndex: 'weight_norm',
      key: 'weight_norm',
      render: (norm?: number) => norm ? norm.toFixed(4) : '-'
    },
    {
      title: '梯度范数',
      dataIndex: 'gradient_norm',
      key: 'gradient_norm',
      render: (norm?: number) => norm ? norm.toFixed(4) : '-'
    }
  ];

  const checkpointColumns = [
    {
      title: '检查点',
      dataIndex: 'checkpoint_id',
      key: 'checkpoint_id'
    },
    {
      title: '步数',
      dataIndex: 'step',
      key: 'step'
    },
    {
      title: '轮次',
      dataIndex: 'epoch',
      key: 'epoch'
    },
    {
      title: '训练损失',
      dataIndex: 'train_loss',
      key: 'train_loss',
      render: (loss: number) => loss.toFixed(4)
    },
    {
      title: '验证损失',
      dataIndex: 'eval_loss',
      key: 'eval_loss',
      render: (loss?: number) => loss ? loss.toFixed(4) : '-'
    },
    {
      title: '最优',
      dataIndex: 'is_best',
      key: 'is_best',
      render: (isBest: boolean) => isBest ? <CheckCircleOutlined style={{ color: '#52c41a' }} /> : null
    },
    {
      title: '大小',
      dataIndex: 'file_size_mb',
      key: 'file_size_mb',
      render: (size: number) => `${size.toFixed(1)} MB`
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (date: string) => new Date(date).toLocaleString()
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record: CheckpointInfo) => (
        <Space size="small">
          <Button
            size="small"
            icon={<DownloadOutlined />}
            onClick={() => message.info('下载功能开发中')}
          />
          {selectedJob && (
            <Button
              size="small"
              onClick={async () => {
                try {
                  await loraTrainingService.loadCheckpoint(selectedJob.job_id, record.checkpoint_id);
                  message.success('检查点加载成功');
                } catch (error) {
                  message.error('加载检查点失败');
                }
              }}
            >
              加载
            </Button>
          )}
        </Space>
      )
    }
  ];

  // ==================== 渲染 ====================

  if (loading && trainingJobs.length === 0) {
    return (
      <div style={{ padding: '24px', textAlign: 'center' }}>
        <Spin size="large" tip="加载训练任务..." />
      </div>
    );
  }

  return (
    <div style={{ padding: '24px' }}>
      {/* 页面标题 */}
      <div style={{ marginBottom: 24 }}>
        <Row align="middle" justify="space-between">
          <Col>
            <Title level={2}>
              <GoldOutlined style={{ marginRight: 8, color: '#faad14' }} />
              LoRA训练管理
            </Title>
            <Text type="secondary">高效参数微调(LoRA/QLoRA)训练任务管理与监控</Text>
          </Col>
          <Col>
            <Space>
              <Switch
                checked={autoRefresh}
                onChange={setAutoRefresh}
                checkedChildren="自动刷新"
                unCheckedChildren="手动"
              />
              <Button
                icon={<ReloadOutlined spin={refreshing} />}
                onClick={handleRefresh}
                loading={refreshing}
              >
                刷新
              </Button>
              <Button
                type="primary"
                icon={<PlayCircleOutlined />}
                onClick={() => setCreateJobModalVisible(true)}
              >
                创建任务
              </Button>
            </Space>
          </Col>
        </Row>
      </div>

      {/* 概览卡片 */}
      {selectedJob && trainingProgress && (
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="训练进度"
                value={trainingProgress.progress_percent}
                suffix="%"
                prefix={<DashboardOutlined />}
                valueStyle={{ color: '#52c41a' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="当前损失"
                value={trainingProgress.train_loss}
                precision={4}
                prefix={<LineChartOutlined />}
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="学习率"
                value={trainingProgress.learning_rate}
                precision={6}
                prefix={<FireOutlined />}
                valueStyle={{ color: '#faad14' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="GPU使用率"
                value={trainingProgress.gpu_utilization || 0}
                suffix="%"
                prefix={<ThunderboltOutlined />}
                valueStyle={{ 
                  color: trainingProgress.gpu_utilization > 90 ? '#ff4d4f' : '#52c41a'
                }}
              />
            </Card>
          </Col>
        </Row>
      )}

      {/* 主内容标签页 */}
      <Card>
        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane tab="任务管理" key="overview">
            <Table
              columns={jobColumns}
              dataSource={trainingJobs}
              rowKey="job_id"
              size="middle"
            />
            
            {selectedJob && (
              <Card title={`任务详情: ${selectedJob.name}`} style={{ marginTop: 16 }}>
                <Descriptions column={3} size="small">
                  <Descriptions.Item label="任务ID">{selectedJob.job_id}</Descriptions.Item>
                  <Descriptions.Item label="基础模型">{selectedJob.base_model}</Descriptions.Item>
                  <Descriptions.Item label="状态">
                    <Tag color={selectedJob.status === 'running' ? 'processing' : 'default'}>
                      {selectedJob.status.toUpperCase()}
                    </Tag>
                  </Descriptions.Item>
                  <Descriptions.Item label="LoRA Rank">{selectedJob.lora_config.r}</Descriptions.Item>
                  <Descriptions.Item label="LoRA Alpha">{selectedJob.lora_config.lora_alpha}</Descriptions.Item>
                  <Descriptions.Item label="Dropout">{selectedJob.lora_config.lora_dropout}</Descriptions.Item>
                  <Descriptions.Item label="轮次">
                    {trainingProgress ? `${trainingProgress.current_epoch}/${trainingProgress.total_epochs}` : '-'}
                  </Descriptions.Item>
                  <Descriptions.Item label="步数">
                    {trainingProgress ? `${trainingProgress.current_step}/${trainingProgress.total_steps}` : '-'}
                  </Descriptions.Item>
                  <Descriptions.Item label="剩余时间">
                    {trainingProgress?.estimated_time_remaining || '-'}
                  </Descriptions.Item>
                </Descriptions>
                
                {trainingProgress && (
                  <div style={{ textAlign: 'center', marginTop: 24 }}>
                    <Gauge {...gaugeConfig} width={200} height={200} />
                  </div>
                )}
              </Card>
            )}
          </TabPane>

          <TabPane tab="训练监控" key="monitor">
            {trainingMetrics.length > 0 ? (
              <Row gutter={[16, 16]}>
                <Col span={24}>
                  <Card title="训练损失曲线" size="small">
                    <Line {...lossChartConfig} height={300} />
                  </Card>
                </Col>
                <Col span={12}>
                  <Card title="学习率变化" size="small">
                    <Line {...learningRateConfig} height={200} />
                  </Card>
                </Col>
                <Col span={12}>
                  <Card title="性能指标" size="small">
                    {trainingProgress && (
                      <Descriptions column={2} size="small">
                        <Descriptions.Item label="样本/秒">
                          {trainingProgress.samples_per_second?.toFixed(2) || '-'}
                        </Descriptions.Item>
                        <Descriptions.Item label="梯度范数">
                          {trainingProgress.grad_norm?.toFixed(4) || '-'}
                        </Descriptions.Item>
                        <Descriptions.Item label="GPU显存">
                          {trainingProgress.memory_usage_gb ? 
                            `${trainingProgress.memory_usage_gb.toFixed(1)} GB` : '-'}
                        </Descriptions.Item>
                        <Descriptions.Item label="训练时长">
                          {trainingProgress.training_time_seconds ? 
                            `${Math.floor(trainingProgress.training_time_seconds / 3600)}小时${Math.floor((trainingProgress.training_time_seconds % 3600) / 60)}分钟` : '-'}
                        </Descriptions.Item>
                      </Descriptions>
                    )}
                  </Card>
                </Col>
              </Row>
            ) : (
              <Alert
                message="请选择一个训练任务查看监控数据"
                type="info"
                showIcon
              />
            )}
          </TabPane>

          <TabPane tab="模型信息" key="model">
            {modelInfo ? (
              <Row gutter={[16, 16]}>
                <Col span={24}>
                  <Card title="模型概览" size="small">
                    <Descriptions column={3} size="small">
                      <Descriptions.Item label="基础模型">{modelInfo.base_model}</Descriptions.Item>
                      <Descriptions.Item label="模型规模">{modelInfo.model_size}</Descriptions.Item>
                      <Descriptions.Item label="总参数量">
                        {(modelInfo.total_params / 1e9).toFixed(2)}B
                      </Descriptions.Item>
                      <Descriptions.Item label="可训练参数">
                        {(modelInfo.trainable_params / 1e6).toFixed(2)}M
                      </Descriptions.Item>
                      <Descriptions.Item label="训练占比">
                        {(modelInfo.trainable_percentage * 100).toFixed(3)}%
                      </Descriptions.Item>
                      <Descriptions.Item label="内存占用">
                        {(modelInfo.memory_footprint_mb / 1024).toFixed(2)} GB
                      </Descriptions.Item>
                      <Descriptions.Item label="磁盘大小">
                        {(modelInfo.disk_size_mb / 1024).toFixed(2)} GB
                      </Descriptions.Item>
                      <Descriptions.Item label="LoRA层数">
                        {modelInfo.lora_layers.length}
                      </Descriptions.Item>
                    </Descriptions>
                  </Card>
                </Col>
                <Col span={24}>
                  <Card 
                    title="LoRA层详情" 
                    size="small"
                    extra={
                      <Button
                        size="small"
                        icon={<SettingOutlined />}
                        onClick={() => setConfigModalVisible(true)}
                      >
                        修改配置
                      </Button>
                    }
                  >
                    <Table
                      columns={loraLayerColumns}
                      dataSource={modelInfo.lora_layers}
                      rowKey="layer_name"
                      size="small"
                      pagination={{ pageSize: 10 }}
                    />
                  </Card>
                </Col>
              </Row>
            ) : (
              <Alert
                message="请选择一个训练任务查看模型信息"
                type="info"
                showIcon
              />
            )}
          </TabPane>

          <TabPane tab="检查点" key="checkpoints">
            {selectedJob ? (
              <Card
                title="检查点管理"
                extra={
                  <Space>
                    <Button
                      icon={<SaveOutlined />}
                      onClick={handleSaveCheckpoint}
                      disabled={selectedJob.status !== 'running'}
                    >
                      保存当前状态
                    </Button>
                    <Button
                      icon={<DownloadOutlined />}
                      onClick={() => message.info('导出功能开发中')}
                    >
                      导出适配器
                    </Button>
                  </Space>
                }
              >
                <Table
                  columns={checkpointColumns}
                  dataSource={checkpoints}
                  rowKey="checkpoint_id"
                  size="small"
                />
              </Card>
            ) : (
              <Alert
                message="请选择一个训练任务查看检查点"
                type="info"
                showIcon
              />
            )}
          </TabPane>

          <TabPane tab="推理测试" key="inference">
            {selectedJob ? (
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <Card title="推理配置">
                    <Form
                      layout="vertical"
                      onFinish={handleTestInference}
                      initialValues={{
                        max_new_tokens: 128,
                        temperature: 0.7,
                        top_p: 0.9,
                        do_sample: true
                      }}
                    >
                      <Form.Item
                        name="input_text"
                        label="输入文本"
                        rules={[{ required: true, message: '请输入测试文本' }]}
                      >
                        <TextArea rows={4} placeholder="输入测试文本..." />
                      </Form.Item>
                      <Row gutter={16}>
                        <Col span={12}>
                          <Form.Item name="max_new_tokens" label="最大生成长度">
                            <InputNumber min={1} max={512} style={{ width: '100%' }} />
                          </Form.Item>
                        </Col>
                        <Col span={12}>
                          <Form.Item name="temperature" label="温度">
                            <Slider min={0} max={2} step={0.1} />
                          </Form.Item>
                        </Col>
                      </Row>
                      <Row gutter={16}>
                        <Col span={12}>
                          <Form.Item name="top_p" label="Top-p">
                            <Slider min={0} max={1} step={0.05} />
                          </Form.Item>
                        </Col>
                        <Col span={12}>
                          <Form.Item name="do_sample" label="采样" valuePropName="checked">
                            <Switch />
                          </Form.Item>
                        </Col>
                      </Row>
                      <Form.Item>
                        <Button
                          type="primary"
                          htmlType="submit"
                          icon={<ExperimentOutlined />}
                          block
                        >
                          开始推理
                        </Button>
                      </Form.Item>
                    </Form>
                  </Card>
                </Col>
                <Col span={12}>
                  <Card title="推理结果">
                    {inferenceResult ? (
                      <div>
                        <Descriptions column={1} size="small" style={{ marginBottom: 16 }}>
                          <Descriptions.Item label="生成时间">
                            {inferenceResult.generation_time_ms} ms
                          </Descriptions.Item>
                          <Descriptions.Item label="生成token数">
                            {inferenceResult.tokens_generated}
                          </Descriptions.Item>
                        </Descriptions>
                        <Divider orientation="left">输入</Divider>
                        <Text>{inferenceResult.input}</Text>
                        <Divider orientation="left">输出</Divider>
                        <Text>{inferenceResult.output}</Text>
                      </div>
                    ) : (
                      <Alert
                        message="暂无推理结果"
                        description="请在左侧配置推理参数并点击开始推理"
                        type="info"
                        showIcon
                      />
                    )}
                  </Card>
                </Col>
              </Row>
            ) : (
              <Alert
                message="请选择一个训练任务进行推理测试"
                type="info"
                showIcon
              />
            )}
          </TabPane>
        </Tabs>
      </Card>

      {/* 配置修改模态框 */}
      <Modal
        title="修改LoRA配置"
        visible={configModalVisible}
        onCancel={() => setConfigModalVisible(false)}
        footer={null}
        width={600}
      >
        <Alert
          message="配置修改将在下次训练启动时生效"
          type="warning"
          showIcon
          style={{ marginBottom: 16 }}
        />
        <Form
          layout="vertical"
          initialValues={selectedJob?.lora_config}
          onFinish={async (values) => {
            if (!selectedJob) return;
            try {
              await loraTrainingService.updateLoRAConfig(selectedJob.job_id, values);
              message.success('配置更新成功');
              setConfigModalVisible(false);
            } catch (error) {
              message.error('配置更新失败');
            }
          }}
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="r" label="LoRA Rank">
                <InputNumber min={1} max={256} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="lora_alpha" label="LoRA Alpha">
                <InputNumber min={1} max={512} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
          </Row>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="lora_dropout" label="Dropout">
                <Slider min={0} max={0.5} step={0.05} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="bias" label="Bias类型">
                <Select>
                  <Option value="none">无</Option>
                  <Option value="all">全部</Option>
                  <Option value="lora_only">仅LoRA</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
          <Form.Item name="target_modules" label="目标模块">
            <Select mode="multiple">
              <Option value="q_proj">Q投影</Option>
              <Option value="k_proj">K投影</Option>
              <Option value="v_proj">V投影</Option>
              <Option value="o_proj">O投影</Option>
              <Option value="gate_proj">门控投影</Option>
              <Option value="up_proj">上投影</Option>
              <Option value="down_proj">下投影</Option>
            </Select>
          </Form.Item>
          <Row gutter={16}>
            <Col span={8}>
              <Form.Item name="use_rslora" label="使用RSLoRA" valuePropName="checked">
                <Switch />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="use_dora" label="使用DoRA" valuePropName="checked">
                <Switch />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="init_lora_weights" label="初始化方式">
                <Select>
                  <Option value="gaussian">高斯</Option>
                  <Option value="pissa">PiSSA</Option>
                  <Option value={true}>默认</Option>
                  <Option value={false}>禁用</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                保存配置
              </Button>
              <Button onClick={() => setConfigModalVisible(false)}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* 创建任务模态框 */}
      <Modal
        title="创建LoRA训练任务"
        visible={createJobModalVisible}
        onCancel={() => setCreateJobModalVisible(false)}
        footer={null}
        width={800}
      >
        <Alert
          message="创建新的LoRA微调训练任务"
          description="选择基础模型、配置LoRA参数并设置训练超参数"
          type="info"
          showIcon
          style={{ marginBottom: 16 }}
        />
        <Form
          layout="vertical"
          onFinish={async (values) => {
            try {
              const job = await loraTrainingService.createTrainingJob({
                name: values.name,
                base_model: values.base_model,
                dataset_id: values.dataset_id,
                lora_config: {
                  r: values.r,
                  lora_alpha: values.lora_alpha,
                  lora_dropout: values.lora_dropout,
                  target_modules: values.target_modules
                },
                training_config: {
                  base_model: values.base_model,
                  model_type: values.model_type,
                  num_train_epochs: values.num_train_epochs,
                  per_device_train_batch_size: values.batch_size,
                  per_device_eval_batch_size: values.batch_size,
                  gradient_accumulation_steps: values.gradient_accumulation_steps,
                  learning_rate: values.learning_rate,
                  warmup_ratio: values.warmup_ratio,
                  lr_scheduler_type: values.lr_scheduler_type,
                  gradient_checkpointing: values.gradient_checkpointing,
                  fp16: values.fp16,
                  optim: 'adamw_torch',
                  weight_decay: 0.001,
                  max_grad_norm: 0.3,
                  max_seq_length: values.max_seq_length,
                  save_steps: values.save_steps,
                  eval_steps: values.eval_steps,
                  logging_steps: 10,
                  save_total_limit: 3,
                  load_best_model_at_end: true,
                  metric_for_best_model: 'eval_loss',
                  greater_is_better: false
                },
                use_qlora: values.use_qlora
              });
              message.success('训练任务创建成功');
              setCreateJobModalVisible(false);
              await loadTrainingJobs();
            } catch (error) {
              message.error('创建任务失败');
            }
          }}
          initialValues={{
            model_type: 'llama',
            r: 16,
            lora_alpha: 32,
            lora_dropout: 0.1,
            target_modules: ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
            num_train_epochs: 3,
            batch_size: 4,
            gradient_accumulation_steps: 4,
            learning_rate: 2e-4,
            warmup_ratio: 0.03,
            lr_scheduler_type: 'cosine',
            max_seq_length: 512,
            save_steps: 100,
            eval_steps: 100,
            gradient_checkpointing: true,
            fp16: true,
            use_qlora: false
          }}
        >
          <Form.Item
            name="name"
            label="任务名称"
            rules={[{ required: true, message: '请输入任务名称' }]}
          >
            <Input placeholder="例如: Llama2-7B 客服对话微调" />
          </Form.Item>
          
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="base_model"
                label="基础模型"
                rules={[{ required: true, message: '请选择基础模型' }]}
              >
                <Select>
                  <Option value="meta-llama/Llama-2-7b-chat-hf">Llama-2-7B-Chat</Option>
                  <Option value="meta-llama/Llama-2-13b-chat-hf">Llama-2-13B-Chat</Option>
                  <Option value="mistralai/Mistral-7B-Instruct-v0.2">Mistral-7B-Instruct</Option>
                  <Option value="Qwen/Qwen1.5-7B-Chat">Qwen1.5-7B-Chat</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="dataset_id"
                label="数据集"
                rules={[{ required: true, message: '请选择数据集' }]}
              >
                <Select>
                  <Option value="dataset-1">客服对话数据集</Option>
                  <Option value="dataset-2">技术文档数据集</Option>
                  <Option value="dataset-3">通用指令数据集</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Divider>LoRA配置</Divider>
          
          <Row gutter={16}>
            <Col span={6}>
              <Form.Item name="r" label="Rank">
                <InputNumber min={1} max={256} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={6}>
              <Form.Item name="lora_alpha" label="Alpha">
                <InputNumber min={1} max={512} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={6}>
              <Form.Item name="lora_dropout" label="Dropout">
                <Slider min={0} max={0.5} step={0.05} />
              </Form.Item>
            </Col>
            <Col span={6}>
              <Form.Item name="use_qlora" label="使用QLoRA" valuePropName="checked">
                <Switch />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item name="target_modules" label="目标模块">
            <Select mode="multiple">
              <Option value="q_proj">Q投影</Option>
              <Option value="k_proj">K投影</Option>
              <Option value="v_proj">V投影</Option>
              <Option value="o_proj">O投影</Option>
            </Select>
          </Form.Item>

          <Divider>训练配置</Divider>
          
          <Row gutter={16}>
            <Col span={8}>
              <Form.Item name="num_train_epochs" label="训练轮次">
                <InputNumber min={1} max={100} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="batch_size" label="批次大小">
                <InputNumber min={1} max={64} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="gradient_accumulation_steps" label="梯度累积">
                <InputNumber min={1} max={32} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={8}>
              <Form.Item name="learning_rate" label="学习率">
                <Input placeholder="2e-4" />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="warmup_ratio" label="预热比例">
                <Slider min={0} max={0.2} step={0.01} />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="lr_scheduler_type" label="调度器">
                <Select>
                  <Option value="linear">线性</Option>
                  <Option value="cosine">余弦</Option>
                  <Option value="polynomial">多项式</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={8}>
              <Form.Item name="max_seq_length" label="最大序列长度">
                <InputNumber min={128} max={4096} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="save_steps" label="保存间隔">
                <InputNumber min={10} max={1000} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="eval_steps" label="评估间隔">
                <InputNumber min={10} max={1000} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={8}>
              <Form.Item name="gradient_checkpointing" label="梯度检查点" valuePropName="checked">
                <Switch />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="fp16" label="FP16训练" valuePropName="checked">
                <Switch />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="model_type" label="模型类型">
                <Select>
                  <Option value="llama">Llama</Option>
                  <Option value="mistral">Mistral</Option>
                  <Option value="qwen">Qwen</Option>
                  <Option value="gpt">GPT</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" icon={<PlayCircleOutlined />}>
                创建并启动
              </Button>
              <Button onClick={() => setCreateJobModalVisible(false)}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default LoRATrainingPage;