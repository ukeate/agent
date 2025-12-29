import React, { useState, useEffect } from 'react';
import { 
import { logger } from '../utils/logger'
  Card, 
  Typography, 
  Row, 
  Col, 
  Statistic, 
  Progress, 
  Descriptions, 
  Table,
  Button,
  Space,
  Tag,
  Alert,
  Tabs,
  Select,
  Switch,
  Tooltip,
  Badge,
  message,
  Spin,
  Modal,
  Form,
  InputNumber,
  Divider
} from 'antd';
import { 
  ClusterOutlined, 
  ThunderboltOutlined, 
  DatabaseOutlined,
  PlayCircleOutlined,
  PauseOutlined,
  StopOutlined,
  ReloadOutlined,
  SettingOutlined,
  CloudServerOutlined,
  SyncOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  RocketOutlined
} from '@ant-design/icons';
import { Line, Column, Gauge } from '@ant-design/charts';
import { 
  distributedTrainingService,
  type ClusterStatus,
  type GPUNode,
  type TrainingJob,
  type DeepSpeedConfig,
  type TrainingMetrics
} from '../services/distributedTrainingService';

const { Title, Text } = Typography;
const { TabPane } = Tabs;

const DistributedTrainingPage: React.FC = () => {
  // ==================== 状态管理 ====================
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');
  
  // 集群数据
  const [clusterStatus, setClusterStatus] = useState<ClusterStatus | null>(null);
  const [gpuNodes, setGPUNodes] = useState<GPUNode[]>([]);
  const [trainingJobs, setTrainingJobs] = useState<TrainingJob[]>([]);
  const [selectedJob, setSelectedJob] = useState<TrainingJob | null>(null);
  const [deepSpeedConfig, setDeepSpeedConfig] = useState<DeepSpeedConfig | null>(null);
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetrics[]>([]);
  
  // 控制状态
  const [configModalVisible, setConfigModalVisible] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(5000);

  // ==================== 数据加载 ====================
  
  const loadClusterData = async () => {
    setLoading(true);
    try {
      const [status, nodes, jobs] = await Promise.all([
        distributedTrainingService.getClusterStatus(),
        distributedTrainingService.getGPUNodes(),
        distributedTrainingService.getTrainingJobs()
      ]);
      
      setClusterStatus(status);
      setGPUNodes(nodes);
      setTrainingJobs(jobs);
      
      // 选择第一个运行中的任务
      const runningJob = jobs.find(j => j.status === 'running');
      if (runningJob && !selectedJob) {
        setSelectedJob(runningJob);
        await loadJobDetails(runningJob.job_id);
      }
    } catch (error) {
      logger.error('加载集群数据失败:', error);
      message.error('加载集群数据失败');
    } finally {
      setLoading(false);
    }
  };

  const loadJobDetails = async (jobId: string) => {
    try {
      const [config, metrics] = await Promise.all([
        distributedTrainingService.getDeepSpeedConfig(jobId),
        distributedTrainingService.getTrainingMetrics(jobId)
      ]);
      
      setDeepSpeedConfig(config);
      setTrainingMetrics(metrics);
    } catch (error) {
      logger.error('加载任务详情失败:', error);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadClusterData();
    setRefreshing(false);
    message.success('数据已刷新');
  };

  // ==================== 任务控制 ====================
  
  const handleJobControl = async (jobId: string, action: 'start' | 'pause' | 'resume' | 'stop') => {
    try {
      const result = await distributedTrainingService.controlTrainingJob(jobId, action);
      if (result.success) {
        message.success(result.message);
        await loadClusterData();
      }
    } catch (error) {
      message.error(`操作失败: ${error}`);
    }
  };

  // ==================== 生命周期 ====================
  
  useEffect(() => {
    loadClusterData();
  }, []);

  useEffect(() => {
    if (!autoRefresh) return;
    
    const timer = setInterval(() => {
      loadClusterData();
    }, refreshInterval);
    
    return () => clearInterval(timer);
  }, [autoRefresh, refreshInterval]);

  // ==================== 图表配置 ====================
  
  const lossChartConfig = {
    data: trainingMetrics.map(m => ({
      time: new Date(m.timestamp).toLocaleTimeString(),
      loss: m.loss,
      accuracy: m.accuracy || 0
    })),
    xField: 'time',
    yField: 'loss',
    smooth: true,
    color: '#1890ff',
    point: { size: 3 },
    yAxis: {
      title: { text: '训练损失' }
    }
  };

  const gpuUtilizationConfig = {
    data: gpuNodes.map(node => ({
      name: node.name,
      usage: node.usage_percent,
      status: node.status
    })),
    xField: 'name',
    yField: 'usage',
    color: ({ status }) => {
      return status === 'training' ? '#52c41a' : status === 'idle' ? '#1890ff' : '#ff4d4f';
    },
    label: {
      position: 'top',
      formatter: (v) => `${v.usage.toFixed(1)}%`
    },
    yAxis: {
      title: { text: 'GPU使用率 (%)' },
      max: 100
    }
  };

  const efficiencyGaugeConfig = {
    percent: (clusterStatus?.cluster_efficiency || 0) / 100,
    range: {
      color: clusterStatus?.cluster_efficiency >= 80 ? '#52c41a' : 
             clusterStatus?.cluster_efficiency >= 60 ? '#faad14' : '#ff4d4f'
    },
    indicator: {
      pointer: { style: { stroke: '#D0D0D0' } },
      pin: { style: { stroke: '#D0D0D0' } }
    },
    statistic: {
      content: {
        style: { fontSize: '24px', lineHeight: '24px' },
        formatter: () => `${clusterStatus?.cluster_efficiency.toFixed(1)}%`
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
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const colorMap: Record<string, string> = {
          running: 'processing',
          pending: 'default',
          completed: 'success',
          failed: 'error',
          paused: 'warning'
        };
        return <Tag color={colorMap[status]}>{status.toUpperCase()}</Tag>;
      }
    },
    {
      title: '策略',
      dataIndex: 'strategy',
      key: 'strategy',
      render: (strategy: string) => (
        <Tag color="blue">{strategy.replace('_', ' ').toUpperCase()}</Tag>
      )
    },
    {
      title: '进度',
      dataIndex: 'progress_percent',
      key: 'progress',
      render: (progress: number) => (
        <Progress percent={progress} size="small" style={{ width: 100 }} />
      )
    },
    {
      title: 'GPU数',
      dataIndex: 'num_workers',
      key: 'num_workers',
      render: (num: number) => `${num} GPUs`
    },
    {
      title: '同步效率',
      dataIndex: 'sync_efficiency',
      key: 'sync_efficiency',
      render: (efficiency: number) => (
        <span style={{ color: efficiency >= 85 ? '#52c41a' : efficiency >= 70 ? '#faad14' : '#ff4d4f' }}>
          {efficiency.toFixed(1)}%
        </span>
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record: TrainingJob) => (
        <Space size="small">
          {record.status === 'running' && (
            <Tooltip title="暂停">
              <Button
                size="small"
                icon={<PauseOutlined />}
                onClick={() => handleJobControl(record.job_id, 'pause')}
              />
            </Tooltip>
          )}
          {record.status === 'paused' && (
            <Tooltip title="继续">
              <Button
                size="small"
                icon={<PlayCircleOutlined />}
                onClick={() => handleJobControl(record.job_id, 'resume')}
              />
            </Tooltip>
          )}
          {record.status === 'pending' && (
            <Tooltip title="启动">
              <Button
                size="small"
                icon={<PlayCircleOutlined />}
                onClick={() => handleJobControl(record.job_id, 'start')}
              />
            </Tooltip>
          )}
          <Tooltip title="停止">
            <Button
              size="small"
              danger
              icon={<StopOutlined />}
              onClick={() => handleJobControl(record.job_id, 'stop')}
            />
          </Tooltip>
        </Space>
      )
    }
  ];

  const gpuColumns = [
    {
      title: 'GPU',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: GPUNode) => (
        <Space>
          <Badge status={record.status === 'training' ? 'processing' : 
                        record.status === 'idle' ? 'success' : 'error'} />
          {name}
        </Space>
      )
    },
    {
      title: '节点',
      dataIndex: 'node_id',
      key: 'node_id'
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const colorMap: Record<string, string> = {
          idle: 'success',
          training: 'processing',
          error: 'error',
          offline: 'default'
        };
        return <Tag color={colorMap[status]}>{status.toUpperCase()}</Tag>;
      }
    },
    {
      title: '使用率',
      dataIndex: 'usage_percent',
      key: 'usage',
      render: (usage: number) => (
        <Progress 
          percent={usage} 
          size="small" 
          strokeColor={usage > 90 ? '#ff4d4f' : usage > 70 ? '#faad14' : '#52c41a'}
        />
      )
    },
    {
      title: '显存',
      key: 'memory',
      render: (_, record: GPUNode) => (
        <span>{record.memory_used_gb.toFixed(1)}/{record.memory_total_gb}GB</span>
      )
    },
    {
      title: '温度',
      dataIndex: 'temperature_celsius',
      key: 'temperature',
      render: (temp: number) => (
        <span style={{ color: temp > 80 ? '#ff4d4f' : temp > 70 ? '#faad14' : '#52c41a' }}>
          {temp.toFixed(1)}°C
        </span>
      )
    },
    {
      title: '功率',
      dataIndex: 'power_watts',
      key: 'power',
      render: (power: number) => `${power.toFixed(0)}W`
    },
    {
      title: '当前任务',
      dataIndex: 'current_task',
      key: 'current_task',
      render: (task?: string) => task || '-'
    }
  ];

  // ==================== 渲染 ====================

  if (loading && !clusterStatus) {
    return (
      <div style={{ padding: '24px', textAlign: 'center' }}>
        <Spin size="large" tip="加载集群数据..." />
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
              <ClusterOutlined style={{ marginRight: 8, color: '#13c2c2' }} />
              分布式训练管理
            </Title>
            <Text type="secondary">管理和监控多GPU分布式训练任务</Text>
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
                icon={<RocketOutlined />}
                onClick={() => message.info('创建任务功能开发中')}
              >
                新建任务
              </Button>
            </Space>
          </Col>
        </Row>
      </div>

      {/* 集群概览卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic 
              title="活跃节点" 
              value={clusterStatus?.active_nodes || 0}
              suffix={`/ ${clusterStatus?.total_nodes || 0}`}
              prefix={<CloudServerOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic 
              title="GPU使用" 
              value={clusterStatus ? clusterStatus.total_gpus - clusterStatus.available_gpus : 0}
              suffix={`/ ${clusterStatus?.total_gpus || 0}`}
              prefix={<ThunderboltOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic 
              title="总显存" 
              value={clusterStatus?.used_memory_gb || 0}
              suffix={`/ ${clusterStatus?.total_memory_gb || 0} GB`}
              prefix={<DatabaseOutlined />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic 
              title="集群效率" 
              value={clusterStatus?.cluster_efficiency || 0}
              suffix="%"
              prefix={<SyncOutlined />}
              valueStyle={{ 
                color: clusterStatus?.cluster_efficiency >= 80 ? '#52c41a' : 
                       clusterStatus?.cluster_efficiency >= 60 ? '#faad14' : '#ff4d4f'
              }}
            />
          </Card>
        </Col>
      </Row>

      {/* 主内容标签页 */}
      <Card>
        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane tab="GPU状态" key="overview">
            <Row gutter={[16, 16]}>
              <Col span={16}>
                <Card title="GPU节点列表" size="small">
                  <Table
                    columns={gpuColumns}
                    dataSource={gpuNodes}
                    rowKey="gpu_id"
                    size="small"
                    pagination={false}
                  />
                </Card>
              </Col>
              <Col span={8}>
                <Card title="GPU使用率分布" size="small" style={{ marginBottom: 16 }}>
                  <Column {...gpuUtilizationConfig} height={200} />
                </Card>
                <Card title="集群效率" size="small">
                  <div style={{ textAlign: 'center', padding: '20px 0' }}>
                    <Gauge {...efficiencyGaugeConfig} width={200} height={200} />
                  </div>
                  <Divider />
                  <Descriptions column={1} size="small">
                    <Descriptions.Item label="网络带宽">
                      {clusterStatus?.network_bandwidth_gbps || 0} Gbps
                    </Descriptions.Item>
                    <Descriptions.Item label="运行任务">
                      {clusterStatus?.jobs_running || 0} 个
                    </Descriptions.Item>
                    <Descriptions.Item label="等待任务">
                      {clusterStatus?.jobs_pending || 0} 个
                    </Descriptions.Item>
                  </Descriptions>
                </Card>
              </Col>
            </Row>
          </TabPane>

          <TabPane tab="训练任务" key="jobs">
            <div style={{ marginBottom: 16 }}>
              <Alert
                message="训练任务管理"
                description={`当前有 ${trainingJobs.filter(j => j.status === 'running').length} 个任务正在运行，${trainingJobs.filter(j => j.status === 'pending').length} 个任务等待中`}
                type="info"
                showIcon
              />
            </div>
            <Table
              columns={jobColumns}
              dataSource={trainingJobs}
              rowKey="job_id"
              size="middle"
            />
            {selectedJob && (
              <Card title={`任务详情: ${selectedJob.name}`} style={{ marginTop: 16 }}>
                <Row gutter={16}>
                  <Col span={12}>
                    <Descriptions column={1} size="small">
                      <Descriptions.Item label="任务ID">{selectedJob.job_id}</Descriptions.Item>
                      <Descriptions.Item label="分布策略">{selectedJob.strategy}</Descriptions.Item>
                      <Descriptions.Item label="工作节点">{selectedJob.num_workers}</Descriptions.Item>
                      <Descriptions.Item label="当前轮次">
                        {selectedJob.current_epoch}/{selectedJob.total_epochs}
                      </Descriptions.Item>
                      <Descriptions.Item label="当前损失">{selectedJob.current_loss.toFixed(4)}</Descriptions.Item>
                      <Descriptions.Item label="预计剩余">
                        {selectedJob.estimated_time_remaining || '计算中...'}
                      </Descriptions.Item>
                    </Descriptions>
                  </Col>
                  <Col span={12}>
                    {trainingMetrics.length > 0 && (
                      <div>
                        <Title level={5}>训练损失曲线</Title>
                        <Line {...lossChartConfig} height={200} />
                      </div>
                    )}
                  </Col>
                </Row>
              </Card>
            )}
          </TabPane>

          <TabPane tab="DeepSpeed配置" key="deepspeed">
            {deepSpeedConfig ? (
              <Card>
                <Row gutter={[16, 16]}>
                  <Col span={12}>
                    <Card title="当前配置" size="small">
                      <Descriptions column={1} size="small">
                        <Descriptions.Item label="ZeRO Stage">{deepSpeedConfig.zero_stage}</Descriptions.Item>
                        <Descriptions.Item label="优化器卸载">
                          {deepSpeedConfig.offload_optimizer ? '启用' : '禁用'}
                        </Descriptions.Item>
                        <Descriptions.Item label="参数卸载">
                          {deepSpeedConfig.offload_params ? '启用' : '禁用'}
                        </Descriptions.Item>
                        <Descriptions.Item label="梯度压缩">
                          {deepSpeedConfig.gradient_compression ? '启用' : '禁用'}
                        </Descriptions.Item>
                        <Descriptions.Item label="通信后端">
                          {deepSpeedConfig.communication_backend.toUpperCase()}
                        </Descriptions.Item>
                        <Descriptions.Item label="FP16">
                          {deepSpeedConfig.fp16_enabled ? '启用' : '禁用'}
                        </Descriptions.Item>
                        <Descriptions.Item label="BF16">
                          {deepSpeedConfig.bf16_enabled ? '启用' : '禁用'}
                        </Descriptions.Item>
                        <Descriptions.Item label="梯度累积步数">
                          {deepSpeedConfig.gradient_accumulation_steps}
                        </Descriptions.Item>
                        <Descriptions.Item label="总批次大小">
                          {deepSpeedConfig.train_batch_size}
                        </Descriptions.Item>
                        <Descriptions.Item label="单GPU批次">
                          {deepSpeedConfig.train_micro_batch_size_per_gpu}
                        </Descriptions.Item>
                      </Descriptions>
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card 
                      title="优化建议" 
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
                      <Alert
                        message="配置优化建议"
                        description={
                          <ul style={{ margin: '8px 0', paddingLeft: 20 }}>
                            <li>当前使用ZeRO-{deepSpeedConfig.zero_stage}，适合中等规模模型</li>
                            <li>已启用FP16混合精度训练，可显著提升训练速度</li>
                            <li>梯度压缩已启用，可减少通信开销</li>
                            <li>建议根据显存使用情况调整批次大小</li>
                            {deepSpeedConfig.zero_stage < 3 && (
                              <li>如遇显存不足，可考虑升级到ZeRO-3</li>
                            )}
                          </ul>
                        }
                        type="success"
                        showIcon
                      />
                      
                      <Divider />
                      
                      <Title level={5}>性能指标</Title>
                      <Row gutter={8}>
                        <Col span={12}>
                          <Card size="small" bordered={false}>
                            <Statistic
                              title="通信效率"
                              value={89.5}
                              suffix="%"
                              valueStyle={{ color: '#52c41a', fontSize: 20 }}
                            />
                          </Card>
                        </Col>
                        <Col span={12}>
                          <Card size="small" bordered={false}>
                            <Statistic
                              title="显存节省"
                              value={35}
                              suffix="%"
                              valueStyle={{ color: '#1890ff', fontSize: 20 }}
                            />
                          </Card>
                        </Col>
                      </Row>
                    </Card>
                  </Col>
                </Row>
              </Card>
            ) : (
              <Alert
                message="请选择一个训练任务查看DeepSpeed配置"
                type="info"
                showIcon
              />
            )}
          </TabPane>

          <TabPane tab="性能监控" key="performance">
            {selectedJob && trainingMetrics.length > 0 ? (
              <Row gutter={[16, 16]}>
                <Col span={24}>
                  <Card title="训练性能指标">
                    <Row gutter={16}>
                      <Col span={8}>
                        <Card size="small">
                          <Statistic
                            title="吞吐量"
                            value={trainingMetrics[trainingMetrics.length - 1]?.throughput_samples_per_sec || 0}
                            suffix="samples/s"
                            prefix={<ThunderboltOutlined />}
                          />
                        </Card>
                      </Col>
                      <Col span={8}>
                        <Card size="small">
                          <Statistic
                            title="网络带宽"
                            value={trainingMetrics[trainingMetrics.length - 1]?.network_bandwidth_mbps || 0}
                            suffix="Mbps"
                            prefix={<CloudServerOutlined />}
                          />
                        </Card>
                      </Col>
                      <Col span={8}>
                        <Card size="small">
                          <Statistic
                            title="同步时间"
                            value={trainingMetrics[trainingMetrics.length - 1]?.sync_time_ms || 0}
                            suffix="ms"
                            prefix={<SyncOutlined />}
                          />
                        </Card>
                      </Col>
                    </Row>
                  </Card>
                </Col>
                <Col span={24}>
                  <Card title="GPU使用率历史">
                    <Line
                      data={trainingMetrics.map(m => ({
                        time: new Date(m.timestamp).toLocaleTimeString(),
                        gpu0: m.gpu_utilization_percent[0] || 0,
                        gpu1: m.gpu_utilization_percent[1] || 0,
                        gpu2: m.gpu_utilization_percent[2] || 0,
                        gpu3: m.gpu_utilization_percent[3] || 0
                      }))}
                      xField="time"
                      yField={['gpu0', 'gpu1', 'gpu2', 'gpu3']}
                      smooth
                      height={300}
                      yAxis={{
                        title: { text: 'GPU使用率 (%)' },
                        max: 100
                      }}
                    />
                  </Card>
                </Col>
              </Row>
            ) : (
              <Alert
                message="请选择一个运行中的任务查看性能监控"
                type="info"
                showIcon
              />
            )}
          </TabPane>
        </Tabs>
      </Card>

      {/* DeepSpeed配置修改弹窗 */}
      <Modal
        title="修改DeepSpeed配置"
        visible={configModalVisible}
        onCancel={() => setConfigModalVisible(false)}
        footer={null}
        width={600}
      >
        <Alert
          message="修改配置将在下次训练启动时生效"
          type="warning"
          showIcon
          style={{ marginBottom: 16 }}
        />
        <Form
          layout="vertical"
          initialValues={deepSpeedConfig}
          onFinish={async (values) => {
            try {
              if (selectedJob) {
                await distributedTrainingService.updateDeepSpeedConfig(selectedJob.job_id, values);
                message.success('配置更新成功');
                setConfigModalVisible(false);
                setDeepSpeedConfig(values);
              }
            } catch (error) {
              message.error('配置更新失败');
            }
          }}
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="zero_stage" label="ZeRO Stage">
                <Select>
                  <Select.Option value={0}>ZeRO-0 (禁用)</Select.Option>
                  <Select.Option value={1}>ZeRO-1 (优化器状态分片)</Select.Option>
                  <Select.Option value={2}>ZeRO-2 (优化器+梯度分片)</Select.Option>
                  <Select.Option value={3}>ZeRO-3 (全部分片)</Select.Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="communication_backend" label="通信后端">
                <Select>
                  <Select.Option value="nccl">NCCL</Select.Option>
                  <Select.Option value="gloo">Gloo</Select.Option>
                  <Select.Option value="mpi">MPI</Select.Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
          <Row gutter={16}>
            <Col span={8}>
              <Form.Item name="offload_optimizer" label="优化器卸载" valuePropName="checked">
                <Switch />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="offload_params" label="参数卸载" valuePropName="checked">
                <Switch />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="gradient_compression" label="梯度压缩" valuePropName="checked">
                <Switch />
              </Form.Item>
            </Col>
          </Row>
          <Row gutter={16}>
            <Col span={8}>
              <Form.Item name="fp16_enabled" label="FP16精度" valuePropName="checked">
                <Switch />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="bf16_enabled" label="BF16精度" valuePropName="checked">
                <Switch />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="gradient_accumulation_steps" label="梯度累积步数">
                <InputNumber min={1} max={32} />
              </Form.Item>
            </Col>
          </Row>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="train_batch_size" label="总批次大小">
                <InputNumber min={1} max={512} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="train_micro_batch_size_per_gpu" label="单GPU批次大小">
                <InputNumber min={1} max={128} />
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
    </div>
  );
};

export default DistributedTrainingPage;