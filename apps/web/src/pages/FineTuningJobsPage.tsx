import React, { useState, useEffect } from 'react'
import { logger } from '../utils/logger'
import {
  Card,
  Table,
  Button,
  Tag,
  Space,
  Progress,
  Typography,
  Row,
  Col,
  Statistic,
  Modal,
  Form,
  Select,
  AutoComplete,
  Input,
  Upload,
  message,
  Tooltip,
  Popconfirm,
  Divider,
} from 'antd'
import {
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  DeleteOutlined,
  EyeOutlined,
  DownloadOutlined,
  PlusOutlined,
  ReloadOutlined,
  FilterOutlined,
  UploadOutlined,
  SettingOutlined,
  MonitorOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  CheckCircleOutlined,
} from '@ant-design/icons'
import {
  fineTuningService,
  TrainingJob,
  TrainingJobRequest,
  ModelInfo,
  Dataset,
} from '../services/fineTuningService'

const { Title, Text } = Typography
const { Option } = Select

// 类型定义
interface JobWithExtras extends TrainingJob {
  model?: string
  type?: string
  dataset?: string
  learningRate?: string
  batchSize?: number
  loraRank?: number
  loraAlpha?: number
  gpuMemory?: string
  gpuCount?: number
  estimatedEndTime?: string
  endTime?: string
  errorTime?: string
  quantization?: string
  scheduledTime?: string
  error?: string
}

const FineTuningJobsPage: React.FC = () => {
  const [jobs, setJobs] = useState<JobWithExtras[]>([])
  const [loading, setLoading] = useState(false)
  const [selectedJob, setSelectedJob] = useState<JobWithExtras | null>(null)
  const [showJobModal, setShowJobModal] = useState(false)
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [filterStatus, setFilterStatus] = useState<string>('all')
  const [supportedModels, setSupportedModels] = useState<ModelInfo | null>(null)
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [jobLogs, setJobLogs] = useState<string[]>([])
  const [jobMetrics, setJobMetrics] = useState<any>(null)
  const [form] = Form.useForm()

  // 加载任务列表
  const loadJobs = async () => {
    try {
      setLoading(true)
      const jobData = await fineTuningService.getTrainingJobs()
      // 处理API返回的数据结构，支持 {jobs: [...]} 和直接数组格式
      const jobsArray = Array.isArray(jobData) ? jobData : jobData.jobs || []
      setJobs(jobsArray)
    } catch (error) {
      logger.error('加载任务列表失败:', error)
      message.error('加载任务列表失败')
    } finally {
      setLoading(false)
    }
  }

  // 加载支持的模型
  const loadSupportedModels = async () => {
    try {
      const models = await fineTuningService.getSupportedModels()
      setSupportedModels(models)
    } catch (error) {
      logger.error('加载支持的模型失败:', error)
    }
  }

  // 加载数据集
  const loadDatasets = async () => {
    try {
      const datasetsData = await fineTuningService.getDatasets()
      setDatasets(datasetsData.datasets)
    } catch (error) {
      logger.error('加载数据集失败:', error)
    }
  }

  // 加载任务日志
  const loadJobLogs = async (jobId: string) => {
    try {
      const logs = await fineTuningService.getTrainingLogs(jobId)
      setJobLogs(logs.logs)
    } catch (error) {
      logger.error('加载任务日志失败:', error)
    }
  }

  // 加载任务指标
  const loadJobMetrics = async (jobId: string) => {
    try {
      const metrics = await fineTuningService.getTrainingMetrics(jobId)
      setJobMetrics(metrics.metrics)
    } catch (error) {
      logger.error('加载任务指标失败:', error)
    }
  }

  // 刷新任务列表
  const refreshJobs = () => {
    loadJobs()
    message.success('正在刷新任务列表...')
  }

  // 获取状态标签
  const getStatusTag = (status: string) => {
    const statusConfig = {
      running: { color: 'processing', text: '运行中' },
      completed: { color: 'success', text: '已完成' },
      failed: { color: 'error', text: '失败' },
      pending: { color: 'default', text: '等待中' },
      paused: { color: 'warning', text: '已暂停' },
    }
    const config = statusConfig[status as keyof typeof statusConfig] || {
      color: 'default',
      text: status,
    }
    return <Tag color={config.color}>{config.text}</Tag>
  }

  // 获取训练类型标签
  const getTypeTag = (type: string) => {
    const typeConfig = {
      LoRA: { color: 'gold', icon: <ThunderboltOutlined /> },
      QLoRA: { color: 'cyan', icon: <DatabaseOutlined /> },
      'Distributed LoRA': { color: 'purple', icon: <MonitorOutlined /> },
    }
    const config = typeConfig[type as keyof typeof typeConfig] || {
      color: 'default',
      icon: null,
    }
    return (
      <Tag color={config.color} icon={config.icon}>
        {type}
      </Tag>
    )
  }

  // 任务操作
  const handleJobAction = async (action: string, jobId: string) => {
    const job = jobs.find(j => j.job_id === jobId)
    try {
      switch (action) {
        case '暂停':
          await fineTuningService.pauseTrainingJob(jobId)
          break
        case '继续':
          await fineTuningService.resumeTrainingJob(jobId)
          break
        case '停止':
          await fineTuningService.cancelTrainingJob(jobId)
          break
        case '下载': {
          const a = document.createElement('a')
          a.href = `/api/v1/fine-tuning/jobs/${jobId}/download`
          a.rel = 'noopener'
          document.body.appendChild(a)
          a.click()
          a.remove()
          break
        }
        case '删除':
          await fineTuningService.deleteTrainingJob(jobId)
          break
        default:
          break
      }
      message.success(`${action} 任务: ${job?.job_name}`)
    } catch (error) {
      logger.error(`${action}任务失败:`, error)
      const detail = (error as any)?.response?.data?.detail
      message.error(
        detail ? `${action}任务失败: ${detail}` : `${action}任务失败`
      )
    } finally {
      await loadJobs()
    }
  }

  // 创建新任务
  const handleCreateJob = async (values: any) => {
    try {
      const jobRequest: TrainingJobRequest = {
        job_name: values.jobName,
        model_name: values.model,
        training_mode: values.trainingType,
        dataset_path: values.dataset,
        learning_rate: parseFloat(values.learningRate),
        num_train_epochs: parseInt(values.epochs),
        per_device_train_batch_size: parseInt(values.batchSize),
        gradient_accumulation_steps: 1,
        warmup_steps: 0,
        max_seq_length: 2048,
        lora_config: {
          rank: parseInt(values.loraRank),
          alpha: parseInt(values.loraAlpha),
          dropout: 0.1,
          bias: 'none',
        },
        use_distributed: values.trainingType === 'distributed',
        use_deepspeed: false,
        use_flash_attention: false,
        use_gradient_checkpointing: true,
        fp16: true,
        bf16: false,
      }

      await fineTuningService.createTrainingJob(jobRequest)
      message.success('任务创建成功')
      setShowCreateModal(false)
      form.resetFields()
      await loadJobs()
    } catch (error) {
      logger.error('创建任务失败:', error)
      message.error('创建任务失败')
    }
  }

  // 表格列定义
  const columns = [
    {
      title: '任务信息',
      key: 'info',
      width: 300,
      render: (_, record: any) => (
        <div>
          <div style={{ fontWeight: 'bold', marginBottom: 4 }}>
            {record.job_name}
          </div>
          <div style={{ color: '#666', fontSize: '12px' }}>
            {record.model || 'N/A'}
          </div>
          <div style={{ marginTop: 4 }}>
            {record.type && getTypeTag(record.type)}
            {getStatusTag(record.status)}
          </div>
        </div>
      ),
    },
    {
      title: '训练进度',
      key: 'progress',
      width: 200,
      render: (_, record: any) => (
        <div>
          <Progress
            percent={record.progress}
            size="small"
            status={record.status === 'failed' ? 'exception' : 'normal'}
          />
          <div style={{ fontSize: '12px', color: '#666', marginTop: 4 }}>
            Epoch {record.current_epoch}/{record.total_epochs}
          </div>
          {record.current_loss && (
            <div style={{ fontSize: '12px', color: '#666' }}>
              Loss: {record.current_loss.toFixed(4)}
            </div>
          )}
        </div>
      ),
    },
    {
      title: '资源使用',
      key: 'resources',
      width: 150,
      render: (_, record: any) => (
        <div style={{ fontSize: '12px' }}>
          {record.gpuMemory && <div>GPU: {record.gpuMemory}</div>}
          {record.gpuCount && <div>GPU数量: {record.gpuCount}</div>}
          <div>批次大小: {record.batchSize}</div>
          <div>学习率: {record.learningRate}</div>
        </div>
      ),
    },
    {
      title: '时间信息',
      key: 'time',
      width: 150,
      render: (_, record: any) => (
        <div style={{ fontSize: '12px' }}>
          <div>
            开始:{' '}
            {record.started_at?.slice(11, 16) ||
              record.created_at?.slice(11, 16)}
          </div>
          {record.estimatedEndTime && (
            <div>预计结束: {record.estimatedEndTime.slice(11, 16)}</div>
          )}
          {record.completed_at && (
            <div>结束: {record.completed_at.slice(11, 16)}</div>
          )}
          {record.error_message && (
            <div style={{ color: '#ff4d4f' }}>
              错误时间: {record.created_at?.slice(11, 16)}
            </div>
          )}
        </div>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      width: 200,
      render: (_, record: any) => (
        <Space size="small">
          <Tooltip title="查看详情">
            <Button
              type="text"
              icon={<EyeOutlined />}
              onClick={async () => {
                setSelectedJob(record)
                setShowJobModal(true)
                await loadJobLogs(record.job_id)
                await loadJobMetrics(record.job_id)
              }}
            />
          </Tooltip>

          {record.status === 'running' && (
            <>
              <Tooltip title="暂停训练">
                <Button
                  type="text"
                  icon={<PauseCircleOutlined />}
                  onClick={() => handleJobAction('暂停', record.job_id)}
                />
              </Tooltip>
              <Tooltip title="停止训练">
                <Popconfirm
                  title="确定停止训练吗？"
                  onConfirm={() => handleJobAction('停止', record.job_id)}
                >
                  <Button type="text" icon={<StopOutlined />} danger />
                </Popconfirm>
              </Tooltip>
            </>
          )}

          {record.status === 'paused' && (
            <Tooltip title="继续训练">
              <Button
                type="text"
                icon={<PlayCircleOutlined />}
                onClick={() => handleJobAction('继续', record.job_id)}
              />
            </Tooltip>
          )}

          {record.status === 'completed' && (
            <Tooltip title="下载模型">
              <Button
                type="text"
                icon={<DownloadOutlined />}
                onClick={() => handleJobAction('下载', record.job_id)}
              />
            </Tooltip>
          )}

          <Tooltip title="删除任务">
            <Popconfirm
              title="确定删除这个任务吗？"
              onConfirm={() => handleJobAction('删除', record.job_id)}
            >
              <Button type="text" icon={<DeleteOutlined />} danger />
            </Popconfirm>
          </Tooltip>
        </Space>
      ),
    },
  ]

  // 过滤后的任务数据
  const filteredJobs =
    filterStatus === 'all'
      ? jobs
      : jobs.filter(job => job.status === filterStatus)

  // 统计数据
  const stats = {
    total: jobs.length,
    running: jobs.filter(j => j.status === 'running').length,
    completed: jobs.filter(j => j.status === 'completed').length,
    failed: jobs.filter(j => j.status === 'failed').length,
  }

  // 组件初始化
  useEffect(() => {
    loadJobs()
    loadSupportedModels()
    loadDatasets()
  }, [])

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>微调任务管理</Title>
        <Text type="secondary">
          管理和监控LoRA/QLoRA微调任务的执行状态，支持任务创建、监控、控制和结果下载
        </Text>
      </div>

      {/* 统计卡片 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="总任务数"
              value={stats.total}
              prefix={<DatabaseOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="运行中"
              value={stats.running}
              valueStyle={{ color: '#1890ff' }}
              prefix={<PlayCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="已完成"
              value={stats.completed}
              valueStyle={{ color: '#52c41a' }}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="失败任务"
              value={stats.failed}
              valueStyle={{ color: '#ff4d4f' }}
              prefix={<StopOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* 任务控制面板 */}
      <Card style={{ marginBottom: '16px' }}>
        <Row justify="space-between" align="middle">
          <Col>
            <Space>
              <Button
                type="primary"
                icon={<PlusOutlined />}
                onClick={() => setShowCreateModal(true)}
              >
                创建新任务
              </Button>
              <Button
                icon={<ReloadOutlined />}
                onClick={refreshJobs}
                loading={loading}
              >
                刷新
              </Button>
              <Select
                value={filterStatus}
                onChange={setFilterStatus}
                style={{ width: 120 }}
                prefix={<FilterOutlined />}
                name="jobStatusFilter"
              >
                <Option value="all">全部状态</Option>
                <Option value="running">运行中</Option>
                <Option value="completed">已完成</Option>
                <Option value="failed">失败</Option>
                <Option value="pending">等待中</Option>
              </Select>
            </Space>
          </Col>
          <Col>
            <Space>
              <Button icon={<SettingOutlined />}>批量操作</Button>
              <Button icon={<MonitorOutlined />}>系统监控</Button>
            </Space>
          </Col>
        </Row>
      </Card>

      {/* 任务列表 */}
      <Card>
        <Table
          columns={columns}
          dataSource={filteredJobs}
          rowKey="job_id"
          loading={loading}
          pagination={{
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: total => `共 ${total} 个任务`,
          }}
          scroll={{ x: 1000 }}
        />
      </Card>

      {/* 任务详情模态框 */}
      <Modal
        title="任务详情"
        open={showJobModal}
        onCancel={() => setShowJobModal(false)}
        footer={null}
        width={800}
      >
        {selectedJob && (
          <div>
            <Row gutter={16}>
              <Col span={12}>
                <Card title="基本信息" size="small">
                  <div>
                    <strong>任务名称：</strong>
                    {selectedJob.job_name}
                  </div>
                  <div>
                    <strong>模型：</strong>
                    {selectedJob.model || 'N/A'}
                  </div>
                  <div>
                    <strong>类型：</strong>
                    {selectedJob.type && getTypeTag(selectedJob.type)}
                  </div>
                  <div>
                    <strong>状态：</strong>
                    {getStatusTag(selectedJob.status)}
                  </div>
                  <div>
                    <strong>数据集：</strong>
                    {selectedJob.dataset || 'N/A'}
                  </div>
                </Card>
              </Col>
              <Col span={12}>
                <Card title="训练参数" size="small">
                  <div>
                    <strong>LoRA Rank：</strong>
                    {selectedJob.loraRank}
                  </div>
                  <div>
                    <strong>LoRA Alpha：</strong>
                    {selectedJob.loraAlpha}
                  </div>
                  <div>
                    <strong>学习率：</strong>
                    {selectedJob.learningRate}
                  </div>
                  <div>
                    <strong>批次大小：</strong>
                    {selectedJob.batchSize}
                  </div>
                  {selectedJob.quantization && (
                    <div>
                      <strong>量化：</strong>
                      {selectedJob.quantization}
                    </div>
                  )}
                </Card>
              </Col>
            </Row>
            <Card title="训练指标" size="small" style={{ marginTop: 16 }}>
              <Row gutter={16}>
                <Col span={8}>
                  <Statistic
                    title="当前Epoch"
                    value={`${selectedJob.current_epoch}/${selectedJob.total_epochs}`}
                  />
                </Col>
                <Col span={8}>
                  <Statistic
                    title="当前Loss"
                    value={selectedJob.current_loss?.toFixed(4) || 'N/A'}
                  />
                </Col>
                <Col span={8}>
                  <Statistic
                    title="最佳Loss"
                    value={selectedJob.best_loss?.toFixed(4) || 'N/A'}
                  />
                </Col>
              </Row>
            </Card>
            {selectedJob.error_message && (
              <Card title="错误信息" size="small" style={{ marginTop: 16 }}>
                <Text type="danger">{selectedJob.error_message}</Text>
              </Card>
            )}
            {jobLogs.length > 0 && (
              <Card title="训练日志" size="small" style={{ marginTop: 16 }}>
                <div
                  style={{
                    maxHeight: 200,
                    overflow: 'auto',
                    background: '#f5f5f5',
                    padding: 8,
                    fontSize: '12px',
                    fontFamily: 'monospace',
                  }}
                >
                  {jobLogs.map((log, index) => (
                    <div key={index}>{log}</div>
                  ))}
                </div>
              </Card>
            )}
            {jobMetrics && (
              <Card title="训练指标" size="small" style={{ marginTop: 16 }}>
                <pre style={{ fontSize: '12px' }}>
                  {JSON.stringify(jobMetrics, null, 2)}
                </pre>
              </Card>
            )}
          </div>
        )}
      </Modal>

      {/* 创建任务模态框 */}
      <Modal
        title="创建微调任务"
        open={showCreateModal}
        onCancel={() => setShowCreateModal(false)}
        width={800}
        footer={null}
      >
        <Form form={form} layout="vertical" onFinish={handleCreateJob}>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                label="任务名称"
                name="jobName"
                required
                rules={[{ required: true, message: '请输入任务名称' }]}
              >
                <Input placeholder="输入任务名称" name="jobName" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                label="训练类型"
                name="trainingType"
                required
                rules={[{ required: true, message: '请选择训练类型' }]}
              >
                <Select placeholder="选择训练类型" name="trainingType">
                  <Option value="lora">LoRA</Option>
                  <Option value="qlora">QLoRA</Option>
                  <Option value="distributed">分布式LoRA</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            label="基础模型"
            name="model"
            required
            rules={[{ required: true, message: '请输入基础模型' }]}
          >
            <AutoComplete
              placeholder="输入或选择基础模型"
              options={supportedModels?.models?.flatMap(group =>
                group.models.map(value => ({ value }))
              )}
              filterOption={(inputValue, option) =>
                (option?.value ?? '')
                  .toLowerCase()
                  .includes(inputValue.toLowerCase())
              }
              name="model"
            />
          </Form.Item>

          <Form.Item
            label="训练数据集"
            name="dataset"
            required
            rules={[{ required: true, message: '请选择数据集' }]}
          >
            <Select placeholder="选择数据集" name="dataset">
              {datasets.map(dataset => (
                <Option key={dataset.filename} value={dataset.path}>
                  {dataset.filename}
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Row gutter={16}>
            <Col span={8}>
              <Form.Item label="LoRA Rank" name="loraRank" initialValue="16">
                <Select name="loraRank">
                  <Option value="8">8</Option>
                  <Option value="16">16</Option>
                  <Option value="32">32</Option>
                  <Option value="64">64</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item label="LoRA Alpha" name="loraAlpha" initialValue="32">
                <Select name="loraAlpha">
                  <Option value="16">16</Option>
                  <Option value="32">32</Option>
                  <Option value="64">64</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item label="学习率" name="learningRate" initialValue="2e-4">
                <Select name="learningRate">
                  <Option value="1e-4">1e-4</Option>
                  <Option value="2e-4">2e-4</Option>
                  <Option value="3e-4">3e-4</Option>
                  <Option value="5e-4">5e-4</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item label="训练轮数" name="epochs" initialValue="3">
                <Select name="epochs">
                  <Option value="1">1</Option>
                  <Option value="3">3</Option>
                  <Option value="5">5</Option>
                  <Option value="10">10</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="批次大小" name="batchSize" initialValue="4">
                <Select name="batchSize">
                  <Option value="2">2</Option>
                  <Option value="4">4</Option>
                  <Option value="8">8</Option>
                  <Option value="16">16</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Divider />
          <Space>
            <Button type="primary" htmlType="submit" loading={loading}>
              创建任务
            </Button>
            <Button onClick={() => setShowCreateModal(false)}>取消</Button>
          </Space>
        </Form>
      </Modal>
    </div>
  )
}

export default FineTuningJobsPage
