import React, { useState, useEffect } from 'react';
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
  Input,
  Upload,
  message,
  Tooltip,
  Popconfirm,
  Divider
} from 'antd';
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
  CheckCircleOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { Option } = Select;

// 模拟训练任务数据
const generateMockJobs = () => [
  {
    id: '1',
    name: 'LLaMA-7B LoRA微调 - 对话数据',
    model: 'meta-llama/Llama-2-7b-chat-hf',
    type: 'LoRA',
    status: 'running',
    progress: 65,
    currentEpoch: 2,
    totalEpochs: 3,
    currentLoss: 0.8423,
    bestLoss: 0.7856,
    startTime: '2025-08-23 14:30:00',
    estimatedEndTime: '2025-08-23 16:45:00',
    dataset: 'conversation_data_zh',
    gpuMemory: '12.5 GB / 24.0 GB',
    learningRate: '2e-4',
    batchSize: 4,
    loraRank: 16,
    loraAlpha: 32
  },
  {
    id: '2', 
    name: 'Mistral-7B QLoRA微调 - 代码生成',
    model: 'mistralai/Mistral-7B-Instruct-v0.1',
    type: 'QLoRA',
    status: 'completed',
    progress: 100,
    currentEpoch: 3,
    totalEpochs: 3,
    currentLoss: 0.6234,
    bestLoss: 0.6234,
    startTime: '2025-08-23 10:15:00',
    endTime: '2025-08-23 13:22:00',
    dataset: 'code_generation_data',
    gpuMemory: '8.2 GB / 24.0 GB',
    learningRate: '1e-4',
    batchSize: 8,
    loraRank: 8,
    loraAlpha: 16,
    quantization: '4-bit NF4'
  },
  {
    id: '3',
    name: 'Qwen-14B LoRA微调 - 文档问答',
    model: 'Qwen/Qwen-14B-Chat',
    type: 'LoRA',
    status: 'failed',
    progress: 23,
    currentEpoch: 1,
    totalEpochs: 5,
    currentLoss: 1.2456,
    bestLoss: 1.1892,
    startTime: '2025-08-23 09:00:00',
    errorTime: '2025-08-23 11:30:00',
    dataset: 'document_qa_data',
    error: 'CUDA out of memory error',
    gpuMemory: '22.8 GB / 24.0 GB',
    learningRate: '3e-4',
    batchSize: 2,
    loraRank: 32,
    loraAlpha: 64
  },
  {
    id: '4',
    name: 'ChatGLM3-6B 分布式微调',
    model: 'THUDM/chatglm3-6b',
    type: 'Distributed LoRA',
    status: 'pending',
    progress: 0,
    currentEpoch: 0,
    totalEpochs: 4,
    dataset: 'multi_domain_data',
    gpuCount: 4,
    learningRate: '1e-4',
    batchSize: 16,
    loraRank: 16,
    loraAlpha: 32,
    scheduledTime: '2025-08-23 18:00:00'
  }
];

const FineTuningJobsPage: React.FC = () => {
  const [jobs, setJobs] = useState(generateMockJobs());
  const [loading, setLoading] = useState(false);
  const [selectedJob, setSelectedJob] = useState<any>(null);
  const [showJobModal, setShowJobModal] = useState(false);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [filterStatus, setFilterStatus] = useState<string>('all');

  // 刷新任务列表
  const refreshJobs = () => {
    setLoading(true);
    setTimeout(() => {
      setJobs(generateMockJobs());
      setLoading(false);
      message.success('任务列表已刷新');
    }, 1000);
  };

  // 获取状态标签
  const getStatusTag = (status: string) => {
    const statusConfig = {
      running: { color: 'processing', text: '运行中' },
      completed: { color: 'success', text: '已完成' },
      failed: { color: 'error', text: '失败' },
      pending: { color: 'default', text: '等待中' },
      paused: { color: 'warning', text: '已暂停' }
    };
    const config = statusConfig[status as keyof typeof statusConfig] || { color: 'default', text: status };
    return <Tag color={config.color}>{config.text}</Tag>;
  };

  // 获取训练类型标签
  const getTypeTag = (type: string) => {
    const typeConfig = {
      'LoRA': { color: 'gold', icon: <ThunderboltOutlined /> },
      'QLoRA': { color: 'cyan', icon: <DatabaseOutlined /> },
      'Distributed LoRA': { color: 'purple', icon: <MonitorOutlined /> },
    };
    const config = typeConfig[type as keyof typeof typeConfig] || { color: 'default', icon: null };
    return (
      <Tag color={config.color} icon={config.icon}>
        {type}
      </Tag>
    );
  };

  // 任务操作
  const handleJobAction = (action: string, jobId: string) => {
    const job = jobs.find(j => j.id === jobId);
    message.success(`${action} 任务: ${job?.name}`);
  };

  // 表格列定义
  const columns = [
    {
      title: '任务信息',
      key: 'info',
      width: 300,
      render: (_, record: any) => (
        <div>
          <div style={{ fontWeight: 'bold', marginBottom: 4 }}>
            {record.name}
          </div>
          <div style={{ color: '#666', fontSize: '12px' }}>
            {record.model}
          </div>
          <div style={{ marginTop: 4 }}>
            {getTypeTag(record.type)}
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
            Epoch {record.currentEpoch}/{record.totalEpochs}
          </div>
          {record.currentLoss && (
            <div style={{ fontSize: '12px', color: '#666' }}>
              Loss: {record.currentLoss.toFixed(4)}
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
          {record.gpuMemory && (
            <div>GPU: {record.gpuMemory}</div>
          )}
          {record.gpuCount && (
            <div>GPU数量: {record.gpuCount}</div>
          )}
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
          <div>开始: {record.startTime?.slice(11, 16)}</div>
          {record.estimatedEndTime && (
            <div>预计结束: {record.estimatedEndTime.slice(11, 16)}</div>
          )}
          {record.endTime && (
            <div>结束: {record.endTime.slice(11, 16)}</div>
          )}
          {record.errorTime && (
            <div style={{ color: '#ff4d4f' }}>
              错误: {record.errorTime.slice(11, 16)}
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
              onClick={() => {
                setSelectedJob(record);
                setShowJobModal(true);
              }}
            />
          </Tooltip>
          
          {record.status === 'running' && (
            <>
              <Tooltip title="暂停训练">
                <Button 
                  type="text" 
                  icon={<PauseCircleOutlined />}
                  onClick={() => handleJobAction('暂停', record.id)}
                />
              </Tooltip>
              <Tooltip title="停止训练">
                <Popconfirm
                  title="确定停止训练吗？"
                  onConfirm={() => handleJobAction('停止', record.id)}
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
                onClick={() => handleJobAction('继续', record.id)}
              />
            </Tooltip>
          )}
          
          {record.status === 'completed' && (
            <Tooltip title="下载模型">
              <Button 
                type="text" 
                icon={<DownloadOutlined />}
                onClick={() => handleJobAction('下载', record.id)}
              />
            </Tooltip>
          )}
          
          <Tooltip title="删除任务">
            <Popconfirm
              title="确定删除这个任务吗？"
              onConfirm={() => handleJobAction('删除', record.id)}
            >
              <Button type="text" icon={<DeleteOutlined />} danger />
            </Popconfirm>
          </Tooltip>
        </Space>
      ),
    },
  ];

  // 过滤后的任务数据
  const filteredJobs = filterStatus === 'all' ? jobs : jobs.filter(job => job.status === filterStatus);

  // 统计数据
  const stats = {
    total: jobs.length,
    running: jobs.filter(j => j.status === 'running').length,
    completed: jobs.filter(j => j.status === 'completed').length,
    failed: jobs.filter(j => j.status === 'failed').length,
  };

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
          rowKey="id"
          loading={loading}
          pagination={{
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total) => `共 ${total} 个任务`,
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
                  <div><strong>任务名称：</strong>{selectedJob.name}</div>
                  <div><strong>模型：</strong>{selectedJob.model}</div>
                  <div><strong>类型：</strong>{getTypeTag(selectedJob.type)}</div>
                  <div><strong>状态：</strong>{getStatusTag(selectedJob.status)}</div>
                  <div><strong>数据集：</strong>{selectedJob.dataset}</div>
                </Card>
              </Col>
              <Col span={12}>
                <Card title="训练参数" size="small">
                  <div><strong>LoRA Rank：</strong>{selectedJob.loraRank}</div>
                  <div><strong>LoRA Alpha：</strong>{selectedJob.loraAlpha}</div>
                  <div><strong>学习率：</strong>{selectedJob.learningRate}</div>
                  <div><strong>批次大小：</strong>{selectedJob.batchSize}</div>
                  {selectedJob.quantization && (
                    <div><strong>量化：</strong>{selectedJob.quantization}</div>
                  )}
                </Card>
              </Col>
            </Row>
            <Card title="训练指标" size="small" style={{ marginTop: 16 }}>
              <Row gutter={16}>
                <Col span={8}>
                  <Statistic title="当前Epoch" value={`${selectedJob.currentEpoch}/${selectedJob.totalEpochs}`} />
                </Col>
                <Col span={8}>
                  <Statistic title="当前Loss" value={selectedJob.currentLoss?.toFixed(4) || 'N/A'} />
                </Col>
                <Col span={8}>
                  <Statistic title="最佳Loss" value={selectedJob.bestLoss?.toFixed(4) || 'N/A'} />
                </Col>
              </Row>
            </Card>
            {selectedJob.error && (
              <Card title="错误信息" size="small" style={{ marginTop: 16 }}>
                <Text type="danger">{selectedJob.error}</Text>
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
        <Form layout="vertical">
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item label="任务名称" required>
                <Input placeholder="输入任务名称" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="训练类型" required>
                <Select placeholder="选择训练类型">
                  <Option value="lora">LoRA</Option>
                  <Option value="qlora">QLoRA</Option>
                  <Option value="distributed">分布式LoRA</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
          
          <Form.Item label="基础模型" required>
            <Select placeholder="选择基础模型">
              <Option value="llama2-7b">LLaMA 2 7B</Option>
              <Option value="llama2-13b">LLaMA 2 13B</Option>
              <Option value="mistral-7b">Mistral 7B</Option>
              <Option value="qwen-14b">Qwen 14B</Option>
              <Option value="chatglm3-6b">ChatGLM3 6B</Option>
            </Select>
          </Form.Item>

          <Form.Item label="训练数据集" required>
            <Upload>
              <Button icon={<UploadOutlined />}>上传数据集</Button>
            </Upload>
          </Form.Item>

          <Row gutter={16}>
            <Col span={8}>
              <Form.Item label="LoRA Rank">
                <Select defaultValue="16">
                  <Option value="8">8</Option>
                  <Option value="16">16</Option>
                  <Option value="32">32</Option>
                  <Option value="64">64</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item label="LoRA Alpha">
                <Select defaultValue="32">
                  <Option value="16">16</Option>
                  <Option value="32">32</Option>
                  <Option value="64">64</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item label="学习率">
                <Select defaultValue="2e-4">
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
              <Form.Item label="训练轮数">
                <Select defaultValue="3">
                  <Option value="1">1</Option>
                  <Option value="3">3</Option>
                  <Option value="5">5</Option>
                  <Option value="10">10</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="批次大小">
                <Select defaultValue="4">
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
            <Button type="primary">创建任务</Button>
            <Button onClick={() => setShowCreateModal(false)}>取消</Button>
          </Space>
        </Form>
      </Modal>
    </div>
  );
};

export default FineTuningJobsPage;