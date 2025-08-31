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
  Tabs,
  Tag,
  Alert,
  Space,
  Modal,
  message,
  Statistic,
  Progress,
  Timeline,
  Descriptions,
  Typography,
  Divider
} from 'antd';
import {
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  PlusOutlined,
  DeleteOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  WarningOutlined,
  SettingOutlined
} from '@ant-design/icons';
import { distributedTaskService, TaskSubmitRequest, SystemStats, ConflictInfo } from '../services/distributedTaskService';

const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;
const { TextArea } = Input;
const { confirm } = Modal;

interface TaskInfo {
  task_id: string;
  status: string;
  assigned_to?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  result?: any;
  error?: string;
}

const DistributedTaskCoordinationPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [engineInitialized, setEngineInitialized] = useState(false);
  const [systemStats, setSystemStats] = useState<SystemStats | null>(null);
  const [tasks, setTasks] = useState<TaskInfo[]>([]);
  const [conflicts, setConflicts] = useState<ConflictInfo[]>([]);
  const [taskForm] = Form.useForm();
  const [initForm] = Form.useForm();

  // 任务优先级映射
  const priorityColors = {
    critical: 'red',
    high: 'orange',
    medium: 'blue',
    low: 'green',
    background: 'gray'
  };

  // 任务状态映射
  const statusColors = {
    pending: 'default',
    assigned: 'processing',
    running: 'processing',
    completed: 'success',
    failed: 'error',
    cancelled: 'warning',
    decomposed: 'cyan'
  };

  // Raft状态映射
  const raftStateColors = {
    leader: 'success',
    follower: 'processing',
    candidate: 'warning'
  };

  // 获取系统统计
  const fetchSystemStats = async () => {
    try {
      const stats = await distributedTaskService.getSystemStats();
      setSystemStats(stats);
      setEngineInitialized(true);
    } catch (error) {
      console.error('Failed to fetch system stats:', error);
      setEngineInitialized(false);
    }
  };

  // 获取任务列表
  const fetchTasks = async () => {
    if (!systemStats) return;
    
    const taskIds = Object.keys(systemStats.stats);
    const taskPromises = taskIds.map(async (taskId) => {
      try {
        return await distributedTaskService.getTaskStatus(taskId);
      } catch (error) {
        console.error(`Failed to fetch task ${taskId}:`, error);
        return null;
      }
    });
    
    const taskResults = await Promise.all(taskPromises);
    setTasks(taskResults.filter(Boolean));
  };

  // 获取冲突列表
  const fetchConflicts = async () => {
    try {
      const conflictList = await distributedTaskService.detectConflicts();
      setConflicts(conflictList);
    } catch (error) {
      console.error('Failed to fetch conflicts:', error);
    }
  };

  // 初始化引擎
  const handleInitializeEngine = async (values: any) => {
    setLoading(true);
    try {
      await distributedTaskService.initializeEngine(values.nodeId, values.clusterNodes.split(',').map((s: string) => s.trim()));
      message.success('引擎初始化成功');
      await fetchSystemStats();
    } catch (error) {
      message.error(`引擎初始化失败: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  // 提交任务
  const handleSubmitTask = async (values: any) => {
    setLoading(true);
    try {
      const request: TaskSubmitRequest = {
        task_type: values.taskType,
        task_data: JSON.parse(values.taskData || '{}'),
        requirements: values.requirements ? JSON.parse(values.requirements) : undefined,
        priority: values.priority,
        decomposition_strategy: values.decompositionStrategy,
        assignment_strategy: values.assignmentStrategy
      };
      
      const result = await distributedTaskService.submitTask(request);
      message.success(`任务提交成功: ${result.task_id}`);
      taskForm.resetFields();
      await fetchSystemStats();
      await fetchTasks();
    } catch (error) {
      message.error(`任务提交失败: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  // 取消任务
  const handleCancelTask = (taskId: string) => {
    confirm({
      title: '确认取消任务',
      icon: <ExclamationCircleOutlined />,
      content: `确定要取消任务 ${taskId} 吗？`,
      onOk: async () => {
        try {
          await distributedTaskService.cancelTask(taskId);
          message.success('任务取消成功');
          await fetchTasks();
          await fetchSystemStats();
        } catch (error) {
          message.error(`任务取消失败: ${error}`);
        }
      }
    });
  };

  // 解决冲突
  const handleResolveConflict = (conflictId: string, strategy: string = 'priority_based') => {
    confirm({
      title: '确认解决冲突',
      icon: <ExclamationCircleOutlined />,
      content: `确定要使用 ${strategy} 策略解决冲突 ${conflictId} 吗？`,
      onOk: async () => {
        try {
          await distributedTaskService.resolveConflict(conflictId, strategy);
          message.success('冲突解决成功');
          await fetchConflicts();
        } catch (error) {
          message.error(`冲突解决失败: ${error}`);
        }
      }
    });
  };

  // 创建检查点
  const handleCreateCheckpoint = () => {
    Modal.prompt({
      title: '创建检查点',
      content: '请输入检查点名称：',
      onOk: async (name: string) => {
        try {
          await distributedTaskService.createCheckpoint(name);
          message.success(`检查点 ${name} 创建成功`);
        } catch (error) {
          message.error(`检查点创建失败: ${error}`);
        }
      }
    });
  };

  // 回滚检查点
  const handleRollbackCheckpoint = () => {
    Modal.prompt({
      title: '回滚检查点',
      content: '请输入要回滚的检查点名称：',
      onOk: async (name: string) => {
        confirm({
          title: '确认回滚',
          icon: <ExclamationCircleOutlined />,
          content: `确定要回滚到检查点 ${name} 吗？这将撤销之后的所有状态更改。`,
          onOk: async () => {
            try {
              await distributedTaskService.rollbackCheckpoint(name);
              message.success(`回滚到检查点 ${name} 成功`);
              await fetchSystemStats();
            } catch (error) {
              message.error(`检查点回滚失败: ${error}`);
            }
          }
        });
      }
    });
  };

  // 关闭引擎
  const handleShutdownEngine = () => {
    confirm({
      title: '确认关闭引擎',
      icon: <ExclamationCircleOutlined />,
      content: '确定要关闭分布式任务协调引擎吗？这将停止所有任务处理。',
      onOk: async () => {
        try {
          await distributedTaskService.shutdownEngine();
          message.success('引擎关闭成功');
          setEngineInitialized(false);
          setSystemStats(null);
          setTasks([]);
          setConflicts([]);
        } catch (error) {
          message.error(`引擎关闭失败: ${error}`);
        }
      }
    });
  };

  // 定期刷新数据
  useEffect(() => {
    if (engineInitialized) {
      const interval = setInterval(async () => {
        await fetchSystemStats();
        await fetchTasks();
        await fetchConflicts();
      }, 5000);
      
      return () => clearInterval(interval);
    }
  }, [engineInitialized]);

  // 初始化时检查引擎状态
  useEffect(() => {
    fetchSystemStats();
  }, []);

  // 任务表格列定义
  const taskColumns = [
    {
      title: '任务ID',
      dataIndex: 'task_id',
      key: 'task_id',
      render: (taskId: string) => (
        <Text code copyable={{ text: taskId }}>
          {taskId.substring(0, 8)}...
        </Text>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={statusColors[status as keyof typeof statusColors] || 'default'}>
          {status.toUpperCase()}
        </Tag>
      )
    },
    {
      title: '分配给',
      dataIndex: 'assigned_to',
      key: 'assigned_to',
      render: (agentId: string) => agentId ? <Tag>{agentId}</Tag> : '-'
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (time: string) => new Date(time).toLocaleString()
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record: TaskInfo) => (
        <Space>
          <Button
            size="small"
            icon={<DeleteOutlined />}
            danger
            onClick={() => handleCancelTask(record.task_id)}
            disabled={['completed', 'failed', 'cancelled'].includes(record.status)}
          >
            取消
          </Button>
        </Space>
      )
    }
  ];

  // 冲突表格列定义
  const conflictColumns = [
    {
      title: '冲突ID',
      dataIndex: 'conflict_id',
      key: 'conflict_id',
      render: (conflictId: string) => (
        <Text code copyable={{ text: conflictId }}>
          {conflictId.substring(0, 8)}...
        </Text>
      )
    },
    {
      title: '类型',
      dataIndex: 'conflict_type',
      key: 'conflict_type',
      render: (type: string) => <Tag color="orange">{type}</Tag>
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description'
    },
    {
      title: '涉及任务',
      dataIndex: 'involved_tasks',
      key: 'involved_tasks',
      render: (tasks: string[]) => tasks.length
    },
    {
      title: '状态',
      dataIndex: 'resolved',
      key: 'resolved',
      render: (resolved: boolean) => (
        <Tag color={resolved ? 'success' : 'error'}>
          {resolved ? '已解决' : '待解决'}
        </Tag>
      )
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record: ConflictInfo) => (
        <Space>
          <Button
            size="small"
            type="primary"
            onClick={() => handleResolveConflict(record.conflict_id, 'priority_based')}
            disabled={record.resolved}
          >
            优先级解决
          </Button>
          <Button
            size="small"
            onClick={() => handleResolveConflict(record.conflict_id, 'load_balancing')}
            disabled={record.resolved}
          >
            负载均衡解决
          </Button>
        </Space>
      )
    }
  ];

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <Title level={2}>分布式任务协调引擎</Title>
      <Paragraph>
        基于Raft共识算法的分布式任务协调系统，支持任务分解、智能分配、状态同步和冲突解决。
      </Paragraph>

      {!engineInitialized ? (
        <Card title="引擎初始化" style={{ marginBottom: 24 }}>
          <Alert
            message="引擎未初始化"
            description="请配置节点信息并启动分布式任务协调引擎。"
            type="warning"
            showIcon
            style={{ marginBottom: 16 }}
          />
          <Form
            form={initForm}
            layout="vertical"
            onFinish={handleInitializeEngine}
          >
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item
                  label="节点ID"
                  name="nodeId"
                  rules={[{ required: true, message: '请输入节点ID' }]}
                >
                  <Input placeholder="例如：node-001" />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item
                  label="集群节点列表"
                  name="clusterNodes"
                  rules={[{ required: true, message: '请输入集群节点列表' }]}
                >
                  <Input placeholder="例如：node-001,node-002,node-003" />
                </Form.Item>
              </Col>
            </Row>
            <Form.Item>
              <Button
                type="primary"
                htmlType="submit"
                loading={loading}
                icon={<PlayCircleOutlined />}
              >
                启动引擎
              </Button>
            </Form.Item>
          </Form>
        </Card>
      ) : (
        <>
          {/* 系统状态概览 */}
          <Card title="系统状态" style={{ marginBottom: 24 }}>
            <Row gutter={16}>
              <Col span={6}>
                <Statistic
                  title="节点ID"
                  value={systemStats?.node_id || '-'}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="Raft状态"
                  value={systemStats?.raft_state || '-'}
                  prefix={
                    <Tag color={raftStateColors[systemStats?.raft_state as keyof typeof raftStateColors] || 'default'}>
                      {systemStats?.raft_state?.toUpperCase()}
                    </Tag>
                  }
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="活跃任务"
                  value={systemStats?.active_tasks || 0}
                  suffix="个"
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="完成任务"
                  value={systemStats?.completed_tasks || 0}
                  suffix="个"
                />
              </Col>
            </Row>
            <Divider />
            <Row gutter={16}>
              <Col span={6}>
                <Statistic
                  title="排队任务"
                  value={systemStats?.queued_tasks || 0}
                  suffix="个"
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="提交总数"
                  value={systemStats?.stats?.tasks_submitted || 0}
                  suffix="个"
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="失败任务"
                  value={systemStats?.stats?.tasks_failed || 0}
                  suffix="个"
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="平均处理时间"
                  value={systemStats?.stats?.average_processing_time || 0}
                  suffix="秒"
                  precision={2}
                />
              </Col>
            </Row>
            <Divider />
            <Space>
              <Button
                icon={<ReloadOutlined />}
                onClick={fetchSystemStats}
              >
                刷新状态
              </Button>
              <Button
                icon={<SettingOutlined />}
                onClick={handleCreateCheckpoint}
              >
                创建检查点
              </Button>
              <Button
                icon={<ReloadOutlined />}
                onClick={handleRollbackCheckpoint}
              >
                回滚检查点
              </Button>
              <Button
                danger
                icon={<PauseCircleOutlined />}
                onClick={handleShutdownEngine}
              >
                关闭引擎
              </Button>
            </Space>
          </Card>

          {/* 主要功能标签页 */}
          <Tabs defaultActiveKey="tasks">
            <TabPane tab="任务管理" key="tasks">
              <Card
                title="提交新任务"
                style={{ marginBottom: 16 }}
                extra={
                  <Button
                    icon={<ReloadOutlined />}
                    onClick={fetchTasks}
                  >
                    刷新任务列表
                  </Button>
                }
              >
                <Form
                  form={taskForm}
                  layout="vertical"
                  onFinish={handleSubmitTask}
                >
                  <Row gutter={16}>
                    <Col span={8}>
                      <Form.Item
                        label="任务类型"
                        name="taskType"
                        rules={[{ required: true, message: '请输入任务类型' }]}
                      >
                        <Select placeholder="选择任务类型">
                          <Option value="data_processing">数据处理</Option>
                          <Option value="batch_processing">批处理</Option>
                          <Option value="pipeline">流水线</Option>
                          <Option value="complex_analysis">复杂分析</Option>
                          <Option value="long_running">长时间运行</Option>
                        </Select>
                      </Form.Item>
                    </Col>
                    <Col span={8}>
                      <Form.Item
                        label="优先级"
                        name="priority"
                        initialValue="medium"
                      >
                        <Select>
                          <Option value="critical">关键</Option>
                          <Option value="high">高</Option>
                          <Option value="medium">中</Option>
                          <Option value="low">低</Option>
                          <Option value="background">后台</Option>
                        </Select>
                      </Form.Item>
                    </Col>
                    <Col span={8}>
                      <Form.Item
                        label="分解策略"
                        name="decompositionStrategy"
                      >
                        <Select placeholder="选择分解策略" allowClear>
                          <Option value="parallel">并行分解</Option>
                          <Option value="sequential">序列分解</Option>
                          <Option value="hierarchical">分层分解</Option>
                          <Option value="pipeline">流水线分解</Option>
                        </Select>
                      </Form.Item>
                    </Col>
                  </Row>
                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item
                        label="任务数据 (JSON)"
                        name="taskData"
                        rules={[
                          { required: true, message: '请输入任务数据' },
                          {
                            validator: (_, value) => {
                              if (!value) return Promise.resolve();
                              try {
                                JSON.parse(value);
                                return Promise.resolve();
                              } catch {
                                return Promise.reject(new Error('请输入有效的JSON格式'));
                              }
                            }
                          }
                        ]}
                      >
                        <TextArea
                          rows={4}
                          placeholder='{"input": "sample_data", "operation": "transform"}'
                        />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item
                        label="任务需求 (JSON)"
                        name="requirements"
                      >
                        <TextArea
                          rows={4}
                          placeholder='{"cpu": 0.5, "memory": 1024, "timeout": 3600}'
                        />
                      </Form.Item>
                    </Col>
                  </Row>
                  <Form.Item
                    label="分配策略"
                    name="assignmentStrategy"
                    initialValue="capability_based"
                  >
                    <Select>
                      <Option value="capability_based">基于能力</Option>
                      <Option value="load_balanced">负载均衡</Option>
                      <Option value="resource_optimized">资源优化</Option>
                      <Option value="deadline_aware">截止时间敏感</Option>
                      <Option value="locality_aware">位置感知</Option>
                    </Select>
                  </Form.Item>
                  <Form.Item>
                    <Button
                      type="primary"
                      htmlType="submit"
                      loading={loading}
                      icon={<PlusOutlined />}
                    >
                      提交任务
                    </Button>
                  </Form.Item>
                </Form>
              </Card>

              <Card title="任务列表">
                <Table
                  columns={taskColumns}
                  dataSource={tasks}
                  rowKey="task_id"
                  pagination={{ pageSize: 10 }}
                  loading={loading}
                />
              </Card>
            </TabPane>

            <TabPane tab="冲突管理" key="conflicts">
              <Card
                title="系统冲突"
                extra={
                  <Button
                    icon={<ReloadOutlined />}
                    onClick={fetchConflicts}
                  >
                    刷新冲突列表
                  </Button>
                }
              >
                {conflicts.length === 0 ? (
                  <Alert
                    message="系统运行正常"
                    description="当前没有检测到任何冲突。"
                    type="success"
                    showIcon
                  />
                ) : (
                  <Table
                    columns={conflictColumns}
                    dataSource={conflicts}
                    rowKey="conflict_id"
                    pagination={{ pageSize: 10 }}
                  />
                )}
              </Card>
            </TabPane>

            <TabPane tab="系统详情" key="details">
              <Card title="详细统计信息">
                <Descriptions bordered column={2}>
                  <Descriptions.Item label="节点ID">{systemStats?.node_id}</Descriptions.Item>
                  <Descriptions.Item label="Raft状态">{systemStats?.raft_state}</Descriptions.Item>
                  <Descriptions.Item label="Leader节点">{systemStats?.leader_id || '未知'}</Descriptions.Item>
                  <Descriptions.Item label="活跃任务数">{systemStats?.active_tasks}</Descriptions.Item>
                  <Descriptions.Item label="完成任务数">{systemStats?.completed_tasks}</Descriptions.Item>
                  <Descriptions.Item label="排队任务数">{systemStats?.queued_tasks}</Descriptions.Item>
                  <Descriptions.Item label="提交总数">{systemStats?.stats?.tasks_submitted}</Descriptions.Item>
                  <Descriptions.Item label="失败总数">{systemStats?.stats?.tasks_failed}</Descriptions.Item>
                  <Descriptions.Item label="取消总数">{systemStats?.stats?.tasks_cancelled}</Descriptions.Item>
                  <Descriptions.Item label="平均处理时间">{systemStats?.stats?.average_processing_time}秒</Descriptions.Item>
                </Descriptions>

                {systemStats?.state_summary && (
                  <>
                    <Title level={4} style={{ marginTop: 24 }}>状态摘要</Title>
                    <pre style={{ background: '#f5f5f5', padding: 16, borderRadius: 4 }}>
                      {JSON.stringify(systemStats.state_summary, null, 2)}
                    </pre>
                  </>
                )}
              </Card>
            </TabPane>
          </Tabs>
        </>
      )}
    </div>
  );
};

export default DistributedTaskCoordinationPage;