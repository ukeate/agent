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
  Tree,
  Tag,
  Alert,
  Space,
  Modal,
  message,
  Statistic,
  Typography,
  Divider,
  Steps,
  Timeline,
  Descriptions,
  Progress
} from 'antd';
import {
  BranchesOutlined,
  NodeIndexOutlined,
  PlayCircleOutlined,
  ReloadOutlined,
  EyeOutlined,
  SettingOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  ForkOutlined
} from '@ant-design/icons';
import type { DataNode } from 'antd/es/tree';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;
const { TextArea } = Input;
const { Step } = Steps;

interface DecompositionTask {
  task_id: string;
  parent_task_id?: string;
  task_type: string;
  data: any;
  requirements: any;
  priority: string;
  status: string;
  dependencies: string[];
  decomposition_strategy?: string;
  level: number;
  created_at: string;
}

interface DecompositionExample {
  name: string;
  description: string;
  strategy: string;
  input: any;
  expected_output: number;
}

const TaskDecomposerPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [decompositionTasks, setDecompositionTasks] = useState<DecompositionTask[]>([]);
  const [selectedStrategy, setSelectedStrategy] = useState<string>('parallel');
  const [treeData, setTreeData] = useState<DataNode[]>([]);
  const [form] = Form.useForm();
  const [previewVisible, setPreviewVisible] = useState(false);
  const [previewData, setPreviewData] = useState<any>(null);

  // 分解策略配置
  const decompositionStrategies = {
    parallel: {
      name: '并行分解',
      description: '将任务分解为可以并行执行的子任务',
      icon: <NodeIndexOutlined />,
      color: 'blue',
      suitable: ['批处理', '数据转换', '独立计算']
    },
    sequential: {
      name: '序列分解',
      description: '将任务分解为按顺序执行的子任务',
      icon: <ClockCircleOutlined />,
      color: 'green',
      suitable: ['流水线处理', '依赖任务', '状态机']
    },
    hierarchical: {
      name: '分层分解',
      description: '将任务分解为多层次的任务树',
      icon: <BranchesOutlined />,
      color: 'orange',
      suitable: ['复杂分析', '多阶段处理', '嵌套任务']
    },
    pipeline: {
      name: '流水线分解',
      description: '将任务分解为流水线阶段',
      icon: <ForkOutlined />,
      color: 'purple',
      suitable: ['数据管道', '实时处理', '流式计算']
    }
  };

  // 示例任务模板
  const exampleTasks: DecompositionExample[] = [
    {
      name: '图像批处理',
      description: '将100张图片进行并行处理',
      strategy: 'parallel',
      input: {
        chunks: Array.from({ length: 5 }, (_, i) => ({
          id: i + 1,
          data: `image_batch_${i + 1}`,
          count: 20
        }))
      },
      expected_output: 5
    },
    {
      name: '数据处理流水线',
      description: '清洗 -> 转换 -> 验证 -> 存储',
      strategy: 'sequential',
      input: {
        steps: [
          { type: 'clean', name: '数据清洗' },
          { type: 'transform', name: '数据转换' },
          { type: 'validate', name: '数据验证' },
          { type: 'store', name: '数据存储' }
        ]
      },
      expected_output: 4
    },
    {
      name: '分层数据分析',
      description: '多层次数据分析任务',
      strategy: 'hierarchical',
      input: {
        hierarchy: {
          phase1: {
            type: 'preprocessing',
            subtasks: {
              task1: { name: '数据加载' },
              task2: { name: '数据清理' }
            }
          },
          phase2: {
            type: 'analysis',
            subtasks: {
              task3: { name: '特征提取' },
              task4: { name: '模型训练' }
            }
          }
        }
      },
      expected_output: 6
    },
    {
      name: '实时数据管道',
      description: '实时数据处理管道',
      strategy: 'pipeline',
      input: {
        stages: [
          { name: 'ingestion', parallel: 3 },
          { name: 'processing', parallel: 2 },
          { name: 'output', parallel: 1 }
        ]
      },
      expected_output: 6
    }
  ];

  // 模拟任务分解
  const simulateDecomposition = (strategy: string, data: any): DecompositionTask[] => {
    const baseTaskId = `task_${Date.now()}`;
    const tasks: DecompositionTask[] = [];

    switch (strategy) {
      case 'parallel':
        if (data.chunks) {
          data.chunks.forEach((chunk: any, index: number) => {
            tasks.push({
              task_id: `${baseTaskId}_chunk_${index}`,
              parent_task_id: baseTaskId,
              task_type: 'chunk_processing',
              data: chunk,
              requirements: { cpu: 0.2, memory: 256 },
              priority: 'medium',
              status: 'pending',
              dependencies: [],
              decomposition_strategy: strategy,
              level: 1,
              created_at: new Date().toISOString()
            });
          });
        }
        break;

      case 'sequential':
        if (data.steps) {
          data.steps.forEach((step: any, index: number) => {
            const dependencies = index > 0 ? [`${baseTaskId}_step_${index - 1}`] : [];
            tasks.push({
              task_id: `${baseTaskId}_step_${index}`,
              parent_task_id: baseTaskId,
              task_type: step.type,
              data: { step: step.name },
              requirements: { cpu: 0.3, memory: 512 },
              priority: 'medium',
              status: 'pending',
              dependencies,
              decomposition_strategy: strategy,
              level: 1,
              created_at: new Date().toISOString()
            });
          });
        }
        break;

      case 'hierarchical':
        if (data.hierarchy) {
          let taskIndex = 0;
          Object.entries(data.hierarchy).forEach(([phaseKey, phase]: [string, any]) => {
            const phaseTaskId = `${baseTaskId}_${phaseKey}`;
            tasks.push({
              task_id: phaseTaskId,
              parent_task_id: baseTaskId,
              task_type: phase.type,
              data: { phase: phaseKey },
              requirements: { cpu: 0.1, memory: 128 },
              priority: 'medium',
              status: 'pending',
              dependencies: [],
              decomposition_strategy: strategy,
              level: 1,
              created_at: new Date().toISOString()
            });

            if (phase.subtasks) {
              Object.entries(phase.subtasks).forEach(([taskKey, task]: [string, any]) => {
                tasks.push({
                  task_id: `${baseTaskId}_${phaseKey}_${taskKey}`,
                  parent_task_id: phaseTaskId,
                  task_type: 'subtask',
                  data: task,
                  requirements: { cpu: 0.2, memory: 256 },
                  priority: 'medium',
                  status: 'pending',
                  dependencies: [phaseTaskId],
                  decomposition_strategy: strategy,
                  level: 2,
                  created_at: new Date().toISOString()
                });
              });
            }
          });
        }
        break;

      case 'pipeline':
        if (data.stages) {
          data.stages.forEach((stage: any, index: number) => {
            for (let i = 0; i < stage.parallel; i++) {
              const dependencies = index > 0 ? [`${baseTaskId}_stage${index - 1}_${i}`] : [];
              tasks.push({
                task_id: `${baseTaskId}_stage${index}_${i}`,
                parent_task_id: baseTaskId,
                task_type: 'pipeline_stage',
                data: { stage: stage.name, instance: i },
                requirements: { cpu: 0.3, memory: 512 },
                priority: 'medium',
                status: 'pending',
                dependencies,
                decomposition_strategy: strategy,
                level: 1,
                created_at: new Date().toISOString()
              });
            }
          });
        }
        break;
    }

    return tasks;
  };

  // 生成树形数据
  const generateTreeData = (tasks: DecompositionTask[]): DataNode[] => {
    const taskMap = new Map<string, DecompositionTask>();
    const rootTasks: DataNode[] = [];

    tasks.forEach(task => taskMap.set(task.task_id, task));

    const buildNode = (task: DecompositionTask): DataNode => {
      const children = tasks
        .filter(t => t.parent_task_id === task.task_id)
        .map(buildNode);

      return {
        title: (
          <Space>
            <Text strong>{task.task_id.split('_').pop()}</Text>
            <Tag color={getStatusColor(task.status)}>{task.status}</Tag>
            {task.dependencies.length > 0 && (
              <Tag color="orange">{task.dependencies.length} deps</Tag>
            )}
          </Space>
        ),
        key: task.task_id,
        children: children.length > 0 ? children : undefined,
        icon: getStrategyIcon(task.decomposition_strategy)
      };
    };

    // 找到根任务（没有parent的任务）
    const roots = tasks.filter(t => !t.parent_task_id);
    roots.forEach(root => {
      rootTasks.push(buildNode(root));
    });

    return rootTasks;
  };

  const getStatusColor = (status: string) => {
    const colors = {
      pending: 'default',
      running: 'processing',
      completed: 'success',
      failed: 'error',
      cancelled: 'warning'
    };
    return colors[status as keyof typeof colors] || 'default';
  };

  const getStrategyIcon = (strategy?: string) => {
    if (!strategy) return null;
    return decompositionStrategies[strategy as keyof typeof decompositionStrategies]?.icon;
  };

  // 加载示例任务
  const loadExample = (example: DecompositionExample) => {
    form.setFieldsValue({
      taskType: example.name,
      taskData: JSON.stringify(example.input, null, 2),
      decompositionStrategy: example.strategy
    });
    setSelectedStrategy(example.strategy);
  };

  // 预览分解结果
  const previewDecomposition = () => {
    form.validateFields().then((values) => {
      try {
        const data = JSON.parse(values.taskData);
        const tasks = simulateDecomposition(values.decompositionStrategy, data);
        setPreviewData({
          strategy: values.decompositionStrategy,
          tasks,
          treeData: generateTreeData(tasks)
        });
        setPreviewVisible(true);
      } catch (error) {
        message.error('任务数据格式错误，请检查JSON格式');
      }
    });
  };

  // 执行分解
  const executeDecomposition = async () => {
    form.validateFields().then(async (values) => {
      setLoading(true);
      try {
        const data = JSON.parse(values.taskData);
        const tasks = simulateDecomposition(values.decompositionStrategy, data);
        setDecompositionTasks(tasks);
        setTreeData(generateTreeData(tasks));
        message.success(`成功分解为 ${tasks.length} 个子任务`);
      } catch (error) {
        message.error('任务分解失败：' + error);
      } finally {
        setLoading(false);
      }
    });
  };

  // 表格列定义
  const columns = [
    {
      title: '任务ID',
      dataIndex: 'task_id',
      key: 'task_id',
      render: (taskId: string) => (
        <Text code copyable={{ text: taskId }}>
          {taskId.substring(taskId.lastIndexOf('_') + 1)}
        </Text>
      )
    },
    {
      title: '层级',
      dataIndex: 'level',
      key: 'level',
      render: (level: number) => <Tag color="blue">L{level}</Tag>
    },
    {
      title: '类型',
      dataIndex: 'task_type',
      key: 'task_type'
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={getStatusColor(status)}>
          {status.toUpperCase()}
        </Tag>
      )
    },
    {
      title: '依赖数',
      dataIndex: 'dependencies',
      key: 'dependencies',
      render: (deps: string[]) => (
        <Tag color={deps.length > 0 ? 'orange' : 'default'}>
          {deps.length}
        </Tag>
      )
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (time: string) => new Date(time).toLocaleTimeString()
    }
  ];

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <Title level={2}>任务分解器</Title>
      <Paragraph>
        智能任务分解系统，支持多种分解策略：并行、序列、分层和流水线分解。
      </Paragraph>

      <Row gutter={24}>
        <Col span={16}>
          <Card title="任务分解配置" style={{ marginBottom: 24 }}>
            <Form
              form={form}
              layout="vertical"
              initialValues={{
                decompositionStrategy: 'parallel',
                priority: 'medium'
              }}
            >
              <Row gutter={16}>
                <Col span={12}>
                  <Form.Item
                    label="任务类型"
                    name="taskType"
                    rules={[{ required: true, message: '请输入任务类型' }]}
                  >
                    <Input placeholder="例如：批处理任务" />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    label="分解策略"
                    name="decompositionStrategy"
                    rules={[{ required: true, message: '请选择分解策略' }]}
                  >
                    <Select onChange={setSelectedStrategy}>
                      {Object.entries(decompositionStrategies).map(([key, strategy]) => (
                        <Option key={key} value={key}>
                          <Space>
                            {strategy.icon}
                            {strategy.name}
                          </Space>
                        </Option>
                      ))}
                    </Select>
                  </Form.Item>
                </Col>
              </Row>

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
                <TextArea rows={8} placeholder="输入任务数据..." />
              </Form.Item>

              <Form.Item>
                <Space>
                  <Button
                    type="primary"
                    onClick={executeDecomposition}
                    loading={loading}
                    icon={<PlayCircleOutlined />}
                  >
                    执行分解
                  </Button>
                  <Button
                    onClick={previewDecomposition}
                    icon={<EyeOutlined />}
                  >
                    预览结果
                  </Button>
                  <Button
                    onClick={() => form.resetFields()}
                    icon={<ReloadOutlined />}
                  >
                    重置
                  </Button>
                </Space>
              </Form.Item>
            </Form>
          </Card>

          {/* 分解结果 */}
          {decompositionTasks.length > 0 && (
            <>
              <Card title="分解统计" style={{ marginBottom: 16 }}>
                <Row gutter={16}>
                  <Col span={6}>
                    <Statistic
                      title="总任务数"
                      value={decompositionTasks.length}
                      suffix="个"
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="层级数"
                      value={Math.max(...decompositionTasks.map(t => t.level))}
                      suffix="层"
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="并行度"
                      value={decompositionTasks.filter(t => t.dependencies.length === 0).length}
                      suffix="个"
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="依赖任务"
                      value={decompositionTasks.filter(t => t.dependencies.length > 0).length}
                      suffix="个"
                    />
                  </Col>
                </Row>
              </Card>

              <Card title="任务树" style={{ marginBottom: 16 }}>
                <Tree
                  treeData={treeData}
                  showIcon
                  defaultExpandAll
                  style={{ background: 'white', padding: 16 }}
                />
              </Card>

              <Card title="任务详情">
                <Table
                  columns={columns}
                  dataSource={decompositionTasks}
                  rowKey="task_id"
                  pagination={{ pageSize: 10 }}
                  size="small"
                />
              </Card>
            </>
          )}
        </Col>

        <Col span={8}>
          {/* 策略说明 */}
          <Card title="策略说明" style={{ marginBottom: 16 }}>
            {selectedStrategy && (
              <div>
                <Space style={{ marginBottom: 16 }}>
                  {decompositionStrategies[selectedStrategy as keyof typeof decompositionStrategies].icon}
                  <Title level={4}>
                    {decompositionStrategies[selectedStrategy as keyof typeof decompositionStrategies].name}
                  </Title>
                </Space>
                <Paragraph>
                  {decompositionStrategies[selectedStrategy as keyof typeof decompositionStrategies].description}
                </Paragraph>
                <Title level={5}>适用场景：</Title>
                <ul>
                  {decompositionStrategies[selectedStrategy as keyof typeof decompositionStrategies].suitable.map((scenario, index) => (
                    <li key={index}>{scenario}</li>
                  ))}
                </ul>
              </div>
            )}
          </Card>

          {/* 示例任务 */}
          <Card title="示例任务">
            <Space direction="vertical" style={{ width: '100%' }}>
              {exampleTasks.map((example, index) => (
                <Card
                  key={index}
                  size="small"
                  title={example.name}
                  extra={
                    <Button
                      size="small"
                      type="link"
                      onClick={() => loadExample(example)}
                    >
                      加载
                    </Button>
                  }
                >
                  <Paragraph style={{ fontSize: '12px', margin: 0 }}>
                    {example.description}
                  </Paragraph>
                  <Space style={{ marginTop: 8 }}>
                    <Tag color={decompositionStrategies[example.strategy as keyof typeof decompositionStrategies].color}>
                      {decompositionStrategies[example.strategy as keyof typeof decompositionStrategies].name}
                    </Tag>
                    <Tag>预期: {example.expected_output}个子任务</Tag>
                  </Space>
                </Card>
              ))}
            </Space>
          </Card>
        </Col>
      </Row>

      {/* 预览模态框 */}
      <Modal
        title="分解预览"
        visible={previewVisible}
        onCancel={() => setPreviewVisible(false)}
        width={800}
        footer={null}
      >
        {previewData && (
          <div>
            <Alert
              message={`${decompositionStrategies[previewData.strategy as keyof typeof decompositionStrategies].name}分解预览`}
              description={`将生成 ${previewData.tasks.length} 个子任务`}
              type="info"
              showIcon
              style={{ marginBottom: 16 }}
            />
            <Tree
              treeData={previewData.treeData}
              showIcon
              defaultExpandAll
            />
          </div>
        )}
      </Modal>
    </div>
  );
};

export default TaskDecomposerPage;