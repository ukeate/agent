import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Table,
  Tag,
  Progress,
  Timeline,
  Badge,
  Button,
  Space,
  Typography,
  Tabs,
  Alert,
  Select,
  DatePicker,
  Tooltip
} from 'antd';
import {
  DashboardOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  LoadingOutlined,
  ReloadOutlined,
  BarChartOutlined,
  LineChartOutlined
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { RangePicker } = DatePicker;

interface TaskMonitorData {
  task_id: string;
  status: 'pending' | 'assigned' | 'running' | 'completed' | 'failed' | 'cancelled';
  assigned_agent: string;
  priority: string;
  progress: number;
  start_time?: string;
  estimated_completion?: string;
  resource_usage: {
    cpu: number;
    memory: number;
    network: number;
  };
  performance_metrics: {
    throughput: number;
    latency: number;
    error_rate: number;
  };
}

const DistributedTaskMonitorPage: React.FC = () => {
  const [tasks, setTasks] = useState<TaskMonitorData[]>([]);
  const [loading, setLoading] = useState(false);

  // 生成模拟数据
  const generateMockTasks = (): TaskMonitorData[] => {
    const statuses: TaskMonitorData['status'][] = ['pending', 'assigned', 'running', 'completed', 'failed', 'cancelled'];
    const priorities = ['critical', 'high', 'medium', 'low'];
    
    return Array.from({ length: 20 }, (_, i) => ({
      task_id: `task_${String(i + 1).padStart(3, '0')}`,
      status: statuses[Math.floor(Math.random() * statuses.length)],
      assigned_agent: `agent_${Math.floor(Math.random() * 8) + 1}`,
      priority: priorities[Math.floor(Math.random() * priorities.length)],
      progress: Math.floor(Math.random() * 100),
      start_time: new Date(Date.now() - Math.random() * 3600000).toISOString(),
      estimated_completion: new Date(Date.now() + Math.random() * 3600000).toISOString(),
      resource_usage: {
        cpu: Math.floor(Math.random() * 80) + 10,
        memory: Math.floor(Math.random() * 70) + 20,
        network: Math.floor(Math.random() * 50) + 10
      },
      performance_metrics: {
        throughput: Math.floor(Math.random() * 1000) + 100,
        latency: Math.floor(Math.random() * 500) + 50,
        error_rate: Math.random() * 5
      }
    }));
  };

  const taskColumns: ColumnsType<TaskMonitorData> = [
    {
      title: '任务ID',
      dataIndex: 'task_id',
      key: 'task_id',
      render: (id: string) => <Text code>{id}</Text>
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const colors = {
          pending: 'default',
          assigned: 'processing',
          running: 'processing',
          completed: 'success',
          failed: 'error',
          cancelled: 'warning'
        };
        const icons = {
          pending: <ClockCircleOutlined />,
          assigned: <LoadingOutlined />,
          running: <LoadingOutlined />,
          completed: <CheckCircleOutlined />,
          failed: <ExclamationCircleOutlined />,
          cancelled: <ExclamationCircleOutlined />
        };
        return (
          <Tag color={colors[status as keyof typeof colors]} icon={icons[status as keyof typeof icons]}>
            {status.toUpperCase()}
          </Tag>
        );
      }
    },
    {
      title: '分配智能体',
      dataIndex: 'assigned_agent',
      key: 'assigned_agent',
      render: (agent: string) => <Tag color="blue">{agent}</Tag>
    },
    {
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      render: (progress: number, record: TaskMonitorData) => (
        <Progress
          percent={progress}
          size="small"
          status={record.status === 'failed' ? 'exception' : undefined}
        />
      )
    },
    {
      title: '资源使用',
      key: 'resource',
      render: (_, record: TaskMonitorData) => (
        <Space direction="vertical" size="small">
          <Text style={{ fontSize: '12px' }}>CPU: {record.resource_usage.cpu}%</Text>
          <Text style={{ fontSize: '12px' }}>MEM: {record.resource_usage.memory}%</Text>
        </Space>
      )
    },
    {
      title: '性能指标',
      key: 'performance',
      render: (_, record: TaskMonitorData) => (
        <Space direction="vertical" size="small">
          <Text style={{ fontSize: '12px' }}>吞吐: {record.performance_metrics.throughput}/s</Text>
          <Text style={{ fontSize: '12px' }}>延迟: {record.performance_metrics.latency}ms</Text>
        </Space>
      )
    }
  ];

  useEffect(() => {
    setTasks(generateMockTasks());
    
    const interval = setInterval(() => {
      setTasks(prev => prev.map(task => ({
        ...task,
        progress: task.status === 'running' ? Math.min(100, task.progress + Math.random() * 5) : task.progress,
        resource_usage: {
          cpu: Math.max(0, task.resource_usage.cpu + (Math.random() - 0.5) * 10),
          memory: Math.max(0, task.resource_usage.memory + (Math.random() - 0.5) * 8),
          network: Math.max(0, task.resource_usage.network + (Math.random() - 0.5) * 15)
        }
      })));
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const taskStats = {
    total: tasks.length,
    running: tasks.filter(t => t.status === 'running').length,
    completed: tasks.filter(t => t.status === 'completed').length,
    failed: tasks.filter(t => t.status === 'failed').length,
    avg_progress: tasks.reduce((sum, t) => sum + t.progress, 0) / tasks.length
  };

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <Title level={2}>分布式任务监控</Title>

      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="总任务数"
              value={taskStats.total}
              prefix={<DashboardOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="运行中"
              value={taskStats.running}
              prefix={<LoadingOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="已完成"
              value={taskStats.completed}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均进度"
              value={taskStats.avg_progress}
              precision={1}
              suffix="%"
              prefix={<BarChartOutlined />}
            />
          </Card>
        </Col>
      </Row>

      <Card
        title="任务监控列表"
        extra={
          <Space>
            <Button icon={<ReloadOutlined />} onClick={() => setTasks(generateMockTasks())}>
              刷新
            </Button>
          </Space>
        }
      >
        <Table
          columns={taskColumns}
          dataSource={tasks}
          rowKey="task_id"
          pagination={{ pageSize: 10 }}
          size="small"
        />
      </Card>
    </div>
  );
};

export default DistributedTaskMonitorPage;