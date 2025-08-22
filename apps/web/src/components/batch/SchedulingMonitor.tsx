import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Row, 
  Col, 
  Statistic, 
  Progress, 
  Table, 
  Tag, 
  Button, 
  Space, 
  Alert,
  Badge
} from 'antd';
import { 
  ThunderboltOutlined,
  BarChartOutlined,
  ReloadOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
} from '@ant-design/icons';

interface WorkerMetrics {
  worker_id: string;
  current_load: number;
  task_completion_rate: number;
  average_task_time: number;
  task_type_performance: Record<string, number>;
  status: 'active' | 'idle' | 'overloaded' | 'offline';
}

interface SLARequirement {
  id: string;
  name: string;
  target_completion_time: number;
  max_failure_rate: number;
  priority_weight: number;
  current_performance: {
    avg_completion_time: number;
    failure_rate: number;
    violation_count: number;
  };
  status: 'met' | 'warning' | 'violated';
}

interface SystemResources {
  cpu_usage: number;
  memory_usage: number;
  io_utilization: number;
  network_usage: number;
  active_connections: number;
  queue_depth: number;
}

interface PredictiveScheduling {
  predicted_completion_times: Record<string, number>;
  scaling_recommendations: {
    action: 'scale_up' | 'scale_down' | 'maintain';
    target_workers: number;
    confidence: number;
    reason: string;
  };
  resource_forecast: {
    next_hour: SystemResources;
    next_24h: SystemResources;
  };
}

interface SchedulingStats {
  workers: WorkerMetrics[];
  sla_requirements: SLARequirement[];
  system_resources: SystemResources;
  predictive_scheduling: PredictiveScheduling;
  total_tasks_scheduled: number;
  load_balancing_efficiency: number;
  sla_compliance_rate: number;
}

const SchedulingMonitor: React.FC = () => {
  const [stats, setStats] = useState<SchedulingStats>({
    workers: [],
    sla_requirements: [],
    system_resources: {
      cpu_usage: 0,
      memory_usage: 0,
      io_utilization: 0,
      network_usage: 0,
      active_connections: 0,
      queue_depth: 0
    },
    predictive_scheduling: {
      predicted_completion_times: {},
      scaling_recommendations: {
        action: 'maintain',
        target_workers: 0,
        confidence: 0,
        reason: ''
      },
      resource_forecast: {
        next_hour: {
          cpu_usage: 0,
          memory_usage: 0,
          io_utilization: 0,
          network_usage: 0,
          active_connections: 0,
          queue_depth: 0
        },
        next_24h: {
          cpu_usage: 0,
          memory_usage: 0,
          io_utilization: 0,
          network_usage: 0,
          active_connections: 0,
          queue_depth: 0
        }
      }
    },
    total_tasks_scheduled: 0,
    load_balancing_efficiency: 0,
    sla_compliance_rate: 0
  });
  const [loading, setLoading] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date>();

  const fetchSchedulingStats = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/v1/batch/scheduling/stats');
      if (response.ok) {
        const data = await response.json();
        setStats(data);
        setLastUpdate(new Date());
      }
    } catch (error) {
      console.error('获取调度统计失败:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSchedulingStats();
    const interval = setInterval(fetchSchedulingStats, 10000);
    return () => clearInterval(interval);
  }, []);

  const getWorkerStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'green';
      case 'idle': return 'blue';
      case 'overloaded': return 'red';
      case 'offline': return 'default';
      default: return 'default';
    }
  };

  const getSLAStatusColor = (status: string) => {
    switch (status) {
      case 'met': return 'green';
      case 'warning': return 'orange';
      case 'violated': return 'red';
      default: return 'default';
    }
  };

  const getResourceStatus = (usage: number) => {
    if (usage < 60) return 'normal';
    if (usage < 80) return 'warning';
    return 'critical';
  };

  const getResourceColor = (usage: number) => {
    if (usage < 60) return '#52c41a';
    if (usage < 80) return '#fa8c16';
    return '#ff4d4f';
  };

  const workerColumns = [
    {
      title: '工作者ID',
      dataIndex: 'worker_id',
      key: 'worker_id',
      render: (id: string) => <code>{id}</code>
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={getWorkerStatusColor(status)}>
          {status.toUpperCase()}
        </Tag>
      )
    },
    {
      title: '当前负载',
      dataIndex: 'current_load',
      key: 'current_load',
      render: (load: number) => (
        <Progress
          percent={Math.round(load * 100)}
          size="small"
          status={load > 0.8 ? 'exception' : 'success'}
        />
      )
    },
    {
      title: '完成率',
      dataIndex: 'task_completion_rate',
      key: 'task_completion_rate',
      render: (rate: number) => `${(rate * 100).toFixed(1)}%`
    },
    {
      title: '平均耗时',
      dataIndex: 'average_task_time',
      key: 'average_task_time',
      render: (time: number) => `${time.toFixed(2)}s`
    }
  ];

  const slaColumns = [
    {
      title: 'SLA名称',
      dataIndex: 'name',
      key: 'name'
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Badge 
          status={getSLAStatusColor(status) as any}
          text={status.toUpperCase()}
        />
      )
    },
    {
      title: '目标完成时间',
      dataIndex: 'target_completion_time',
      key: 'target_completion_time',
      render: (time: number) => `${time}s`
    },
    {
      title: '当前性能',
      key: 'current_performance',
      render: (_: any, record: SLARequirement) => (
        <div>
          <div>完成时间: {record.current_performance.avg_completion_time.toFixed(2)}s</div>
          <div>失败率: {(record.current_performance.failure_rate * 100).toFixed(1)}%</div>
          <div>违规次数: {record.current_performance.violation_count}</div>
        </div>
      )
    },
    {
      title: '优先级权重',
      dataIndex: 'priority_weight',
      key: 'priority_weight',
      render: (weight: number) => weight.toFixed(2)
    }
  ];

  const getScalingActionIcon = (action: string) => {
    switch (action) {
      case 'scale_up': return <BarChartOutlined style={{ color: '#52c41a' }} />;
      case 'scale_down': return <BarChartOutlined style={{ color: '#1890ff' }} />;
      case 'maintain': return <CheckCircleOutlined style={{ color: '#faad14' }} />;
      default: return <ExclamationCircleOutlined />;
    }
  };

  return (
    <div className="scheduling-monitor">
      {/* 系统概览 */}
      <Row gutter={[16, 16]} className="mb-4">
        <Col span={6}>
          <Card>
            <Statistic
              title="调度任务总数"
              value={stats.total_tasks_scheduled}
              prefix={<ThunderboltOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="负载均衡效率"
              value={stats.load_balancing_efficiency}
              suffix="%"
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="SLA合规率"
              value={stats.sla_compliance_rate}
              suffix="%"
              valueStyle={{ 
                color: stats.sla_compliance_rate >= 95 ? '#52c41a' : '#ff4d4f' 
              }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="活跃工作者"
              value={stats.workers.filter(w => w.status === 'active').length}
              suffix={`/ ${stats.workers.length}`}
            />
          </Card>
        </Col>
      </Row>

      {/* 系统资源监控 */}
      <Row gutter={[16, 16]} className="mb-4">
        <Col span={12}>
          <Card title="系统资源使用率" size="small">
            <Row gutter={[8, 8]}>
              <Col span={12}>
                <div className="mb-2">
                  <div className="flex justify-between mb-1">
                    <span>CPU</span>
                    <span>{stats.system_resources.cpu_usage.toFixed(1)}%</span>
                  </div>
                  <Progress 
                    percent={stats.system_resources.cpu_usage}
                    strokeColor={getResourceColor(stats.system_resources.cpu_usage)}
                    size="small"
                  />
                </div>
                <div>
                  <div className="flex justify-between mb-1">
                    <span>内存</span>
                    <span>{stats.system_resources.memory_usage.toFixed(1)}%</span>
                  </div>
                  <Progress 
                    percent={stats.system_resources.memory_usage}
                    strokeColor={getResourceColor(stats.system_resources.memory_usage)}
                    size="small"
                  />
                </div>
              </Col>
              <Col span={12}>
                <div className="mb-2">
                  <div className="flex justify-between mb-1">
                    <span>I/O</span>
                    <span>{stats.system_resources.io_utilization.toFixed(1)}%</span>
                  </div>
                  <Progress 
                    percent={stats.system_resources.io_utilization}
                    strokeColor={getResourceColor(stats.system_resources.io_utilization)}
                    size="small"
                  />
                </div>
                <div>
                  <div className="flex justify-between mb-1">
                    <span>网络</span>
                    <span>{stats.system_resources.network_usage.toFixed(1)}%</span>
                  </div>
                  <Progress 
                    percent={stats.system_resources.network_usage}
                    strokeColor={getResourceColor(stats.system_resources.network_usage)}
                    size="small"
                  />
                </div>
              </Col>
            </Row>
            <Row className="mt-3">
              <Col span={12}>
                <Statistic 
                  title="活跃连接" 
                  value={stats.system_resources.active_connections}
                />
              </Col>
              <Col span={12}>
                <Statistic 
                  title="队列深度" 
                  value={stats.system_resources.queue_depth}
                />
              </Col>
            </Row>
          </Card>
        </Col>
        <Col span={12}>
          <Card title="预测性调度建议" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div className="flex items-center space-x-2">
                {getScalingActionIcon(stats.predictive_scheduling.scaling_recommendations.action)}
                <span className="font-medium">
                  {stats.predictive_scheduling.scaling_recommendations.action === 'scale_up' && '建议扩容'}
                  {stats.predictive_scheduling.scaling_recommendations.action === 'scale_down' && '建议缩容'}
                  {stats.predictive_scheduling.scaling_recommendations.action === 'maintain' && '保持当前规模'}
                </span>
              </div>
              <div className="text-sm text-gray-600">
                目标工作者数: {stats.predictive_scheduling.scaling_recommendations.target_workers}
              </div>
              <div className="text-sm text-gray-600">
                置信度: {(stats.predictive_scheduling.scaling_recommendations.confidence * 100).toFixed(1)}%
              </div>
              <Alert 
                message={stats.predictive_scheduling.scaling_recommendations.reason}
                type="info"
                showIcon
              />
            </Space>
          </Card>
        </Col>
      </Row>

      {/* 工作者状态 */}
      <Row gutter={[16, 16]} className="mb-4">
        <Col span={24}>
          <Card 
            title="工作者状态"
            extra={
              <Button 
                icon={<ReloadOutlined />} 
                onClick={fetchSchedulingStats}
                loading={loading}
              >
                刷新
              </Button>
            }
          >
            <Table
              columns={workerColumns}
              dataSource={stats.workers}
              rowKey="worker_id"
              loading={loading}
              pagination={false}
              size="small"
            />
          </Card>
        </Col>
      </Row>

      {/* SLA监控 */}
      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Card title="SLA监控">
            <Table
              columns={slaColumns}
              dataSource={stats.sla_requirements}
              rowKey="id"
              loading={loading}
              pagination={false}
              size="small"
            />
          </Card>
        </Col>
      </Row>

      {/* 更新时间 */}
      {lastUpdate && (
        <div className="text-center text-gray-500 mt-4">
          最后更新: {lastUpdate.toLocaleString()}
        </div>
      )}
    </div>
  );
};

export default SchedulingMonitor;