import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Badge, Progress, Statistic, Alert, Timeline, Table, Tag, Space, Button, Select, Tooltip } from 'antd';
import { Line, Gauge, Liquid } from '@ant-design/plots';
import { 
  HeartOutlined, 
  CheckCircleOutlined, 
  ExclamationCircleOutlined,
  CloseCircleOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  CloudServerOutlined,
  ApiOutlined,
  SafetyCertificateOutlined,
  MonitorOutlined
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';

const { Option } = Select;

interface SystemComponent {
  id: string;
  name: string;
  type: 'service' | 'database' | 'cache' | 'queue';
  status: 'healthy' | 'warning' | 'error' | 'offline';
  uptime: number;
  responseTime: number;
  lastCheck: string;
  details: string;
  dependencies: string[];
}

interface HealthMetric {
  timestamp: string;
  cpu: number;
  memory: number;
  disk: number;
  network: number;
  connections: number;
}

interface ServiceHealth {
  service: string;
  availability: number;
  errorRate: number;
  throughput: number;
  latency: number;
  health: number;
}

interface HealthEvent {
  id: string;
  timestamp: string;
  type: 'info' | 'warning' | 'error' | 'recovery';
  component: string;
  message: string;
  duration?: number;
}

const RLSystemHealthPage: React.FC = () => {
  const [timeRange, setTimeRange] = useState('1h');
  const [components, setComponents] = useState<SystemComponent[]>([]);
  const [healthMetrics, setHealthMetrics] = useState<HealthMetric[]>([]);
  const [serviceHealth, setServiceHealth] = useState<ServiceHealth[]>([]);
  const [healthEvents, setHealthEvents] = useState<HealthEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [systemHealth, setSystemHealth] = useState(0.95);

  // 生成模拟数据
  const generateComponents = (): SystemComponent[] => [
    {
      id: 'rl-api',
      name: 'RL推荐API服务',
      type: 'service',
      status: 'healthy',
      uptime: 99.8,
      responseTime: 15.2,
      lastCheck: '2025-08-22 14:25:30',
      details: '3个实例运行正常，负载均衡器工作正常',
      dependencies: ['postgres', 'redis', 'qdrant']
    },
    {
      id: 'postgres',
      name: 'PostgreSQL数据库',
      type: 'database',
      status: 'healthy',
      uptime: 99.95,
      responseTime: 3.8,
      lastCheck: '2025-08-22 14:25:25',
      details: '主从复制正常，连接池使用率68%',
      dependencies: []
    },
    {
      id: 'redis',
      name: 'Redis缓存',
      type: 'cache',
      status: 'warning',
      uptime: 99.2,
      responseTime: 1.2,
      lastCheck: '2025-08-22 14:25:28',
      details: '内存使用率85%，建议清理过期键',
      dependencies: []
    },
    {
      id: 'qdrant',
      name: 'Qdrant向量数据库',
      type: 'database',
      status: 'healthy',
      uptime: 99.7,
      responseTime: 8.5,
      lastCheck: '2025-08-22 14:25:32',
      details: '向量索引正常，查询响应时间稳定',
      dependencies: []
    },
    {
      id: 'message-queue',
      name: '消息队列',
      type: 'queue',
      status: 'healthy',
      uptime: 99.9,
      responseTime: 5.1,
      lastCheck: '2025-08-22 14:25:26',
      details: '队列长度正常，无积压消息',
      dependencies: []
    },
    {
      id: 'monitoring',
      name: '监控系统',
      type: 'service',
      status: 'healthy',
      uptime: 99.85,
      responseTime: 12.3,
      lastCheck: '2025-08-22 14:25:29',
      details: 'Prometheus和Grafana运行正常',
      dependencies: []
    }
  ];

  const generateHealthMetrics = (): HealthMetric[] => {
    const data: HealthMetric[] = [];
    const points = timeRange === '1h' ? 60 : timeRange === '24h' ? 144 : 168;
    const interval = timeRange === '1h' ? 60000 : timeRange === '24h' ? 600000 : 3600000;

    for (let i = points; i >= 0; i--) {
      const timestamp = new Date(Date.now() - i * interval).toLocaleTimeString();
      data.push({
        timestamp,
        cpu: 40 + Math.random() * 30 + Math.sin(i * 0.1) * 10,
        memory: 60 + Math.random() * 20 + Math.sin(i * 0.15) * 5,
        disk: 30 + Math.random() * 10,
        network: 20 + Math.random() * 40 + Math.sin(i * 0.08) * 15,
        connections: 150 + Math.random() * 100 + Math.sin(i * 0.12) * 50
      });
    }
    return data;
  };

  const generateServiceHealth = (): ServiceHealth[] => [
    {
      service: 'UCB算法服务',
      availability: 99.8,
      errorRate: 0.08,
      throughput: 156.8,
      latency: 12.3,
      health: 95
    },
    {
      service: 'Thompson Sampling服务',
      availability: 99.5,
      errorRate: 0.12,
      throughput: 142.3,
      latency: 15.7,
      health: 92
    },
    {
      service: 'Q-Learning服务',
      availability: 98.9,
      errorRate: 0.05,
      throughput: 89.2,
      latency: 28.4,
      health: 88
    },
    {
      service: '推荐缓存服务',
      availability: 99.9,
      errorRate: 0.02,
      throughput: 1250.5,
      latency: 1.2,
      health: 98
    },
    {
      service: '用户画像服务',
      availability: 99.2,
      errorRate: 0.15,
      throughput: 68.4,
      latency: 45.2,
      health: 85
    }
  ];

  const generateHealthEvents = (): HealthEvent[] => [
    {
      id: '1',
      timestamp: '2025-08-22 14:20:15',
      type: 'warning',
      component: 'Redis缓存',
      message: '内存使用率达到85%，接近告警阈值'
    },
    {
      id: '2',
      timestamp: '2025-08-22 14:15:30',
      type: 'recovery',
      component: 'RL推荐API服务',
      message: '服务恢复正常，响应时间已降低到正常范围',
      duration: 1800
    },
    {
      id: '3',
      timestamp: '2025-08-22 13:45:22',
      type: 'error',
      component: 'Q-Learning服务',
      message: '模型训练任务失败，错误率临时上升',
      duration: 900
    },
    {
      id: '4',
      timestamp: '2025-08-22 13:30:10',
      type: 'info',
      component: '监控系统',
      message: '系统自动扩容，新增2个API服务实例'
    },
    {
      id: '5',
      timestamp: '2025-08-22 12:15:45',
      type: 'warning',
      component: 'PostgreSQL数据库',
      message: '连接池使用率达到75%，建议检查慢查询'
    }
  ];

  useEffect(() => {
    setLoading(true);
    setTimeout(() => {
      setComponents(generateComponents());
      setHealthMetrics(generateHealthMetrics());
      setServiceHealth(generateServiceHealth());
      setHealthEvents(generateHealthEvents());
      
      // 计算系统整体健康度
      const avgHealth = generateServiceHealth().reduce((sum, s) => sum + s.health, 0) / generateServiceHealth().length;
      setSystemHealth(avgHealth / 100);
      
      setLoading(false);
    }, 1000);
  }, [timeRange]);

  // 系统健康趋势图配置
  const healthTrendConfig = {
    data: healthMetrics.map(m => [
      { timestamp: m.timestamp, metric: 'CPU使用率', value: m.cpu },
      { timestamp: m.timestamp, metric: '内存使用率', value: m.memory },
      { timestamp: m.timestamp, metric: '磁盘使用率', value: m.disk },
      { timestamp: m.timestamp, metric: '网络使用率', value: m.network }
    ]).flat(),
    xField: 'timestamp',
    yField: 'value',
    seriesField: 'metric',
    smooth: true,
    color: ['#1890ff', '#52c41a', '#faad14', '#f5222d'],
    legend: { position: 'top' },
  };

  // 系统健康仪表盘配置
  const healthGaugeConfig = {
    percent: systemHealth,
    range: {
      ticks: [0, 1/3, 2/3, 1],
      color: ['#F4664A', '#FAAD14', '#30BF78'],
    },
    indicator: {
      pointer: {
        style: {
          stroke: '#D0D0D0',
        },
      },
      pin: {
        style: {
          stroke: '#D0D0D0',
        },
      },
    },
    statistic: {
      content: {
        style: {
          fontSize: '36px',
          lineHeight: '36px',
        },
        formatter: () => (systemHealth * 100).toFixed(1) + '%',
      },
    },
  };

  const componentColumns: ColumnsType<SystemComponent> = [
    {
      title: '组件',
      dataIndex: 'name',
      key: 'name',
      render: (text, record) => (
        <div style={{ display: 'flex', alignItems: 'center' }}>
          {record.type === 'service' && <ApiOutlined style={{ marginRight: '8px', color: '#1890ff' }} />}
          {record.type === 'database' && <DatabaseOutlined style={{ marginRight: '8px', color: '#52c41a' }} />}
          {record.type === 'cache' && <CloudServerOutlined style={{ marginRight: '8px', color: '#faad14' }} />}
          {record.type === 'queue' && <MonitorOutlined style={{ marginRight: '8px', color: '#722ed1' }} />}
          <div>
            <strong>{text}</strong>
            <div style={{ fontSize: '12px', color: '#666' }}>{record.details}</div>
          </div>
        </div>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => {
        const config = {
          healthy: { color: 'success', icon: <CheckCircleOutlined />, text: '健康' },
          warning: { color: 'warning', icon: <ExclamationCircleOutlined />, text: '警告' },
          error: { color: 'error', icon: <CloseCircleOutlined />, text: '错误' },
          offline: { color: 'default', icon: <CloseCircleOutlined />, text: '离线' }
        };
        return <Badge status={config[status].color} text={config[status].text} />;
      },
    },
    {
      title: '可用性',
      dataIndex: 'uptime',
      key: 'uptime',
      render: (uptime) => (
        <div>
          <Progress
            percent={uptime}
            size="small"
            status={uptime > 99 ? 'success' : uptime > 95 ? 'normal' : 'exception'}
          />
          <span style={{ fontSize: '12px' }}>{uptime.toFixed(2)}%</span>
        </div>
      ),
      sorter: (a, b) => a.uptime - b.uptime
    },
    {
      title: '响应时间',
      dataIndex: 'responseTime',
      key: 'responseTime',
      render: (time) => (
        <Tag color={time < 10 ? 'green' : time < 50 ? 'orange' : 'red'}>
          {time.toFixed(1)}ms
        </Tag>
      ),
      sorter: (a, b) => a.responseTime - b.responseTime
    },
    {
      title: '最后检查',
      dataIndex: 'lastCheck',
      key: 'lastCheck',
    },
    {
      title: '依赖',
      dataIndex: 'dependencies',
      key: 'dependencies',
      render: (deps) => (
        <div>
          {deps.map((dep: string) => (
            <Tag key={dep} size="small">{dep}</Tag>
          ))}
        </div>
      ),
    }
  ];

  const serviceColumns: ColumnsType<ServiceHealth> = [
    {
      title: '服务',
      dataIndex: 'service',
      key: 'service',
    },
    {
      title: '可用性',
      dataIndex: 'availability',
      key: 'availability',
      render: (value) => (
        <Tooltip title={`目标: 99.5%`}>
          <Progress
            percent={value}
            size="small"
            status={value > 99 ? 'success' : 'exception'}
            format={() => `${value.toFixed(1)}%`}
          />
        </Tooltip>
      ),
      sorter: (a, b) => a.availability - b.availability
    },
    {
      title: '错误率',
      dataIndex: 'errorRate',
      key: 'errorRate',
      render: (value) => (
        <Tag color={value < 0.1 ? 'green' : value < 0.5 ? 'orange' : 'red'}>
          {(value * 100).toFixed(2)}%
        </Tag>
      ),
      sorter: (a, b) => a.errorRate - b.errorRate
    },
    {
      title: '吞吐量',
      dataIndex: 'throughput',
      key: 'throughput',
      render: (value) => `${value.toFixed(1)} req/s`,
      sorter: (a, b) => a.throughput - b.throughput
    },
    {
      title: '延迟',
      dataIndex: 'latency',
      key: 'latency',
      render: (value) => `${value.toFixed(1)}ms`,
      sorter: (a, b) => a.latency - b.latency
    },
    {
      title: '健康分数',
      dataIndex: 'health',
      key: 'health',
      render: (value) => (
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <Progress
            percent={value}
            size="small"
            status={value > 90 ? 'success' : value > 80 ? 'normal' : 'exception'}
            style={{ width: '60px', marginRight: '8px' }}
          />
          <span>{value}</span>
        </div>
      ),
      sorter: (a, b) => a.health - b.health
    }
  ];

  const healthyComponents = components.filter(c => c.status === 'healthy').length;
  const warningComponents = components.filter(c => c.status === 'warning').length;
  const errorComponents = components.filter(c => c.status === 'error').length;
  const avgResponseTime = components.reduce((sum, c) => sum + c.responseTime, 0) / components.length;

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <h1 style={{ margin: 0, display: 'flex', alignItems: 'center' }}>
          <HeartOutlined style={{ marginRight: '8px' }} />
          强化学习系统健康监控
        </h1>
        <Space>
          <Select value={timeRange} onChange={setTimeRange} style={{ width: 120 }}>
            <Option value="1h">最近1小时</Option>
            <Option value="24h">最近24小时</Option>
            <Option value="7d">最近7天</Option>
          </Select>
          <Button 
            type="primary" 
            icon={<ThunderboltOutlined />}
            loading={loading}
            onClick={() => {
              setComponents(generateComponents());
              setHealthMetrics(generateHealthMetrics());
              setServiceHealth(generateServiceHealth());
            }}
          >
            刷新状态
          </Button>
        </Space>
      </div>

      {/* 系统状态警告 */}
      {(errorComponents > 0 || warningComponents > 0) && (
        <Alert
          message="系统健康警告"
          description={`发现 ${errorComponents} 个错误组件和 ${warningComponents} 个警告组件，请及时处理`}
          type={errorComponents > 0 ? 'error' : 'warning'}
          showIcon
          closable
          style={{ marginBottom: '24px' }}
        />
      )}

      {/* 系统健康概览 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={6} md={4}>
          <Card>
            <Statistic
              title="系统健康度"
              value={(systemHealth * 100).toFixed(1)}
              prefix={<HeartOutlined />}
              suffix="%"
              valueStyle={{ color: systemHealth > 0.9 ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6} md={4}>
          <Card>
            <Statistic
              title="健康组件"
              value={healthyComponents}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6} md={4}>
          <Card>
            <Statistic
              title="警告组件"
              value={warningComponents}
              prefix={<ExclamationCircleOutlined />}
              valueStyle={{ color: warningComponents > 0 ? '#faad14' : '#3f8600' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6} md={4}>
          <Card>
            <Statistic
              title="错误组件"
              value={errorComponents}
              prefix={<CloseCircleOutlined />}
              valueStyle={{ color: errorComponents > 0 ? '#cf1322' : '#3f8600' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6} md={4}>
          <Card>
            <Statistic
              title="平均响应时间"
              value={avgResponseTime.toFixed(1)}
              suffix="ms"
              valueStyle={{ color: avgResponseTime < 20 ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6} md={4}>
          <Card>
            <Statistic
              title="在线服务"
              value={components.filter(c => c.status !== 'offline').length}
              suffix={`/${components.length}`}
              prefix={<SafetyCertificateOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 系统健康仪表盘和趋势 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} lg={8}>
          <Card title="系统整体健康度" loading={loading}>
            <Gauge {...healthGaugeConfig} height={250} />
          </Card>
        </Col>
        <Col xs={24} lg={16}>
          <Card title="系统资源使用趋势" loading={loading}>
            <Line {...healthTrendConfig} height={250} />
          </Card>
        </Col>
      </Row>

      {/* 组件状态详情 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} lg={16}>
          <Card title="组件健康状态" loading={loading}>
            <Table
              dataSource={components}
              columns={componentColumns}
              rowKey="id"
              pagination={false}
              size="middle"
            />
          </Card>
        </Col>
        <Col xs={24} lg={8}>
          <Card title="健康事件时间线" loading={loading}>
            <Timeline mode="left" style={{ marginTop: '16px' }}>
              {healthEvents.map(event => (
                <Timeline.Item
                  key={event.id}
                  color={
                    event.type === 'error' ? 'red' :
                    event.type === 'warning' ? 'orange' :
                    event.type === 'recovery' ? 'green' : 'blue'
                  }
                  dot={
                    event.type === 'error' ? <CloseCircleOutlined /> :
                    event.type === 'warning' ? <ExclamationCircleOutlined /> :
                    event.type === 'recovery' ? <CheckCircleOutlined /> :
                    <SafetyCertificateOutlined />
                  }
                >
                  <div>
                    <div style={{ fontSize: '12px', color: '#999' }}>
                      {event.timestamp}
                    </div>
                    <div style={{ fontWeight: 'bold' }}>
                      {event.component}
                    </div>
                    <div style={{ fontSize: '14px', marginTop: '4px' }}>
                      {event.message}
                    </div>
                    {event.duration && (
                      <div style={{ fontSize: '12px', color: '#666', marginTop: '2px' }}>
                        持续时间: {Math.floor(event.duration / 60)}分{event.duration % 60}秒
                      </div>
                    )}
                  </div>
                </Timeline.Item>
              ))}
            </Timeline>
          </Card>
        </Col>
      </Row>

      {/* 服务健康详情 */}
      <Card title="服务健康详情" loading={loading}>
        <Table
          dataSource={serviceHealth}
          columns={serviceColumns}
          rowKey="service"
          pagination={false}
          size="middle"
        />
      </Card>
    </div>
  );
};

export default RLSystemHealthPage;