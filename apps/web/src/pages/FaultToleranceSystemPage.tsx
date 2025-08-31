import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Row, 
  Col, 
  Statistic, 
  Progress, 
  Alert, 
  Button, 
  Space, 
  Tabs, 
  Tag,
  Timeline,
  Descriptions,
  Table,
  Badge
} from 'antd';
import {
  SafetyOutlined,
  ExclamationTriangleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  WarningOutlined,
  ReloadOutlined,
  SettingOutlined,
  MonitorOutlined,
  DatabaseOutlined,
  ClusterOutlined,
  HeartOutlined
} from '@ant-design/icons';

interface SystemStatus {
  system_started: boolean;
  health_summary: {
    total_components: number;
    health_ratio: number;
    active_faults: number;
    status_counts: {
      healthy: number;
      degraded: number;
      unhealthy: number;
    };
  };
  recovery_statistics: {
    success_rate: number;
    recent_recoveries: any[];
  };
  consistency_statistics: {
    consistency_rate: number;
  };
  backup_statistics: {
    total_backups: number;
    components: Record<string, any>;
  };
  active_faults: any[];
  last_updated: string;
}

interface SystemMetrics {
  fault_detection_metrics: {
    total_components: number;
    healthy_components: number;
  };
  recovery_metrics: {
    success_rate: number;
  };
  backup_metrics: {
    total_backups: number;
  };
  consistency_metrics: {
    consistency_rate: number;
  };
  system_availability: number;
  last_updated: string;
}

interface ComponentHealth {
  component_id: string;
  status: 'healthy' | 'degraded' | 'unhealthy' | 'unknown';
  last_check: string;
  response_time: number;
  error_rate: number;
  resource_usage: {
    cpu: number;
    memory: number;
  };
}

const FaultToleranceSystemPage: React.FC = () => {
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [components, setComponents] = useState<ComponentHealth[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');

  const fetchSystemStatus = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/v1/fault-tolerance/status');
      if (response.ok) {
        const data = await response.json();
        setSystemStatus(data);
      }
    } catch (error) {
      console.error('获取系统状态失败:', error);
    }
  };

  const fetchSystemMetrics = async () => {
    try {
      const response = await fetch('/api/v1/fault-tolerance/metrics');
      if (response.ok) {
        const data = await response.json();
        setSystemMetrics(data);
      }
    } catch (error) {
      console.error('获取系统指标失败:', error);
    }
  };

  const fetchComponentsList = async () => {
    try {
      const response = await fetch('/api/v1/fault-tolerance/health');
      if (response.ok) {
        const data = await response.json();
        // 模拟组件列表，实际应该从API获取
        const mockComponents: ComponentHealth[] = [
          {
            component_id: 'agent-1',
            status: 'healthy',
            last_check: new Date().toISOString(),
            response_time: 0.5,
            error_rate: 0.01,
            resource_usage: { cpu: 45, memory: 60 }
          },
          {
            component_id: 'agent-2',
            status: 'degraded',
            last_check: new Date().toISOString(),
            response_time: 1.2,
            error_rate: 0.05,
            resource_usage: { cpu: 75, memory: 80 }
          },
          {
            component_id: 'agent-3',
            status: 'unhealthy',
            last_check: new Date().toISOString(),
            response_time: 5.0,
            error_rate: 0.15,
            resource_usage: { cpu: 95, memory: 90 }
          }
        ];
        setComponents(mockComponents);
      }
    } catch (error) {
      console.error('获取组件列表失败:', error);
    }
  };

  const startSystem = async () => {
    try {
      const response = await fetch('/api/v1/fault-tolerance/system/start', {
        method: 'POST'
      });
      if (response.ok) {
        await fetchSystemStatus();
      }
    } catch (error) {
      console.error('启动系统失败:', error);
    }
  };

  const stopSystem = async () => {
    try {
      const response = await fetch('/api/v1/fault-tolerance/system/stop', {
        method: 'POST'
      });
      if (response.ok) {
        await fetchSystemStatus();
      }
    } catch (error) {
      console.error('停止系统失败:', error);
    }
  };

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await Promise.all([
        fetchSystemStatus(),
        fetchSystemMetrics(),
        fetchComponentsList()
      ]);
      setLoading(false);
    };

    loadData();
    const interval = setInterval(loadData, 10000); // 每10秒刷新
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'success';
      case 'degraded': return 'warning';
      case 'unhealthy': return 'error';
      default: return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'degraded': return <WarningOutlined style={{ color: '#faad14' }} />;
      case 'unhealthy': return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />;
      default: return <ExclamationTriangleOutlined style={{ color: '#d9d9d9' }} />;
    }
  };

  const componentColumns = [
    {
      title: '组件ID',
      dataIndex: 'component_id',
      key: 'component_id',
      render: (id: string) => <Badge status="processing" text={id} />
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={getStatusColor(status)} icon={getStatusIcon(status)}>
          {status.toUpperCase()}
        </Tag>
      )
    },
    {
      title: '响应时间',
      dataIndex: 'response_time',
      key: 'response_time',
      render: (time: number) => `${time.toFixed(2)}s`
    },
    {
      title: '错误率',
      dataIndex: 'error_rate',
      key: 'error_rate',
      render: (rate: number) => `${(rate * 100).toFixed(2)}%`
    },
    {
      title: 'CPU使用率',
      dataIndex: ['resource_usage', 'cpu'],
      key: 'cpu',
      render: (cpu: number) => (
        <Progress
          percent={cpu}
          size="small"
          status={cpu > 80 ? 'exception' : cpu > 60 ? 'active' : 'success'}
        />
      )
    },
    {
      title: '内存使用率',
      dataIndex: ['resource_usage', 'memory'],
      key: 'memory',
      render: (memory: number) => (
        <Progress
          percent={memory}
          size="small"
          status={memory > 80 ? 'exception' : memory > 60 ? 'active' : 'success'}
        />
      )
    }
  ];

  const overviewTab = (
    <div>
      <Row gutter={[16, 16]} className="mb-6">
        <Col span={24}>
          <Alert
            message="故障容错和恢复系统"
            description="实时监控分布式智能体系统的健康状态，提供自动故障检测、任务重分配和数据一致性保障"
            type="info"
            showIcon
            action={
              <Space>
                {systemStatus?.system_started ? (
                  <Button size="small" danger onClick={stopSystem}>
                    停止系统
                  </Button>
                ) : (
                  <Button size="small" type="primary" onClick={startSystem}>
                    启动系统
                  </Button>
                )}
                <Button 
                  size="small" 
                  icon={<ReloadOutlined />} 
                  onClick={() => {
                    fetchSystemStatus();
                    fetchSystemMetrics();
                    fetchComponentsList();
                  }}
                  loading={loading}
                >
                  刷新
                </Button>
              </Space>
            }
          />
        </Col>
      </Row>

      {/* 系统状态指标 */}
      <Row gutter={[16, 16]} className="mb-6">
        <Col span={6}>
          <Card>
            <Statistic
              title="系统可用性"
              value={systemMetrics?.system_availability || 0}
              suffix="%"
              precision={2}
              valueStyle={{ 
                color: (systemMetrics?.system_availability || 0) > 99 ? '#3f8600' : '#cf1322' 
              }}
              prefix={<SafetyOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="健康组件比率"
              value={systemStatus ? systemStatus.health_summary.health_ratio * 100 : 0}
              suffix="%"
              precision={1}
              valueStyle={{ 
                color: systemStatus && systemStatus.health_summary.health_ratio > 0.9 ? '#3f8600' : '#cf1322' 
              }}
              prefix={<HeartOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="恢复成功率"
              value={systemStatus ? systemStatus.recovery_statistics.success_rate * 100 : 0}
              suffix="%"
              precision={1}
              valueStyle={{ 
                color: systemStatus && systemStatus.recovery_statistics.success_rate > 0.95 ? '#3f8600' : '#cf1322' 
              }}
              prefix={<SyncOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="活跃故障数"
              value={systemStatus?.active_faults.length || 0}
              valueStyle={{ 
                color: (systemStatus?.active_faults.length || 0) === 0 ? '#3f8600' : '#cf1322' 
              }}
              prefix={<ExclamationTriangleOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* 组件状态分布 */}
      <Row gutter={[16, 16]} className="mb-6">
        <Col span={8}>
          <Card title="组件状态分布" size="small">
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="flex items-center">
                  <CheckCircleOutlined style={{ color: '#52c41a', marginRight: 8 }} />
                  健康组件
                </span>
                <Badge 
                  count={systemStatus?.health_summary.status_counts.healthy || 0} 
                  style={{ backgroundColor: '#52c41a' }}
                />
              </div>
              <div className="flex justify-between items-center">
                <span className="flex items-center">
                  <WarningOutlined style={{ color: '#faad14', marginRight: 8 }} />
                  降级组件
                </span>
                <Badge 
                  count={systemStatus?.health_summary.status_counts.degraded || 0} 
                  style={{ backgroundColor: '#faad14' }}
                />
              </div>
              <div className="flex justify-between items-center">
                <span className="flex items-center">
                  <CloseCircleOutlined style={{ color: '#ff4d4f', marginRight: 8 }} />
                  异常组件
                </span>
                <Badge 
                  count={systemStatus?.health_summary.status_counts.unhealthy || 0} 
                  style={{ backgroundColor: '#ff4d4f' }}
                />
              </div>
            </div>
          </Card>
        </Col>
        <Col span={8}>
          <Card title="数据一致性" size="small">
            <div className="text-center">
              <Progress
                type="circle"
                percent={systemStatus ? systemStatus.consistency_statistics.consistency_rate * 100 : 0}
                format={(percent) => `${percent?.toFixed(1)}%`}
                status={systemStatus && systemStatus.consistency_statistics.consistency_rate > 0.98 ? 'success' : 'active'}
              />
              <div className="mt-2 text-gray-600">数据一致性率</div>
            </div>
          </Card>
        </Col>
        <Col span={8}>
          <Card title="备份统计" size="small">
            <Descriptions size="small" column={1}>
              <Descriptions.Item label="备份总数">
                {systemStatus?.backup_statistics.total_backups || 0}
              </Descriptions.Item>
              <Descriptions.Item label="备份组件">
                {Object.keys(systemStatus?.backup_statistics.components || {}).length}
              </Descriptions.Item>
              <Descriptions.Item label="最后更新">
                {systemStatus ? new Date(systemStatus.last_updated).toLocaleString() : '-'}
              </Descriptions.Item>
            </Descriptions>
          </Card>
        </Col>
      </Row>

      {/* 组件详情表格 */}
      <Card title="组件状态详情">
        <Table
          columns={componentColumns}
          dataSource={components}
          rowKey="component_id"
          loading={loading}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total) => `共 ${total} 个组件`
          }}
        />
      </Card>
    </div>
  );

  const faultHistoryTab = (
    <Card title="故障历史记录">
      <Timeline>
        {systemStatus?.active_faults.map((fault, index) => (
          <Timeline.Item
            key={index}
            color={fault.severity === 'high' ? 'red' : fault.severity === 'medium' ? 'orange' : 'blue'}
            dot={<ExclamationTriangleOutlined />}
          >
            <div>
              <strong>{fault.fault_type}</strong> - {fault.description}
              <br />
              <span className="text-gray-500">
                影响组件: {fault.affected_components.join(', ')} | 
                严重程度: {fault.severity} | 
                检测时间: {new Date(fault.detected_at).toLocaleString()}
              </span>
            </div>
          </Timeline.Item>
        ))}
        {(!systemStatus?.active_faults || systemStatus.active_faults.length === 0) && (
          <Timeline.Item color="green" dot={<CheckCircleOutlined />}>
            <div>
              <strong>系统正常</strong>
              <br />
              <span className="text-gray-500">当前没有活跃的故障事件</span>
            </div>
          </Timeline.Item>
        )}
      </Timeline>
    </Card>
  );

  const tabItems = [
    {
      key: 'overview',
      label: (
        <span>
          <MonitorOutlined />
          系统总览
        </span>
      ),
      children: overviewTab
    },
    {
      key: 'faults',
      label: (
        <span>
          <ExclamationTriangleOutlined />
          故障历史
        </span>
      ),
      children: faultHistoryTab
    }
  ];

  return (
    <div className="fault-tolerance-system-page p-6">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold mb-2">故障容错和恢复系统</h1>
          <p className="text-gray-600">智能体故障检测、任务重分配、数据备份和一致性保障</p>
        </div>
        <Space>
          <Button icon={<SettingOutlined />} href="/fault-tolerance/settings">
            系统设置
          </Button>
          <Button type="primary" icon={<DatabaseOutlined />} href="/fault-tolerance/backup">
            备份管理
          </Button>
        </Space>
      </div>

      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        items={tabItems}
        size="large"
      />
    </div>
  );
};

export default FaultToleranceSystemPage;