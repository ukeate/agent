import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Progress,
  Badge,
  Typography,
  Alert,
  Descriptions,
  Tag,
  Space,
  Button,
  Timeline,
  Table
} from 'antd';
import {
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  ClockCircleOutlined,
  DatabaseOutlined,
  ClusterOutlined,
  ThunderboltOutlined,
  ReloadOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;

interface SystemStatus {
  cluster_health: 'healthy' | 'degraded' | 'critical';
  total_nodes: number;
  active_nodes: number;
  leader_node: string;
  consensus_status: 'stable' | 'electing' | 'split';
  task_throughput: number;
  avg_response_time: number;
  error_rate: number;
  uptime: number;
}

const DistributedTaskSystemStatusPage: React.FC = () => {
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    cluster_health: 'healthy',
    total_nodes: 5,
    active_nodes: 5,
    leader_node: 'node_1',
    consensus_status: 'stable',
    task_throughput: 1250,
    avg_response_time: 125,
    error_rate: 0.5,
    uptime: 99.8
  });

  const [alerts] = useState([
    {
      id: '1',
      level: 'info',
      message: '系统运行正常',
      timestamp: new Date().toISOString()
    },
    {
      id: '2', 
      level: 'warning',
      message: 'node_3 CPU使用率较高',
      timestamp: new Date(Date.now() - 300000).toISOString()
    }
  ]);

  useEffect(() => {
    const interval = setInterval(() => {
      setSystemStatus(prev => ({
        ...prev,
        task_throughput: prev.task_throughput + (Math.random() - 0.5) * 100,
        avg_response_time: Math.max(50, prev.avg_response_time + (Math.random() - 0.5) * 20),
        error_rate: Math.max(0, prev.error_rate + (Math.random() - 0.5) * 0.2)
      }));
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const getHealthColor = (health: string) => {
    switch (health) {
      case 'healthy': return 'success';
      case 'degraded': return 'warning'; 
      case 'critical': return 'error';
      default: return 'default';
    }
  };

  const getConsensusColor = (status: string) => {
    switch (status) {
      case 'stable': return 'success';
      case 'electing': return 'processing';
      case 'split': return 'error';
      default: return 'default';
    }
  };

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <Title level={2}>系统状态总览</Title>
      
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={8}>
          <Card>
            <Statistic
              title="集群健康状态"
              value={systemStatus.cluster_health}
              prefix={<Badge status={getHealthColor(systemStatus.cluster_health)} />}
              valueStyle={{ 
                color: systemStatus.cluster_health === 'healthy' ? '#3f8600' : 
                       systemStatus.cluster_health === 'degraded' ? '#faad14' : '#f5222d' 
              }}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic
              title="活跃节点"
              value={systemStatus.active_nodes}
              suffix={`/ ${systemStatus.total_nodes}`}
              prefix={<ClusterOutlined />}
              valueStyle={{ color: systemStatus.active_nodes === systemStatus.total_nodes ? '#3f8600' : '#faad14' }}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic
              title="系统可用性"
              value={systemStatus.uptime}
              precision={2}
              suffix="%"
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="任务吞吐量"
              value={Math.floor(systemStatus.task_throughput)}
              suffix="tasks/min"
              prefix={<ThunderboltOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均响应时间"
              value={Math.floor(systemStatus.avg_response_time)}
              suffix="ms"
              prefix={<ClockCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="错误率"
              value={systemStatus.error_rate}
              precision={2}
              suffix="%"
              prefix={<ExclamationCircleOutlined />}
              valueStyle={{ color: systemStatus.error_rate > 1 ? '#f5222d' : '#3f8600' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="共识状态"
              value={systemStatus.consensus_status}
              prefix={<Badge status={getConsensusColor(systemStatus.consensus_status)} />}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={16}>
        <Col span={16}>
          <Card 
            title="系统详细信息"
            extra={<Button icon={<ReloadOutlined />}>刷新</Button>}
          >
            <Descriptions bordered column={2}>
              <Descriptions.Item label="Leader节点">
                <Tag color="gold">{systemStatus.leader_node}</Tag>
              </Descriptions.Item>
              <Descriptions.Item label="Raft任期">15</Descriptions.Item>
              <Descriptions.Item label="日志索引">2,847</Descriptions.Item>
              <Descriptions.Item label="提交索引">2,847</Descriptions.Item>
              <Descriptions.Item label="网络分区">无</Descriptions.Item>
              <Descriptions.Item label="存储使用">
                <Progress percent={68} size="small" />
              </Descriptions.Item>
              <Descriptions.Item label="内存使用">
                <Progress percent={45} size="small" />
              </Descriptions.Item>
              <Descriptions.Item label="CPU使用">
                <Progress percent={32} size="small" />
              </Descriptions.Item>
            </Descriptions>
          </Card>
        </Col>
        <Col span={8}>
          <Card title="系统告警">
            <Timeline size="small">
              {alerts.map(alert => (
                <Timeline.Item
                  key={alert.id}
                  color={alert.level === 'warning' ? 'orange' : 'green'}
                >
                  <div>
                    <Text>{alert.message}</Text>
                    <br />
                    <Text type="secondary" style={{ fontSize: '12px' }}>
                      {new Date(alert.timestamp).toLocaleString()}
                    </Text>
                  </div>
                </Timeline.Item>
              ))}
            </Timeline>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default DistributedTaskSystemStatusPage;