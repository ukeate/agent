import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Tabs, 
  Typography, 
  Row, 
  Col, 
  Button, 
  Alert, 
  Tag,
  Statistic,
  Space,
  Progress,
  List
} from 'antd';
import { 
  WifiOutlined, 
  SignalFilled, 
  ThunderboltOutlined, 
  ClockCircleOutlined, 
  DatabaseOutlined,
  ReloadOutlined, 
  LoadingOutlined,
  WarningOutlined,
  CheckCircleOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { TabPane } = Tabs;

interface NetworkMetrics {
  timestamp: string;
  latency_ms: number;
  packet_loss: number;
  bandwidth_mbps: number;
  connection_quality: number;
  dns_resolution_ms: number;
}

interface NetworkAlert {
  id: string;
  type: 'high_latency' | 'packet_loss' | 'connection_drop' | 'dns_failure';
  severity: 'warning' | 'error' | 'critical';
  message: string;
  timestamp: string;
  resolved: boolean;
}

const NetworkMonitorDetailPage: React.FC = () => {
  const [metrics, setMetrics] = useState<NetworkMetrics[]>([]);
  const [alerts, setAlerts] = useState<NetworkAlert[]>([]);
  const [currentMetrics, setCurrentMetrics] = useState<NetworkMetrics | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchNetworkData = async () => {
    try {
      // 模拟网络指标数据
      const now = Date.now();
      const mockMetrics: NetworkMetrics[] = Array.from({ length: 10 }, (_, i) => ({
        timestamp: new Date(now - (i * 30000)).toISOString(), // 每30秒一个数据点
        latency_ms: 20 + Math.random() * 80,
        packet_loss: Math.random() * 0.05, // 0-5% 丢包率
        bandwidth_mbps: 95 + Math.random() * 10,
        connection_quality: 0.8 + Math.random() * 0.2,
        dns_resolution_ms: 5 + Math.random() * 15
      })).reverse();

      const mockAlerts: NetworkAlert[] = [
        {
          id: 'alert-001',
          type: 'high_latency',
          severity: 'warning',
          message: '网络延迟超过阈值 (>100ms)',
          timestamp: new Date(now - 180000).toISOString(),
          resolved: false
        },
        {
          id: 'alert-002',
          type: 'packet_loss',
          severity: 'error',
          message: '检测到丢包率过高 (>3%)',
          timestamp: new Date(now - 300000).toISOString(),
          resolved: true
        }
      ];

      setMetrics(mockMetrics);
      setCurrentMetrics(mockMetrics[mockMetrics.length - 1]);
      setAlerts(mockAlerts);
    } catch (error) {
      console.error('获取网络监控数据失败:', error);
    }
  };

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await fetchNetworkData();
      setLoading(false);
    };

    loadData();
    const interval = setInterval(loadData, 10000);
    return () => clearInterval(interval);
  }, []);

  const getAlertSeverityColor = (severity: string) => {
    switch (severity) {
      case 'warning': return 'orange';
      case 'error': return 'red';
      case 'critical': return 'error';
      default: return 'default';
    }
  };

  const getAlertTypeIcon = (type: string) => {
    switch (type) {
      case 'high_latency': return <ClockCircleOutlined />;
      case 'packet_loss': return <WarningOutlined />;
      case 'connection_drop': return <WifiOutlined />;
      case 'dns_failure': return <DatabaseOutlined />;
      default: return <WarningOutlined />;
    }
  };

  const getConnectionQualityStatus = (quality: number) => {
    if (quality >= 0.9) return { color: 'success', text: '优秀' };
    if (quality >= 0.7) return { color: 'processing', text: '良好' };
    if (quality >= 0.5) return { color: 'warning', text: '一般' };
    return { color: 'error', text: '差' };
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '50px' }}>
        <LoadingOutlined style={{ fontSize: 24 }} />
        <div style={{ marginTop: 16 }}>加载网络监控数据中...</div>
      </div>
    );
  }

  const qualityStatus = getConnectionQualityStatus(currentMetrics?.connection_quality || 0);

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Title level={2}>
          <WifiOutlined style={{ marginRight: '12px' }} />
          网络监控详情
        </Title>
        <Space>
          <Button onClick={() => window.location.reload()} icon={<ReloadOutlined />}>
            刷新
          </Button>
        </Space>
      </div>

      {/* 实时指标概览 */}
      <Row gutter={24} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="当前延迟"
              value={currentMetrics?.latency_ms.toFixed(1)}
              suffix="ms"
              prefix={<ClockCircleOutlined />}
              valueStyle={{ 
                color: (currentMetrics?.latency_ms || 0) > 100 ? '#ff4d4f' : '#3f8600' 
              }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="丢包率"
              value={((currentMetrics?.packet_loss || 0) * 100).toFixed(2)}
              suffix="%"
              prefix={<WarningOutlined />}
              valueStyle={{ 
                color: (currentMetrics?.packet_loss || 0) > 0.03 ? '#ff4d4f' : '#3f8600' 
              }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="带宽利用率"
              value={currentMetrics?.bandwidth_mbps.toFixed(1)}
              suffix="Mbps"
              prefix={<ThunderboltOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="连接质量"
              value={((currentMetrics?.connection_quality || 0) * 100).toFixed(1)}
              suffix="%"
              prefix={<SignalFilled />}
              valueStyle={{ color: qualityStatus.color === 'success' ? '#3f8600' : qualityStatus.color === 'error' ? '#ff4d4f' : '#faad14' }}
            />
            <div style={{ marginTop: '8px' }}>
              <Tag color={qualityStatus.color}>{qualityStatus.text}</Tag>
            </div>
          </Card>
        </Col>
      </Row>

      <Tabs defaultActiveKey="realtime">
        <TabPane
          tab={
            <span>
              <SignalFilled />
              实时监控
            </span>
          }
          key="realtime"
        >
          <Row gutter={16}>
            <Col span={12}>
              <Card title="网络延迟趋势" style={{ marginBottom: '16px' }}>
                <div style={{ height: '200px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Text type="secondary">延迟图表 (模拟数据可视化)</Text>
                </div>
                <div style={{ marginTop: '16px' }}>
                  <Text strong>最近10个数据点平均延迟: </Text>
                  <Text>
                    {(metrics.reduce((sum, m) => sum + m.latency_ms, 0) / metrics.length).toFixed(1)} ms
                  </Text>
                </div>
              </Card>
            </Col>
            <Col span={12}>
              <Card title="连接质量分析" style={{ marginBottom: '16px' }}>
                <div style={{ marginBottom: '16px' }}>
                  <Text strong>当前连接质量:</Text>
                  <Progress 
                    percent={(currentMetrics?.connection_quality || 0) * 100}
                    status={qualityStatus.color === 'success' ? 'success' : qualityStatus.color === 'error' ? 'exception' : 'active'}
                    style={{ marginTop: '8px' }}
                  />
                </div>
                
                <div style={{ marginBottom: '12px' }}>
                  <Text strong>DNS解析时间: </Text>
                  <Text>{currentMetrics?.dns_resolution_ms.toFixed(1)} ms</Text>
                </div>

                <div>
                  <Text strong>网络稳定性: </Text>
                  <Text>
                    {metrics.filter(m => m.packet_loss < 0.01).length}/{metrics.length} 个数据点稳定
                  </Text>
                </div>
              </Card>
            </Col>
          </Row>

          <Card title="历史指标详情">
            <List
              dataSource={metrics.slice().reverse().slice(0, 5)} // 显示最近5条
              renderItem={(metric, index) => (
                <List.Item>
                  <Row style={{ width: '100%' }} gutter={16}>
                    <Col span={4}>
                      <Text strong>{new Date(metric.timestamp).toLocaleTimeString()}</Text>
                    </Col>
                    <Col span={4}>
                      <Text>延迟: {metric.latency_ms.toFixed(1)}ms</Text>
                    </Col>
                    <Col span={4}>
                      <Text>丢包: {(metric.packet_loss * 100).toFixed(2)}%</Text>
                    </Col>
                    <Col span={4}>
                      <Text>带宽: {metric.bandwidth_mbps.toFixed(1)}Mbps</Text>
                    </Col>
                    <Col span={4}>
                      <Text>质量: {(metric.connection_quality * 100).toFixed(1)}%</Text>
                    </Col>
                    <Col span={4}>
                      <Tag color={metric.connection_quality > 0.8 ? 'success' : metric.connection_quality > 0.6 ? 'warning' : 'error'}>
                        {metric.connection_quality > 0.8 ? '优秀' : metric.connection_quality > 0.6 ? '良好' : '需优化'}
                      </Tag>
                    </Col>
                  </Row>
                </List.Item>
              )}
            />
          </Card>
        </TabPane>

        <TabPane
          tab={
            <span>
              <WarningOutlined />
              告警管理
            </span>
          }
          key="alerts"
        >
          <div style={{ marginBottom: '16px' }}>
            <Alert
              message="网络告警概览"
              description={`当前有 ${alerts.filter(a => !a.resolved).length} 个未解决的告警，${alerts.filter(a => a.resolved).length} 个已解决的告警。`}
              type={alerts.filter(a => !a.resolved).length > 0 ? 'warning' : 'success'}
              showIcon
            />
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {alerts.map((alert) => (
              <Card key={alert.id} size="small">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    {getAlertTypeIcon(alert.type)}
                    <div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
                        <Text strong>{alert.message}</Text>
                        <Tag color={getAlertSeverityColor(alert.severity)}>
                          {alert.severity}
                        </Tag>
                        {alert.resolved ? (
                          <Tag color="success">
                            <CheckCircleOutlined />
                            已解决
                          </Tag>
                        ) : (
                          <Tag color="processing">处理中</Tag>
                        )}
                      </div>
                      <Text type="secondary" style={{ fontSize: '12px' }}>
                        {new Date(alert.timestamp).toLocaleString()}
                      </Text>
                    </div>
                  </div>
                  
                  {!alert.resolved && (
                    <Button size="small" type="primary">
                      处理告警
                    </Button>
                  )}
                </div>
              </Card>
            ))}
          </div>
        </TabPane>

        <TabPane
          tab={
            <span>
              <DatabaseOutlined />
              诊断信息
            </span>
          }
          key="diagnostics"
        >
          <Row gutter={16}>
            <Col span={12}>
              <Card title="网络诊断" style={{ marginBottom: '16px' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  <div>
                    <Text strong>网络接口状态: </Text>
                    <Tag color="success">正常</Tag>
                  </div>
                  <div>
                    <Text strong>路由表状态: </Text>
                    <Tag color="success">正常</Tag>
                  </div>
                  <div>
                    <Text strong>DNS服务器: </Text>
                    <Text>8.8.8.8, 8.8.4.4</Text>
                  </div>
                  <div>
                    <Text strong>网关响应: </Text>
                    <Tag color="success">正常</Tag>
                  </div>
                </div>
              </Card>
            </Col>
            
            <Col span={12}>
              <Card title="系统资源" style={{ marginBottom: '16px' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  <div>
                    <Text strong>网络缓冲区使用率:</Text>
                    <Progress percent={25} size="small" style={{ marginTop: '4px' }} />
                  </div>
                  <div>
                    <Text strong>连接数:</Text>
                    <Progress percent={60} size="small" style={{ marginTop: '4px' }} />
                  </div>
                  <div>
                    <Text strong>套接字使用率:</Text>
                    <Progress percent={35} size="small" style={{ marginTop: '4px' }} />
                  </div>
                </div>
              </Card>
            </Col>
          </Row>

          <Card title="性能建议">
            <List
              dataSource={[
                {
                  type: 'success',
                  message: '网络延迟在正常范围内，连接质量良好'
                },
                {
                  type: 'warning', 
                  message: '建议监控丢包率，如持续超过1%需要检查网络设备'
                },
                {
                  type: 'info',
                  message: 'DNS解析时间正常，建议定期清理DNS缓存'
                }
              ]}
              renderItem={(item) => (
                <List.Item>
                  <Alert
                    message={item.message}
                    type={item.type as any}
                    showIcon
                    style={{ width: '100%' }}
                  />
                </List.Item>
              )}
            />
          </Card>
        </TabPane>
      </Tabs>
    </div>
  );
};

export default NetworkMonitorDetailPage;