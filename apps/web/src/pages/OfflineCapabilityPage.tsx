import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Tabs, 
  Typography, 
  Row, 
  Col, 
  Button, 
  Alert, 
  Progress,
  Tag,
  Statistic,
  Space,
  Divider,
  Badge
} from 'antd';
import { 
  WifiOutlined, 
  DatabaseOutlined, 
  HddOutlined, 
  BranchesOutlined, 
  WarningOutlined, 
  CheckCircleOutlined, 
  ClockCircleOutlined,
  DashboardOutlined, 
  ThunderboltOutlined, 
  ReloadOutlined, 
  DownloadOutlined, 
  UploadOutlined,
  LoadingOutlined,
  ExclamationCircleOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { TabPane } = Tabs;

interface OfflineStatus {
  mode: string;
  network_status: string;
  connection_quality: number;
  pending_operations: number;
  has_conflicts: boolean;
  sync_in_progress: boolean;
  last_sync_at: string | null;
  last_heartbeat: string;
}

interface NetworkStats {
  network: {
    current_status: string;
    current_latency_ms: number;
    current_packet_loss: number;
    connection_quality: number;
    uptime_percentage: number;
    average_latency_ms: number;
  };
  mode_switcher: {
    current_mode: string;
    last_online_time: string;
    last_offline_time: string | null;
  };
}

interface OfflineStatistics {
  session: {
    session_id: string;
    mode: string;
    uptime_hours: number;
  };
  memory: {
    total_memories: number;
    memory_types: Record<string, number>;
    storage_usage_mb: number;
  };
  model_cache: {
    cached_models: number;
    cache_size_mb: number;
    cache_hit_rate: number;
  };
  inference: {
    total_requests: number;
    offline_requests: number;
    cache_hit_rate: number;
  };
  reasoning: {
    total_workflows: number;
    active_workflows: number;
    completed_workflows: number;
  };
  network: NetworkStats['network'];
}

const OfflineCapabilityPage: React.FC = () => {
  const [offlineStatus, setOfflineStatus] = useState<OfflineStatus | null>(null);
  const [networkStats, setNetworkStats] = useState<NetworkStats | null>(null);
  const [offlineStats, setOfflineStats] = useState<OfflineStatistics | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchOfflineStatus = async () => {
    try {
      const response = await fetch('/api/v1/offline/status');
      const data = await response.json();
      setOfflineStatus(data);
    } catch (error) {
      console.error('获取离线状态失败:', error);
    }
  };

  const fetchNetworkStats = async () => {
    try {
      const response = await fetch('/api/v1/offline/network');
      const data = await response.json();
      setNetworkStats(data);
    } catch (error) {
      console.error('获取网络统计失败:', error);
    }
  };

  const fetchOfflineStats = async () => {
    try {
      const response = await fetch('/api/v1/offline/statistics');
      const data = await response.json();
      setOfflineStats(data);
    } catch (error) {
      console.error('获取离线统计失败:', error);
    }
  };

  const handleForceSync = async () => {
    try {
      await fetch('/api/v1/offline/sync', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ force: true, batch_size: 100 })
      });
      await fetchOfflineStatus();
    } catch (error) {
      console.error('强制同步失败:', error);
    }
  };

  const handleModeSwitch = async (mode: string) => {
    try {
      await fetch(`/api/v1/offline/mode/${mode}`, { method: 'POST' });
      await fetchOfflineStatus();
    } catch (error) {
      console.error('切换模式失败:', error);
    }
  };

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await Promise.all([
        fetchOfflineStatus(),
        fetchNetworkStats(),
        fetchOfflineStats()
      ]);
      setLoading(false);
    };

    loadData();
    const interval = setInterval(loadData, 5000); // 5秒刷新

    return () => clearInterval(interval);
  }, []);

  const getNetworkStatusIcon = (status: string) => {
    switch (status) {
      case 'connected': return <WifiOutlined style={{ color: '#52c41a' }} />;
      case 'weak': return <WifiOutlined style={{ color: '#faad14' }} />;
      case 'disconnected': return <WifiOutlined style={{ color: '#ff4d4f' }} />;
      default: return <WifiOutlined style={{ color: '#8c8c8c' }} />;
    }
  };

  const getNetworkStatusColor = (status: string) => {
    switch (status) {
      case 'connected': return 'success';
      case 'weak': return 'warning';
      case 'disconnected': return 'error';
      default: return 'default';
    }
  };

  const getModeColor = (mode: string) => {
    switch (mode) {
      case 'online': return 'success';
      case 'offline': return 'error';
      case 'auto': return 'processing';
      default: return 'default';
    }
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '50px' }}>
        <LoadingOutlined style={{ fontSize: 24 }} />
        <div style={{ marginTop: 16 }}>加载离线状态中...</div>
      </div>
    );
  }

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Title level={2}>
          <WifiOutlined style={{ marginRight: '12px' }} />
          离线能力监控
        </Title>
        <Space>
          <Button onClick={() => window.location.reload()} icon={<ReloadOutlined />}>
            刷新
          </Button>
          <Button onClick={handleForceSync} icon={<BranchesOutlined />}>
            强制同步
          </Button>
        </Space>
      </div>

      {/* 核心状态卡片 */}
      <Row gutter={24} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card 
            title={
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Text>网络状态</Text>
                {getNetworkStatusIcon(offlineStatus?.network_status || 'unknown')}
              </div>
            }
          >
            <div style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '8px' }}>
              <Tag color={getNetworkStatusColor(offlineStatus?.network_status || 'unknown')}>
                {offlineStatus?.network_status || 'Unknown'}
              </Tag>
            </div>
            <Text type="secondary">
              连接质量: {((offlineStatus?.connection_quality || 0) * 100).toFixed(1)}%
            </Text>
          </Card>
        </Col>

        <Col span={6}>
          <Card 
            title={
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Text>工作模式</Text>
                <DashboardOutlined />
              </div>
            }
          >
            <div style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '8px' }}>
              <Tag color={getModeColor(offlineStatus?.mode || 'unknown')}>
                {offlineStatus?.mode || 'Unknown'}
              </Tag>
            </div>
            <div style={{ display: 'flex', gap: '4px', marginTop: '8px' }}>
              <Button size="small" onClick={() => handleModeSwitch('online')}>
                在线
              </Button>
              <Button size="small" onClick={() => handleModeSwitch('offline')}>
                离线
              </Button>
              <Button size="small" onClick={() => handleModeSwitch('auto')}>
                自动
              </Button>
            </div>
          </Card>
        </Col>

        <Col span={6}>
          <Card 
            title={
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Text>待同步操作</Text>
                <DatabaseOutlined />
              </div>
            }
          >
            <div style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '8px' }}>
              {offlineStatus?.pending_operations || 0}
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '8px' }}>
              {offlineStatus?.sync_in_progress ? (
                <Tag color="processing">
                  <ReloadOutlined style={{ marginRight: '4px' }} />
                  同步中
                </Tag>
              ) : (
                <Tag>空闲</Tag>
              )}
            </div>
          </Card>
        </Col>

        <Col span={6}>
          <Card 
            title={
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Text>冲突状态</Text>
                {offlineStatus?.has_conflicts ? (
                  <ExclamationCircleOutlined style={{ color: '#faad14' }} />
                ) : (
                  <CheckCircleOutlined style={{ color: '#52c41a' }} />
                )}
              </div>
            }
          >
            <div style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '8px' }}>
              {offlineStatus?.has_conflicts ? (
                <Tag color="error">有冲突</Tag>
              ) : (
                <Tag color="success">无冲突</Tag>
              )}
            </div>
            <Text type="secondary" style={{ fontSize: '12px' }}>
              最后同步: {offlineStatus?.last_sync_at ? 
                new Date(offlineStatus.last_sync_at).toLocaleString() : '从未同步'}
            </Text>
          </Card>
        </Col>
      </Row>

      {/* 详细监控面板 */}
      <Tabs defaultActiveKey="network">
        <TabPane
          tab={
            <span>
              <WifiOutlined />
              网络监控
            </span>
          }
          key="network"
        >
          <Card 
            title={
              <div style={{ display: 'flex', alignItems: 'center' }}>
                <WifiOutlined style={{ marginRight: '8px' }} />
                网络质量监控
              </div>
            }
          >
            <Row gutter={16} style={{ marginBottom: '16px' }}>
              <Col span={8}>
                <Text strong>当前延迟</Text>
                <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                  {networkStats?.network?.current_latency_ms?.toFixed(1) || 0} ms
                </div>
              </Col>
              <Col span={8}>
                <Text strong>丢包率</Text>
                <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                  {((networkStats?.network?.current_packet_loss || 0) * 100).toFixed(2)}%
                </div>
              </Col>
              <Col span={8}>
                <Text strong>在线时间</Text>
                <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                  {networkStats?.network?.uptime_percentage?.toFixed(1) || 0}%
                </div>
              </Col>
            </Row>
            
            <div style={{ marginBottom: '16px' }}>
              <Text strong>连接质量</Text>
              <Progress percent={(networkStats?.network?.connection_quality || 0) * 100} style={{ marginTop: '8px' }} />
            </div>

            <Row gutter={16}>
              <Col span={12}>
                <Text strong>当前模式</Text>
                <div style={{ marginTop: '4px' }}>
                  <Tag color={getModeColor(networkStats?.mode_switcher?.current_mode || 'unknown')}>
                    {networkStats?.mode_switcher?.current_mode || 'Unknown'}
                  </Tag>
                </div>
              </Col>
              <Col span={12}>
                <Text strong>最后在线时间</Text>
                <div style={{ marginTop: '4px' }}>
                  <Text>{networkStats?.mode_switcher?.last_online_time ? 
                    new Date(networkStats.mode_switcher.last_online_time).toLocaleString() : 'Unknown'}</Text>
                </div>
              </Col>
            </Row>
          </Card>
        </TabPane>

        <TabPane
          tab={
            <span>
              <HddOutlined />
              存储状态
            </span>
          }
          key="storage"
        >
          <Row gutter={16}>
            <Col span={12}>
              <Card 
                title={
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <HddOutlined style={{ marginRight: '8px' }} />
                    模型缓存
                  </div>
                }
              >
                <div style={{ marginBottom: '16px' }}>
                  <Text strong>缓存模型数量</Text>
                  <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                    {offlineStats?.model_cache?.cached_models || 0}
                  </div>
                </div>
                <div style={{ marginBottom: '16px' }}>
                  <Text strong>缓存大小</Text>
                  <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                    {offlineStats?.model_cache?.cache_size_mb?.toFixed(1) || 0} MB
                  </div>
                </div>
                <div>
                  <Text strong>缓存命中率</Text>
                  <Progress 
                    percent={(offlineStats?.model_cache?.cache_hit_rate || 0) * 100} 
                    style={{ marginTop: '8px' }} 
                  />
                  <Text type="secondary" style={{ fontSize: '12px', marginTop: '4px' }}>
                    {((offlineStats?.model_cache?.cache_hit_rate || 0) * 100).toFixed(1)}%
                  </Text>
                </div>
              </Card>
            </Col>

            <Col span={12}>
              <Card 
                title={
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <DatabaseOutlined style={{ marginRight: '8px' }} />
                    记忆存储
                  </div>
                }
              >
                <div style={{ marginBottom: '16px' }}>
                  <Text strong>总记忆数量</Text>
                  <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                    {offlineStats?.memory?.total_memories || 0}
                  </div>
                </div>
                <div style={{ marginBottom: '16px' }}>
                  <Text strong>存储使用</Text>
                  <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                    {offlineStats?.memory?.storage_usage_mb?.toFixed(1) || 0} MB
                  </div>
                </div>
                <div>
                  <Text strong>记忆类型分布</Text>
                  <div style={{ marginTop: '8px' }}>
                    {Object.entries(offlineStats?.memory?.memory_types || {}).map(([type, count]) => (
                      <div key={type} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                        <Text style={{ fontSize: '12px' }}>{type}</Text>
                        <Text strong style={{ fontSize: '12px' }}>{count}</Text>
                      </div>
                    ))}
                  </div>
                </div>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane
          tab={
            <span>
              <ThunderboltOutlined />
              推理引擎
            </span>
          }
          key="inference"
        >
          <Card 
            title={
              <div style={{ display: 'flex', alignItems: 'center' }}>
                <ThunderboltOutlined style={{ marginRight: '8px' }} />
                推理引擎状态
              </div>
            }
          >
            <Row gutter={16} style={{ marginBottom: '16px' }}>
              <Col span={8}>
                <Text strong>总请求数</Text>
                <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                  {offlineStats?.inference?.total_requests || 0}
                </div>
              </Col>
              <Col span={8}>
                <Text strong>离线请求数</Text>
                <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                  {offlineStats?.inference?.offline_requests || 0}
                </div>
              </Col>
              <Col span={8}>
                <Text strong>离线占比</Text>
                <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                  {offlineStats?.inference?.total_requests ? 
                    ((offlineStats.inference.offline_requests / offlineStats.inference.total_requests) * 100).toFixed(1) : 0}%
                </div>
              </Col>
            </Row>
            
            <div>
              <Text strong>推理缓存命中率</Text>
              <Progress 
                percent={(offlineStats?.inference?.cache_hit_rate || 0) * 100} 
                style={{ marginTop: '8px' }} 
              />
              <Text type="secondary" style={{ fontSize: '12px' }}>
                {((offlineStats?.inference?.cache_hit_rate || 0) * 100).toFixed(1)}%
              </Text>
            </div>
          </Card>
        </TabPane>

        <TabPane
          tab={
            <span>
              <DatabaseOutlined />
              记忆系统
            </span>
          }
          key="memory"
        >
          <Card 
            title={
              <div style={{ display: 'flex', alignItems: 'center' }}>
                <DatabaseOutlined style={{ marginRight: '8px' }} />
                记忆系统详情
              </div>
            }
          >
            <Row gutter={16} style={{ marginBottom: '16px' }}>
              <Col span={12}>
                <Text strong>会话ID</Text>
                <div style={{ 
                  fontSize: '12px', 
                  fontFamily: 'monospace', 
                  backgroundColor: '#f5f5f5', 
                  padding: '4px', 
                  borderRadius: '4px',
                  marginTop: '4px'
                }}>
                  {offlineStats?.session?.session_id || 'Unknown'}
                </div>
              </Col>
              <Col span={12}>
                <Text strong>运行时间</Text>
                <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                  {offlineStats?.session?.uptime_hours?.toFixed(1) || 0} 小时
                </div>
              </Col>
            </Row>
            
            <div>
              <Text strong>记忆类型分布</Text>
              <div style={{ marginTop: '8px' }}>
                {Object.entries(offlineStats?.memory?.memory_types || {}).map(([type, count]) => (
                  <div key={type} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                    <Text style={{ fontSize: '14px' }}>{type}</Text>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div style={{ 
                        width: '80px', 
                        height: '8px', 
                        backgroundColor: '#f0f0f0', 
                        borderRadius: '4px',
                        overflow: 'hidden'
                      }}>
                        <div 
                          style={{ 
                            height: '100%',
                            backgroundColor: '#1890ff', 
                            width: `${(count / (offlineStats?.memory?.total_memories || 1)) * 100}%` 
                          }}
                        ></div>
                      </div>
                      <Text strong style={{ width: '32px', textAlign: 'right' }}>{count}</Text>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </Card>
        </TabPane>

        <TabPane
          tab={
            <span>
              <BranchesOutlined />
              推理工作流
            </span>
          }
          key="reasoning"
        >
          <Card 
            title={
              <div style={{ display: 'flex', alignItems: 'center' }}>
                <BranchesOutlined style={{ marginRight: '8px' }} />
                推理工作流状态
              </div>
            }
          >
            <Row gutter={16} style={{ marginBottom: '16px' }}>
              <Col span={8}>
                <Text strong>总工作流</Text>
                <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                  {offlineStats?.reasoning?.total_workflows || 0}
                </div>
              </Col>
              <Col span={8}>
                <Text strong>活跃工作流</Text>
                <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#52c41a' }}>
                  {offlineStats?.reasoning?.active_workflows || 0}
                </div>
              </Col>
              <Col span={8}>
                <Text strong>已完成</Text>
                <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#1890ff' }}>
                  {offlineStats?.reasoning?.completed_workflows || 0}
                </div>
              </Col>
            </Row>
            
            <div>
              <Text strong>完成率</Text>
              <Progress 
                percent={offlineStats?.reasoning?.total_workflows ? 
                  (offlineStats.reasoning.completed_workflows / offlineStats.reasoning.total_workflows) * 100 : 0}
                style={{ marginTop: '8px' }}
              />
              <Text type="secondary" style={{ fontSize: '12px' }}>
                {offlineStats?.reasoning?.total_workflows ? 
                  ((offlineStats.reasoning.completed_workflows / offlineStats.reasoning.total_workflows) * 100).toFixed(1) : 0}%
              </Text>
            </div>
          </Card>
        </TabPane>
      </Tabs>
    </div>
  );
};

export default OfflineCapabilityPage;