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
  Divider
} from 'antd';
import { 
  BranchesOutlined, 
  DatabaseOutlined, 
  ClockCircleOutlined, 
  WarningOutlined, 
  CheckCircleOutlined, 
  ReloadOutlined, 
  UploadOutlined, 
  DownloadOutlined, 
  ArrowUpOutlined,
  ArrowDownOutlined, 
  DashboardOutlined,
  FileTextOutlined, 
  NumberOutlined, 
  ClockCircleOutlined as TimerOutlined, 
  ThunderboltOutlined, 
  RiseOutlined, 
  SettingOutlined,
  LoadingOutlined,
  BarChartOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { TabPane } = Tabs;

interface SyncTask {
  task_id: string;
  session_id: string;
  direction: 'upload' | 'download' | 'bidirectional';
  priority: number;
  status: 'pending' | 'in_progress' | 'completed' | 'failed' | 'paused' | 'cancelled';
  progress: number;
  total_operations: number;
  completed_operations: number;
  failed_operations: number;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
  retry_count: number;
  checkpoint_data: Record<string, any>;
}

interface SyncStatistics {
  active_tasks: number;
  queued_tasks: number;
  total_tasks: number;
  status_distribution: Record<string, number>;
  priority_distribution: Record<string, number>;
  total_synced_operations: number;
  total_failed_operations: number;
  total_conflicts_resolved: number;
  last_sync_time: string | null;
  sync_efficiency: number;
}

interface VectorClockStats {
  total_syncs: number;
  conflicts_detected: number;
  conflict_rate: number;
  active_nodes: number;
  recent_sync_time: string | null;
}

interface DeltaStats {
  total_deltas: number;
  total_original_size: number;
  total_compressed_size: number;
  average_compression_ratio: number;
  compression_algorithms_used: string[];
}

const SyncManagementPage: React.FC = () => {
  const [syncTasks, setSyncTasks] = useState<SyncTask[]>([]);
  const [syncStats, setSyncStats] = useState<SyncStatistics | null>(null);
  const [vectorClockStats, setVectorClockStats] = useState<VectorClockStats | null>(null);
  const [deltaStats, setDeltaStats] = useState<DeltaStats | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchSyncTasks = async () => {
    try {
      // 模拟API调用
      const mockTasks: SyncTask[] = [
        {
          task_id: 'task-001',
          session_id: 'session-user-123',
          direction: 'upload',
          priority: 1,
          status: 'in_progress',
          progress: 0.65,
          total_operations: 150,
          completed_operations: 98,
          failed_operations: 2,
          created_at: new Date(Date.now() - 300000).toISOString(),
          started_at: new Date(Date.now() - 240000).toISOString(),
          retry_count: 0,
          checkpoint_data: { completed_batches: 3, last_checkpoint: new Date().toISOString() }
        },
        {
          task_id: 'task-002',
          session_id: 'session-user-456',
          direction: 'bidirectional',
          priority: 2,
          status: 'pending',
          progress: 0,
          total_operations: 75,
          completed_operations: 0,
          failed_operations: 0,
          created_at: new Date(Date.now() - 120000).toISOString(),
          retry_count: 0,
          checkpoint_data: {}
        },
        {
          task_id: 'task-003',
          session_id: 'session-user-789',
          direction: 'download',
          priority: 3,
          status: 'completed',
          progress: 1.0,
          total_operations: 42,
          completed_operations: 42,
          failed_operations: 0,
          created_at: new Date(Date.now() - 600000).toISOString(),
          started_at: new Date(Date.now() - 580000).toISOString(),
          completed_at: new Date(Date.now() - 480000).toISOString(),
          retry_count: 0,
          checkpoint_data: {}
        }
      ];
      setSyncTasks(mockTasks);
    } catch (error) {
      console.error('获取同步任务失败:', error);
    }
  };

  const fetchSyncStats = async () => {
    try {
      // 模拟API调用
      const mockStats: SyncStatistics = {
        active_tasks: 1,
        queued_tasks: 1,
        total_tasks: 3,
        status_distribution: {
          'pending': 1,
          'in_progress': 1,
          'completed': 1,
          'failed': 0
        },
        priority_distribution: {
          '1': 1,
          '2': 1,
          '3': 1
        },
        total_synced_operations: 1250,
        total_failed_operations: 15,
        total_conflicts_resolved: 8,
        last_sync_time: new Date(Date.now() - 300000).toISOString(),
        sync_efficiency: 0.988
      };
      setSyncStats(mockStats);
    } catch (error) {
      console.error('获取同步统计失败:', error);
    }
  };

  const fetchVectorClockStats = async () => {
    try {
      const mockVectorStats: VectorClockStats = {
        total_syncs: 342,
        conflicts_detected: 28,
        conflict_rate: 0.082,
        active_nodes: 5,
        recent_sync_time: new Date(Date.now() - 180000).toISOString()
      };
      setVectorClockStats(mockVectorStats);
    } catch (error) {
      console.error('获取向量时钟统计失败:', error);
    }
  };

  const fetchDeltaStats = async () => {
    try {
      const mockDeltaStats: DeltaStats = {
        total_deltas: 156,
        total_original_size: 2048576, // 2MB
        total_compressed_size: 819200, // 800KB
        average_compression_ratio: 0.6,
        compression_algorithms_used: ['gzip', 'json_diff']
      };
      setDeltaStats(mockDeltaStats);
    } catch (error) {
      console.error('获取增量统计失败:', error);
    }
  };

  const createSyncTask = async (direction: 'upload' | 'download' | 'bidirectional') => {
    try {
      // 模拟创建同步任务
      console.log(`创建${direction}同步任务`);
      await fetchSyncTasks();
    } catch (error) {
      console.error('创建同步任务失败:', error);
    }
  };

  const pauseTask = async (taskId: string) => {
    try {
      console.log(`暂停任务 ${taskId}`);
      await fetchSyncTasks();
    } catch (error) {
      console.error('暂停任务失败:', error);
    }
  };

  const resumeTask = async (taskId: string) => {
    try {
      console.log(`恢复任务 ${taskId}`);
      await fetchSyncTasks();
    } catch (error) {
      console.error('恢复任务失败:', error);
    }
  };

  const cancelTask = async (taskId: string) => {
    try {
      console.log(`取消任务 ${taskId}`);
      await fetchSyncTasks();
    } catch (error) {
      console.error('取消任务失败:', error);
    }
  };

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await Promise.all([
        fetchSyncTasks(),
        fetchSyncStats(),
        fetchVectorClockStats(),
        fetchDeltaStats()
      ]);
      setLoading(false);
    };

    loadData();
    const interval = setInterval(loadData, 3000); // 3秒刷新

    return () => clearInterval(interval);
  }, []);

  const getDirectionIcon = (direction: string) => {
    switch (direction) {
      case 'upload': return <UploadOutlined />;
      case 'download': return <DownloadOutlined />;
      case 'bidirectional': return <ArrowUpOutlined />;
      default: return <ThunderboltOutlined />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success';
      case 'in_progress': return 'processing';
      case 'pending': return 'warning';
      case 'failed': return 'error';
      case 'paused': return 'default';
      case 'cancelled': return 'error';
      default: return 'default';
    }
  };

  const getPriorityColor = (priority: number) => {
    if (priority === 1) return 'red';
    if (priority === 2) return 'orange';
    if (priority === 3) return 'yellow';
    if (priority === 4) return 'blue';
    return 'default';
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '50px' }}>
        <LoadingOutlined style={{ fontSize: 24 }} />
        <div style={{ marginTop: 16 }}>加载同步数据中...</div>
      </div>
    );
  }

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Title level={2}>
          <BranchesOutlined style={{ marginRight: '12px' }} />
          数据同步管理
        </Title>
        <Space>
          <Button onClick={() => createSyncTask('upload')} type="primary" icon={<UploadOutlined />}>
            上传同步
          </Button>
          <Button onClick={() => createSyncTask('download')} icon={<DownloadOutlined />}>
            下载同步
          </Button>
          <Button onClick={() => createSyncTask('bidirectional')} icon={<ArrowUpOutlined />}>
            双向同步
          </Button>
        </Space>
      </div>

      {/* 同步统计概览 */}
      <Row gutter={24} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="活跃任务"
              value={syncStats?.active_tasks || 0}
              prefix={<ThunderboltOutlined />}
              suffix={
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  队列中: {syncStats?.queued_tasks || 0}
                </Text>
              }
            />
          </Card>
        </Col>

        <Col span={6}>
          <Card>
            <Statistic
              title="同步效率"
              value={((syncStats?.sync_efficiency || 0) * 100).toFixed(1)}
              prefix={<RiseOutlined />}
              suffix="%"
            />
            <Text type="secondary" style={{ fontSize: '12px' }}>
              成功: {syncStats?.total_synced_operations || 0} / 失败: {syncStats?.total_failed_operations || 0}
            </Text>
          </Card>
        </Col>

        <Col span={6}>
          <Card>
            <Statistic
              title="冲突解决"
              value={syncStats?.total_conflicts_resolved || 0}
              prefix={<WarningOutlined />}
            />
            <Text type="secondary" style={{ fontSize: '12px' }}>
              冲突率: {((vectorClockStats?.conflict_rate || 0) * 100).toFixed(1)}%
            </Text>
          </Card>
        </Col>

        <Col span={6}>
          <Card>
            <Statistic
              title="数据压缩"
              value={((deltaStats?.average_compression_ratio || 0) * 100).toFixed(1)}
              prefix={<DatabaseOutlined />}
              suffix="%"
            />
            <Text type="secondary" style={{ fontSize: '12px' }}>
              节省: {formatBytes((deltaStats?.total_original_size || 0) - (deltaStats?.total_compressed_size || 0))}
            </Text>
          </Card>
        </Col>
      </Row>

      {/* 详细管理面板 */}
      <Card>
        <Tabs defaultActiveKey="tasks" size="large">

          <TabPane
            tab={
              <span>
                <BranchesOutlined />
                同步任务
              </span>
            }
            key="tasks"
          >
            <Card title="同步任务列表">
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                {syncTasks.map((task) => (
                  <Card key={task.task_id} style={{ padding: '16px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                        {getDirectionIcon(task.direction)}
                        <div>
                          <Text strong>{task.task_id}</Text>
                          <br />
                          <Text type="secondary">{task.session_id}</Text>
                        </div>
                      </div>
                      <Space>
                        <Tag color={getStatusColor(task.status)}>
                          {task.status}
                        </Tag>
                        <Tag color={getPriorityColor(task.priority)}>
                          P{task.priority}
                        </Tag>
                      </Space>
                    </div>

                    <Row gutter={16}>
                      <Col span={8}>
                        <Text strong>进度</Text>
                        <Progress percent={task.progress * 100} style={{ marginTop: '4px' }} />
                        <Text type="secondary" style={{ fontSize: '12px' }}>
                          {task.completed_operations}/{task.total_operations} 操作
                        </Text>
                      </Col>
                      <Col span={8}>
                        <Text strong>创建时间</Text>
                        <br />
                        <Text>{new Date(task.created_at).toLocaleString()}</Text>
                      </Col>
                      <Col span={8}>
                        <Text strong>重试次数</Text>
                        <br />
                        <Text>{task.retry_count}</Text>
                      </Col>
                    </Row>

                    {task.failed_operations > 0 && (
                      <Alert
                        message={`失败操作: ${task.failed_operations}`}
                        description={task.error_message}
                        variant="destructive"
                        showIcon
                        style={{ marginTop: '12px' }}
                      />
                    )}

                    {Object.keys(task.checkpoint_data).length > 0 && (
                      <Alert
                        message="检查点数据"
                        description={
                          <pre style={{ fontSize: '12px', margin: 0 }}>
                            {JSON.stringify(task.checkpoint_data, null, 2)}
                          </pre>
                        }
                        variant="default"
                        showIcon
                        style={{ marginTop: '12px' }}
                      />
                    )}

                    <Space style={{ marginTop: '12px' }}>
                      {task.status === 'in_progress' && (
                        <Button size="small" onClick={() => pauseTask(task.task_id)}>
                          暂停
                        </Button>
                      )}
                      {task.status === 'paused' && (
                        <Button size="small" onClick={() => resumeTask(task.task_id)}>
                          恢复
                        </Button>
                      )}
                      {['pending', 'in_progress', 'paused'].includes(task.status) && (
                        <Button size="small" danger onClick={() => cancelTask(task.task_id)}>
                          取消
                        </Button>
                      )}
                    </Space>
                  </Card>
                ))}
              </div>
            </Card>
          </TabPane>

          <TabPane
            tab={
              <span>
                <ClockCircleOutlined />
                向量时钟
              </span>
            }
            key="vector-clock"
          >
            <Card title="向量时钟统计">
              <Row gutter={16} style={{ marginBottom: '16px' }}>
                <Col span={6}>
                  <Text strong>总同步次数</Text>
                  <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                    {vectorClockStats?.total_syncs || 0}
                  </div>
                </Col>
                <Col span={6}>
                  <Text strong>检测到冲突</Text>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#fa8c16' }}>
                    {vectorClockStats?.conflicts_detected || 0}
                  </div>
                </Col>
                <Col span={6}>
                  <Text strong>冲突率</Text>
                  <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                    {((vectorClockStats?.conflict_rate || 0) * 100).toFixed(2)}%
                  </div>
                </Col>
                <Col span={6}>
                  <Text strong>活跃节点</Text>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#52c41a' }}>
                    {vectorClockStats?.active_nodes || 0}
                  </div>
                </Col>
              </Row>

              <div style={{ marginBottom: '16px' }}>
                <Text strong>冲突率趋势</Text>
                <Progress 
                  percent={(vectorClockStats?.conflict_rate || 0) * 100} 
                  style={{ marginTop: '8px' }} 
                />
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  最近同步: {vectorClockStats?.recent_sync_time ? 
                    new Date(vectorClockStats.recent_sync_time).toLocaleString() : '无'}
                </Text>
              </div>

              <Alert
                message="向量时钟工作原理"
                description="向量时钟用于检测分布式系统中事件的因果关系。每个节点维护一个时钟向量，记录来自所有节点的逻辑时间戳。当检测到并发事件时，会标记为潜在冲突。"
                variant="default"
                showIcon
              />
          </Card>
          </TabPane>

          <TabPane
            tab={
              <span>
                <NumberOutlined />
                增量计算
              </span>
            }
            key="delta"
          >
            <Card 
              title={
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <NumberOutlined style={{ marginRight: '8px' }} />
                  增量计算统计
                </div>
              }
            >
              <Row gutter={16} style={{ marginBottom: '16px' }}>
                <Col span={6}>
                  <Text strong>处理增量</Text>
                  <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                    {deltaStats?.total_deltas || 0}
                  </div>
                </Col>
                <Col span={6}>
                  <Text strong>原始大小</Text>
                  <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                    {formatBytes(deltaStats?.total_original_size || 0)}
                  </div>
                </Col>
                <Col span={6}>
                  <Text strong>压缩后大小</Text>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#52c41a' }}>
                    {formatBytes(deltaStats?.total_compressed_size || 0)}
                  </div>
                </Col>
                <Col span={6}>
                  <Text strong>压缩率</Text>
                  <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                    {((deltaStats?.average_compression_ratio || 0) * 100).toFixed(1)}%
                  </div>
                </Col>
              </Row>

              <div style={{ marginBottom: '16px' }}>
                <Text strong>数据压缩效果</Text>
                <Row gutter={16} style={{ marginTop: '8px' }}>
                  <Col span={12}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                      <span>原始</span>
                      <span>{formatBytes(deltaStats?.total_original_size || 0)}</span>
                    </div>
                    <div style={{ 
                      width: '100%', 
                      height: '8px', 
                      backgroundColor: '#f0f0f0', 
                      borderRadius: '4px',
                      overflow: 'hidden'
                    }}>
                      <div style={{ backgroundColor: '#ff4d4f', height: '100%', width: '100%' }}></div>
                    </div>
                  </Col>
                  <Col span={12}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                      <span>压缩后</span>
                      <span>{formatBytes(deltaStats?.total_compressed_size || 0)}</span>
                    </div>
                    <div style={{ 
                      width: '100%', 
                      height: '8px', 
                      backgroundColor: '#f0f0f0', 
                      borderRadius: '4px',
                      overflow: 'hidden'
                    }}>
                      <div 
                        style={{ 
                          backgroundColor: '#52c41a', 
                          height: '100%',
                          width: `${(deltaStats?.total_compressed_size || 0) / (deltaStats?.total_original_size || 1) * 100}%`
                        }}
                      ></div>
                    </div>
                  </Col>
                </Row>
              </div>

              <div style={{ marginBottom: '16px' }}>
                <Text strong>压缩算法使用</Text>
                <div style={{ marginTop: '8px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                  {deltaStats?.compression_algorithms_used.map((algorithm) => (
                    <Tag key={algorithm}>
                      {algorithm}
                    </Tag>
                  ))}
                </div>
              </div>

              <Alert
                message="增量计算优势"
                description={`通过计算数据差异而非传输完整数据，显著减少网络传输量。结合智能压缩算法，平均可减少 ${((deltaStats?.average_compression_ratio || 0) * 100).toFixed(0)}% 的数据传输。`}
                type="success"
                showIcon
              />
            </Card>
          </TabPane>

          <TabPane
            tab={
              <span>
                <BarChartOutlined />
                统计分析
              </span>
            }
            key="statistics"
          >
            <Row gutter={16} style={{ marginBottom: '16px' }}>
              <Col span={12}>
                <Card 
                  title={
                    <div style={{ display: 'flex', alignItems: 'center' }}>
                      <FileTextOutlined style={{ marginRight: '8px' }} />
                      任务状态分布
                    </div>
                  }
                >
                  <div>
                    {Object.entries(syncStats?.status_distribution || {}).map(([status, count]) => (
                      <div key={status} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <Tag color={getStatusColor(status)}>{status}</Tag>
                        </div>
                        <Text strong>{count}</Text>
                      </div>
                    ))}
                  </div>
                </Card>
              </Col>

              <Col span={12}>
                <Card 
                  title={
                    <div style={{ display: 'flex', alignItems: 'center' }}>
                      <SettingOutlined style={{ marginRight: '8px' }} />
                      优先级分布
                    </div>
                  }
                >
                  <div>
                    {Object.entries(syncStats?.priority_distribution || {}).map(([priority, count]) => (
                      <div key={priority} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <Tag color={getPriorityColor(parseInt(priority))}>优先级 {priority}</Tag>
                        </div>
                        <Text strong>{count}</Text>
                      </div>
                    ))}
                  </div>
                </Card>
              </Col>
            </Row>

            <Card 
              title={
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <TimerOutlined style={{ marginRight: '8px' }} />
                  同步性能指标
                </div>
              }
            >
              <Row gutter={16}>
                <Col span={8}>
                  <Text strong>最后同步时间</Text>
                  <div style={{ marginTop: '4px' }}>
                    <Text>{syncStats?.last_sync_time ? 
                      new Date(syncStats.last_sync_time).toLocaleString() : '从未同步'}</Text>
                  </div>
                </Col>
                <Col span={8}>
                  <Text strong>总操作数</Text>
                  <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
                    {(syncStats?.total_synced_operations || 0) + (syncStats?.total_failed_operations || 0)}
                  </div>
                </Col>
                <Col span={8}>
                  <Text strong>成功率</Text>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#52c41a' }}>
                    {((syncStats?.sync_efficiency || 0) * 100).toFixed(2)}%
                  </div>
                </Col>
              </Row>
            </Card>
          </TabPane>
        </Tabs>
      </Card>
    </div>
  );
};

export default SyncManagementPage;