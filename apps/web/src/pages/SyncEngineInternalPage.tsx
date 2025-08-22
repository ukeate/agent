import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Progress, Badge, Timeline, Tag, Table, Alert, Switch, Button, Select, Space, Statistic } from 'antd';
import { SyncOutlined, ClockCircleOutlined, WarningOutlined, CheckCircleOutlined, LoadingOutlined, PauseCircleOutlined, StopOutlined } from '@ant-design/icons';

interface SyncTask {
  id: string;
  sessionId: string;
  direction: 'upload' | 'download' | 'bidirectional';
  priority: number;
  status: 'pending' | 'in_progress' | 'completed' | 'failed' | 'paused' | 'cancelled';
  progress: number;
  totalOperations: number;
  completedOperations: number;
  failedOperations: number;
  createdAt: string;
  startedAt?: string;
  errorMessage?: string;
  retryCount: number;
  checkpointData?: {
    completedBatches: number;
    completedOperations: number;
    lastCheckpoint: string;
  };
}

interface SyncOperation {
  id: string;
  type: 'PUT' | 'DELETE' | 'PATCH';
  tableName: string;
  objectId: string;
  timestamp: string;
  size: number;
  conflicted: boolean;
}

const SyncEngineInternalPage: React.FC = () => {
  const [activeTasks, setActiveTasks] = useState<SyncTask[]>([]);
  const [queuedTasks, setQueuedTasks] = useState<SyncTask[]>([]);
  const [operations, setOperations] = useState<SyncOperation[]>([]);
  const [syncStats, setSyncStats] = useState({
    totalSynced: 1247,
    totalFailed: 23,
    totalConflicts: 8,
    efficiency: 98.2,
    avgThroughput: 156.7
  });
  const [engineConfig, setEngineConfig] = useState({
    maxConcurrentTasks: 3,
    batchSize: 100,
    checkpointInterval: 50,
    retryDelay: 30
  });
  const [realTimeMode, setRealTimeMode] = useState(true);

  // 模拟数据生成
  useEffect(() => {
    generateMockData();
    
    const interval = setInterval(() => {
      if (realTimeMode) {
        updateTaskProgress();
      }
    }, 2000);
    
    return () => clearInterval(interval);
  }, [realTimeMode]);

  const generateMockData = () => {
    // 生成活跃任务
    const activeTasksData: SyncTask[] = [
      {
        id: 'task-001',
        sessionId: 'session-123',
        direction: 'upload',
        priority: 1,
        status: 'in_progress',
        progress: 0.67,
        totalOperations: 250,
        completedOperations: 168,
        failedOperations: 3,
        createdAt: '2024-01-15T10:30:00Z',
        startedAt: '2024-01-15T10:32:00Z',
        retryCount: 0,
        checkpointData: {
          completedBatches: 3,
          completedOperations: 150,
          lastCheckpoint: '2024-01-15T10:45:00Z'
        }
      },
      {
        id: 'task-002',
        sessionId: 'session-124',
        direction: 'bidirectional',
        priority: 2,
        status: 'in_progress',
        progress: 0.23,
        totalOperations: 450,
        completedOperations: 103,
        failedOperations: 1,
        createdAt: '2024-01-15T10:25:00Z',
        startedAt: '2024-01-15T10:35:00Z',
        retryCount: 1
      }
    ];

    // 生成队列任务
    const queuedTasksData: SyncTask[] = [
      {
        id: 'task-003',
        sessionId: 'session-125',
        direction: 'download',
        priority: 3,
        status: 'pending',
        progress: 0,
        totalOperations: 180,
        completedOperations: 0,
        failedOperations: 0,
        createdAt: '2024-01-15T10:40:00Z',
        retryCount: 0
      },
      {
        id: 'task-004',
        sessionId: 'session-126',
        direction: 'upload',
        priority: 4,
        status: 'pending',
        progress: 0,
        totalOperations: 75,
        completedOperations: 0,
        failedOperations: 0,
        createdAt: '2024-01-15T10:42:00Z',
        retryCount: 0
      }
    ];

    // 生成操作数据
    const operationsData: SyncOperation[] = Array.from({ length: 20 }, (_, i) => ({
      id: `op-${i + 1}`,
      type: ['PUT', 'DELETE', 'PATCH'][Math.floor(Math.random() * 3)] as any,
      tableName: ['users', 'documents', 'settings', 'cache'][Math.floor(Math.random() * 4)],
      objectId: `obj-${i + 1}`,
      timestamp: new Date(Date.now() - Math.random() * 3600000).toISOString(),
      size: Math.floor(Math.random() * 10000) + 1000,
      conflicted: Math.random() < 0.1
    }));

    setActiveTasks(activeTasksData);
    setQueuedTasks(queuedTasksData);
    setOperations(operationsData);
  };

  const updateTaskProgress = () => {
    setActiveTasks(prev => prev.map(task => {
      if (task.status === 'in_progress') {
        const newProgress = Math.min(task.progress + Math.random() * 0.1, 1);
        const newCompleted = Math.floor(newProgress * task.totalOperations);
        return {
          ...task,
          progress: newProgress,
          completedOperations: newCompleted
        };
      }
      return task;
    }));
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'in_progress': return <LoadingOutlined spin style={{ color: '#1890ff' }} />;
      case 'completed': return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'failed': return <WarningOutlined style={{ color: '#ff4d4f' }} />;
      case 'paused': return <PauseCircleOutlined style={{ color: '#faad14' }} />;
      case 'cancelled': return <StopOutlined style={{ color: '#d9d9d9' }} />;
      default: return <ClockCircleOutlined style={{ color: '#d9d9d9' }} />;
    }
  };

  const getPriorityLabel = (priority: number) => {
    const labels = ['', '关键', '高', '普通', '低', '后台'];
    const colors = ['', 'red', 'orange', 'blue', 'green', 'default'];
    return <Tag color={colors[priority]}>{labels[priority]}</Tag>;
  };

  const getDirectionLabel = (direction: string) => {
    const labels = {
      upload: '上传',
      download: '下载',
      bidirectional: '双向'
    };
    const colors = {
      upload: 'green',
      download: 'blue',
      bidirectional: 'purple'
    };
    return <Tag color={colors[direction as keyof typeof colors]}>{labels[direction as keyof typeof labels]}</Tag>;
  };

  const tasksColumns = [
    {
      title: '任务ID',
      dataIndex: 'id',
      key: 'id',
      render: (id: string) => <code>{id}</code>
    },
    {
      title: '方向',
      dataIndex: 'direction',
      key: 'direction',
      render: (direction: string) => getDirectionLabel(direction)
    },
    {
      title: '优先级',
      dataIndex: 'priority',
      key: 'priority',
      render: (priority: number) => getPriorityLabel(priority)
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string, record: SyncTask) => (
        <Space>
          {getStatusIcon(status)}
          <span>{status}</span>
          {record.retryCount > 0 && <Badge count={record.retryCount} size="small" />}
        </Space>
      )
    },
    {
      title: '进度',
      key: 'progress',
      render: (record: SyncTask) => (
        <div>
          <Progress 
            percent={Math.round(record.progress * 100)} 
            size="small" 
            status={record.status === 'failed' ? 'exception' : 'active'}
          />
          <div style={{ fontSize: '12px', color: '#666' }}>
            {record.completedOperations}/{record.totalOperations} 操作
            {record.failedOperations > 0 && (
              <span style={{ color: '#ff4d4f' }}> ({record.failedOperations} 失败)</span>
            )}
          </div>
        </div>
      )
    },
    {
      title: '断点数据',
      key: 'checkpoint',
      render: (record: SyncTask) => {
        if (!record.checkpointData) return '-';
        return (
          <div style={{ fontSize: '12px' }}>
            <div>批次: {record.checkpointData.completedBatches}</div>
            <div>操作: {record.checkpointData.completedOperations}</div>
          </div>
        );
      }
    }
  ];

  const operationsColumns = [
    {
      title: '操作ID',
      dataIndex: 'id',
      key: 'id',
      render: (id: string) => <code>{id}</code>
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => {
        const colors = { PUT: 'green', DELETE: 'red', PATCH: 'orange' };
        return <Tag color={colors[type as keyof typeof colors]}>{type}</Tag>;
      }
    },
    {
      title: '表名',
      dataIndex: 'tableName',
      key: 'tableName'
    },
    {
      title: '对象ID',
      dataIndex: 'objectId',
      key: 'objectId'
    },
    {
      title: '大小',
      dataIndex: 'size',
      key: 'size',
      render: (size: number) => `${(size / 1024).toFixed(1)} KB`
    },
    {
      title: '状态',
      key: 'status',
      render: (record: SyncOperation) => record.conflicted ? 
        <Tag color="red">冲突</Tag> : <Tag color="green">正常</Tag>
    }
  ];

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <h1>🔄 同步引擎内部机制展示</h1>
        <p>深入了解数据同步引擎的内部工作原理，包括任务调度、批处理、断点续传等核心机制。</p>
      </div>

      {/* 控制面板 */}
      <Card title="引擎控制面板" style={{ marginBottom: '24px' }}>
        <Row gutter={16}>
          <Col span={6}>
            <Space direction="vertical">
              <span>实时模式</span>
              <Switch checked={realTimeMode} onChange={setRealTimeMode} />
            </Space>
          </Col>
          <Col span={6}>
            <Space direction="vertical">
              <span>最大并发任务</span>
              <Select value={engineConfig.maxConcurrentTasks} style={{ width: '100%' }}>
                <Select.Option value={1}>1</Select.Option>
                <Select.Option value={2}>2</Select.Option>
                <Select.Option value={3}>3</Select.Option>
                <Select.Option value={4}>4</Select.Option>
              </Select>
            </Space>
          </Col>
          <Col span={6}>
            <Space direction="vertical">
              <span>批处理大小</span>
              <Select value={engineConfig.batchSize} style={{ width: '100%' }}>
                <Select.Option value={50}>50</Select.Option>
                <Select.Option value={100}>100</Select.Option>
                <Select.Option value={200}>200</Select.Option>
              </Select>
            </Space>
          </Col>
          <Col span={6}>
            <Space direction="vertical">
              <span>检查点间隔</span>
              <Select value={engineConfig.checkpointInterval} style={{ width: '100%' }}>
                <Select.Option value={25}>25 操作</Select.Option>
                <Select.Option value={50}>50 操作</Select.Option>
                <Select.Option value={100}>100 操作</Select.Option>
              </Select>
            </Space>
          </Col>
        </Row>
      </Card>

      {/* 引擎统计 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={4}>
          <Card>
            <Statistic 
              title="已同步操作" 
              value={syncStats.totalSynced} 
              prefix={<SyncOutlined />}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card>
            <Statistic 
              title="失败操作" 
              value={syncStats.totalFailed} 
              valueStyle={{ color: '#cf1322' }}
              prefix={<WarningOutlined />}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card>
            <Statistic 
              title="冲突解决" 
              value={syncStats.totalConflicts} 
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card>
            <Statistic 
              title="同步效率" 
              value={syncStats.efficiency} 
              precision={1}
              suffix="%" 
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card>
            <Statistic 
              title="平均吞吐量" 
              value={syncStats.avgThroughput} 
              precision={1}
              suffix="ops/s" 
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card>
            <Statistic 
              title="活跃任务" 
              value={activeTasks.length} 
              suffix={`/ ${engineConfig.maxConcurrentTasks}`}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={16}>
        {/* 活跃任务 */}
        <Col span={12}>
          <Card title="🏃‍♂️ 活跃同步任务" style={{ marginBottom: '24px' }}>
            <Table 
              dataSource={activeTasks} 
              columns={tasksColumns}
              rowKey="id"
              size="small"
              pagination={false}
            />
            
            {activeTasks.length > 0 && (
              <Alert 
                style={{ marginTop: '16px' }}
                message="任务调度算法"
                description="同步引擎按照优先级（数值越小优先级越高）和创建时间进行任务调度。支持断点续传，每50个操作创建一个检查点。"
                variant="default"
                showIcon
              />
            )}
          </Card>
        </Col>

        {/* 队列任务 */}
        <Col span={12}>
          <Card title="⏳ 等待队列任务" style={{ marginBottom: '24px' }}>
            <Table 
              dataSource={queuedTasks} 
              columns={tasksColumns}
              rowKey="id"
              size="small"
              pagination={false}
            />
            
            {queuedTasks.length > 0 && (
              <Alert 
                style={{ marginTop: '16px' }}
                message="优先级队列"
                description="任务按优先级排序：关键(1) > 高(2) > 普通(3) > 低(4) > 后台(5)。同优先级按创建时间排序。"
                variant="warning"
                showIcon
              />
            )}
          </Card>
        </Col>
      </Row>

      {/* 操作批处理机制 */}
      <Card title="📦 操作批处理机制" style={{ marginBottom: '24px' }}>
        <Row gutter={16}>
          <Col span={16}>
            <Table 
              dataSource={operations.slice(0, 10)} 
              columns={operationsColumns}
              rowKey="id"
              size="small"
              pagination={false}
              title={() => '当前批次操作 (批大小: ' + engineConfig.batchSize + ')'}
            />
          </Col>
          <Col span={8}>
            <div style={{ background: '#f5f5f5', padding: '16px', borderRadius: '6px' }}>
              <h4>批处理优化策略</h4>
              <Timeline size="small">
                <Timeline.Item color="blue">
                  操作分组：按表名和操作类型分组
                </Timeline.Item>
                <Timeline.Item color="green">
                  批量执行：减少网络往返次数
                </Timeline.Item>
                <Timeline.Item color="orange">
                  断点续传：定期保存处理进度
                </Timeline.Item>
                <Timeline.Item color="red">
                  失败重试：指数退避重试策略
                </Timeline.Item>
                <Timeline.Item color="purple">
                  冲突检测：向量时钟并发检测
                </Timeline.Item>
              </Timeline>
            </div>
          </Col>
        </Row>
      </Card>

      {/* 同步流程图 */}
      <Card title="🔄 同步流程可视化">
        <Row gutter={16}>
          <Col span={8}>
            <Card size="small" title="上传流程">
              <Timeline size="small">
                <Timeline.Item color="blue">创建同步任务</Timeline.Item>
                <Timeline.Item color="green">获取待同步操作</Timeline.Item>
                <Timeline.Item color="orange">按批大小分组</Timeline.Item>
                <Timeline.Item color="purple">逐批上传操作</Timeline.Item>
                <Timeline.Item color="red">创建检查点</Timeline.Item>
                <Timeline.Item color="gray">标记已同步</Timeline.Item>
              </Timeline>
            </Card>
          </Col>
          <Col span={8}>
            <Card size="small" title="下载流程">
              <Timeline size="small">
                <Timeline.Item color="blue">获取服务器更新</Timeline.Item>
                <Timeline.Item color="green">检测本地冲突</Timeline.Item>
                <Timeline.Item color="orange">解决冲突策略</Timeline.Item>
                <Timeline.Item color="purple">应用到本地</Timeline.Item>
                <Timeline.Item color="red">更新向量时钟</Timeline.Item>
                <Timeline.Item color="gray">完成同步</Timeline.Item>
              </Timeline>
            </Card>
          </Col>
          <Col span={8}>
            <Card size="small" title="双向流程">
              <Timeline size="small">
                <Timeline.Item color="blue">执行上传阶段</Timeline.Item>
                <Timeline.Item color="green">等待上传完成</Timeline.Item>
                <Timeline.Item color="orange">执行下载阶段</Timeline.Item>
                <Timeline.Item color="purple">合并同步结果</Timeline.Item>
                <Timeline.Item color="red">计算总体统计</Timeline.Item>
                <Timeline.Item color="gray">返回合并结果</Timeline.Item>
              </Timeline>
            </Card>
          </Col>
        </Row>

        <Alert 
          style={{ marginTop: '16px' }}
          message="增量同步机制"
          description="引擎支持增量数据同步，只传输变更的数据。使用Delta计算器计算数据差异，大幅减少网络传输量和存储需求。支持断点续传，网络中断后可从最后检查点继续。"
          type="success"
          showIcon
        />
      </Card>
    </div>
  );
};

export default SyncEngineInternalPage;