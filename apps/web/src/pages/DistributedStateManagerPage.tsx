import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Button,
  Table,
  Tag,
  Alert,
  Space,
  Typography,
  Statistic,
  Form,
  Input,
  Select,
  Modal,
  message,
  Tabs,
  Tree,
  Progress,
  Badge,
  Tooltip,
  Timeline,
  Descriptions,
  Switch
} from 'antd';
import {
  DatabaseOutlined,
  LockOutlined,
  UnlockOutlined,
  SaveOutlined,
  RollbackOutlined,
  SyncOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  DeleteOutlined,
  EditOutlined,
  EyeOutlined,
  CopyOutlined,
  SettingOutlined,
  WarningOutlined
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import type { DataNode } from 'antd/es/tree';

const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;
const { TextArea } = Input;
const { confirm } = Modal;

interface StateEntry {
  key: string;
  value: any;
  version: number;
  timestamp: string;
  locked: boolean;
  lock_holder?: string;
  lock_timeout?: string;
  checkpoints: string[];
  replicated: boolean;
  consistency_level: 'strong' | 'eventual' | 'weak';
}

interface Lock {
  resource: string;
  holder: string;
  acquired_at: string;
  expires_at: string;
  timeout: number;
  renewable: boolean;
  metadata: any;
}

interface Checkpoint {
  name: string;
  created_at: string;
  state_snapshot: Record<string, any>;
  version: number;
  description?: string;
  size: number;
}

interface Operation {
  id: string;
  type: 'get' | 'set' | 'delete' | 'lock' | 'unlock' | 'checkpoint' | 'rollback';
  key?: string;
  value?: any;
  success: boolean;
  timestamp: string;
  duration: number;
  node_id: string;
  error?: string;
}

const DistributedStateManagerPage: React.FC = () => {
  const [stateEntries, setStateEntries] = useState<StateEntry[]>([]);
  const [locks, setLocks] = useState<Lock[]>([]);
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [operations, setOperations] = useState<Operation[]>([]);
  const [loading, setLoading] = useState(false);
  const [stateModalVisible, setStateModalVisible] = useState(false);
  const [lockModalVisible, setLockModalVisible] = useState(false);
  const [checkpointModalVisible, setCheckpointModalVisible] = useState(false);
  const [selectedEntry, setSelectedEntry] = useState<StateEntry | null>(null);
  const [form] = Form.useForm();
  const [lockForm] = Form.useForm();
  const [checkpointForm] = Form.useForm();

  // 模拟状态数据
  const generateMockState = () => {
    const mockEntries: StateEntry[] = [
      {
        key: 'user:1001:profile',
        value: { name: 'Alice', email: 'alice@example.com', age: 28 },
        version: 3,
        timestamp: new Date(Date.now() - 5000).toISOString(),
        locked: false,
        checkpoints: ['checkpoint_1', 'checkpoint_2'],
        replicated: true,
        consistency_level: 'strong'
      },
      {
        key: 'config:database:connection',
        value: { host: 'db.example.com', port: 5432, database: 'myapp' },
        version: 1,
        timestamp: new Date(Date.now() - 30000).toISOString(),
        locked: true,
        lock_holder: 'node_2',
        lock_timeout: new Date(Date.now() + 10000).toISOString(),
        checkpoints: ['checkpoint_1'],
        replicated: true,
        consistency_level: 'strong'
      },
      {
        key: 'counter:global',
        value: 42,
        version: 15,
        timestamp: new Date(Date.now() - 2000).toISOString(),
        locked: false,
        checkpoints: ['checkpoint_2'],
        replicated: true,
        consistency_level: 'eventual'
      },
      {
        key: 'cache:session:abc123',
        value: { user_id: 1001, expires: Date.now() + 3600000 },
        version: 1,
        timestamp: new Date(Date.now() - 1000).toISOString(),
        locked: false,
        checkpoints: [],
        replicated: false,
        consistency_level: 'weak'
      }
    ];

    const mockLocks: Lock[] = [
      {
        resource: 'config:database:connection',
        holder: 'node_2',
        acquired_at: new Date(Date.now() - 5000).toISOString(),
        expires_at: new Date(Date.now() + 10000).toISOString(),
        timeout: 15000,
        renewable: true,
        metadata: { operation: 'migration', priority: 'high' }
      },
      {
        resource: 'global:maintenance',
        holder: 'node_1',
        acquired_at: new Date(Date.now() - 2000).toISOString(),
        expires_at: new Date(Date.now() + 30000).toISOString(),
        timeout: 60000,
        renewable: false,
        metadata: { operation: 'system_update' }
      }
    ];

    const mockCheckpoints: Checkpoint[] = [
      {
        name: 'checkpoint_1',
        created_at: new Date(Date.now() - 60000).toISOString(),
        state_snapshot: {
          'user:1001:profile': { name: 'Alice', email: 'alice@example.com', age: 27 },
          'config:database:connection': { host: 'db.example.com', port: 5432, database: 'myapp' },
          'counter:global': 40
        },
        version: 1,
        description: '系统维护前备份',
        size: 2048
      },
      {
        name: 'checkpoint_2',
        created_at: new Date(Date.now() - 30000).toISOString(),
        state_snapshot: {
          'user:1001:profile': { name: 'Alice', email: 'alice@example.com', age: 28 },
          'counter:global': 42
        },
        version: 2,
        description: '功能更新后备份',
        size: 1536
      }
    ];

    setStateEntries(mockEntries);
    setLocks(mockLocks);
    setCheckpoints(mockCheckpoints);
  };

  // 添加操作记录
  const addOperation = (op: Partial<Operation>) => {
    const operation: Operation = {
      id: Date.now().toString(),
      type: op.type!,
      key: op.key,
      value: op.value,
      success: op.success ?? true,
      timestamp: new Date().toISOString(),
      duration: Math.floor(Math.random() * 100) + 10,
      node_id: 'local_node',
      error: op.error,
      ...op
    };

    setOperations(prev => [operation, ...prev.slice(0, 19)]);
  };

  // 设置状态值
  const setState = async (key: string, value: any, consistencyLevel: 'strong' | 'eventual' | 'weak' = 'strong') => {
    setLoading(true);
    try {
      // 检查是否被锁定
      const existingEntry = stateEntries.find(e => e.key === key);
      if (existingEntry?.locked) {
        throw new Error('资源被锁定，无法修改');
      }

      // 模拟设置操作
      await new Promise(resolve => setTimeout(resolve, Math.random() * 500 + 100));

      const newEntry: StateEntry = {
        key,
        value,
        version: existingEntry ? existingEntry.version + 1 : 1,
        timestamp: new Date().toISOString(),
        locked: false,
        checkpoints: existingEntry?.checkpoints || [],
        replicated: consistencyLevel !== 'weak',
        consistency_level: consistencyLevel
      };

      setStateEntries(prev => {
        const filtered = prev.filter(e => e.key !== key);
        return [newEntry, ...filtered];
      });

      addOperation({
        type: 'set',
        key,
        value,
        success: true
      });

      message.success(`状态 ${key} 设置成功`);
    } catch (error) {
      addOperation({
        type: 'set',
        key,
        value,
        success: false,
        error: String(error)
      });
      message.error(`设置失败: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  // 获取状态值
  const getState = async (key: string) => {
    setLoading(true);
    try {
      await new Promise(resolve => setTimeout(resolve, Math.random() * 200 + 50));
      
      const entry = stateEntries.find(e => e.key === key);
      
      addOperation({
        type: 'get',
        key,
        success: !!entry
      });

      if (entry) {
        message.info(`获取成功: ${JSON.stringify(entry.value)}`);
        return entry.value;
      } else {
        message.warning(`键 ${key} 不存在`);
        return null;
      }
    } catch (error) {
      addOperation({
        type: 'get',
        key,
        success: false,
        error: String(error)
      });
      message.error(`获取失败: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  // 删除状态
  const deleteState = async (key: string) => {
    setLoading(true);
    try {
      const entry = stateEntries.find(e => e.key === key);
      if (entry?.locked) {
        throw new Error('资源被锁定，无法删除');
      }

      await new Promise(resolve => setTimeout(resolve, Math.random() * 300 + 100));

      setStateEntries(prev => prev.filter(e => e.key !== key));

      addOperation({
        type: 'delete',
        key,
        success: true
      });

      message.success(`状态 ${key} 删除成功`);
    } catch (error) {
      addOperation({
        type: 'delete',
        key,
        success: false,
        error: String(error)
      });
      message.error(`删除失败: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  // 获取锁
  const acquireLock = async (resource: string, timeout: number = 30000) => {
    setLoading(true);
    try {
      // 检查是否已被锁定
      const existingLock = locks.find(l => l.resource === resource);
      if (existingLock) {
        throw new Error(`资源已被 ${existingLock.holder} 锁定`);
      }

      await new Promise(resolve => setTimeout(resolve, Math.random() * 200 + 100));

      const newLock: Lock = {
        resource,
        holder: 'local_node',
        acquired_at: new Date().toISOString(),
        expires_at: new Date(Date.now() + timeout).toISOString(),
        timeout,
        renewable: true,
        metadata: { acquired_by: 'web_ui' }
      };

      setLocks(prev => [newLock, ...prev]);

      // 更新状态条目的锁定状态
      setStateEntries(prev =>
        prev.map(entry =>
          entry.key === resource
            ? { ...entry, locked: true, lock_holder: 'local_node', lock_timeout: newLock.expires_at }
            : entry
        )
      );

      addOperation({
        type: 'lock',
        key: resource,
        success: true
      });

      message.success(`锁定 ${resource} 成功`);
    } catch (error) {
      addOperation({
        type: 'lock',
        key: resource,
        success: false,
        error: String(error)
      });
      message.error(`锁定失败: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  // 释放锁
  const releaseLock = async (resource: string) => {
    setLoading(true);
    try {
      const lock = locks.find(l => l.resource === resource);
      if (!lock) {
        throw new Error('锁不存在');
      }
      if (lock.holder !== 'local_node') {
        throw new Error('无权释放此锁');
      }

      await new Promise(resolve => setTimeout(resolve, Math.random() * 200 + 100));

      setLocks(prev => prev.filter(l => l.resource !== resource));

      // 更新状态条目的锁定状态
      setStateEntries(prev =>
        prev.map(entry =>
          entry.key === resource
            ? { ...entry, locked: false, lock_holder: undefined, lock_timeout: undefined }
            : entry
        )
      );

      addOperation({
        type: 'unlock',
        key: resource,
        success: true
      });

      message.success(`释放锁 ${resource} 成功`);
    } catch (error) {
      addOperation({
        type: 'unlock',
        key: resource,
        success: false,
        error: String(error)
      });
      message.error(`释放锁失败: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  // 创建检查点
  const createCheckpoint = async (name: string, description?: string) => {
    setLoading(true);
    try {
      if (checkpoints.find(c => c.name === name)) {
        throw new Error('检查点名称已存在');
      }

      await new Promise(resolve => setTimeout(resolve, Math.random() * 1000 + 500));

      const snapshot: Record<string, any> = {};
      stateEntries.forEach(entry => {
        snapshot[entry.key] = entry.value;
      });

      const checkpoint: Checkpoint = {
        name,
        created_at: new Date().toISOString(),
        state_snapshot: snapshot,
        version: checkpoints.length + 1,
        description,
        size: JSON.stringify(snapshot).length
      };

      setCheckpoints(prev => [checkpoint, ...prev]);

      // 更新状态条目的检查点引用
      setStateEntries(prev =>
        prev.map(entry => ({
          ...entry,
          checkpoints: [...entry.checkpoints, name]
        }))
      );

      addOperation({
        type: 'checkpoint',
        key: name,
        success: true
      });

      message.success(`检查点 ${name} 创建成功`);
    } catch (error) {
      addOperation({
        type: 'checkpoint',
        key: name,
        success: false,
        error: String(error)
      });
      message.error(`创建检查点失败: ${error}`);
    } finally {
      setLoading(false);
    }
  };

  // 回滚到检查点
  const rollbackToCheckpoint = async (name: string) => {
    confirm({
      title: '确认回滚',
      icon: <ExclamationCircleOutlined />,
      content: `确定要回滚到检查点 ${name} 吗？这将覆盖当前所有状态。`,
      onOk: async () => {
        setLoading(true);
        try {
          const checkpoint = checkpoints.find(c => c.name === name);
          if (!checkpoint) {
            throw new Error('检查点不存在');
          }

          await new Promise(resolve => setTimeout(resolve, Math.random() * 1000 + 500));

          // 清除所有锁
          setLocks([]);

          // 恢复状态
          const restoredEntries: StateEntry[] = Object.entries(checkpoint.state_snapshot).map(([key, value]) => ({
            key,
            value,
            version: 1,
            timestamp: new Date().toISOString(),
            locked: false,
            checkpoints: [name],
            replicated: true,
            consistency_level: 'strong' as const
          }));

          setStateEntries(restoredEntries);

          addOperation({
            type: 'rollback',
            key: name,
            success: true
          });

          message.success(`回滚到检查点 ${name} 成功`);
        } catch (error) {
          addOperation({
            type: 'rollback',
            key: name,
            success: false,
            error: String(error)
          });
          message.error(`回滚失败: ${error}`);
        } finally {
          setLoading(false);
        }
      }
    });
  };

  // 初始化数据
  useEffect(() => {
    generateMockState();
  }, []);

  // 定时清理过期锁
  useEffect(() => {
    const interval = setInterval(() => {
      const now = Date.now();
      setLocks(prev => {
        const activeLocks = prev.filter(lock => {
          const expired = new Date(lock.expires_at).getTime() < now;
          if (expired) {
            // 释放过期锁对应的状态条目
            setStateEntries(entries =>
              entries.map(entry =>
                entry.key === lock.resource
                  ? { ...entry, locked: false, lock_holder: undefined, lock_timeout: undefined }
                  : entry
              )
            );
          }
          return !expired;
        });
        return activeLocks;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  // 状态表格列
  const stateColumns: ColumnsType<StateEntry> = [
    {
      title: '键',
      dataIndex: 'key',
      key: 'key',
      render: (key: string, record: StateEntry) => (
        <Space>
          <Text code copyable={{ text: key }}>{key}</Text>
          {record.locked && <LockOutlined style={{ color: '#fa8c16' }} />}
        </Space>
      )
    },
    {
      title: '值',
      dataIndex: 'value',
      key: 'value',
      render: (value: any) => (
        <Text ellipsis style={{ maxWidth: 200 }} title={JSON.stringify(value)}>
          {typeof value === 'object' ? JSON.stringify(value) : String(value)}
        </Text>
      )
    },
    {
      title: '版本',
      dataIndex: 'version',
      key: 'version',
      render: (version: number) => <Tag color="blue">v{version}</Tag>
    },
    {
      title: '一致性',
      dataIndex: 'consistency_level',
      key: 'consistency_level',
      render: (level: string) => {
        const colors = { strong: 'green', eventual: 'orange', weak: 'red' };
        return <Tag color={colors[level as keyof typeof colors]}>{level}</Tag>;
      }
    },
    {
      title: '状态',
      key: 'status',
      render: (_, record: StateEntry) => (
        <Space>
          {record.locked && (
            <Tag color="orange" icon={<LockOutlined />}>
              锁定中
            </Tag>
          )}
          {record.replicated && <Tag color="green">已复制</Tag>}
          {record.checkpoints.length > 0 && (
            <Tag color="blue">{record.checkpoints.length} 检查点</Tag>
          )}
        </Space>
      )
    },
    {
      title: '更新时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (time: string) => new Date(time).toLocaleString()
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record: StateEntry) => (
        <Space>
          <Button
            size="small"
            icon={<EyeOutlined />}
            onClick={() => {
              setSelectedEntry(record);
              Modal.info({
                title: `状态详情: ${record.key}`,
                width: 600,
                content: (
                  <div>
                    <pre style={{ background: '#f5f5f5', padding: 16, marginTop: 16 }}>
                      {JSON.stringify(record, null, 2)}
                    </pre>
                  </div>
                )
              });
            }}
          >
            查看
          </Button>
          <Button
            size="small"
            icon={<EditOutlined />}
            onClick={() => {
              setSelectedEntry(record);
              form.setFieldsValue({
                key: record.key,
                value: JSON.stringify(record.value, null, 2),
                consistency: record.consistency_level
              });
              setStateModalVisible(true);
            }}
            disabled={record.locked}
          >
            编辑
          </Button>
          {record.locked ? (
            <Button
              size="small"
              icon={<UnlockOutlined />}
              onClick={() => releaseLock(record.key)}
              disabled={record.lock_holder !== 'local_node'}
            >
              解锁
            </Button>
          ) : (
            <Button
              size="small"
              icon={<LockOutlined />}
              onClick={() => {
                lockForm.setFieldsValue({ resource: record.key });
                setLockModalVisible(true);
              }}
            >
              锁定
            </Button>
          )}
          <Button
            size="small"
            danger
            icon={<DeleteOutlined />}
            onClick={() => deleteState(record.key)}
            disabled={record.locked}
          >
            删除
          </Button>
        </Space>
      )
    }
  ];

  // 锁表格列
  const lockColumns: ColumnsType<Lock> = [
    {
      title: '资源',
      dataIndex: 'resource',
      key: 'resource',
      render: (resource: string) => <Text code>{resource}</Text>
    },
    {
      title: '持有者',
      dataIndex: 'holder',
      key: 'holder',
      render: (holder: string) => <Tag color="blue">{holder}</Tag>
    },
    {
      title: '获取时间',
      dataIndex: 'acquired_at',
      key: 'acquired_at',
      render: (time: string) => new Date(time).toLocaleString()
    },
    {
      title: '过期时间',
      dataIndex: 'expires_at',
      key: 'expires_at',
      render: (time: string) => {
        const expiry = new Date(time).getTime();
        const now = Date.now();
        const remaining = Math.max(0, expiry - now);
        const seconds = Math.floor(remaining / 1000);
        
        return (
          <Space>
            <Text style={{ color: remaining < 10000 ? '#f5222d' : '#52c41a' }}>
              {seconds}秒后过期
            </Text>
          </Space>
        );
      }
    },
    {
      title: '可续期',
      dataIndex: 'renewable',
      key: 'renewable',
      render: (renewable: boolean) => (
        <Tag color={renewable ? 'green' : 'red'}>
          {renewable ? '是' : '否'}
        </Tag>
      )
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record: Lock) => (
        <Space>
          <Button
            size="small"
            danger
            icon={<UnlockOutlined />}
            onClick={() => releaseLock(record.resource)}
            disabled={record.holder !== 'local_node'}
          >
            释放
          </Button>
        </Space>
      )
    }
  ];

  // 检查点表格列
  const checkpointColumns: ColumnsType<Checkpoint> = [
    {
      title: '名称',
      dataIndex: 'name',
      key: 'name',
      render: (name: string) => <Text strong>{name}</Text>
    },
    {
      title: '版本',
      dataIndex: 'version',
      key: 'version',
      render: (version: number) => <Tag color="blue">v{version}</Tag>
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
      ellipsis: true
    },
    {
      title: '大小',
      dataIndex: 'size',
      key: 'size',
      render: (size: number) => `${(size / 1024).toFixed(2)} KB`
    },
    {
      title: '状态数',
      dataIndex: 'state_snapshot',
      key: 'state_count',
      render: (snapshot: Record<string, any>) => (
        <Tag>{Object.keys(snapshot).length} 个状态</Tag>
      )
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (time: string) => new Date(time).toLocaleString()
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record: Checkpoint) => (
        <Space>
          <Button
            size="small"
            icon={<EyeOutlined />}
            onClick={() => {
              Modal.info({
                title: `检查点详情: ${record.name}`,
                width: 800,
                content: (
                  <div>
                    <Descriptions column={2} style={{ marginBottom: 16 }}>
                      <Descriptions.Item label="版本">{record.version}</Descriptions.Item>
                      <Descriptions.Item label="大小">{(record.size / 1024).toFixed(2)} KB</Descriptions.Item>
                      <Descriptions.Item label="状态数">{Object.keys(record.state_snapshot).length}</Descriptions.Item>
                      <Descriptions.Item label="创建时间">{new Date(record.created_at).toLocaleString()}</Descriptions.Item>
                    </Descriptions>
                    <Title level={5}>状态快照:</Title>
                    <pre style={{ background: '#f5f5f5', padding: 16, maxHeight: 400, overflow: 'auto' }}>
                      {JSON.stringify(record.state_snapshot, null, 2)}
                    </pre>
                  </div>
                )
              });
            }}
          >
            查看
          </Button>
          <Button
            size="small"
            icon={<RollbackOutlined />}
            onClick={() => rollbackToCheckpoint(record.name)}
          >
            回滚
          </Button>
        </Space>
      )
    }
  ];

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <Title level={2}>分布式状态管理器</Title>
      <Paragraph>
        分布式状态同步与管理系统，支持强一致性保证、分布式锁、状态检查点和回滚机制。
      </Paragraph>

      {/* 状态概览 */}
      <Card title="系统状态概览" style={{ marginBottom: 24 }}>
        <Row gutter={16}>
          <Col span={6}>
            <Statistic
              title="状态条目"
              value={stateEntries.length}
              prefix={<DatabaseOutlined />}
              suffix="个"
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="活跃锁"
              value={locks.length}
              prefix={<LockOutlined />}
              suffix="个"
              valueStyle={{ color: locks.length > 0 ? '#faad14' : '#52c41a' }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="检查点"
              value={checkpoints.length}
              prefix={<SaveOutlined />}
              suffix="个"
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="操作总数"
              value={operations.length}
              prefix={<SyncOutlined />}
              suffix="次"
            />
          </Col>
        </Row>
      </Card>

      <Tabs defaultActiveKey="states">
        <TabPane tab="状态管理" key="states">
          <Card
            title="分布式状态"
            extra={
              <Space>
                <Button
                  type="primary"
                  icon={<DatabaseOutlined />}
                  onClick={() => {
                    form.resetFields();
                    setSelectedEntry(null);
                    setStateModalVisible(true);
                  }}
                >
                  添加状态
                </Button>
                <Button
                  icon={<SyncOutlined />}
                  onClick={generateMockState}
                >
                  刷新数据
                </Button>
              </Space>
            }
          >
            <Table
              columns={stateColumns}
              dataSource={stateEntries}
              rowKey="key"
              pagination={{ pageSize: 8 }}
              loading={loading}
              size="small"
            />
          </Card>
        </TabPane>

        <TabPane tab="分布式锁" key="locks">
          <Card
            title="分布式锁管理"
            extra={
              <Button
                type="primary"
                icon={<LockOutlined />}
                onClick={() => {
                  lockForm.resetFields();
                  setLockModalVisible(true);
                }}
              >
                获取锁
              </Button>
            }
          >
            {locks.length === 0 ? (
              <Alert
                message="当前没有活跃的锁"
                description="所有资源都处于未锁定状态"
                type="success"
                showIcon
              />
            ) : (
              <Table
                columns={lockColumns}
                dataSource={locks}
                rowKey="resource"
                pagination={false}
                size="small"
              />
            )}
          </Card>
        </TabPane>

        <TabPane tab="状态检查点" key="checkpoints">
          <Card
            title="检查点管理"
            extra={
              <Button
                type="primary"
                icon={<SaveOutlined />}
                onClick={() => {
                  checkpointForm.resetFields();
                  setCheckpointModalVisible(true);
                }}
              >
                创建检查点
              </Button>
            }
          >
            <Table
              columns={checkpointColumns}
              dataSource={checkpoints}
              rowKey="name"
              pagination={{ pageSize: 6 }}
              size="small"
            />
          </Card>
        </TabPane>

        <TabPane tab="操作日志" key="operations">
          <Card title="操作历史">
            <Timeline>
              {operations.slice(0, 10).map((op, index) => (
                <Timeline.Item
                  key={op.id}
                  color={op.success ? 'green' : 'red'}
                  dot={op.success ? <CheckCircleOutlined /> : <ExclamationCircleOutlined />}
                >
                  <div>
                    <Space>
                      <Tag color="blue">{op.type.toUpperCase()}</Tag>
                      {op.key && <Text code>{op.key}</Text>}
                      <Text type="secondary">{op.duration}ms</Text>
                      <Text type="secondary">{new Date(op.timestamp).toLocaleTimeString()}</Text>
                    </Space>
                    {op.error && (
                      <div style={{ marginTop: 4 }}>
                        <Text type="danger">错误: {op.error}</Text>
                      </div>
                    )}
                  </div>
                </Timeline.Item>
              ))}
            </Timeline>
          </Card>
        </TabPane>
      </Tabs>

      {/* 状态编辑模态框 */}
      <Modal
        title={selectedEntry ? '编辑状态' : '添加状态'}
        visible={stateModalVisible}
        onCancel={() => {
          setStateModalVisible(false);
          form.resetFields();
          setSelectedEntry(null);
        }}
        onOk={async () => {
          try {
            const values = await form.validateFields();
            const parsedValue = JSON.parse(values.value);
            await setState(values.key, parsedValue, values.consistency);
            setStateModalVisible(false);
            form.resetFields();
            setSelectedEntry(null);
          } catch (error) {
            message.error('数据格式错误，请检查JSON格式');
          }
        }}
        confirmLoading={loading}
      >
        <Form form={form} layout="vertical">
          <Form.Item
            label="键"
            name="key"
            rules={[{ required: true, message: '请输入状态键' }]}
          >
            <Input placeholder="例如: user:1001:profile" disabled={!!selectedEntry} />
          </Form.Item>
          <Form.Item
            label="值 (JSON格式)"
            name="value"
            rules={[
              { required: true, message: '请输入状态值' },
              {
                validator: (_, value) => {
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
            <TextArea
              rows={6}
              placeholder='{"key": "value"}'
            />
          </Form.Item>
          <Form.Item
            label="一致性级别"
            name="consistency"
            initialValue="strong"
          >
            <Select>
              <Option value="strong">强一致性</Option>
              <Option value="eventual">最终一致性</Option>
              <Option value="weak">弱一致性</Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>

      {/* 锁获取模态框 */}
      <Modal
        title="获取分布式锁"
        visible={lockModalVisible}
        onCancel={() => {
          setLockModalVisible(false);
          lockForm.resetFields();
        }}
        onOk={async () => {
          try {
            const values = await lockForm.validateFields();
            await acquireLock(values.resource, values.timeout * 1000);
            setLockModalVisible(false);
            lockForm.resetFields();
          } catch (error) {
            // 错误已在 acquireLock 中处理
          }
        }}
        confirmLoading={loading}
      >
        <Form form={lockForm} layout="vertical">
          <Form.Item
            label="资源名称"
            name="resource"
            rules={[{ required: true, message: '请输入资源名称' }]}
          >
            <Input placeholder="例如: config:database:connection" />
          </Form.Item>
          <Form.Item
            label="超时时间 (秒)"
            name="timeout"
            initialValue={30}
            rules={[{ required: true, type: 'number', min: 1, max: 3600 }]}
          >
            <Input type="number" />
          </Form.Item>
        </Form>
      </Modal>

      {/* 检查点创建模态框 */}
      <Modal
        title="创建状态检查点"
        visible={checkpointModalVisible}
        onCancel={() => {
          setCheckpointModalVisible(false);
          checkpointForm.resetFields();
        }}
        onOk={async () => {
          try {
            const values = await checkpointForm.validateFields();
            await createCheckpoint(values.name, values.description);
            setCheckpointModalVisible(false);
            checkpointForm.resetFields();
          } catch (error) {
            // 错误已在 createCheckpoint 中处理
          }
        }}
        confirmLoading={loading}
      >
        <Form form={checkpointForm} layout="vertical">
          <Form.Item
            label="检查点名称"
            name="name"
            rules={[{ required: true, message: '请输入检查点名称' }]}
          >
            <Input placeholder="例如: checkpoint_backup_2024" />
          </Form.Item>
          <Form.Item
            label="描述"
            name="description"
          >
            <TextArea rows={3} placeholder="检查点描述信息（可选）" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default DistributedStateManagerPage;