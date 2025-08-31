import React, { useState, useEffect } from 'react';
import {
  Card,
  Table,
  Tag,
  Button,
  Space,
  Row,
  Col,
  Statistic,
  Progress,
  Modal,
  Form,
  Select,
  Input,
  Alert,
  Descriptions,
  Tabs,
  Upload,
  message,
  Popconfirm,
  Badge,
  Timeline
} from 'antd';
import {
  DatabaseOutlined,
  CloudUploadOutlined,
  CloudDownloadOutlined,
  DeleteOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  PlayCircleOutlined,
  SettingOutlined,
  ReloadOutlined,
  EyeOutlined,
  SafetyOutlined,
  HistoryOutlined,
  FileTextOutlined,
  ClusterOutlined
} from '@ant-design/icons';

const { Option } = Select;
const { TextArea } = Input;

interface BackupRecord {
  backup_id: string;
  component_id: string;
  backup_type: 'full_backup' | 'incremental_backup' | 'snapshot_backup';
  created_at: string;
  size: number;
  checksum: string;
  metadata: Record<string, any>;
  storage_path: string;
  valid: boolean;
  description?: string;
}

interface BackupStatistics {
  total_backups: number;
  total_size: number;
  success_rate: number;
  last_backup_time: string;
  components: Record<string, {
    backup_count: number;
    last_backup: string;
    total_size: number;
  }>;
}

interface BackupJob {
  job_id: string;
  component_ids: string[];
  backup_type: string;
  status: 'running' | 'completed' | 'failed' | 'queued';
  progress: number;
  started_at: string;
  completed_at?: string;
  error?: string;
}

const BackupManagementPage: React.FC = () => {
  const [backupRecords, setBackupRecords] = useState<BackupRecord[]>([]);
  const [backupStats, setBackupStats] = useState<BackupStatistics | null>(null);
  const [runningJobs, setRunningJobs] = useState<BackupJob[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedRecord, setSelectedRecord] = useState<BackupRecord | null>(null);
  const [detailsVisible, setDetailsVisible] = useState(false);
  const [manualBackupVisible, setManualBackupVisible] = useState(false);
  const [restoreVisible, setRestoreVisible] = useState(false);
  const [form] = Form.useForm();
  const [restoreForm] = Form.useForm();
  const [activeTab, setActiveTab] = useState('backups');

  const fetchBackupRecords = async () => {
    try {
      // 模拟获取备份记录
      const mockRecords: BackupRecord[] = [
        {
          backup_id: 'backup_001',
          component_id: 'agent-1',
          backup_type: 'full_backup',
          created_at: new Date().toISOString(),
          size: 1024 * 1024 * 150, // 150MB
          checksum: 'sha256:abc123def456',
          metadata: { version: '1.0', config_hash: 'xyz789' },
          storage_path: '/backups/agent-1/backup_001.tar.gz',
          valid: true,
          description: 'Scheduled full backup'
        },
        {
          backup_id: 'backup_002',
          component_id: 'agent-2',
          backup_type: 'incremental_backup',
          created_at: new Date(Date.now() - 3600000).toISOString(),
          size: 1024 * 1024 * 45, // 45MB
          checksum: 'sha256:def456ghi789',
          metadata: { version: '1.0', config_hash: 'abc123' },
          storage_path: '/backups/agent-2/backup_002.tar.gz',
          valid: true,
          description: 'Incremental backup after update'
        },
        {
          backup_id: 'backup_003',
          component_id: 'database-1',
          backup_type: 'snapshot_backup',
          created_at: new Date(Date.now() - 7200000).toISOString(),
          size: 1024 * 1024 * 500, // 500MB
          checksum: 'sha256:ghi789jkl012',
          metadata: { version: '2.1', rows: 150000 },
          storage_path: '/backups/database-1/backup_003.sql.gz',
          valid: false,
          description: 'Database snapshot before migration'
        }
      ];
      setBackupRecords(mockRecords);
    } catch (error) {
      console.error('获取备份记录失败:', error);
    }
  };

  const fetchBackupStatistics = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/v1/fault-tolerance/backup/statistics');
      if (response.ok) {
        const data = await response.json();
        setBackupStats(data);
      } else {
        // 如果API调用失败，使用模拟数据
        const mockStats: BackupStatistics = {
          total_backups: 45,
          total_size: 1024 * 1024 * 1024 * 2.3, // 2.3GB
          success_rate: 0.96,
          last_backup_time: new Date().toISOString(),
          components: {
            'agent-1': { backup_count: 15, last_backup: new Date().toISOString(), total_size: 1024 * 1024 * 800 },
            'agent-2': { backup_count: 12, last_backup: new Date(Date.now() - 3600000).toISOString(), total_size: 1024 * 1024 * 600 },
            'database-1': { backup_count: 8, last_backup: new Date(Date.now() - 7200000).toISOString(), total_size: 1024 * 1024 * 1200 },
            'service-1': { backup_count: 10, last_backup: new Date(Date.now() - 86400000).toISOString(), total_size: 1024 * 1024 * 400 }
          }
        };
        setBackupStats(mockStats);
      }
    } catch (error) {
      console.error('获取备份统计失败:', error);
      // 使用模拟数据作为fallback
      const mockStats: BackupStatistics = {
        total_backups: 45,
        total_size: 1024 * 1024 * 1024 * 2.3, // 2.3GB
        success_rate: 0.96,
        last_backup_time: new Date().toISOString(),
        components: {
          'agent-1': { backup_count: 15, last_backup: new Date().toISOString(), total_size: 1024 * 1024 * 800 },
          'agent-2': { backup_count: 12, last_backup: new Date(Date.now() - 3600000).toISOString(), total_size: 1024 * 1024 * 600 },
          'database-1': { backup_count: 8, last_backup: new Date(Date.now() - 7200000).toISOString(), total_size: 1024 * 1024 * 1200 },
          'service-1': { backup_count: 10, last_backup: new Date(Date.now() - 86400000).toISOString(), total_size: 1024 * 1024 * 400 }
        }
      };
      setBackupStats(mockStats);
    }
  };

  const fetchRunningJobs = async () => {
    try {
      // 模拟运行中的备份任务
      const mockJobs: BackupJob[] = [
        {
          job_id: 'job_001',
          component_ids: ['agent-3', 'agent-4'],
          backup_type: 'full_backup',
          status: 'running',
          progress: 75,
          started_at: new Date().toISOString()
        }
      ];
      setRunningJobs(mockJobs);
    } catch (error) {
      console.error('获取运行任务失败:', error);
    }
  };

  const triggerManualBackup = async (values: any) => {
    try {
      const response = await fetch('/api/v1/fault-tolerance/backup/manual', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          component_ids: values.component_ids,
          backup_type: values.backup_type
        })
      });

      if (response.ok) {
        const result = await response.json();
        message.success(`备份任务已启动，成功率: ${result.success_count}/${result.total_count}`);
        setManualBackupVisible(false);
        form.resetFields();
        
        // 刷新数据
        setTimeout(() => {
          fetchRunningJobs();
          fetchBackupRecords();
        }, 1000);
      }
    } catch (error) {
      console.error('启动手动备份失败:', error);
      message.error('启动备份失败');
    }
  };

  const restoreBackup = async (values: any) => {
    try {
      const response = await fetch('/api/v1/fault-tolerance/backup/restore', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          backup_id: values.backup_id,
          target_component_id: values.target_component_id || undefined
        })
      });

      if (response.ok) {
        message.success('备份恢复任务已启动');
        setRestoreVisible(false);
        restoreForm.resetFields();
      } else {
        message.error('恢复失败');
      }
    } catch (error) {
      console.error('恢复备份失败:', error);
      message.error('恢复操作失败');
    }
  };

  const validateBackups = async () => {
    try {
      const response = await fetch('/api/v1/fault-tolerance/backup/validate', {
        method: 'POST'
      });

      if (response.ok) {
        const result = await response.json();
        Modal.success({
          title: '备份验证完成',
          content: `验证了 ${result.total_count} 个备份，其中 ${result.valid_count} 个有效，验证率: ${(result.validation_rate * 100).toFixed(1)}%`
        });
        fetchBackupRecords();
      }
    } catch (error) {
      console.error('验证备份失败:', error);
      message.error('备份验证失败');
    }
  };

  const deleteBackup = async (backupId: string) => {
    try {
      // TODO: 调用删除API
      console.log('Deleting backup:', backupId);
      message.success('备份已删除');
      fetchBackupRecords();
    } catch (error) {
      console.error('删除备份失败:', error);
      message.error('删除失败');
    }
  };

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await Promise.all([
        fetchBackupRecords(),
        fetchBackupStatistics(),
        fetchRunningJobs()
      ]);
      setLoading(false);
    };

    loadData();
    const interval = setInterval(loadData, 15000); // 每15秒刷新
    return () => clearInterval(interval);
  }, []);

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  };

  const getBackupTypeColor = (type: string) => {
    switch (type) {
      case 'full_backup': return 'blue';
      case 'incremental_backup': return 'green';
      case 'snapshot_backup': return 'purple';
      default: return 'default';
    }
  };

  const getBackupTypeName = (type: string) => {
    const names: Record<string, string> = {
      full_backup: '完整备份',
      incremental_backup: '增量备份',
      snapshot_backup: '快照备份'
    };
    return names[type] || type;
  };

  const backupColumns = [
    {
      title: '备份ID',
      dataIndex: 'backup_id',
      key: 'backup_id',
      render: (id: string) => <code>{id.slice(0, 12)}...</code>
    },
    {
      title: '组件',
      dataIndex: 'component_id',
      key: 'component_id',
      render: (id: string) => <Badge status="processing" text={id} />
    },
    {
      title: '类型',
      dataIndex: 'backup_type',
      key: 'backup_type',
      render: (type: string) => (
        <Tag color={getBackupTypeColor(type)}>
          {getBackupTypeName(type)}
        </Tag>
      )
    },
    {
      title: '大小',
      dataIndex: 'size',
      key: 'size',
      render: (size: number) => formatSize(size)
    },
    {
      title: '状态',
      dataIndex: 'valid',
      key: 'valid',
      render: (valid: boolean) => (
        <Tag color={valid ? 'green' : 'red'} icon={valid ? <CheckCircleOutlined /> : <ExclamationCircleOutlined />}>
          {valid ? '有效' : '无效'}
        </Tag>
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
      key: 'actions',
      render: (_: any, record: BackupRecord) => (
        <Space>
          <Button 
            size="small" 
            icon={<EyeOutlined />}
            onClick={() => {
              setSelectedRecord(record);
              setDetailsVisible(true);
            }}
          >
            详情
          </Button>
          <Button 
            size="small" 
            icon={<CloudDownloadOutlined />}
            type="primary"
            onClick={() => {
              restoreForm.setFieldsValue({ backup_id: record.backup_id });
              setRestoreVisible(true);
            }}
          >
            恢复
          </Button>
          <Popconfirm
            title="确定要删除这个备份吗？"
            onConfirm={() => deleteBackup(record.backup_id)}
            okText="确定"
            cancelText="取消"
          >
            <Button 
              size="small" 
              danger
              icon={<DeleteOutlined />}
            >
              删除
            </Button>
          </Popconfirm>
        </Space>
      )
    }
  ];

  const jobColumns = [
    {
      title: '任务ID',
      dataIndex: 'job_id',
      key: 'job_id',
      render: (id: string) => <code>{id}</code>
    },
    {
      title: '组件',
      dataIndex: 'component_ids',
      key: 'component_ids',
      render: (ids: string[]) => (
        <div>
          {ids.map(id => (
            <Tag key={id} size="small">{id}</Tag>
          ))}
        </div>
      )
    },
    {
      title: '类型',
      dataIndex: 'backup_type',
      key: 'backup_type',
      render: (type: string) => (
        <Tag color={getBackupTypeColor(type)}>
          {getBackupTypeName(type)}
        </Tag>
      )
    },
    {
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      render: (progress: number, record: BackupJob) => (
        <Progress 
          percent={progress} 
          size="small" 
          status={record.status === 'failed' ? 'exception' : 'active'}
        />
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const colors = {
          running: 'blue',
          completed: 'green',
          failed: 'red',
          queued: 'orange'
        };
        const names = {
          running: '运行中',
          completed: '已完成',
          failed: '失败',
          queued: '排队中'
        };
        return <Tag color={colors[status as keyof typeof colors]}>{names[status as keyof typeof names]}</Tag>;
      }
    },
    {
      title: '开始时间',
      dataIndex: 'started_at',
      key: 'started_at',
      render: (time: string) => new Date(time).toLocaleString()
    }
  ];

  return (
    <div className="backup-management-page p-6">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold mb-2">备份管理中心</h1>
          <p className="text-gray-600">管理和监控系统备份，确保数据安全和可恢复性</p>
        </div>
        <Space>
          <Button 
            icon={<CloudUploadOutlined />} 
            onClick={() => setManualBackupVisible(true)}
            type="primary"
          >
            手动备份
          </Button>
          <Button 
            icon={<SafetyOutlined />} 
            onClick={validateBackups}
          >
            验证备份
          </Button>
          <Button 
            icon={<SettingOutlined />} 
            href="/fault-tolerance/backup/settings"
          >
            备份设置
          </Button>
          <Button 
            icon={<ReloadOutlined />} 
            onClick={() => {
              fetchBackupRecords();
              fetchBackupStatistics();
              fetchRunningJobs();
            }}
            loading={loading}
          >
            刷新数据
          </Button>
        </Space>
      </div>

      {/* 备份统计概览 */}
      <Row gutter={[16, 16]} className="mb-6">
        <Col span={6}>
          <Card>
            <Statistic
              title="总备份数"
              value={backupStats?.total_backups || 0}
              prefix={<DatabaseOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="总存储大小"
              value={formatSize(backupStats?.total_size || 0)}
              prefix={<ClusterOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="备份成功率"
              value={backupStats ? backupStats.success_rate * 100 : 0}
              suffix="%"
              precision={1}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ 
                color: backupStats && backupStats.success_rate > 0.95 ? '#3f8600' : '#cf1322' 
              }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="运行中任务"
              value={runningJobs.length}
              prefix={<PlayCircleOutlined />}
              valueStyle={{ 
                color: runningJobs.length > 0 ? '#fa8c16' : '#52c41a' 
              }}
            />
          </Card>
        </Col>
      </Row>

      {/* 组件备份状态 */}
      <Row gutter={[16, 16]} className="mb-6">
        <Col span={24}>
          <Card title="组件备份状态">
            <Row gutter={[16, 16]}>
              {backupStats && Object.entries(backupStats.components).map(([componentId, info]) => (
                <Col span={6} key={componentId}>
                  <Card size="small" className="text-center">
                    <Badge status="processing" text={componentId} />
                    <div className="mt-2 space-y-1">
                      <div>
                        <span className="text-gray-600">备份数: </span>
                        <span className="font-semibold">{info.backup_count}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">大小: </span>
                        <span className="font-semibold">{formatSize(info.total_size)}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">最后备份: </span>
                        <span className="text-sm">{new Date(info.last_backup).toLocaleDateString()}</span>
                      </div>
                    </div>
                  </Card>
                </Col>
              ))}
            </Row>
          </Card>
        </Col>
      </Row>

      {/* 运行中的备份任务 */}
      {runningJobs.length > 0 && (
        <Card title="运行中的备份任务" className="mb-6">
          <Alert
            message={`当前有 ${runningJobs.length} 个备份任务正在进行`}
            type="info"
            showIcon
            className="mb-4"
          />
          <Table
            columns={jobColumns}
            dataSource={runningJobs}
            rowKey="job_id"
            loading={loading}
            pagination={false}
          />
        </Card>
      )}

      {/* 主要内容标签页 */}
      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <Tabs.TabPane
          tab={
            <span>
              <DatabaseOutlined />
              备份记录
            </span>
          }
          key="backups"
        >
          <Table
            columns={backupColumns}
            dataSource={backupRecords}
            rowKey="backup_id"
            loading={loading}
            pagination={{
              pageSize: 20,
              showSizeChanger: true,
              showQuickJumper: true,
              showTotal: (total) => `共 ${total} 个备份`
            }}
          />
        </Tabs.TabPane>
      </Tabs>

      {/* 备份详情弹窗 */}
      <Modal
        title="备份详情"
        open={detailsVisible}
        onCancel={() => setDetailsVisible(false)}
        footer={[
          <Button key="close" onClick={() => setDetailsVisible(false)}>
            关闭
          </Button>
        ]}
        width={800}
      >
        {selectedRecord && (
          <div>
            <Descriptions bordered column={2}>
              <Descriptions.Item label="备份ID" span={2}>
                <code>{selectedRecord.backup_id}</code>
              </Descriptions.Item>
              <Descriptions.Item label="组件ID">
                <Badge status="processing" text={selectedRecord.component_id} />
              </Descriptions.Item>
              <Descriptions.Item label="备份类型">
                <Tag color={getBackupTypeColor(selectedRecord.backup_type)}>
                  {getBackupTypeName(selectedRecord.backup_type)}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="文件大小">
                {formatSize(selectedRecord.size)}
              </Descriptions.Item>
              <Descriptions.Item label="状态">
                <Tag color={selectedRecord.valid ? 'green' : 'red'}>
                  {selectedRecord.valid ? '有效' : '无效'}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="创建时间" span={2}>
                {new Date(selectedRecord.created_at).toLocaleString()}
              </Descriptions.Item>
              <Descriptions.Item label="校验和" span={2}>
                <code>{selectedRecord.checksum}</code>
              </Descriptions.Item>
              <Descriptions.Item label="存储路径" span={2}>
                <code>{selectedRecord.storage_path}</code>
              </Descriptions.Item>
              {selectedRecord.description && (
                <Descriptions.Item label="描述" span={2}>
                  {selectedRecord.description}
                </Descriptions.Item>
              )}
            </Descriptions>

            {Object.keys(selectedRecord.metadata).length > 0 && (
              <div className="mt-4">
                <h4>元数据信息:</h4>
                <pre className="bg-gray-100 p-3 rounded">
                  {JSON.stringify(selectedRecord.metadata, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}
      </Modal>

      {/* 手动备份弹窗 */}
      <Modal
        title="启动手动备份"
        open={manualBackupVisible}
        onCancel={() => setManualBackupVisible(false)}
        onOk={() => form.submit()}
        okText="开始备份"
        cancelText="取消"
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={triggerManualBackup}
        >
          <Form.Item
            name="component_ids"
            label="选择组件"
            rules={[{ required: true, message: '请选择要备份的组件' }]}
          >
            <Select
              mode="multiple"
              placeholder="选择要备份的组件"
              options={[
                { label: 'agent-1', value: 'agent-1' },
                { label: 'agent-2', value: 'agent-2' },
                { label: 'agent-3', value: 'agent-3' },
                { label: 'database-1', value: 'database-1' },
                { label: 'service-1', value: 'service-1' }
              ]}
            />
          </Form.Item>
          <Form.Item
            name="backup_type"
            label="备份类型"
            rules={[{ required: true, message: '请选择备份类型' }]}
            initialValue="full_backup"
          >
            <Select>
              <Option value="full_backup">完整备份</Option>
              <Option value="incremental_backup">增量备份</Option>
              <Option value="snapshot_backup">快照备份</Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>

      {/* 恢复备份弹窗 */}
      <Modal
        title="恢复备份"
        open={restoreVisible}
        onCancel={() => setRestoreVisible(false)}
        onOk={() => restoreForm.submit()}
        okText="开始恢复"
        cancelText="取消"
      >
        <Form
          form={restoreForm}
          layout="vertical"
          onFinish={restoreBackup}
        >
          <Form.Item
            name="backup_id"
            label="备份ID"
            rules={[{ required: true, message: '请输入备份ID' }]}
          >
            <Input placeholder="输入要恢复的备份ID" />
          </Form.Item>
          <Form.Item
            name="target_component_id"
            label="目标组件"
            help="留空则恢复到原组件"
          >
            <Select
              placeholder="选择目标组件（可选）"
              allowClear
              options={[
                { label: 'agent-1', value: 'agent-1' },
                { label: 'agent-2', value: 'agent-2' },
                { label: 'agent-3', value: 'agent-3' },
                { label: 'database-1', value: 'database-1' },
                { label: 'service-1', value: 'service-1' }
              ]}
            />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default BackupManagementPage;