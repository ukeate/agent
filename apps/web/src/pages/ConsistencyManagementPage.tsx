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
  Timeline,
  Badge,
  Popconfirm,
  message,
  Tabs,
  List
} from 'antd';
import {
  SyncOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  WarningOutlined,
  SearchOutlined,
  SettingOutlined,
  ReloadOutlined,
  EyeOutlined,
  PlayCircleOutlined,
  BugOutlined,
  DatabaseOutlined,
  ClusterOutlined,
  SecurityScanOutlined,
  HddOutlined
} from '@ant-design/icons';

const { Option } = Select;
const { TextArea } = Input;

interface ConsistencyCheck {
  check_id: string;
  checked_at: string;
  components: string[];
  data_keys: string[];
  consistent: boolean;
  inconsistencies: Array<{
    type: string;
    data_key: string;
    components: string[];
    description: string;
  }>;
  repair_actions: string[];
  duration: number;
}

interface ConsistencyStatistics {
  total_checks: number;
  consistency_rate: number;
  last_check_time: string;
  avg_check_duration: number;
  components_status: Record<string, {
    consistent: boolean;
    last_check: string;
    inconsistency_count: number;
  }>;
}

interface DataInconsistency {
  inconsistency_id: string;
  data_key: string;
  type: 'value_mismatch' | 'missing_data' | 'timestamp_conflict' | 'checksum_mismatch';
  affected_components: string[];
  detected_at: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  resolved: boolean;
  description: string;
  repair_status?: 'pending' | 'in_progress' | 'completed' | 'failed';
}

interface RepairJob {
  repair_id: string;
  check_id: string;
  data_keys: string[];
  repair_strategy: string;
  status: 'running' | 'completed' | 'failed';
  progress: number;
  started_at: string;
  completed_at?: string;
  error?: string;
}

const ConsistencyManagementPage: React.FC = () => {
  const [consistencyStats, setConsistencyStats] = useState<ConsistencyStatistics | null>(null);
  const [checkHistory, setCheckHistory] = useState<ConsistencyCheck[]>([]);
  const [inconsistencies, setInconsistencies] = useState<DataInconsistency[]>([]);
  const [repairJobs, setRepairJobs] = useState<RepairJob[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedCheck, setSelectedCheck] = useState<ConsistencyCheck | null>(null);
  const [checkDetailsVisible, setCheckDetailsVisible] = useState(false);
  const [manualCheckVisible, setManualCheckVisible] = useState(false);
  const [forceRepairVisible, setForceRepairVisible] = useState(false);
  const [form] = Form.useForm();
  const [forceRepairForm] = Form.useForm();
  const [activeTab, setActiveTab] = useState('overview');

  const fetchConsistencyStats = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/v1/fault-tolerance/consistency/statistics');
      if (response.ok) {
        const data = await response.json();
        setConsistencyStats(data);
      } else {
        // 如果API调用失败，使用模拟数据
        const mockStats: ConsistencyStatistics = {
          total_checks: 156,
          consistency_rate: 0.94,
          last_check_time: new Date().toISOString(),
          avg_check_duration: 2.3,
          components_status: {
            'agent-1': { consistent: true, last_check: new Date().toISOString(), inconsistency_count: 0 },
            'agent-2': { consistent: false, last_check: new Date(Date.now() - 3600000).toISOString(), inconsistency_count: 2 },
            'database-1': { consistent: true, last_check: new Date(Date.now() - 1800000).toISOString(), inconsistency_count: 0 },
            'service-1': { consistent: false, last_check: new Date(Date.now() - 7200000).toISOString(), inconsistency_count: 1 }
          }
        };
        setConsistencyStats(mockStats);
      }
    } catch (error) {
      console.error('获取一致性统计失败:', error);
      // 使用模拟数据作为fallback
      const mockStats: ConsistencyStatistics = {
        total_checks: 156,
        consistency_rate: 0.94,
        last_check_time: new Date().toISOString(),
        avg_check_duration: 2.3,
        components_status: {
          'agent-1': { consistent: true, last_check: new Date().toISOString(), inconsistency_count: 0 },
          'agent-2': { consistent: false, last_check: new Date(Date.now() - 3600000).toISOString(), inconsistency_count: 2 },
          'database-1': { consistent: true, last_check: new Date(Date.now() - 1800000).toISOString(), inconsistency_count: 0 },
          'service-1': { consistent: false, last_check: new Date(Date.now() - 7200000).toISOString(), inconsistency_count: 1 }
        }
      };
      setConsistencyStats(mockStats);
    }
  };

  const fetchCheckHistory = async () => {
    try {
      // 模拟获取检查历史
      const mockHistory: ConsistencyCheck[] = [
        {
          check_id: 'check_001',
          checked_at: new Date().toISOString(),
          components: ['agent-1', 'agent-2', 'database-1'],
          data_keys: ['cluster_state', 'task_assignments'],
          consistent: false,
          inconsistencies: [
            {
              type: 'value_mismatch',
              data_key: 'cluster_state',
              components: ['agent-1', 'agent-2'],
              description: 'Cluster membership view differs between agents'
            }
          ],
          repair_actions: ['synchronize_cluster_view', 'update_node_status'],
          duration: 1.8
        },
        {
          check_id: 'check_002',
          checked_at: new Date(Date.now() - 1800000).toISOString(),
          components: ['agent-1', 'agent-2', 'agent-3'],
          data_keys: ['agent_metadata', 'configuration'],
          consistent: true,
          inconsistencies: [],
          repair_actions: [],
          duration: 2.1
        }
      ];
      setCheckHistory(mockHistory);
    } catch (error) {
      console.error('获取检查历史失败:', error);
    }
  };

  const fetchInconsistencies = async () => {
    try {
      // 模拟获取不一致性列表
      const mockInconsistencies: DataInconsistency[] = [
        {
          inconsistency_id: 'inc_001',
          data_key: 'cluster_state',
          type: 'value_mismatch',
          affected_components: ['agent-1', 'agent-2'],
          detected_at: new Date().toISOString(),
          severity: 'high',
          resolved: false,
          description: 'Agent-1 shows 5 nodes, Agent-2 shows 4 nodes in cluster',
          repair_status: 'pending'
        },
        {
          inconsistency_id: 'inc_002',
          data_key: 'task_assignments',
          type: 'missing_data',
          affected_components: ['service-1'],
          detected_at: new Date(Date.now() - 3600000).toISOString(),
          severity: 'medium',
          resolved: true,
          description: 'Task assignment data missing for service-1',
          repair_status: 'completed'
        }
      ];
      setInconsistencies(mockInconsistencies);
    } catch (error) {
      console.error('获取不一致性列表失败:', error);
    }
  };

  const fetchRepairJobs = async () => {
    try {
      // 模拟获取修复任务
      const mockJobs: RepairJob[] = [
        {
          repair_id: 'repair_001',
          check_id: 'check_001',
          data_keys: ['cluster_state'],
          repair_strategy: 'majority_wins',
          status: 'running',
          progress: 60,
          started_at: new Date().toISOString()
        }
      ];
      setRepairJobs(mockJobs);
    } catch (error) {
      console.error('获取修复任务失败:', error);
    }
  };

  const triggerManualCheck = async (values: any) => {
    try {
      const response = await fetch('/api/v1/fault-tolerance/consistency/check', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data_keys: values.data_keys
        })
      });

      if (response.ok) {
        const result = await response.json();
        message.success(`一致性检查已启动，检查ID: ${result.check_id}`);
        setManualCheckVisible(false);
        form.resetFields();
        
        setTimeout(() => {
          fetchCheckHistory();
          fetchInconsistencies();
        }, 1000);
      }
    } catch (error) {
      console.error('启动一致性检查失败:', error);
      message.error('启动检查失败');
    }
  };

  const triggerForceRepair = async (values: any) => {
    try {
      const response = await fetch('/api/v1/fault-tolerance/consistency/force-repair', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data_key: values.data_key,
          authoritative_component_id: values.authoritative_component
        })
      });

      if (response.ok) {
        message.success('强制修复已启动');
        setForceRepairVisible(false);
        forceRepairForm.resetFields();
        
        setTimeout(() => {
          fetchRepairJobs();
          fetchInconsistencies();
        }, 1000);
      }
    } catch (error) {
      console.error('启动强制修复失败:', error);
      message.error('强制修复失败');
    }
  };

  const repairInconsistency = async (checkId: string) => {
    try {
      const response = await fetch(`/api/v1/fault-tolerance/consistency/${checkId}/repair`, {
        method: 'POST'
      });

      if (response.ok) {
        message.success('修复任务已启动');
        fetchRepairJobs();
        fetchInconsistencies();
      }
    } catch (error) {
      console.error('启动修复失败:', error);
      message.error('启动修复失败');
    }
  };

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await Promise.all([
        fetchConsistencyStats(),
        fetchCheckHistory(),
        fetchInconsistencies(),
        fetchRepairJobs()
      ]);
      setLoading(false);
    };

    loadData();
    const interval = setInterval(loadData, 12000); // 每12秒刷新
    return () => clearInterval(interval);
  }, []);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'low': return 'blue';
      case 'medium': return 'orange';
      case 'high': return 'red';
      case 'critical': return 'red';
      default: return 'default';
    }
  };

  const getInconsistencyTypeColor = (type: string) => {
    switch (type) {
      case 'value_mismatch': return 'red';
      case 'missing_data': return 'orange';
      case 'timestamp_conflict': return 'purple';
      case 'checksum_mismatch': return 'magenta';
      default: return 'default';
    }
  };

  const getInconsistencyTypeName = (type: string) => {
    const names: Record<string, string> = {
      value_mismatch: '值不匹配',
      missing_data: '数据缺失',
      timestamp_conflict: '时间戳冲突',
      checksum_mismatch: '校验和不匹配'
    };
    return names[type] || type;
  };

  const checkColumns = [
    {
      title: '检查ID',
      dataIndex: 'check_id',
      key: 'check_id',
      render: (id: string) => <code>{id}</code>
    },
    {
      title: '检查组件',
      dataIndex: 'components',
      key: 'components',
      render: (components: string[]) => (
        <div>
          {components.map(comp => (
            <Tag key={comp} size="small">{comp}</Tag>
          ))}
        </div>
      )
    },
    {
      title: '数据键',
      dataIndex: 'data_keys',
      key: 'data_keys',
      render: (keys: string[]) => (
        <div>
          {keys.map(key => (
            <Tag key={key} color="blue" size="small">{key}</Tag>
          ))}
        </div>
      )
    },
    {
      title: '一致性',
      dataIndex: 'consistent',
      key: 'consistent',
      render: (consistent: boolean) => (
        <Tag color={consistent ? 'green' : 'red'} icon={consistent ? <CheckCircleOutlined /> : <ExclamationCircleOutlined />}>
          {consistent ? '一致' : '不一致'}
        </Tag>
      )
    },
    {
      title: '检查时间',
      dataIndex: 'checked_at',
      key: 'checked_at',
      render: (time: string) => new Date(time).toLocaleString()
    },
    {
      title: '持续时间',
      dataIndex: 'duration',
      key: 'duration',
      render: (duration: number) => `${duration.toFixed(2)}s`
    },
    {
      title: '操作',
      key: 'actions',
      render: (_: any, record: ConsistencyCheck) => (
        <Space>
          <Button 
            size="small" 
            icon={<EyeOutlined />}
            onClick={() => {
              setSelectedCheck(record);
              setCheckDetailsVisible(true);
            }}
          >
            详情
          </Button>
          {!record.consistent && (
            <Button 
              size="small" 
              type="primary"
              icon={<SyncOutlined />}
              onClick={() => repairInconsistency(record.check_id)}
            >
              修复
            </Button>
          )}
        </Space>
      )
    }
  ];

  const inconsistencyColumns = [
    {
      title: '不一致ID',
      dataIndex: 'inconsistency_id',
      key: 'inconsistency_id',
      render: (id: string) => <code>{id.slice(0, 12)}...</code>
    },
    {
      title: '数据键',
      dataIndex: 'data_key',
      key: 'data_key',
      render: (key: string) => <Tag color="blue">{key}</Tag>
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => (
        <Tag color={getInconsistencyTypeColor(type)}>
          {getInconsistencyTypeName(type)}
        </Tag>
      )
    },
    {
      title: '严重程度',
      dataIndex: 'severity',
      key: 'severity',
      render: (severity: string) => (
        <Tag color={getSeverityColor(severity)}>
          {severity.toUpperCase()}
        </Tag>
      )
    },
    {
      title: '影响组件',
      dataIndex: 'affected_components',
      key: 'affected_components',
      render: (components: string[]) => (
        <div>
          {components.map(comp => (
            <Tag key={comp} size="small">{comp}</Tag>
          ))}
        </div>
      )
    },
    {
      title: '状态',
      dataIndex: 'resolved',
      key: 'resolved',
      render: (resolved: boolean, record: DataInconsistency) => {
        if (resolved) {
          return <Tag color="green" icon={<CheckCircleOutlined />}>已解决</Tag>;
        }
        if (record.repair_status) {
          const colors = {
            pending: 'orange',
            in_progress: 'blue',
            completed: 'green',
            failed: 'red'
          };
          const names = {
            pending: '待修复',
            in_progress: '修复中',
            completed: '已完成',
            failed: '修复失败'
          };
          return <Tag color={colors[record.repair_status]}>{names[record.repair_status]}</Tag>;
        }
        return <Tag color="red">未解决</Tag>;
      }
    },
    {
      title: '检测时间',
      dataIndex: 'detected_at',
      key: 'detected_at',
      render: (time: string) => new Date(time).toLocaleString()
    }
  ];

  const repairJobColumns = [
    {
      title: '修复ID',
      dataIndex: 'repair_id',
      key: 'repair_id',
      render: (id: string) => <code>{id}</code>
    },
    {
      title: '数据键',
      dataIndex: 'data_keys',
      key: 'data_keys',
      render: (keys: string[]) => (
        <div>
          {keys.map(key => (
            <Tag key={key} color="blue" size="small">{key}</Tag>
          ))}
        </div>
      )
    },
    {
      title: '修复策略',
      dataIndex: 'repair_strategy',
      key: 'repair_strategy',
      render: (strategy: string) => <Tag color="purple">{strategy}</Tag>
    },
    {
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      render: (progress: number, record: RepairJob) => (
        <Progress 
          percent={progress} 
          size="small" 
          status={record.status === 'failed' ? 'exception' : record.status === 'completed' ? 'success' : 'active'}
        />
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const colors = { running: 'blue', completed: 'green', failed: 'red' };
        const names = { running: '运行中', completed: '已完成', failed: '失败' };
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
    <div className="consistency-management-page p-6">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold mb-2">数据一致性管理</h1>
          <p className="text-gray-600">监控和管理分布式系统数据一致性，自动检测和修复不一致问题</p>
        </div>
        <Space>
          <Button 
            icon={<SearchOutlined />} 
            onClick={() => setManualCheckVisible(true)}
            type="primary"
          >
            手动检查
          </Button>
          <Button 
            icon={<BugOutlined />} 
            onClick={() => setForceRepairVisible(true)}
            danger
          >
            强制修复
          </Button>
          <Button 
            icon={<SettingOutlined />} 
            href="/fault-tolerance/consistency/settings"
          >
            一致性配置
          </Button>
          <Button 
            icon={<ReloadOutlined />} 
            onClick={() => {
              fetchConsistencyStats();
              fetchCheckHistory();
              fetchInconsistencies();
              fetchRepairJobs();
            }}
            loading={loading}
          >
            刷新数据
          </Button>
        </Space>
      </div>

      {/* 一致性统计概览 */}
      <Row gutter={[16, 16]} className="mb-6">
        <Col span={6}>
          <Card>
            <Statistic
              title="总检查次数"
              value={consistencyStats?.total_checks || 0}
              prefix={<SearchOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="一致性率"
              value={consistencyStats ? consistencyStats.consistency_rate * 100 : 0}
              suffix="%"
              precision={1}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ 
                color: consistencyStats && consistencyStats.consistency_rate > 0.95 ? '#3f8600' : '#cf1322' 
              }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均检查时间"
              value={consistencyStats?.avg_check_duration || 0}
              suffix="s"
              precision={1}
              prefix={<HddOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="活跃不一致"
              value={inconsistencies.filter(i => !i.resolved).length}
              prefix={<ExclamationCircleOutlined />}
              valueStyle={{ 
                color: inconsistencies.filter(i => !i.resolved).length === 0 ? '#3f8600' : '#cf1322' 
              }}
            />
          </Card>
        </Col>
      </Row>

      {/* 组件一致性状态 */}
      <Row gutter={[16, 16]} className="mb-6">
        <Col span={24}>
          <Card title="组件一致性状态">
            <Row gutter={[16, 16]}>
              {consistencyStats && Object.entries(consistencyStats.components_status).map(([componentId, status]) => (
                <Col span={6} key={componentId}>
                  <Card size="small" className="text-center">
                    <Badge 
                      status={status.consistent ? 'success' : 'error'} 
                      text={componentId} 
                    />
                    <div className="mt-2 space-y-1">
                      <div>
                        <Tag color={status.consistent ? 'green' : 'red'}>
                          {status.consistent ? '一致' : '不一致'}
                        </Tag>
                      </div>
                      <div>
                        <span className="text-gray-600">不一致数: </span>
                        <span className="font-semibold">{status.inconsistency_count}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">最后检查: </span>
                        <span className="text-sm">{new Date(status.last_check).toLocaleString()}</span>
                      </div>
                    </div>
                  </Card>
                </Col>
              ))}
            </Row>
          </Card>
        </Col>
      </Row>

      {/* 活跃修复任务 */}
      {repairJobs.length > 0 && (
        <Card title="进行中的修复任务" className="mb-6">
          <Alert
            message={`当前有 ${repairJobs.length} 个修复任务正在进行`}
            type="info"
            showIcon
            className="mb-4"
          />
          <Table
            columns={repairJobColumns}
            dataSource={repairJobs}
            rowKey="repair_id"
            loading={loading}
            pagination={false}
          />
        </Card>
      )}

      {/* 主要内容标签页 */}
      <Tabs 
        activeKey={activeTab} 
        onChange={setActiveTab}
        items={[
          {
            key: "checks",
            label: (
              <span>
                <SearchOutlined />
                检查历史
              </span>
            ),
            children: (
              <Table
                columns={checkColumns}
                dataSource={checkHistory}
                rowKey="check_id"
                loading={loading}
                pagination={{
                  pageSize: 20,
                  showSizeChanger: true,
                  showQuickJumper: true,
                  showTotal: (total) => `共 ${total} 次检查`
                }}
              />
            )
          },
          {
            key: "inconsistencies",
            label: (
              <span>
                <ExclamationCircleOutlined />
                不一致问题
              </span>
            ),
            children: (
              <Table
                columns={inconsistencyColumns}
                dataSource={inconsistencies}
                rowKey="inconsistency_id"
                loading={loading}
                pagination={{
                  pageSize: 20,
                  showSizeChanger: true,
                  showQuickJumper: true,
                  showTotal: (total) => `共 ${total} 个不一致问题`
                }}
              />
            )
          }
        ]}
      />

      {/* 检查详情弹窗 */}
      <Modal
        title="一致性检查详情"
        open={checkDetailsVisible}
        onCancel={() => setCheckDetailsVisible(false)}
        footer={[
          <Button key="close" onClick={() => setCheckDetailsVisible(false)}>
            关闭
          </Button>
        ]}
        width={800}
      >
        {selectedCheck && (
          <div>
            <Descriptions bordered column={2} className="mb-4">
              <Descriptions.Item label="检查ID" span={2}>
                <code>{selectedCheck.check_id}</code>
              </Descriptions.Item>
              <Descriptions.Item label="检查时间">
                {new Date(selectedCheck.checked_at).toLocaleString()}
              </Descriptions.Item>
              <Descriptions.Item label="持续时间">
                {selectedCheck.duration.toFixed(2)}s
              </Descriptions.Item>
              <Descriptions.Item label="检查组件" span={2}>
                {selectedCheck.components.map(comp => (
                  <Tag key={comp}>{comp}</Tag>
                ))}
              </Descriptions.Item>
              <Descriptions.Item label="数据键" span={2}>
                {selectedCheck.data_keys.map(key => (
                  <Tag key={key} color="blue">{key}</Tag>
                ))}
              </Descriptions.Item>
              <Descriptions.Item label="一致性结果" span={2}>
                <Tag color={selectedCheck.consistent ? 'green' : 'red'}>
                  {selectedCheck.consistent ? '一致' : '不一致'}
                </Tag>
              </Descriptions.Item>
            </Descriptions>

            {selectedCheck.inconsistencies.length > 0 && (
              <div className="mb-4">
                <h4>发现的不一致问题:</h4>
                <List
                  dataSource={selectedCheck.inconsistencies}
                  renderItem={(item, index) => (
                    <List.Item key={index}>
                      <List.Item.Meta
                        title={
                          <div>
                            <Tag color={getInconsistencyTypeColor(item.type)}>
                              {getInconsistencyTypeName(item.type)}
                            </Tag>
                            <span className="ml-2">{item.data_key}</span>
                          </div>
                        }
                        description={
                          <div>
                            <div>影响组件: {item.components.join(', ')}</div>
                            <div className="mt-1">{item.description}</div>
                          </div>
                        }
                      />
                    </List.Item>
                  )}
                />
              </div>
            )}

            {selectedCheck.repair_actions.length > 0 && (
              <div>
                <h4>修复操作:</h4>
                <Timeline>
                  {selectedCheck.repair_actions.map((action, index) => (
                    <Timeline.Item key={index}>
                      {action}
                    </Timeline.Item>
                  ))}
                </Timeline>
              </div>
            )}
          </div>
        )}
      </Modal>

      {/* 手动检查弹窗 */}
      <Modal
        title="启动手动一致性检查"
        open={manualCheckVisible}
        onCancel={() => setManualCheckVisible(false)}
        onOk={() => form.submit()}
        okText="开始检查"
        cancelText="取消"
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={triggerManualCheck}
        >
          <Form.Item
            name="data_keys"
            label="检查数据键"
            rules={[{ required: true, message: '请选择要检查的数据键' }]}
          >
            <Select
              mode="multiple"
              placeholder="选择要检查一致性的数据键"
              options={[
                { label: 'cluster_state', value: 'cluster_state' },
                { label: 'task_assignments', value: 'task_assignments' },
                { label: 'agent_metadata', value: 'agent_metadata' },
                { label: 'configuration', value: 'configuration' },
                { label: 'node_status', value: 'node_status' }
              ]}
            />
          </Form.Item>
        </Form>
      </Modal>

      {/* 强制修复弹窗 */}
      <Modal
        title="强制修复数据不一致"
        open={forceRepairVisible}
        onCancel={() => setForceRepairVisible(false)}
        onOk={() => forceRepairForm.submit()}
        okText="强制修复"
        cancelText="取消"
      >
        <Alert
          message="警告"
          description="强制修复会覆盖冲突的数据，请确保选择的权威组件数据是正确的。"
          type="warning"
          showIcon
          className="mb-4"
        />
        <Form
          form={forceRepairForm}
          layout="vertical"
          onFinish={triggerForceRepair}
        >
          <Form.Item
            name="data_key"
            label="数据键"
            rules={[{ required: true, message: '请输入要修复的数据键' }]}
          >
            <Input placeholder="输入要强制修复的数据键" />
          </Form.Item>
          <Form.Item
            name="authoritative_component"
            label="权威组件"
            rules={[{ required: true, message: '请选择权威组件' }]}
            help="选择作为标准数据源的组件"
          >
            <Select placeholder="选择权威组件">
              <Option value="agent-1">agent-1</Option>
              <Option value="agent-2">agent-2</Option>
              <Option value="database-1">database-1</Option>
              <Option value="service-1">service-1</Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default ConsistencyManagementPage;