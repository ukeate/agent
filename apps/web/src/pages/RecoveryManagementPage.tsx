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
  Timeline,
  Modal,
  Form,
  Select,
  Input,
  Alert,
  Descriptions,
  Badge,
  Tabs,
  List
} from 'antd';
import {
  SyncOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  WarningOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  SettingOutlined,
  EyeOutlined,
  ThunderboltOutlined,
  SafetyOutlined,
  ClusterOutlined,
  HistoryOutlined
} from '@ant-design/icons';

const { Option } = Select;
const { TextArea } = Input;
const { TabPane } = Tabs;

interface RecoveryRecord {
  fault_id: string;
  fault_type: string;
  recovery_success: boolean;
  recovery_time: number;
  recovery_actions: Array<{
    strategy: string;
    success: boolean;
    timestamp: string;
    details?: string;
  }>;
  started_at: string;
  completed_at?: string;
}

interface RecoveryStatistics {
  total_recoveries: number;
  success_rate: number;
  avg_recovery_time: number;
  strategy_success_rates: Record<string, number>;
  recent_recoveries: RecoveryRecord[];
}

interface RunningRecovery {
  fault_id: string;
  fault_type: string;
  current_strategy: string;
  started_at: string;
  affected_components: string[];
  progress: number;
  status: 'running' | 'waiting' | 'retry';
}

const RecoveryManagementPage: React.FC = () => {
  const [recoveryStats, setRecoveryStats] = useState<RecoveryStatistics | null>(null);
  const [runningRecoveries, setRunningRecoveries] = useState<RunningRecovery[]>([]);
  const [recoveryHistory, setRecoveryHistory] = useState<RecoveryRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedRecord, setSelectedRecord] = useState<RecoveryRecord | null>(null);
  const [modalVisible, setModalVisible] = useState(false);
  const [manualRecoveryVisible, setManualRecoveryVisible] = useState(false);
  const [form] = Form.useForm();

  const fetchRecoveryStats = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/v1/fault-tolerance/recovery/statistics');
      if (response.ok) {
        const data = await response.json();
        setRecoveryStats(data);
      } else {
        // 如果API调用失败，使用模拟数据
        const mockStats: RecoveryStatistics = {
          total_recoveries: 127,
          success_rate: 0.94,
          avg_recovery_time: 45.2,
          strategy_success_rates: {
            immediate_restart: 0.98,
            graceful_restart: 0.92,
            task_migration: 0.87,
            service_degradation: 0.78,
            manual_intervention: 0.65
          },
          recent_recoveries: []
        };
        setRecoveryStats(mockStats);
      }
    } catch (error) {
      console.error('获取恢复统计失败:', error);
      // 使用模拟数据作为fallback
      const mockStats: RecoveryStatistics = {
        total_recoveries: 127,
        success_rate: 0.94,
        avg_recovery_time: 45.2,
        strategy_success_rates: {
          immediate_restart: 0.98,
          graceful_restart: 0.92,
          task_migration: 0.87,
          service_degradation: 0.78,
          manual_intervention: 0.65
        },
        recent_recoveries: []
      };
      setRecoveryStats(mockStats);
    }
  };

  const fetchRunningRecoveries = async () => {
    try {
      // 模拟运行中的恢复操作
      const mockRunning: RunningRecovery[] = [
        {
          fault_id: 'fault_001',
          fault_type: 'agent_unresponsive',
          current_strategy: 'graceful_restart',
          started_at: new Date().toISOString(),
          affected_components: ['agent-3'],
          progress: 65,
          status: 'running'
        }
      ];
      setRunningRecoveries(mockRunning);
    } catch (error) {
      console.error('获取运行中恢复失败:', error);
    }
  };

  const fetchRecoveryHistory = async () => {
    try {
      // 模拟历史恢复记录
      const mockHistory: RecoveryRecord[] = [
        {
          fault_id: 'fault_002',
          fault_type: 'performance_degradation',
          recovery_success: true,
          recovery_time: 32.5,
          recovery_actions: [
            {
              strategy: 'graceful_restart',
              success: true,
              timestamp: new Date(Date.now() - 300000).toISOString(),
              details: 'Agent restarted successfully'
            }
          ],
          started_at: new Date(Date.now() - 400000).toISOString(),
          completed_at: new Date(Date.now() - 300000).toISOString()
        },
        {
          fault_id: 'fault_003',
          fault_type: 'resource_exhaustion',
          recovery_success: false,
          recovery_time: 120.0,
          recovery_actions: [
            {
              strategy: 'immediate_restart',
              success: false,
              timestamp: new Date(Date.now() - 600000).toISOString(),
              details: 'Restart failed, insufficient resources'
            },
            {
              strategy: 'task_migration',
              success: false,
              timestamp: new Date(Date.now() - 500000).toISOString(),
              details: 'No healthy agents available for migration'
            }
          ],
          started_at: new Date(Date.now() - 700000).toISOString(),
          completed_at: new Date(Date.now() - 500000).toISOString()
        }
      ];
      setRecoveryHistory(mockHistory);
    } catch (error) {
      console.error('获取恢复历史失败:', error);
    }
  };

  const triggerManualRecovery = async (values: any) => {
    try {
      // TODO: 实际调用API
      console.log('Triggering manual recovery:', values);
      setManualRecoveryVisible(false);
      form.resetFields();
      
      Modal.success({
        title: '手动恢复已启动',
        content: `已为故障 ${values.fault_id} 启动手动恢复操作`
      });
      
      // 刷新数据
      setTimeout(() => {
        fetchRunningRecoveries();
        fetchRecoveryHistory();
      }, 1000);
    } catch (error) {
      console.error('启动手动恢复失败:', error);
    }
  };

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await Promise.all([
        fetchRecoveryStats(),
        fetchRunningRecoveries(),
        fetchRecoveryHistory()
      ]);
      setLoading(false);
    };

    loadData();
    const interval = setInterval(loadData, 10000); // 每10秒刷新
    return () => clearInterval(interval);
  }, []);

  const getStrategyColor = (strategy: string) => {
    switch (strategy) {
      case 'immediate_restart': return 'red';
      case 'graceful_restart': return 'orange';
      case 'task_migration': return 'blue';
      case 'service_degradation': return 'purple';
      case 'manual_intervention': return 'gray';
      default: return 'default';
    }
  };

  const getStrategyName = (strategy: string) => {
    const names: Record<string, string> = {
      immediate_restart: '立即重启',
      graceful_restart: '优雅重启',
      task_migration: '任务迁移',
      service_degradation: '服务降级',
      manual_intervention: '手动干预'
    };
    return names[strategy] || strategy;
  };

  const runningRecoveryColumns = [
    {
      title: '故障ID',
      dataIndex: 'fault_id',
      key: 'fault_id',
      render: (id: string) => <code>{id}</code>
    },
    {
      title: '故障类型',
      dataIndex: 'fault_type',
      key: 'fault_type',
      render: (type: string) => <Tag color="blue">{type}</Tag>
    },
    {
      title: '当前策略',
      dataIndex: 'current_strategy',
      key: 'current_strategy',
      render: (strategy: string) => (
        <Tag color={getStrategyColor(strategy)}>
          {getStrategyName(strategy)}
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
            <Tag key={comp}>{comp}</Tag>
          ))}
        </div>
      )
    },
    {
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      render: (progress: number) => (
        <Progress percent={progress} size="small" />
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={status === 'running' ? 'green' : status === 'retry' ? 'orange' : 'blue'}>
          {status === 'running' ? '运行中' : status === 'retry' ? '重试中' : '等待中'}
        </Tag>
      )
    },
    {
      title: '开始时间',
      dataIndex: 'started_at',
      key: 'started_at',
      render: (time: string) => new Date(time).toLocaleString()
    }
  ];

  const historyColumns = [
    {
      title: '故障ID',
      dataIndex: 'fault_id',
      key: 'fault_id',
      render: (id: string) => <code>{id}</code>
    },
    {
      title: '故障类型',
      dataIndex: 'fault_type',
      key: 'fault_type',
      render: (type: string) => <Tag color="blue">{type}</Tag>
    },
    {
      title: '恢复结果',
      dataIndex: 'recovery_success',
      key: 'recovery_success',
      render: (success: boolean) => (
        <Tag color={success ? 'green' : 'red'} icon={success ? <CheckCircleOutlined /> : <CloseCircleOutlined />}>
          {success ? '成功' : '失败'}
        </Tag>
      )
    },
    {
      title: '恢复时间',
      dataIndex: 'recovery_time',
      key: 'recovery_time',
      render: (time: number) => `${time.toFixed(1)}s`
    },
    {
      title: '使用策略',
      dataIndex: 'recovery_actions',
      key: 'strategies',
      render: (actions: any[]) => (
        <div>
          {actions.map((action, index) => (
            <Tag 
              key={index} 
              color={action.success ? 'green' : 'red'} 
              style={{ marginBottom: 2 }}
            >
              {getStrategyName(action.strategy)}
            </Tag>
          ))}
        </div>
      )
    },
    {
      title: '完成时间',
      dataIndex: 'completed_at',
      key: 'completed_at',
      render: (time: string) => time ? new Date(time).toLocaleString() : '-'
    },
    {
      title: '操作',
      key: 'actions',
      render: (_: any, record: RecoveryRecord) => (
        <Button 
          type="link"
          icon={<EyeOutlined />}
          onClick={() => {
            setSelectedRecord(record);
            setModalVisible(true);
          }}
        >
          详情
        </Button>
      )
    }
  ];

  return (
    <div className="recovery-management-page p-6">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold mb-2">恢复管理中心</h1>
          <p className="text-gray-600">监控和管理故障恢复策略，优化系统恢复效果</p>
        </div>
        <Space>
          <Button 
            icon={<ThunderboltOutlined />} 
            onClick={() => setManualRecoveryVisible(true)}
            type="primary"
          >
            手动恢复
          </Button>
          <Button 
            icon={<SettingOutlined />} 
            href="/fault-tolerance/recovery/settings"
          >
            恢复策略配置
          </Button>
          <Button 
            icon={<ReloadOutlined />} 
            onClick={() => {
              fetchRecoveryStats();
              fetchRunningRecoveries();
              fetchRecoveryHistory();
            }}
            loading={loading}
          >
            刷新数据
          </Button>
        </Space>
      </div>

      {/* 恢复统计概览 */}
      <Row gutter={[16, 16]} className="mb-6">
        <Col span={6}>
          <Card>
            <Statistic
              title="总恢复次数"
              value={recoveryStats?.total_recoveries || 0}
              prefix={<SyncOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="恢复成功率"
              value={recoveryStats ? recoveryStats.success_rate * 100 : 0}
              suffix="%"
              precision={1}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ 
                color: recoveryStats && recoveryStats.success_rate > 0.9 ? '#3f8600' : '#cf1322' 
              }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均恢复时间"
              value={recoveryStats?.avg_recovery_time || 0}
              suffix="s"
              precision={1}
              prefix={<HistoryOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="运行中恢复"
              value={runningRecoveries.length}
              prefix={<PlayCircleOutlined />}
              valueStyle={{ 
                color: runningRecoveries.length > 0 ? '#fa8c16' : '#52c41a' 
              }}
            />
          </Card>
        </Col>
      </Row>

      {/* 策略成功率分析 */}
      <Row gutter={[16, 16]} className="mb-6">
        <Col span={24}>
          <Card title="恢复策略效果分析">
            <Row gutter={[16, 16]}>
              {recoveryStats && Object.entries(recoveryStats.strategy_success_rates).map(([strategy, rate]) => (
                <Col span={4.8} key={strategy}>
                  <Card size="small" className="text-center">
                    <div className="mb-2">
                      <Tag color={getStrategyColor(strategy)}>
                        {getStrategyName(strategy)}
                      </Tag>
                    </div>
                    <Progress
                      type="circle"
                      percent={rate * 100}
                      size={80}
                      format={() => `${(rate * 100).toFixed(1)}%`}
                      status={rate > 0.9 ? 'success' : rate > 0.7 ? 'active' : 'exception'}
                    />
                  </Card>
                </Col>
              ))}
            </Row>
          </Card>
        </Col>
      </Row>

      {/* 运行中的恢复操作 */}
      {runningRecoveries.length > 0 && (
        <Card title="运行中的恢复操作" className="mb-6">
          <Alert
            message={`当前有 ${runningRecoveries.length} 个恢复操作正在进行`}
            type="info"
            showIcon
            className="mb-4"
          />
          <Table
            columns={runningRecoveryColumns}
            dataSource={runningRecoveries}
            rowKey="fault_id"
            loading={loading}
            pagination={false}
          />
        </Card>
      )}

      {/* 恢复历史记录 */}
      <Card title="恢复历史记录">
        <Table
          columns={historyColumns}
          dataSource={recoveryHistory}
          rowKey="fault_id"
          loading={loading}
          pagination={{
            pageSize: 20,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total) => `共 ${total} 条恢复记录`
          }}
        />
      </Card>

      {/* 恢复详情弹窗 */}
      <Modal
        title="恢复详情"
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setModalVisible(false)}>
            关闭
          </Button>
        ]}
        width={800}
      >
        {selectedRecord && (
          <div>
            <Descriptions bordered column={2} className="mb-4">
              <Descriptions.Item label="故障ID" span={2}>
                <code>{selectedRecord.fault_id}</code>
              </Descriptions.Item>
              <Descriptions.Item label="故障类型">
                <Tag color="blue">{selectedRecord.fault_type}</Tag>
              </Descriptions.Item>
              <Descriptions.Item label="恢复结果">
                <Tag color={selectedRecord.recovery_success ? 'green' : 'red'}>
                  {selectedRecord.recovery_success ? '成功' : '失败'}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="恢复时间">
                {selectedRecord.recovery_time.toFixed(1)}s
              </Descriptions.Item>
              <Descriptions.Item label="开始时间">
                {new Date(selectedRecord.started_at).toLocaleString()}
              </Descriptions.Item>
              {selectedRecord.completed_at && (
                <Descriptions.Item label="完成时间" span={2}>
                  {new Date(selectedRecord.completed_at).toLocaleString()}
                </Descriptions.Item>
              )}
            </Descriptions>

            <h4>恢复操作序列:</h4>
            <Timeline>
              {selectedRecord.recovery_actions.map((action, index) => (
                <Timeline.Item
                  key={index}
                  color={action.success ? 'green' : 'red'}
                  dot={action.success ? <CheckCircleOutlined /> : <CloseCircleOutlined />}
                >
                  <div>
                    <strong>
                      <Tag color={getStrategyColor(action.strategy)}>
                        {getStrategyName(action.strategy)}
                      </Tag>
                    </strong>
                    <span className={action.success ? 'text-green-600' : 'text-red-600'}>
                      {action.success ? ' 执行成功' : ' 执行失败'}
                    </span>
                    <br />
                    <small className="text-gray-500">
                      {new Date(action.timestamp).toLocaleString()}
                    </small>
                    {action.details && (
                      <div className="mt-2 p-2 bg-gray-100 rounded text-sm">
                        {action.details}
                      </div>
                    )}
                  </div>
                </Timeline.Item>
              ))}
            </Timeline>
          </div>
        )}
      </Modal>

      {/* 手动恢复弹窗 */}
      <Modal
        title="启动手动恢复"
        open={manualRecoveryVisible}
        onCancel={() => setManualRecoveryVisible(false)}
        onOk={() => form.submit()}
        okText="启动恢复"
        cancelText="取消"
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={triggerManualRecovery}
        >
          <Form.Item
            name="fault_id"
            label="故障ID"
            rules={[{ required: true, message: '请输入故障ID' }]}
          >
            <Input placeholder="输入要恢复的故障ID" />
          </Form.Item>
          <Form.Item
            name="strategy"
            label="恢复策略"
            rules={[{ required: true, message: '请选择恢复策略' }]}
          >
            <Select placeholder="选择恢复策略">
              <Option value="immediate_restart">立即重启</Option>
              <Option value="graceful_restart">优雅重启</Option>
              <Option value="task_migration">任务迁移</Option>
              <Option value="service_degradation">服务降级</Option>
              <Option value="manual_intervention">手动干预</Option>
            </Select>
          </Form.Item>
          <Form.Item
            name="reason"
            label="启动原因"
          >
            <TextArea 
              rows={3} 
              placeholder="请说明启动手动恢复的原因"
            />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default RecoveryManagementPage;