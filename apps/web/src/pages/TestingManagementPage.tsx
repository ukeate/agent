import React, { useState, useEffect } from 'react';
import { 
import { logger } from '../utils/logger'
  Card, Button, Table, Tabs, Form, Input, Select, Space, message, 
  Row, Col, Statistic, Progress, Tag, Modal, InputNumber, Switch, 
  Descriptions, List, Alert, Timeline, Spin, Badge, Drawer
} from 'antd';
import { 
  PlayCircleOutlined, PauseCircleOutlined, StopOutlined, 
  EyeOutlined, DownloadOutlined, SecurityScanOutlined,
  BugOutlined, ThunderboltOutlined, BarChartOutlined,
  CheckCircleOutlined, CloseCircleOutlined, ClockCircleOutlined,
  WarningOutlined, RocketOutlined, FireOutlined
} from '@ant-design/icons';
import { 
  testingService, TestResult, BenchmarkResult, SecurityScanResult, 
  SystemHealthStatus, TestSuiteRequest, BenchmarkRequest, 
  LoadTestRequest, StressTestRequest, SecurityTestRequest 
} from '../services/testingService';

const { Option } = Select;
const { TabPane } = Tabs;
const { TextArea } = Input;

const TestingManagementPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('integration');
  const [loading, setLoading] = useState(false);
  const [runningTests, setRunningTests] = useState<TestResult[]>([]);
  const [systemHealth, setSystemHealth] = useState<SystemHealthStatus | null>(null);
  const [testHistory, setTestHistory] = useState<TestResult[]>([]);
  const [benchmarkHistory, setBenchmarkHistory] = useState<BenchmarkResult[]>([]);
  const [securityHistory, setSecurityHistory] = useState<SecurityScanResult[]>([]);
  const [modalVisible, setModalVisible] = useState(false);
  const [modalType, setModalType] = useState<string>('');
  const [detailDrawerVisible, setDetailDrawerVisible] = useState(false);
  const [selectedTest, setSelectedTest] = useState<any>(null);
  const [form] = Form.useForm();

  useEffect(() => {
    loadInitialData();
    const interval = setInterval(loadRunningTests, 5000);
    return () => clearInterval(interval);
  }, []);

  const loadInitialData = async () => {
    try {
      setLoading(true);
      await Promise.all([
        loadRunningTests(),
        loadSystemHealth(),
        loadTestHistory(),
        loadBenchmarkHistory(),
        loadSecurityHistory()
      ]);
    } catch (error) {
      logger.error('加载初始数据失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadRunningTests = async () => {
    try {
      const tests = await testingService.getRunningTests();
      setRunningTests(tests || []);
    } catch (error) {
      logger.error('加载运行中测试失败:', error);
    }
  };

  const loadSystemHealth = async () => {
    try {
      const health = await testingService.getSystemHealth();
      setSystemHealth(health);
    } catch (error) {
      logger.error('加载系统健康状态失败:', error);
    }
  };

  const loadTestHistory = async () => {
    try {
      const history = await testingService.getIntegrationTestHistory();
      setTestHistory(history || []);
    } catch (error) {
      logger.error('加载测试历史失败:', error);
    }
  };

  const loadBenchmarkHistory = async () => {
    try {
      const history = await testingService.getBenchmarkHistory();
      setBenchmarkHistory(history || []);
    } catch (error) {
      logger.error('加载基准测试历史失败:', error);
    }
  };

  const loadSecurityHistory = async () => {
    try {
      const history = await testingService.getSecurityScanHistory();
      setSecurityHistory(history || []);
    } catch (error) {
      logger.error('加载安全扫描历史失败:', error);
    }
  };

  const handleRunTest = async (testType: string, values: any) => {
    try {
      setLoading(true);
      let result: any;
      
      switch (testType) {
        case 'integration':
          result = await testingService.runIntegrationTests(values);
          break;
        case 'benchmark':
          result = await testingService.runBenchmarkTest(values);
          break;
        case 'load':
          result = await testingService.runLoadTest(values);
          break;
        case 'stress':
          result = await testingService.runStressTest(values);
          break;
        case 'security':
          result = await testingService.runSecurityTest(values);
          break;
      }
      
      message.success(`${testType} 测试已启动`);
      setModalVisible(false);
      form.resetFields();
      loadRunningTests();
      
    } catch (error) {
      message.error(`启动 ${testType} 测试失败`);
      logger.error('启动测试失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleStopTest = async (testId: string, testType: string) => {
    try {
      if (testType === 'load') {
        await testingService.stopLoadTest(testId);
      } else if (testType === 'stress') {
        await testingService.stopStressTest(testId);
      } else {
        await testingService.cancelTest(testId);
      }
      message.success('测试已停止');
      loadRunningTests();
    } catch (error) {
      message.error('停止测试失败');
      logger.error('停止测试失败:', error);
    }
  };

  const handleViewDetails = (test: any) => {
    setSelectedTest(test);
    setDetailDrawerVisible(true);
  };

  const openTestModal = (testType: string) => {
    setModalType(testType);
    setModalVisible(true);
    form.resetFields();
  };

  const getStatusTag = (status: string) => {
    const statusMap = {
      running: { color: 'processing', text: '运行中' },
      completed: { color: 'success', text: '已完成' },
      failed: { color: 'error', text: '失败' },
      cancelled: { color: 'default', text: '已取消' }
    };
    const config = statusMap[status as keyof typeof statusMap] || { color: 'default', text: status };
    return <Tag color={config.color}>{config.text}</Tag>;
  };

  const getHealthStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'warning': return <WarningOutlined style={{ color: '#faad14' }} />;
      case 'critical': return <CloseCircleOutlined style={{ color: '#f5222d' }} />;
      default: return <ClockCircleOutlined style={{ color: '#d9d9d9' }} />;
    }
  };

  const runningTestColumns = [
    {
      title: '测试ID',
      dataIndex: 'test_id',
      key: 'test_id',
      width: 120,
      render: (text: string) => <span style={{ fontFamily: 'monospace' }}>{text.slice(0, 8)}...</span>
    },
    {
      title: '测试类型',
      dataIndex: 'test_type',
      key: 'test_type',
      render: (text: string) => <Tag color="blue">{text}</Tag>
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: getStatusTag
    },
    {
      title: '开始时间',
      dataIndex: 'start_time',
      key: 'start_time',
      render: (text: string) => new Date(text).toLocaleString()
    },
    {
      title: '运行时长',
      key: 'duration',
      render: (record: TestResult) => {
        if (record.status === 'running') {
          const duration = Math.floor((new Date().getTime() - new Date(record.start_time).getTime()) / 1000);
          return `${Math.floor(duration / 60)}分${duration % 60}秒`;
        }
        return record.duration ? `${record.duration}秒` : '-';
      }
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: TestResult) => (
        <Space>
          <Button 
            size="small" 
            icon={<EyeOutlined />}
            onClick={() => handleViewDetails(record)}
          >
            详情
          </Button>
          {record.status === 'running' && (
            <Button 
              size="small" 
              danger
              icon={<StopOutlined />}
              onClick={() => handleStopTest(record.test_id, record.test_type)}
            >
              停止
            </Button>
          )}
        </Space>
      )
    }
  ];

  const renderTestForm = () => {
    const formItems = {
      integration: (
        <>
          <Form.Item name="suite_name" label="测试套件名称" rules={[{ required: true }]}>
            <Input placeholder="输入测试套件名称" />
          </Form.Item>
          <Form.Item name="test_types" label="测试类型" rules={[{ required: true }]}>
            <Select mode="multiple" placeholder="选择测试类型">
              <Option value="unit">单元测试</Option>
              <Option value="integration">集成测试</Option>
              <Option value="e2e">端到端测试</Option>
              <Option value="api">API测试</Option>
            </Select>
          </Form.Item>
          <Form.Item name="async_execution" label="异步执行" valuePropName="checked">
            <Switch />
          </Form.Item>
        </>
      ),
      benchmark: (
        <>
          <Form.Item name="benchmark_types" label="基准测试类型" rules={[{ required: true }]}>
            <Select mode="multiple" placeholder="选择基准测试类型">
              <Option value="cpu">CPU性能</Option>
              <Option value="memory">内存使用</Option>
              <Option value="io">I/O性能</Option>
              <Option value="network">网络性能</Option>
              <Option value="database">数据库性能</Option>
            </Select>
          </Form.Item>
          <Form.Item name="compare_with_baseline" label="与基线对比" valuePropName="checked">
            <Switch defaultChecked />
          </Form.Item>
        </>
      ),
      load: (
        <>
          <Form.Item name="target_qps" label="目标QPS" rules={[{ required: true }]}>
            <InputNumber min={1} max={10000} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item name="duration_minutes" label="持续时间(分钟)" rules={[{ required: true }]}>
            <InputNumber min={1} max={120} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item name="ramp_up_seconds" label="升压时间(秒)" rules={[{ required: true }]}>
            <InputNumber min={1} max={300} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item name="endpoint_patterns" label="测试端点">
            <Select mode="tags" placeholder="输入要测试的端点模式">
              <Option value="/api/v1/*">API v1</Option>
              <Option value="/health">健康检查</Option>
            </Select>
          </Form.Item>
        </>
      ),
      stress: (
        <>
          <Form.Item name="max_concurrent_users" label="最大并发用户数" rules={[{ required: true }]}>
            <InputNumber min={1} max={10000} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item name="duration_minutes" label="持续时间(分钟)" rules={[{ required: true }]}>
            <InputNumber min={1} max={60} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item name="failure_threshold" label="失败阈值(%)" rules={[{ required: true }]}>
            <InputNumber min={1} max={100} style={{ width: '100%' }} />
          </Form.Item>
        </>
      ),
      security: (
        <>
          <Form.Item name="test_categories" label="测试类别" rules={[{ required: true }]}>
            <Select mode="multiple" placeholder="选择安全测试类别">
              <Option value="owasp">OWASP Top 10</Option>
              <Option value="injection">注入攻击</Option>
              <Option value="xss">跨站脚本</Option>
              <Option value="csrf">CSRF攻击</Option>
              <Option value="authentication">认证测试</Option>
              <Option value="authorization">授权测试</Option>
            </Select>
          </Form.Item>
          <Form.Item name="target_endpoints" label="目标端点">
            <Select mode="tags" placeholder="输入要测试的端点">
              <Option value="/api/v1/auth/*">认证端点</Option>
              <Option value="/api/v1/users/*">用户端点</Option>
            </Select>
          </Form.Item>
          <Form.Item name="severity_levels" label="严重程度" rules={[{ required: true }]}>
            <Select mode="multiple" placeholder="选择要检测的严重程度">
              <Option value="low">低</Option>
              <Option value="medium">中</Option>
              <Option value="high">高</Option>
              <Option value="critical">严重</Option>
            </Select>
          </Form.Item>
        </>
      )
    };

    return formItems[modalType as keyof typeof formItems] || null;
  };

  return (
    <div style={{ padding: '24px' }}>
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="运行中测试"
              value={runningTests.length}
              prefix={<RocketOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="系统健康状态"
              value={systemHealth?.overall_status || '未知'}
              prefix={getHealthStatusIcon(systemHealth?.overall_status || 'unknown')}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="今日完成测试"
              value={testHistory.filter(t => 
                new Date(t.start_time).toDateString() === new Date().toDateString()
              ).length}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="系统运行时间"
              value={systemHealth?.uptime ? `${Math.floor(systemHealth.uptime / 3600)}小时` : '未知'}
              prefix={<ClockCircleOutlined />}
            />
          </Card>
        </Col>
      </Row>

      <Card
        title="测试管理中心"
        extra={
          <Space>
            <Button 
              type="primary" 
              icon={<PlayCircleOutlined />}
              onClick={() => openTestModal('integration')}
            >
              集成测试
            </Button>
            <Button 
              icon={<BarChartOutlined />}
              onClick={() => openTestModal('benchmark')}
            >
              性能基准
            </Button>
            <Button 
              icon={<ThunderboltOutlined />}
              onClick={() => openTestModal('load')}
            >
              负载测试
            </Button>
            <Button 
              icon={<FireOutlined />}
              onClick={() => openTestModal('stress')}
            >
              压力测试
            </Button>
            <Button 
              icon={<SecurityScanOutlined />}
              onClick={() => openTestModal('security')}
            >
              安全测试
            </Button>
          </Space>
        }
      >
        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane tab="运行中测试" key="running">
            {runningTests.length > 0 ? (
              <Table
                columns={runningTestColumns}
                dataSource={runningTests}
                rowKey="test_id"
                loading={loading}
                pagination={{ pageSize: 10 }}
              />
            ) : (
              <div style={{ textAlign: 'center', padding: '40px' }}>
                <RocketOutlined style={{ fontSize: '48px', color: '#d9d9d9' }} />
                <p style={{ marginTop: '16px', color: '#999' }}>暂无运行中的测试</p>
              </div>
            )}
          </TabPane>

          <TabPane tab="系统健康状态" key="health">
            {systemHealth ? (
              <Row gutter={[16, 16]}>
                <Col span={24}>
                  <Alert
                    message={`系统整体状态: ${systemHealth.overall_status}`}
                    type={systemHealth.overall_status === 'healthy' ? 'success' : 
                          systemHealth.overall_status === 'warning' ? 'warning' : 'error'}
                    showIcon
                    style={{ marginBottom: 16 }}
                  />
                </Col>
                <Col span={24}>
                  <Card title="组件健康状态">
                    <List
                      dataSource={systemHealth.components}
                      renderItem={(component) => (
                        <List.Item>
                          <List.Item.Meta
                            avatar={getHealthStatusIcon(component.status)}
                            title={component.name}
                            description={
                              <Space>
                                <span>响应时间: {component.response_time}ms</span>
                                <span>错误率: {(component.error_rate * 100).toFixed(2)}%</span>
                                <span>最后检查: {new Date(component.last_check).toLocaleString()}</span>
                              </Space>
                            }
                          />
                          <Badge 
                            status={component.status === 'healthy' ? 'success' : 
                                   component.status === 'degraded' ? 'warning' : 'error'} 
                            text={component.status}
                          />
                        </List.Item>
                      )}
                    />
                  </Card>
                </Col>
              </Row>
            ) : (
              <Spin size="large" style={{ display: 'block', textAlign: 'center', padding: '40px' }} />
            )}
          </TabPane>

          <TabPane tab="测试历史" key="history">
            <Table
              columns={runningTestColumns.filter(col => col.key !== 'actions')}
              dataSource={testHistory}
              rowKey="test_id"
              pagination={{ pageSize: 20 }}
            />
          </TabPane>
        </Tabs>
      </Card>

      <Modal
        title={`运行 ${modalType} 测试`}
        open={modalVisible}
        onCancel={() => {
          setModalVisible(false);
          form.resetFields();
        }}
        footer={null}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={(values) => handleRunTest(modalType, values)}
        >
          {renderTestForm()}
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={loading}>
                启动测试
              </Button>
              <Button onClick={() => {
                setModalVisible(false);
                form.resetFields();
              }}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      <Drawer
        title="测试详情"
        placement="right"
        width={800}
        open={detailDrawerVisible}
        onClose={() => setDetailDrawerVisible(false)}
      >
        {selectedTest && (
          <div>
            <Descriptions column={2} bordered>
              <Descriptions.Item label="测试ID">
                <span style={{ fontFamily: 'monospace' }}>{selectedTest.test_id}</span>
              </Descriptions.Item>
              <Descriptions.Item label="测试类型">{selectedTest.test_type}</Descriptions.Item>
              <Descriptions.Item label="状态">{getStatusTag(selectedTest.status)}</Descriptions.Item>
              <Descriptions.Item label="开始时间">
                {new Date(selectedTest.start_time).toLocaleString()}
              </Descriptions.Item>
              {selectedTest.end_time && (
                <Descriptions.Item label="结束时间">
                  {new Date(selectedTest.end_time).toLocaleString()}
                </Descriptions.Item>
              )}
              {selectedTest.duration && (
                <Descriptions.Item label="持续时间">{selectedTest.duration}秒</Descriptions.Item>
              )}
            </Descriptions>

            {selectedTest.metrics && (
              <Card title="测试指标" style={{ marginTop: 16 }}>
                <Row gutter={[16, 16]}>
                  <Col span={8}>
                    <Statistic title="总测试数" value={selectedTest.metrics.total_tests} />
                  </Col>
                  <Col span={8}>
                    <Statistic 
                      title="通过率" 
                      value={((selectedTest.metrics.passed_tests / selectedTest.metrics.total_tests) * 100).toFixed(1)}
                      suffix="%"
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic title="执行时间" value={selectedTest.metrics.execution_time} suffix="ms" />
                  </Col>
                </Row>
                
                {selectedTest.metrics.performance_metrics && (
                  <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
                    <Col span={12}>
                      <Statistic 
                        title="平均响应时间" 
                        value={selectedTest.metrics.performance_metrics.avg_response_time} 
                        suffix="ms" 
                      />
                    </Col>
                    <Col span={12}>
                      <Statistic 
                        title="吞吐量" 
                        value={selectedTest.metrics.performance_metrics.throughput} 
                        suffix="req/s" 
                      />
                    </Col>
                  </Row>
                )}
              </Card>
            )}

            {selectedTest.results && (
              <Card title="测试结果" style={{ marginTop: 16 }}>
                <pre style={{ 
                  background: '#f5f5f5', 
                  padding: '12px', 
                  borderRadius: '4px',
                  fontSize: '12px',
                  maxHeight: '300px',
                  overflow: 'auto'
                }}>
                  {JSON.stringify(selectedTest.results, null, 2)}
                </pre>
              </Card>
            )}
          </div>
        )}
      </Drawer>
    </div>
  );
};

export default TestingManagementPage;
