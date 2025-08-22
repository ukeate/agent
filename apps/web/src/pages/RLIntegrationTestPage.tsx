import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Button, Table, Progress, Tag, Alert, Select, Space, Statistic, Timeline, Modal, Tabs } from 'antd';
import { 
  ExperimentOutlined, 
  PlayCircleOutlined, 
  CheckCircleOutlined, 
  CloseCircleOutlined,
  ClockCircleOutlined,
  BugOutlined,
  ThunderboltOutlined,
  SettingOutlined,
  FileTextOutlined
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';

const { Option } = Select;
const { TabPane } = Tabs;

interface TestSuite {
  id: string;
  name: string;
  description: string;
  status: 'pending' | 'running' | 'passed' | 'failed' | 'skipped';
  duration: number;
  progress: number;
  tests: TestCase[];
  lastRun: string;
}

interface TestCase {
  id: string;
  name: string;
  description: string;
  status: 'pending' | 'running' | 'passed' | 'failed' | 'skipped';
  duration: number;
  error?: string;
  assertions: number;
  passedAssertions: number;
}

interface TestResult {
  suite: string;
  testCase: string;
  timestamp: string;
  status: 'passed' | 'failed';
  duration: number;
  error?: string;
}

const RLIntegrationTestPage: React.FC = () => {
  const [testSuites, setTestSuites] = useState<TestSuite[]>([]);
  const [runningTests, setRunningTests] = useState<Set<string>>(new Set());
  const [testResults, setTestResults] = useState<TestResult[]>([]);
  const [selectedSuite, setSelectedSuite] = useState<TestSuite | null>(null);
  const [showDetailsModal, setShowDetailsModal] = useState(false);
  const [testEnv, setTestEnv] = useState('staging');

  // 初始化测试套件数据
  useEffect(() => {
    const suites: TestSuite[] = [
      {
        id: 'end-to-end',
        name: '端到端集成测试',
        description: '完整的推荐工作流测试，包括用户请求、算法选择、结果返回',
        status: 'passed',
        duration: 45000,
        progress: 100,
        lastRun: '2025-08-22 14:20:15',
        tests: [
          {
            id: 'full-workflow',
            name: '完整推荐工作流',
            description: '测试从用户请求到推荐返回的完整流程',
            status: 'passed',
            duration: 12000,
            assertions: 15,
            passedAssertions: 15
          },
          {
            id: 'multi-user',
            name: '多用户并发测试',
            description: '测试系统处理多个用户并发请求的能力',
            status: 'passed',
            duration: 18000,
            assertions: 25,
            passedAssertions: 25
          },
          {
            id: 'algorithm-comparison',
            name: '算法性能对比',
            description: '对比不同推荐算法的性能表现',
            status: 'passed',
            duration: 15000,
            assertions: 20,
            passedAssertions: 20
          }
        ]
      },
      {
        id: 'performance',
        name: '性能集成测试',
        description: '系统性能基准测试，验证延迟、吞吐量等指标',
        status: 'failed',
        duration: 60000,
        progress: 75,
        lastRun: '2025-08-22 14:15:30',
        tests: [
          {
            id: 'latency-benchmark',
            name: '延迟基准测试',
            description: '测试系统响应时间是否满足SLA要求',
            status: 'passed',
            duration: 20000,
            assertions: 10,
            passedAssertions: 10
          },
          {
            id: 'throughput-test',
            name: '吞吐量测试',
            description: '测试系统在高负载下的处理能力',
            status: 'failed',
            duration: 25000,
            error: 'QPS 达到 8500 req/s 时系统响应时间超过 100ms 阈值',
            assertions: 8,
            passedAssertions: 6
          },
          {
            id: 'cache-performance',
            name: '缓存性能测试',
            description: '测试缓存系统的命中率和性能',
            status: 'passed',
            duration: 15000,
            assertions: 12,
            passedAssertions: 12
          }
        ]
      },
      {
        id: 'algorithm-integration',
        name: '算法集成测试',
        description: '测试各个推荐算法的集成和切换功能',
        status: 'passed',
        duration: 30000,
        progress: 100,
        lastRun: '2025-08-22 14:18:45',
        tests: [
          {
            id: 'ucb-integration',
            name: 'UCB算法集成',
            description: '测试UCB算法的推荐准确性和性能',
            status: 'passed',
            duration: 8000,
            assertions: 8,
            passedAssertions: 8
          },
          {
            id: 'thompson-sampling',
            name: 'Thompson Sampling集成',
            description: '测试Thompson Sampling算法的功能',
            status: 'passed',
            duration: 10000,
            assertions: 10,
            passedAssertions: 10
          },
          {
            id: 'epsilon-greedy',
            name: 'Epsilon Greedy集成',
            description: '测试Epsilon Greedy算法的集成',
            status: 'passed',
            duration: 7000,
            assertions: 6,
            passedAssertions: 6
          },
          {
            id: 'q-learning',
            name: 'Q-Learning集成',
            description: '测试Q-Learning算法的学习和推荐功能',
            status: 'passed',
            duration: 5000,
            assertions: 12,
            passedAssertions: 12
          }
        ]
      },
      {
        id: 'fault-tolerance',
        name: '容错集成测试',
        description: '测试系统在各种故障场景下的恢复能力',
        status: 'running',
        duration: 0,
        progress: 40,
        lastRun: '2025-08-22 14:25:00',
        tests: [
          {
            id: 'database-failure',
            name: '数据库故障恢复',
            description: '测试数据库连接失败时的系统行为',
            status: 'passed',
            duration: 8000,
            assertions: 5,
            passedAssertions: 5
          },
          {
            id: 'cache-failure',
            name: '缓存故障恢复',
            description: '测试Redis缓存故障时的降级处理',
            status: 'running',
            duration: 0,
            assertions: 8,
            passedAssertions: 3
          },
          {
            id: 'algorithm-failure',
            name: '算法故障切换',
            description: '测试主算法故障时的备选算法切换',
            status: 'pending',
            duration: 0,
            assertions: 6,
            passedAssertions: 0
          }
        ]
      }
    ];
    setTestSuites(suites);

    // 生成历史测试结果
    const results: TestResult[] = [];
    for (let i = 0; i < 50; i++) {
      results.push({
        suite: suites[Math.floor(Math.random() * suites.length)].name,
        testCase: `test-case-${i}`,
        timestamp: new Date(Date.now() - Math.random() * 86400000 * 7).toLocaleString(),
        status: Math.random() > 0.2 ? 'passed' : 'failed',
        duration: Math.floor(Math.random() * 30000 + 5000),
        error: Math.random() > 0.8 ? '连接超时或断言失败' : undefined
      });
    }
    setTestResults(results.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()));
  }, []);

  const runTestSuite = async (suiteId: string) => {
    setRunningTests(prev => new Set([...prev, suiteId]));
    
    setTestSuites(prev => prev.map(suite => 
      suite.id === suiteId 
        ? { ...suite, status: 'running', progress: 0 }
        : suite
    ));

    // 模拟测试执行
    for (let progress = 0; progress <= 100; progress += 10) {
      await new Promise(resolve => setTimeout(resolve, 500));
      setTestSuites(prev => prev.map(suite => 
        suite.id === suiteId 
          ? { ...suite, progress }
          : suite
      ));
    }

    // 模拟测试完成
    const success = Math.random() > 0.3;
    setTestSuites(prev => prev.map(suite => 
      suite.id === suiteId 
        ? { 
            ...suite, 
            status: success ? 'passed' : 'failed',
            duration: Math.floor(Math.random() * 60000 + 20000),
            lastRun: new Date().toLocaleString(),
            progress: 100
          }
        : suite
    ));

    setRunningTests(prev => {
      const newSet = new Set(prev);
      newSet.delete(suiteId);
      return newSet;
    });
  };

  const runAllTests = async () => {
    for (const suite of testSuites) {
      if (!runningTests.has(suite.id)) {
        await runTestSuite(suite.id);
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
  };

  const suiteColumns: ColumnsType<TestSuite> = [
    {
      title: '测试套件',
      dataIndex: 'name',
      key: 'name',
      render: (text, record) => (
        <div>
          <strong>{text}</strong>
          <div style={{ fontSize: '12px', color: '#666' }}>{record.description}</div>
        </div>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status, record) => {
        const config = {
          pending: { color: 'default', icon: <ClockCircleOutlined />, text: '等待中' },
          running: { color: 'processing', icon: <PlayCircleOutlined />, text: '运行中' },
          passed: { color: 'success', icon: <CheckCircleOutlined />, text: '通过' },
          failed: { color: 'error', icon: <CloseCircleOutlined />, text: '失败' },
          skipped: { color: 'warning', icon: <ClockCircleOutlined />, text: '跳过' }
        };
        return (
          <div>
            <Tag color={config[status].color} icon={config[status].icon}>
              {config[status].text}
            </Tag>
            {status === 'running' && (
              <Progress 
                percent={record.progress} 
                size="small" 
                style={{ marginTop: '4px', width: '100px' }}
              />
            )}
          </div>
        );
      },
    },
    {
      title: '测试用例',
      dataIndex: 'tests',
      key: 'tests',
      render: (tests: TestCase[]) => (
        <div>
          <div>总计: {tests.length}</div>
          <div style={{ fontSize: '12px', color: '#666' }}>
            通过: {tests.filter(t => t.status === 'passed').length} | 
            失败: {tests.filter(t => t.status === 'failed').length}
          </div>
        </div>
      ),
    },
    {
      title: '耗时',
      dataIndex: 'duration',
      key: 'duration',
      render: (duration) => duration > 0 ? `${(duration / 1000).toFixed(1)}s` : '-',
    },
    {
      title: '最后运行',
      dataIndex: 'lastRun',
      key: 'lastRun',
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Button
            type="primary"
            size="small"
            icon={<PlayCircleOutlined />}
            loading={runningTests.has(record.id)}
            onClick={() => runTestSuite(record.id)}
            disabled={runningTests.size > 0}
          >
            运行
          </Button>
          <Button
            size="small"
            icon={<FileTextOutlined />}
            onClick={() => {
              setSelectedSuite(record);
              setShowDetailsModal(true);
            }}
          >
            详情
          </Button>
        </Space>
      ),
    },
  ];

  const testCaseColumns: ColumnsType<TestCase> = [
    {
      title: '测试用例',
      dataIndex: 'name',
      key: 'name',
      render: (text, record) => (
        <div>
          <strong>{text}</strong>
          <div style={{ fontSize: '12px', color: '#666' }}>{record.description}</div>
        </div>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => {
        const config = {
          pending: { color: 'default', icon: <ClockCircleOutlined /> },
          running: { color: 'processing', icon: <PlayCircleOutlined /> },
          passed: { color: 'success', icon: <CheckCircleOutlined /> },
          failed: { color: 'error', icon: <CloseCircleOutlined /> },
          skipped: { color: 'warning', icon: <ClockCircleOutlined /> }
        };
        return <Tag color={config[status].color} icon={config[status].icon}>{status}</Tag>;
      },
    },
    {
      title: '断言',
      key: 'assertions',
      render: (_, record) => (
        <div>
          <Progress 
            percent={(record.passedAssertions / record.assertions) * 100}
            size="small"
            format={() => `${record.passedAssertions}/${record.assertions}`}
            status={record.passedAssertions === record.assertions ? 'success' : 'exception'}
          />
        </div>
      ),
    },
    {
      title: '耗时',
      dataIndex: 'duration',
      key: 'duration',
      render: (duration) => duration > 0 ? `${(duration / 1000).toFixed(1)}s` : '-',
    },
    {
      title: '错误信息',
      dataIndex: 'error',
      key: 'error',
      render: (error) => error ? (
        <Tag color="red" icon={<BugOutlined />}>
          {error.length > 30 ? `${error.substring(0, 30)}...` : error}
        </Tag>
      ) : '-',
    }
  ];

  const resultColumns: ColumnsType<TestResult> = [
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 150,
    },
    {
      title: '测试套件',
      dataIndex: 'suite',
      key: 'suite',
    },
    {
      title: '测试用例',
      dataIndex: 'testCase',
      key: 'testCase',
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => (
        <Tag color={status === 'passed' ? 'green' : 'red'}>
          {status === 'passed' ? '通过' : '失败'}
        </Tag>
      ),
    },
    {
      title: '耗时',
      dataIndex: 'duration',
      key: 'duration',
      render: (duration) => `${(duration / 1000).toFixed(1)}s`,
    },
    {
      title: '错误',
      dataIndex: 'error',
      key: 'error',
      render: (error) => error || '-',
    }
  ];

  const passedTests = testSuites.filter(s => s.status === 'passed').length;
  const failedTests = testSuites.filter(s => s.status === 'failed').length;
  const runningTestsCount = testSuites.filter(s => s.status === 'running').length;

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <h1 style={{ margin: 0, display: 'flex', alignItems: 'center' }}>
          <ExperimentOutlined style={{ marginRight: '8px' }} />
          强化学习集成测试
        </h1>
        <Space>
          <Select value={testEnv} onChange={setTestEnv} style={{ width: 120 }}>
            <Option value="local">本地环境</Option>
            <Option value="staging">测试环境</Option>
            <Option value="production">生产环境</Option>
          </Select>
          <Button 
            type="primary" 
            icon={<PlayCircleOutlined />}
            loading={runningTestsCount > 0}
            onClick={runAllTests}
            disabled={runningTestsCount > 0}
          >
            运行所有测试
          </Button>
          <Button 
            icon={<SettingOutlined />}
          >
            配置
          </Button>
        </Space>
      </div>

      {/* 测试概览统计 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="总测试套件"
              value={testSuites.length}
              prefix={<ExperimentOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="通过"
              value={passedTests}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="失败"
              value={failedTests}
              prefix={<CloseCircleOutlined />}
              valueStyle={{ color: '#cf1322' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <Statistic
              title="运行中"
              value={runningTestsCount}
              prefix={<PlayCircleOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 运行状态警告 */}
      {failedTests > 0 && (
        <Alert
          message="测试失败警告"
          description={`有 ${failedTests} 个测试套件执行失败，请检查详细错误信息并修复问题`}
          variant="destructive"
          showIcon
          closable
          style={{ marginBottom: '24px' }}
        />
      )}

      {/* 测试套件表格 */}
      <Card title="测试套件" style={{ marginBottom: '24px' }}>
        <Table
          dataSource={testSuites}
          columns={suiteColumns}
          rowKey="id"
          pagination={false}
          size="middle"
        />
      </Card>

      {/* 测试历史 */}
      <Card title="测试历史">
        <Table
          dataSource={testResults}
          columns={resultColumns}
          rowKey={(record, index) => `${record.suite}-${record.testCase}-${index}`}
          pagination={{ pageSize: 10 }}
          size="small"
        />
      </Card>

      {/* 测试详情弹窗 */}
      <Modal
        title={`测试套件详情: ${selectedSuite?.name}`}
        visible={showDetailsModal}
        onCancel={() => setShowDetailsModal(false)}
        footer={null}
        width={800}
      >
        {selectedSuite && (
          <Tabs defaultActiveKey="cases">
            <TabPane tab="测试用例" key="cases">
              <Table
                dataSource={selectedSuite.tests}
                columns={testCaseColumns}
                rowKey="id"
                pagination={false}
                size="small"
              />
            </TabPane>
            <TabPane tab="执行时间线" key="timeline">
              <Timeline>
                {selectedSuite.tests.map(test => (
                  <Timeline.Item
                    key={test.id}
                    color={test.status === 'passed' ? 'green' : test.status === 'failed' ? 'red' : 'blue'}
                    dot={test.status === 'passed' ? <CheckCircleOutlined /> : 
                         test.status === 'failed' ? <CloseCircleOutlined /> : 
                         <ClockCircleOutlined />}
                  >
                    <div>
                      <strong>{test.name}</strong>
                      <div style={{ fontSize: '12px', color: '#666' }}>
                        {test.description}
                      </div>
                      {test.duration > 0 && (
                        <div style={{ fontSize: '12px', color: '#999' }}>
                          耗时: {(test.duration / 1000).toFixed(1)}s
                        </div>
                      )}
                      {test.error && (
                        <div style={{ fontSize: '12px', color: '#f5222d' }}>
                          错误: {test.error}
                        </div>
                      )}
                    </div>
                  </Timeline.Item>
                ))}
              </Timeline>
            </TabPane>
          </Tabs>
        )}
      </Modal>
    </div>
  );
};

export default RLIntegrationTestPage;