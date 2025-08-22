import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Tabs, 
  Typography, 
  Row, 
  Col, 
  Button, 
  Table, 
  Tag,
  Statistic,
  Space,
  Form,
  Input,
  Select,
  InputNumber,
  Switch,
  Divider,
  Progress,
  Alert,
  Modal,
  Descriptions,
  List,
  message
} from 'antd';
import { 
  SettingOutlined, 
  ThunderboltOutlined, 
  DatabaseOutlined, 
  BranchesOutlined as PipelineOutlined,
  PlayCircleOutlined,
  ReloadOutlined,
  BarChartOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  RiseOutlined,
  NodeIndexOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

// 处理模式枚举
const ProcessingModes = {
  STREAM: 'stream',
  BATCH: 'batch', 
  HYBRID: 'hybrid',
  AUTO: 'auto',
  PIPELINE: 'pipeline'
};

// 选择策略枚举
const SelectionStrategies = {
  HEURISTIC: 'heuristic',
  PERFORMANCE_BASED: 'performance',
  LOAD_AWARE: 'load_aware',
  HYBRID: 'hybrid'
};

interface ProcessingItem {
  id: string;
  data: any;
  priority: number;
  metadata: Record<string, any>;
}

interface ProcessingRequest {
  session_id: string;
  items: ProcessingItem[];
  mode?: string;
  requires_real_time: boolean;
  streaming_enabled: boolean;
  batch_size?: number;
  max_parallel_tasks: number;
  requires_aggregation: boolean;
  aggregation_strategy: string;
  timeout?: number;
}

interface ProcessingResponse {
  request_id: string;
  session_id: string;
  mode_used: string;
  status: string;
  progress: number;
  results: any[];
  aggregated_result?: any;
  processing_time?: number;
  errors: any[];
  success_rate: number;
}

interface SystemMetrics {
  total_requests: number;
  total_items_processed: number;
  active_sessions: number;
  processing_history_size: number;
  average_processing_time: number;
  success_rate: number;
  mode_usage_stats: Record<string, number>;
  default_mode: string;
}

interface ModeRecommendation {
  mode: string;
  score: number;
  heuristic_score: number;
  performance_score: number;
  request_count: number;
  success_rate: number;
  avg_processing_time: number;
}

const UnifiedEnginePage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null);
  const [processingHistory, setProcessingHistory] = useState<ProcessingResponse[]>([]);
  const [modeRecommendations, setModeRecommendations] = useState<ModeRecommendation[]>([]);
  const [activeRequests, setActiveRequests] = useState<Record<string, ProcessingResponse>>({});
  
  // 表单状态
  const [form] = Form.useForm();
  const [currentSessionId, setCurrentSessionId] = useState(`session_${Date.now()}`);
  
  // 模态框状态
  const [recommendModalVisible, setRecommendModalVisible] = useState(false);
  const [requestModalVisible, setRequestModalVisible] = useState(false);
  const [selectedRequest, setSelectedRequest] = useState<ProcessingResponse | null>(null);

  // 获取系统指标
  const fetchSystemMetrics = async () => {
    try {
      const response = await fetch('/api/v1/unified/metrics');
      if (response.ok) {
        const data = await response.json();
        setSystemMetrics(data);
      }
    } catch (error) {
      console.error('获取系统指标失败:', error);
    }
  };

  // 获取处理历史
  const fetchProcessingHistory = async () => {
    try {
      const response = await fetch('/api/v1/unified/history?limit=20');
      if (response.ok) {
        const data = await response.json();
        setProcessingHistory(data);
      }
    } catch (error) {
      console.error('获取处理历史失败:', error);
    }
  };

  // 获取模式推荐
  const fetchModeRecommendations = async (request: ProcessingRequest) => {
    try {
      const response = await fetch('/api/v1/unified/mode/recommendations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request)
      });
      
      if (response.ok) {
        const data = await response.json();
        setModeRecommendations(data);
        setRecommendModalVisible(true);
      }
    } catch (error) {
      console.error('获取模式推荐失败:', error);
      message.error('获取模式推荐失败');
    }
  };

  // 提交处理请求
  const submitProcessingRequest = async (values: any) => {
    setLoading(true);
    
    try {
      // 构建处理项目
      const items: ProcessingItem[] = [];
      const itemCount = values.item_count || 5;
      
      for (let i = 0; i < itemCount; i++) {
        items.push({
          id: `item_${i + 1}`,
          data: values.sample_data || `示例数据项目 ${i + 1}`,
          priority: values.priority || 5,
          metadata: {
            created_at: new Date().toISOString(),
            index: i
          }
        });
      }
      
      const request: ProcessingRequest = {
        session_id: currentSessionId,
        items,
        mode: values.mode === 'auto' ? undefined : values.mode,
        requires_real_time: values.requires_real_time || false,
        streaming_enabled: values.streaming_enabled !== false,
        batch_size: values.batch_size,
        max_parallel_tasks: values.max_parallel_tasks || 10,
        requires_aggregation: values.requires_aggregation || false,
        aggregation_strategy: values.aggregation_strategy || 'collect',
        timeout: values.timeout
      };
      
      const response = await fetch('/api/v1/unified/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request)
      });
      
      if (response.ok) {
        const result = await response.json();
        setActiveRequests(prev => ({
          ...prev,
          [currentSessionId]: result
        }));
        
        message.success('处理请求已提交');
        
        // 生成新的会话ID用于下次请求
        setCurrentSessionId(`session_${Date.now()}`);
        
        // 刷新数据
        await Promise.all([
          fetchSystemMetrics(),
          fetchProcessingHistory()
        ]);
      } else {
        const error = await response.json();
        message.error(`提交失败: ${error.detail || '未知错误'}`);
      }
    } catch (error) {
      console.error('提交处理请求失败:', error);
      message.error('提交处理请求失败');
    } finally {
      setLoading(false);
    }
  };

  // 获取推荐（不提交请求）
  const getRecommendationsOnly = async () => {
    const values = form.getFieldsValue();
    
    // 构建处理项目
    const items: ProcessingItem[] = [];
    const itemCount = values.item_count || 5;
    
    for (let i = 0; i < itemCount; i++) {
      items.push({
        id: `item_${i + 1}`,
        data: values.sample_data || `示例数据项目 ${i + 1}`,
        priority: values.priority || 5,
        metadata: { index: i }
      });
    }
    
    const request: ProcessingRequest = {
      session_id: currentSessionId,
      items,
      requires_real_time: values.requires_real_time || false,
      streaming_enabled: values.streaming_enabled !== false,
      batch_size: values.batch_size,
      max_parallel_tasks: values.max_parallel_tasks || 10,
      requires_aggregation: values.requires_aggregation || false,
      aggregation_strategy: values.aggregation_strategy || 'collect',
      timeout: values.timeout
    };
    
    await fetchModeRecommendations(request);
  };

  useEffect(() => {
    fetchSystemMetrics();
    fetchProcessingHistory();
    
    // 定期刷新数据
    const interval = setInterval(() => {
      fetchSystemMetrics();
      fetchProcessingHistory();
    }, 5000);
    
    return () => clearInterval(interval);
  }, []);

  const getModeIcon = (mode: string) => {
    switch (mode) {
      case 'stream': return <ThunderboltOutlined />;
      case 'batch': return <DatabaseOutlined />;
      case 'hybrid': return <BranchesOutlined />;
      case 'pipeline': return <PipelineOutlined />;
      default: return <SettingOutlined />;
    }
  };

  const getModeColor = (mode: string) => {
    switch (mode) {
      case 'stream': return 'blue';
      case 'batch': return 'green';
      case 'hybrid': return 'orange';
      case 'pipeline': return 'purple';
      default: return 'default';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success';
      case 'processing': return 'processing';
      case 'failed': return 'error';
      default: return 'default';
    }
  };

  // 历史记录表格列定义
  const historyColumns = [
    {
      title: '请求ID',
      dataIndex: 'request_id',
      key: 'request_id',
      width: 120,
      render: (text: string) => (
        <Text code style={{ fontSize: '12px' }}>
          {text.slice(-8)}
        </Text>
      )
    },
    {
      title: '模式',
      dataIndex: 'mode_used',
      key: 'mode_used',
      width: 100,
      render: (mode: string) => (
        <Tag color={getModeColor(mode)} icon={getModeIcon(mode)}>
          {mode.toUpperCase()}
        </Tag>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 80,
      render: (status: string) => (
        <Tag color={getStatusColor(status)}>
          {status}
        </Tag>
      )
    },
    {
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      width: 120,
      render: (progress: number) => (
        <Progress 
          percent={Math.round(progress * 100)} 
          size="small" 
          status={progress === 1 ? 'success' : 'active'}
        />
      )
    },
    {
      title: '项目数',
      dataIndex: 'results',
      key: 'item_count',
      width: 80,
      render: (results: any[]) => results.length
    },
    {
      title: '成功率',
      dataIndex: 'success_rate',
      key: 'success_rate',
      width: 80,
      render: (rate: number) => `${(rate * 100).toFixed(1)}%`
    },
    {
      title: '处理时间',
      dataIndex: 'processing_time',
      key: 'processing_time',
      width: 100,
      render: (time: number) => time ? `${time.toFixed(2)}s` : '-'
    },
    {
      title: '操作',
      key: 'actions',
      width: 80,
      render: (_, record: ProcessingResponse) => (
        <Button 
          size="small" 
          onClick={() => {
            setSelectedRequest(record);
            setRequestModalVisible(true);
          }}
        >
          详情
        </Button>
      )
    }
  ];

  return (
    <div style={{ padding: '24px' }}>
      {/* 页面标题 */}
      <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Title level={2}>
          <SettingOutlined style={{ marginRight: '12px' }} />
          统一处理引擎 (流批一体)
        </Title>
        <Space>
          <Button onClick={fetchSystemMetrics} icon={<ReloadOutlined />}>
            刷新指标
          </Button>
        </Space>
      </div>

      {/* 系统概览指标 */}
      <Row gutter={24} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="总处理请求"
              value={systemMetrics?.total_requests || 0}
              prefix={<PlayCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="活跃会话"
              value={systemMetrics?.active_sessions || 0}
              prefix={<ThunderboltOutlined />}
              suffix={
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  历史: {systemMetrics?.processing_history_size || 0}
                </Text>
              }
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均处理时间"
              value={systemMetrics?.average_processing_time || 0}
              precision={2}
              suffix="s"
              prefix={<ClockCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="成功率"
              value={(systemMetrics?.success_rate || 0) * 100}
              precision={1}
              suffix="%"
              prefix={<RiseOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* 主要功能区域 */}
      <Card>
        <Tabs defaultActiveKey="processing" size="large">
          
          {/* 处理请求标签页 */}
          <TabPane
            tab={
              <span>
                <PlayCircleOutlined />
                处理请求
              </span>
            }
            key="processing"
          >
            <Row gutter={24}>
              <Col span={16}>
                <Card title="提交处理请求" style={{ marginBottom: '16px' }}>
                  <Form
                    form={form}
                    layout="vertical"
                    onFinish={submitProcessingRequest}
                    initialValues={{
                      mode: 'auto',
                      item_count: 5,
                      priority: 5,
                      max_parallel_tasks: 10,
                      streaming_enabled: true,
                      aggregation_strategy: 'collect'
                    }}
                  >
                    <Row gutter={16}>
                      <Col span={12}>
                        <Form.Item label="处理模式" name="mode">
                          <Select>
                            <Option value="auto">
                              <SettingOutlined /> 自动选择
                            </Option>
                            <Option value="stream">
                              <ThunderboltOutlined /> 流式处理
                            </Option>
                            <Option value="batch">
                              <DatabaseOutlined /> 批处理
                            </Option>
                            <Option value="hybrid">
                              <BranchesOutlined /> 混合处理
                            </Option>
                            <Option value="pipeline">
                              <PipelineOutlined /> 流水线处理
                            </Option>
                          </Select>
                        </Form.Item>
                      </Col>
                      <Col span={12}>
                        <Form.Item label="数据项目数量" name="item_count">
                          <InputNumber min={1} max={100} style={{ width: '100%' }} />
                        </Form.Item>
                      </Col>
                    </Row>

                    <Row gutter={16}>
                      <Col span={12}>
                        <Form.Item label="优先级" name="priority">
                          <InputNumber min={1} max={10} style={{ width: '100%' }} />
                        </Form.Item>
                      </Col>
                      <Col span={12}>
                        <Form.Item label="最大并行任务" name="max_parallel_tasks">
                          <InputNumber min={1} max={50} style={{ width: '100%' }} />
                        </Form.Item>
                      </Col>
                    </Row>

                    <Row gutter={16}>
                      <Col span={8}>
                        <Form.Item label="实时处理" name="requires_real_time" valuePropName="checked">
                          <Switch />
                        </Form.Item>
                      </Col>
                      <Col span={8}>
                        <Form.Item label="启用流式" name="streaming_enabled" valuePropName="checked">
                          <Switch defaultChecked />
                        </Form.Item>
                      </Col>
                      <Col span={8}>
                        <Form.Item label="需要聚合" name="requires_aggregation" valuePropName="checked">
                          <Switch />
                        </Form.Item>
                      </Col>
                    </Row>

                    <Form.Item label="示例数据" name="sample_data">
                      <Input.TextArea 
                        rows={3} 
                        placeholder="输入要处理的示例数据..." 
                      />
                    </Form.Item>

                    <Row gutter={16}>
                      <Col span={12}>
                        <Form.Item label="批处理大小" name="batch_size">
                          <InputNumber min={1} max={1000} style={{ width: '100%' }} placeholder="可选" />
                        </Form.Item>
                      </Col>
                      <Col span={12}>
                        <Form.Item label="超时时间(秒)" name="timeout">
                          <InputNumber min={1} max={3600} style={{ width: '100%' }} placeholder="可选" />
                        </Form.Item>
                      </Col>
                    </Row>

                    <Form.Item label="聚合策略" name="aggregation_strategy">
                      <Select>
                        <Option value="collect">收集</Option>
                        <Option value="merge">合并</Option>
                        <Option value="concat">连接</Option>
                        <Option value="sum">求和</Option>
                      </Select>
                    </Form.Item>

                    <Divider />

                    <Space>
                      <Button 
                        type="primary" 
                        htmlType="submit" 
                        loading={loading}
                        icon={<PlayCircleOutlined />}
                      >
                        提交处理请求
                      </Button>
                      <Button 
                        onClick={getRecommendationsOnly}
                        icon={<BarChartOutlined />}
                      >
                        获取模式推荐
                      </Button>
                    </Space>
                  </Form>
                </Card>
              </Col>

              <Col span={8}>
                <Card title="模式使用统计" size="small">
                  {systemMetrics?.mode_usage_stats && (
                    <div>
                      {Object.entries(systemMetrics.mode_usage_stats).map(([mode, count]) => (
                        <div key={mode} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                          <Tag color={getModeColor(mode)} icon={getModeIcon(mode)}>
                            {mode.toUpperCase()}
                          </Tag>
                          <Text strong>{count}</Text>
                        </div>
                      ))}
                    </div>
                  )}
                </Card>

                <Card title="当前会话" size="small" style={{ marginTop: '16px' }}>
                  <Text code style={{ fontSize: '12px' }}>
                    {currentSessionId}
                  </Text>
                </Card>
              </Col>
            </Row>
          </TabPane>

          {/* 处理历史标签页 */}
          <TabPane
            tab={
              <span>
                <DatabaseOutlined />
                处理历史
              </span>
            }
            key="history"
          >
            <Card title="最近处理历史">
              <Table
                columns={historyColumns}
                dataSource={processingHistory}
                rowKey="request_id"
                size="small"
                scroll={{ x: 900 }}
                pagination={{
                  pageSize: 10,
                  showSizeChanger: true,
                  showQuickJumper: true,
                  showTotal: (total) => `共 ${total} 条记录`
                }}
              />
            </Card>
          </TabPane>

          {/* 性能分析标签页 */}
          <TabPane
            tab={
              <span>
                <BarChartOutlined />
                性能分析
              </span>
            }
            key="analytics"
          >
            <Row gutter={16}>
              <Col span={12}>
                <Card title="处理模式性能对比" style={{ marginBottom: '16px' }}>
                  {systemMetrics && (
                    <div>
                      <Alert
                        message="性能指标"
                        description={`总处理项目: ${systemMetrics.total_items_processed} | 平均处理时间: ${systemMetrics.average_processing_time.toFixed(2)}s`}
                        variant="default"
                        style={{ marginBottom: '16px' }}
                      />
                      
                      <div>
                        {Object.entries(systemMetrics.mode_usage_stats).map(([mode, count]) => {
                          const percentage = systemMetrics.total_requests > 0 
                            ? (count / systemMetrics.total_requests) * 100 
                            : 0;
                          
                          return (
                            <div key={mode} style={{ marginBottom: '12px' }}>
                              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                                <span>
                                  <Tag color={getModeColor(mode)} icon={getModeIcon(mode)}>
                                    {mode.toUpperCase()}
                                  </Tag>
                                </span>
                                <span>{count} 次 ({percentage.toFixed(1)}%)</span>
                              </div>
                              <Progress 
                                percent={percentage} 
                                strokeColor={
                                  mode === 'stream' ? '#1890ff' :
                                  mode === 'batch' ? '#52c41a' :
                                  mode === 'hybrid' ? '#fa8c16' :
                                  mode === 'pipeline' ? '#722ed1' : '#d9d9d9'
                                }
                                size="small"
                              />
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </Card>
              </Col>

              <Col span={12}>
                <Card title="系统状态" style={{ marginBottom: '16px' }}>
                  <Descriptions column={1} size="small">
                    <Descriptions.Item label="默认模式">
                      <Tag color={getModeColor(systemMetrics?.default_mode || 'auto')}>
                        {(systemMetrics?.default_mode || 'auto').toUpperCase()}
                      </Tag>
                    </Descriptions.Item>
                    <Descriptions.Item label="活跃会话">
                      {systemMetrics?.active_sessions || 0}
                    </Descriptions.Item>
                    <Descriptions.Item label="历史记录">
                      {systemMetrics?.processing_history_size || 0}
                    </Descriptions.Item>
                    <Descriptions.Item label="总处理项目">
                      {systemMetrics?.total_items_processed || 0}
                    </Descriptions.Item>
                  </Descriptions>
                </Card>
              </Col>
            </Row>
          </TabPane>

        </Tabs>
      </Card>

      {/* 模式推荐模态框 */}
      <Modal
        title="处理模式推荐"
        visible={recommendModalVisible}
        onCancel={() => setRecommendModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setRecommendModalVisible(false)}>
            关闭
          </Button>
        ]}
        width={800}
      >
        <List
          dataSource={modeRecommendations}
          renderItem={(item: ModeRecommendation) => (
            <List.Item>
              <Card size="small" style={{ width: '100%' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div>
                    <Tag color={getModeColor(item.mode)} icon={getModeIcon(item.mode)}>
                      {item.mode.toUpperCase()}
                    </Tag>
                    <Text strong style={{ marginLeft: '8px' }}>
                      综合评分: {(item.score * 100).toFixed(1)}%
                    </Text>
                  </div>
                  <div>
                    <Text type="secondary">
                      历史请求: {item.request_count} | 成功率: {(item.success_rate * 100).toFixed(1)}%
                    </Text>
                  </div>
                </div>
                <div style={{ marginTop: '8px' }}>
                  <Progress 
                    percent={item.score * 100} 
                    size="small" 
                    strokeColor={getModeColor(item.mode) === 'blue' ? '#1890ff' : 
                               getModeColor(item.mode) === 'green' ? '#52c41a' :
                               getModeColor(item.mode) === 'orange' ? '#fa8c16' : '#722ed1'}
                  />
                </div>
                <div style={{ marginTop: '8px', fontSize: '12px' }}>
                  <Space>
                    <Text type="secondary">启发式: {(item.heuristic_score * 100).toFixed(0)}%</Text>
                    <Text type="secondary">性能: {(item.performance_score * 100).toFixed(0)}%</Text>
                    <Text type="secondary">平均耗时: {item.avg_processing_time.toFixed(2)}s</Text>
                  </Space>
                </div>
              </Card>
            </List.Item>
          )}
        />
      </Modal>

      {/* 请求详情模态框 */}
      <Modal
        title="处理请求详情"
        visible={requestModalVisible}
        onCancel={() => setRequestModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setRequestModalVisible(false)}>
            关闭
          </Button>
        ]}
        width={800}
      >
        {selectedRequest && (
          <div>
            <Descriptions column={2} bordered size="small">
              <Descriptions.Item label="请求ID">
                <Text code>{selectedRequest.request_id}</Text>
              </Descriptions.Item>
              <Descriptions.Item label="会话ID">
                <Text code>{selectedRequest.session_id}</Text>
              </Descriptions.Item>
              <Descriptions.Item label="处理模式">
                <Tag color={getModeColor(selectedRequest.mode_used)} icon={getModeIcon(selectedRequest.mode_used)}>
                  {selectedRequest.mode_used.toUpperCase()}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="状态">
                <Tag color={getStatusColor(selectedRequest.status)}>
                  {selectedRequest.status}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="进度">
                <Progress percent={Math.round(selectedRequest.progress * 100)} size="small" />
              </Descriptions.Item>
              <Descriptions.Item label="成功率">
                {(selectedRequest.success_rate * 100).toFixed(1)}%
              </Descriptions.Item>
              <Descriptions.Item label="处理时间">
                {selectedRequest.processing_time ? `${selectedRequest.processing_time.toFixed(2)}s` : '-'}
              </Descriptions.Item>
              <Descriptions.Item label="结果数量">
                {selectedRequest.results.length}
              </Descriptions.Item>
            </Descriptions>

            {selectedRequest.errors.length > 0 && (
              <div style={{ marginTop: '16px' }}>
                <Alert
                  message="处理错误"
                  description={
                    <ul>
                      {selectedRequest.errors.map((error, index) => (
                        <li key={index}>{JSON.stringify(error)}</li>
                      ))}
                    </ul>
                  }
                  variant="destructive"
                  showIcon
                />
              </div>
            )}

            {selectedRequest.aggregated_result && (
              <div style={{ marginTop: '16px' }}>
                <Card title="聚合结果" size="small">
                  <pre style={{ fontSize: '12px', margin: 0, whiteSpace: 'pre-wrap' }}>
                    {JSON.stringify(selectedRequest.aggregated_result, null, 2)}
                  </pre>
                </Card>
              </div>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
};

export default UnifiedEnginePage;