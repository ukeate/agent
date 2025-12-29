import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useState, useEffect } from 'react';
import {
import { logger } from '../utils/logger'
  Card,
  Table,
  Tag,
  Button,
  Space,
  Row,
  Col,
  Statistic,
  Select,
  Input,
  DatePicker,
  Alert,
  Modal,
  Form,
  Descriptions,
  Progress,
  Timeline,
  Badge
} from 'antd';
import {
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  WarningOutlined,
  ReloadOutlined,
  EyeOutlined,
  SettingOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  SearchOutlined,
  FilterOutlined,
  BugOutlined
} from '@ant-design/icons';
import { RangePickerProps } from 'antd/es/date-picker';
import dayjs from 'dayjs';

const { Option } = Select;
const { Search } = Input;
const { RangePicker } = DatePicker;

interface FaultEvent {
  fault_id: string;
  fault_type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  affected_components: string[];
  detected_at: string;
  resolved_at?: string;
  description: string;
  resolved: boolean;
  context: Record<string, any>;
  recovery_actions?: string[];
}

interface ComponentHealth {
  component_id: string;
  status: 'healthy' | 'degraded' | 'unhealthy' | 'unknown';
  last_check: string;
  response_time: number;
  error_rate: number;
  resource_usage: {
    cpu: number;
    memory: number;
    disk?: number;
  };
  custom_metrics?: Record<string, any>;
}

interface HealthSummary {
  total_components: number;
  status_counts: {
    healthy: number;
    degraded: number;
    unhealthy: number;
    unknown: number;
  };
  avg_response_time: number;
  avg_error_rate: number;
  health_ratio: number;
  active_faults: number;
  last_update: string;
}

const FaultDetectionPage: React.FC = () => {
  const [faultEvents, setFaultEvents] = useState<FaultEvent[]>([]);
  const [healthSummary, setHealthSummary] = useState<HealthSummary | null>(null);
  const [components, setComponents] = useState<ComponentHealth[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedFault, setSelectedFault] = useState<FaultEvent | null>(null);
  const [modalVisible, setModalVisible] = useState(false);
  
  // 过滤器状态
  const [filters, setFilters] = useState({
    fault_type: '',
    severity: '',
    resolved: '',
    search: '',
    date_range: null as [dayjs.Dayjs, dayjs.Dayjs] | null
  });

  const fetchFaultEvents = async () => {
    try {
      const params = new URLSearchParams();
      if (filters.fault_type) params.append('fault_type', filters.fault_type);
      if (filters.severity) params.append('severity', filters.severity);
      if (filters.resolved !== '') params.append('resolved', filters.resolved);
      params.append('limit', '100');

      const response = await apiFetch(buildApiUrl(`/api/v1/fault-tolerance/faults?${params}`));
      const data = await response.json();
      setFaultEvents(data);
    } catch (error) {
      logger.error('获取故障事件失败:', error);
    }
  };

  const fetchHealthSummary = async () => {
    try {
      const response = await apiFetch(buildApiUrl('/api/v1/fault-tolerance/health'));
      const data = await response.json();
      setHealthSummary(data);
    } catch (error) {
      logger.error('获取健康摘要失败:', error);
    }
  };

  const fetchComponents = async () => {
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/fault-tolerance/health'));
      const data = await res.json();
      setComponents(data?.components || []);
      setHealthSummary((prev) => prev || data);
    } catch (error) {
      logger.error('获取组件状态失败:', error);
      setComponents([]);
    }
  };

  const injectTestFault = async () => {
    try {
      const response = await apiFetch(buildApiUrl('/api/v1/fault-tolerance/testing/inject-fault'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          component_id: 'agent-test',
          fault_type: 'agent_error',
          duration_seconds: 60
        })
      });
      await response.json().catch(() => null);
      Modal.success({
        title: '故障注入成功',
        content: '测试故障已成功注入，将在60秒后自动恢复'
      });
      setTimeout(fetchFaultEvents, 1000);
    } catch (error) {
      logger.error('故障注入失败:', error);
    }
  };

  useEffect(() => {
    const loadData = async () => {
      setLoading(true);
      await Promise.all([
        fetchFaultEvents(),
        fetchHealthSummary(),
        fetchComponents()
      ]);
      setLoading(false);
    };

    loadData();
    const interval = setInterval(loadData, 15000); // 每15秒刷新
    return () => clearInterval(interval);
  }, [filters]);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'low': return 'blue';
      case 'medium': return 'orange';
      case 'high': return 'red';
      case 'critical': return 'red';
      default: return 'default';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'success';
      case 'degraded': return 'warning';
      case 'unhealthy': return 'error';
      default: return 'default';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'degraded': return <WarningOutlined style={{ color: '#faad14' }} />;
      case 'unhealthy': return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />;
      default: return <ExclamationCircleOutlined style={{ color: '#d9d9d9' }} />;
    }
  };

  const faultColumns = [
    {
      title: '故障ID',
      dataIndex: 'fault_id',
      key: 'fault_id',
      render: (id: string) => <code>{id.slice(0, 12)}...</code>
    },
    {
      title: '故障类型',
      dataIndex: 'fault_type',
      key: 'fault_type',
      render: (type: string) => <Tag color="blue">{type}</Tag>
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
      title: '检测时间',
      dataIndex: 'detected_at',
      key: 'detected_at',
      render: (time: string) => new Date(time).toLocaleString()
    },
    {
      title: '状态',
      dataIndex: 'resolved',
      key: 'resolved',
      render: (resolved: boolean) => (
        <Tag color={resolved ? 'green' : 'red'}>
          {resolved ? '已解决' : '未解决'}
        </Tag>
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (_: any, record: FaultEvent) => (
        <Space>
          <Button 
            size="small" 
            icon={<EyeOutlined />}
            onClick={() => {
              setSelectedFault(record);
              setModalVisible(true);
            }}
          >
            详情
          </Button>
        </Space>
      )
    }
  ];

  const componentColumns = [
    {
      title: '组件ID',
      dataIndex: 'component_id',
      key: 'component_id',
      render: (id: string) => <Badge status="processing" text={id} />
    },
    {
      title: '健康状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={getStatusColor(status)} icon={getStatusIcon(status)}>
          {status.toUpperCase()}
        </Tag>
      )
    },
    {
      title: '响应时间',
      dataIndex: 'response_time',
      key: 'response_time',
      render: (time: number) => (
        <span className={time > 2 ? 'text-red-500' : time > 1 ? 'text-yellow-500' : 'text-green-500'}>
          {time.toFixed(2)}s
        </span>
      )
    },
    {
      title: '错误率',
      dataIndex: 'error_rate',
      key: 'error_rate',
      render: (rate: number) => (
        <span className={rate > 0.1 ? 'text-red-500' : rate > 0.05 ? 'text-yellow-500' : 'text-green-500'}>
          {(rate * 100).toFixed(2)}%
        </span>
      )
    },
    {
      title: 'CPU',
      dataIndex: ['resource_usage', 'cpu'],
      key: 'cpu',
      render: (cpu: number) => (
        <Progress
          percent={cpu}
          size="small"
          status={cpu > 90 ? 'exception' : cpu > 70 ? 'active' : 'success'}
          format={() => `${cpu}%`}
        />
      )
    },
    {
      title: '内存',
      dataIndex: ['resource_usage', 'memory'],
      key: 'memory',
      render: (memory: number) => (
        <Progress
          percent={memory}
          size="small"
          status={memory > 90 ? 'exception' : memory > 70 ? 'active' : 'success'}
          format={() => `${memory}%`}
        />
      )
    },
    {
      title: '最后检查',
      dataIndex: 'last_check',
      key: 'last_check',
      render: (time: string) => new Date(time).toLocaleTimeString()
    }
  ];

  return (
    <div className="fault-detection-page p-6">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold mb-2">故障检测与监控</h1>
          <p className="text-gray-600">实时监控系统健康状态，检测和分析故障事件</p>
        </div>
        <Space>
          <Button 
            icon={<BugOutlined />} 
            onClick={injectTestFault}
            type="dashed"
          >
            注入测试故障
          </Button>
          <Button 
            icon={<SettingOutlined />} 
            href="/fault-tolerance/detection/settings"
          >
            检测设置
          </Button>
          <Button 
            icon={<ReloadOutlined />} 
            onClick={() => {
              fetchFaultEvents();
              fetchHealthSummary();
              fetchComponents();
            }}
            loading={loading}
          >
            刷新数据
          </Button>
        </Space>
      </div>

      {/* 系统健康概览 */}
      <Row gutter={[16, 16]} className="mb-6">
        <Col span={6}>
          <Card>
            <Statistic
              title="健康组件"
              value={healthSummary?.status_counts.healthy || 0}
              suffix={`/ ${healthSummary?.total_components || 0}`}
              valueStyle={{ color: '#3f8600' }}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="降级组件"
              value={healthSummary?.status_counts.degraded || 0}
              valueStyle={{ color: '#faad14' }}
              prefix={<WarningOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="异常组件"
              value={healthSummary?.status_counts.unhealthy || 0}
              valueStyle={{ color: '#cf1322' }}
              prefix={<CloseCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均响应时间"
              value={healthSummary?.avg_response_time || 0}
              suffix="ms"
              precision={2}
              valueStyle={{ 
                color: (healthSummary?.avg_response_time || 0) > 1000 ? '#cf1322' : '#3f8600' 
              }}
              prefix={<ExclamationCircleOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* 活跃故障告警 */}
      {faultEvents.filter(f => !f.resolved).length > 0 && (
        <Alert
          message={`检测到 ${faultEvents.filter(f => !f.resolved).length} 个未解决故障`}
          description="请及时关注和处理活跃的故障事件"
          type="error"
          showIcon
          closable
          className="mb-6"
        />
      )}

      {/* 故障事件列表 */}
      <Card 
        title="故障事件记录" 
        className="mb-6"
        extra={
          <Space>
            <Select
              placeholder="故障类型"
              allowClear
              style={{ width: 120 }}
              onChange={(value) => setFilters({...filters, fault_type: value || ''})}
            >
              <Option value="agent_error">智能体错误</Option>
              <Option value="agent_unresponsive">智能体无响应</Option>
              <Option value="performance_degradation">性能降级</Option>
              <Option value="resource_exhaustion">资源耗尽</Option>
              <Option value="network_partition">网络分区</Option>
            </Select>
            <Select
              placeholder="严重程度"
              allowClear
              style={{ width: 100 }}
              onChange={(value) => setFilters({...filters, severity: value || ''})}
            >
              <Option value="low">低</Option>
              <Option value="medium">中</Option>
              <Option value="high">高</Option>
              <Option value="critical">严重</Option>
            </Select>
            <Select
              placeholder="状态"
              allowClear
              style={{ width: 100 }}
              onChange={(value) => setFilters({...filters, resolved: value || ''})}
            >
              <Option value="true">已解决</Option>
              <Option value="false">未解决</Option>
            </Select>
          </Space>
        }
      >
        <Table
          columns={faultColumns}
          dataSource={faultEvents}
          rowKey="fault_id"
          loading={loading}
          pagination={{
            pageSize: 20,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total) => `共 ${total} 个故障事件`
          }}
        />
      </Card>

      {/* 组件健康状态 */}
      <Card title="组件健康状态">
        <Table
          columns={componentColumns}
          dataSource={components}
          rowKey="component_id"
          loading={loading}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total) => `共 ${total} 个组件`
          }}
        />
      </Card>

      {/* 故障详情弹窗 */}
      <Modal
        title="故障详情"
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setModalVisible(false)}>
            关闭
          </Button>
        ]}
        width={800}
      >
        {selectedFault && (
          <div>
            <Descriptions bordered column={2}>
              <Descriptions.Item label="故障ID" span={2}>
                <code>{selectedFault.fault_id}</code>
              </Descriptions.Item>
              <Descriptions.Item label="故障类型">
                <Tag color="blue">{selectedFault.fault_type}</Tag>
              </Descriptions.Item>
              <Descriptions.Item label="严重程度">
                <Tag color={getSeverityColor(selectedFault.severity)}>
                  {selectedFault.severity.toUpperCase()}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="状态">
                <Tag color={selectedFault.resolved ? 'green' : 'red'}>
                  {selectedFault.resolved ? '已解决' : '未解决'}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="检测时间">
                {new Date(selectedFault.detected_at).toLocaleString()}
              </Descriptions.Item>
              {selectedFault.resolved_at && (
                <Descriptions.Item label="解决时间">
                  {new Date(selectedFault.resolved_at).toLocaleString()}
                </Descriptions.Item>
              )}
              <Descriptions.Item label="描述" span={2}>
                {selectedFault.description}
              </Descriptions.Item>
              <Descriptions.Item label="影响组件" span={2}>
                {selectedFault.affected_components.map(comp => (
                  <Tag key={comp}>{comp}</Tag>
                ))}
              </Descriptions.Item>
            </Descriptions>

            {selectedFault.recovery_actions && selectedFault.recovery_actions.length > 0 && (
              <div className="mt-4">
                <h4>恢复操作记录:</h4>
                <Timeline>
                  {selectedFault.recovery_actions.map((action, index) => (
                    <Timeline.Item key={index}>
                      {action}
                    </Timeline.Item>
                  ))}
                </Timeline>
              </div>
            )}

            {Object.keys(selectedFault.context).length > 0 && (
              <div className="mt-4">
                <h4>上下文信息:</h4>
                <pre className="bg-gray-100 p-3 rounded">
                  {JSON.stringify(selectedFault.context, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
};

export default FaultDetectionPage;
