import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Table, Button, Modal, Form, Input, Select, Tag, Space, Tabs, Switch, message, Tooltip, Popconfirm, InputNumber, Progress } from 'antd';
import { ApiOutlined, PlusOutlined, EditOutlined, DeleteOutlined, EyeOutlined, PlayCircleOutlined, PauseCircleOutlined, SettingOutlined, SafetyCertificateOutlined, KeyOutlined, FieldTimeOutlined, DatabaseOutlined } from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import dayjs from 'dayjs';

const { Option } = Select;
const { TextArea } = Input;
const { TabPane } = Tabs;

// 数据接口定义
interface ApiEndpoint {
  id: string;
  name: string;
  path: string;
  method: 'GET' | 'POST' | 'PUT' | 'DELETE';
  description: string;
  status: 'active' | 'disabled' | 'deprecated';
  category: 'evaluation' | 'model' | 'benchmark' | 'report' | 'admin';
  version: string;
  rateLimitPer24h: number;
  currentUsage: number;
  authentication: 'none' | 'api_key' | 'oauth' | 'jwt';
  security: {
    requiresAuth: boolean;
    permissions: string[];
    rateLimited: boolean;
    encrypted: boolean;
  };
  metrics: {
    totalRequests: number;
    successRate: number;
    avgResponseTime: number;
    errorRate: number;
  };
  lastUpdated: string;
  createdBy: string;
}

interface ApiKey {
  id: string;
  name: string;
  key: string;
  status: 'active' | 'expired' | 'revoked';
  permissions: string[];
  rateLimitPer24h: number;
  currentUsage: number;
  lastUsed: string;
  expiresAt: string;
  createdBy: string;
  createdAt: string;
}

interface ApiLog {
  id: string;
  timestamp: string;
  endpoint: string;
  method: string;
  status: number;
  responseTime: number;
  userAgent: string;
  ipAddress: string;
  apiKey?: string;
  error?: string;
}

const EvaluationApiManagementPage: React.FC = () => {
  // 状态管理
  const [endpoints, setEndpoints] = useState<ApiEndpoint[]>([]);
  const [apiKeys, setApiKeys] = useState<ApiKey[]>([]);
  const [apiLogs, setApiLogs] = useState<ApiLog[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('endpoints');

  // 模态框状态
  const [isEndpointModalVisible, setIsEndpointModalVisible] = useState(false);
  const [isApiKeyModalVisible, setIsApiKeyModalVisible] = useState(false);
  const [isLogModalVisible, setIsLogModalVisible] = useState(false);
  const [editingEndpoint, setEditingEndpoint] = useState<ApiEndpoint | null>(null);
  const [editingApiKey, setApiKey] = useState<ApiKey | null>(null);

  // 表单实例
  const [endpointForm] = Form.useForm();
  const [apiKeyForm] = Form.useForm();

  // 模拟数据加载
  useEffect(() => {
    loadMockData();
  }, []);

  const loadMockData = () => {
    const mockEndpoints: ApiEndpoint[] = [
      {
        id: '1',
        name: '模型评估接口',
        path: '/api/v1/evaluation/models',
        method: 'POST',
        description: '启动新的模型评估任务',
        status: 'active',
        category: 'evaluation',
        version: 'v1.2.0',
        rateLimitPer24h: 1000,
        currentUsage: 245,
        authentication: 'api_key',
        security: {
          requiresAuth: true,
          permissions: ['evaluation:create'],
          rateLimited: true,
          encrypted: true,
        },
        metrics: {
          totalRequests: 15420,
          successRate: 98.5,
          avgResponseTime: 1250,
          errorRate: 1.5,
        },
        lastUpdated: '2024-01-20T10:30:00',
        createdBy: 'admin',
      },
      {
        id: '2',
        name: '基准测试查询',
        path: '/api/v1/benchmarks',
        method: 'GET',
        description: '获取可用的基准测试列表',
        status: 'active',
        category: 'benchmark',
        version: 'v1.0.0',
        rateLimitPer24h: 5000,
        currentUsage: 1205,
        authentication: 'jwt',
        security: {
          requiresAuth: false,
          permissions: ['benchmark:read'],
          rateLimited: true,
          encrypted: false,
        },
        metrics: {
          totalRequests: 45230,
          successRate: 99.8,
          avgResponseTime: 150,
          errorRate: 0.2,
        },
        lastUpdated: '2024-01-18T14:15:00',
        createdBy: 'system',
      },
    ];

    const mockApiKeys: ApiKey[] = [
      {
        id: '1',
        name: '生产环境密钥',
        key: 'ak_prod_1234567890abcdef',
        status: 'active',
        permissions: ['evaluation:create', 'benchmark:read', 'report:read'],
        rateLimitPer24h: 2000,
        currentUsage: 456,
        lastUsed: '2024-01-20T09:45:00',
        expiresAt: '2024-06-20T00:00:00',
        createdBy: 'admin',
        createdAt: '2024-01-01T00:00:00',
      },
      {
        id: '2',
        name: '开发测试密钥',
        key: 'ak_dev_abcdef1234567890',
        status: 'active',
        permissions: ['benchmark:read'],
        rateLimitPer24h: 500,
        currentUsage: 89,
        lastUsed: '2024-01-19T16:30:00',
        expiresAt: '2024-12-31T00:00:00',
        createdBy: 'developer',
        createdAt: '2024-01-10T10:00:00',
      },
    ];

    const mockApiLogs: ApiLog[] = [
      {
        id: '1',
        timestamp: '2024-01-20T10:45:00',
        endpoint: '/api/v1/evaluation/models',
        method: 'POST',
        status: 200,
        responseTime: 1250,
        userAgent: 'curl/7.68.0',
        ipAddress: '192.168.1.100',
        apiKey: 'ak_prod_1234...cdef',
      },
      {
        id: '2',
        timestamp: '2024-01-20T10:44:30',
        endpoint: '/api/v1/benchmarks',
        method: 'GET',
        status: 200,
        responseTime: 145,
        userAgent: 'Mozilla/5.0',
        ipAddress: '10.0.0.50',
      },
      {
        id: '3',
        timestamp: '2024-01-20T10:44:00',
        endpoint: '/api/v1/evaluation/models',
        method: 'POST',
        status: 429,
        responseTime: 50,
        userAgent: 'python-requests/2.31.0',
        ipAddress: '172.16.0.10',
        error: 'Rate limit exceeded',
      },
    ];

    setEndpoints(mockEndpoints);
    setApiKeys(mockApiKeys);
    setApiLogs(mockApiLogs);
  };

  // API端点操作
  const handleCreateEndpoint = () => {
    setEditingEndpoint(null);
    endpointForm.resetFields();
    setIsEndpointModalVisible(true);
  };

  const handleEditEndpoint = (endpoint: ApiEndpoint) => {
    setEditingEndpoint(endpoint);
    endpointForm.setFieldsValue(endpoint);
    setIsEndpointModalVisible(true);
  };

  const handleDeleteEndpoint = (id: string) => {
    setEndpoints(endpoints.filter(ep => ep.id !== id));
    message.success('API端点已删除');
  };

  const handleToggleEndpoint = (id: string, status: ApiEndpoint['status']) => {
    setEndpoints(endpoints.map(ep => 
      ep.id === id 
        ? { ...ep, status: status === 'active' ? 'disabled' : 'active' }
        : ep
    ));
    message.success(`API端点已${status === 'active' ? '禁用' : '启用'}`);
  };

  const handleSaveEndpoint = async () => {
    try {
      const values = await endpointForm.validateFields();
      
      if (editingEndpoint) {
        // 更新现有端点
        setEndpoints(endpoints.map(ep => 
          ep.id === editingEndpoint.id 
            ? { ...ep, ...values, lastUpdated: new Date().toISOString() }
            : ep
        ));
        message.success('API端点已更新');
      } else {
        // 创建新端点
        const newEndpoint: ApiEndpoint = {
          ...values,
          id: Date.now().toString(),
          rateLimitPer24h: values.rateLimitPer24h || 1000,
          currentUsage: 0,
          security: {
            requiresAuth: values.authentication !== 'none',
            permissions: values.permissions || [],
            rateLimited: true,
            encrypted: values.authentication !== 'none',
          },
          metrics: {
            totalRequests: 0,
            successRate: 100,
            avgResponseTime: 0,
            errorRate: 0,
          },
          lastUpdated: new Date().toISOString(),
          createdBy: 'current_user',
        };
        setEndpoints([...endpoints, newEndpoint]);
        message.success('API端点已创建');
      }
      
      setIsEndpointModalVisible(false);
      endpointForm.resetFields();
    } catch (error) {
      message.error('保存失败，请检查输入');
    }
  };

  // API密钥操作
  const handleCreateApiKey = () => {
    setApiKey(null);
    apiKeyForm.resetFields();
    setIsApiKeyModalVisible(true);
  };

  const handleRevokeApiKey = (id: string) => {
    setApiKeys(apiKeys.map(key => 
      key.id === id 
        ? { ...key, status: 'revoked' }
        : key
    ));
    message.success('API密钥已撤销');
  };

  const handleSaveApiKey = async () => {
    try {
      const values = await apiKeyForm.validateFields();
      
      // 生成新的API密钥
      const newApiKey: ApiKey = {
        ...values,
        id: Date.now().toString(),
        key: `ak_${values.name.toLowerCase().replace(/\s+/g, '_')}_${Math.random().toString(36).substr(2, 16)}`,
        status: 'active',
        currentUsage: 0,
        lastUsed: '',
        createdBy: 'current_user',
        createdAt: new Date().toISOString(),
      };
      
      setApiKeys([...apiKeys, newApiKey]);
      message.success('API密钥已创建');
      
      setIsApiKeyModalVisible(false);
      apiKeyForm.resetFields();
    } catch (error) {
      message.error('创建失败，请检查输入');
    }
  };

  // 表格列定义
  const endpointColumns: ColumnsType<ApiEndpoint> = [
    {
      title: '端点名称',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: ApiEndpoint) => (
        <div>
          <div style={{ fontWeight: 500 }}>{text}</div>
          <div style={{ fontSize: '12px', color: '#666' }}>
            {record.method} {record.path}
          </div>
        </div>
      ),
    },
    {
      title: '分类',
      dataIndex: 'category',
      key: 'category',
      render: (category: string) => {
        const colors = {
          evaluation: 'blue',
          model: 'green',
          benchmark: 'orange',
          report: 'purple',
          admin: 'red',
        };
        return <Tag color={colors[category as keyof typeof colors]}>{category}</Tag>;
      },
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string, record: ApiEndpoint) => {
        const colors = {
          active: 'green',
          disabled: 'orange',
          deprecated: 'red',
        };
        return (
          <div>
            <Tag color={colors[status as keyof typeof colors]}>{status}</Tag>
            <div style={{ fontSize: '10px', color: '#999', marginTop: '2px' }}>
              {record.currentUsage}/{record.rateLimitPer24h} 次/日
            </div>
          </div>
        );
      },
    },
    {
      title: '性能指标',
      key: 'metrics',
      render: (record: ApiEndpoint) => (
        <div>
          <div>成功率: {record.metrics.successRate}%</div>
          <div>响应时间: {record.metrics.avgResponseTime}ms</div>
        </div>
      ),
    },
    {
      title: '认证方式',
      dataIndex: 'authentication',
      key: 'authentication',
      render: (auth: string) => {
        const icons = {
          none: null,
          api_key: <KeyOutlined />,
          oauth: <SafetyCertificateOutlined />,
          jwt: <SafetyCertificateOutlined />,
        };
        return (
          <div>
            {icons[auth as keyof typeof icons]} {auth}
          </div>
        );
      },
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: ApiEndpoint) => (
        <Space>
          <Tooltip title="查看详情">
            <Button type="text" icon={<EyeOutlined />} size="small" />
          </Tooltip>
          <Tooltip title="编辑">
            <Button 
              type="text" 
              icon={<EditOutlined />} 
              size="small"
              onClick={() => handleEditEndpoint(record)}
            />
          </Tooltip>
          <Tooltip title={record.status === 'active' ? '禁用' : '启用'}>
            <Button 
              type="text" 
              icon={record.status === 'active' ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
              size="small"
              onClick={() => handleToggleEndpoint(record.id, record.status)}
            />
          </Tooltip>
          <Popconfirm
            title="确定要删除这个API端点吗？"
            onConfirm={() => handleDeleteEndpoint(record.id)}
          >
            <Tooltip title="删除">
              <Button type="text" icon={<DeleteOutlined />} size="small" danger />
            </Tooltip>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  const apiKeyColumns: ColumnsType<ApiKey> = [
    {
      title: '密钥名称',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: ApiKey) => (
        <div>
          <div style={{ fontWeight: 500 }}>{text}</div>
          <div style={{ fontSize: '12px', color: '#666', fontFamily: 'monospace' }}>
            {record.key}
          </div>
        </div>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const colors = {
          active: 'green',
          expired: 'orange',
          revoked: 'red',
        };
        return <Tag color={colors[status as keyof typeof colors]}>{status}</Tag>;
      },
    },
    {
      title: '使用情况',
      key: 'usage',
      render: (record: ApiKey) => (
        <div>
          <Progress 
            percent={(record.currentUsage / record.rateLimitPer24h) * 100}
            size="small"
            showInfo={false}
          />
          <div style={{ fontSize: '12px', marginTop: '4px' }}>
            {record.currentUsage}/{record.rateLimitPer24h} 次/日
          </div>
        </div>
      ),
    },
    {
      title: '权限',
      dataIndex: 'permissions',
      key: 'permissions',
      render: (permissions: string[]) => (
        <div>
          {permissions.map(perm => (
            <Tag key={perm} size="small">{perm}</Tag>
          ))}
        </div>
      ),
    },
    {
      title: '到期时间',
      dataIndex: 'expiresAt',
      key: 'expiresAt',
      render: (date: string) => dayjs(date).format('YYYY-MM-DD'),
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: ApiKey) => (
        <Space>
          <Tooltip title="查看详情">
            <Button type="text" icon={<EyeOutlined />} size="small" />
          </Tooltip>
          {record.status === 'active' && (
            <Popconfirm
              title="确定要撤销这个API密钥吗？"
              onConfirm={() => handleRevokeApiKey(record.id)}
            >
              <Tooltip title="撤销">
                <Button type="text" icon={<DeleteOutlined />} size="small" danger />
              </Tooltip>
            </Popconfirm>
          )}
        </Space>
      ),
    },
  ];

  const logColumns: ColumnsType<ApiLog> = [
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (date: string) => dayjs(date).format('MM-DD HH:mm:ss'),
    },
    {
      title: '端点',
      key: 'endpoint',
      render: (record: ApiLog) => (
        <div>
          <Tag>{record.method}</Tag> {record.endpoint}
        </div>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: number) => {
        const color = status < 400 ? 'green' : status < 500 ? 'orange' : 'red';
        return <Tag color={color}>{status}</Tag>;
      },
    },
    {
      title: '响应时间',
      dataIndex: 'responseTime',
      key: 'responseTime',
      render: (time: number) => `${time}ms`,
    },
    {
      title: 'IP地址',
      dataIndex: 'ipAddress',
      key: 'ipAddress',
    },
    {
      title: 'API密钥',
      dataIndex: 'apiKey',
      key: 'apiKey',
      render: (key: string) => key || '无',
    },
    {
      title: '错误信息',
      dataIndex: 'error',
      key: 'error',
      render: (error: string) => error || '无',
    },
  ];

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <h1>API接口管理</h1>
        <p>管理模型评估系统的API端点、密钥和访问日志</p>
      </div>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="API端点管理" key="endpoints">
          <div style={{ marginBottom: '16px' }}>
            <Button 
              type="primary" 
              icon={<PlusOutlined />}
              onClick={handleCreateEndpoint}
            >
              新建API端点
            </Button>
          </div>
          
          <Table
            columns={endpointColumns}
            dataSource={endpoints}
            rowKey="id"
            loading={loading}
            pagination={{ pageSize: 10 }}
          />
        </TabPane>

        <TabPane tab="API密钥管理" key="apikeys">
          <div style={{ marginBottom: '16px' }}>
            <Button 
              type="primary" 
              icon={<PlusOutlined />}
              onClick={handleCreateApiKey}
            >
              创建API密钥
            </Button>
          </div>
          
          <Table
            columns={apiKeyColumns}
            dataSource={apiKeys}
            rowKey="id"
            loading={loading}
            pagination={{ pageSize: 10 }}
          />
        </TabPane>

        <TabPane tab="访问日志" key="logs">
          <Table
            columns={logColumns}
            dataSource={apiLogs}
            rowKey="id"
            loading={loading}
            pagination={{ pageSize: 20 }}
            scroll={{ x: 1200 }}
          />
        </TabPane>
      </Tabs>

      {/* API端点编辑模态框 */}
      <Modal
        title={editingEndpoint ? "编辑API端点" : "新建API端点"}
        open={isEndpointModalVisible}
        onOk={handleSaveEndpoint}
        onCancel={() => {
          setIsEndpointModalVisible(false);
          endpointForm.resetFields();
        }}
        width={800}
      >
        <Form
          form={endpointForm}
          layout="vertical"
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="name"
                label="端点名称"
                rules={[{ required: true, message: '请输入端点名称' }]}
              >
                <Input placeholder="输入端点名称" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="method"
                label="HTTP方法"
                rules={[{ required: true, message: '请选择HTTP方法' }]}
              >
                <Select placeholder="选择HTTP方法">
                  <Option value="GET">GET</Option>
                  <Option value="POST">POST</Option>
                  <Option value="PUT">PUT</Option>
                  <Option value="DELETE">DELETE</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="path"
                label="API路径"
                rules={[{ required: true, message: '请输入API路径' }]}
              >
                <Input placeholder="/api/v1/..." />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="category"
                label="分类"
                rules={[{ required: true, message: '请选择分类' }]}
              >
                <Select placeholder="选择分类">
                  <Option value="evaluation">evaluation</Option>
                  <Option value="model">model</Option>
                  <Option value="benchmark">benchmark</Option>
                  <Option value="report">report</Option>
                  <Option value="admin">admin</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            name="description"
            label="描述"
            rules={[{ required: true, message: '请输入描述' }]}
          >
            <TextArea rows={3} placeholder="输入API端点描述" />
          </Form.Item>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="authentication"
                label="认证方式"
                rules={[{ required: true, message: '请选择认证方式' }]}
              >
                <Select placeholder="选择认证方式">
                  <Option value="none">无认证</Option>
                  <Option value="api_key">API密钥</Option>
                  <Option value="oauth">OAuth</Option>
                  <Option value="jwt">JWT</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="rateLimitPer24h"
                label="每日限额"
              >
                <InputNumber 
                  style={{ width: '100%' }}
                  min={1}
                  placeholder="请求次数限制"
                />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            name="version"
            label="版本"
            rules={[{ required: true, message: '请输入版本号' }]}
          >
            <Input placeholder="v1.0.0" />
          </Form.Item>
        </Form>
      </Modal>

      {/* API密钥创建模态框 */}
      <Modal
        title="创建API密钥"
        open={isApiKeyModalVisible}
        onOk={handleSaveApiKey}
        onCancel={() => {
          setIsApiKeyModalVisible(false);
          apiKeyForm.resetFields();
        }}
      >
        <Form
          form={apiKeyForm}
          layout="vertical"
        >
          <Form.Item
            name="name"
            label="密钥名称"
            rules={[{ required: true, message: '请输入密钥名称' }]}
          >
            <Input placeholder="输入密钥名称" />
          </Form.Item>

          <Form.Item
            name="permissions"
            label="权限"
            rules={[{ required: true, message: '请选择权限' }]}
          >
            <Select
              mode="multiple"
              placeholder="选择权限"
            >
              <Option value="evaluation:create">evaluation:create</Option>
              <Option value="evaluation:read">evaluation:read</Option>
              <Option value="benchmark:read">benchmark:read</Option>
              <Option value="benchmark:create">benchmark:create</Option>
              <Option value="report:read">report:read</Option>
              <Option value="report:create">report:create</Option>
              <Option value="admin:manage">admin:manage</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="rateLimitPer24h"
            label="每日限额"
            rules={[{ required: true, message: '请输入每日限额' }]}
          >
            <InputNumber 
              style={{ width: '100%' }}
              min={1}
              placeholder="请求次数限制"
            />
          </Form.Item>

          <Form.Item
            name="expiresAt"
            label="到期时间"
            rules={[{ required: true, message: '请选择到期时间' }]}
          >
            <Input type="date" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default EvaluationApiManagementPage;