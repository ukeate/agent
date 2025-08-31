import React, { useState, useEffect } from 'react'
import { Card, Row, Col, Form, Input, Select, Switch, InputNumber, Button, Space, Typography, Divider, Alert, Table, Tag, Modal, Drawer, Tabs, Tree, Tooltip } from 'antd'
import { 
  SettingOutlined, 
  SaveOutlined, 
  ReloadOutlined, 
  ExportOutlined, 
  ImportOutlined,
  CopyOutlined,
  EditOutlined,
  DeleteOutlined,
  EyeOutlined,
  PlusOutlined,
  FileTextOutlined,
  GlobalOutlined,
  DatabaseOutlined,
  ThunderboltOutlined,
  SecurityScanOutlined,
  MonitorOutlined,
  ApiOutlined,
  CloudServerOutlined,
  BranchesOutlined,
  HistoryOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined
} from '@ant-design/icons'

const { Title, Paragraph, Text } = Typography
const { Option } = Select
const { TextArea } = Input
const { TabPane } = Tabs

interface ServiceConfigManagementPageProps {}

interface ConfigItem {
  id: string
  category: string
  key: string
  value: string | number | boolean
  type: 'string' | 'number' | 'boolean' | 'json' | 'array'
  description: string
  required: boolean
  sensitive: boolean
  defaultValue: any
  validation?: {
    min?: number
    max?: number
    pattern?: string
    options?: string[]
  }
  lastModified: string
  modifiedBy: string
}

interface ConfigTemplate {
  id: string
  name: string
  description: string
  category: string
  configs: ConfigItem[]
  version: string
  created: string
  author: string
}

interface ConfigHistory {
  id: string
  configKey: string
  oldValue: any
  newValue: any
  timestamp: string
  user: string
  reason: string
  environment: string
}

const ServiceConfigManagementPage: React.FC<ServiceConfigManagementPageProps> = () => {
  const [activeTab, setActiveTab] = useState('global')
  const [selectedCategory, setSelectedCategory] = useState('system')
  const [configModalVisible, setConfigModalVisible] = useState(false)
  const [templateDrawerVisible, setTemplateDrawerVisible] = useState(false)
  const [historyDrawerVisible, setHistoryDrawerVisible] = useState(false)
  const [editingConfig, setEditingConfig] = useState<ConfigItem | null>(null)
  const [loading, setSaving] = useState(false)

  const [form] = Form.useForm()

  const [configs, setConfigs] = useState<ConfigItem[]>([
    {
      id: 'cfg-001',
      category: 'system',
      key: 'service_discovery.etcd.endpoints',
      value: 'http://etcd1:2379,http://etcd2:2379,http://etcd3:2379',
      type: 'array',
      description: 'etcd集群端点列表',
      required: true,
      sensitive: false,
      defaultValue: 'http://localhost:2379',
      validation: { pattern: '^https?://.*:\\d+$' },
      lastModified: '2024-08-26T14:25:00Z',
      modifiedBy: 'admin'
    },
    {
      id: 'cfg-002',
      category: 'system',
      key: 'service_discovery.etcd.timeout',
      value: 5000,
      type: 'number',
      description: 'etcd连接超时时间（毫秒）',
      required: true,
      sensitive: false,
      defaultValue: 5000,
      validation: { min: 1000, max: 30000 },
      lastModified: '2024-08-26T14:20:00Z',
      modifiedBy: 'system'
    },
    {
      id: 'cfg-003',
      category: 'load_balancer',
      key: 'load_balancer.default_strategy',
      value: 'capability_based',
      type: 'string',
      description: '默认负载均衡策略',
      required: true,
      sensitive: false,
      defaultValue: 'round_robin',
      validation: { 
        options: ['round_robin', 'least_connections', 'capability_based', 'geographic', 'response_time'] 
      },
      lastModified: '2024-08-26T14:15:00Z',
      modifiedBy: 'admin'
    },
    {
      id: 'cfg-004',
      category: 'health_check',
      key: 'health_check.default_interval',
      value: 30,
      type: 'number',
      description: '默认健康检查间隔（秒）',
      required: true,
      sensitive: false,
      defaultValue: 30,
      validation: { min: 5, max: 300 },
      lastModified: '2024-08-26T14:10:00Z',
      modifiedBy: 'system'
    },
    {
      id: 'cfg-005',
      category: 'health_check',
      key: 'health_check.failure_threshold',
      value: 3,
      type: 'number',
      description: '健康检查失败阈值',
      required: true,
      sensitive: false,
      defaultValue: 3,
      validation: { min: 1, max: 10 },
      lastModified: '2024-08-26T14:05:00Z',
      modifiedBy: 'admin'
    },
    {
      id: 'cfg-006',
      category: 'security',
      key: 'security.api_key',
      value: '****-****-****-****',
      type: 'string',
      description: 'API密钥',
      required: true,
      sensitive: true,
      defaultValue: '',
      lastModified: '2024-08-26T13:45:00Z',
      modifiedBy: 'admin'
    },
    {
      id: 'cfg-007',
      category: 'monitoring',
      key: 'monitoring.metrics_enabled',
      value: true,
      type: 'boolean',
      description: '启用指标收集',
      required: false,
      sensitive: false,
      defaultValue: true,
      lastModified: '2024-08-26T13:30:00Z',
      modifiedBy: 'system'
    },
    {
      id: 'cfg-008',
      category: 'advanced',
      key: 'advanced.custom_headers',
      value: '{"X-Service": "discovery", "X-Version": "1.0"}',
      type: 'json',
      description: '自定义HTTP头',
      required: false,
      sensitive: false,
      defaultValue: '{}',
      lastModified: '2024-08-26T13:15:00Z',
      modifiedBy: 'developer'
    }
  ])

  const [configTemplates] = useState<ConfigTemplate[]>([
    {
      id: 'tpl-001',
      name: '生产环境模板',
      description: '适用于生产环境的标准配置模板',
      category: 'production',
      version: '1.0.0',
      created: '2024-08-20T10:30:00Z',
      author: 'ops-team',
      configs: [
        {
          id: 'tpl-cfg-001',
          category: 'system',
          key: 'service_discovery.etcd.timeout',
          value: 3000,
          type: 'number',
          description: '生产环境超时时间',
          required: true,
          sensitive: false,
          defaultValue: 3000,
          lastModified: '2024-08-20T10:30:00Z',
          modifiedBy: 'template'
        }
      ]
    },
    {
      id: 'tpl-002',
      name: '开发环境模板',
      description: '适用于开发环境的配置模板',
      category: 'development',
      version: '1.0.0',
      created: '2024-08-20T10:35:00Z',
      author: 'dev-team',
      configs: []
    }
  ])

  const [configHistory] = useState<ConfigHistory[]>([
    {
      id: 'hist-001',
      configKey: 'service_discovery.etcd.timeout',
      oldValue: 10000,
      newValue: 5000,
      timestamp: '2024-08-26T14:20:00Z',
      user: 'admin',
      reason: '优化响应时间',
      environment: 'production'
    },
    {
      id: 'hist-002',
      configKey: 'load_balancer.default_strategy',
      oldValue: 'round_robin',
      newValue: 'capability_based',
      timestamp: '2024-08-26T14:15:00Z',
      user: 'admin',
      reason: '改进负载均衡效果',
      environment: 'production'
    },
    {
      id: 'hist-003',
      configKey: 'health_check.failure_threshold',
      oldValue: 5,
      newValue: 3,
      timestamp: '2024-08-26T14:05:00Z',
      user: 'admin',
      reason: '提高故障检测敏感度',
      environment: 'production'
    }
  ])

  const categories = [
    { key: 'system', label: '系统配置', icon: <DatabaseOutlined /> },
    { key: 'load_balancer', label: '负载均衡', icon: <ThunderboltOutlined /> },
    { key: 'health_check', label: '健康检查', icon: <MonitorOutlined /> },
    { key: 'security', label: '安全配置', icon: <SecurityScanOutlined /> },
    { key: 'monitoring', label: '监控配置', icon: <BarChartOutlined /> },
    { key: 'advanced', label: '高级配置', icon: <SettingOutlined /> }
  ]

  const filteredConfigs = configs.filter(config => 
    selectedCategory === 'all' || config.category === selectedCategory
  )

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'string': return 'blue'
      case 'number': return 'green'
      case 'boolean': return 'orange'
      case 'json': return 'purple'
      case 'array': return 'cyan'
      default: return 'default'
    }
  }

  const handleSaveConfig = async (values: any) => {
    try {
      setSaving(true)
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      if (editingConfig) {
        // 更新配置
        setConfigs(prev => prev.map(config => 
          config.id === editingConfig.id 
            ? { 
                ...config, 
                ...values, 
                lastModified: new Date().toISOString(),
                modifiedBy: 'admin'
              }
            : config
        ))
      } else {
        // 新增配置
        const newConfig: ConfigItem = {
          id: `cfg-${Date.now()}`,
          category: values.category,
          key: values.key,
          value: values.value,
          type: values.type,
          description: values.description || '',
          required: values.required || false,
          sensitive: values.sensitive || false,
          defaultValue: values.defaultValue,
          lastModified: new Date().toISOString(),
          modifiedBy: 'admin'
        }
        setConfigs(prev => [...prev, newConfig])
      }
      
      setConfigModalVisible(false)
      setEditingConfig(null)
      form.resetFields()
      
      Modal.success({
        title: '配置保存成功',
        content: '配置已更新，将在下次服务重启时生效。'
      })
    } catch (error) {
      Modal.error({
        title: '配置保存失败',
        content: '保存配置时出现错误，请重试。'
      })
    } finally {
      setSaving(false)
    }
  }

  const handleEditConfig = (config: ConfigItem) => {
    setEditingConfig(config)
    form.setFieldsValue({
      category: config.category,
      key: config.key,
      value: config.sensitive ? '' : config.value,
      type: config.type,
      description: config.description,
      required: config.required,
      sensitive: config.sensitive,
      defaultValue: config.defaultValue
    })
    setConfigModalVisible(true)
  }

  const handleDeleteConfig = (configId: string) => {
    Modal.confirm({
      title: '确认删除',
      content: '确定要删除这个配置项吗？此操作不可撤销。',
      onOk: () => {
        setConfigs(prev => prev.filter(config => config.id !== configId))
      }
    })
  }

  const handleExportConfig = () => {
    const exportData = {
      timestamp: new Date().toISOString(),
      version: '1.0.0',
      configs: configs.map(config => ({
        ...config,
        value: config.sensitive ? '***SENSITIVE***' : config.value
      }))
    }
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `service-discovery-config-${Date.now()}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const renderConfigValue = (config: ConfigItem) => {
    if (config.sensitive) {
      return <Text code>****</Text>
    }
    
    switch (config.type) {
      case 'boolean':
        return <Switch checked={config.value as boolean} disabled />
      case 'json':
        return (
          <Text code style={{ fontSize: '12px' }}>
            {JSON.stringify(config.value).substring(0, 50)}...
          </Text>
        )
      case 'array':
        return (
          <div>
            {String(config.value).split(',').map((item, index) => (
              <Tag key={index} size="small">{item.trim()}</Tag>
            ))}
          </div>
        )
      default:
        return <Text code>{String(config.value)}</Text>
    }
  }

  const configColumns = [
    {
      title: '配置项',
      key: 'config',
      render: (_, config: ConfigItem) => (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
            <Text strong>{config.key}</Text>
            {config.required && <Tag color="red" size="small" style={{ marginLeft: 8 }}>必需</Tag>}
            {config.sensitive && <Tag color="orange" size="small" style={{ marginLeft: 4 }}>敏感</Tag>}
          </div>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {config.description}
          </Text>
        </div>
      )
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => (
        <Tag color={getTypeColor(type)}>{type.toUpperCase()}</Tag>
      )
    },
    {
      title: '当前值',
      key: 'value',
      render: (_, config: ConfigItem) => renderConfigValue(config)
    },
    {
      title: '默认值',
      key: 'defaultValue',
      render: (_, config: ConfigItem) => (
        <Text code type="secondary">{String(config.defaultValue)}</Text>
      )
    },
    {
      title: '最后修改',
      key: 'lastModified',
      render: (_, config: ConfigItem) => (
        <div>
          <Text style={{ fontSize: '12px' }}>
            {new Date(config.lastModified).toLocaleString()}
          </Text>
          <br />
          <Text type="secondary" style={{ fontSize: '11px' }}>
            by {config.modifiedBy}
          </Text>
        </div>
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, config: ConfigItem) => (
        <Space size="small">
          <Tooltip title="编辑">
            <Button size="small" icon={<EditOutlined />} onClick={() => handleEditConfig(config)} />
          </Tooltip>
          <Tooltip title="复制">
            <Button size="small" icon={<CopyOutlined />} onClick={() => {
              navigator.clipboard.writeText(`${config.key}=${config.value}`)
            }} />
          </Tooltip>
          <Tooltip title="删除">
            <Button size="small" danger icon={<DeleteOutlined />} onClick={() => handleDeleteConfig(config.id)} />
          </Tooltip>
        </Space>
      )
    }
  ]

  const historyColumns = [
    {
      title: '配置项',
      dataIndex: 'configKey',
      key: 'configKey',
      render: (key: string) => <Text code>{key}</Text>
    },
    {
      title: '旧值',
      dataIndex: 'oldValue',
      key: 'oldValue',
      render: (value: any) => <Text code type="secondary">{String(value)}</Text>
    },
    {
      title: '新值',
      dataIndex: 'newValue',
      key: 'newValue',
      render: (value: any) => <Text code>{String(value)}</Text>
    },
    {
      title: '修改原因',
      dataIndex: 'reason',
      key: 'reason'
    },
    {
      title: '操作用户',
      dataIndex: 'user',
      key: 'user'
    },
    {
      title: '修改时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (timestamp: string) => new Date(timestamp).toLocaleString()
    }
  ]

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
        {/* 页面标题 */}
        <div style={{ marginBottom: '24px' }}>
          <Title level={2}>
            <SettingOutlined /> 系统配置管理
          </Title>
          <Paragraph>
            管理服务发现系统的所有配置项，包括系统参数、负载均衡、健康检查和安全设置。
          </Paragraph>
        </div>

        {/* 操作栏 */}
        <Card style={{ marginBottom: '16px' }}>
          <Row justify="space-between" align="middle">
            <Col>
              <Space>
                <Button type="primary" icon={<PlusOutlined />} onClick={() => {
                  setEditingConfig(null)
                  form.resetFields()
                  setConfigModalVisible(true)
                }}>
                  添加配置
                </Button>
                <Button icon={<ImportOutlined />}>
                  导入配置
                </Button>
                <Button icon={<ExportOutlined />} onClick={handleExportConfig}>
                  导出配置
                </Button>
                <Button icon={<FileTextOutlined />} onClick={() => setTemplateDrawerVisible(true)}>
                  配置模板
                </Button>
                <Button icon={<HistoryOutlined />} onClick={() => setHistoryDrawerVisible(true)}>
                  修改历史
                </Button>
              </Space>
            </Col>
            <Col>
              <Space>
                <Text>配置分类:</Text>
                <Select value={selectedCategory} onChange={setSelectedCategory} style={{ width: 150 }}>
                  <Option value="all">全部分类</Option>
                  {categories.map(cat => (
                    <Option key={cat.key} value={cat.key}>
                      {cat.icon} {cat.label}
                    </Option>
                  ))}
                </Select>
              </Space>
            </Col>
          </Row>
        </Card>

        {/* 配置分类统计 */}
        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          {categories.map(category => {
            const count = configs.filter(c => c.category === category.key).length
            return (
              <Col xs={24} sm={12} lg={4} key={category.key}>
                <Card
                  size="small"
                  hoverable
                  onClick={() => setSelectedCategory(category.key)}
                  style={{ 
                    border: selectedCategory === category.key ? '2px solid #1890ff' : '1px solid #d9d9d9',
                    cursor: 'pointer'
                  }}
                >
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '24px', color: '#1890ff', marginBottom: '8px' }}>
                      {category.icon}
                    </div>
                    <div style={{ fontSize: '18px', fontWeight: 'bold', marginBottom: '4px' }}>
                      {count}
                    </div>
                    <Text type="secondary" style={{ fontSize: '12px' }}>
                      {category.label}
                    </Text>
                  </div>
                </Card>
              </Col>
            )
          })}
        </Row>

        {/* 配置项列表 */}
        <Card title={`配置项列表 - ${categories.find(c => c.key === selectedCategory)?.label || '全部分类'}`}>
          <Table
            columns={configColumns}
            dataSource={filteredConfigs}
            rowKey="id"
            size="small"
            pagination={{
              pageSize: 15,
              showSizeChanger: true,
              showTotal: (total, range) => `第 ${range[0]}-${range[1]} 条，共 ${total} 条配置`
            }}
          />
        </Card>

        {/* 配置编辑Modal */}
        <Modal
          title={editingConfig ? "编辑配置项" : "添加配置项"}
          visible={configModalVisible}
          onOk={form.submit}
          onCancel={() => {
            setConfigModalVisible(false)
            setEditingConfig(null)
            form.resetFields()
          }}
          width={600}
          confirmLoading={loading}
        >
          <Form
            form={form}
            layout="vertical"
            onFinish={handleSaveConfig}
          >
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item
                  name="category"
                  label="配置分类"
                  rules={[{ required: true, message: '请选择配置分类' }]}
                >
                  <Select placeholder="选择分类">
                    {categories.map(cat => (
                      <Option key={cat.key} value={cat.key}>
                        {cat.icon} {cat.label}
                      </Option>
                    ))}
                  </Select>
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item
                  name="type"
                  label="数据类型"
                  rules={[{ required: true, message: '请选择数据类型' }]}
                >
                  <Select placeholder="选择类型">
                    <Option value="string">字符串</Option>
                    <Option value="number">数字</Option>
                    <Option value="boolean">布尔值</Option>
                    <Option value="json">JSON对象</Option>
                    <Option value="array">数组</Option>
                  </Select>
                </Form.Item>
              </Col>
            </Row>
            
            <Form.Item
              name="key"
              label="配置键名"
              rules={[
                { required: true, message: '请输入配置键名' },
                { pattern: /^[a-z][a-z0-9._]*$/, message: '键名只能包含小写字母、数字、点和下划线，且以字母开头' }
              ]}
            >
              <Input placeholder="例如: service_discovery.timeout" />
            </Form.Item>

            <Form.Item
              name="value"
              label="配置值"
              rules={[{ required: true, message: '请输入配置值' }]}
            >
              <TextArea rows={3} placeholder="输入配置值" />
            </Form.Item>

            <Form.Item name="description" label="描述信息">
              <Input placeholder="配置项的用途描述" />
            </Form.Item>

            <Form.Item name="defaultValue" label="默认值">
              <Input placeholder="配置项的默认值" />
            </Form.Item>

            <Row gutter={16}>
              <Col span={12}>
                <Form.Item name="required" valuePropName="checked">
                  <Switch /> 必需配置
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item name="sensitive" valuePropName="checked">
                  <Switch /> 敏感信息
                </Form.Item>
              </Col>
            </Row>
          </Form>
        </Modal>

        {/* 配置模板Drawer */}
        <Drawer
          title="配置模板管理"
          visible={templateDrawerVisible}
          onClose={() => setTemplateDrawerVisible(false)}
          width={600}
        >
          <div>
            {configTemplates.map(template => (
              <Card key={template.id} size="small" style={{ marginBottom: '16px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
                  <div>
                    <Title level={5}>{template.name}</Title>
                    <Text type="secondary">{template.description}</Text>
                    <br />
                    <Tag color="blue">v{template.version}</Tag>
                    <Tag>{template.category}</Tag>
                    <br />
                    <Text type="secondary" style={{ fontSize: '12px' }}>
                      作者: {template.author} | 创建时间: {new Date(template.created).toLocaleDateString()}
                    </Text>
                  </div>
                  <Space>
                    <Button size="small" icon={<EyeOutlined />}>查看</Button>
                    <Button size="small" type="primary">应用</Button>
                  </Space>
                </div>
              </Card>
            ))}
          </div>
        </Drawer>

        {/* 修改历史Drawer */}
        <Drawer
          title="配置修改历史"
          visible={historyDrawerVisible}
          onClose={() => setHistoryDrawerVisible(false)}
          width={800}
        >
          <Table
            columns={historyColumns}
            dataSource={configHistory}
            rowKey="id"
            size="small"
            pagination={{
              pageSize: 10,
              showTotal: (total, range) => `第 ${range[0]}-${range[1]} 条，共 ${total} 条记录`
            }}
          />
        </Drawer>
      </div>
    </div>
  )
}

export default ServiceConfigManagementPage