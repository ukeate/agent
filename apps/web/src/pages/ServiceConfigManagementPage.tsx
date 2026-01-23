import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Form,
  Input,
  Select,
  Switch,
  Button,
  Space,
  Typography,
  Alert,
  Table,
  Tag,
  Modal,
  Drawer,
  Tooltip,
  Upload,
} from 'antd'
import {
  SettingOutlined,
  ExportOutlined,
  ImportOutlined,
  CopyOutlined,
  EditOutlined,
  DeleteOutlined,
  EyeOutlined,
  PlusOutlined,
  FileTextOutlined,
  DatabaseOutlined,
  BarChartOutlined,
  ThunderboltOutlined,
  SecurityScanOutlined,
  MonitorOutlined,
  HistoryOutlined,
  SearchOutlined,
} from '@ant-design/icons'

const { Title, Paragraph, Text } = Typography
const { Option } = Select
const { TextArea } = Input

interface ServiceConfigManagementPageProps {}

interface ConfigItem {
  id: string
  category: string
  key: string
  value: any
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

const SENSITIVE_PLACEHOLDER = '***SENSITIVE***'

const ServiceConfigManagementPage: React.FC<
  ServiceConfigManagementPageProps
> = () => {
  const [selectedCategory, setSelectedCategory] = useState('system')
  const [configModalVisible, setConfigModalVisible] = useState(false)
  const [templateDrawerVisible, setTemplateDrawerVisible] = useState(false)
  const [historyDrawerVisible, setHistoryDrawerVisible] = useState(false)
  const [previewTemplate, setPreviewTemplate] =
    useState<ConfigTemplate | null>(null)
  const [editingConfig, setEditingConfig] = useState<ConfigItem | null>(null)
  const [loading, setSaving] = useState(false)
  const [searchKeyword, setSearchKeyword] = useState('')

  const [form] = Form.useForm()

  const [configs, setConfigs] = useState<ConfigItem[]>([])
  const [configTemplates, setConfigTemplates] = useState<ConfigTemplate[]>([])
  const [configHistory, setConfigHistory] = useState<ConfigHistory[]>([])
  const [loadingConfigs, setLoadingConfigs] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)

  const categories = [
    { key: 'system', label: '系统配置', icon: <DatabaseOutlined /> },
    { key: 'load_balancer', label: '负载均衡', icon: <ThunderboltOutlined /> },
    { key: 'health_check', label: '健康检查', icon: <MonitorOutlined /> },
    { key: 'security', label: '安全配置', icon: <SecurityScanOutlined /> },
    { key: 'monitoring', label: '监控配置', icon: <BarChartOutlined /> },
    { key: 'advanced', label: '高级配置', icon: <SettingOutlined /> },
  ]

  const parseValue = (raw: any, type: string) => {
    if (type === 'number') {
      if (raw === '' || raw === null || raw === undefined) return null
      const num = Number(raw)
      return Number.isNaN(num) ? raw : num
    }
    if (type === 'boolean') {
      if (typeof raw === 'boolean') return raw
      if (raw === 'true') return true
      if (raw === 'false') return false
      return Boolean(raw)
    }
    if (type === 'json') {
      if (typeof raw === 'string' && raw.trim() !== '') {
        return JSON.parse(raw)
      }
      return raw ?? {}
    }
    if (type === 'array') {
      if (Array.isArray(raw)) return raw
      if (typeof raw === 'string') {
        return raw
          .split(',')
          .map(item => item.trim())
          .filter(Boolean)
      }
      return raw ?? []
    }
    return raw
  }

  const toInputValue = (value: any, type: string) => {
    if (type === 'json') {
      return typeof value === 'string'
        ? value
        : JSON.stringify(value ?? {}, null, 2)
    }
    if (type === 'array') {
      if (Array.isArray(value)) return value.join(', ')
      return value ?? ''
    }
    if (value === null || value === undefined) return ''
    return String(value)
  }

  const formatValue = (value: any) => {
    if (value === null || value === undefined) return ''
    if (typeof value === 'string') return value
    try {
      return JSON.stringify(value)
    } catch {
      return String(value)
    }
  }

  const loadAll = async () => {
    setLoadingConfigs(true)
    setLoadError(null)
    try {
      const [configsRes, templatesRes, historyRes] = await Promise.all([
        apiFetch(buildApiUrl('/api/v1/service-config/configs')),
        apiFetch(buildApiUrl('/api/v1/service-config/templates')),
        apiFetch(buildApiUrl('/api/v1/service-config/history?limit=50')),
      ])
      const configsData = await configsRes.json()
      const templatesData = await templatesRes.json()
      const historyData = await historyRes.json()
      setConfigs(configsData?.configs || [])
      setConfigTemplates(templatesData?.templates || [])
      setConfigHistory(historyData?.history || [])
    } catch (error: any) {
      setLoadError(error?.message || '加载失败')
    } finally {
      setLoadingConfigs(false)
    }
  }

  const applyConfigBatch = async (
    items: Array<Partial<ConfigItem>>,
    fallbackCategory: string
  ) => {
    if (!Array.isArray(items) || items.length === 0) {
      return { success: 0, failed: 0, skipped: 0 }
    }
    const existingByKey = new Map(configs.map(config => [config.key, config]))
    const requests: Array<Promise<Response>> = []
    let skipped = 0

    for (const item of items) {
      const key = typeof item.key === 'string' ? item.key.trim() : ''
      const type = typeof item.type === 'string' ? item.type : ''
      if (!key || !type) {
        skipped += 1
        continue
      }
      const category = item.category || fallbackCategory || 'system'
      const payload: any = { category, key, type, modifiedBy: 'admin' }
      if (typeof item.value !== 'undefined') payload.value = item.value
      if (typeof item.description !== 'undefined')
        payload.description = item.description || ''
      if (typeof item.required !== 'undefined')
        payload.required = Boolean(item.required)
      if (typeof item.sensitive !== 'undefined')
        payload.sensitive = Boolean(item.sensitive)
      if (typeof item.defaultValue !== 'undefined')
        payload.defaultValue = item.defaultValue

      const maskedValue = payload.value === SENSITIVE_PLACEHOLDER
      if (maskedValue) delete payload.value

      const existing = existingByKey.get(key)
      if (existing) {
        requests.push(
          apiFetch(buildApiUrl(`/api/v1/service-config/configs/${existing.id}`), {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
          })
        )
        continue
      }

      if (maskedValue || typeof payload.value === 'undefined') {
        skipped += 1
        continue
      }

      requests.push(
        apiFetch(buildApiUrl('/api/v1/service-config/configs'), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        })
      )
    }

    if (requests.length === 0) {
      return { success: 0, failed: 0, skipped }
    }

    const results = await Promise.allSettled(requests)
    const failed = results.filter(result => result.status === 'rejected').length
    return { success: requests.length - failed, failed, skipped }
  }

  useEffect(() => {
    loadAll()
  }, [])

  const normalizedKeyword = searchKeyword.trim().toLowerCase()
  const filteredConfigs = configs.filter(config => {
    if (selectedCategory !== 'all' && config.category !== selectedCategory)
      return false
    if (!normalizedKeyword) return true
    const target = `${config.key} ${config.description || ''}`.toLowerCase()
    return target.includes(normalizedKeyword)
  })

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'string':
        return 'blue'
      case 'number':
        return 'green'
      case 'boolean':
        return 'orange'
      case 'json':
        return 'purple'
      case 'array':
        return 'cyan'
      default:
        return 'default'
    }
  }

  const handleSaveConfig = async (values: any) => {
    try {
      setSaving(true)
      let parsedValue: any
      let parsedDefault: any
      const shouldSkipValue =
        editingConfig?.sensitive &&
        (values.value === '' || values.value === null || values.value === undefined)
      try {
        parsedDefault = parseValue(values.defaultValue, values.type)
        if (!shouldSkipValue) {
          parsedValue = parseValue(values.value, values.type)
        }
      } catch (error: any) {
        Modal.error({
          title: '配置解析失败',
          content: error?.message || '配置值解析失败',
        })
        return
      }

      const payload: any = {
        category: values.category,
        key: values.key,
        type: values.type,
        description: values.description || '',
        required: values.required || false,
        sensitive: values.sensitive || false,
        defaultValue: parsedDefault,
        modifiedBy: 'admin',
      }

      if (!shouldSkipValue) payload.value = parsedValue

      const url = buildApiUrl(
        editingConfig
          ? `/api/v1/service-config/configs/${editingConfig.id}`
          : '/api/v1/service-config/configs'
      )
      const response = await apiFetch(url, {
        method: editingConfig ? 'PUT' : 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      await response.json()
      await loadAll()

      setConfigModalVisible(false)
      setEditingConfig(null)
      form.resetFields()

      Modal.success({
        title: '配置保存成功',
        content: '配置已更新，将在下次服务重启时生效。',
      })
    } catch (error) {
      Modal.error({
        title: '配置保存失败',
        content: '保存配置时出现错误，请重试。',
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
      value: config.sensitive ? '' : toInputValue(config.value, config.type),
      type: config.type,
      description: config.description,
      required: config.required,
      sensitive: config.sensitive,
      defaultValue: toInputValue(config.defaultValue, config.type),
    })
    setConfigModalVisible(true)
  }

  const handleDeleteConfig = (configId: string) => {
    Modal.confirm({
      title: '确认删除',
      content: '确定要删除这个配置项吗？此操作不可撤销。',
      onOk: async () => {
        try {
          const response = await apiFetch(
            buildApiUrl(`/api/v1/service-config/configs/${configId}`),
            { method: 'DELETE' }
          )
          await response.json().catch(() => null)
          await loadAll()
        } catch (error) {
          Modal.error({
            title: '删除失败',
            content: '删除配置项时出现错误，请重试。',
          })
        }
      },
    })
  }

  const handleExportConfig = () => {
    const exportData = {
      timestamp: new Date().toISOString(),
      version: '1.0.0',
      configs: configs.map(config => ({
        ...config,
        value: config.sensitive ? SENSITIVE_PLACEHOLDER : config.value,
      })),
    }

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json',
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `service-discovery-config-${Date.now()}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const showBatchResult = (
    label: string,
    result: { success: number; failed: number; skipped: number }
  ) => {
    const content = `成功 ${result.success} 项，失败 ${result.failed} 项，跳过 ${result.skipped} 项`
    if (result.failed > 0) {
      Modal.warning({ title: `${label}完成（部分失败）`, content })
      return
    }
    Modal.success({ title: `${label}完成`, content })
  }

  const handleImportConfig = (file: File) => {
    const importConfigs = async () => {
      try {
        const text = await file.text()
        const data = JSON.parse(text)
        const items = Array.isArray(data?.configs)
          ? data.configs
          : Array.isArray(data)
            ? data
            : null
        if (!items) {
          throw new Error('配置文件格式不正确')
        }
        const fallbackCategory =
          selectedCategory === 'all' ? 'system' : selectedCategory
        const result = await applyConfigBatch(items, fallbackCategory)
        await loadAll()
        showBatchResult('导入', result)
      } catch (error: any) {
        Modal.error({
          title: '导入失败',
          content: error?.message || '无法解析配置文件',
        })
      }
    }
    importConfigs()
    return false
  }

  const handleApplyTemplate = (template: ConfigTemplate) => {
    if (!Array.isArray(template.configs) || template.configs.length === 0) {
      Modal.warning({ title: '模板为空', content: '模板没有可应用的配置项' })
      return
    }
    Modal.confirm({
      title: '应用配置模板',
      content: `将模板「${template.name}」应用到当前配置，存在同名配置将被覆盖。`,
      onOk: async () => {
        try {
          const result = await applyConfigBatch(
            template.configs,
            template.category || 'system'
          )
          await loadAll()
          showBatchResult('模板应用', result)
        } catch (error: any) {
          Modal.error({
            title: '模板应用失败',
            content: error?.message || '应用模板时发生错误',
          })
        }
      },
    })
  }

  const renderConfigValue = (config: ConfigItem) => {
    if (config.sensitive) {
      return <Text code>****</Text>
    }

    switch (config.type) {
      case 'boolean':
        return <Switch checked={config.value as boolean} disabled />
      case 'json':
        {
          const text = formatValue(config.value)
          const short = text.length > 60 ? `${text.slice(0, 60)}...` : text
          return (
            <Text code style={{ fontSize: '12px' }}>
              {short || '-'}
            </Text>
          )
        }
      case 'array':
        {
          const list = Array.isArray(config.value)
            ? config.value
            : String(config.value ?? '')
                .split(',')
                .map(item => item.trim())
                .filter(Boolean)
          return (
            <div>
              {list.length === 0 ? (
                <Text type="secondary">-</Text>
              ) : (
                list.map((item, index) => (
                  <Tag key={`${config.id}-${index}`} size="small">
                    {String(item)}
                  </Tag>
                ))
              )}
            </div>
          )
        }
      default:
        {
          const text = formatValue(config.value)
          return <Text code>{text || '-'}</Text>
        }
    }
  }

  const renderTemplateValue = (item: Partial<ConfigItem>) => {
    if (item.sensitive) {
      return <Text code>****</Text>
    }
    const text = formatValue(item.value ?? item.defaultValue)
    const short = text.length > 60 ? `${text.slice(0, 60)}...` : text
    return <Text code>{short || '-'}</Text>
  }

  const configColumns = [
    {
      title: '配置项',
      key: 'config',
      render: (_, config: ConfigItem) => (
        <div>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              marginBottom: '4px',
            }}
          >
            <Text strong>{config.key}</Text>
            {config.required && (
              <Tag color="red" size="small" style={{ marginLeft: 8 }}>
                必需
              </Tag>
            )}
            {config.sensitive && (
              <Tag color="orange" size="small" style={{ marginLeft: 4 }}>
                敏感
              </Tag>
            )}
          </div>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {config.description}
          </Text>
        </div>
      ),
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => (
        <Tag color={getTypeColor(type)}>{type.toUpperCase()}</Tag>
      ),
    },
    {
      title: '当前值',
      key: 'value',
      render: (_, config: ConfigItem) => renderConfigValue(config),
    },
    {
      title: '默认值',
      key: 'defaultValue',
      render: (_, config: ConfigItem) => {
        const text = formatValue(config.defaultValue)
        return (
          <Text code type="secondary">
            {text || '-'}
          </Text>
        )
      },
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
      ),
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, config: ConfigItem) => (
        <Space size="small">
          <Tooltip title="编辑">
            <Button
              size="small"
              icon={<EditOutlined />}
              onClick={() => handleEditConfig(config)}
            />
          </Tooltip>
          <Tooltip title="复制">
            <Button
              size="small"
              icon={<CopyOutlined />}
              onClick={() => {
                const value = config.sensitive
                  ? ''
                  : formatValue(config.value)
                navigator.clipboard.writeText(`${config.key}=${value}`)
              }}
            />
          </Tooltip>
          <Tooltip title="删除">
            <Button
              size="small"
              danger
              icon={<DeleteOutlined />}
              onClick={() => handleDeleteConfig(config.id)}
            />
          </Tooltip>
        </Space>
      ),
    },
  ]

  const templateColumns = [
    {
      title: '键名',
      dataIndex: 'key',
      key: 'key',
      render: (key: string) => <Text code>{key}</Text>,
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) =>
        type ? (
          <Tag color={getTypeColor(type)}>{type.toUpperCase()}</Tag>
        ) : (
          <Text type="secondary">-</Text>
        ),
    },
    {
      title: '值/默认值',
      key: 'value',
      render: (_: unknown, item: Partial<ConfigItem>) =>
        renderTemplateValue(item),
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
      render: (text: string) => text || '-',
    },
  ]

  const historyColumns = [
    {
      title: '配置项',
      dataIndex: 'configKey',
      key: 'configKey',
      render: (key: string) => <Text code>{key}</Text>,
    },
    {
      title: '旧值',
      dataIndex: 'oldValue',
      key: 'oldValue',
      render: (value: any) => (
        <Text code type="secondary">
          {formatValue(value) || '-'}
        </Text>
      ),
    },
    {
      title: '新值',
      dataIndex: 'newValue',
      key: 'newValue',
      render: (value: any) => <Text code>{formatValue(value) || '-'}</Text>,
    },
    {
      title: '修改原因',
      dataIndex: 'reason',
      key: 'reason',
    },
    {
      title: '操作用户',
      dataIndex: 'user',
      key: 'user',
    },
    {
      title: '修改时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (timestamp: string) => new Date(timestamp).toLocaleString(),
    },
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
                <Button
                  type="primary"
                  icon={<PlusOutlined />}
                  onClick={() => {
                    setEditingConfig(null)
                    form.resetFields()
                    setConfigModalVisible(true)
                  }}
                >
                  添加配置
                </Button>
                <Upload
                  accept=".json"
                  beforeUpload={handleImportConfig}
                  showUploadList={false}
                >
                  <Button icon={<ImportOutlined />}>导入配置</Button>
                </Upload>
                <Button icon={<ExportOutlined />} onClick={handleExportConfig}>
                  导出配置
                </Button>
                <Button
                  icon={<FileTextOutlined />}
                  onClick={() => setTemplateDrawerVisible(true)}
                >
                  配置模板
                </Button>
                <Button
                  icon={<HistoryOutlined />}
                  onClick={() => setHistoryDrawerVisible(true)}
                >
                  修改历史
                </Button>
              </Space>
            </Col>
            <Col>
              <Space>
                <Text>配置分类:</Text>
                <Select
                  value={selectedCategory}
                  onChange={setSelectedCategory}
                  style={{ width: 150 }}
                >
                  <Option value="all">全部分类</Option>
                  {categories.map(cat => (
                    <Option key={cat.key} value={cat.key}>
                      {cat.icon} {cat.label}
                    </Option>
                  ))}
                </Select>
                <Input
                  allowClear
                  prefix={<SearchOutlined />}
                  placeholder="搜索键名/描述"
                  value={searchKeyword}
                  onChange={e => setSearchKeyword(e.target.value)}
                  style={{ width: 220 }}
                />
              </Space>
            </Col>
          </Row>
        </Card>

        {loadError && (
          <Alert
            type="error"
            showIcon
            message="加载失败"
            description={loadError}
            style={{ marginBottom: '16px' }}
          />
        )}

        {/* 配置分类统计 */}
        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          {categories.map(category => {
            const count = configs.filter(
              c => c.category === category.key
            ).length
            return (
              <Col xs={24} sm={12} lg={4} key={category.key}>
                <Card
                  size="small"
                  hoverable
                  onClick={() => setSelectedCategory(category.key)}
                  style={{
                    border:
                      selectedCategory === category.key
                        ? '2px solid #1890ff'
                        : '1px solid #d9d9d9',
                    cursor: 'pointer',
                  }}
                >
                  <div style={{ textAlign: 'center' }}>
                    <div
                      style={{
                        fontSize: '24px',
                        color: '#1890ff',
                        marginBottom: '8px',
                      }}
                    >
                      {category.icon}
                    </div>
                    <div
                      style={{
                        fontSize: '18px',
                        fontWeight: 'bold',
                        marginBottom: '4px',
                      }}
                    >
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
        <Card
          title={`配置项列表 - ${categories.find(c => c.key === selectedCategory)?.label || '全部分类'}`}
        >
          <Table
            columns={configColumns}
            dataSource={filteredConfigs}
            rowKey="id"
            size="small"
            loading={loadingConfigs}
            pagination={{
              pageSize: 15,
              showSizeChanger: true,
              showTotal: (total, range) =>
                `第 ${range[0]}-${range[1]} 条，共 ${total} 条配置`,
            }}
          />
        </Card>

        {/* 配置编辑Modal */}
        <Modal
          title={editingConfig ? '编辑配置项' : '添加配置项'}
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
          <Form form={form} layout="vertical" onFinish={handleSaveConfig}>
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
                {
                  pattern: /^[a-z][a-z0-9._]*$/,
                  message:
                    '键名只能包含小写字母、数字、点和下划线，且以字母开头',
                },
              ]}
            >
              <Input placeholder="例如: service_discovery.timeout" />
            </Form.Item>

            <Form.Item
              name="value"
              label="配置值"
              rules={[
                {
                  validator: (_, value) => {
                    if (
                      editingConfig?.sensitive &&
                      (value === '' || value === null || value === undefined)
                    ) {
                      return Promise.resolve()
                    }
                    if (value === '' || value === null || value === undefined) {
                      return Promise.reject(new Error('请输入配置值'))
                    }
                    return Promise.resolve()
                  },
                },
              ]}
            >
              <TextArea
                rows={3}
                placeholder={
                  editingConfig?.sensitive ? '留空保持原值' : '输入配置值'
                }
              />
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
              <Card
                key={template.id}
                size="small"
                style={{ marginBottom: '16px' }}
              >
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'start',
                  }}
                >
                  <div>
                    <Title level={5}>{template.name}</Title>
                    <Text type="secondary">{template.description}</Text>
                    <br />
                    <Tag color="blue">v{template.version}</Tag>
                    <Tag>{template.category}</Tag>
                    <br />
                    <Text type="secondary" style={{ fontSize: '12px' }}>
                      作者: {template.author} | 创建时间:{' '}
                      {new Date(template.created).toLocaleDateString()}
                    </Text>
                  </div>
                  <Space>
                    <Button
                      size="small"
                      icon={<EyeOutlined />}
                      onClick={() => setPreviewTemplate(template)}
                    >
                      查看
                    </Button>
                    <Button
                      size="small"
                      type="primary"
                      onClick={() => handleApplyTemplate(template)}
                    >
                      应用
                    </Button>
                  </Space>
                </div>
              </Card>
            ))}
          </div>
        </Drawer>

        {/* 模板预览Modal */}
        <Modal
          title={
            previewTemplate
              ? `模板预览 - ${previewTemplate.name}`
              : '模板预览'
          }
          visible={Boolean(previewTemplate)}
          onCancel={() => setPreviewTemplate(null)}
          footer={null}
          width={760}
        >
          {previewTemplate && (
            <div>
              <Space size={8} style={{ marginBottom: '12px' }}>
                <Tag color="blue">v{previewTemplate.version}</Tag>
                <Tag>{previewTemplate.category}</Tag>
                <Text type="secondary">作者: {previewTemplate.author}</Text>
              </Space>
              {previewTemplate.description && (
                <Paragraph style={{ marginBottom: '12px' }}>
                  {previewTemplate.description}
                </Paragraph>
              )}
              <Table
                columns={templateColumns}
                dataSource={previewTemplate.configs || []}
                rowKey={(record, index) =>
                  record.key || `template-${index}`
                }
                size="small"
                pagination={{ pageSize: 8 }}
              />
            </div>
          )}
        </Modal>

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
            loading={loadingConfigs}
            pagination={{
              pageSize: 10,
              showTotal: (total, range) =>
                `第 ${range[0]}-${range[1]} 条，共 ${total} 条记录`,
            }}
          />
        </Drawer>
      </div>
    </div>
  )
}

export default ServiceConfigManagementPage
