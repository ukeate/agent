import React, { useState, useEffect } from 'react'
import {
  Card,
  Table,
  Button,
  Modal,
  Form,
  Input,
  Select,
  Space,
  Tag,
  Typography,
  Row,
  Col,
  Statistic,
  Progress,
  Alert,
  Tabs,
  Timeline,
  Divider,
  Tooltip,
  Popconfirm,
  message,
  Upload,
  Switch
} from 'antd'
import {
  FileTextOutlined,
  PlusOutlined,
  DownloadOutlined,
  EyeOutlined,
  EditOutlined,
  DeleteOutlined,
  ReloadOutlined,
  BookOutlined,
  ApiOutlined,
  SettingOutlined,
  CloudUploadOutlined,
  FilePdfOutlined,
  FileWordOutlined,
  FileMarkdownOutlined,
  SyncOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  ExclamationCircleOutlined
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import type { UploadFile } from 'antd/lib/upload/interface'

const { Title, Text, Paragraph } = Typography
const { Option } = Select
const { TextArea } = Input
const { TabPane } = Tabs

interface Documentation {
  doc_id: string
  title: string
  type: 'user_guide' | 'api_documentation' | 'deployment_guide' | 'troubleshooting' | 'technical_spec'
  format: 'markdown' | 'html' | 'pdf' | 'json'
  status: 'generating' | 'completed' | 'failed' | 'outdated'
  version: string
  content?: string
  file_path?: string
  created_at: string
  updated_at: string
  auto_update: boolean
  metadata: Record<string, any>
}

interface GenerationTask {
  task_id: string
  doc_type: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  started_at?: string
  completed_at?: string
  error_message?: string
}

interface Template {
  template_id: string
  name: string
  type: string
  description: string
  content: string
}

const DocumentationManagementPage: React.FC = () => {
  const [docs, setDocs] = useState<Documentation[]>([])
  const [tasks, setTasks] = useState<GenerationTask[]>([])
  const [templates, setTemplates] = useState<Template[]>([])
  const [loading, setLoading] = useState(false)
  const [modalVisible, setModalVisible] = useState(false)
  const [previewModalVisible, setPreviewModalVisible] = useState(false)
  const [selectedDoc, setSelectedDoc] = useState<Documentation | null>(null)
  const [form] = Form.useForm()

  useEffect(() => {
    fetchDocumentationData()
    const interval = setInterval(fetchGenerationTasks, 5000)
    return () => clearInterval(interval)
  }, [])

  const fetchDocumentationData = async () => {
    setLoading(true)
    try {
      const [docsRes, tasksRes, templatesRes] = await Promise.all([
        fetch('/api/v1/platform-integration/documentation'),
        fetch('/api/v1/platform-integration/documentation/tasks'),
        fetch('/api/v1/platform-integration/documentation/templates')
      ])

      if (docsRes.ok) {
        const data = await docsRes.json()
        setDocs(data.documents || [])
      }

      if (tasksRes.ok) {
        const data = await tasksRes.json()
        setTasks(data.tasks || [])
      }

      if (templatesRes.ok) {
        const data = await templatesRes.json()
        setTemplates(data.templates || [])
      }
    } catch (error) {
      console.error('获取文档数据失败:', error)
    } finally {
      setLoading(false)
    }
  }

  const fetchGenerationTasks = async () => {
    try {
      const response = await fetch('/api/v1/platform-integration/documentation/tasks')
      if (response.ok) {
        const data = await response.json()
        setTasks(data.tasks || [])
      }
    } catch (error) {
      console.error('获取生成任务失败:', error)
    }
  }

  const handleGenerateDoc = async (values: any) => {
    try {
      const response = await fetch('/api/v1/platform-integration/documentation/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          doc_type: values.type,
          format: values.format,
          title: values.title,
          auto_update: values.auto_update || false,
          template_id: values.template_id,
          parameters: values.parameters ? JSON.parse(values.parameters) : {}
        })
      })

      if (response.ok) {
        message.success('文档生成任务已启动')
        setModalVisible(false)
        form.resetFields()
        fetchDocumentationData()
      } else {
        const error = await response.json()
        message.error(`生成失败: ${error.detail}`)
      }
    } catch (error) {
      message.error('生成失败')
    }
  }

  const handleDownloadDoc = async (docId: string, format: string) => {
    try {
      const response = await fetch(`/api/v1/platform-integration/documentation/${docId}/download`)
      if (response.ok) {
        const blob = await response.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `document.${format}`
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        window.URL.revokeObjectURL(url)
      } else {
        message.error('下载失败')
      }
    } catch (error) {
      message.error('下载失败')
    }
  }

  const handleDeleteDoc = async (docId: string) => {
    try {
      const response = await fetch(`/api/v1/platform-integration/documentation/${docId}`, {
        method: 'DELETE'
      })

      if (response.ok) {
        message.success('文档已删除')
        fetchDocumentationData()
      } else {
        message.error('删除失败')
      }
    } catch (error) {
      message.error('删除失败')
    }
  }

  const handlePreviewDoc = async (doc: Documentation) => {
    setSelectedDoc(doc)
    if (doc.content) {
      setPreviewModalVisible(true)
    } else {
      try {
        const response = await fetch(`/api/v1/platform-integration/documentation/${doc.doc_id}/content`)
        if (response.ok) {
          const data = await response.json()
          setSelectedDoc({ ...doc, content: data.content })
          setPreviewModalVisible(true)
        } else {
          message.error('获取文档内容失败')
        }
      } catch (error) {
        message.error('获取文档内容失败')
      }
    }
  }

  const handleToggleAutoUpdate = async (docId: string, autoUpdate: boolean) => {
    try {
      const response = await fetch(`/api/v1/platform-integration/documentation/${docId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ auto_update: autoUpdate })
      })

      if (response.ok) {
        message.success(autoUpdate ? '自动更新已启用' : '自动更新已禁用')
        fetchDocumentationData()
      } else {
        message.error('操作失败')
      }
    } catch (error) {
      message.error('操作失败')
    }
  }

  const getStatusColor = (status: string) => {
    const colors = {
      generating: 'processing',
      completed: 'success',
      failed: 'error',
      outdated: 'warning'
    }
    return colors[status] || 'default'
  }

  const getStatusText = (status: string) => {
    const texts = {
      generating: '生成中',
      completed: '已完成',
      failed: '失败',
      outdated: '已过期'
    }
    return texts[status] || status
  }

  const getTypeText = (type: string) => {
    const texts = {
      user_guide: '用户指南',
      api_documentation: 'API文档',
      deployment_guide: '部署指南',
      troubleshooting: '故障排除',
      technical_spec: '技术规格'
    }
    return texts[type] || type
  }

  const getFormatIcon = (format: string) => {
    const icons = {
      markdown: <FileMarkdownOutlined />,
      html: <FileTextOutlined />,
      pdf: <FilePdfOutlined />,
      json: <ApiOutlined />
    }
    return icons[format] || <FileTextOutlined />
  }

  const docColumns: ColumnsType<Documentation> = [
    {
      title: '标题',
      dataIndex: 'title',
      key: 'title',
      render: (text, record) => (
        <Space>
          {getFormatIcon(record.format)}
          <div>
            <div>{text}</div>
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {getTypeText(record.type)} • v{record.version}
            </Text>
          </div>
        </Space>
      )
    },
    {
      title: '格式',
      dataIndex: 'format',
      key: 'format',
      render: (format) => (
        <Tag>{format.toUpperCase()}</Tag>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => (
        <Tag color={getStatusColor(status)}>
          {getStatusText(status)}
        </Tag>
      )
    },
    {
      title: '自动更新',
      dataIndex: 'auto_update',
      key: 'auto_update',
      render: (autoUpdate, record) => (
        <Switch
          size="small"
          checked={autoUpdate}
          onChange={(checked) => handleToggleAutoUpdate(record.doc_id, checked)}
        />
      )
    },
    {
      title: '更新时间',
      dataIndex: 'updated_at',
      key: 'updated_at',
      render: (time) => new Date(time).toLocaleString()
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Tooltip title="预览">
            <Button
              type="text"
              icon={<EyeOutlined />}
              onClick={() => handlePreviewDoc(record)}
            />
          </Tooltip>
          <Tooltip title="下载">
            <Button
              type="text"
              icon={<DownloadOutlined />}
              onClick={() => handleDownloadDoc(record.doc_id, record.format)}
            />
          </Tooltip>
          <Popconfirm
            title="确认删除此文档？"
            onConfirm={() => handleDeleteDoc(record.doc_id)}
          >
            <Tooltip title="删除">
              <Button type="text" danger icon={<DeleteOutlined />} />
            </Tooltip>
          </Popconfirm>
        </Space>
      )
    }
  ]

  const taskColumns: ColumnsType<GenerationTask> = [
    {
      title: '文档类型',
      dataIndex: 'doc_type',
      key: 'doc_type',
      render: (type) => getTypeText(type)
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => {
        const statusConfig = {
          pending: { color: 'default', text: '等待中', icon: <ClockCircleOutlined /> },
          running: { color: 'processing', text: '生成中', icon: <SyncOutlined spin /> },
          completed: { color: 'success', text: '已完成', icon: <CheckCircleOutlined /> },
          failed: { color: 'error', text: '失败', icon: <ExclamationCircleOutlined /> }
        }
        const config = statusConfig[status]
        return <Tag color={config.color} icon={config.icon}>{config.text}</Tag>
      }
    },
    {
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      render: (progress) => <Progress percent={progress} size="small" />
    },
    {
      title: '开始时间',
      dataIndex: 'started_at',
      key: 'started_at',
      render: (time) => time ? new Date(time).toLocaleString() : '-'
    },
    {
      title: '完成时间',
      dataIndex: 'completed_at',
      key: 'completed_at',
      render: (time) => time ? new Date(time).toLocaleString() : '-'
    }
  ]

  const summary = docs.reduce((acc, doc) => {
    acc.total++
    acc[doc.status] = (acc[doc.status] || 0) + 1
    return acc
  }, { total: 0, generating: 0, completed: 0, failed: 0, outdated: 0 } as any)

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: 24, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Title level={2}>文档管理</Title>
        <Space>
          <Button icon={<ReloadOutlined />} onClick={fetchDocumentationData}>
            刷新
          </Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setModalVisible(true)}>
            生成文档
          </Button>
        </Space>
      </div>

      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="文档总数"
              value={summary.total}
              prefix={<FileTextOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="生成中"
              value={summary.generating}
              prefix={<SyncOutlined />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="已完成"
              value={summary.completed}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="需更新"
              value={summary.outdated}
              prefix={<ExclamationCircleOutlined />}
              valueStyle={{ color: '#f5222d' }}
            />
          </Card>
        </Col>
      </Row>

      <Tabs defaultActiveKey="documents">
        <TabPane tab="文档列表" key="documents">
          <Table
            columns={docColumns}
            dataSource={docs}
            rowKey="doc_id"
            loading={loading}
            pagination={{
              pageSize: 10,
              showSizeChanger: true,
              showQuickJumper: true,
              showTotal: (total) => `共 ${total} 个文档`
            }}
          />
        </TabPane>
        <TabPane tab={`生成任务 (${tasks.filter(t => t.status === 'running').length})`} key="tasks">
          <Table
            columns={taskColumns}
            dataSource={tasks}
            rowKey="task_id"
            pagination={{ pageSize: 10 }}
            size="small"
          />
        </TabPane>
      </Tabs>

      <Modal
        title="生成新文档"
        open={modalVisible}
        onCancel={() => {
          setModalVisible(false)
          form.resetFields()
        }}
        footer={null}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleGenerateDoc}
        >
          <Form.Item
            name="title"
            label="文档标题"
            rules={[{ required: true, message: '请输入文档标题' }]}
          >
            <Input placeholder="文档标题" />
          </Form.Item>

          <Form.Item
            name="type"
            label="文档类型"
            rules={[{ required: true, message: '请选择文档类型' }]}
          >
            <Select placeholder="选择文档类型">
              <Option value="user_guide">用户指南</Option>
              <Option value="api_documentation">API文档</Option>
              <Option value="deployment_guide">部署指南</Option>
              <Option value="troubleshooting">故障排除</Option>
              <Option value="technical_spec">技术规格</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="format"
            label="输出格式"
            rules={[{ required: true, message: '请选择输出格式' }]}
          >
            <Select placeholder="选择输出格式">
              <Option value="markdown">Markdown</Option>
              <Option value="html">HTML</Option>
              <Option value="pdf">PDF</Option>
              <Option value="json">JSON</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="template_id"
            label="文档模板"
          >
            <Select placeholder="选择文档模板（可选）" allowClear>
              {templates.map(template => (
                <Option key={template.template_id} value={template.template_id}>
                  {template.name}
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item
            name="parameters"
            label="生成参数 (JSON格式)"
          >
            <TextArea
              rows={4}
              placeholder='{"include_examples": true, "detail_level": "comprehensive"}'
            />
          </Form.Item>

          <Form.Item
            name="auto_update"
            valuePropName="checked"
          >
            <Switch /> 启用自动更新
          </Form.Item>

          <Form.Item style={{ marginBottom: 0 }}>
            <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
              <Button onClick={() => {
                setModalVisible(false)
                form.resetFields()
              }}>
                取消
              </Button>
              <Button type="primary" htmlType="submit">
                生成
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title={selectedDoc?.title}
        open={previewModalVisible}
        onCancel={() => setPreviewModalVisible(false)}
        width={800}
        footer={[
          <Button key="close" onClick={() => setPreviewModalVisible(false)}>
            关闭
          </Button>,
          <Button
            key="download"
            type="primary"
            icon={<DownloadOutlined />}
            onClick={() => selectedDoc && handleDownloadDoc(selectedDoc.doc_id, selectedDoc.format)}
          >
            下载
          </Button>
        ]}
      >
        {selectedDoc && (
          <div>
            <div style={{ marginBottom: 16 }}>
              <Space>
                <Tag>{getTypeText(selectedDoc.type)}</Tag>
                <Tag>{selectedDoc.format.toUpperCase()}</Tag>
                <Tag color={getStatusColor(selectedDoc.status)}>
                  {getStatusText(selectedDoc.status)}
                </Tag>
              </Space>
            </div>
            <Divider />
            <div style={{ 
              maxHeight: 400, 
              overflowY: 'auto', 
              padding: 16, 
              background: '#f5f5f5',
              borderRadius: 4 
            }}>
              {selectedDoc.format === 'markdown' ? (
                <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'monospace' }}>
                  {selectedDoc.content || '正在加载内容...'}
                </pre>
              ) : selectedDoc.format === 'html' ? (
                <div dangerouslySetInnerHTML={{ __html: selectedDoc.content || '正在加载内容...' }} />
              ) : (
                <pre style={{ whiteSpace: 'pre-wrap' }}>
                  {selectedDoc.content || '正在加载内容...'}
                </pre>
              )}
            </div>
          </div>
        )}
      </Modal>
    </div>
  )
}

export default DocumentationManagementPage