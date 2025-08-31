import React, { useState, useEffect } from 'react'
import {
  Card,
  Table,
  Button,
  Space,
  Tag,
  Upload,
  Modal,
  Form,
  Input,
  Select,
  message,
  Divider,
  Statistic,
  Row,
  Col,
  Tooltip,
  Progress
} from 'antd'
import {
  PlusOutlined,
  UploadOutlined,
  DeleteOutlined,
  EyeOutlined,
  DownloadOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  CloudUploadOutlined,
  DatabaseOutlined
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'

const { Option } = Select
const { TextArea } = Input

interface ModelInfo {
  model_id: string
  name: string
  version: string
  format: string
  framework: string
  description?: string
  tags: string[]
  model_size_mb?: number
  parameter_count?: number
  created_at: string
  updated_at: string
}

const ModelRegistryPage: React.FC = () => {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [loading, setLoading] = useState(false)
  const [uploadModalVisible, setUploadModalVisible] = useState(false)
  const [hubModalVisible, setHubModalVisible] = useState(false)
  const [detailModalVisible, setDetailModalVisible] = useState(false)
  const [selectedModel, setSelectedModel] = useState<ModelInfo | null>(null)
  const [statistics, setStatistics] = useState({
    totalModels: 0,
    totalSize: 0,
    formats: {},
    frameworks: {}
  })
  
  const [form] = Form.useForm()
  const [hubForm] = Form.useForm()

  useEffect(() => {
    loadModels()
    loadStatistics()
  }, [])

  const loadModels = async () => {
    setLoading(true)
    try {
      // 模拟API调用
      const mockModels: ModelInfo[] = [
        {
          model_id: 'model-001',
          name: 'bert-base-chinese',
          version: '1.0.0',
          format: 'huggingface',
          framework: 'transformers',
          description: 'BERT中文基础模型',
          tags: ['nlp', 'bert', 'chinese'],
          model_size_mb: 410.5,
          parameter_count: 102000000,
          created_at: '2024-01-15T10:30:00Z',
          updated_at: '2024-01-15T10:30:00Z'
        },
        {
          model_id: 'model-002',
          name: 'resnet50-classifier',
          version: '2.1.0',
          format: 'pytorch',
          framework: 'pytorch',
          description: 'ResNet50图像分类模型',
          tags: ['vision', 'classification', 'resnet'],
          model_size_mb: 97.8,
          parameter_count: 25557032,
          created_at: '2024-01-10T14:20:00Z',
          updated_at: '2024-01-12T09:15:00Z'
        },
        {
          model_id: 'model-003',
          name: 'yolo-v8-detection',
          version: '1.5.0',
          format: 'onnx',
          framework: 'onnx',
          description: 'YOLOv8目标检测模型',
          tags: ['vision', 'detection', 'yolo'],
          model_size_mb: 52.1,
          parameter_count: 11200000,
          created_at: '2024-01-08T16:45:00Z',
          updated_at: '2024-01-08T16:45:00Z'
        }
      ]
      setModels(mockModels)
    } catch (error) {
      message.error('加载模型列表失败')
    }
    setLoading(false)
  }

  const loadStatistics = async () => {
    try {
      // 模拟统计数据
      setStatistics({
        totalModels: 15,
        totalSize: 2847.3,
        formats: { pytorch: 8, huggingface: 5, onnx: 2 },
        frameworks: { pytorch: 8, transformers: 5, onnx: 2 }
      })
    } catch (error) {
      console.error('加载统计信息失败:', error)
    }
  }

  const handleUpload = async (values: any) => {
    try {
      // 模拟上传过程
      message.loading('正在上传模型...', 2)
      setTimeout(() => {
        message.success('模型上传成功')
        setUploadModalVisible(false)
        form.resetFields()
        loadModels()
      }, 2000)
    } catch (error) {
      message.error('模型上传失败')
    }
  }

  const handleHubImport = async (values: any) => {
    try {
      message.loading('正在从Hub导入模型...', 3)
      setTimeout(() => {
        message.success('模型导入成功')
        setHubModalVisible(false)
        hubForm.resetFields()
        loadModels()
      }, 3000)
    } catch (error) {
      message.error('模型导入失败')
    }
  }

  const handleDelete = (modelId: string) => {
    Modal.confirm({
      title: '确认删除',
      content: '确定要删除这个模型吗？此操作不可恢复。',
      onOk: async () => {
        try {
          message.success('模型删除成功')
          loadModels()
        } catch (error) {
          message.error('删除失败')
        }
      }
    })
  }

  const showModelDetail = (model: ModelInfo) => {
    setSelectedModel(model)
    setDetailModalVisible(true)
  }

  const validateModel = async (modelId: string) => {
    try {
      message.loading('正在验证模型...', 2)
      setTimeout(() => {
        message.success('模型验证通过')
      }, 2000)
    } catch (error) {
      message.error('模型验证失败')
    }
  }

  const formatSize = (sizeInMB: number) => {
    if (sizeInMB < 1024) {
      return `${sizeInMB.toFixed(1)} MB`
    }
    return `${(sizeInMB / 1024).toFixed(1)} GB`
  }

  const formatNumber = (num: number) => {
    if (num >= 1000000) {
      return `${(num / 1000000).toFixed(1)}M`
    }
    if (num >= 1000) {
      return `${(num / 1000).toFixed(1)}K`
    }
    return num.toString()
  }

  const columns: ColumnsType<ModelInfo> = [
    {
      title: '模型名称',
      dataIndex: 'name',
      key: 'name',
      render: (text, record) => (
        <Space direction="vertical" size={0}>
          <strong>{text}</strong>
          <span style={{ color: '#666', fontSize: '12px' }}>v{record.version}</span>
        </Space>
      )
    },
    {
      title: '格式/框架',
      key: 'format',
      render: (_, record) => (
        <Space direction="vertical" size={0}>
          <Tag color="blue">{record.format.toUpperCase()}</Tag>
          <span style={{ fontSize: '12px', color: '#666' }}>{record.framework}</span>
        </Space>
      )
    },
    {
      title: '标签',
      dataIndex: 'tags',
      key: 'tags',
      render: (tags: string[]) => (
        <Space wrap>
          {tags.map(tag => (
            <Tag key={tag} size="small">{tag}</Tag>
          ))}
        </Space>
      )
    },
    {
      title: '大小',
      dataIndex: 'model_size_mb',
      key: 'size',
      render: (size) => size ? formatSize(size) : '-'
    },
    {
      title: '参数量',
      dataIndex: 'parameter_count',
      key: 'parameters',
      render: (count) => count ? formatNumber(count) : '-'
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (date) => new Date(date).toLocaleDateString()
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Tooltip title="查看详情">
            <Button
              type="text"
              icon={<EyeOutlined />}
              onClick={() => showModelDetail(record)}
            />
          </Tooltip>
          <Tooltip title="验证模型">
            <Button
              type="text"
              icon={<CheckCircleOutlined />}
              onClick={() => validateModel(record.model_id)}
            />
          </Tooltip>
          <Tooltip title="下载模型">
            <Button
              type="text"
              icon={<DownloadOutlined />}
              onClick={() => message.info('下载功能开发中')}
            />
          </Tooltip>
          <Tooltip title="删除模型">
            <Button
              type="text"
              danger
              icon={<DeleteOutlined />}
              onClick={() => handleDelete(record.model_id)}
            />
          </Tooltip>
        </Space>
      )
    }
  ]

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <h2>模型注册表</h2>
        <p>统一管理和版本控制所有AI模型</p>
      </div>

      {/* 统计卡片 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="总模型数"
              value={statistics.totalModels}
              prefix={<DatabaseOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="总存储大小"
              value={formatSize(statistics.totalSize)}
              prefix={<CloudUploadOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <div>
              <div style={{ marginBottom: '8px' }}>模型格式分布</div>
              {Object.entries(statistics.formats).map(([format, count]) => (
                <div key={format} style={{ marginBottom: '4px' }}>
                  <Tag size="small">{format.toUpperCase()}</Tag>
                  <span style={{ marginLeft: '8px' }}>{count}</span>
                </div>
              ))}
            </div>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <div>
              <div style={{ marginBottom: '8px' }}>框架分布</div>
              {Object.entries(statistics.frameworks).map(([framework, count]) => (
                <div key={framework} style={{ marginBottom: '4px' }}>
                  <Tag size="small" color="green">{framework}</Tag>
                  <span style={{ marginLeft: '8px' }}>{count}</span>
                </div>
              ))}
            </div>
          </Card>
        </Col>
      </Row>

      <Card>
        <div style={{ marginBottom: '16px', display: 'flex', justifyContent: 'space-between' }}>
          <div>
            <Space>
              <Button
                type="primary"
                icon={<UploadOutlined />}
                onClick={() => setUploadModalVisible(true)}
              >
                上传模型
              </Button>
              <Button
                icon={<CloudUploadOutlined />}
                onClick={() => setHubModalVisible(true)}
              >
                从Hub导入
              </Button>
            </Space>
          </div>
          <Button onClick={loadModels}>刷新</Button>
        </div>

        <Table
          columns={columns}
          dataSource={models}
          rowKey="model_id"
          loading={loading}
          pagination={{
            total: models.length,
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total) => `共 ${total} 条`
          }}
        />
      </Card>

      {/* 上传模型对话框 */}
      <Modal
        title="上传模型"
        open={uploadModalVisible}
        onCancel={() => setUploadModalVisible(false)}
        footer={null}
        width={600}
      >
        <Form form={form} layout="vertical" onFinish={handleUpload}>
          <Form.Item
            name="name"
            label="模型名称"
            rules={[{ required: true, message: '请输入模型名称' }]}
          >
            <Input placeholder="例如: bert-base-chinese" />
          </Form.Item>
          
          <Form.Item
            name="version"
            label="版本号"
            rules={[{ required: true, message: '请输入版本号' }]}
          >
            <Input placeholder="例如: 1.0.0" />
          </Form.Item>

          <Form.Item
            name="format"
            label="模型格式"
            rules={[{ required: true, message: '请选择模型格式' }]}
          >
            <Select placeholder="选择模型格式">
              <Option value="pytorch">PyTorch</Option>
              <Option value="onnx">ONNX</Option>
              <Option value="huggingface">HuggingFace</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="framework"
            label="框架"
            rules={[{ required: true, message: '请输入框架名称' }]}
          >
            <Input placeholder="例如: pytorch, transformers, onnx" />
          </Form.Item>

          <Form.Item
            name="description"
            label="描述"
          >
            <TextArea rows={3} placeholder="模型描述信息" />
          </Form.Item>

          <Form.Item
            name="tags"
            label="标签"
          >
            <Select
              mode="tags"
              placeholder="添加标签"
              tokenSeparators={[',']}
            />
          </Form.Item>

          <Form.Item
            name="file"
            label="模型文件"
            rules={[{ required: true, message: '请选择模型文件' }]}
          >
            <Upload
              beforeUpload={() => false}
              maxCount={1}
            >
              <Button icon={<UploadOutlined />}>选择文件</Button>
            </Upload>
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                上传
              </Button>
              <Button onClick={() => setUploadModalVisible(false)}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* HuggingFace Hub导入对话框 */}
      <Modal
        title="从HuggingFace Hub导入模型"
        open={hubModalVisible}
        onCancel={() => setHubModalVisible(false)}
        footer={null}
        width={500}
      >
        <Form form={hubForm} layout="vertical" onFinish={handleHubImport}>
          <Form.Item
            name="hub_model_name"
            label="Hub模型名称"
            rules={[{ required: true, message: '请输入Hub模型名称' }]}
          >
            <Input placeholder="例如: bert-base-chinese" />
          </Form.Item>

          <Form.Item
            name="local_name"
            label="本地名称"
            rules={[{ required: true, message: '请输入本地名称' }]}
          >
            <Input placeholder="本地存储的模型名称" />
          </Form.Item>

          <Form.Item
            name="version"
            label="版本号"
            rules={[{ required: true, message: '请输入版本号' }]}
            initialValue="1.0.0"
          >
            <Input />
          </Form.Item>

          <Form.Item
            name="description"
            label="描述"
          >
            <TextArea rows={3} placeholder="模型描述" />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                导入
              </Button>
              <Button onClick={() => setHubModalVisible(false)}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* 模型详情对话框 */}
      <Modal
        title="模型详情"
        open={detailModalVisible}
        onCancel={() => setDetailModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setDetailModalVisible(false)}>
            关闭
          </Button>
        ]}
        width={700}
      >
        {selectedModel && (
          <div>
            <Row gutter={16}>
              <Col span={12}>
                <Card size="small" title="基本信息">
                  <p><strong>模型ID:</strong> {selectedModel.model_id}</p>
                  <p><strong>名称:</strong> {selectedModel.name}</p>
                  <p><strong>版本:</strong> {selectedModel.version}</p>
                  <p><strong>格式:</strong> <Tag color="blue">{selectedModel.format.toUpperCase()}</Tag></p>
                  <p><strong>框架:</strong> {selectedModel.framework}</p>
                </Card>
              </Col>
              <Col span={12}>
                <Card size="small" title="模型规格">
                  <p><strong>文件大小:</strong> {selectedModel.model_size_mb ? formatSize(selectedModel.model_size_mb) : '未知'}</p>
                  <p><strong>参数数量:</strong> {selectedModel.parameter_count ? formatNumber(selectedModel.parameter_count) : '未知'}</p>
                  <p><strong>创建时间:</strong> {new Date(selectedModel.created_at).toLocaleString()}</p>
                  <p><strong>更新时间:</strong> {new Date(selectedModel.updated_at).toLocaleString()}</p>
                </Card>
              </Col>
            </Row>

            <Card size="small" title="描述" style={{ marginTop: '16px' }}>
              <p>{selectedModel.description || '暂无描述'}</p>
            </Card>

            <Card size="small" title="标签" style={{ marginTop: '16px' }}>
              <Space wrap>
                {selectedModel.tags.map(tag => (
                  <Tag key={tag}>{tag}</Tag>
                ))}
              </Space>
            </Card>
          </div>
        )}
      </Modal>
    </div>
  )
}

export default ModelRegistryPage