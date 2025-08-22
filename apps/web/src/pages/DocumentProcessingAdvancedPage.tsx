import React, { useState, useEffect } from 'react'
import { 
  Card, 
  Row, 
  Col, 
  Button, 
  Space, 
  Table, 
  Upload,
  Progress,
  Tag,
  Statistic,
  Alert,
  Typography,
  Divider,
  Tabs,
  List,
  Timeline,
  Badge,
  Tooltip,
  Modal,
  Form,
  Input,
  Select,
  Switch,
  Result,
  Steps,
  Drawer,
  message
} from 'antd'
import { 
  FileTextOutlined,
  FilePdfOutlined,
  FileWordOutlined,
  FileExcelOutlined,
  FileImageOutlined,
  FileMarkdownOutlined,
  FileZipOutlined,
  CloudUploadOutlined,
  ScanOutlined,
  TranslationOutlined,
  SearchOutlined,
  HighlightOutlined,
  RobotOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  SyncOutlined,
  ExportOutlined,
  ImportOutlined,
  FilterOutlined,
  EditOutlined,
  DeleteOutlined
} from '@ant-design/icons'
import { 
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip as RechartsTooltip
} from 'recharts'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs
const { Step } = Steps
const { Option } = Select
const { Dragger } = Upload

// 文档类型枚举
enum DocumentType {
  PDF = 'pdf',
  WORD = 'word',
  EXCEL = 'excel',
  TEXT = 'text',
  MARKDOWN = 'markdown',
  IMAGE = 'image',
  CODE = 'code',
  JSON = 'json',
  XML = 'xml',
  HTML = 'html'
}

// 处理状态枚举
enum ProcessingStatus {
  PENDING = 'pending',
  PROCESSING = 'processing',
  COMPLETED = 'completed',
  FAILED = 'failed'
}

// 生成文档数据
const generateDocuments = () => {
  const documents = []
  const types = Object.values(DocumentType)
  const statuses = Object.values(ProcessingStatus)
  
  for (let i = 0; i < 20; i++) {
    documents.push({
      id: `doc_${i + 1}`,
      name: `文档_${i + 1}.${types[Math.floor(Math.random() * types.length)]}`,
      type: types[Math.floor(Math.random() * types.length)],
      size: Math.floor(Math.random() * 10000000) + 100000,
      uploadTime: new Date(Date.now() - Math.random() * 30 * 24 * 3600 * 1000),
      status: statuses[Math.floor(Math.random() * statuses.length)],
      pages: Math.floor(Math.random() * 100) + 1,
      extractedText: Math.random() > 0.5,
      entities: Math.floor(Math.random() * 50),
      language: ['中文', '英文', '混合'][Math.floor(Math.random() * 3)],
      confidence: Math.random() * 40 + 60
    })
  }
  
  return documents
}

// 生成处理统计
const generateProcessingStats = () => {
  const data = []
  
  for (let i = 0; i < 7; i++) {
    const date = new Date()
    date.setDate(date.getDate() - i)
    
    data.push({
      date: date.toLocaleDateString(),
      processed: Math.floor(Math.random() * 100) + 20,
      failed: Math.floor(Math.random() * 10),
      pending: Math.floor(Math.random() * 30)
    })
  }
  
  return data.reverse()
}

// 生成实体提取结果
const generateEntities = () => {
  return [
    { type: '人名', count: 45, examples: ['张三', '李四', '王五'] },
    { type: '地点', count: 32, examples: ['北京', '上海', '深圳'] },
    { type: '组织', count: 28, examples: ['公司A', '部门B', '团队C'] },
    { type: '日期', count: 23, examples: ['2024-01-01', '上周', '明天'] },
    { type: '金额', count: 18, examples: ['100元', '$500', '€200'] },
    { type: '产品', count: 15, examples: ['产品X', '服务Y', '方案Z'] }
  ]
}

const DocumentProcessingAdvancedPage: React.FC = () => {
  const [documents, setDocuments] = useState(() => generateDocuments())
  const [processingStats] = useState(() => generateProcessingStats())
  const [entities] = useState(() => generateEntities())
  const [selectedDoc, setSelectedDoc] = useState<any>(null)
  const [drawerVisible, setDrawerVisible] = useState(false)
  const [processingStep, setProcessingStep] = useState(0)
  const [isProcessing, setIsProcessing] = useState(false)

  // 获取文档类型图标
  const getDocumentIcon = (type: DocumentType) => {
    const icons = {
      [DocumentType.PDF]: <FilePdfOutlined style={{ color: '#ff4d4f' }} />,
      [DocumentType.WORD]: <FileWordOutlined style={{ color: '#1890ff' }} />,
      [DocumentType.EXCEL]: <FileExcelOutlined style={{ color: '#52c41a' }} />,
      [DocumentType.TEXT]: <FileTextOutlined style={{ color: '#666' }} />,
      [DocumentType.MARKDOWN]: <FileMarkdownOutlined style={{ color: '#722ed1' }} />,
      [DocumentType.IMAGE]: <FileImageOutlined style={{ color: '#fa8c16' }} />,
      [DocumentType.CODE]: <FileTextOutlined style={{ color: '#13c2c2' }} />,
      [DocumentType.JSON]: <FileTextOutlined style={{ color: '#eb2f96' }} />,
      [DocumentType.XML]: <FileTextOutlined style={{ color: '#a0d911' }} />,
      [DocumentType.HTML]: <FileTextOutlined style={{ color: '#fa541c' }} />
    }
    return icons[type] || <FileTextOutlined />
  }

  // 获取状态颜色
  const getStatusColor = (status: ProcessingStatus): string => {
    const colors = {
      [ProcessingStatus.PENDING]: 'default',
      [ProcessingStatus.PROCESSING]: 'processing',
      [ProcessingStatus.COMPLETED]: 'success',
      [ProcessingStatus.FAILED]: 'error'
    }
    return colors[status]
  }

  // 格式化文件大小
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  // 计算统计数据
  const stats = {
    totalDocuments: documents.length,
    processedDocuments: documents.filter(d => d.status === ProcessingStatus.COMPLETED).length,
    pendingDocuments: documents.filter(d => d.status === ProcessingStatus.PENDING).length,
    totalEntities: documents.reduce((sum, d) => sum + d.entities, 0),
    averageConfidence: documents.reduce((sum, d) => sum + d.confidence, 0) / documents.length
  }

  // 统计卡片
  const StatsCards = () => (
    <Row gutter={16}>
      <Col span={6}>
        <Card>
          <Statistic
            title="文档总数"
            value={stats.totalDocuments}
            prefix={<FileTextOutlined />}
          />
        </Card>
      </Col>
      <Col span={6}>
        <Card>
          <Statistic
            title="已处理"
            value={stats.processedDocuments}
            prefix={<CheckCircleOutlined />}
            valueStyle={{ color: '#52c41a' }}
          />
        </Card>
      </Col>
      <Col span={6}>
        <Card>
          <Statistic
            title="待处理"
            value={stats.pendingDocuments}
            prefix={<ClockCircleOutlined />}
            valueStyle={{ color: '#faad14' }}
          />
        </Card>
      </Col>
      <Col span={6}>
        <Card>
          <Statistic
            title="提取实体"
            value={stats.totalEntities}
            prefix={<HighlightOutlined />}
            valueStyle={{ color: '#1890ff' }}
          />
        </Card>
      </Col>
    </Row>
  )

  // 文档表格
  const DocumentsTable = () => {
    const columns = [
      {
        title: '文档名称',
        dataIndex: 'name',
        key: 'name',
        render: (name: string, record: any) => (
          <Space>
            {getDocumentIcon(record.type)}
            <Text>{name}</Text>
          </Space>
        )
      },
      {
        title: '类型',
        dataIndex: 'type',
        key: 'type',
        render: (type: DocumentType) => (
          <Tag>{type.toUpperCase()}</Tag>
        )
      },
      {
        title: '大小',
        dataIndex: 'size',
        key: 'size',
        render: (size: number) => formatFileSize(size)
      },
      {
        title: '状态',
        dataIndex: 'status',
        key: 'status',
        render: (status: ProcessingStatus) => (
          <Badge status={getStatusColor(status)} text={status} />
        )
      },
      {
        title: '页数',
        dataIndex: 'pages',
        key: 'pages'
      },
      {
        title: '语言',
        dataIndex: 'language',
        key: 'language',
        render: (lang: string) => (
          <Tag color={lang === '中文' ? 'red' : lang === '英文' ? 'blue' : 'purple'}>
            {lang}
          </Tag>
        )
      },
      {
        title: '置信度',
        dataIndex: 'confidence',
        key: 'confidence',
        render: (confidence: number) => (
          <Progress 
            percent={confidence} 
            size="small" 
            format={percent => `${percent?.toFixed(1)}%`}
            strokeColor={confidence > 80 ? '#52c41a' : confidence > 60 ? '#faad14' : '#ff4d4f'}
          />
        )
      },
      {
        title: '操作',
        key: 'actions',
        render: (record: any) => (
          <Space>
            <Button 
              icon={<ScanOutlined />} 
              size="small"
              onClick={() => handleProcessDocument(record)}
            >
              处理
            </Button>
            <Button 
              icon={<SearchOutlined />} 
              size="small"
              onClick={() => {
                setSelectedDoc(record)
                setDrawerVisible(true)
              }}
            >
              详情
            </Button>
          </Space>
        )
      }
    ]

    return (
      <Card title="文档列表" size="small">
        <Table
          columns={columns}
          dataSource={documents}
          rowKey="id"
          size="small"
          pagination={{ pageSize: 10 }}
        />
      </Card>
    )
  }

  // 处理文档
  const handleProcessDocument = async (doc: any) => {
    setIsProcessing(true)
    setProcessingStep(0)
    
    // 模拟处理步骤
    const steps = [
      () => setProcessingStep(1), // 文本提取
      () => setProcessingStep(2), // 实体识别
      () => setProcessingStep(3), // 语义分析
      () => setProcessingStep(4)  // 完成
    ]
    
    for (let i = 0; i < steps.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 1000))
      steps[i]()
    }
    
    setIsProcessing(false)
    message.success('文档处理完成！')
    
    // 更新文档状态
    setDocuments(docs => docs.map(d => 
      d.id === doc.id ? { ...d, status: ProcessingStatus.COMPLETED } : d
    ))
  }

  // 处理统计图表
  const ProcessingChart = () => (
    <Card title="处理趋势" size="small">
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={processingStats}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis />
          <RechartsTooltip />
          <Legend />
          <Line type="monotone" dataKey="processed" stroke="#52c41a" name="已处理" />
          <Line type="monotone" dataKey="failed" stroke="#ff4d4f" name="失败" />
          <Line type="monotone" dataKey="pending" stroke="#faad14" name="待处理" />
        </LineChart>
      </ResponsiveContainer>
    </Card>
  )

  // 文档类型分布
  const DocumentTypeChart = () => {
    const typeData = Object.values(DocumentType).map(type => ({
      name: type.toUpperCase(),
      value: documents.filter(d => d.type === type).length
    }))

    const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8']

    return (
      <Card title="文档类型分布" size="small">
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={typeData}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              outerRadius={80}
              fill="#8884d8"
              dataKey="value"
            >
              {typeData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <RechartsTooltip />
          </PieChart>
        </ResponsiveContainer>
      </Card>
    )
  }

  // 实体提取结果
  const EntityExtractionResults = () => (
    <Card title="实体提取结果" size="small">
      <List
        dataSource={entities}
        renderItem={entity => (
          <List.Item>
            <List.Item.Meta
              title={
                <Space>
                  <Tag color="blue">{entity.type}</Tag>
                  <Text>{entity.count} 个</Text>
                </Space>
              }
              description={
                <Space>
                  <Text type="secondary">示例:</Text>
                  {entity.examples.map(ex => (
                    <Tag key={ex}>{ex}</Tag>
                  ))}
                </Space>
              }
            />
          </List.Item>
        )}
      />
    </Card>
  )

  // OCR处理配置
  const OCRConfiguration = () => (
    <Card title="OCR处理配置" size="small">
      <Form layout="vertical">
        <Row gutter={16}>
          <Col span={12}>
            <Form.Item label="识别语言">
              <Select defaultValue="auto" style={{ width: '100%' }}>
                <Option value="auto">自动检测</Option>
                <Option value="zh">中文</Option>
                <Option value="en">英文</Option>
                <Option value="mixed">混合</Option>
              </Select>
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item label="输出格式">
              <Select defaultValue="text" style={{ width: '100%' }}>
                <Option value="text">纯文本</Option>
                <Option value="json">JSON</Option>
                <Option value="xml">XML</Option>
                <Option value="markdown">Markdown</Option>
              </Select>
            </Form.Item>
          </Col>
        </Row>
        <Row gutter={16}>
          <Col span={12}>
            <Form.Item label="图像预处理">
              <Space>
                <Switch defaultChecked /> 启用
                <Tag>去噪、倾斜校正、对比度增强</Tag>
              </Space>
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item label="表格识别">
              <Space>
                <Switch defaultChecked /> 启用
                <Tag>保留表格结构</Tag>
              </Space>
            </Form.Item>
          </Col>
        </Row>
      </Form>
    </Card>
  )

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <FileTextOutlined /> 智能文档处理中心
      </Title>
      <Paragraph type="secondary">
        支持多种文档格式的智能处理，包括OCR识别、实体提取、语义分析、多语言翻译等功能
      </Paragraph>
      
      <Divider />

      <Tabs defaultActiveKey="1">
        <TabPane tab="文档管理" key="1">
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            <StatsCards />

            {/* 上传区域 */}
            <Card title="文档上传" size="small">
              <Dragger
                name="file"
                multiple
                action="/api/v1/documents/upload"
                onChange={(info) => {
                  const { status } = info.file
                  if (status === 'done') {
                    message.success(`${info.file.name} 上传成功`)
                  } else if (status === 'error') {
                    message.error(`${info.file.name} 上传失败`)
                  }
                }}
              >
                <p className="ant-upload-drag-icon">
                  <CloudUploadOutlined />
                </p>
                <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
                <p className="ant-upload-hint">
                  支持PDF、Word、Excel、图片等多种格式，单个文件不超过100MB
                </p>
              </Dragger>
            </Card>

            <DocumentsTable />
          </Space>
        </TabPane>

        <TabPane tab="处理流程" key="2">
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            {/* 处理步骤 */}
            <Card title="文档处理流程" size="small">
              <Steps current={processingStep}>
                <Step title="文档上传" description="选择并上传文档" />
                <Step title="文本提取" description="OCR识别与文本提取" />
                <Step title="实体识别" description="NER实体识别" />
                <Step title="语义分析" description="语义理解与分析" />
                <Step title="处理完成" description="生成结构化数据" />
              </Steps>
              
              {isProcessing && (
                <Alert
                  message="处理中"
                  description="正在处理文档，请稍候..."
                  variant="default"
                  showIcon
                  icon={<SyncOutlined spin />}
                  style={{ marginTop: 16 }}
                />
              )}
            </Card>

            <OCRConfiguration />
          </Space>
        </TabPane>

        <TabPane tab="分析结果" key="3">
          <Row gutter={16}>
            <Col span={12}>
              <ProcessingChart />
            </Col>
            <Col span={12}>
              <DocumentTypeChart />
            </Col>
          </Row>
          
          <div style={{ marginTop: 16 }}>
            <EntityExtractionResults />
          </div>
        </TabPane>

        <TabPane tab="批量处理" key="4">
          <Card title="批量处理任务" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Alert
                message="批量处理功能"
                description="可以选择多个文档进行批量OCR识别、实体提取、翻译等操作"
                variant="default"
                showIcon
              />
              
              <Form layout="inline">
                <Form.Item label="处理类型">
                  <Select defaultValue="ocr" style={{ width: 150 }}>
                    <Option value="ocr">OCR识别</Option>
                    <Option value="ner">实体提取</Option>
                    <Option value="translate">文档翻译</Option>
                    <Option value="summary">摘要生成</Option>
                  </Select>
                </Form.Item>
                <Form.Item label="并发数">
                  <Input defaultValue="5" style={{ width: 80 }} />
                </Form.Item>
                <Form.Item>
                  <Button type="primary" icon={<RobotOutlined />}>
                    开始批量处理
                  </Button>
                </Form.Item>
              </Form>
              
              <Table
                columns={[
                  { title: '文档名', dataIndex: 'name', key: 'name' },
                  { title: '状态', dataIndex: 'status', key: 'status' },
                  { title: '进度', dataIndex: 'progress', key: 'progress',
                    render: (progress: number) => <Progress percent={progress} size="small" />
                  }
                ]}
                dataSource={documents.slice(0, 5).map(d => ({
                  ...d,
                  progress: Math.floor(Math.random() * 100)
                }))}
                size="small"
              />
            </Space>
          </Card>
        </TabPane>

        <TabPane tab="智能问答" key="5">
          <Card title="文档智能问答" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Alert
                message="基于文档的智能问答"
                description="上传文档后，可以对文档内容进行智能问答，系统会根据文档内容回答您的问题"
                variant="default"
                showIcon
              />
              
              <Form.Item label="选择文档">
                <Select placeholder="请选择已处理的文档" style={{ width: '100%' }}>
                  {documents.filter(d => d.status === ProcessingStatus.COMPLETED).map(d => (
                    <Option key={d.id} value={d.id}>{d.name}</Option>
                  ))}
                </Select>
              </Form.Item>
              
              <Form.Item label="输入问题">
                <Input.TextArea 
                  rows={4} 
                  placeholder="请输入您想询问的问题..."
                />
              </Form.Item>
              
              <Button type="primary" icon={<SearchOutlined />}>
                提问
              </Button>
              
              <Card title="问答历史" size="small" style={{ marginTop: 16 }}>
                <Timeline>
                  <Timeline.Item>
                    <Text strong>问：这份文档的主要内容是什么？</Text>
                    <br />
                    <Text>答：这份文档主要介绍了...</Text>
                  </Timeline.Item>
                  <Timeline.Item>
                    <Text strong>问：文档中提到了哪些关键日期？</Text>
                    <br />
                    <Text>答：文档中提到的关键日期包括...</Text>
                  </Timeline.Item>
                </Timeline>
              </Card>
            </Space>
          </Card>
        </TabPane>
      </Tabs>

      {/* 文档详情抽屉 */}
      <Drawer
        title="文档详情"
        placement="right"
        onClose={() => setDrawerVisible(false)}
        visible={drawerVisible}
        width={600}
      >
        {selectedDoc && (
          <div>
            <Descriptions bordered column={1}>
              <Descriptions.Item label="文档名称">{selectedDoc.name}</Descriptions.Item>
              <Descriptions.Item label="文档类型">
                <Tag>{selectedDoc.type.toUpperCase()}</Tag>
              </Descriptions.Item>
              <Descriptions.Item label="文件大小">
                {formatFileSize(selectedDoc.size)}
              </Descriptions.Item>
              <Descriptions.Item label="页数">{selectedDoc.pages}</Descriptions.Item>
              <Descriptions.Item label="语言">
                <Tag color={selectedDoc.language === '中文' ? 'red' : 'blue'}>
                  {selectedDoc.language}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="处理状态">
                <Badge status={getStatusColor(selectedDoc.status)} text={selectedDoc.status} />
              </Descriptions.Item>
              <Descriptions.Item label="置信度">
                <Progress percent={selectedDoc.confidence} />
              </Descriptions.Item>
              <Descriptions.Item label="提取实体">{selectedDoc.entities} 个</Descriptions.Item>
              <Descriptions.Item label="上传时间">
                {selectedDoc.uploadTime.toLocaleString()}
              </Descriptions.Item>
            </Descriptions>
            
            <Divider />
            
            <Title level={5}>提取的文本内容</Title>
            <Card size="small">
              <Paragraph>
                这是从文档中提取的示例文本内容...
              </Paragraph>
            </Card>
          </div>
        )}
      </Drawer>
    </div>
  )
}

export default DocumentProcessingAdvancedPage