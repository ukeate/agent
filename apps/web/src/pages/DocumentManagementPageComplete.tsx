import React, { useState, useEffect } from 'react'
import { logger } from '../utils/logger'
import {
  Card,
  Tabs,
  Upload,
  Button,
  Table,
  Tag,
  Space,
  Modal,
  Form,
  Input,
  Select,
  Switch,
  message,
  Progress,
  Row,
  Col,
  Statistic,
  List,
  Typography,
  Tooltip,
  Popconfirm,
  InputNumber,
  Divider,
  Badge,
  Timeline,
  Alert,
} from 'antd'
import {
  InboxOutlined,
  FileTextOutlined,
  UploadOutlined,
  EyeOutlined,
  EditOutlined,
  DeleteOutlined,
  TagsOutlined,
  BranchesOutlined,
  HistoryOutlined,
  RollbackOutlined,
  DownloadOutlined,
  ReloadOutlined,
  SettingOutlined,
  FileImageOutlined,
  FileMarkdownOutlined,
  FilePdfOutlined,
  FileWordOutlined,
  FileExcelOutlined,
  CodeOutlined,
} from '@ant-design/icons'
import { documentsService } from '../services/documentsService'

const { TabPane } = Tabs
const { Dragger } = Upload
const { TextArea } = Input
const { Option } = Select
const { Title, Text, Paragraph } = Typography

interface DocumentInfo {
  doc_id: string
  title: string
  file_type: string
  file_size?: number
  created_at: string
  processing_info?: {
    chunks?: Array<{
      chunk_id: string
      content: string
      type: string
      index: number
    }>
    auto_tags?: Array<{
      tag: string
      category: string
      confidence: number
    }>
    total_chunks?: number
  }
  version?: {
    version_id: string
    version_number: number
  }
}

interface DocumentVersion {
  version_id: string
  version_number: number
  created_at: string
  change_summary: string
  is_current: boolean
}

interface DocumentRelationship {
  source: string
  target: string
  type: string
  confidence: number
}

const DocumentManagementPageComplete: React.FC = () => {
  const [documents, setDocuments] = useState<DocumentInfo[]>([])
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [selectedDocument, setSelectedDocument] = useState<DocumentInfo | null>(
    null
  )
  const [documentVersions, setDocumentVersions] = useState<DocumentVersion[]>(
    []
  )
  const [documentRelationships, setDocumentRelationships] = useState<
    DocumentRelationship[]
  >([])
  const [supportedFormats, setSupportedFormats] = useState<any>({})
  const [activeTab, setActiveTab] = useState('upload')

  // 模态框状态
  const [uploadModalVisible, setUploadModalVisible] = useState(false)
  const [detailModalVisible, setDetailModalVisible] = useState(false)
  const [versionsModalVisible, setVersionsModalVisible] = useState(false)
  const [relationshipsModalVisible, setRelationshipsModalVisible] =
    useState(false)
  const [tagsModalVisible, setTagsModalVisible] = useState(false)

  // 表单和输入状态
  const [uploadOptions, setUploadOptions] = useState({
    enableOcr: false,
    extractImages: true,
    autoTag: true,
    chunkStrategy: 'semantic',
  })
  const [tagForm] = Form.useForm()
  const [batchUploadProgress, setBatchUploadProgress] = useState(0)

  useEffect(() => {
    loadSupportedFormats()
  }, [])

  // 加载支持的文档格式
  const loadSupportedFormats = async () => {
    try {
      const formats = await documentsService.getSupportedFormats()
      setSupportedFormats(formats)
    } catch (error) {
      logger.error('加载支持格式失败:', error)
    }
  }

  // 获取文件类型图标
  const getFileIcon = (fileType: string) => {
    if (fileType.includes('pdf'))
      return <FilePdfOutlined style={{ color: '#ff4d4f' }} />
    if (fileType.includes('word') || fileType.includes('docx'))
      return <FileWordOutlined style={{ color: '#1890ff' }} />
    if (fileType.includes('excel') || fileType.includes('xlsx'))
      return <FileExcelOutlined style={{ color: '#52c41a' }} />
    if (fileType.includes('image'))
      return <FileImageOutlined style={{ color: '#722ed1' }} />
    if (fileType.includes('markdown'))
      return <FileMarkdownOutlined style={{ color: '#13c2c2' }} />
    if (
      fileType.includes('code') ||
      fileType.includes('javascript') ||
      fileType.includes('python')
    )
      return <CodeOutlined style={{ color: '#fa8c16' }} />
    return <FileTextOutlined />
  }

  // 单文档上传
  const handleSingleUpload = async (file: File) => {
    setUploading(true)
    try {
      const result = await documentsService.uploadDocument(file, uploadOptions)

      const newDoc: DocumentInfo = {
        doc_id: result.doc_id || `doc_${Date.now()}`,
        title: result.source_file || file.name,
        file_type: result.content_type || file.type,
        created_at: new Date().toISOString(),
        processing_info: {
          total_chunks: result.num_text_chunks || 0,
          chunks: result.chunks || [],
          auto_tags: result.auto_tags || [],
        },
      }

      setDocuments(prev => [...prev, newDoc])
      message.success(`文档上传成功: ${file.name}`)
    } catch (error) {
      logger.error('单文档上传失败:', error)
      message.error(`上传失败: ${file.name}`)
    } finally {
      setUploading(false)
    }
  }

  // 批量文档上传
  const handleBatchUpload = async (files: File[]) => {
    setUploading(true)
    setBatchUploadProgress(0)

    try {
      const result = await documentsService.batchUploadDocuments(files, {
        concurrentLimit: 3,
        continueOnError: true,
      })

      if (result.results) {
        const newDocs: DocumentInfo[] = result.results
          .filter((r: any) => r.success)
          .map((r: any) => ({
            doc_id: r.doc_id,
            title: r.filename,
            file_type: 'unknown',
            created_at: new Date().toISOString(),
          }))

        setDocuments(prev => [...prev, ...newDocs])
        message.success(`批量上传完成: ${result.success}/${result.total} 成功`)
      }
    } catch (error) {
      logger.error('批量上传失败:', error)
      message.error('批量上传失败')
    } finally {
      setUploading(false)
      setBatchUploadProgress(0)
    }
  }

  // 分析文档关系
  const analyzeDocumentRelationships = async (docId: string) => {
    setLoading(true)
    try {
      const relatedDocIds = documents
        .filter(doc => doc.doc_id !== docId)
        .slice(0, 5)
        .map(doc => doc.doc_id)

      const result = await documentsService.analyzeDocumentRelationships(
        docId,
        relatedDocIds
      )
      setDocumentRelationships(result.relationships || [])
      setRelationshipsModalVisible(true)
    } catch (error) {
      logger.error('关系分析失败:', error)
      message.error('关系分析失败')
    } finally {
      setLoading(false)
    }
  }

  // 生成文档标签
  const generateDocumentTags = async (docId: string, content: string) => {
    try {
      const result = await documentsService.generateDocumentTags(docId, content)
      return result.tags || []
    } catch (error) {
      logger.error('标签生成失败:', error)
      return []
    }
  }

  // 获取文档版本历史
  const getDocumentVersions = async (docId: string) => {
    setLoading(true)
    try {
      const result = await documentsService.getDocumentVersionHistory(docId)
      setDocumentVersions(result.versions || [])
      setVersionsModalVisible(true)
    } catch (error) {
      logger.error('获取版本历史失败:', error)
      message.error('获取版本历史失败')
    } finally {
      setLoading(false)
    }
  }

  // 回滚文档版本
  const rollbackDocumentVersion = async (docId: string, versionId: string) => {
    try {
      await documentsService.rollbackDocumentVersion(docId, versionId)
      message.success('版本回滚成功')
      setVersionsModalVisible(false)
    } catch (error) {
      logger.error('版本回滚失败:', error)
      message.error('版本回滚失败')
    }
  }

  // 文档表格列定义
  const documentColumns = [
    {
      title: '文档',
      dataIndex: 'title',
      key: 'title',
      render: (text: string, record: DocumentInfo) => (
        <Space>
          {getFileIcon(record.file_type)}
          <span>{text}</span>
        </Space>
      ),
    },
    {
      title: '类型',
      dataIndex: 'file_type',
      key: 'file_type',
      render: (type: string) => <Tag color="blue">{type.toUpperCase()}</Tag>,
    },
    {
      title: '分块数',
      key: 'chunks',
      render: (record: DocumentInfo) => (
        <Badge
          count={record.processing_info?.total_chunks || 0}
          color="#52c41a"
        />
      ),
    },
    {
      title: '标签数',
      key: 'tags',
      render: (record: DocumentInfo) => (
        <Badge
          count={record.processing_info?.auto_tags?.length || 0}
          color="#1890ff"
        />
      ),
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (date: string) => new Date(date).toLocaleString(),
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: DocumentInfo) => (
        <Space>
          <Tooltip title="查看详情">
            <Button
              type="text"
              icon={<EyeOutlined />}
              onClick={() => {
                setSelectedDocument(record)
                setDetailModalVisible(true)
              }}
            />
          </Tooltip>
          <Tooltip title="分析关系">
            <Button
              type="text"
              icon={<BranchesOutlined />}
              onClick={() => analyzeDocumentRelationships(record.doc_id)}
            />
          </Tooltip>
          <Tooltip title="版本历史">
            <Button
              type="text"
              icon={<HistoryOutlined />}
              onClick={() => getDocumentVersions(record.doc_id)}
            />
          </Tooltip>
          <Tooltip title="管理标签">
            <Button
              type="text"
              icon={<TagsOutlined />}
              onClick={() => {
                setSelectedDocument(record)
                setTagsModalVisible(true)
              }}
            />
          </Tooltip>
          <Popconfirm
            title="确定删除这个文档吗？"
            onConfirm={() => {
              setDocuments(prev =>
                prev.filter(doc => doc.doc_id !== record.doc_id)
              )
              message.success('文档删除成功')
            }}
          >
            <Button type="text" danger icon={<DeleteOutlined />} />
          </Popconfirm>
        </Space>
      ),
    },
  ]

  return (
    <div style={{ padding: '24px' }}>
      <Card>
        <Title level={2}>
          <FileTextOutlined /> 智能文档管理系统
        </Title>
        <Paragraph>
          支持多种格式文档的上传、处理、分析和版本管理。包含智能分块、自动标签生成、关系分析等功能。
        </Paragraph>

        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane tab="文档上传" key="upload">
            <Row gutter={24}>
              <Col span={16}>
                <Card title="单文档上传" size="small">
                  <Dragger
                    multiple={false}
                    beforeUpload={file => {
                      handleSingleUpload(file)
                      return false
                    }}
                    accept={Object.values(supportedFormats.categories || {})
                      .flat()
                      .join(',')}
                    style={{ marginBottom: 16 }}
                  >
                    <p className="ant-upload-drag-icon">
                      <InboxOutlined />
                    </p>
                    <p className="ant-upload-text">
                      点击或拖拽文件到此区域上传
                    </p>
                    <p className="ant-upload-hint">
                      支持 PDF, Word, Excel, PowerPoint, Markdown, 代码文件等
                    </p>
                  </Dragger>

                  <Divider />

                  <Card title="批量上传" size="small">
                    <Upload
                      multiple
                      beforeUpload={() => false}
                      onChange={info => {
                        if (info.fileList.length > 0) {
                          const files = info.fileList
                            .map(file => file.originFileObj)
                            .filter(Boolean) as File[]
                          if (files.length > 0) {
                            handleBatchUpload(files)
                          }
                        }
                      }}
                      showUploadList={false}
                    >
                      <Button icon={<UploadOutlined />} loading={uploading}>
                        选择多个文件批量上传
                      </Button>
                    </Upload>
                    {batchUploadProgress > 0 && (
                      <Progress
                        percent={batchUploadProgress}
                        style={{ marginTop: 16 }}
                      />
                    )}
                  </Card>
                </Card>
              </Col>

              <Col span={8}>
                <Card title="上传配置" size="small">
                  <Form layout="vertical">
                    <Form.Item label="启用OCR识别">
                      <Switch
                        checked={uploadOptions.enableOcr}
                        onChange={checked =>
                          setUploadOptions(prev => ({
                            ...prev,
                            enableOcr: checked,
                          }))
                        }
                      />
                    </Form.Item>

                    <Form.Item label="提取图像">
                      <Switch
                        checked={uploadOptions.extractImages}
                        onChange={checked =>
                          setUploadOptions(prev => ({
                            ...prev,
                            extractImages: checked,
                          }))
                        }
                      />
                    </Form.Item>

                    <Form.Item label="自动生成标签">
                      <Switch
                        checked={uploadOptions.autoTag}
                        onChange={checked =>
                          setUploadOptions(prev => ({
                            ...prev,
                            autoTag: checked,
                          }))
                        }
                      />
                    </Form.Item>

                    <Form.Item label="分块策略">
                      <Select
                        value={uploadOptions.chunkStrategy}
                        onChange={value =>
                          setUploadOptions(prev => ({
                            ...prev,
                            chunkStrategy: value,
                          }))
                        }
                      >
                        <Option value="semantic">语义分块</Option>
                        <Option value="fixed">固定大小</Option>
                        <Option value="adaptive">自适应</Option>
                        <Option value="sliding_window">滑动窗口</Option>
                        <Option value="hierarchical">分层结构</Option>
                      </Select>
                    </Form.Item>
                  </Form>
                </Card>

                <Card title="支持格式" size="small" style={{ marginTop: 16 }}>
                  {Object.entries(supportedFormats.categories || {}).map(
                    ([category, formats]: [string, any]) => (
                      <div key={category} style={{ marginBottom: 8 }}>
                        <Text strong>{category}:</Text>
                        <div>
                          {formats.map((format: string) => (
                            <Tag key={format} size="small">
                              {format}
                            </Tag>
                          ))}
                        </div>
                      </div>
                    )
                  )}
                </Card>
              </Col>
            </Row>
          </TabPane>

          <TabPane tab="文档管理" key="management">
            <Space style={{ marginBottom: 16 }}>
              <Button
                icon={<ReloadOutlined />}
                onClick={() => {
                  // 这里可以添加刷新文档列表的逻辑
                  message.info('刷新文档列表')
                }}
              >
                刷新
              </Button>
              <Button
                icon={<DownloadOutlined />}
                onClick={() => {
                  // 这里可以添加导出功能
                  message.info('导出文档列表')
                }}
              >
                导出列表
              </Button>
            </Space>

            <Table
              dataSource={documents}
              columns={documentColumns}
              rowKey="doc_id"
              loading={loading}
              pagination={{
                pageSize: 10,
                showSizeChanger: true,
                showQuickJumper: true,
                showTotal: total => `共 ${total} 个文档`,
              }}
            />
          </TabPane>

          <TabPane tab="统计分析" key="statistics">
            <Row gutter={24}>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="总文档数"
                    value={documents.length}
                    prefix={<FileTextOutlined />}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="总分块数"
                    value={documents.reduce(
                      (sum, doc) =>
                        sum + (doc.processing_info?.total_chunks || 0),
                      0
                    )}
                    prefix={<BranchesOutlined />}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="标签总数"
                    value={documents.reduce(
                      (sum, doc) =>
                        sum + (doc.processing_info?.auto_tags?.length || 0),
                      0
                    )}
                    prefix={<TagsOutlined />}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="今日上传"
                    value={
                      documents.filter(
                        doc =>
                          new Date(doc.created_at).toDateString() ===
                          new Date().toDateString()
                      ).length
                    }
                    prefix={<UploadOutlined />}
                  />
                </Card>
              </Col>
            </Row>

            <Card title="文档类型分布" style={{ marginTop: 24 }}>
              <List
                dataSource={Object.entries(
                  documents.reduce(
                    (acc, doc) => {
                      acc[doc.file_type] = (acc[doc.file_type] || 0) + 1
                      return acc
                    },
                    {} as Record<string, number>
                  )
                )}
                renderItem={([type, count]) => (
                  <List.Item>
                    <Space>
                      {getFileIcon(type)}
                      <span>{type}</span>
                      <Badge count={count} />
                    </Space>
                  </List.Item>
                )}
              />
            </Card>
          </TabPane>
        </Tabs>

        {/* 文档详情模态框 */}
        <Modal
          title="文档详情"
          open={detailModalVisible}
          onCancel={() => setDetailModalVisible(false)}
          footer={null}
          width={800}
        >
          {selectedDocument && (
            <div>
              <Space style={{ marginBottom: 16 }}>
                {getFileIcon(selectedDocument.file_type)}
                <Title level={4}>{selectedDocument.title}</Title>
              </Space>

              <Row gutter={16}>
                <Col span={12}>
                  <p>
                    <strong>文件类型:</strong> {selectedDocument.file_type}
                  </p>
                  <p>
                    <strong>创建时间:</strong>{' '}
                    {new Date(selectedDocument.created_at).toLocaleString()}
                  </p>
                </Col>
                <Col span={12}>
                  <p>
                    <strong>分块数量:</strong>{' '}
                    {selectedDocument.processing_info?.total_chunks || 0}
                  </p>
                  <p>
                    <strong>标签数量:</strong>{' '}
                    {selectedDocument.processing_info?.auto_tags?.length || 0}
                  </p>
                </Col>
              </Row>

              {selectedDocument.processing_info?.auto_tags && (
                <div style={{ marginTop: 16 }}>
                  <Title level={5}>自动标签</Title>
                  <Space wrap>
                    {selectedDocument.processing_info.auto_tags.map(
                      (tag, index) => (
                        <Tag key={index} color="blue">
                          {tag.tag} ({Math.round(tag.confidence * 100)}%)
                        </Tag>
                      )
                    )}
                  </Space>
                </div>
              )}

              {selectedDocument.processing_info?.chunks && (
                <div style={{ marginTop: 16 }}>
                  <Title level={5}>内容分块 (前5个)</Title>
                  <List
                    dataSource={selectedDocument.processing_info.chunks.slice(
                      0,
                      5
                    )}
                    renderItem={(chunk, index) => (
                      <List.Item>
                        <List.Item.Meta
                          title={`分块 ${chunk.index + 1}`}
                          description={
                            chunk.content.substring(0, 200) +
                            (chunk.content.length > 200 ? '...' : '')
                          }
                        />
                      </List.Item>
                    )}
                  />
                </div>
              )}
            </div>
          )}
        </Modal>

        {/* 版本历史模态框 */}
        <Modal
          title="版本历史"
          open={versionsModalVisible}
          onCancel={() => setVersionsModalVisible(false)}
          footer={null}
          width={600}
        >
          <Timeline>
            {documentVersions.map(version => (
              <Timeline.Item
                key={version.version_id}
                color={version.is_current ? 'green' : 'blue'}
              >
                <div>
                  <Space>
                    <strong>版本 {version.version_number}</strong>
                    {version.is_current && <Tag color="green">当前版本</Tag>}
                  </Space>
                  <p>{version.change_summary}</p>
                  <p style={{ color: '#666', fontSize: '12px' }}>
                    {new Date(version.created_at).toLocaleString()}
                  </p>
                  {!version.is_current && (
                    <Button
                      size="small"
                      icon={<RollbackOutlined />}
                      onClick={() =>
                        rollbackDocumentVersion(
                          selectedDocument?.doc_id || '',
                          version.version_id
                        )
                      }
                    >
                      回滚到此版本
                    </Button>
                  )}
                </div>
              </Timeline.Item>
            ))}
          </Timeline>
        </Modal>

        {/* 关系分析模态框 */}
        <Modal
          title="文档关系分析"
          open={relationshipsModalVisible}
          onCancel={() => setRelationshipsModalVisible(false)}
          footer={null}
          width={600}
        >
          {documentRelationships.length > 0 ? (
            <List
              dataSource={documentRelationships}
              renderItem={rel => (
                <List.Item>
                  <Space>
                    <span>{rel.source}</span>
                    <span>→</span>
                    <span>{rel.target}</span>
                    <Tag color="blue">{rel.type}</Tag>
                    <span>置信度: {Math.round(rel.confidence * 100)}%</span>
                  </Space>
                </List.Item>
              )}
            />
          ) : (
            <Alert message="未发现明显的文档关系" type="info" />
          )}
        </Modal>

        {/* 标签管理模态框 */}
        <Modal
          title="标签管理"
          open={tagsModalVisible}
          onCancel={() => setTagsModalVisible(false)}
          footer={null}
        >
          <Form form={tagForm} layout="vertical">
            <Form.Item label="生成新标签">
              <Button
                type="primary"
                onClick={async () => {
                  if (selectedDocument) {
                    const tags = await generateDocumentTags(
                      selectedDocument.doc_id,
                      '示例文档内容用于标签生成'
                    )
                    message.success(`生成了 ${tags.length} 个标签`)
                  }
                }}
              >
                智能生成标签
              </Button>
            </Form.Item>
          </Form>
        </Modal>
      </Card>
    </div>
  )
}

export default DocumentManagementPageComplete
