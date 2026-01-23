import React, { useState } from 'react'
import {
  Card,
  Button,
  Input,
  Upload,
  List,
  Tag,
  Alert,
  Tabs,
  Space,
  Progress,
  Row,
  Col,
  Statistic,
  Table,
  Divider,
  Badge,
  message,
  Spin,
  Empty,
  Descriptions,
  Modal,
  Form,
  Select,
} from 'antd'
import {
  SearchOutlined,
  UploadOutlined,
  FileTextOutlined,
  DatabaseOutlined,
  CloudUploadOutlined,
  DeleteOutlined,
  ReloadOutlined,
  BookOutlined,
  FileSearchOutlined,
  FolderOpenOutlined,
} from '@ant-design/icons'
import ragService from '../services/ragService'

const { TextArea } = Input
const { TabPane } = Tabs
const { Dragger } = Upload

interface Document {
  id: string
  name: string
  size: number
  chunks: number
  status: 'indexed' | 'processing' | 'error'
  uploadedAt: string
}

interface SearchResult {
  title: string
  content: string
  relevance: number
  source: string
}

const TEXT_EXTENSIONS = new Set([
  'txt',
  'md',
  'json',
  'yaml',
  'yml',
  'py',
  'js',
  'ts',
  'tsx',
  'jsx',
  'java',
  'cpp',
  'c',
  'cs',
  'go',
  'rs',
])

const RAGEnhancedPage: React.FC = () => {
  const [query, setQuery] = useState('')
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [documents, setDocuments] = useState<Document[]>([])
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [activeTab, setActiveTab] = useState('search')
  const [selectedDocs, setSelectedDocs] = useState<string[]>([])
  const [showUploadModal, setShowUploadModal] = useState(false)
  const [form] = Form.useForm()
  const [response, setResponse] = useState('')
  const [confidence, setConfidence] = useState(0)

  // 执行RAG查询
  const handleSearch = async () => {
    if (!query.trim()) {
      message.warning('请输入查询内容')
      return
    }

    setLoading(true)
    try {
      const res = await ragService.query({
        query: query.trim(),
        search_type: 'hybrid',
        limit: 5,
        score_threshold: 0.5,
      })

      if (!res.success) {
        throw new Error(res.error || '查询失败')
      }

      const mappedResults: SearchResult[] = (res.results || []).map(item => ({
        title:
          item.metadata?.title ||
          item.metadata?.filename ||
          item.file_path?.split('/').pop() ||
          '文档片段',
        content: item.content,
        relevance: item.score || 0,
        source:
          item.file_path ||
          item.metadata?.source ||
          item.metadata?.filename ||
          'unknown',
      }))

      const topResult = mappedResults[0]
      setResponse(topResult?.content || '')
      setConfidence(topResult ? Math.min(topResult.relevance * 100, 100) : 0)
      setSearchResults(mappedResults)
      message.success('查询成功')
    } catch (error: any) {
      message.error('查询失败: ' + error.message)
      setResponse('')
      setConfidence(0)
      setSearchResults([])
    } finally {
      setLoading(false)
    }
  }

  // 上传文档
  const handleUpload = async (file: any) => {
    setUploading(true)
    try {
      const extension = file.name.split('.').pop()?.toLowerCase() || ''
      const isTextFile =
        file.type.startsWith('text/') ||
        file.type === 'application/json' ||
        file.type === 'application/x-yaml' ||
        file.type === 'application/yaml' ||
        TEXT_EXTENSIONS.has(extension)

      if (!isTextFile) {
        message.error('仅支持文本类文件')
        return false
      }

      const content = await file.text()
      if (!content.trim()) {
        message.error('文档内容为空')
        return false
      }

      const res = await ragService.addDocument({
        text: content,
        metadata: {
          filename: file.name,
          size: file.size,
          type: file.type,
        },
      })

      if (!res.success) {
        throw new Error(res.error || '文档添加失败')
      }

      const newDoc: Document = {
        id: res.document_id || file.uid,
        name: file.name,
        size: file.size,
        chunks: res.chunks || 0,
        status: 'indexed',
        uploadedAt: new Date().toISOString(),
      }

      setDocuments(prev => [...prev, newDoc])
      message.success('文档上传成功')
    } catch (error: any) {
      message.error('文档上传失败: ' + error.message)
    } finally {
      setUploading(false)
    }

    return false // 阻止默认上传
  }

  // 删除文档
  const handleDeleteDoc = async (docId: string) => {
    try {
      const res = await ragService.deleteDocument(docId)
      if (!res.success) {
        throw new Error(res.error || '文档删除失败')
      }
      setDocuments(prev => prev.filter(doc => doc.id !== docId))
      message.success('文档删除成功')
    } catch (error) {
      message.error('文档删除失败')
    }
  }

  // 文档列表列定义
  const docColumns = [
    {
      title: '文档名称',
      dataIndex: 'name',
      key: 'name',
      render: (text: string) => (
        <Space>
          <FileTextOutlined />
          {text}
        </Space>
      ),
    },
    {
      title: '大小',
      dataIndex: 'size',
      key: 'size',
      render: (size: number) => `${(size / 1024).toFixed(2)} KB`,
    },
    {
      title: '分块数',
      dataIndex: 'chunks',
      key: 'chunks',
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag
          color={
            status === 'indexed'
              ? 'success'
              : status === 'processing'
                ? 'processing'
                : 'error'
          }
        >
          {status === 'indexed'
            ? '已索引'
            : status === 'processing'
              ? '处理中'
              : '错误'}
        </Tag>
      ),
    },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: Document) => (
        <Button
          type="link"
          danger
          icon={<DeleteOutlined />}
          onClick={() => handleDeleteDoc(record.id)}
        >
          删除
        </Button>
      ),
    },
  ]

  return (
    <div style={{ padding: '24px' }}>
      <Card
        title={
          <Space>
            <DatabaseOutlined />
            <span>RAG 检索增强生成系统 - 增强版</span>
          </Space>
        }
        extra={
          <Space>
            <Button
              icon={<CloudUploadOutlined />}
              onClick={() => setShowUploadModal(true)}
            >
              上传文档
            </Button>
            <Button
              icon={<ReloadOutlined />}
              onClick={() => window.location.reload()}
            >
              刷新
            </Button>
          </Space>
        }
      >
        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane tab="智能搜索" key="search">
            <Row gutter={16}>
              <Col span={24}>
                <Card size="small" style={{ marginBottom: 16 }}>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <TextArea
                      rows={3}
                      value={query}
                      onChange={e => setQuery(e.target.value)}
                      placeholder="输入您的问题，系统将基于知识库为您提供答案..."
                      onPressEnter={e => {
                        if (e.shiftKey) return
                        e.preventDefault()
                        handleSearch()
                      }}
                    />

                    <Row justify="space-between" align="middle">
                      <Col>
                        <Select
                          mode="multiple"
                          style={{ width: 300 }}
                          placeholder="选择特定文档（可选）"
                          value={selectedDocs}
                          onChange={setSelectedDocs}
                          options={documents.map(doc => ({
                            label: doc.name,
                            value: doc.id,
                          }))}
                        />
                      </Col>
                      <Col>
                        <Button
                          type="primary"
                          icon={<SearchOutlined />}
                          loading={loading}
                          onClick={handleSearch}
                          size="large"
                        >
                          智能搜索
                        </Button>
                      </Col>
                    </Row>
                  </Space>
                </Card>

                {loading ? (
                  <Card>
                    <Spin tip="正在搜索..." />
                  </Card>
                ) : response ? (
                  <>
                    <Card title="AI 回答" style={{ marginBottom: 16 }}>
                      <div style={{ whiteSpace: 'pre-wrap' }}>{response}</div>
                      <Divider />
                      <Row>
                        <Col span={12}>
                          <Statistic
                            title="置信度"
                            value={confidence}
                            precision={1}
                            suffix="%"
                            valueStyle={{
                              color: confidence > 80 ? '#52c41a' : '#faad14',
                            }}
                          />
                        </Col>
                        <Col span={12}>
                          <Statistic
                            title="参考文档"
                            value={searchResults.length}
                            suffix="篇"
                          />
                        </Col>
                      </Row>
                    </Card>

                    <Card title="参考来源">
                      <List
                        dataSource={searchResults}
                        renderItem={item => (
                          <List.Item>
                            <List.Item.Meta
                              avatar={<FileSearchOutlined />}
                              title={
                                <Space>
                                  {item.title}
                                  <Tag color="blue">
                                    {(item.relevance * 100).toFixed(1)}%
                                  </Tag>
                                </Space>
                              }
                              description={item.content}
                            />
                            <div>{item.source}</div>
                          </List.Item>
                        )}
                      />
                    </Card>
                  </>
                ) : (
                  <Empty description="输入问题开始搜索" />
                )}
              </Col>
            </Row>
          </TabPane>

          <TabPane tab="知识库管理" key="documents">
            <Card
              title="文档列表"
              extra={
                <Space>
                  <Statistic title="总文档" value={documents.length} />
                  <Statistic
                    title="总分块"
                    value={documents.reduce((sum, doc) => sum + doc.chunks, 0)}
                  />
                </Space>
              }
            >
              <Table
                dataSource={documents}
                columns={docColumns}
                rowKey="id"
                pagination={{ pageSize: 10 }}
              />
            </Card>
          </TabPane>

          <TabPane tab="系统状态" key="status">
            <Row gutter={16}>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="文档总数"
                    value={documents.length}
                    prefix={<FolderOpenOutlined />}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="向量总数"
                    value={documents.reduce((sum, doc) => sum + doc.chunks, 0)}
                    prefix={<DatabaseOutlined />}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="查询次数"
                    value={searchResults.length > 0 ? 1 : 0}
                    prefix={<SearchOutlined />}
                  />
                </Card>
              </Col>
              <Col span={6}>
                <Card>
                  <Statistic
                    title="平均相关度"
                    value={
                      searchResults.length > 0
                        ? (searchResults.reduce(
                            (sum, r) => sum + r.relevance,
                            0
                          ) /
                            searchResults.length) *
                          100
                        : 0
                    }
                    precision={1}
                    suffix="%"
                  />
                </Card>
              </Col>
            </Row>

            <Card title="系统信息" style={{ marginTop: 16 }}>
              <Descriptions column={2}>
                <Descriptions.Item label="向量数据库">Qdrant</Descriptions.Item>
                <Descriptions.Item label="嵌入模型">
                  text-embedding-ada-002
                </Descriptions.Item>
                <Descriptions.Item label="LLM模型">
                  Claude 3.5
                </Descriptions.Item>
                <Descriptions.Item label="检索算法">
                  余弦相似度
                </Descriptions.Item>
                <Descriptions.Item label="分块策略">滑动窗口</Descriptions.Item>
                <Descriptions.Item label="最大分块">
                  1000 tokens
                </Descriptions.Item>
              </Descriptions>
            </Card>
          </TabPane>
        </Tabs>
      </Card>

      <Modal
        title="上传知识库文档"
        visible={showUploadModal}
        onCancel={() => setShowUploadModal(false)}
        footer={null}
        width={600}
      >
        <Dragger
          beforeUpload={handleUpload}
          multiple
          showUploadList={false}
          disabled={uploading}
        >
          <p className="ant-upload-drag-icon">
            <UploadOutlined />
          </p>
          <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
          <p className="ant-upload-hint">
            支持 PDF、TXT、MD、DOCX 等文档格式，文件将被自动分块并建立索引
          </p>
        </Dragger>

        {uploading && (
          <div style={{ textAlign: 'center', marginTop: 16 }}>
            <Spin tip="正在处理文档..." />
          </div>
        )}
      </Modal>
    </div>
  )
}

export default RAGEnhancedPage
