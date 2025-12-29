import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Row, Col, Button, Upload, Typography, List, message, Tag, Divider, Spin } from 'antd'
import { CloudUploadOutlined, FileTextOutlined, ReloadOutlined, CheckCircleOutlined, CloseCircleOutlined } from '@ant-design/icons'

const { Title, Text } = Typography
const { Dragger } = Upload

interface DocumentVersion {
  version_id: string
  created_at: string
  change_summary?: string
}

interface ProcessedDocument {
  document_id: string
  filename: string
  file_type: string
  upload_time: string
  status: string
  processing_info?: any
  versions?: DocumentVersion[]
}

const DocumentProcessingAdvancedPage: React.FC = () => {
  const [docs, setDocs] = useState<ProcessedDocument[]>([])
  const [loading, setLoading] = useState(false)

  const fetchDocs = async () => {
    setLoading(true)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/documents/list'))
      const data = await res.json()
      const items = Array.isArray(data?.documents) ? data.documents : []
      setDocs(items.map((doc: any) => ({
        document_id: doc.doc_id,
        filename: doc.title || doc.doc_id,
        file_type: doc.file_type,
        upload_time: doc.created_at,
        status: doc.status,
        processing_info: doc.processing_info,
        versions: doc.version ? [{
          version_id: doc.version.version_id,
          created_at: doc.created_at,
          change_summary: doc.version.version_number
        }] : []
      })))
    } catch (e) {
      setDocs([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchDocs()
  }, [])

  const uploadProps = {
    name: 'file',
    multiple: false,
    action: buildApiUrl('/api/v1/documents/upload'),
    onChange(info: any) {
      const { status } = info.file
      if (status === 'done') {
        message.success(`${info.file.name} 上传并处理成功`)
        fetchDocs()
      } else if (status === 'error') {
        message.error(`${info.file.name} 上传失败`)
      }
    }
  }

  return (
    <div style={{ padding: 24 }}>
      <Row justify="space-between" align="middle" style={{ marginBottom: 16 }}>
        <Col>
          <Title level={3}>文档处理中心</Title>
          <Text type="secondary">所有数据直接来源 /api/v1/documents，无本地静态数据</Text>
        </Col>
        <Col>
          <Button icon={<ReloadOutlined />} onClick={fetchDocs} loading={loading}>
            刷新
          </Button>
        </Col>
      </Row>

      <Card style={{ marginBottom: 16 }}>
        <Dragger {...uploadProps} accept=".pdf,.doc,.docx,.txt,.md,.json,.xml,.html,.png,.jpg">
          <p className="ant-upload-drag-icon">
            <CloudUploadOutlined />
          </p>
          <p className="ant-upload-text">点击或拖拽文件到此上传并处理</p>
          <p className="ant-upload-hint">支持OCR、自动标签与分块，处理结果由后端返回</p>
        </Dragger>
      </Card>

      <Card title="已处理文档" extra={loading && <Spin size="small" />}>
        {docs.length === 0 && (
          <div style={{ textAlign: 'center', padding: 24, color: '#888' }}>
            暂无文档，请先上传
          </div>
        )}
        {docs.length > 0 && (
          <List
            dataSource={docs}
            renderItem={doc => (
              <List.Item>
                <List.Item.Meta
                  avatar={<FileTextOutlined style={{ fontSize: 24 }} />}
                  title={doc.filename}
                  description={
                    <div>
                      <Text type="secondary">类型: {doc.file_type}，上传时间: {new Date(doc.upload_time).toLocaleString()}</Text>
                      <div style={{ marginTop: 4 }}>
                        <Tag color={doc.status === 'completed' ? 'green' : 'orange'}>
                          {doc.status}
                        </Tag>
                      </div>
                    </div>
                  }
                />
                {doc.processing_info?.auto_tags && (
                  <div>
                    <Divider style={{ margin: '8px 0' }} />
                    <Text>标签:</Text>{' '}
                    {doc.processing_info.auto_tags.map((t: any) => (
                      <Tag key={t.tag}>{t.tag}</Tag>
                    ))}
                  </div>
                )}
              </List.Item>
            )}
          />
        )}
      </Card>
    </div>
  )
}

export default DocumentProcessingAdvancedPage
