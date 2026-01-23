/**
 * 文档上传和处理可视化组件
 * 展示文档解析管道的技术细节
 */

import React, { useState } from 'react'
import {
  Card,
  Upload,
  Button,
  Tag,
  Space,
  Typography,
  Timeline,
  Alert,
  Divider,
  Row,
  Col,
  Statistic,
} from 'antd'
import {
  UploadOutlined,
  FileTextOutlined,
  FileImageOutlined,
  FileExcelOutlined,
  FilePdfOutlined,
  CheckCircleOutlined,
  LoadingOutlined,
  ClockCircleOutlined,
} from '@ant-design/icons'
import type { UploadFile } from 'antd/es/upload/interface'
import apiClient from '../../services/apiClient'

const { Text, Title } = Typography

interface ProcessingStep {
  step: string
  status: 'waiting' | 'processing' | 'completed' | 'error'
  duration?: number
  details?: string
}

interface DocumentMetadata {
  chunks: number
  images: number
  tables: number
  processingTimeMs: number
  embeddingDimension?: number
}

interface DocumentUploaderProps {
  onUploadSuccess?: () => void
}

const DocumentUploader: React.FC<DocumentUploaderProps> = ({
  onUploadSuccess,
}) => {
  const [fileList, setFileList] = useState<UploadFile[]>([])
  const [processing, setProcessing] = useState(false)
  const [currentFile, setCurrentFile] = useState<string | null>(null)
  const [processingSteps, setProcessingSteps] = useState<ProcessingStep[]>([])
  const [documentMetadata, setDocumentMetadata] =
    useState<DocumentMetadata | null>(null)

  const getFileIcon = (fileName: string) => {
    const ext = fileName.split('.').pop()?.toLowerCase()
    if (ext === 'pdf') return <FilePdfOutlined style={{ color: '#ff4d4f' }} />
    if (['jpg', 'jpeg', 'png', 'gif'].includes(ext || ''))
      return <FileImageOutlined style={{ color: '#722ed1' }} />
    if (['xlsx', 'xls', 'csv'].includes(ext || ''))
      return <FileExcelOutlined style={{ color: '#52c41a' }} />
    return <FileTextOutlined style={{ color: '#1890ff' }} />
  }

  const processFile = async (file: File) => {
    setProcessing(true)
    setCurrentFile(file.name)

    const steps: ProcessingStep[] = [
      { step: '文件验证', status: 'waiting' },
      { step: '上传并处理', status: 'waiting' },
      { step: '读取向量库状态', status: 'waiting' },
    ]
    setProcessingSteps(steps)

    try {
      const t0 = performance.now()
      steps[0].status = 'processing'
      setProcessingSteps([...steps])
      if (!file.name) {
        throw new Error('文件名不能为空')
      }
      if (file.size <= 0) {
        throw new Error('文件为空')
      }
      steps[0].status = 'completed'
      steps[0].duration = Math.round(performance.now() - t0)
      steps[0].details = `类型: ${file.type || 'unknown'}, 大小: ${(file.size / 1024).toFixed(2)}KB`
      setProcessingSteps([...steps])

      const t1 = performance.now()
      steps[1].status = 'processing'
      setProcessingSteps([...steps])
      const formData = new FormData()
      formData.append('file', file)
      const uploadResp = await apiClient.post<{
        doc_id: string
        source_file: string
        content_type: string
        num_text_chunks: number
        num_images: number
        num_tables: number
        processing_time: number
      }>('/multimodal-rag/upload-document', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      steps[1].status = 'completed'
      steps[1].duration = Math.round(performance.now() - t1)
      steps[1].details = `doc_id=${uploadResp.data.doc_id}, processing_time=${Math.round(uploadResp.data.processing_time * 1000)}ms`
      setProcessingSteps([...steps])

      const t2 = performance.now()
      steps[2].status = 'processing'
      setProcessingSteps([...steps])
      const statusResp = await apiClient.get<{ embedding_dimension: number }>(
        '/multimodal-rag/status'
      )
      steps[2].status = 'completed'
      steps[2].duration = Math.round(performance.now() - t2)
      steps[2].details = `embedding_dimension=${statusResp.data.embedding_dimension}`
      setProcessingSteps([...steps])

      setDocumentMetadata({
        chunks: uploadResp.data.num_text_chunks,
        images: uploadResp.data.num_images,
        tables: uploadResp.data.num_tables,
        processingTimeMs: Math.round(uploadResp.data.processing_time * 1000),
        embeddingDimension: statusResp.data.embedding_dimension,
      })
    } catch (e: any) {
      let last = -1
      for (let i = steps.length - 1; i >= 0; i--) {
        if (steps[i].status === 'processing' || steps[i].status === 'waiting') {
          last = i
          break
        }
      }
      if (last >= 0) {
        steps[last].status = 'error'
        steps[last].details = e?.message || '处理失败'
        setProcessingSteps([...steps])
      }
      throw e
    } finally {
      setProcessing(false)
      setCurrentFile(null)
    }

    onUploadSuccess?.()
  }

  const handleUpload = async () => {
    try {
      for (const file of fileList) {
        if (file.originFileObj) {
          await processFile(file.originFileObj)
        }
      }
      setFileList([])
    } catch {}
  }

  const getStepIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />
      case 'processing':
        return <LoadingOutlined style={{ color: '#1890ff' }} />
      case 'error':
        return <ClockCircleOutlined style={{ color: '#ff4d4f' }} />
      default:
        return <ClockCircleOutlined style={{ color: '#d9d9d9' }} />
    }
  }

  return (
    <Card title={<span>文档处理管道</span>}>
      <Row gutter={16}>
        <Col span={12}>
          <Title level={5}>文档上传</Title>
          <Upload
            fileList={fileList}
            onChange={({ fileList }) => setFileList(fileList)}
            beforeUpload={() => false}
            multiple
            disabled={processing}
          >
            <Button icon={<UploadOutlined />} disabled={processing}>
              选择文档
            </Button>
          </Upload>

          <div className="mt-3">
            {fileList.map(file => (
              <Tag
                key={file.uid}
                icon={getFileIcon(file.name)}
                closable
                onClose={() => {
                  setFileList(fileList.filter(f => f.uid !== file.uid))
                }}
              >
                {file.name}
              </Tag>
            ))}
          </div>

          <Button
            type="primary"
            onClick={handleUpload}
            disabled={fileList.length === 0 || processing}
            loading={processing}
            className="mt-3"
            block
          >
            开始处理
          </Button>

          <Alert
            message="支持的格式"
            description="PDF, Word, Excel, Text, Markdown, HTML, 图片 (PNG/JPG)"
            type="info"
            className="mt-3"
          />
        </Col>

        <Col span={12}>
          <Title level={5}>处理流程可视化</Title>

          {processing && currentFile && (
            <Alert
              message={`正在处理: ${currentFile}`}
              type="info"
              showIcon
              className="mb-3"
            />
          )}

          <Timeline>
            {processingSteps.map((step, index) => (
              <Timeline.Item
                key={index}
                dot={getStepIcon(step.status)}
                color={
                  step.status === 'completed'
                    ? 'green'
                    : step.status === 'processing'
                      ? 'blue'
                      : 'gray'
                }
              >
                <Space direction="vertical" size="small">
                  <Text strong>{step.step}</Text>
                  {step.duration && (
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      耗时: {step.duration}ms
                    </Text>
                  )}
                  {step.details && (
                    <Text code style={{ fontSize: 11 }}>
                      {step.details}
                    </Text>
                  )}
                </Space>
              </Timeline.Item>
            ))}
          </Timeline>
        </Col>
      </Row>

      {documentMetadata && (
        <>
          <Divider />
          <Title level={5}>处理结果统计</Title>
          <Row gutter={16}>
            <Col span={6}>
              <Statistic
                title="文本块"
                value={documentMetadata.chunks}
                prefix={<FileTextOutlined />}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="图像"
                value={documentMetadata.images}
                prefix={<FileImageOutlined />}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="表格"
                value={documentMetadata.tables}
                prefix={<FileExcelOutlined />}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="嵌入耗时"
                value={documentMetadata.processingTimeMs}
                suffix="ms"
              />
            </Col>
          </Row>

          {documentMetadata.embeddingDimension != null && (
            <Alert
              message="向量维度"
              description={`embedding_dimension=${documentMetadata.embeddingDimension}`}
              type="success"
              className="mt-3"
            />
          )}
        </>
      )}
    </Card>
  )
}

export default DocumentUploader
