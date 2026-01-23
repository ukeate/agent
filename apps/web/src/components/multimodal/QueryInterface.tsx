/**
 * 多模态查询输入界面
 * 展示查询输入、文件上传、参数配置等
 */

import React, { useState } from 'react'
import {
  Card,
  Input,
  Button,
  Upload,
  Space,
  Form,
  Slider,
  Switch,
  InputNumber,
  Tag,
  Tooltip,
  message,
} from 'antd'
import {
  SendOutlined,
  UploadOutlined,
  SettingOutlined,
  FileImageOutlined,
  FileTextOutlined,
  FileExcelOutlined,
} from '@ant-design/icons'
import type { UploadFile } from 'antd/es/upload/interface'

const { TextArea } = Input

interface QueryInterfaceProps {
  onQuery: (query: string, files?: File[]) => void
  loading?: boolean
}

const QueryInterface: React.FC<QueryInterfaceProps> = ({
  onQuery,
  loading,
}) => {
  const [form] = Form.useForm()
  const [fileList, setFileList] = useState<UploadFile[]>([])
  const [showAdvanced, setShowAdvanced] = useState(false)

  const handleSubmit = (values: any) => {
    const files = fileList.map(f => f.originFileObj as File).filter(Boolean)
    onQuery(values.query, files.length > 0 ? files : undefined)
  }

  const getFileIcon = (fileName: string) => {
    const ext = fileName.split('.').pop()?.toLowerCase()
    if (['jpg', 'jpeg', 'png', 'gif'].includes(ext || '')) {
      return <FileImageOutlined />
    }
    if (['xlsx', 'xls', 'csv'].includes(ext || '')) {
      return <FileExcelOutlined />
    }
    return <FileTextOutlined />
  }

  const beforeUpload = (file: File) => {
    // 验证文件类型
    const supportedTypes = [
      'application/pdf',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'application/msword',
      'text/plain',
      'text/markdown',
      'text/html',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'application/vnd.ms-excel',
      'text/csv',
      'image/png',
      'image/jpeg',
      'image/jpg',
    ]

    if (!supportedTypes.includes(file.type)) {
      message.error('不支持的文件类型')
      return false
    }

    return false // 阻止自动上传
  }

  return (
    <Card
      title="多模态查询输入"
      extra={
        <Button
          icon={<SettingOutlined />}
          type="text"
          onClick={() => setShowAdvanced(!showAdvanced)}
        >
          高级选项
        </Button>
      }
    >
      <Form
        form={form}
        layout="vertical"
        onFinish={handleSubmit}
        initialValues={{
          query: '',
          temperature: 0.7,
          maxTokens: 1000,
          topK: 10,
          includeImages: true,
          includeTables: true,
          stream: false,
        }}
      >
        <Form.Item
          name="query"
          label="查询内容"
          rules={[{ required: true, message: '请输入查询内容' }]}
        >
          <TextArea
            rows={4}
            placeholder="输入您的查询，例如：分析文档中的销售数据趋势..."
            disabled={loading}
          />
        </Form.Item>

        <Form.Item label="上传文件（可选）">
          <Upload
            fileList={fileList}
            onChange={({ fileList }) => setFileList(fileList)}
            beforeUpload={beforeUpload}
            multiple
            disabled={loading}
          >
            <Button icon={<UploadOutlined />} disabled={loading}>
              选择文件
            </Button>
          </Upload>
          <div className="mt-2">
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
        </Form.Item>

        {showAdvanced && (
          <Space direction="vertical" style={{ width: '100%' }}>
            <Form.Item
              name="temperature"
              label={
                <Tooltip title="控制生成的随机性，0为确定性，1为最大随机性">
                  温度参数
                </Tooltip>
              }
            >
              <Slider
                min={0}
                max={1}
                step={0.1}
                marks={{
                  0: '0.0',
                  0.5: '0.5',
                  1: '1.0',
                }}
              />
            </Form.Item>

            <Form.Item
              name="maxTokens"
              label={
                <Tooltip title="生成回答的最大token数">最大Tokens</Tooltip>
              }
            >
              <InputNumber
                min={100}
                max={4000}
                step={100}
                style={{ width: '100%' }}
              />
            </Form.Item>

            <Form.Item
              name="topK"
              label={<Tooltip title="检索的文档数量">检索数量 (Top-K)</Tooltip>}
            >
              <InputNumber min={1} max={50} style={{ width: '100%' }} />
            </Form.Item>

            <Form.Item name="includeImages" valuePropName="checked">
              <Switch /> 包含图像检索
            </Form.Item>

            <Form.Item name="includeTables" valuePropName="checked">
              <Switch /> 包含表格检索
            </Form.Item>

            <Form.Item name="stream" valuePropName="checked">
              <Switch /> 流式响应
            </Form.Item>
          </Space>
        )}

        <Form.Item>
          <Button
            type="primary"
            htmlType="submit"
            icon={<SendOutlined />}
            loading={loading}
            block
          >
            执行查询
          </Button>
        </Form.Item>
      </Form>
    </Card>
  )
}

export default QueryInterface
