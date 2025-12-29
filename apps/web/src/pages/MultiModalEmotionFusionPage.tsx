import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useState } from 'react'
import { Card, Row, Col, Typography, Space, Button, Upload, Input, Alert, Spin, message } from 'antd'
import { CloudUploadOutlined, ReloadOutlined, PlayCircleOutlined } from '@ant-design/icons'
import type { UploadProps } from 'antd'

const { Title } = Typography
const { Dragger } = Upload
const { TextArea } = Input

const MultiModalEmotionFusionPage: React.FC = () => {
  const [text, setText] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)

  const analyze = async () => {
    if (!text && !file) {
      message.warning('请提供文本或上传文件')
      return
    }
    setLoading(true)
    setError(null)
    try {
      const form = new FormData()
      if (text) form.append('text', text)
      if (file) form.append('file', file)
      const res = await apiFetch(buildApiUrl('/api/v1/multimodal/emotion/fusion'), {
        method: 'POST',
        body: form
      })
      const data = await res.json()
      setResult(data)
    } catch (e: any) {
      setError(e?.message || '分析失败')
    } finally {
      setLoading(false)
    }
  }

  const uploadProps: UploadProps = {
    maxCount: 1,
    beforeUpload: (f) => {
      setFile(f)
      return false
    }
  }

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Space align="center" style={{ justifyContent: 'space-between', width: '100%' }}>
          <Title level={3} style={{ margin: 0 }}>
            多模态情感融合
          </Title>
          <Space>
            <Button icon={<ReloadOutlined />} onClick={() => { setResult(null); setText(''); setFile(null); }}>
              重置
            </Button>
            <Button type="primary" icon={<PlayCircleOutlined />} onClick={analyze} loading={loading}>
              分析
            </Button>
          </Space>
        </Space>

        {error && <Alert type="error" message="分析失败" description={error} />}

        <Card title="输入">
          <Row gutter={16}>
            <Col span={12}>
              <TextArea
                rows={6}
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="输入文本描述"
              />
            </Col>
            <Col span={12}>
              <Dragger {...uploadProps}>
                <p className="ant-upload-drag-icon">
                  <CloudUploadOutlined />
                </p>
                <p className="ant-upload-text">上传语音/图像文件（可选）</p>
              </Dragger>
            </Col>
          </Row>
        </Card>

        <Card title="结果">
          {loading ? (
            <Spin />
          ) : result ? (
            <pre style={{ whiteSpace: 'pre-wrap' }}>{JSON.stringify(result, null, 2)}</pre>
          ) : (
            <Alert type="info" message="提交后显示融合结果。" />
          )}
        </Card>
      </Space>
    </div>
  )
}

export default MultiModalEmotionFusionPage
