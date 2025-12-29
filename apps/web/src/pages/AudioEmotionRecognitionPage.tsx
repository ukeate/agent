import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useState } from 'react'
import { Card, Upload, Button, Space, Typography, Alert, Spin, Descriptions, message } from 'antd'
import { UploadOutlined, PlayCircleOutlined } from '@ant-design/icons'

const { Title } = Typography
const { Dragger } = Upload

const AudioEmotionRecognitionPage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)

  const props = {
    accept: '.wav,.mp3,.m4a',
    beforeUpload: (f: File) => {
      setFile(f)
      return false
    },
    maxCount: 1
  }

  const analyze = async () => {
    if (!file) {
      message.warning('请先选择音频文件')
      return
    }
    const formData = new FormData()
    formData.append('audio_file', file)
    setLoading(true)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/emotion-recognition/analyze/audio'), {
        method: 'POST',
        body: formData
      })
      const data = await res.json()
      setResult(data)
      message.success('分析完成')
    } catch (e: any) {
      message.error(e?.message || '分析失败')
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <Title level={3}>音频情感识别</Title>

        <Card title="上传音频">
          <Dragger {...props} disabled={loading} style={{ padding: 16 }}>
            <p>将音频文件拖到这里，或点击选择</p>
          </Dragger>
          <Button
            type="primary"
            icon={<PlayCircleOutlined />}
            style={{ marginTop: 16 }}
            onClick={analyze}
            loading={loading}
          >
            开始分析
          </Button>
          <Alert
            style={{ marginTop: 12 }}
            type="info"
            showIcon
            message="调用 /api/v1/emotion-recognition/analyze/audio 返回真实模型推理结果。"
          />
        </Card>

        <Card title="分析结果">
          {loading ? (
            <Spin />
          ) : !result ? (
            <Alert type="info" showIcon message="尚无结果" />
          ) : (
            <Descriptions column={1} bordered>
              <Descriptions.Item label="主情感">{result?.primaryEmotion || result?.primary_emotion || '未知'}</Descriptions.Item>
              <Descriptions.Item label="置信度">{result?.confidence}</Descriptions.Item>
              <Descriptions.Item label="处理耗时(ms)">{result?.processingTime || result?.processing_time_ms}</Descriptions.Item>
              <Descriptions.Item label="原始响应">
                <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all', margin: 0 }}>
                  {JSON.stringify(result, null, 2)}
                </pre>
              </Descriptions.Item>
            </Descriptions>
          )}
        </Card>
      </Space>
    </div>
  )
}

export default AudioEmotionRecognitionPage
