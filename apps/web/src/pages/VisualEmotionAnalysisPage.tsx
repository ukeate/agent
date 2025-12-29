/**
 * 视觉情感分析页面
 * Story 11.1: 视觉情感识别器
 */

import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useState, useRef, useCallback } from 'react'
import {
import { logger } from '../utils/logger'
  Card,
  Row,
  Col,
  Typography,
  Space,
  Button,
  Upload,
  Progress,
  Tag,
  Image,
  Alert,
  Statistic,
  List,
  Badge,
  Select,
  Switch,
  Slider,
  Divider,
  message,
  Spin,
  Empty,
  Tooltip,
  Avatar
} from 'antd'
import {
  CameraOutlined,
  UploadOutlined,
  VideoCameraOutlined,
  PictureOutlined,
  FaceRecognitionOutlined,
  ScanOutlined,
  EyeOutlined,
  SmileOutlined,
  FrownOutlined,
  MehOutlined,
  ThunderboltOutlined,
  HeartOutlined,
  ExclamationCircleOutlined,
  CloudUploadOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  SettingOutlined,
  ZoomInOutlined,
  BarChartOutlined,
  RadarChartOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  UserOutlined
} from '@ant-design/icons'
import { Column, Pie, Radar, Rose } from '@ant-design/plots'
import type { UploadProps } from 'antd'

const { Title, Text, Paragraph } = Typography
const { Dragger } = Upload
const { Option } = Select

const VisualEmotionAnalysisPage: React.FC = () => {
  const [analyzing, setAnalyzing] = useState(false)
  const [imageFile, setImageFile] = useState<any>(null)
  const [imageUrl, setImageUrl] = useState<string>('')
  const [analysisResult, setAnalysisResult] = useState<any>(null)
  const [videoMode, setVideoMode] = useState(false)
  const [streaming, setStreaming] = useState(false)
  const [detectedFaces, setDetectedFaces] = useState<any[]>([])
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)

  const [settings, setSettings] = useState({
    model: 'fer2013-cnn',
    faceDetection: 'opencv',
    minFaceSize: 40,
    confidenceThreshold: 0.5,
    multipleFaces: true,
    enableLandmarks: true,
    frameRate: 5
  })

  // 情感图标和颜色映射
  const emotionConfig: Record<string, { icon: React.ReactNode; color: string }> = {
    happy: { 
      icon: <SmileOutlined style={{ fontSize: 24, color: '#52c41a' }} />, 
      color: 'green' 
    },
    sad: { 
      icon: <FrownOutlined style={{ fontSize: 24, color: '#1890ff' }} />, 
      color: 'blue' 
    },
    angry: { 
      icon: <ThunderboltOutlined style={{ fontSize: 24, color: '#f5222d' }} />, 
      color: 'red' 
    },
    fear: { 
      icon: <ExclamationCircleOutlined style={{ fontSize: 24, color: '#faad14' }} />, 
      color: 'orange' 
    },
    surprise: { 
      icon: <SmileOutlined style={{ fontSize: 24, color: '#722ed1' }} />, 
      color: 'purple' 
    },
    neutral: { 
      icon: <MehOutlined style={{ fontSize: 24, color: '#8c8c8c' }} />, 
      color: 'default' 
    },
    disgust: { 
      icon: <FrownOutlined style={{ fontSize: 24, color: '#13c2c2' }} />, 
      color: 'cyan' 
    },
    contempt: { 
      icon: <MehOutlined style={{ fontSize: 24, color: '#fa8c16' }} />, 
      color: 'gold' 
    }
  }

  // 开始视频流
  const startVideoStream = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: 640, 
          height: 480,
          frameRate: settings.frameRate 
        } 
      })
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        streamRef.current = stream
        setStreaming(true)
        message.success('摄像头已启动')
        
        // 开始实时分析
        startRealtimeAnalysis()
      }
    } catch (error) {
      message.error('无法访问摄像头')
    }
  }

  // 停止视频流
  const stopVideoStream = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
      setStreaming(false)
      message.info('摄像头已停止')
    }
  }

  // 实时分析
  const startRealtimeAnalysis = () => {
    const analyzeFrame = async () => {
      if (!streaming || !videoRef.current) return
      try {
        const canvas = document.createElement('canvas')
        canvas.width = videoRef.current.videoWidth
        canvas.height = videoRef.current.videoHeight
        const ctx = canvas.getContext('2d')
        if (ctx) {
          ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height)
          const blob: Blob = await new Promise(resolve => canvas.toBlob(b => resolve(b as Blob), 'image/jpeg'))
          const formData = new FormData()
          formData.append('image_file', blob)
          const res = await apiFetch(buildApiUrl('/api/v1/emotion-recognition/analyze/visual'), {
            method: 'POST',
            body: formData
          })
          const data = await res.json()
          const faces = (data?.faces || []).map((f: any, idx: number) => ({
            id: idx,
            emotion: f.primary_emotion || f.emotion,
            confidence: f.confidence || f.score || 0,
            position: f.bounding_box || {}
          }))
          setDetectedFaces(faces)
        }
      } catch (e) {
        logger.error('实时视觉分析失败', e)
      } finally {
        if (streaming) {
          setTimeout(analyzeFrame, 1000 / settings.frameRate)
        }
      }
    }
    analyzeFrame()
  }

  // 分析图像
  const analyzeImage = useCallback(async () => {
    if (!imageFile && !imageUrl) {
      message.warning('请先上传图像或启动摄像头')
      return
    }

    setAnalyzing(true)

    try {
      const formData = new FormData()
      if (imageFile) formData.append('image_file', imageFile)
      if (imageUrl) formData.append('image_url', imageUrl)
      const res = await apiFetch(buildApiUrl('/api/v1/emotion-recognition/analyze/visual'), {
        method: 'POST',
        body: formData
      })
      const data = await res.json()
      setAnalysisResult(data)
      message.success('视觉情感分析完成')
    } catch (error: any) {
      message.error(error?.message || '视觉情感分析失败')
      setAnalysisResult(null)
    } finally {
      setAnalyzing(false)
    }
  }, [imageFile, imageUrl])

  // 上传配置
  const uploadProps: UploadProps = {
    name: 'image',
    accept: 'image/*',
    maxCount: 1,
    beforeUpload: (file) => {
      const isImage = file.type.startsWith('image/')
      if (!isImage) {
        message.error('只能上传图片文件!')
        return false
      }
      const isLt5M = file.size / 1024 / 1024 < 5
      if (!isLt5M) {
        message.error('图片必须小于5MB!')
        return false
      }
      
      // 创建图片预览URL
      const reader = new FileReader()
      reader.onload = (e) => {
        setImageUrl(e.target?.result as string)
      }
      reader.readAsDataURL(file)
      
      setImageFile(file)
      return false
    },
    onRemove: () => {
      setImageFile(null)
      setImageUrl('')
      setAnalysisResult(null)
    }
  }

  // 面部情感分布图配置
  const faceEmotionConfig = analysisResult ? {
    data: analysisResult.emotions.filter((e: any) => e.count > 0),
    xField: 'label',
    yField: 'avgConfidence',
    seriesField: 'label',
    radius: 0.9,
    label: {
      offset: -15,
    },
    interactions: [{ type: 'element-active' }],
  } : null

  // 情感统计饼图配置
  const emotionPieConfig = analysisResult ? {
    appendPadding: 10,
    data: analysisResult.emotions.filter((e: any) => e.count > 0).map((e: any) => ({
      type: e.label,
      value: e.count
    })),
    angleField: 'value',
    colorField: 'type',
    radius: 0.8,
    label: {
      type: 'outer',
      content: '{name} {percentage}'
    },
    interactions: [{ type: 'pie-legend-active' }],
  } : null

  return (
    <div style={{ padding: '24px' }}>
      {/* 页面标题 */}
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>
          <Space>
            <CameraOutlined />
            视觉情感分析
          </Space>
        </Title>
        <Paragraph type="secondary">
          基于深度学习的面部表情识别，支持图像上传和实时视频流分析
        </Paragraph>
      </div>

      {/* 模式切换提示 */}
      {videoMode && (
        <Alert
          message="视频模式"
          description="系统正在实时分析视频流中的面部表情"
          type="info"
          showIcon
          closable
          style={{ marginBottom: 24 }}
        />
      )}

      <Row gutter={[24, 24]}>
        {/* 左侧输入区 */}
        <Col xs={24} lg={12}>
          {/* 输入模式切换 */}
          <Card 
            title="输入源" 
            extra={
              <Space>
                <Button
                  type={!videoMode ? 'primary' : 'default'}
                  icon={<PictureOutlined />}
                  onClick={() => {
                    setVideoMode(false)
                    stopVideoStream()
                  }}
                >
                  图片
                </Button>
                <Button
                  type={videoMode ? 'primary' : 'default'}
                  icon={<VideoCameraOutlined />}
                  onClick={() => setVideoMode(true)}
                >
                  视频
                </Button>
              </Space>
            }
          >
            {!videoMode ? (
              <Space direction="vertical" style={{ width: '100%' }}>
                {/* 图片上传 */}
                <Dragger {...uploadProps}>
                  <p className="ant-upload-drag-icon">
                    <CloudUploadOutlined style={{ fontSize: 48, color: '#1890ff' }} />
                  </p>
                  <p className="ant-upload-text">点击或拖拽图片到此区域</p>
                  <p className="ant-upload-hint">
                    支持 JPG, PNG, GIF 等常见图片格式，文件大小不超过5MB
                  </p>
                </Dragger>

                {/* 图片预览 */}
                {imageUrl && (
                  <div style={{ marginTop: 16, textAlign: 'center' }}>
                    <Image
                      src={imageUrl}
                      alt="预览"
                      style={{ maxHeight: 300 }}
                      preview={{
                        mask: <Space><ZoomInOutlined /> 查看大图</Space>
                      }}
                    />
                    {analysisResult && detectedFaces.length > 0 && (
                      <div style={{ marginTop: 8 }}>
                        <Badge count={`检测到 ${analysisResult.numFaces} 张面部`} />
                      </div>
                    )}
                  </div>
                )}
              </Space>
            ) : (
              <Space direction="vertical" style={{ width: '100%' }}>
                {/* 视频流显示 */}
                <div style={{ position: 'relative', width: '100%', backgroundColor: '#000', borderRadius: 8, overflow: 'hidden' }}>
                  <video
                    ref={videoRef}
                    autoPlay
                    muted
                    style={{ width: '100%', height: 'auto' }}
                  />
                  
                  {/* 实时检测覆盖层 */}
                  {streaming && detectedFaces.map((face) => (
                    <div
                      key={face.id}
                      style={{
                        position: 'absolute',
                        left: face.position.x,
                        top: face.position.y,
                        width: face.position.width,
                        height: face.position.height,
                        border: '2px solid #52c41a',
                        borderRadius: 4
                      }}
                    >
                      <Tag 
                        color={emotionConfig[face.emotion]?.color}
                        style={{ position: 'absolute', top: -25 }}
                      >
                        {face.emotion} ({(face.confidence * 100).toFixed(0)}%)
                      </Tag>
                    </div>
                  ))}
                </div>

                {/* 视频控制 */}
                <div style={{ textAlign: 'center' }}>
                  <Space size="large">
                    {!streaming ? (
                      <Button
                        type="primary"
                        size="large"
                        icon={<PlayCircleOutlined />}
                        onClick={startVideoStream}
                      >
                        启动摄像头
                      </Button>
                    ) : (
                      <Button
                        type="danger"
                        size="large"
                        icon={<PauseCircleOutlined />}
                        onClick={stopVideoStream}
                      >
                        停止摄像头
                      </Button>
                    )}
                  </Space>
                </div>
              </Space>
            )}

            {/* 分析按钮 */}
            {!videoMode && (
              <Button
                type="primary"
                size="large"
                icon={<ScanOutlined />}
                loading={analyzing}
                onClick={analyzeImage}
                disabled={!imageFile}
                block
                style={{ marginTop: 16 }}
              >
                开始分析
              </Button>
            )}
          </Card>

          {/* 设置面板 */}
          <Card title="分析设置" style={{ marginTop: 24 }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Text>识别模型：</Text>
                <Select 
                  value={settings.model} 
                  style={{ width: '100%', marginTop: 8 }}
                  onChange={v => setSettings({...settings, model: v})}
                >
                  <Option value="fer2013-cnn">FER2013 CNN (推荐)</Option>
                  <Option value="affectnet">AffectNet</Option>
                  <Option value="vggface">VGGFace</Option>
                  <Option value="resnet50">ResNet50</Option>
                </Select>
              </div>

              <div>
                <Text>最小面部尺寸：{settings.minFaceSize}px</Text>
                <Slider
                  min={20}
                  max={100}
                  value={settings.minFaceSize}
                  onChange={v => setSettings({...settings, minFaceSize: v})}
                />
              </div>

              <div>
                <Space style={{ width: '100%', justifyContent: 'space-between' }}>
                  <Text>多人脸检测</Text>
                  <Switch 
                    checked={settings.multipleFaces}
                    onChange={v => setSettings({...settings, multipleFaces: v})}
                  />
                </Space>
              </div>

              <div>
                <Space style={{ width: '100%', justifyContent: 'space-between' }}>
                  <Text>面部关键点</Text>
                  <Switch 
                    checked={settings.enableLandmarks}
                    onChange={v => setSettings({...settings, enableLandmarks: v})}
                  />
                </Space>
              </div>

              {videoMode && (
                <div>
                  <Text>帧率：{settings.frameRate} FPS</Text>
                  <Slider
                    min={1}
                    max={30}
                    value={settings.frameRate}
                    onChange={v => setSettings({...settings, frameRate: v})}
                  />
                </div>
              )}
            </Space>
          </Card>
        </Col>

        {/* 右侧结果区 */}
        <Col xs={24} lg={12}>
          {analyzing ? (
            <Card style={{ textAlign: 'center', padding: '60px 0' }}>
              <Spin size="large" tip="正在分析面部表情..." />
            </Card>
          ) : analysisResult ? (
            <>
              {/* 主要结果 */}
              <Card title="分析结果">
                <Row gutter={16}>
                  <Col span={8}>
                    <Statistic
                      title="检测面部"
                      value={analysisResult.numFaces}
                      prefix={<UserOutlined />}
                      suffix="个"
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="主要情感"
                      value={analysisResult.aggregatedEmotion.emotion}
                      prefix={emotionConfig[analysisResult.aggregatedEmotion.emotion]?.icon}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="综合置信度"
                      value={(analysisResult.aggregatedEmotion.confidence * 100).toFixed(1)}
                      suffix="%"
                      valueStyle={{ color: '#52c41a' }}
                    />
                  </Col>
                </Row>

                <Divider />

                {/* 面部列表 */}
                <List
                  header={<Text strong>检测到的面部</Text>}
                  dataSource={analysisResult.faces}
                  renderItem={(face: any) => (
                    <List.Item
                      actions={[
                        <Tag color={emotionConfig[face.emotion]?.color}>
                          {(face.confidence * 100).toFixed(1)}%
                        </Tag>
                      ]}
                    >
                      <List.Item.Meta
                        avatar={
                          <Avatar 
                            size={48} 
                            icon={emotionConfig[face.emotion]?.icon}
                            style={{ backgroundColor: emotionConfig[face.emotion]?.color }}
                          />
                        }
                        title={`面部 ${face.id}: ${face.emotion}`}
                        description={
                          <Space direction="vertical" size="small">
                            <Text type="secondary">
                              位置: ({face.boundingBox.x}, {face.boundingBox.y})
                            </Text>
                            <Text type="secondary">
                              强度: {(face.intensity * 100).toFixed(1)}%
                            </Text>
                          </Space>
                        }
                      />
                    </List.Item>
                  )}
                />

                <Divider />

                {/* 处理信息 */}
                <Space>
                  <Tag icon={<ClockCircleOutlined />}>
                    {analysisResult.processingTime}ms
                  </Tag>
                  <Tag icon={<CheckCircleOutlined />} color="success">
                    分析完成
                  </Tag>
                </Space>
              </Card>

              {/* 情感分布 */}
              {emotionPieConfig && (
                <Card title="情感分布" style={{ marginTop: 24 }}>
                  <Pie {...emotionPieConfig} />
                </Card>
              )}

              {/* 置信度分析 */}
              {faceEmotionConfig && (
                <Card title="情感置信度" style={{ marginTop: 24 }}>
                  <Rose {...faceEmotionConfig} />
                </Card>
              )}
            </>
          ) : (
            <Card>
              <Empty
                description="暂无分析结果"
                image={Empty.PRESENTED_IMAGE_SIMPLE}
              >
                <Button type="primary">
                  上传图片开始分析
                </Button>
              </Empty>
            </Card>
          )}

          {/* 实时检测统计 (视频模式) */}
          {videoMode && streaming && detectedFaces.length > 0 && (
            <Card title="实时检测" style={{ marginTop: 24 }}>
              <Space direction="vertical" style={{ width: '100%' }}>
                <Text>当前检测到 {detectedFaces.length} 个面部</Text>
                <div>
                  {detectedFaces.map((face, index) => (
                    <Tag
                      key={index}
                      icon={emotionConfig[face.emotion]?.icon}
                      color={emotionConfig[face.emotion]?.color}
                      style={{ marginBottom: 8 }}
                    >
                      面部{index + 1}: {face.emotion} ({(face.confidence * 100).toFixed(0)}%)
                    </Tag>
                  ))}
                </div>
              </Space>
            </Card>
          )}
        </Col>
      </Row>
    </div>
  )
}

export default VisualEmotionAnalysisPage
