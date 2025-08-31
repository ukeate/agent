/**
 * 音频情感识别页面
 * Story 11.1: 语音情感识别器
 */

import React, { useState, useRef, useCallback } from 'react'
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Button,
  Upload,
  Progress,
  Tag,
  List,
  Alert,
  Statistic,
  Timeline,
  Badge,
  Select,
  Slider,
  Switch,
  Divider,
  message,
  Spin,
  Empty,
  Tooltip
} from 'antd'
import {
  SoundOutlined,
  AudioOutlined,
  UploadOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  SettingOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  BarChartOutlined,
  LineChartOutlined,
  RadarChartOutlined,
  WaveformOutlined,
  ThunderboltOutlined,
  HeartOutlined,
  SmileOutlined,
  FrownOutlined,
  MehOutlined,
  ExclamationCircleOutlined,
  CloudUploadOutlined,
  HistoryOutlined,
  InfoCircleOutlined,
  AudioMutedOutlined
} from '@ant-design/icons'
import { Line, Area, Column, Radar } from '@ant-design/plots'
import type { UploadProps } from 'antd'

const { Title, Text, Paragraph } = Typography
const { Dragger } = Upload
const { Option } = Select

const AudioEmotionRecognitionPage: React.FC = () => {
  const [analyzing, setAnalyzing] = useState(false)
  const [recording, setRecording] = useState(false)
  const [audioFile, setAudioFile] = useState<any>(null)
  const [analysisResult, setAnalysisResult] = useState<any>(null)
  const [waveformData, setWaveformData] = useState<any[]>([])
  const [realtimeMode, setRealtimeMode] = useState(false)
  const [history, setHistory] = useState<any[]>([])
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])

  const [settings, setSettings] = useState({
    model: 'wav2vec2-large',
    sampleRate: 16000,
    chunkSize: 5,
    enableNoiseReduction: true,
    enableVAD: true,
    confidenceThreshold: 0.5,
    realtimeBufferSize: 3
  })

  // 情感图标映射
  const emotionIcons: Record<string, React.ReactNode> = {
    happy: <SmileOutlined style={{ color: '#52c41a' }} />,
    sad: <FrownOutlined style={{ color: '#1890ff' }} />,
    angry: <ThunderboltOutlined style={{ color: '#f5222d' }} />,
    fear: <ExclamationCircleOutlined style={{ color: '#faad14' }} />,
    surprise: <SmileOutlined style={{ color: '#722ed1' }} />,
    neutral: <MehOutlined style={{ color: '#8c8c8c' }} />,
    calm: <HeartOutlined style={{ color: '#13c2c2' }} />,
    disgust: <FrownOutlined style={{ color: '#fa8c16' }} />
  }

  // 开始录音
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder
      audioChunksRef.current = []

      mediaRecorder.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data)
        
        // 实时模式下处理音频块
        if (realtimeMode && audioChunksRef.current.length % 5 === 0) {
          processRealtimeChunk()
        }
      }

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' })
        setAudioFile(audioBlob)
        stream.getTracks().forEach(track => track.stop())
      }

      mediaRecorder.start(100)
      setRecording(true)
      message.info('录音已开始')

      // 生成模拟波形数据
      generateWaveform()
    } catch (error) {
      message.error('无法访问麦克风')
    }
  }

  // 停止录音
  const stopRecording = () => {
    if (mediaRecorderRef.current && recording) {
      mediaRecorderRef.current.stop()
      setRecording(false)
      message.success('录音已停止')
    }
  }

  // 生成波形数据
  const generateWaveform = () => {
    const data = []
    for (let i = 0; i < 100; i++) {
      data.push({
        time: i,
        amplitude: Math.sin(i / 10) * Math.random() * 100
      })
    }
    setWaveformData(data)
  }

  // 处理实时音频块
  const processRealtimeChunk = () => {
    // 模拟实时处理
    const emotions = ['happy', 'sad', 'angry', 'neutral', 'calm']
    const emotion = emotions[Math.floor(Math.random() * emotions.length)]
    const confidence = Math.random() * 0.4 + 0.6
    
    message.info(`实时检测: ${emotion} (${(confidence * 100).toFixed(1)}%)`)
  }

  // 分析音频
  const analyzeAudio = useCallback(async () => {
    if (!audioFile) {
      message.warning('请先上传或录制音频')
      return
    }

    setAnalyzing(true)

    // 模拟API调用
    setTimeout(() => {
      const mockResult = {
        primaryEmotion: 'happy',
        confidence: 0.88,
        intensity: 0.75,
        emotions: [
          { label: 'happy', score: 0.88 },
          { label: 'excited', score: 0.65 },
          { label: 'calm', score: 0.42 },
          { label: 'neutral', score: 0.25 },
          { label: 'sad', score: 0.12 }
        ],
        features: {
          duration: 5.2,
          pitchMean: 220.5,
          pitchStd: 45.3,
          energyMean: 0.68,
          energyStd: 0.15,
          spectralCentroid: 1850.2,
          zeroCrossingRate: 0.042,
          mfcc: [12.3, -5.2, 8.7, -2.1, 6.5, -3.8, 4.2, -1.5, 3.1, -0.8, 2.4, -0.3, 1.8]
        },
        processingTime: 342,
        timestamp: new Date().toISOString()
      }

      setAnalysisResult(mockResult)
      setHistory([mockResult, ...history.slice(0, 9)])
      setAnalyzing(false)
      message.success('音频情感分析完成')
    }, 2000)
  }, [audioFile, history])

  // 上传配置
  const uploadProps: UploadProps = {
    name: 'audio',
    accept: 'audio/*',
    maxCount: 1,
    beforeUpload: (file) => {
      const isAudio = file.type.startsWith('audio/')
      if (!isAudio) {
        message.error('只能上传音频文件!')
        return false
      }
      const isLt10M = file.size / 1024 / 1024 < 10
      if (!isLt10M) {
        message.error('音频文件必须小于10MB!')
        return false
      }
      setAudioFile(file)
      generateWaveform()
      return false
    },
    onRemove: () => {
      setAudioFile(null)
      setAnalysisResult(null)
      setWaveformData([])
    }
  }

  // 音频特征雷达图配置
  const featureRadarConfig = analysisResult ? {
    data: [
      { name: '音高', value: (analysisResult.features.pitchMean / 300) * 100 },
      { name: '能量', value: analysisResult.features.energyMean * 100 },
      { name: '频谱中心', value: (analysisResult.features.spectralCentroid / 3000) * 100 },
      { name: '韵律变化', value: (analysisResult.features.pitchStd / 100) * 100 },
      { name: '情感强度', value: analysisResult.intensity * 100 }
    ],
    xField: 'name',
    yField: 'value',
    meta: {
      value: {
        alias: '特征值',
        min: 0,
        max: 100
      }
    },
    xAxis: {
      line: null,
      tickLine: null
    },
    point: {
      size: 4
    },
    area: {}
  } : null

  // 波形图配置
  const waveformConfig = {
    data: waveformData,
    xField: 'time',
    yField: 'amplitude',
    smooth: true,
    areaStyle: {
      fill: 'l(270) 0:#ffffff 1:#1890ff'
    },
    line: {
      color: '#1890ff'
    }
  }

  // 情感分布图配置
  const emotionBarConfig = analysisResult ? {
    data: analysisResult.emotions,
    xField: 'label',
    yField: 'score',
    label: {
      position: 'top',
      style: {
        fill: '#666'
      }
    },
    color: (datum: any) => {
      const colors: Record<string, string> = {
        happy: '#52c41a',
        sad: '#1890ff',
        angry: '#f5222d',
        fear: '#faad14',
        neutral: '#8c8c8c',
        calm: '#13c2c2',
        excited: '#fa8c16'
      }
      return colors[datum.label] || '#666'
    }
  } : null

  return (
    <div style={{ padding: '24px' }}>
      {/* 页面标题 */}
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>
          <Space>
            <SoundOutlined />
            语音情感识别
          </Space>
        </Title>
        <Paragraph type="secondary">
          基于Wav2Vec2的实时语音情感分析，支持音频文件上传和实时录音分析
        </Paragraph>
      </div>

      {/* 实时模式提示 */}
      {realtimeMode && (
        <Alert
          message="实时模式已启用"
          description="系统将在录音过程中实时分析情感变化"
          type="info"
          showIcon
          closable
          style={{ marginBottom: 24 }}
        />
      )}

      <Row gutter={[24, 24]}>
        {/* 左侧输入区 */}
        <Col xs={24} lg={12}>
          {/* 音频输入卡片 */}
          <Card title="音频输入">
            <Space direction="vertical" style={{ width: '100%' }}>
              {/* 录音控制 */}
              <div style={{ textAlign: 'center', marginBottom: 24 }}>
                <Space size="large">
                  <Button
                    type={recording ? 'danger' : 'primary'}
                    size="large"
                    shape="circle"
                    icon={recording ? <AudioMutedOutlined /> : <AudioOutlined />}
                    onClick={recording ? stopRecording : startRecording}
                    style={{ width: 80, height: 80 }}
                  />
                  {audioFile && (
                    <Button
                      size="large"
                      shape="circle"
                      icon={<PlayCircleOutlined />}
                      style={{ width: 60, height: 60 }}
                    />
                  )}
                </Space>
                <div style={{ marginTop: 16 }}>
                  <Text type={recording ? 'danger' : 'secondary'}>
                    {recording ? '正在录音...' : '点击开始录音'}
                  </Text>
                </div>
              </div>

              <Divider>或</Divider>

              {/* 文件上传 */}
              <Dragger {...uploadProps}>
                <p className="ant-upload-drag-icon">
                  <CloudUploadOutlined style={{ fontSize: 48, color: '#1890ff' }} />
                </p>
                <p className="ant-upload-text">点击或拖拽音频文件到此区域</p>
                <p className="ant-upload-hint">
                  支持 WAV, MP3, M4A 等常见音频格式，文件大小不超过10MB
                </p>
              </Dragger>

              {/* 分析按钮 */}
              <Button
                type="primary"
                size="large"
                icon={<ThunderboltOutlined />}
                loading={analyzing}
                onClick={analyzeAudio}
                disabled={!audioFile}
                block
              >
                开始分析
              </Button>
            </Space>
          </Card>

          {/* 波形显示 */}
          {waveformData.length > 0 && (
            <Card title="音频波形" style={{ marginTop: 24 }}>
              <Area {...waveformConfig} />
            </Card>
          )}

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
                  <Option value="wav2vec2-large">Wav2Vec2-Large (推荐)</Option>
                  <Option value="wav2vec2-base">Wav2Vec2-Base</Option>
                  <Option value="hubert-large">HuBERT-Large</Option>
                  <Option value="whisper">Whisper (多语言)</Option>
                </Select>
              </div>

              <div>
                <Text>采样率：{settings.sampleRate} Hz</Text>
                <Slider
                  min={8000}
                  max={48000}
                  step={8000}
                  value={settings.sampleRate}
                  onChange={v => setSettings({...settings, sampleRate: v})}
                />
              </div>

              <div>
                <Space style={{ width: '100%', justifyContent: 'space-between' }}>
                  <Text>实时模式</Text>
                  <Switch 
                    checked={realtimeMode}
                    onChange={setRealtimeMode}
                  />
                </Space>
              </div>

              <div>
                <Space style={{ width: '100%', justifyContent: 'space-between' }}>
                  <Text>噪声抑制</Text>
                  <Switch 
                    checked={settings.enableNoiseReduction}
                    onChange={v => setSettings({...settings, enableNoiseReduction: v})}
                  />
                </Space>
              </div>

              <div>
                <Space style={{ width: '100%', justifyContent: 'space-between' }}>
                  <Text>语音活动检测(VAD)</Text>
                  <Switch 
                    checked={settings.enableVAD}
                    onChange={v => setSettings({...settings, enableVAD: v})}
                  />
                </Space>
              </div>
            </Space>
          </Card>
        </Col>

        {/* 右侧结果区 */}
        <Col xs={24} lg={12}>
          {analyzing ? (
            <Card style={{ textAlign: 'center', padding: '60px 0' }}>
              <Spin size="large" tip="正在分析音频情感..." />
            </Card>
          ) : analysisResult ? (
            <>
              {/* 主要结果 */}
              <Card title="分析结果">
                <Row gutter={16}>
                  <Col span={8}>
                    <Statistic
                      title="主要情感"
                      value={analysisResult.primaryEmotion}
                      prefix={emotionIcons[analysisResult.primaryEmotion]}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="置信度"
                      value={(analysisResult.confidence * 100).toFixed(1)}
                      suffix="%"
                      valueStyle={{ color: '#52c41a' }}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="强度"
                      value={(analysisResult.intensity * 100).toFixed(1)}
                      suffix="%"
                      valueStyle={{ color: '#fa8c16' }}
                    />
                  </Col>
                </Row>

                <Divider />

                {/* 音频特征 */}
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Text strong>音频特征：</Text>
                  <Row gutter={[16, 16]}>
                    <Col span={12}>
                      <Text type="secondary">时长：{analysisResult.features.duration.toFixed(1)}秒</Text>
                    </Col>
                    <Col span={12}>
                      <Text type="secondary">平均音高：{analysisResult.features.pitchMean.toFixed(1)} Hz</Text>
                    </Col>
                    <Col span={12}>
                      <Text type="secondary">能量均值：{analysisResult.features.energyMean.toFixed(2)}</Text>
                    </Col>
                    <Col span={12}>
                      <Text type="secondary">频谱中心：{analysisResult.features.spectralCentroid.toFixed(1)} Hz</Text>
                    </Col>
                  </Row>

                  <Space wrap style={{ marginTop: 16 }}>
                    <Tag icon={<ClockCircleOutlined />}>
                      处理时间: {analysisResult.processingTime}ms
                    </Tag>
                    <Tag icon={<CheckCircleOutlined />} color="success">
                      分析成功
                    </Tag>
                  </Space>
                </Space>
              </Card>

              {/* 情感分布 */}
              <Card title="情感分布" style={{ marginTop: 24 }}>
                {emotionBarConfig && <Column {...emotionBarConfig} />}
              </Card>

              {/* 音频特征雷达图 */}
              <Card title="音频特征分析" style={{ marginTop: 24 }}>
                {featureRadarConfig && <Radar {...featureRadarConfig} />}
              </Card>
            </>
          ) : (
            <Card>
              <Empty
                description="暂无分析结果"
                image={Empty.PRESENTED_IMAGE_SIMPLE}
              >
                <Space>
                  <Button type="primary" onClick={startRecording}>
                    开始录音
                  </Button>
                  <Button>上传音频文件</Button>
                </Space>
              </Empty>
            </Card>
          )}

          {/* 历史记录 */}
          {history.length > 0 && (
            <Card 
              title={
                <Space>
                  <HistoryOutlined />
                  分析历史
                </Space>
              } 
              style={{ marginTop: 24 }}
            >
              <Timeline>
                {history.slice(0, 5).map((item, index) => (
                  <Timeline.Item
                    key={index}
                    dot={emotionIcons[item.primaryEmotion]}
                  >
                    <Space direction="vertical" size="small">
                      <Space>
                        <Tag color="blue">{item.primaryEmotion}</Tag>
                        <Text type="secondary">
                          置信度: {(item.confidence * 100).toFixed(1)}%
                        </Text>
                      </Space>
                      <Text type="secondary">
                        {new Date(item.timestamp).toLocaleTimeString()}
                      </Text>
                    </Space>
                  </Timeline.Item>
                ))}
              </Timeline>
            </Card>
          )}
        </Col>
      </Row>
    </div>
  )
}

export default AudioEmotionRecognitionPage