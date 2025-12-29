/**
 * 文本情感分析页面
 * Story 11.1: 文本情感分析器
 */

import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useState, useCallback } from 'react'
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Input,
  Button,
  Tag,
  Progress,
  List,
  Avatar,
  Divider,
  Alert,
  Select,
  Switch,
  Slider,
  Timeline,
  Badge,
  Statistic,
  Tooltip,
  message,
  Spin,
  Empty
} from 'antd'
import {
  FileTextOutlined,
  SendOutlined,
  ClearOutlined,
  SettingOutlined,
  ExperimentOutlined,
  SmileOutlined,
  FrownOutlined,
  MehOutlined,
  ThunderboltOutlined,
  HeartOutlined,
  BulbOutlined,
  ClockCircleOutlined,
  BarChartOutlined,
  InfoCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  TagOutlined,
  GlobalOutlined,
  TranslationOutlined,
  HistoryOutlined,
  RobotOutlined
} from '@ant-design/icons'
import { Line, Column, Radar } from '@ant-design/plots'

const { Title, Text, Paragraph } = Typography
const { TextArea } = Input
const { Option } = Select

const TextEmotionAnalysisPage: React.FC = () => {
  const [inputText, setInputText] = useState('')
  const [analyzing, setAnalyzing] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<any>(null)
  const [history, setHistory] = useState<any[]>([])
  const [settings, setSettings] = useState({
    model: 'distilroberta-base',
    language: 'en',
    includeContext: true,
    confidenceThreshold: 0.5,
    maxLength: 512,
    temperature: 1.0,
    detailLevel: 'high'
  })

  // 情感图标映射
  const emotionIcons: Record<string, React.ReactNode> = {
    happiness: <SmileOutlined style={{ color: '#52c41a' }} />,
    sadness: <FrownOutlined style={{ color: '#1890ff' }} />,
    anger: <ThunderboltOutlined style={{ color: '#f5222d' }} />,
    fear: <ExclamationCircleOutlined style={{ color: '#faad14' }} />,
    surprise: <BulbOutlined style={{ color: '#722ed1' }} />,
    neutral: <MehOutlined style={{ color: '#8c8c8c' }} />,
    joy: <HeartOutlined style={{ color: '#eb2f96' }} />,
    disgust: <FrownOutlined style={{ color: '#13c2c2' }} />,
    trust: <CheckCircleOutlined style={{ color: '#52c41a' }} />,
    anticipation: <ClockCircleOutlined style={{ color: '#fa8c16' }} />
  }

  // 情感颜色映射
  const emotionColors: Record<string, string> = {
    happiness: 'green',
    sadness: 'blue',
    anger: 'red',
    fear: 'orange',
    surprise: 'purple',
    neutral: 'default',
    joy: 'magenta',
    disgust: 'cyan',
    trust: 'lime',
    anticipation: 'gold'
  }

  // 示例文本
  const sampleTexts = [
    "I'm absolutely thrilled about the new project! Can't wait to get started.",
    "这个产品真的让我很失望，完全没有达到预期的效果。",
    "The weather is nice today, I might go for a walk in the park.",
    "¡Estoy muy emocionado por las vacaciones! Será increíble.",
    "Je suis vraiment heureux de vous voir aujourd'hui!"
  ]

  // 分析文本
  const analyzeText = useCallback(async () => {
    if (!inputText.trim()) {
      message.warning('请输入要分析的文本')
      return
    }

    setAnalyzing(true)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/emotion-recognition/analyze/text'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: inputText,
          language: settings.language,
          analyze_intensity: true,
          include_details: true
        })
      })
      const data = await res.json()
      setAnalysisResult(data)
      setHistory([data, ...history.slice(0, 9)])
      message.success('文本情感分析完成')
    } catch (error: any) {
      message.error(error?.message || '文本情感分析失败')
      setAnalysisResult(null)
    } finally {
      setAnalyzing(false)
    }
  }, [inputText, history, settings.language])

  // VAD模型雷达图配置
  const vadRadarConfig = analysisResult ? {
    data: [
      { name: '效价(Valence)', value: analysisResult.valence * 100 },
      { name: '唤醒度(Arousal)', value: analysisResult.arousal * 100 },
      { name: '支配性(Dominance)', value: analysisResult.dominance * 100 }
    ],
    xField: 'name',
    yField: 'value',
    meta: {
      value: {
        alias: '强度',
        min: 0,
        max: 100
      }
    },
    xAxis: {
      line: null,
      tickLine: null
    },
    yAxis: {
      line: null,
      tickLine: null,
      grid: {
        line: {
          type: 'line'
        }
      }
    },
    point: {
      size: 4
    },
    area: {}
  } : null

  // 情感分布柱状图配置
  const emotionBarConfig = analysisResult ? {
    data: analysisResult.emotions,
    xField: 'label',
    yField: 'score',
    label: {
      position: 'middle',
      style: {
        fill: '#FFFFFF',
        opacity: 0.6
      }
    },
    xAxis: {
      label: {
        autoHide: true,
        autoRotate: false
      }
    },
    meta: {
      score: {
        alias: '置信度'
      }
    }
  } : null

  return (
    <div style={{ padding: '24px' }}>
      {/* 页面标题 */}
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>
          <Space>
            <FileTextOutlined />
            文本情感分析器
          </Space>
        </Title>
        <Paragraph type="secondary">
          基于Transformer的高精度文本情感识别，支持20+情感类别和细粒度情感分析
        </Paragraph>
      </div>

      <Row gutter={[24, 24]}>
        {/* 左侧输入区 */}
        <Col xs={24} lg={12}>
          {/* 输入卡片 */}
          <Card 
            title="文本输入" 
            extra={
              <Space>
                <Select 
                  name="emotionLanguage"
                  value={settings.language} 
                  style={{ width: 100 }}
                  onChange={v => setSettings({...settings, language: v})}
                >
                  <Option value="en">English</Option>
                  <Option value="zh">中文</Option>
                  <Option value="es">Español</Option>
                  <Option value="fr">Français</Option>
                  <Option value="auto">自动检测</Option>
                </Select>
                <Button 
                  icon={<ClearOutlined />} 
                  onClick={() => {
                    setInputText('')
                    setAnalysisResult(null)
                  }}
                >
                  清空
                </Button>
              </Space>
            }
          >
            <TextArea
              name="emotionTextInput"
              rows={6}
              value={inputText}
              onChange={e => setInputText(e.target.value)}
              placeholder="请输入要分析的文本内容..."
              maxLength={settings.maxLength}
              showCount
            />
            
            {/* 示例文本 */}
            <div style={{ marginTop: 16 }}>
              <Text type="secondary">快速示例：</Text>
              <Space wrap style={{ marginTop: 8 }}>
                {sampleTexts.map((text, index) => (
                  <Tag 
                    key={index}
                    color="blue"
                    style={{ cursor: 'pointer' }}
                    onClick={() => setInputText(text)}
                  >
                    示例 {index + 1}
                  </Tag>
                ))}
              </Space>
            </div>

            {/* 分析按钮 */}
            <Button
              type="primary"
              size="large"
              icon={<SendOutlined />}
              loading={analyzing}
              onClick={analyzeText}
              style={{ marginTop: 16, width: '100%' }}
            >
              开始分析
            </Button>
          </Card>

          {/* 设置卡片 */}
          <Card title="分析设置" style={{ marginTop: 24 }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Text>模型选择：</Text>
                <Select 
                  name="emotionModel"
                  value={settings.model} 
                  style={{ width: '100%', marginTop: 8 }}
                  onChange={v => setSettings({...settings, model: v})}
                >
                  <Option value="distilroberta-base">DistilRoBERTa (推荐)</Option>
                  <Option value="bert-base">BERT Base</Option>
                  <Option value="xlm-roberta">XLM-RoBERTa (多语言)</Option>
                  <Option value="albert-base">ALBERT Base (轻量)</Option>
                </Select>
              </div>

              <div>
                <Text>置信度阈值：{settings.confidenceThreshold}</Text>
                <Slider
                  min={0}
                  max={1}
                  step={0.1}
                  value={settings.confidenceThreshold}
                  onChange={v => setSettings({...settings, confidenceThreshold: v})}
                />
              </div>

              <div>
                <Space style={{ width: '100%', justifyContent: 'space-between' }}>
                  <Text>包含上下文分析</Text>
                  <Switch 
                    checked={settings.includeContext}
                    onChange={v => setSettings({...settings, includeContext: v})}
                  />
                </Space>
              </div>

              <div>
                <Text>温度参数：{settings.temperature}</Text>
                <Slider
                  min={0.1}
                  max={2}
                  step={0.1}
                  value={settings.temperature}
                  onChange={v => setSettings({...settings, temperature: v})}
                />
              </div>
            </Space>
          </Card>
        </Col>

        {/* 右侧结果区 */}
        <Col xs={24} lg={12}>
          {analyzing ? (
            <Card style={{ textAlign: 'center', padding: '60px 0' }}>
              <Spin size="large" tip="正在分析文本情感..." />
            </Card>
          ) : analysisResult ? (
            <>
              {/* 主要结果卡片 */}
              <Card title="分析结果">
                <Row gutter={16}>
                  <Col span={8}>
                    <Statistic
                      title="主要情感"
                      value={analysisResult.primaryEmotion}
                      prefix={emotionIcons[analysisResult.primaryEmotion]}
                      valueStyle={{ color: emotionColors[analysisResult.primaryEmotion] }}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="置信度"
                      value={analysisResult.confidence}
                      precision={2}
                      suffix="%"
                      valueStyle={{ color: '#3f8600' }}
                      formatter={(value: any) => `${(value * 100).toFixed(1)}`}
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="强度"
                      value={analysisResult.intensity}
                      precision={2}
                      suffix="%"
                      valueStyle={{ color: '#cf1322' }}
                      formatter={(value: any) => `${(value * 100).toFixed(1)}`}
                    />
                  </Col>
                </Row>

                <Divider />

                {/* 情感标签 */}
                <div style={{ marginBottom: 16 }}>
                  <Text strong>检测到的情感：</Text>
                  <div style={{ marginTop: 8 }}>
                    {analysisResult.emotions.map((emotion: any) => (
                      <Tag
                        key={emotion.label}
                        icon={emotionIcons[emotion.label]}
                        color={emotionColors[emotion.label]}
                        style={{ marginBottom: 8 }}
                      >
                        {emotion.label} ({(emotion.score * 100).toFixed(1)}%)
                      </Tag>
                    ))}
                  </div>
                </div>

                {/* 文本特征 */}
                <div>
                  <Text strong>文本特征：</Text>
                  <Row gutter={16} style={{ marginTop: 8 }}>
                    <Col span={12}>
                      <Text type="secondary">词数：{analysisResult.features.wordCount}</Text>
                    </Col>
                    <Col span={12}>
                      <Text type="secondary">句数：{analysisResult.features.sentenceCount}</Text>
                    </Col>
                    <Col span={12}>
                      <Text type="secondary">感叹号：{analysisResult.features.exclamationCount}</Text>
                    </Col>
                    <Col span={12}>
                      <Text type="secondary">问号：{analysisResult.features.questionCount}</Text>
                    </Col>
                  </Row>
                </div>

                <Divider />

                {/* 处理信息 */}
                <Space>
                  <Tag icon={<ClockCircleOutlined />}>
                    {analysisResult.processingTime}ms
                  </Tag>
                  <Tag icon={<GlobalOutlined />}>
                    {analysisResult.language.toUpperCase()}
                  </Tag>
                  <Tag icon={<RobotOutlined />}>
                    {settings.model}
                  </Tag>
                </Space>
              </Card>

              {/* 情感维度分析 */}
              <Card title="VAD情感维度" style={{ marginTop: 24 }}>
                {vadRadarConfig && <Radar {...vadRadarConfig} />}
              </Card>

              {/* 情感分布图 */}
              <Card title="情感分布" style={{ marginTop: 24 }}>
                {emotionBarConfig && <Column {...emotionBarConfig} />}
              </Card>
            </>
          ) : (
            <Card>
              <Empty
                description="暂无分析结果"
                image={Empty.PRESENTED_IMAGE_SIMPLE}
              >
                <Button type="primary" onClick={() => setInputText(sampleTexts[0])}>
                  试试示例文本
                </Button>
              </Empty>
            </Card>
          )}

          {/* 历史记录 */}
          {history.length > 0 && (
            <Card 
              title={
                <Space>
                  <HistoryOutlined />
                  历史记录
                </Space>
              } 
              style={{ marginTop: 24 }}
            >
              <Timeline>
                {history.slice(0, 5).map((item, index) => (
                  <Timeline.Item
                    key={index}
                    color={emotionColors[item.primaryEmotion]}
                    dot={emotionIcons[item.primaryEmotion]}
                  >
                    <Space direction="vertical" size="small">
                      <Text ellipsis style={{ maxWidth: '100%' }}>
                        {item.text.substring(0, 50)}...
                      </Text>
                      <Space>
                        <Tag color={emotionColors[item.primaryEmotion]}>
                          {item.primaryEmotion}
                        </Tag>
                        <Text type="secondary">
                          {new Date(item.timestamp).toLocaleTimeString()}
                        </Text>
                      </Space>
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

export default TextEmotionAnalysisPage
