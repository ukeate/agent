/**
 * 多模态情感融合页面
 * Story 11.1: 多模态融合引擎
 */

import React, { useState, useCallback } from 'react'
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Button,
  Upload,
  Input,
  Progress,
  Tag,
  Image,
  Alert,
  Statistic,
  Timeline,
  Badge,
  Select,
  Radio,
  Switch,
  Divider,
  message,
  Spin,
  Empty,
  Tooltip,
  Steps,
  Result,
  Avatar,
  List
} from 'antd'
import {
  RobotOutlined,
  MergeCellsOutlined,
  FileTextOutlined,
  SoundOutlined,
  CameraOutlined,
  ThunderboltOutlined,
  SyncOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  SettingOutlined,
  RadarChartOutlined,
  BarChartOutlined,
  LineChartOutlined,
  SmileOutlined,
  FrownOutlined,
  MehOutlined,
  HeartOutlined,
  ExclamationCircleOutlined,
  CloudUploadOutlined,
  SendOutlined,
  AudioOutlined,
  PictureOutlined,
  PlayCircleOutlined,
  ExperimentOutlined
} from '@ant-design/icons'
import { Radar, Pie, Column, Area } from '@ant-design/plots'
import type { UploadProps } from 'antd'

const { Title, Text, Paragraph } = Typography
const { TextArea } = Input
const { Step } = Steps
const { Option } = Select
const { Dragger } = Upload

const MultiModalEmotionFusionPage: React.FC = () => {
  const [currentStep, setCurrentStep] = useState(0)
  const [analyzing, setAnalyzing] = useState(false)
  const [fusionResult, setFusionResult] = useState<any>(null)
  
  const [inputs, setInputs] = useState({
    text: '',
    audio: null as any,
    image: null as any
  })
  
  const [modalityResults, setModalityResults] = useState({
    text: null as any,
    audio: null as any,
    visual: null as any
  })

  const [fusionSettings, setFusionSettings] = useState({
    strategy: 'dynamic_adaptive',
    conflictResolution: 'confidence',
    enableTemporal: true,
    temporalWindow: 3,
    weights: {
      text: 0.4,
      audio: 0.35,
      visual: 0.25
    }
  })

  // 融合策略选项
  const fusionStrategies = [
    { value: 'weighted_average', label: '加权平均', description: '基于预定义权重进行融合' },
    { value: 'confidence_based', label: '置信度优先', description: '选择置信度最高的模态' },
    { value: 'dynamic_adaptive', label: '动态自适应', description: '根据一致性动态调整策略' },
    { value: 'hierarchical', label: '层次融合', description: '按优先级顺序融合' },
    { value: 'voting', label: '投票机制', description: '多数投票决定最终结果' }
  ]

  // 处理文本输入
  const handleTextInput = (text: string) => {
    setInputs({ ...inputs, text })
    if (text.trim()) {
      // 模拟文本分析
      setTimeout(() => {
        setModalityResults({
          ...modalityResults,
          text: {
            emotion: 'happiness',
            confidence: 0.85,
            intensity: 0.72,
            processed: true
          }
        })
      }, 500)
    }
  }

  // 音频上传配置
  const audioUploadProps: UploadProps = {
    accept: 'audio/*',
    maxCount: 1,
    beforeUpload: (file) => {
      setInputs({ ...inputs, audio: file })
      // 模拟音频分析
      setTimeout(() => {
        setModalityResults({
          ...modalityResults,
          audio: {
            emotion: 'excitement',
            confidence: 0.78,
            intensity: 0.81,
            processed: true
          }
        })
      }, 800)
      return false
    }
  }

  // 图像上传配置
  const imageUploadProps: UploadProps = {
    accept: 'image/*',
    maxCount: 1,
    beforeUpload: (file) => {
      setInputs({ ...inputs, image: file })
      // 模拟图像分析
      setTimeout(() => {
        setModalityResults({
          ...modalityResults,
          visual: {
            emotion: 'joy',
            confidence: 0.82,
            intensity: 0.69,
            processed: true
          }
        })
      }, 600)
      return false
    }
  }

  // 开始融合分析
  const startFusionAnalysis = useCallback(async () => {
    // 检查是否有输入
    const hasInput = inputs.text || inputs.audio || inputs.image
    if (!hasInput) {
      message.warning('请至少输入一种模态的数据')
      return
    }

    setAnalyzing(true)
    setCurrentStep(1)

    // 模拟融合分析过程
    setTimeout(() => {
      setCurrentStep(2)
      
      setTimeout(() => {
        const mockFusionResult = {
          primaryEmotion: 'happiness',
          secondaryEmotions: [
            { emotion: 'excitement', confidence: 0.65 },
            { emotion: 'joy', confidence: 0.52 },
            { emotion: 'contentment', confidence: 0.38 }
          ],
          overallConfidence: 0.92,
          intensityLevel: 0.74,
          dimensions: {
            valence: 0.75,
            arousal: 0.68,
            dominance: 0.62
          },
          modalityWeights: {
            text: 0.42,
            audio: 0.33,
            visual: 0.25
          },
          modalityContributions: [
            { source: 'text', target: 'fusion', value: 0.42 },
            { source: 'audio', target: 'fusion', value: 0.33 },
            { source: 'visual', target: 'fusion', value: 0.25 }
          ],
          consistency: 0.78,
          fusionStrategy: fusionSettings.strategy,
          processingTime: 485,
          timestamp: new Date().toISOString()
        }
        
        setFusionResult(mockFusionResult)
        setCurrentStep(3)
        setAnalyzing(false)
        message.success('多模态情感融合分析完成')
      }, 2000)
    }, 1500)
  }, [inputs, fusionSettings])

  // 重置分析
  const resetAnalysis = () => {
    setCurrentStep(0)
    setInputs({ text: '', audio: null, image: null })
    setModalityResults({ text: null, audio: null, visual: null })
    setFusionResult(null)
  }

  // VAD维度雷达图配置
  const vadRadarConfig = fusionResult ? {
    data: [
      { name: '效价 Valence', value: fusionResult.dimensions.valence * 100 },
      { name: '唤醒度 Arousal', value: fusionResult.dimensions.arousal * 100 },
      { name: '支配性 Dominance', value: fusionResult.dimensions.dominance * 100 }
    ],
    xField: 'name',
    yField: 'value',
    meta: {
      value: { alias: '强度', min: 0, max: 100 }
    },
    area: {},
    point: { size: 4 }
  } : null

  // 模态权重饼图配置
  const modalityPieConfig = fusionResult ? {
    data: [
      { type: '文本', value: fusionResult.modalityWeights.text },
      { type: '音频', value: fusionResult.modalityWeights.audio },
      { type: '视觉', value: fusionResult.modalityWeights.visual }
    ],
    angleField: 'value',
    colorField: 'type',
    radius: 0.8,
    label: {
      type: 'spider',
      labelHeight: 28,
      content: '{name}\n{percentage}'
    },
    interactions: [{ type: 'element-active' }]
  } : null

  // 置信度进度配置
  const confidencePercent = fusionResult ? (fusionResult.overallConfidence * 100) : 0

  // 获取模态图标
  const getModalityIcon = (modality: string) => {
    const icons: Record<string, React.ReactNode> = {
      text: <FileTextOutlined style={{ fontSize: 24 }} />,
      audio: <SoundOutlined style={{ fontSize: 24 }} />,
      visual: <CameraOutlined style={{ fontSize: 24 }} />
    }
    return icons[modality] || <RobotOutlined style={{ fontSize: 24 }} />
  }

  // 获取情感图标
  const getEmotionIcon = (emotion: string) => {
    const icons: Record<string, React.ReactNode> = {
      happiness: <SmileOutlined style={{ color: '#52c41a' }} />,
      joy: <SmileOutlined style={{ color: '#eb2f96' }} />,
      excitement: <SmileOutlined style={{ color: '#fa8c16' }} />,
      sadness: <FrownOutlined style={{ color: '#1890ff' }} />,
      anger: <ThunderboltOutlined style={{ color: '#f5222d' }} />,
      neutral: <MehOutlined style={{ color: '#8c8c8c' }} />,
      contentment: <HeartOutlined style={{ color: '#13c2c2' }} />
    }
    return icons[emotion] || <MehOutlined />
  }

  return (
    <div style={{ padding: '24px' }}>
      {/* 页面标题 */}
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>
          <Space>
            <MergeCellsOutlined />
            多模态情感融合
          </Space>
        </Title>
        <Paragraph type="secondary">
          融合文本、语音、图像多种模态的情感分析结果，提供更准确和全面的情感理解
        </Paragraph>
      </div>

      {/* 分析步骤 */}
      <Card style={{ marginBottom: 24 }}>
        <Steps current={currentStep}>
          <Step title="输入数据" description="上传多模态数据" icon={<CloudUploadOutlined />} />
          <Step title="独立分析" description="各模态分析" icon={<ExperimentOutlined />} />
          <Step title="融合计算" description="多模态融合" icon={<MergeCellsOutlined />} />
          <Step title="结果展示" description="综合结果" icon={<CheckCircleOutlined />} />
        </Steps>
      </Card>

      <Row gutter={[24, 24]}>
        {/* 左侧输入和设置区 */}
        <Col xs={24} lg={12}>
          {/* 多模态输入 */}
          <Card title="多模态输入" extra={
            <Button 
              type="link" 
              onClick={resetAnalysis}
              disabled={analyzing}
            >
              重置
            </Button>
          }>
            <Space direction="vertical" style={{ width: '100%' }} size="large">
              {/* 文本输入 */}
              <div>
                <Space style={{ marginBottom: 8 }}>
                  <FileTextOutlined />
                  <Text strong>文本输入</Text>
                  {modalityResults.text?.processed && (
                    <Tag icon={<CheckCircleOutlined />} color="success">
                      已分析
                    </Tag>
                  )}
                </Space>
                <TextArea
                  rows={3}
                  value={inputs.text}
                  onChange={e => handleTextInput(e.target.value)}
                  placeholder="输入要分析的文本..."
                  disabled={analyzing}
                />
                {modalityResults.text && (
                  <div style={{ marginTop: 8 }}>
                    <Tag color="blue">
                      {modalityResults.text.emotion} ({(modalityResults.text.confidence * 100).toFixed(0)}%)
                    </Tag>
                  </div>
                )}
              </div>

              {/* 音频输入 */}
              <div>
                <Space style={{ marginBottom: 8 }}>
                  <SoundOutlined />
                  <Text strong>音频输入</Text>
                  {modalityResults.audio?.processed && (
                    <Tag icon={<CheckCircleOutlined />} color="success">
                      已分析
                    </Tag>
                  )}
                </Space>
                <Upload {...audioUploadProps}>
                  <Button icon={<AudioOutlined />} disabled={analyzing}>
                    {inputs.audio ? inputs.audio.name : '上传音频文件'}
                  </Button>
                </Upload>
                {modalityResults.audio && (
                  <div style={{ marginTop: 8 }}>
                    <Tag color="green">
                      {modalityResults.audio.emotion} ({(modalityResults.audio.confidence * 100).toFixed(0)}%)
                    </Tag>
                  </div>
                )}
              </div>

              {/* 图像输入 */}
              <div>
                <Space style={{ marginBottom: 8 }}>
                  <CameraOutlined />
                  <Text strong>图像输入</Text>
                  {modalityResults.visual?.processed && (
                    <Tag icon={<CheckCircleOutlined />} color="success">
                      已分析
                    </Tag>
                  )}
                </Space>
                <Upload {...imageUploadProps}>
                  <Button icon={<PictureOutlined />} disabled={analyzing}>
                    {inputs.image ? inputs.image.name : '上传图像文件'}
                  </Button>
                </Upload>
                {modalityResults.visual && (
                  <div style={{ marginTop: 8 }}>
                    <Tag color="purple">
                      {modalityResults.visual.emotion} ({(modalityResults.visual.confidence * 100).toFixed(0)}%)
                    </Tag>
                  </div>
                )}
              </div>

              <Divider />

              {/* 开始融合按钮 */}
              <Button
                type="primary"
                size="large"
                icon={<ThunderboltOutlined />}
                loading={analyzing}
                onClick={startFusionAnalysis}
                block
              >
                开始融合分析
              </Button>
            </Space>
          </Card>

          {/* 融合设置 */}
          <Card title="融合设置" style={{ marginTop: 24 }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Text strong>融合策略：</Text>
                <Radio.Group 
                  value={fusionSettings.strategy} 
                  onChange={e => setFusionSettings({...fusionSettings, strategy: e.target.value})}
                  style={{ marginTop: 8 }}
                >
                  <Space direction="vertical">
                    {fusionStrategies.map(strategy => (
                      <Radio key={strategy.value} value={strategy.value}>
                        <Space>
                          <Text strong>{strategy.label}</Text>
                          <Text type="secondary">- {strategy.description}</Text>
                        </Space>
                      </Radio>
                    ))}
                  </Space>
                </Radio.Group>
              </div>

              <Divider />

              <div>
                <Text strong>模态权重分配：</Text>
                <div style={{ marginTop: 12 }}>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Progress 
                      percent={fusionSettings.weights.text * 100} 
                      strokeColor="#1890ff" 
                      format={() => `文本 ${(fusionSettings.weights.text * 100).toFixed(0)}%`} 
                    />
                    <Progress 
                      percent={fusionSettings.weights.audio * 100} 
                      strokeColor="#52c41a" 
                      format={() => `音频 ${(fusionSettings.weights.audio * 100).toFixed(0)}%`} 
                    />
                    <Progress 
                      percent={fusionSettings.weights.visual * 100} 
                      strokeColor="#722ed1" 
                      format={() => `视觉 ${(fusionSettings.weights.visual * 100).toFixed(0)}%`} 
                    />
                  </Space>
                </div>
              </div>

              <Divider />

              <div>
                <Space style={{ width: '100%', justifyContent: 'space-between' }}>
                  <Text>时间融合</Text>
                  <Switch 
                    checked={fusionSettings.enableTemporal}
                    onChange={v => setFusionSettings({...fusionSettings, enableTemporal: v})}
                  />
                </Space>
              </div>

              <div>
                <Text>冲突解决：</Text>
                <Select 
                  value={fusionSettings.conflictResolution} 
                  style={{ width: '100%', marginTop: 8 }}
                  onChange={v => setFusionSettings({...fusionSettings, conflictResolution: v})}
                >
                  <Option value="confidence">置信度优先</Option>
                  <Option value="voting">投票决定</Option>
                  <Option value="average">加权平均</Option>
                </Select>
              </div>
            </Space>
          </Card>
        </Col>

        {/* 右侧结果区 */}
        <Col xs={24} lg={12}>
          {analyzing && currentStep > 0 ? (
            <Card style={{ textAlign: 'center', padding: '60px 0' }}>
              <Spin size="large" tip={
                currentStep === 1 ? "正在进行独立模态分析..." : "正在进行多模态融合..."
              } />
            </Card>
          ) : fusionResult ? (
            <>
              {/* 融合结果 */}
              <Card title="融合分析结果">
                <Result
                  icon={getEmotionIcon(fusionResult.primaryEmotion)}
                  title={fusionResult.primaryEmotion.toUpperCase()}
                  subTitle={`综合置信度: ${(fusionResult.overallConfidence * 100).toFixed(1)}% | 强度: ${(fusionResult.intensityLevel * 100).toFixed(1)}%`}
                  extra={[
                    <Button key="detail" type="primary">查看详情</Button>,
                    <Button key="save">保存结果</Button>
                  ]}
                />

                <Divider />

                {/* 次要情感 */}
                <div style={{ marginBottom: 16 }}>
                  <Text strong>次要情感：</Text>
                  <div style={{ marginTop: 8 }}>
                    {fusionResult.secondaryEmotions.map((item: any) => (
                      <Tag
                        key={item.emotion}
                        icon={getEmotionIcon(item.emotion)}
                        style={{ marginBottom: 8 }}
                      >
                        {item.emotion} ({(item.confidence * 100).toFixed(1)}%)
                      </Tag>
                    ))}
                  </div>
                </div>

                {/* 模态贡献 */}
                <div>
                  <Text strong>模态贡献度：</Text>
                  <Row gutter={16} style={{ marginTop: 12 }}>
                    <Col span={8}>
                      <Card size="small" style={{ textAlign: 'center' }}>
                        <FileTextOutlined style={{ fontSize: 24, color: '#1890ff' }} />
                        <div>{(fusionResult.modalityWeights.text * 100).toFixed(1)}%</div>
                      </Card>
                    </Col>
                    <Col span={8}>
                      <Card size="small" style={{ textAlign: 'center' }}>
                        <SoundOutlined style={{ fontSize: 24, color: '#52c41a' }} />
                        <div>{(fusionResult.modalityWeights.audio * 100).toFixed(1)}%</div>
                      </Card>
                    </Col>
                    <Col span={8}>
                      <Card size="small" style={{ textAlign: 'center' }}>
                        <CameraOutlined style={{ fontSize: 24, color: '#722ed1' }} />
                        <div>{(fusionResult.modalityWeights.visual * 100).toFixed(1)}%</div>
                      </Card>
                    </Col>
                  </Row>
                </div>

                <Divider />

                {/* 融合信息 */}
                <Space wrap>
                  <Tag icon={<MergeCellsOutlined />}>
                    策略: {fusionResult.fusionStrategy}
                  </Tag>
                  <Tag icon={<SyncOutlined />}>
                    一致性: {(fusionResult.consistency * 100).toFixed(1)}%
                  </Tag>
                  <Tag icon={<ClockCircleOutlined />}>
                    {fusionResult.processingTime}ms
                  </Tag>
                </Space>
              </Card>

              {/* VAD维度分析 */}
              <Card title="情感维度分析" style={{ marginTop: 24 }}>
                {vadRadarConfig && <Radar {...vadRadarConfig} />}
              </Card>

              {/* 模态权重分布 */}
              <Card title="模态权重分布" style={{ marginTop: 24 }}>
                {modalityPieConfig && <Pie {...modalityPieConfig} />}
              </Card>

              {/* 置信度进度条 */}
              <Card title="融合置信度" style={{ marginTop: 24 }}>
                <div style={{ textAlign: 'center', padding: '20px 0' }}>
                  <Progress 
                    type="circle" 
                    percent={confidencePercent} 
                    width={120}
                    strokeColor={{
                      '0%': '#108ee9',
                      '100%': '#87d068',
                    }}
                  />
                  <div style={{ marginTop: 16 }}>
                    <Text strong>综合置信度: {confidencePercent.toFixed(1)}%</Text>
                  </div>
                </div>
              </Card>
            </>
          ) : (
            <Card>
              <Empty
                description="暂无融合结果"
                image={Empty.PRESENTED_IMAGE_SIMPLE}
              >
                <Text type="secondary">请输入多模态数据开始分析</Text>
              </Empty>
            </Card>
          )}
        </Col>
      </Row>
    </div>
  )
}

export default MultiModalEmotionFusionPage