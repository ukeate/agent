import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Button,
  Steps,
  Space,
  Typography,
  Tabs,
  Progress,
  Tag,
  Statistic,
  Alert,
  Select,
  Input,
  Form,
  Radio,
  Divider,
  List,
  Avatar,
  Badge,
  message,
  Modal,
  Tooltip,
  Slider,
  Rate,
  Tree,
  Table,
  Switch,
  Checkbox,
  InputNumber,
  Empty
} from 'antd'
import {
  BulbOutlined,
  RocketOutlined,
  AimOutlined,
  ThunderboltOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  InfoCircleOutlined,
  StarOutlined,
  TrophyOutlined,
  ClockCircleOutlined,
  DatabaseOutlined,
  SwapOutlined,
  SettingOutlined,
  ExperimentOutlined,
  BarChartOutlined,
  LineChartOutlined,
  EyeOutlined,
  DownloadOutlined,
  PlayCircleOutlined,
  WifiOutlined,
  MobileOutlined,
  CloudOutlined,
  DesktopOutlined
} from '@ant-design/icons'
import { Column, Line, Pie } from '@ant-design/plots'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs
const { Step } = Steps
const { Option } = Select
const { TextArea } = Input

interface ModelRequirement {
  modelType: 'cnn' | 'transformer' | 'rnn' | 'gan' | 'other'
  modelSize: number // MB
  accuracy: number // %
  targetDevice: 'mobile' | 'edge' | 'cloud' | 'embedded'
  latencyRequirement: number // ms
  memoryLimit: number // MB
  powerConstraint: number // W
  batchSize: number
  inputShape: string
}

interface CompressionStrategy {
  id: string
  name: string
  type: 'quantization' | 'pruning' | 'distillation' | 'mixed'
  description: string
  advantages: string[]
  disadvantages: string[]
  suitableFor: string[]
  expectedCompressionRatio: number
  expectedSpeedup: number
  expectedAccuracyLoss: number
  complexity: 'low' | 'medium' | 'high'
  implementationTime: number // hours
  score: number
  confidence: number
}

interface RecommendationResult {
  strategies: CompressionStrategy[]
  reasoning: string
  tradeOffs: {
    factor: string
    impact: 'high' | 'medium' | 'low'
    description: string
  }[]
  timeline: {
    phase: string
    duration: number
    tasks: string[]
  }[]
}

const CompressionStrategyRecommendationPage: React.FC = () => {
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('wizard')
  const [currentStep, setCurrentStep] = useState(0)
  const [requirements, setRequirements] = useState<Partial<ModelRequirement>>({})
  const [recommendationResult, setRecommendationResult] = useState<RecommendationResult | null>(null)
  const [selectedStrategies, setSelectedStrategies] = useState<string[]>([])
  const [strategyModalVisible, setStrategyModalVisible] = useState(false)
  const [selectedStrategy, setSelectedStrategy] = useState<CompressionStrategy | null>(null)

  useEffect(() => {
    // 预设推荐结果（模拟）
    setRecommendationResult({
      strategies: [
        {
          id: 'quantization-int8',
          name: 'INT8量化',
          type: 'quantization',
          description: '将FP32权重量化为INT8，显著减少模型大小和内存使用',
          advantages: [
            '实现简单，工具支持良好',
            '压缩比高（4x）',
            '兼容性好，支持多种硬件',
            '推理速度提升明显'
          ],
          disadvantages: [
            '需要校准数据集',
            '可能有精度损失',
            '某些层可能不适合量化'
          ],
          suitableFor: ['CNN模型', '大型模型', '移动端部署', '边缘设备'],
          expectedCompressionRatio: 4.0,
          expectedSpeedup: 2.5,
          expectedAccuracyLoss: 1.2,
          complexity: 'low',
          implementationTime: 8,
          score: 92,
          confidence: 88
        },
        {
          id: 'structured-pruning',
          name: '结构化剪枝',
          type: 'pruning',
          description: '移除整个通道或滤波器，保持模型结构规整',
          advantages: [
            '硬件友好，无需特殊支持',
            '可以显著减少FLOPs',
            '与其他技术兼容性好',
            '精度保持相对较好'
          ],
          disadvantages: [
            '压缩比相对有限',
            '需要精心设计剪枝策略',
            '可能需要重新训练'
          ],
          suitableFor: ['ResNet系列', 'MobileNet系列', '实时推理', '资源受限环境'],
          expectedCompressionRatio: 2.8,
          expectedSpeedup: 2.1,
          expectedAccuracyLoss: 0.8,
          complexity: 'medium',
          implementationTime: 16,
          score: 85,
          confidence: 82
        },
        {
          id: 'knowledge-distillation',
          name: '知识蒸馏',
          type: 'distillation',
          description: '训练小模型学习大模型的知识，在保持精度的同时减少模型大小',
          advantages: [
            '精度保持最好',
            '可以大幅减少模型大小',
            '适合复杂任务',
            '可与其他技术结合'
          ],
          disadvantages: [
            '需要原始训练数据',
            '训练时间长',
            '实现复杂度高',
            '需要设计学生模型'
          ],
          suitableFor: ['Transformer模型', 'NLP任务', '复杂分类任务', '高精度要求'],
          expectedCompressionRatio: 5.2,
          expectedSpeedup: 3.8,
          expectedAccuracyLoss: 0.5,
          complexity: 'high',
          implementationTime: 32,
          score: 88,
          confidence: 75
        },
        {
          id: 'mixed-strategy',
          name: '混合压缩策略',
          type: 'mixed',
          description: '结合量化、剪枝和蒸馏技术，达到最佳压缩效果',
          advantages: [
            '压缩效果最佳',
            '可以针对性优化',
            '充分利用各技术优势',
            '适应性强'
          ],
          disadvantages: [
            '实现复杂度很高',
            '需要大量实验调优',
            '时间成本高',
            '调试困难'
          ],
          suitableFor: ['关键应用', '极限压缩需求', '充足开发资源', '长期项目'],
          expectedCompressionRatio: 6.5,
          expectedSpeedup: 4.2,
          expectedAccuracyLoss: 1.0,
          complexity: 'high',
          implementationTime: 48,
          score: 95,
          confidence: 85
        }
      ],
      reasoning: `基于您的需求分析，我们推荐采用分阶段的压缩策略：

1. **首选方案：INT8量化**
   - 实现简单，效果明显
   - 满足您的移动端部署需求
   - 在${requirements.latencyRequirement || 50}ms延迟要求内表现良好

2. **次选方案：结构化剪枝**
   - 硬件兼容性优秀
   - 可与量化技术结合
   - 适合您的${requirements.modelType || 'CNN'}模型类型

3. **高级方案：混合策略**
   - 如果需要极限压缩，可考虑此方案
   - 需要投入更多开发资源
   - 适合长期项目规划`,

      tradeOffs: [
        {
          factor: '压缩比 vs 精度',
          impact: 'high',
          description: '更高的压缩比通常意味着更多的精度损失，需要在两者之间找到平衡点'
        },
        {
          factor: '实现复杂度 vs 效果',
          impact: 'medium',
          description: '简单的方法实现快但效果有限，复杂的方法效果好但开发成本高'
        },
        {
          factor: '通用性 vs 定制化',
          impact: 'medium',
          description: '通用方法适用性广但可能不是最优，定制化方法效果好但移植性差'
        }
      ],

      timeline: [
        {
          phase: '准备阶段',
          duration: 3,
          tasks: ['收集校准数据', '环境搭建', '基准测试']
        },
        {
          phase: '量化实施',
          duration: 5,
          tasks: ['模型量化', '精度验证', '性能测试']
        },
        {
          phase: '优化调试',
          duration: 2,
          tasks: ['精度优化', '性能调优', '最终验证']
        }
      ]
    })
  }, [requirements])

  const getDeviceIcon = (device: string) => {
    switch (device) {
      case 'mobile': return <MobileOutlined />
      case 'edge': return <WifiOutlined />
      case 'cloud': return <CloudOutlined />
      case 'embedded': return <DesktopOutlined />
      default: return <DesktopOutlined />
    }
  }

  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case 'low': return '#52c41a'
      case 'medium': return '#faad14'
      case 'high': return '#ff4d4f'
      default: return '#d9d9d9'
    }
  }

  const getStrategyTypeColor = (type: string) => {
    switch (type) {
      case 'quantization': return '#1890ff'
      case 'pruning': return '#722ed1'
      case 'distillation': return '#52c41a'
      case 'mixed': return '#fa8c16'
      default: return '#d9d9d9'
    }
  }

  const handleGenerateRecommendation = async () => {
    setLoading(true)
    try {
      const values = await form.validateFields()
      setRequirements({ ...requirements, ...values })
      
      // 模拟推荐生成过程
      await new Promise(resolve => setTimeout(resolve, 2000))
      
      message.success('推荐策略已生成！')
      setActiveTab('recommendations')
    } catch (error) {
      console.error('生成推荐失败:', error)
    } finally {
      setLoading(false)
    }
  }

  const strategyColumns = [
    {
      title: '策略名称',
      key: 'name',
      render: (record: CompressionStrategy) => (
        <div>
          <Space>
            <Avatar 
              size="small" 
              style={{ backgroundColor: getStrategyTypeColor(record.type) }}
              icon={record.type === 'quantization' ? <DatabaseOutlined /> : 
                   record.type === 'pruning' ? <SwapOutlined /> :
                   record.type === 'distillation' ? <ExperimentOutlined /> : 
                   <SettingOutlined />}
            />
            <Text strong>{record.name}</Text>
          </Space>
          <div style={{ fontSize: '12px', color: '#666', marginTop: '4px' }}>
            {record.description}
          </div>
        </div>
      )
    },
    {
      title: '类型',
      key: 'type',
      render: (record: CompressionStrategy) => (
        <Tag color={getStrategyTypeColor(record.type)}>
          {record.type.toUpperCase()}
        </Tag>
      )
    },
    {
      title: '压缩比',
      key: 'compression',
      render: (record: CompressionStrategy) => (
        <Text strong style={{ color: '#722ed1' }}>
          {record.expectedCompressionRatio.toFixed(1)}x
        </Text>
      )
    },
    {
      title: '加速比',
      key: 'speedup',
      render: (record: CompressionStrategy) => (
        <Text strong style={{ color: '#1890ff' }}>
          {record.expectedSpeedup.toFixed(1)}x
        </Text>
      )
    },
    {
      title: '精度损失',
      key: 'accuracy',
      render: (record: CompressionStrategy) => (
        <Text style={{ color: record.expectedAccuracyLoss > 2 ? '#ff4d4f' : '#52c41a' }}>
          {record.expectedAccuracyLoss.toFixed(1)}%
        </Text>
      )
    },
    {
      title: '复杂度',
      key: 'complexity',
      render: (record: CompressionStrategy) => (
        <Tag color={getComplexityColor(record.complexity)}>
          {record.complexity.toUpperCase()}
        </Tag>
      )
    },
    {
      title: '推荐度',
      key: 'score',
      render: (record: CompressionStrategy) => (
        <div>
          <Progress
            percent={record.score}
            size="small"
            status={record.score >= 90 ? 'success' : record.score >= 80 ? 'normal' : 'exception'}
          />
          <Text style={{ fontSize: '12px' }}>
            置信度: {record.confidence}%
          </Text>
        </div>
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: CompressionStrategy) => (
        <Space>
          <Button
            type="link"
            size="small"
            icon={<EyeOutlined />}
            onClick={() => {
              setSelectedStrategy(record)
              setStrategyModalVisible(true)
            }}
          >
            详情
          </Button>
          <Button
            type="link"
            size="small"
            icon={<PlayCircleOutlined />}
          >
            应用
          </Button>
        </Space>
      )
    }
  ]

  const renderWizard = () => (
    <Card title="智能压缩策略推荐向导">
      <Steps current={currentStep} style={{ marginBottom: '24px' }}>
        <Step title="模型信息" icon={<DatabaseOutlined />} />
        <Step title="部署需求" icon={<AimOutlined />} />
        <Step title="性能要求" icon={<ThunderboltOutlined />} />
        <Step title="生成推荐" icon={<RocketOutlined />} />
      </Steps>

      <Form form={form} layout="vertical">
        {currentStep === 0 && (
          <Row gutter={[16, 16]}>
            <Col span={12}>
              <Form.Item
                name="modelType"
                label="模型类型"
                rules={[{ required: true, message: '请选择模型类型' }]}
              >
                <Select placeholder="选择您的模型类型">
                  <Option value="cnn">卷积神经网络 (CNN)</Option>
                  <Option value="transformer">Transformer</Option>
                  <Option value="rnn">循环神经网络 (RNN)</Option>
                  <Option value="gan">生成对抗网络 (GAN)</Option>
                  <Option value="other">其他</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="modelSize"
                label="当前模型大小 (MB)"
                rules={[{ required: true, message: '请输入模型大小' }]}
              >
                <InputNumber min={1} max={10000} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="accuracy"
                label="当前模型精度 (%)"
                rules={[{ required: true, message: '请输入模型精度' }]}
              >
                <InputNumber min={0} max={100} precision={1} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="inputShape"
                label="输入形状"
                rules={[{ required: true, message: '请输入输入形状' }]}
              >
                <Input placeholder="例如: 224x224x3" />
              </Form.Item>
            </Col>
          </Row>
        )}

        {currentStep === 1 && (
          <Row gutter={[16, 16]}>
            <Col span={24}>
              <Form.Item
                name="targetDevice"
                label="目标部署设备"
                rules={[{ required: true, message: '请选择部署设备' }]}
              >
                <Radio.Group>
                  <Space direction="vertical">
                    <Radio value="mobile">
                      <Space>
                        <MobileOutlined />
                        <div>
                          <Text strong>移动端</Text>
                          <div style={{ fontSize: '12px', color: '#666' }}>
                            Android/iOS设备，内存和计算资源受限
                          </div>
                        </div>
                      </Space>
                    </Radio>
                    <Radio value="edge">
                      <Space>
                        <WifiOutlined />
                        <div>
                          <Text strong>边缘设备</Text>
                          <div style={{ fontSize: '12px', color: '#666' }}>
                            IoT设备、嵌入式系统，功耗敏感
                          </div>
                        </div>
                      </Space>
                    </Radio>
                    <Radio value="cloud">
                      <Space>
                        <CloudOutlined />
                        <div>
                          <Text strong>云端服务</Text>
                          <div style={{ fontSize: '12px', color: '#666' }}>
                            服务器部署，资源相对充足，注重吞吐量
                          </div>
                        </div>
                      </Space>
                    </Radio>
                    <Radio value="embedded">
                      <Space>
                        <DesktopOutlined />
                        <div>
                          <Text strong>嵌入式设备</Text>
                          <div style={{ fontSize: '12px', color: '#666' }}>
                            专用硬件，严格的资源约束
                          </div>
                        </div>
                      </Space>
                    </Radio>
                  </Space>
                </Radio.Group>
              </Form.Item>
            </Col>
          </Row>
        )}

        {currentStep === 2 && (
          <Row gutter={[16, 16]}>
            <Col span={12}>
              <Form.Item
                name="latencyRequirement"
                label="延迟要求 (ms)"
                rules={[{ required: true, message: '请输入延迟要求' }]}
              >
                <InputNumber min={1} max={1000} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="memoryLimit"
                label="内存限制 (MB)"
                rules={[{ required: true, message: '请输入内存限制' }]}
              >
                <InputNumber min={1} max={10000} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="powerConstraint"
                label="功耗约束 (W)"
              >
                <InputNumber min={0.1} max={500} precision={1} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="batchSize"
                label="批处理大小"
                rules={[{ required: true, message: '请输入批处理大小' }]}
              >
                <InputNumber min={1} max={512} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
          </Row>
        )}

        {currentStep === 3 && (
          <div style={{ textAlign: 'center', padding: '40px 0' }}>
            <Avatar size={80} icon={<BulbOutlined />} style={{ backgroundColor: '#1890ff', marginBottom: '16px' }} />
            <Title level={3}>准备生成智能推荐</Title>
            <Paragraph style={{ fontSize: '16px', color: '#666' }}>
              基于您提供的信息，我们将分析最适合的压缩策略
            </Paragraph>
            
            <Alert
              message="推荐将包含"
              description={
                <ul style={{ textAlign: 'left', margin: 0 }}>
                  <li>个性化的压缩策略组合</li>
                  <li>预期的性能提升效果</li>
                  <li>实施难度和时间评估</li>
                  <li>详细的实施指导</li>
                </ul>
              }
              type="info"
              showIcon
              style={{ marginBottom: '24px', textAlign: 'left' }}
            />

            <Button
              type="primary"
              size="large"
              loading={loading}
              onClick={handleGenerateRecommendation}
              icon={<RocketOutlined />}
            >
              生成智能推荐
            </Button>
          </div>
        )}

        <div style={{ marginTop: '24px', textAlign: 'right' }}>
          <Space>
            {currentStep > 0 && (
              <Button onClick={() => setCurrentStep(currentStep - 1)}>
                上一步
              </Button>
            )}
            {currentStep < 3 && (
              <Button 
                type="primary" 
                onClick={() => setCurrentStep(currentStep + 1)}
              >
                下一步
              </Button>
            )}
          </Space>
        </div>
      </Form>
    </Card>
  )

  const renderRecommendations = () => {
    if (!recommendationResult) {
      return <Empty description="请先完成推荐向导" />
    }

    return (
      <div>
        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          <Col span={18}>
            <Alert
              message="智能推荐结果"
              description={recommendationResult.reasoning}
              type="success"
              showIcon
            />
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="推荐策略数量"
                value={recommendationResult.strategies.length}
                prefix={<BulbOutlined />}
              />
            </Card>
          </Col>
        </Row>

        <Card title="推荐策略详情" style={{ marginBottom: '16px' }}>
          <Table
            columns={strategyColumns}
            dataSource={recommendationResult.strategies}
            rowKey="id"
            pagination={false}
          />
        </Card>

        <Row gutter={[16, 16]}>
          <Col span={12}>
            <Card title="性能权衡分析" extra={<BarChartOutlined />}>
              <List
                dataSource={recommendationResult.tradeOffs}
                renderItem={tradeOff => (
                  <List.Item>
                    <List.Item.Meta
                      avatar={
                        <Avatar 
                          size="small"
                          style={{ 
                            backgroundColor: tradeOff.impact === 'high' ? '#ff4d4f' :
                                              tradeOff.impact === 'medium' ? '#faad14' : '#52c41a'
                          }}
                          icon={<ExclamationCircleOutlined />}
                        />
                      }
                      title={tradeOff.factor}
                      description={tradeOff.description}
                    />
                    <Tag color={
                      tradeOff.impact === 'high' ? 'red' :
                      tradeOff.impact === 'medium' ? 'orange' : 'green'
                    }>
                      {tradeOff.impact.toUpperCase()}
                    </Tag>
                  </List.Item>
                )}
              />
            </Card>
          </Col>

          <Col span={12}>
            <Card title="实施时间线" extra={<ClockCircleOutlined />}>
              <List
                dataSource={recommendationResult.timeline}
                renderItem={(phase, index) => (
                  <List.Item>
                    <List.Item.Meta
                      avatar={
                        <Avatar 
                          size="small"
                          style={{ backgroundColor: '#1890ff' }}
                        >
                          {index + 1}
                        </Avatar>
                      }
                      title={`${phase.phase} (${phase.duration}天)`}
                      description={
                        <div>
                          {phase.tasks.map(task => (
                            <Tag key={task} size="small" style={{ margin: '2px' }}>
                              {task}
                            </Tag>
                          ))}
                        </div>
                      }
                    />
                  </List.Item>
                )}
              />
            </Card>
          </Col>
        </Row>
      </div>
    )
  }

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2} style={{ margin: 0, color: '#1a1a1a' }}>
          <BulbOutlined style={{ marginRight: '12px', color: '#1890ff' }} />
          智能压缩策略推荐
        </Title>
        <Paragraph style={{ marginTop: '8px', color: '#666', fontSize: '16px' }}>
          基于AI分析您的模型特性和需求，推荐最适合的压缩策略组合
        </Paragraph>
      </div>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="智能推荐向导" key="wizard">
          {renderWizard()}
        </TabPane>

        <TabPane tab="推荐结果" key="recommendations">
          {renderRecommendations()}
        </TabPane>

        <TabPane tab="策略对比" key="comparison">
          <Card title="策略效果对比">
            {recommendationResult && (
              <div>
                <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
                  {recommendationResult.strategies.map(strategy => (
                    <Col span={6} key={strategy.id}>
                      <Card size="small" style={{ height: '100%' }}>
                        <div style={{ textAlign: 'center' }}>
                          <Avatar
                            size={48}
                            style={{ backgroundColor: getStrategyTypeColor(strategy.type) }}
                            icon={strategy.type === 'quantization' ? <DatabaseOutlined /> : 
                                 strategy.type === 'pruning' ? <SwapOutlined /> :
                                 strategy.type === 'distillation' ? <ExperimentOutlined /> : 
                                 <SettingOutlined />}
                          />
                          <div style={{ marginTop: '8px' }}>
                            <Text strong>{strategy.name}</Text>
                            <div style={{ marginTop: '8px' }}>
                              <Rate
                                value={Math.round(strategy.score / 20)}
                                disabled
                                style={{ fontSize: '12px' }}
                              />
                              <div style={{ fontSize: '12px', color: '#666' }}>
                                评分: {strategy.score}/100
                              </div>
                            </div>
                          </div>
                        </div>
                      </Card>
                    </Col>
                  ))}
                </Row>

                <Row gutter={[16, 16]}>
                  <Col span={8}>
                    <Card title="压缩比对比" size="small">
                      <Column
                        data={recommendationResult.strategies.map(s => ({
                          name: s.name,
                          value: s.expectedCompressionRatio
                        }))}
                        xField="name"
                        yField="value"
                        height={200}
                      />
                    </Card>
                  </Col>
                  <Col span={8}>
                    <Card title="加速比对比" size="small">
                      <Column
                        data={recommendationResult.strategies.map(s => ({
                          name: s.name,
                          value: s.expectedSpeedup
                        }))}
                        xField="name"
                        yField="value"
                        height={200}
                      />
                    </Card>
                  </Col>
                  <Col span={8}>
                    <Card title="精度损失对比" size="small">
                      <Column
                        data={recommendationResult.strategies.map(s => ({
                          name: s.name,
                          value: s.expectedAccuracyLoss
                        }))}
                        xField="name"
                        yField="value"
                        height={200}
                      />
                    </Card>
                  </Col>
                </Row>
              </div>
            )}
          </Card>
        </TabPane>
      </Tabs>

      {/* 策略详情模态框 */}
      <Modal
        title="压缩策略详情"
        open={strategyModalVisible}
        onCancel={() => setStrategyModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setStrategyModalVisible(false)}>
            关闭
          </Button>,
          <Button key="apply" type="primary" icon={<PlayCircleOutlined />}>
            应用此策略
          </Button>
        ]}
        width={800}
      >
        {selectedStrategy && (
          <div>
            <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
              <Col span={16}>
                <Title level={4}>{selectedStrategy.name}</Title>
                <Paragraph>{selectedStrategy.description}</Paragraph>
                <div>
                  <Text strong>适用场景: </Text>
                  {selectedStrategy.suitableFor.map(scenario => (
                    <Tag key={scenario} style={{ marginRight: '4px' }}>
                      {scenario}
                    </Tag>
                  ))}
                </div>
              </Col>
              <Col span={8}>
                <div style={{ textAlign: 'center' }}>
                  <Avatar
                    size={80}
                    style={{ backgroundColor: getStrategyTypeColor(selectedStrategy.type) }}
                    icon={selectedStrategy.type === 'quantization' ? <DatabaseOutlined /> : 
                         selectedStrategy.type === 'pruning' ? <SwapOutlined /> :
                         selectedStrategy.type === 'distillation' ? <ExperimentOutlined /> : 
                         <SettingOutlined />}
                  />
                  <div style={{ marginTop: '8px' }}>
                    <Rate
                      value={Math.round(selectedStrategy.score / 20)}
                      disabled
                    />
                    <div style={{ fontSize: '12px', color: '#666' }}>
                      推荐度: {selectedStrategy.score}/100
                    </div>
                  </div>
                </div>
              </Col>
            </Row>

            <Row gutter={[16, 16]}>
              <Col span={8}>
                <Statistic
                  title="预期压缩比"
                  value={selectedStrategy.expectedCompressionRatio}
                  precision={1}
                  suffix="x"
                  valueStyle={{ color: '#722ed1' }}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="预期加速比"
                  value={selectedStrategy.expectedSpeedup}
                  precision={1}
                  suffix="x"
                  valueStyle={{ color: '#1890ff' }}
                />
              </Col>
              <Col span={8}>
                <Statistic
                  title="预期精度损失"
                  value={selectedStrategy.expectedAccuracyLoss}
                  precision={1}
                  suffix="%"
                  valueStyle={{ color: '#ff4d4f' }}
                />
              </Col>
            </Row>

            <Divider />

            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Title level={5}>✅ 优势</Title>
                <List
                  size="small"
                  dataSource={selectedStrategy.advantages}
                  renderItem={advantage => (
                    <List.Item>
                      <CheckCircleOutlined style={{ color: '#52c41a', marginRight: '8px' }} />
                      {advantage}
                    </List.Item>
                  )}
                />
              </Col>
              <Col span={12}>
                <Title level={5}>⚠️ 注意事项</Title>
                <List
                  size="small"
                  dataSource={selectedStrategy.disadvantages}
                  renderItem={disadvantage => (
                    <List.Item>
                      <ExclamationCircleOutlined style={{ color: '#faad14', marginRight: '8px' }} />
                      {disadvantage}
                    </List.Item>
                  )}
                />
              </Col>
            </Row>

            <Divider />

            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Text strong>实现复杂度: </Text>
                <Tag color={getComplexityColor(selectedStrategy.complexity)}>
                  {selectedStrategy.complexity.toUpperCase()}
                </Tag>
              </Col>
              <Col span={12}>
                <Text strong>预计实施时间: </Text>
                <Text>{selectedStrategy.implementationTime} 小时</Text>
              </Col>
            </Row>
          </div>
        )}
      </Modal>
    </div>
  )
}

export default CompressionStrategyRecommendationPage