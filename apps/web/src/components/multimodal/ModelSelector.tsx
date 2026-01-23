import React from 'react'
import {
  Select,
  Radio,
  Space,
  Card,
  Tag,
  Tooltip,
  Row,
  Col,
  Typography,
} from 'antd'
import {
  DollarOutlined,
  RocketOutlined,
  ThunderboltOutlined,
  BankOutlined,
  CrownOutlined,
  SafetyCertificateOutlined,
} from '@ant-design/icons'

const { Option } = Select
const { Text } = Typography

interface ModelConfig {
  name: string
  displayName: string
  capabilities: string[]
  maxTokens: number
  costPerKTokens: { input: number; output: number }
  bestFor: string[]
  supportsVision: boolean
  supportsFileUpload: boolean
  color: string
  icon: React.ReactNode
}

const modelConfigs: Record<string, ModelConfig> = {
  'gpt-4o': {
    name: 'gpt-4o',
    displayName: 'GPT-4o',
    capabilities: ['text', 'image', 'pdf'],
    maxTokens: 4096,
    costPerKTokens: { input: 5, output: 15 },
    bestFor: ['高质量输出', '复杂推理', 'PDF处理'],
    supportsVision: true,
    supportsFileUpload: true,
    color: 'blue',
    icon: <CrownOutlined />,
  },
  'gpt-4o-mini': {
    name: 'gpt-4o-mini',
    displayName: 'GPT-4o Mini',
    capabilities: ['text', 'image', 'pdf'],
    maxTokens: 16384,
    costPerKTokens: { input: 0.15, output: 0.6 },
    bestFor: ['性价比', '简单任务', '高吞吐量'],
    supportsVision: true,
    supportsFileUpload: true,
    color: 'green',
    icon: <DollarOutlined />,
  },
  'gpt-4o-2024-11-20': {
    name: 'gpt-4o-2024-11-20',
    displayName: 'GPT-4o Latest',
    capabilities: ['text', 'image', 'pdf'],
    maxTokens: 16384,
    costPerKTokens: { input: 2.75, output: 11 },
    bestFor: ['最新功能', '结构化输出'],
    supportsVision: true,
    supportsFileUpload: true,
    color: 'purple',
    icon: <RocketOutlined />,
  },
  'gpt-5': {
    name: 'gpt-5',
    displayName: 'GPT-5',
    capabilities: ['text', 'image', 'pdf', 'video'],
    maxTokens: 8192,
    costPerKTokens: { input: 12.5, output: 25 },
    bestFor: ['最强推理', '多模态', '视频分析'],
    supportsVision: true,
    supportsFileUpload: true,
    color: 'gold',
    icon: <SafetyCertificateOutlined />,
  },
  'gpt-5-mini': {
    name: 'gpt-5-mini',
    displayName: 'GPT-5 Mini',
    capabilities: ['text', 'image', 'pdf'],
    maxTokens: 16384,
    costPerKTokens: { input: 2.5, output: 5 },
    bestFor: ['平衡性能', '中等复杂度'],
    supportsVision: true,
    supportsFileUpload: true,
    color: 'orange',
    icon: <BankOutlined />,
  },
  'gpt-5-nano': {
    name: 'gpt-5-nano',
    displayName: 'GPT-5 Nano',
    capabilities: ['text', 'image'],
    maxTokens: 128000,
    costPerKTokens: { input: 0.05, output: 0.4 },
    bestFor: ['超低成本', '简单分类', '高容量'],
    supportsVision: true,
    supportsFileUpload: false,
    color: 'cyan',
    icon: <ThunderboltOutlined />,
  },
}

interface ModelSelectorProps {
  selectedModel: string
  onModelChange: (model: string) => void
  priority: string
  onPriorityChange: (priority: string) => void
  complexity: string
  onComplexityChange: (complexity: string) => void
}

const ModelSelector: React.FC<ModelSelectorProps> = ({
  selectedModel,
  onModelChange,
  priority,
  onPriorityChange,
  complexity,
  onComplexityChange,
}) => {
  const currentModel = modelConfigs[selectedModel] || modelConfigs['gpt-4o']

  return (
    <div className="space-y-4">
      {/* 模型选择 */}
      <div>
        <Text strong className="mb-2 block">
          选择模型：
        </Text>
        <Select
          value={selectedModel}
          onChange={onModelChange}
          style={{ width: '100%' }}
          size="large"
        >
          {Object.values(modelConfigs).map(model => (
            <Option key={model.name} value={model.name}>
              <div className="flex items-center justify-between">
                <span className="flex items-center">
                  {model.icon}
                  <span className="ml-2">{model.displayName}</span>
                </span>
                <div>
                  {model.capabilities.map(cap => (
                    <Tag
                      key={cap}
                      color={model.color}
                      style={{ marginLeft: 4 }}
                    >
                      {cap}
                    </Tag>
                  ))}
                </div>
              </div>
            </Option>
          ))}
        </Select>
      </div>

      {/* 模型信息展示 */}
      <Card size="small" className="bg-gray-50">
        <Row gutter={[16, 8]}>
          <Col span={12}>
            <Text type="secondary">输入成本：</Text>
            <Text strong> ${currentModel.costPerKTokens.input}/1K tokens</Text>
          </Col>
          <Col span={12}>
            <Text type="secondary">输出成本：</Text>
            <Text strong> ${currentModel.costPerKTokens.output}/1K tokens</Text>
          </Col>
          <Col span={12}>
            <Text type="secondary">最大Tokens：</Text>
            <Text strong> {currentModel.maxTokens.toLocaleString()}</Text>
          </Col>
          <Col span={12}>
            <Text type="secondary">视觉支持：</Text>
            <Text strong> {currentModel.supportsVision ? '✅' : '❌'}</Text>
          </Col>
          <Col span={24}>
            <Text type="secondary">最佳用途：</Text>
            <div className="mt-1">
              {currentModel.bestFor.map(use => (
                <Tag key={use} color={currentModel.color}>
                  {use}
                </Tag>
              ))}
            </div>
          </Col>
        </Row>
      </Card>

      {/* 优先级选择 */}
      <div>
        <Text strong className="mb-2 block">
          处理优先级：
        </Text>
        <Radio.Group
          value={priority}
          onChange={e => onPriorityChange(e.target.value)}
          buttonStyle="solid"
          style={{ width: '100%' }}
        >
          <Radio.Button
            value="cost"
            style={{ width: '25%', textAlign: 'center' }}
          >
            <Tooltip title="最低成本，使用经济型模型">
              <DollarOutlined /> 成本
            </Tooltip>
          </Radio.Button>
          <Radio.Button
            value="quality"
            style={{ width: '25%', textAlign: 'center' }}
          >
            <Tooltip title="最高质量，使用顶级模型">
              <CrownOutlined /> 质量
            </Tooltip>
          </Radio.Button>
          <Radio.Button
            value="speed"
            style={{ width: '25%', textAlign: 'center' }}
          >
            <Tooltip title="最快速度，使用轻量模型">
              <ThunderboltOutlined /> 速度
            </Tooltip>
          </Radio.Button>
          <Radio.Button
            value="balanced"
            style={{ width: '25%', textAlign: 'center' }}
          >
            <Tooltip title="平衡各项指标">
              <BankOutlined /> 平衡
            </Tooltip>
          </Radio.Button>
        </Radio.Group>
      </div>

      {/* 复杂度选择 */}
      <div>
        <Text strong className="mb-2 block">
          任务复杂度：
        </Text>
        <Radio.Group
          value={complexity}
          onChange={e => onComplexityChange(e.target.value)}
          style={{ width: '100%' }}
        >
          <Radio value="simple">简单（基础识别和提取）</Radio>
          <Radio value="medium">中等（详细分析和理解）</Radio>
          <Radio value="complex">复杂（深度推理和综合）</Radio>
        </Radio.Group>
      </div>

      {/* 智能推荐 */}
      <Card
        size="small"
        className="bg-blue-50 border-blue-200"
        title={
          <Text type="secondary">
            <RocketOutlined /> 智能推荐
          </Text>
        }
      >
        <Text>
          基于您选择的
          <Tag color="blue">
            {priority === 'cost'
              ? '成本优先'
              : priority === 'quality'
                ? '质量优先'
                : priority === 'speed'
                  ? '速度优先'
                  : '平衡模式'}
          </Tag>
          和
          <Tag color="orange">
            {complexity === 'simple'
              ? '简单任务'
              : complexity === 'complex'
                ? '复杂任务'
                : '中等任务'}
          </Tag>
          ，系统将自动选择最适合的模型进行处理。
        </Text>
      </Card>
    </div>
  )
}

export default ModelSelector
