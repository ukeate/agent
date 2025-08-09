import React, { useState } from 'react'
import { Steps, Button, Typography, Space } from 'antd'
import { 
  BulbOutlined, 
  PlayCircleOutlined, 
  EyeOutlined,
  EyeInvisibleOutlined,
  ThunderboltOutlined 
} from '@ant-design/icons'
import { ReasoningStep } from '@/types'

const { Text } = Typography

interface ReasoningStepsProps {
  steps: ReasoningStep[]
}

const ReasoningSteps: React.FC<ReasoningStepsProps> = ({ steps }) => {
  const [showSteps, setShowSteps] = useState(false)

  if (!steps || steps.length === 0) {
    return null
  }

  const getStepIcon = (type: ReasoningStep['type']) => {
    switch (type) {
      case 'thought':
        return <BulbOutlined />
      case 'action':
        return <PlayCircleOutlined />
      case 'observation':
        return <EyeOutlined />
      default:
        return <ThunderboltOutlined />
    }
  }

  const getStepTitle = (type: ReasoningStep['type']) => {
    switch (type) {
      case 'thought':
        return '思考'
      case 'action':
        return '行动'
      case 'observation':
        return '观察'
      default:
        return '步骤'
    }
  }

  const stepItems = steps.map((step, index) => ({
    title: `${getStepTitle(step.type)} ${index + 1}`,
    description: step.content,
    icon: getStepIcon(step.type),
    status: 'finish' as const,
  }))

  return (
    <div className="mt-2 mb-3">
      <div className="flex items-center justify-between mb-2">
        <Space size="small">
          <ThunderboltOutlined className="text-purple-500" />
          <Text type="secondary" className="text-sm">
            推理过程 ({steps.length} 步)
          </Text>
        </Space>
        <Button
          type="text"
          size="small"
          icon={showSteps ? <EyeInvisibleOutlined /> : <EyeOutlined />}
          onClick={() => setShowSteps(!showSteps)}
          className="text-xs"
        >
          {showSteps ? '隐藏' : '展开'}
        </Button>
      </div>

      {showSteps && (
        <div className="bg-purple-50 p-3 rounded-lg">
          <Steps
            direction="vertical"
            size="small"
            items={stepItems}
            className="reasoning-steps"
          />
        </div>
      )}
    </div>
  )
}

export default ReasoningSteps