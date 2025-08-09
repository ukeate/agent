import React, { useState } from 'react'
import { Collapse, Tag, Typography, Space, Button } from 'antd'
import { 
  ToolOutlined, 
  CheckCircleOutlined, 
  CloseCircleOutlined, 
  LoadingOutlined,
  EyeOutlined,
  EyeInvisibleOutlined 
} from '@ant-design/icons'
import { ToolCall } from '@/types'

const { Text } = Typography
const { Panel } = Collapse

interface ToolCallDisplayProps {
  toolCalls: ToolCall[]
}

const ToolCallDisplay: React.FC<ToolCallDisplayProps> = ({ toolCalls }) => {
  const [showDetails, setShowDetails] = useState(false)

  if (!toolCalls || toolCalls.length === 0) {
    return null
  }

  const getStatusIcon = (status: ToolCall['status']) => {
    switch (status) {
      case 'success':
        return <CheckCircleOutlined className="text-green-500" />
      case 'error':
        return <CloseCircleOutlined className="text-red-500" />
      case 'pending':
        return <LoadingOutlined className="text-blue-500" />
      default:
        return <ToolOutlined />
    }
  }

  const getStatusColor = (status: ToolCall['status']) => {
    switch (status) {
      case 'success':
        return 'success'
      case 'error':
        return 'error'
      case 'pending':
        return 'processing'
      default:
        return 'default'
    }
  }

  return (
    <div className="mt-4 p-3 bg-gray-50 rounded-lg">
      <div className="flex items-center justify-between mb-3">
        <Space>
          <ToolOutlined className="text-blue-500" />
          <Text strong className="text-sm">
            工具调用 ({toolCalls.length})
          </Text>
        </Space>
        <Button
          type="text"
          size="small"
          icon={showDetails ? <EyeInvisibleOutlined /> : <EyeOutlined />}
          onClick={() => setShowDetails(!showDetails)}
        >
          {showDetails ? '隐藏详情' : '显示详情'}
        </Button>
      </div>

      <div className="space-y-2">
        {toolCalls.map((toolCall) => (
          <div key={toolCall.id} className="bg-white p-3 rounded border">
            <div className="flex items-center justify-between mb-2">
              <Space>
                {getStatusIcon(toolCall.status)}
                <Text strong>{toolCall.name}</Text>
                <Tag color={getStatusColor(toolCall.status)}>
                  {toolCall.status === 'success' ? '成功' : 
                   toolCall.status === 'error' ? '失败' : '执行中'}
                </Tag>
              </Space>
              <Text type="secondary" className="text-xs">
                {new Date(toolCall.timestamp).toLocaleTimeString()}
              </Text>
            </div>

            {showDetails && (
              <Collapse ghost size="small">
                <Panel header="参数" key="args">
                  <pre className="text-xs bg-gray-100 p-2 rounded overflow-x-auto">
                    {JSON.stringify(toolCall.args, null, 2)}
                  </pre>
                </Panel>
                {toolCall.result && (
                  <Panel header="结果" key="result">
                    <pre className="text-xs bg-gray-100 p-2 rounded overflow-x-auto">
                      {typeof toolCall.result === 'string' 
                        ? toolCall.result 
                        : JSON.stringify(toolCall.result, null, 2)}
                    </pre>
                  </Panel>
                )}
              </Collapse>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

export default ToolCallDisplay