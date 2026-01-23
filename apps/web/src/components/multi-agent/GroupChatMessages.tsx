import React from 'react'
import { MultiAgentMessage, Agent } from '../../stores/multiAgentStore'
import { AgentAvatar, RoleBadge } from './AgentAvatar'
import MarkdownRenderer from '../ui/MarkdownRenderer'

interface GroupChatMessagesProps {
  messages: MultiAgentMessage[]
  agents: Agent[]
  currentSpeaker?: string
  className?: string
}

export const GroupChatMessages: React.FC<GroupChatMessagesProps> = ({
  messages,
  agents,
  currentSpeaker,
  className = '',
}) => {
  // 根据发送者名称查找智能体信息
  const getAgentBySender = (senderName: string): Agent | undefined => {
    return agents.find(agent => agent.name === senderName)
  }

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString('zh-CN', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    })
  }

  return (
    <div className={`flex flex-col gap-4 ${className}`}>
      {messages.map(message => {
        const agent = getAgentBySender(message.sender)
        const isCurrentSpeaker = currentSpeaker === message.sender
        const isUserMessage = message.role === 'user'

        return (
          <div
            key={message.id}
            className={`
              flex gap-3 p-4 rounded-lg transition-all
              ${
                isCurrentSpeaker
                  ? 'bg-blue-50 border-2 border-blue-200 shadow-md'
                  : 'bg-white border border-gray-200 hover:shadow-sm'
              }
              ${isUserMessage ? 'bg-gray-50' : ''}
            `}
          >
            {/* 发送者头像 */}
            <div className="flex-shrink-0">
              {agent ? (
                <AgentAvatar
                  agent={agent}
                  size="md"
                  showStatus={isCurrentSpeaker}
                />
              ) : (
                <div className="w-12 h-12 bg-gray-300 rounded-full flex items-center justify-center text-white font-medium">
                  👤
                </div>
              )}
            </div>

            {/* 消息内容 */}
            <div className="flex-1 min-w-0">
              {/* 消息头部 */}
              <div className="flex items-center gap-2 mb-2">
                <div className="flex items-center gap-2">
                  <span className="font-medium text-gray-900">
                    {message.sender}
                  </span>
                  {agent && (
                    <RoleBadge
                      role={agent.role}
                      capabilities={agent.capabilities.slice(0, 2)}
                    />
                  )}
                </div>

                {isCurrentSpeaker && (
                  <span className="text-xs bg-blue-100 text-blue-600 px-2 py-1 rounded-full">
                    正在发言
                  </span>
                )}

                <span className="text-xs text-gray-500 ml-auto">
                  {formatTimestamp(message.timestamp)}
                </span>
              </div>

              {/* 消息正文 */}
              <div className="prose prose-sm max-w-none">
                <MarkdownRenderer content={message.content} />

                {/* 流式消息打字机效果指示器 */}
                {message.isStreaming && !message.streamingComplete && (
                  <div className="inline-flex items-center gap-1 ml-2">
                    <div className="flex space-x-1">
                      <div
                        className="w-1 h-1 bg-blue-400 rounded-full animate-bounce"
                        style={{ animationDelay: '0ms' }}
                      />
                      <div
                        className="w-1 h-1 bg-blue-400 rounded-full animate-bounce"
                        style={{ animationDelay: '150ms' }}
                      />
                      <div
                        className="w-1 h-1 bg-blue-400 rounded-full animate-bounce"
                        style={{ animationDelay: '300ms' }}
                      />
                    </div>
                    <span className="text-xs text-blue-500 ml-1">
                      正在输入...
                    </span>
                  </div>
                )}

                {/* 流式消息完成指示器 */}
                {message.streamingComplete && (
                  <div className="inline-flex items-center gap-1 ml-2 text-xs text-green-500">
                    <span>✓</span>
                    <span>完成</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        )
      })}

      {/* 空状态 */}
      {messages.length === 0 && (
        <section className="text-center py-12 text-gray-500">
          <div className="text-4xl mb-4">💬</div>
          <div className="text-lg font-medium mb-2">多智能体对话还未开始</div>
          <div className="text-sm">发送消息启动智能体协作讨论</div>
        </section>
      )}
    </div>
  )
}

// 发言轮次指示器组件
interface AgentTurnIndicatorProps {
  agents: Agent[]
  currentSpeaker?: string
  currentRound?: number
  className?: string
}

export const AgentTurnIndicator: React.FC<AgentTurnIndicatorProps> = ({
  agents,
  currentSpeaker,
  currentRound = 0,
  className = '',
}) => {
  return (
    <div
      className={`bg-white border border-gray-200 rounded-lg p-4 ${className}`}
    >
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-900">参与者状态</h3>
        <div className="text-xs text-gray-500">第 {currentRound} 轮</div>
      </div>

      <div className="space-y-2">
        {agents.map(agent => {
          const isCurrent = currentSpeaker === agent.name

          return (
            <div
              key={agent.id}
              className={`
                flex items-center gap-3 p-2 rounded-md transition-all
                ${
                  isCurrent
                    ? 'bg-blue-50 border border-blue-200'
                    : 'hover:bg-gray-50'
                }
              `}
            >
              <AgentAvatar agent={agent} size="sm" showStatus={true} />

              <div className="flex-1">
                <div className="text-sm font-medium text-gray-900">
                  {agent.name}
                </div>
                <div className="text-xs text-gray-500">{agent.role}</div>
              </div>

              {isCurrent && (
                <div className="flex items-center gap-1 text-xs text-blue-600">
                  <div className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
                  发言中
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
