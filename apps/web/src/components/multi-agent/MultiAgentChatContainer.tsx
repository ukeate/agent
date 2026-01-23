import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react'
import { Button, Space, Input } from 'antd'
import { ClearOutlined, ReloadOutlined } from '@ant-design/icons'
import { useMultiAgentStore } from '../../stores/multiAgentStore'
import { useMultiAgentWebSocket } from '../../hooks/useMultiAgentWebSocket'
import { useSmartAutoScroll } from '../../hooks/useSmartAutoScroll'
import { GroupChatMessages, AgentTurnIndicator } from './GroupChatMessages'
import { SessionControls } from './SessionControls'
import { AgentAvatar } from './AgentAvatar'
import { multiAgentService } from '../../services/multiAgentService'

import { logger } from '../../utils/logger'
interface MultiAgentChatContainerProps {
  className?: string
}

export const MultiAgentChatContainer: React.FC<
  MultiAgentChatContainerProps
> = ({ className = '' }) => {
  const {
    // 状态
    agents,
    currentSession,
    currentMessages,
    currentSpeaker,
    loading,
    error,

    // Actions
    setCurrentSession,
    clearMessages,
    setError,
    setAgents,
    loadSessionHistory,
    createConversation,
    startConversation,
    pauseConversation,
    resumeConversation,
    terminateConversation,
  } = useMultiAgentStore()

  const [initialMessage, setInitialMessage] = useState('')
  const [selectedAgents, setSelectedAgents] = useState<string[]>([])
  const [agentKeyword, setAgentKeyword] = useState('')
  const [agentsLoading, setAgentsLoading] = useState(false)
  const hasInitializedSelectionRef = useRef(false)
  const isWebsocketSession = !!currentSession?.session_id?.startsWith('session-')
  // WebSocket集成
  const { connected: wsConnected, sendMessage: sendWsMessage } =
    useMultiAgentWebSocket({
      sessionId: currentSession?.session_id,
      enabled: !!currentSession && isWebsocketSession,
    })

  // 智能自动滚动
  const { containerRef } = useSmartAutoScroll({
    messages: currentMessages,
    enabled: !!currentSession,
    threshold: 100,
    behavior: 'smooth',
    resetKey: currentSession?.session_id,
  })

  // 从API加载Agent
  const loadAgents = useCallback(async () => {
    try {
      setAgentsLoading(true)
      const agentList = await multiAgentService.listAgents()
      logger.log('从API加载到Agent:', agentList)
      setAgents(agentList)
      setSelectedAgents(prev => {
        if (agentList.length === 0) return []
        if (!hasInitializedSelectionRef.current) {
          hasInitializedSelectionRef.current = true
          return agentList.map(agent => agent.id)
        }
        return prev.filter(id => agentList.some(agent => agent.id === id))
      })
      setError(null)
    } catch (error) {
      logger.error('加载Agent失败:', error)
      setError(error instanceof Error ? error.message : '加载Agent失败')
    } finally {
      setAgentsLoading(false)
    }
  }, [setAgents, setError])

  useEffect(() => {
    loadAgents()
  }, [loadAgents])

  // 加载会话消息历史
  const handleLoadConversationHistory = async (conversationId: string) => {
    try {
      await loadSessionHistory(conversationId)
    } catch (error) {
      logger.error('加载对话历史失败:', error)
    }
  }

  const handleCreateConversation = async () => {
    if (selectedAgents.length === 0) {
      setError('请选择至少一个Agent参与对话')
      return
    }

    if (!initialMessage.trim()) {
      setError('请输入初始讨论话题')
      return
    }

    try {
      await createConversation(selectedAgents, initialMessage)
      setInitialMessage('')

      logger.log('对话创建完成，会话将自动启动流式响应...')
    } catch (error) {
      logger.error('创建对话失败:', error)
    }
  }

  const handleStartConversation = async () => {
    if (!currentSession || !initialMessage.trim()) {
      setError('请输入初始消息')
      return
    }

    const trimmedMessage = initialMessage.trim()
    const participants = Array.from(
      new Set(
        currentSession.participants
          .map(participant => participant.role)
          .filter(Boolean)
      )
    )
    if (participants.length === 0) {
      setError('未找到参与者角色')
      return
    }

    const sent = sendWsMessage({
      type: 'start_conversation',
      data: {
        message: trimmedMessage,
        participants,
      },
    })
    if (!sent) return

    try {
      await startConversation(currentSession.session_id, trimmedMessage)
      setInitialMessage('')
    } catch (error) {
      logger.error('启动对话失败:', error)
    }
  }

  const handleClearConversation = () => {
    // 清空当前对话消息
    clearMessages()

    // 如果有活跃会话，终止它
    if (
      currentSession &&
      (currentSession.status === 'active' || currentSession.status === 'paused')
    ) {
      terminateConversation(currentSession.session_id, '用户清空对话')
    }

    // 重置会话状态
    setCurrentSession(null)
    setError(null)

    logger.log('对话已清空')
  }

  const filteredAgents = useMemo(() => {
    const normalized = agentKeyword.trim().toLowerCase()
    if (!normalized) return agents
    return agents.filter(agent => {
      const combined = [
        agent.name,
        agent.role,
        (agent.capabilities || []).join(' '),
      ]
        .join(' ')
        .toLowerCase()
      return combined.includes(normalized)
    })
  }, [agents, agentKeyword])

  const handleSelectAllAgents = () => {
    if (agents.length === 0) return
    setSelectedAgents(agents.map(agent => agent.id))
  }

  const handleClearSelection = () => {
    setSelectedAgents([])
  }

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* 错误提示 */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg mb-4">
          <div className="flex items-center gap-2">
            <span>⚠️</span>
            <span>{error}</span>
            <button
              onClick={() => setError(null)}
              className="ml-auto text-red-500 hover:text-red-700"
            >
              ✕
            </button>
          </div>
        </div>
      )}

      <div className="flex-1 flex gap-4 min-h-0">
        {/* 主对话区域 */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* 对话历史 */}
          <div className="flex-1 flex flex-col bg-gray-50 rounded-lg overflow-hidden">
            {/* 工具栏 - 始终显示 */}
            <div className="border-b border-gray-200 bg-white px-4 py-3 rounded-t-lg">
              <Space>
                <Button
                  type="text"
                  size="small"
                  icon={<ClearOutlined />}
                  onClick={handleClearConversation}
                  className="text-gray-500 hover:text-red-500"
                >
                  {currentMessages.length > 0 ? '清空对话' : '新建对话'}
                </Button>
                {currentSession && (
                  <Button
                    type="text"
                    size="small"
                    onClick={() =>
                      handleLoadConversationHistory(currentSession.session_id)
                    }
                    loading={loading}
                    disabled={loading}
                    className="text-blue-500 hover:text-blue-700"
                  >
                    📜 加载历史
                  </Button>
                )}
                {currentMessages.length > 0 && (
                  <>
                    <span className="text-gray-400 text-xs">
                      共 {currentMessages.length} 条消息
                    </span>
                    {currentSession && (
                      <span className="text-gray-400 text-xs">
                        第 {currentSession.round_count} 轮
                      </span>
                    )}
                  </>
                )}
              </Space>
            </div>

            {/* 消息区域 */}
            <div
              ref={containerRef}
              className="flex-1 overflow-y-auto p-4 bg-gray-50 custom-scrollbar"
            >
              <GroupChatMessages
                messages={currentMessages}
                agents={agents}
                currentSpeaker={currentSpeaker || undefined}
              />
            </div>
          </div>

          {/* 消息输入区 */}
          <div className="mt-4 space-y-3">
            {!currentSession ? (
              /* 创建对话 */
              <div className="bg-white border border-gray-200 rounded-lg p-4">
                <h3 className="text-lg font-medium text-gray-900 mb-3">
                  创建多智能体对话 / 创建Multi-Agent对话
                </h3>

                {/* 智能体选择 */}
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    选择参与Agent
                  </label>
                  <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
                    <Input
                      value={agentKeyword}
                      onChange={e => setAgentKeyword(e.target.value)}
                      placeholder="搜索智能体或能力"
                      allowClear
                      size="small"
                      className="w-full sm:w-56"
                    />
                    <Space size={8} wrap>
                      <Button
                        size="small"
                        onClick={handleSelectAllAgents}
                        disabled={agentsLoading || agents.length === 0}
                      >
                        全选
                      </Button>
                      <Button
                        size="small"
                        onClick={handleClearSelection}
                        disabled={agentsLoading || selectedAgents.length === 0}
                      >
                        清空
                      </Button>
                      <Button
                        size="small"
                        icon={<ReloadOutlined />}
                        onClick={loadAgents}
                        loading={agentsLoading}
                        disabled={agentsLoading}
                      >
                        刷新
                      </Button>
                    </Space>
                  </div>
                  {filteredAgents.length === 0 ? (
                    <div className="text-sm text-gray-500">
                      {agentsLoading
                        ? '正在加载智能体...'
                        : '未找到匹配的智能体'}
                    </div>
                  ) : (
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                    {filteredAgents.map(agent => (
                      <label
                        key={agent.id}
                        className={`
                          flex flex-col gap-2 p-3 rounded-lg border cursor-pointer
                          transition-colors
                          ${
                            selectedAgents.includes(agent.id)
                              ? 'border-blue-500 bg-blue-50'
                              : 'border-gray-200 hover:bg-gray-50'
                          }
                        `}
                      >
                        <div className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={selectedAgents.includes(agent.id)}
                            onChange={e => {
                              if (e.target.checked) {
                                setSelectedAgents([...selectedAgents, agent.id])
                              } else {
                                setSelectedAgents(
                                  selectedAgents.filter(id => id !== agent.id)
                                )
                              }
                            }}
                            className="text-blue-600"
                          />
                          <AgentAvatar agent={agent} size="sm" />
                          <div className="flex-1 min-w-0">
                            <div className="text-sm font-medium">
                              {agent.name}
                            </div>
                            <div className="text-xs text-gray-500">
                              {agent.role}
                            </div>
                          </div>
                          {/* 状态指示器 */}
                          <div
                            className={`w-2 h-2 rounded-full ${
                              agent.status === 'active'
                                ? 'bg-green-400'
                                : 'bg-gray-400'
                            }`}
                          />
                        </div>

                        {/* 能力展示 */}
                        {agent.capabilities &&
                          agent.capabilities.length > 0 && (
                            <div className="mt-1">
                              <div className="text-xs text-gray-600 mb-1">
                                能力:
                              </div>
                              <div className="flex flex-wrap gap-1">
                                {agent.capabilities.map((capability, idx) => (
                                  <span
                                    key={idx}
                                    className="inline-block px-2 py-0.5 text-xs bg-gray-100 text-gray-700 rounded"
                                  >
                                    {capability}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}

                        {/* 最后活跃时间 */}
                        {agent.last_active && (
                          <div className="text-xs text-gray-500">
                            最后活跃:{' '}
                            {new Date(agent.last_active).toLocaleString(
                              'zh-CN'
                            )}
                          </div>
                        )}
                      </label>
                    ))}
                  </div>
                  )}
                </div>

                {/* 初始话题 */}
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    初始讨论话题
                  </label>
                  <textarea
                    value={initialMessage}
                    onChange={e => setInitialMessage(e.target.value)}
                    placeholder="请输入要讨论的话题或问题..."
                    rows={3}
                    className="
                      w-full px-3 py-2 border border-gray-300 rounded-md
                      focus:outline-none focus:ring-2 focus:ring-blue-500
                      resize-none
                    "
                  />
                </div>

                <button
                  onClick={handleCreateConversation}
                  disabled={
                    loading ||
                    agentsLoading ||
                    selectedAgents.length === 0 ||
                    !initialMessage.trim()
                  }
                  className="
                    w-full bg-blue-500 hover:bg-blue-600 text-white
                    px-4 py-2 rounded-md font-medium
                    disabled:opacity-50 disabled:cursor-not-allowed
                    transition-colors
                  "
                >
                  {loading ? '创建中...' : '开始多智能体讨论'}
                </button>
              </div>
            ) : currentSession.status === 'created' ? (
              /* 启动对话 */
              <div className="bg-white border border-gray-200 rounded-lg p-4">
                <h3 className="text-lg font-medium text-gray-900 mb-3">
                  启动对话
                </h3>
                <div className="flex gap-2">
                  <textarea
                    value={initialMessage}
                    onChange={e => setInitialMessage(e.target.value)}
                    placeholder="输入消息开始对话..."
                    rows={2}
                    className="
                      flex-1 px-3 py-2 border border-gray-300 rounded-md
                      focus:outline-none focus:ring-2 focus:ring-blue-500
                      resize-none
                    "
                  />
                  <button
                    onClick={handleStartConversation}
                    disabled={loading || !initialMessage.trim()}
                    className="
                      bg-green-500 hover:bg-green-600 text-white
                      px-4 py-2 rounded-md font-medium
                      disabled:opacity-50 disabled:cursor-not-allowed
                      transition-colors whitespace-nowrap
                    "
                  >
                    {loading ? '启动中...' : '💬 启动'}
                  </button>
                </div>
              </div>
            ) : currentSession.status === 'active' ? (
              /* 对话进行中状态提示 */
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <div className="flex items-center gap-2 text-blue-700">
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
                  <span className="font-medium">
                    对话进行中 - Agent正在协作讨论
                  </span>
                </div>
              </div>
            ) : (
              /* 对话已完成或其他状态 */
              <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-gray-600">
                    <div className="w-2 h-2 bg-gray-400 rounded-full" />
                    <span className="font-medium">
                      对话已
                      {currentSession.status === 'completed'
                        ? '完成'
                        : currentSession.status === 'paused'
                          ? '暂停'
                          : currentSession.status === 'terminated'
                            ? '终止'
                            : '结束'}
                    </span>
                  </div>
                  <button
                    onClick={() => {
                      setCurrentSession(null)
                      clearMessages()
                      setError(null)
                    }}
                    className="
                      bg-blue-500 hover:bg-blue-600 text-white
                      px-4 py-2 rounded-md text-sm font-medium
                      transition-colors
                    "
                  >
                    🆕 开始新对话
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* 侧边栏 */}
        <div className="w-72 space-y-4 flex-shrink-0">
          {/* 代理系统状态 */}
          <div className="bg-white border border-gray-200 rounded-lg p-4">
            <h3 className="text-sm font-medium text-gray-900 mb-3">系统状态</h3>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-600">总代理数:</span>
                <span className="font-medium">{agents.length}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-600">活跃代理:</span>
                <span className="font-medium text-green-600">
                  {agents.filter(a => a.status === 'active').length}
                </span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-600">选中代理:</span>
                <span className="font-medium text-blue-600">
                  {selectedAgents.length}
                </span>
              </div>
            </div>

            {/* 代理能力统计 */}
            {agents.length > 0 && (
              <div className="mt-4">
                <div className="text-sm font-medium text-gray-700 mb-2">
                  系统能力覆盖
                </div>
                <div className="flex flex-wrap gap-1">
                  {Array.from(
                    new Set(agents.flatMap(a => a.capabilities || []))
                  ).map((capability, idx) => (
                    <span
                      key={idx}
                      className="inline-block px-2 py-1 text-xs bg-blue-100 text-blue-700 rounded"
                    >
                      {capability}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* 参与者状态 */}
          <AgentTurnIndicator
            agents={agents.filter(
              a =>
                currentSession?.participants.some(p => p.name === a.name) ||
                selectedAgents.includes(a.id)
            )}
            currentSpeaker={currentSpeaker || undefined}
            currentRound={currentSession?.round_count}
          />

          {/* 会话控制 - 始终显示当前会话存在时 */}
          {currentSession && (
            <SessionControls
              session={currentSession}
              loading={loading}
              onPause={() => pauseConversation(currentSession.session_id)}
              onResume={() => resumeConversation(currentSession.session_id)}
              onTerminate={reason =>
                terminateConversation(currentSession.session_id, reason)
              }
            />
          )}

          {/* WebSocket状态 */}
          <div className="bg-white border border-gray-200 rounded-lg p-3">
            <div className="flex items-center gap-2 text-sm">
              <div
                className={`w-2 h-2 rounded-full ${
                  !currentSession
                    ? 'bg-gray-400'
                    : currentSession.status === 'completed' ||
                        currentSession.status === 'terminated'
                      ? 'bg-gray-400'
                      : wsConnected
                        ? 'bg-green-400'
                        : 'bg-orange-400'
                }`}
              />
              <span className="text-gray-600">
                实时连接:{' '}
                {!currentSession
                  ? '待机中'
                  : currentSession.status === 'completed' ||
                      currentSession.status === 'terminated'
                    ? '已断开'
                    : wsConnected
                      ? '已连接'
                      : '连接中...'}
              </span>
            </div>
            {currentSession && !wsConnected && (
              <div className="text-xs text-orange-600 mt-1">
                正在建立连接，请稍候...
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
