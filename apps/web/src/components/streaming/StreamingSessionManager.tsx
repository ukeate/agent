/**
 * 流式会话管理器
 *
 * 提供创建、管理和监控流式处理会话的界面
 */

import React, { useState, useEffect, useRef } from 'react'
import {
  streamingService,
  SessionMetrics,
} from '../../services/streamingService'

import { logger } from '../../utils/logger'
interface StreamingSessionManagerProps {
  onSessionCreated?: (sessionId: string) => void
  onSessionStopped?: (sessionId: string) => void
}

interface StreamEvent {
  type: string
  data: any
  metadata?: Record<string, any>
}

export const StreamingSessionManager: React.FC<
  StreamingSessionManagerProps
> = ({ onSessionCreated, onSessionStopped }) => {
  const [sessions, setSessions] = useState<Record<string, SessionMetrics>>({})
  const [createForm, setCreateForm] = useState({
    agent_id: 'test_agent',
    message: 'Hello streaming!',
    buffer_size: 100,
  })
  const [activeStreams, setActiveStreams] = useState<
    Record<
      string,
      {
        connection: EventSource | WebSocket
        messages: StreamEvent[]
        status: 'connected' | 'disconnected' | 'error'
      }
    >
  >({})
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const streamContainerRef = useRef<HTMLDivElement>(null)
  const activeStreamsRef = useRef(activeStreams)

  useEffect(() => {
    activeStreamsRef.current = activeStreams
  }, [activeStreams])

  // 获取会话列表
  const fetchSessions = async () => {
    try {
      const response = await streamingService.getSessions()
      setSessions(response.sessions)
    } catch (error) {
      logger.error('获取会话列表失败:', error)
      setError(error instanceof Error ? error.message : '获取会话列表失败')
    }
  }

  useEffect(() => {
    fetchSessions()
    const interval = setInterval(fetchSessions, 5000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    return () => {
      Object.values(activeStreamsRef.current).forEach(stream => {
        if (stream.connection instanceof EventSource) {
          stream.connection.close()
        } else if (stream.connection instanceof WebSocket) {
          stream.connection.close()
        }
      })
    }
  }, [])

  // 创建新会话
  const handleCreateSession = async () => {
    setLoading(true)
    setError(null)

    try {
      const response = await streamingService.createSession(createForm)

      if (onSessionCreated) {
        onSessionCreated(response.session_id)
      }

      // 刷新会话列表
      await fetchSessions()

      // 重置表单
      setCreateForm({
        agent_id: 'test_agent',
        message: 'Hello streaming!',
        buffer_size: 100,
      })
    } catch (error) {
      logger.error('创建会话失败:', error)
      setError(error instanceof Error ? error.message : '创建会话失败')
    } finally {
      setLoading(false)
    }
  }

  // 停止会话
  const handleStopSession = async (sessionId: string) => {
    try {
      await streamingService.stopSession(sessionId)

      // 断开流连接
      if (activeStreams[sessionId]) {
        const stream = activeStreams[sessionId]
        if (stream.connection instanceof EventSource) {
          stream.connection.close()
        } else if (stream.connection instanceof WebSocket) {
          stream.connection.close()
        }

        setActiveStreams(prev => {
          const newStreams = { ...prev }
          delete newStreams[sessionId]
          return newStreams
        })
      }

      if (onSessionStopped) {
        onSessionStopped(sessionId)
      }

      await fetchSessions()
    } catch (error) {
      logger.error('停止会话失败:', error)
      setError(error instanceof Error ? error.message : '停止会话失败')
    }
  }

  // 开始SSE流
  const startSSEStream = (sessionId: string, message: string) => {
    if (activeStreams[sessionId]) {
      return // 已经有活跃连接
    }

    const eventSource = streamingService.createSSEConnection(sessionId, message)

    setActiveStreams(prev => ({
      ...prev,
      [sessionId]: {
        connection: eventSource,
        messages: [],
        status: 'connected',
      },
    }))

    eventSource.onmessage = event => {
      try {
        const data = JSON.parse(event.data)
        setActiveStreams(prev => ({
          ...prev,
          [sessionId]: {
            ...prev[sessionId],
            messages: [...(prev[sessionId]?.messages || []), data],
          },
        }))

        // 自动滚动到底部
        if (streamContainerRef.current) {
          streamContainerRef.current.scrollTop =
            streamContainerRef.current.scrollHeight
        }
      } catch (error) {
        logger.error('解析SSE消息失败:', error)
      }
    }

    eventSource.onerror = error => {
      logger.error('SSE连接错误:', error)
      eventSource.close()
      setActiveStreams(prev => ({
        ...prev,
        [sessionId]: {
          ...prev[sessionId],
          status: 'error',
        },
      }))
    }

    eventSource.onopen = () => {
      setActiveStreams(prev => ({
        ...prev,
        [sessionId]: {
          ...prev[sessionId],
          status: 'connected',
        },
      }))
    }
  }

  // 开始WebSocket流
  const startWebSocketStream = (sessionId: string) => {
    if (activeStreams[sessionId]) {
      return // 已经有活跃连接
    }

    const ws = streamingService.createWebSocketConnection(sessionId)

    setActiveStreams(prev => ({
      ...prev,
      [sessionId]: {
        connection: ws,
        messages: [],
        status: 'connected',
      },
    }))

    ws.onopen = () => {
      setActiveStreams(prev => ({
        ...prev,
        [sessionId]: {
          ...prev[sessionId],
          status: 'connected',
        },
      }))

      // 发送初始消息
      ws.send(createForm.message)
    }

    ws.onmessage = event => {
      try {
        const data = JSON.parse(event.data)
        setActiveStreams(prev => ({
          ...prev,
          [sessionId]: {
            ...prev[sessionId],
            messages: [...(prev[sessionId]?.messages || []), data],
          },
        }))

        // 自动滚动到底部
        if (streamContainerRef.current) {
          streamContainerRef.current.scrollTop =
            streamContainerRef.current.scrollHeight
        }
      } catch (error) {
        logger.error('解析WebSocket消息失败:', error)
      }
    }

    ws.onerror = error => {
      logger.error('WebSocket连接错误:', error)
      setActiveStreams(prev => ({
        ...prev,
        [sessionId]: {
          ...prev[sessionId],
          status: 'error',
        },
      }))
    }

    ws.onclose = () => {
      setActiveStreams(prev => ({
        ...prev,
        [sessionId]: {
          ...prev[sessionId],
          status: 'disconnected',
        },
      }))
    }
  }

  // 断开流连接
  const disconnectStream = (sessionId: string) => {
    const stream = activeStreams[sessionId]
    if (stream) {
      if (stream.connection instanceof EventSource) {
        stream.connection.close()
      } else if (stream.connection instanceof WebSocket) {
        stream.connection.close()
      }

      setActiveStreams(prev => {
        const newStreams = { ...prev }
        delete newStreams[sessionId]
        return newStreams
      })
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'processing':
        return 'text-blue-600 bg-blue-100'
      case 'completed':
        return 'text-green-600 bg-green-100'
      case 'error':
        return 'text-red-600 bg-red-100'
      case 'cancelled':
        return 'text-gray-600 bg-gray-100'
      default:
        return 'text-gray-600 bg-gray-100'
    }
  }

  const getConnectionStatusColor = (status: string) => {
    switch (status) {
      case 'connected':
        return 'text-green-600'
      case 'disconnected':
        return 'text-gray-600'
      case 'error':
        return 'text-red-600'
      default:
        return 'text-gray-600'
    }
  }

  return (
    <div className="space-y-6">
      {/* 创建会话表单 */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          创建流式会话
        </h3>

        {error && (
          <div className="mb-4 bg-red-50 border border-red-200 rounded-md p-3">
            <p className="text-sm text-red-600">{error}</p>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              智能体ID
            </label>
            <input
              type="text"
              value={createForm.agent_id}
              onChange={e =>
                setCreateForm(prev => ({ ...prev, agent_id: e.target.value }))
              }
              className="w-full border border-gray-300 rounded-md px-3 py-2"
              placeholder="智能体ID"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              消息内容
            </label>
            <input
              type="text"
              value={createForm.message}
              onChange={e =>
                setCreateForm(prev => ({ ...prev, message: e.target.value }))
              }
              className="w-full border border-gray-300 rounded-md px-3 py-2"
              placeholder="消息内容"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              缓冲区大小
            </label>
            <input
              type="number"
              value={createForm.buffer_size}
              onChange={e =>
                setCreateForm(prev => ({
                  ...prev,
                  buffer_size: Number(e.target.value),
                }))
              }
              className="w-full border border-gray-300 rounded-md px-3 py-2"
              min="10"
              max="1000"
            />
          </div>
        </div>

        <button
          onClick={handleCreateSession}
          disabled={loading}
          className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:opacity-50"
        >
          {loading ? '创建中...' : '创建会话'}
        </button>
      </div>

      {/* 会话列表 */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">会话管理</h3>

        {Object.keys(sessions).length === 0 ? (
          <p className="text-gray-500 text-center py-8">暂无活跃会话</p>
        ) : (
          <div className="space-y-4">
            {Object.entries(sessions).map(([sessionId, session]) => (
              <div key={sessionId} className="border rounded-lg p-4">
                <div className="flex justify-between items-start mb-3">
                  <div>
                    <h4 className="font-medium text-gray-900">
                      会话 {sessionId.substring(0, 8)}...
                    </h4>
                    <p className="text-sm text-gray-600">
                      智能体: {session.agent_id}
                    </p>
                  </div>

                  <div className="flex items-center space-x-2">
                    <span
                      className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getStatusColor(session.status)}`}
                    >
                      {session.status}
                    </span>

                    {activeStreams[sessionId] && (
                      <span
                        className={`text-xs ${getConnectionStatusColor(activeStreams[sessionId].status)}`}
                      >
                        ● {activeStreams[sessionId].status}
                      </span>
                    )}
                  </div>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm mb-3">
                  <div>
                    <span className="text-gray-600">Token数:</span>
                    <span className="ml-1 font-medium">
                      {session.token_count}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-600">事件数:</span>
                    <span className="ml-1 font-medium">
                      {session.event_count}
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-600">速率:</span>
                    <span className="ml-1 font-medium">
                      {session.tokens_per_second.toFixed(1)} t/s
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-600">错误数:</span>
                    <span className="ml-1 font-medium">
                      {session.error_count}
                    </span>
                  </div>
                </div>

                <div className="flex space-x-2">
                  {!activeStreams[sessionId] ? (
                    <>
                      <button
                        onClick={() =>
                          startSSEStream(sessionId, createForm.message)
                        }
                        className="bg-green-500 text-white px-3 py-1 rounded text-sm hover:bg-green-600"
                      >
                        启动SSE流
                      </button>
                      <button
                        onClick={() => startWebSocketStream(sessionId)}
                        className="bg-blue-500 text-white px-3 py-1 rounded text-sm hover:bg-blue-600"
                      >
                        启动WebSocket流
                      </button>
                    </>
                  ) : (
                    <button
                      onClick={() => disconnectStream(sessionId)}
                      className="bg-yellow-500 text-white px-3 py-1 rounded text-sm hover:bg-yellow-600"
                    >
                      断开流连接
                    </button>
                  )}

                  <button
                    onClick={() => handleStopSession(sessionId)}
                    className="bg-red-500 text-white px-3 py-1 rounded text-sm hover:bg-red-600"
                  >
                    停止会话
                  </button>
                </div>

                {/* 流消息显示 */}
                {activeStreams[sessionId] &&
                  activeStreams[sessionId].messages.length > 0 && (
                    <div className="mt-4">
                      <h5 className="font-medium text-gray-700 mb-2">
                        流消息:
                      </h5>
                      <div
                        ref={streamContainerRef}
                        className="bg-gray-50 border rounded p-3 max-h-40 overflow-y-auto"
                      >
                        {activeStreams[sessionId].messages.map(
                          (message, index) => (
                            <div key={index} className="mb-2 text-xs">
                              <span
                                className={`inline-block px-2 py-1 rounded text-white text-xs mr-2 ${
                                  message.type === 'token'
                                    ? 'bg-blue-500'
                                    : message.type === 'complete'
                                      ? 'bg-green-500'
                                      : message.type === 'error'
                                        ? 'bg-red-500'
                                        : 'bg-gray-500'
                                }`}
                              >
                                {message.type}
                              </span>
                              <span className="font-mono">
                                {JSON.stringify(message.data)}
                              </span>
                            </div>
                          )
                        )}
                      </div>
                    </div>
                  )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
