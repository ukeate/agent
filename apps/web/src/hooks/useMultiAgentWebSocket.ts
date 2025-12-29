import { buildWsUrl } from '../utils/apiBase'
import { useEffect, useRef, useCallback } from 'react'
import { useMultiAgentStore } from '../stores/multiAgentStore'

import { logger } from '../utils/logger'
interface WebSocketMessage {
  type: 'new_message' | 'speaker_change' | 'conversation_completed' | 'conversation_error' | 'conversation_created' | 'conversation_started' | 'conversation_resumed' | 'agent_message' | 'status_change' | 'round_change' | 'session_update' | 'error' | 'connection_established' | 'pong' | 'streaming_token' | 'streaming_complete' | 'streaming_error'
  data: any
  timestamp: string
}

interface UseMultiAgentWebSocketOptions {
  sessionId?: string
  enabled?: boolean
  reconnectAttempts?: number
  reconnectDelay?: number
}

export const useMultiAgentWebSocket = ({
  sessionId,
  enabled = true,
  reconnectAttempts = 5,
  reconnectDelay = 3000,
}: UseMultiAgentWebSocketOptions = {}) => {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectAttemptsRef = useRef(0)
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout>>()
  const postOpenTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const {
    addMessage,
    updateAgentStatus,
    updateSessionStatus,
    setWebsocketConnected,
    setError,
    setCurrentSpeaker,
    currentSession,
    setCurrentSession,
    updateSessionId,
    addStreamingToken,
    completeStreamingMessage,
    handleStreamingError,
  } = useMultiAgentStore()

  // 处理接收到的消息
  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const message: WebSocketMessage = JSON.parse(event.data)
      logger.log('WebSocket收到消息:', message.type, message.data)
      
      switch (message.type) {
        case 'connection_established':
          // WebSocket连接已确认
          logger.log('WebSocket连接已确认:', message.data)
          // 重置错误状态
          setError(null)
          break

        case 'pong':
          // ping-pong心跳响应
          logger.log('收到pong响应:', message.data)
          break

        case 'conversation_created':
          // 对话创建成功，更新当前会话的真实session ID
          logger.log('🔥 收到conversation_created消息:', message.data)
          logger.log('🔥 当前会话:', currentSession)
          
          if (message.data.conversation_id && currentSession) {
            logger.log('🔥 对话创建成功，保持原有会话连接')
            // 不更新sessionId，保持原有连接稳定性
            // 只更新会话状态为active
            updateSessionStatus(currentSession.session_id, 'active')
            
            // 将conversation_id存储在会话数据中，但不改变连接ID
            setCurrentSession({
              ...currentSession,
              conversation_id: message.data.conversation_id,
              status: 'active'
            })
            
            logger.log('🔥 会话状态更新完成，连接保持稳定')
          } else {
            logger.log('🔥 会话状态更新失败:', { 
              hasConversationId: !!message.data.conversation_id, 
              hasCurrentSession: !!currentSession,
              conversationId: message.data.conversation_id
            })
          }
          break


        case 'conversation_started':
          // 对话开始
          logger.log('对话已开始')
          break

        case 'conversation_resumed':
          // 对话恢复
          logger.log('对话已恢复:', message.data)
          // 可以添加恢复成功的提示
          break

        case 'new_message':
          // 新消息（智能体响应）
          logger.log('收到new_message数据结构:', message.data)
          
          // 兼容两种数据格式：message.data.message（旧格式）或直接在message.data中的消息（新格式）
          const messageData = message.data.message || message.data
          if (messageData && messageData.content) {
            logger.log('收到新消息:', messageData)
            addMessage({
              id: messageData.id || `msg-${Date.now()}`,
              role: messageData.role || 'assistant',
              sender: messageData.sender,
              content: messageData.content,
              timestamp: messageData.timestamp || new Date().toISOString(),
              round: messageData.round || 0,
            })
          } else {
            logger.warn('new_message格式不正确:', message.data)
          }
          break

        case 'speaker_change':
          // 发言者变更
          logger.log('发言者变更:', message.data.current_speaker, '轮次:', message.data.round)
          if (message.data.current_speaker) {
            setCurrentSpeaker(message.data.current_speaker)
          }
          break

        case 'conversation_completed':
          // 对话完成
          logger.log('对话已完成')
          if (message.data.session_id) {
            updateSessionStatus(message.data.session_id, 'completed')
          }
          break

        case 'conversation_error':
          // 对话错误
          logger.error('对话出现错误:', message.data.error)
          setError(`对话错误: ${message.data.error}`)
          if (message.data.session_id) {
            updateSessionStatus(message.data.session_id, 'error')
          }
          break

        case 'agent_message':
          // 兼容旧格式的智能体消息
          addMessage({
            id: message.data.id || `msg-${Date.now()}`,
            role: message.data.role || 'assistant',
            sender: message.data.sender,
            content: message.data.content,
            timestamp: message.data.timestamp || new Date().toISOString(),
            round: message.data.round || 0,
          })
          break

        case 'status_change':
          // 更新智能体状态
          if (message.data.agent_id && message.data.status) {
            updateAgentStatus(message.data.agent_id, message.data.status)
          }
          break

        case 'round_change':
          // 轮次变更
          logger.log('轮次变更:', message.data)
          break

        case 'session_update':
          // 会话状态更新
          if (message.data.session_id && message.data.status) {
            updateSessionStatus(message.data.session_id, message.data.status)
          }
          break

        case 'streaming_token':
          // 流式Token - 实时显示每个token
          logger.log('收到流式token:', message.data)
          if (message.data.message_id && message.data.token) {
            addStreamingToken(message.data.message_id, {
              agentName: message.data.agent_name,
              token: message.data.token,
              fullContent: message.data.full_content,
              round: message.data.round,
              isComplete: message.data.is_complete
            })
          }
          break

        case 'streaming_complete':
          // 流式响应完成
          logger.log('流式响应完成:', message.data)
          if (message.data.message_id) {
            completeStreamingMessage(message.data.message_id, {
              agentName: message.data.agent_name,
              fullContent: message.data.full_content,
              round: message.data.round
            })
          }
          break

        case 'streaming_error':
          // 流式响应错误
          logger.error('流式响应错误:', message.data)
          if (message.data.message_id) {
            handleStreamingError(message.data.message_id, {
              agentName: message.data.agent_name,
              error: message.data.error,
              fullContent: message.data.full_content,
              round: message.data.round
            })
          }
          break

        case 'error':
          // 错误消息
          setError(`WebSocket错误: ${message.data.message || '未知错误'}`)
          break

        default:
          logger.warn('未知的WebSocket消息类型:', message.type, message.data)
      }
    } catch (error) {
      logger.error('解析WebSocket消息失败:', error, event.data)
      // 不要因为单个消息解析失败就设置错误状态，防止影响后续消息处理
      logger.warn('跳过此消息继续处理后续消息')
    }
  }, [addMessage, updateAgentStatus, updateSessionStatus, setError, setCurrentSpeaker, currentSession, setCurrentSession, updateSessionId])

  // 连接WebSocket
  const connect = useCallback(() => {
    logger.log('connect函数调用:', { enabled, sessionId })
    
    if (!enabled || !sessionId) {
      logger.log('连接条件不满足:', { enabled, sessionId })
      return
    }

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = undefined
    }

    // 如果已有连接且状态正常，不重复创建
    if (wsRef.current && (wsRef.current.readyState === WebSocket.CONNECTING || wsRef.current.readyState === WebSocket.OPEN)) {
      logger.log('WebSocket已连接，跳过重复连接:', wsRef.current.readyState)
      return
    }
    
    // 如果有连接正在关闭，等待关闭完成再重连
    if (wsRef.current && wsRef.current.readyState === WebSocket.CLOSING) {
      logger.log('WebSocket正在关闭，等待完成后重连')
      return
    }

    try {
      const wsUrl = buildWsUrl(`/multi-agent/ws/${sessionId}`)
      
      logger.log('连接WebSocket:', wsUrl)
      
      const ws = new WebSocket(wsUrl)
      wsRef.current = ws

      ws.onopen = () => {
        logger.log('WebSocket连接已建立，readyState:', ws.readyState)
        setWebsocketConnected(true)
        reconnectAttemptsRef.current = 0
        setError(null)
        
        // 立即验证连接状态
        logger.log('WebSocket连接验证:', {
          readyState: ws.readyState,
          OPEN: WebSocket.OPEN,
          isOpen: ws.readyState === WebSocket.OPEN
        })
        
        // 发送ping测试连接并检查是否需要自动启动对话
        if (postOpenTimerRef.current) {
          clearTimeout(postOpenTimerRef.current)
        }
        postOpenTimerRef.current = setTimeout(() => {
          if (ws.readyState === WebSocket.OPEN) {
            logger.log('发送ping测试消息')
            ws.send(JSON.stringify({
              type: 'ping',
              data: { test: true },
              timestamp: new Date().toISOString()
            }))
            
            // 检查是否有待启动的对话
            const currentState = useMultiAgentStore.getState()
            if (currentState.currentSession && 
                currentState.currentSession.status === 'created' && 
                currentState.currentMessages.length > 0) {
              logger.log('检测到待启动对话，自动发送启动消息')
              
              // 获取初始消息
              const initialMessage = currentState.currentMessages.find(msg => msg.role === 'user')?.content
              const participants = currentState.currentSession.participants.map(p => p.role)
              
              if (initialMessage && participants.length > 0) {
                logger.log('自动发送对话启动消息:', { initialMessage, participants })
                ws.send(JSON.stringify({
                  type: 'start_conversation',
                  data: {
                    message: initialMessage,
                    participants: participants
                  },
                  timestamp: new Date().toISOString()
                }))
              }
            }
          }
        }, 1000)
      }

      ws.onmessage = handleMessage

      ws.onclose = (event) => {
        logger.log('WebSocket连接已关闭:', event.code, event.reason)
        setWebsocketConnected(false)
        if (postOpenTimerRef.current) {
          clearTimeout(postOpenTimerRef.current)
          postOpenTimerRef.current = null
        }
        
        // 清理当前连接引用
        if (wsRef.current === ws) {
          wsRef.current = null
        }
        
        // 只有在非正常关闭且需要重连时才进行重连
        // 1006: 异常关闭，1012: 服务重启，1011: 服务器错误
        const shouldReconnect = enabled && 
                              reconnectAttemptsRef.current < reconnectAttempts &&
                              [1006, 1012, 1011].includes(event.code)
        
        if (shouldReconnect) {
          reconnectAttemptsRef.current++
          logger.log(`WebSocket异常关闭(${event.code})，准备重连 (${reconnectAttemptsRef.current}/${reconnectAttempts})`)
          
          // 使用指数退避策略，避免频繁重连
          const delay = Math.min(reconnectDelay * Math.pow(2, reconnectAttemptsRef.current - 1), 30000)
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connect()
          }, delay)
        } else if (reconnectAttemptsRef.current >= reconnectAttempts) {
          setError('WebSocket连接失败，已达到最大重连次数')
        } else {
          logger.log('WebSocket正常关闭或不需要重连:', event.code)
        }
      }

      ws.onerror = (error) => {
        logger.error('WebSocket错误:', error)
        setError('WebSocket连接出现错误')
        if (ws.readyState !== WebSocket.CLOSING && ws.readyState !== WebSocket.CLOSED) {
          ws.close()
        }
      }

    } catch (error) {
      logger.error('创建WebSocket连接失败:', error)
      setError('无法创建WebSocket连接')
    }
  }, [enabled, sessionId, handleMessage, setWebsocketConnected, setError, reconnectAttempts, reconnectDelay])

  // 断开连接
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = undefined
    }

    if (postOpenTimerRef.current) {
      clearTimeout(postOpenTimerRef.current)
      postOpenTimerRef.current = null
    }
    
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    
    setWebsocketConnected(false)
  }, [setWebsocketConnected])

  // 发送消息
  const sendMessage = useCallback((message: Omit<WebSocketMessage, 'timestamp'>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      try {
        const fullMessage: WebSocketMessage = {
          ...message,
          timestamp: new Date().toISOString(),
        }
        
        logger.log('发送WebSocket消息:', fullMessage)
        wsRef.current.send(JSON.stringify(fullMessage))
        return true
      } catch (error) {
        logger.error('发送消息失败:', error)
        setError('发送消息失败')
        return false
      }
    } else {
      logger.warn('WebSocket未连接，无法发送消息，当前状态:', wsRef.current?.readyState)
      setError('WebSocket连接未就绪')
      return false
    }
  }, [setError])

  // 监听sessionId变化
  useEffect(() => {
    logger.log('WebSocket useEffect 触发:', { enabled, sessionId, hasCurrentRef: !!wsRef.current })
    
    if (enabled && sessionId) {
      logger.log('准备建立WebSocket连接:', sessionId)
      // 延迟连接以避免过快连接
      const timer = setTimeout(() => {
        logger.log('延迟后开始连接WebSocket:', sessionId)
        connect()
      }, 100)
      
      return () => {
        logger.log('清理WebSocket连接:', sessionId)
        clearTimeout(timer)
        disconnect()
      }
    } else {
      logger.log('WebSocket条件不满足，断开连接:', { enabled, sessionId })
      disconnect()
    }
  }, [sessionId, enabled, connect, disconnect])

  // 组件卸载时清理
  useEffect(() => {
    return () => {
      disconnect()
    }
  }, [disconnect])

  return {
    connected: wsRef.current?.readyState === WebSocket.OPEN,
    connect,
    disconnect,
    sendMessage,
  }
}
