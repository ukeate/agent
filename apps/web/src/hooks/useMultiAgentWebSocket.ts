import { useEffect, useRef, useCallback } from 'react'
import { useMultiAgentStore } from '../stores/multiAgentStore'

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
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>()

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
      console.log('WebSocket收到消息:', message.type, message.data)
      
      switch (message.type) {
        case 'connection_established':
          // WebSocket连接已确认
          console.log('WebSocket连接已确认:', message.data)
          // 重置错误状态
          setError(null)
          break

        case 'pong':
          // ping-pong心跳响应
          console.log('收到pong响应:', message.data)
          break

        case 'conversation_created':
          // 对话创建成功，更新当前会话的真实session ID
          console.log('🔥 收到conversation_created消息:', message.data)
          console.log('🔥 当前会话:', currentSession)
          
          if (message.data.conversation_id && currentSession) {
            console.log('🔥 对话创建成功，同步会话ID:', currentSession.session_id, '->', message.data.conversation_id)
            // updateSessionId函数已经会同时设置状态为'active'
            updateSessionId(currentSession.session_id, message.data.conversation_id)
            console.log('🔥 会话ID映射和状态更新完成')
          } else {
            console.log('🔥 会话ID映射失败:', { 
              hasConversationId: !!message.data.conversation_id, 
              hasCurrentSession: !!currentSession,
              conversationId: message.data.conversation_id
            })
          }
          break


        case 'conversation_started':
          // 对话开始
          console.log('对话已开始')
          break

        case 'conversation_resumed':
          // 对话恢复
          console.log('对话已恢复:', message.data)
          // 可以添加恢复成功的提示
          break

        case 'new_message':
          // 新消息（智能体响应）
          console.log('收到new_message数据结构:', message.data)
          
          // 兼容两种数据格式：message.data.message（旧格式）或直接在message.data中的消息（新格式）
          const messageData = message.data.message || message.data
          if (messageData && messageData.content) {
            console.log('收到新消息:', messageData)
            addMessage({
              id: messageData.id || `msg-${Date.now()}`,
              role: messageData.role || 'assistant',
              sender: messageData.sender,
              content: messageData.content,
              timestamp: messageData.timestamp || new Date().toISOString(),
              round: messageData.round || 0,
            })
          } else {
            console.warn('new_message格式不正确:', message.data)
          }
          break

        case 'speaker_change':
          // 发言者变更
          console.log('发言者变更:', message.data.current_speaker, '轮次:', message.data.round)
          if (message.data.current_speaker) {
            setCurrentSpeaker(message.data.current_speaker)
          }
          break

        case 'conversation_completed':
          // 对话完成
          console.log('对话已完成')
          if (message.data.session_id) {
            updateSessionStatus(message.data.session_id, 'completed')
          }
          break

        case 'conversation_error':
          // 对话错误
          console.error('对话出现错误:', message.data.error)
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
          console.log('Round changed:', message.data)
          break

        case 'session_update':
          // 会话状态更新
          if (message.data.session_id && message.data.status) {
            updateSessionStatus(message.data.session_id, message.data.status)
          }
          break

        case 'streaming_token':
          // 流式Token - 实时显示每个token
          console.log('收到流式token:', message.data)
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
          console.log('流式响应完成:', message.data)
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
          console.error('流式响应错误:', message.data)
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
          console.warn('未知的WebSocket消息类型:', message.type, message.data)
      }
    } catch (error) {
      console.error('解析WebSocket消息失败:', error, event.data)
      // 不要因为单个消息解析失败就设置错误状态，防止影响后续消息处理
      console.warn('跳过此消息继续处理后续消息')
    }
  }, [addMessage, updateAgentStatus, updateSessionStatus, setError, setCurrentSpeaker, currentSession, setCurrentSession, updateSessionId])

  // 连接WebSocket
  const connect = useCallback(() => {
    console.log('connect函数调用:', { enabled, sessionId })
    
    if (!enabled || !sessionId) {
      console.log('连接条件不满足:', { enabled, sessionId })
      return
    }

    // 如果已有连接且状态正常，不重复创建
    if (wsRef.current && (wsRef.current.readyState === WebSocket.CONNECTING || wsRef.current.readyState === WebSocket.OPEN)) {
      console.log('WebSocket已连接，跳过重复连接:', wsRef.current.readyState)
      return
    }

    try {
      // 构建WebSocket URL - 开发环境直接连接到后端服务器
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const isDev = import.meta.env.DEV
      const host = isDev ? 'localhost:8000' : window.location.host
      const wsUrl = `${protocol}//${host}/api/v1/multi-agent/ws/${sessionId}`
      
      console.log('连接WebSocket:', wsUrl)
      
      const ws = new WebSocket(wsUrl)
      wsRef.current = ws

      ws.onopen = () => {
        console.log('WebSocket连接已建立，readyState:', ws.readyState)
        setWebsocketConnected(true)
        reconnectAttemptsRef.current = 0
        setError(null)
        
        // 立即验证连接状态
        console.log('WebSocket连接验证:', {
          readyState: ws.readyState,
          OPEN: WebSocket.OPEN,
          isOpen: ws.readyState === WebSocket.OPEN
        })
        
        // 发送ping测试连接并检查是否需要自动启动对话
        setTimeout(() => {
          if (ws.readyState === WebSocket.OPEN) {
            console.log('发送ping测试消息')
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
              console.log('检测到待启动对话，自动发送启动消息')
              
              // 获取初始消息
              const initialMessage = currentState.currentMessages.find(msg => msg.role === 'user')?.content
              const participants = currentState.currentSession.participants.map(p => `${p.role}-1`)
              
              if (initialMessage && participants.length > 0) {
                console.log('自动发送对话启动消息:', { initialMessage, participants })
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
        console.log('WebSocket连接已关闭:', event.code, event.reason)
        setWebsocketConnected(false)
        
        // 自动重连
        if (enabled && reconnectAttemptsRef.current < reconnectAttempts) {
          reconnectAttemptsRef.current++
          console.log(`WebSocket重连 (${reconnectAttemptsRef.current}/${reconnectAttempts})`)
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connect()
          }, reconnectDelay)
        } else if (reconnectAttemptsRef.current >= reconnectAttempts) {
          setError('WebSocket连接失败，已达到最大重连次数')
        }
      }

      ws.onerror = (error) => {
        console.error('WebSocket错误:', error)
        setError('WebSocket连接出现错误')
      }

    } catch (error) {
      console.error('创建WebSocket连接失败:', error)
      setError('无法创建WebSocket连接')
    }
  }, [enabled, sessionId, handleMessage, setWebsocketConnected, setError, reconnectAttempts, reconnectDelay])

  // 断开连接
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
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
        
        console.log('发送WebSocket消息:', fullMessage)
        wsRef.current.send(JSON.stringify(fullMessage))
        return true
      } catch (error) {
        console.error('发送消息失败:', error)
        setError('发送消息失败')
        return false
      }
    } else {
      console.warn('WebSocket未连接，无法发送消息，当前状态:', wsRef.current?.readyState)
      setError('WebSocket连接未就绪')
      return false
    }
  }, [setError])

  // 监听sessionId变化
  useEffect(() => {
    console.log('WebSocket useEffect 触发:', { enabled, sessionId, hasCurrentRef: !!wsRef.current })
    
    if (enabled && sessionId) {
      console.log('准备建立WebSocket连接:', sessionId)
      // 延迟连接以避免过快连接
      const timer = setTimeout(() => {
        console.log('延迟后开始连接WebSocket:', sessionId)
        connect()
      }, 100)
      
      return () => {
        console.log('清理WebSocket连接:', sessionId)
        clearTimeout(timer)
        disconnect()
      }
    } else {
      console.log('WebSocket条件不满足，断开连接:', { enabled, sessionId })
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